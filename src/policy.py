"""Separable decision policy: parse model output, apply routing rules.

The SLM only classifies (SAFE/UNSAFE + confidence). This module maps that
classification onto ALLOW / BLOCK / ESCALATE. Policy can force escalation
even when the model is highly confident.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Literal, NamedTuple, Optional

Action = Literal["ALLOW", "BLOCK", "ESCALATE"]
PolicyVerdict = Literal["ALLOW", "BLOCK"]

KNOWN_CATEGORIES = (
    "VIOLENCE",
    "EXPLOITATION",
    "HARASSMENT",
    "PII_LEAK",
    "FINANCIAL_FRAUD",
    "JAILBREAK",
)

_NONE_CATEGORY_TOKENS = {"none", "n/a", "na", "-", "null", "nil"}
_REFUSAL_RE = re.compile(
    r"(?i)\bi\s+(can('t|not)|won't|will not)\s+(fulfill|provide|help|assist|do that)"
)
# Instruct models often echo the taxonomy line instead of selecting categories.
_SCHEMA_ECHO_MIN = 5
# Refusal is an implicit high-confidence UNSAFE classification, not a parse miss.
_REFUSAL_CONFIDENCE = 0.90


@dataclass
class DecisionPolicy:
    confidence_threshold: float = 0.80
    always_escalate_categories: set[str] = field(
        default_factory=lambda: {"EXPLOITATION"}
    )
    always_escalate_if_medical_advice: bool = True
    always_escalate_if_legal_interpretation: bool = True
    always_escalate_if_hr_sensitive: bool = True
    always_escalate_if_workplace_complaint: bool = True
    always_escalate_if_operational_pii: bool = True
    always_escalate_if_regulatory_avoidance: bool = True


class ParsedClassification(NamedTuple):
    is_safe: bool
    categories: List[str]
    confidence: float
    reason: str


def looks_like_medical_advice(text: str) -> bool:
    """Keyword heuristic: clinical decision-support, not a mention of healthcare."""
    t = text.lower()
    cues = (
        "patient",
        "dosage",
        "medication",
        "troponin",
        "chest pain",
        "shortness of breath",
        "prescribe",
        "next medication",
        "what should i order",
        "order right now",
        "clinical presentation",
    )
    return sum(1 for cue in cues if cue in t) >= 2


def looks_like_legal_question(text: str) -> bool:
    """Keyword heuristic: asking the model to interpret law / contract / regulation."""
    t = text.lower()
    return any(
        phrase in t
        for phrase in (
            "does this violate",
            "violate our current",
            "gdpr requirements",
            "data processing agreement",
            "legal advice",
            "against the law",
            "regulatory interpretation",
        )
    )


def looks_like_hr_sensitive(text: str) -> bool:
    """Keyword heuristic: high-stakes workforce / internal HR communications."""
    t = text.lower()
    return any(
        phrase in t
        for phrase in (
            "reduction in force",
            "planned reduction",
            "layoff",
            "layoffs",
            "rif ",
        )
    )


def looks_like_workplace_complaint(text: str) -> bool:
    """Keyword heuristic: gray-area professional complaint about a person."""
    t = text.lower()
    if "complaint" not in t:
        return False
    return any(
        phrase in t
        for phrase in (
            "strongly worded",
            "professional complaint",
            "support engineer",
            "coworker",
            "colleague",
        )
    )


_EXPLOITATION_PERSON_RE = re.compile(
    r"\b(minors?|children|child|underage|kids?)\b",
    re.I,
)
# 3B often dumps EXPLOITATION on clear fraud / weapons how-tos. These cues
# identify a competing clear-BLOCK category so the high-severity rule does not
# steal a correct BLOCK.
_COMPETING_CLEAR_BLOCK_CUES = (
    "for fraud",
    "synthetic identit",
    "improvised explosive",
    "explosive",
    "household chemicals",
)


def looks_like_exploitation(text: str) -> bool:
    """Person-targeted exploitation (minors / children), not 'exploit a system'."""
    return bool(_EXPLOITATION_PERSON_RE.search(text))


def _ignore_exploitation_label(text: str) -> bool:
    """True when EXPLOITATION is a noisy label on a clear competing BLOCK."""
    if looks_like_exploitation(text):
        return False
    t = text.lower()
    return any(cue in t for cue in _COMPETING_CLEAR_BLOCK_CUES)


def looks_like_regulatory_avoidance(text: str) -> bool:
    """Actionable request to evade named reporting (CTR/SAR), not clear fraud how-to.

    High-confidence UNSAFE is still the wrong auto-action when the ask is
    'how do I avoid the report' rather than 'how do I commit fraud'.
    """
    t = text.lower()
    has_regime = bool(re.search(r"\b(ctr|sar)\b", t)) or any(
        phrase in t
        for phrase in (
            "currency transaction report",
            "suspicious activity report",
        )
    )
    has_evasion = any(
        phrase in t
        for phrase in (
            "without triggering",
            "avoid reporting",
            "evade reporting",
            "without reporting",
        )
    )
    return has_regime and has_evasion


_ACCOUNT_NUMBER_RE = re.compile(r"\b\d{4}[-\s]\d{4}[-\s]\d{4}(?:[-\s]\d{4})?\b")
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_BULK_PII_DUMP_CUES = (
    "list every",
    "list all",
    "dump every",
    "exfiltrate",
)


def looks_like_operational_pii(text: str) -> bool:
    """Live identifiers in operational content, not a bulk PII dump.

    Requires two independent cues so a lone 'SSN' in a clear exfil request
    does not steal a confident BLOCK.
    """
    t = text.lower()
    if any(phrase in t for phrase in _BULK_PII_DUMP_CUES):
        return False

    signals = (
        "account number" in t or "full account" in t,
        "date of birth" in t or bool(re.search(r"\bdob\b", t)),
        bool(re.search(r"\bssn\b", t)) or "social security" in t,
        "routing number" in t,
        bool(_ACCOUNT_NUMBER_RE.search(text)),
        bool(_SSN_RE.search(text)),
    )
    return sum(1 for hit in signals if hit) >= 2


def parse_model_output(raw: str) -> Optional[ParsedClassification]:
    """Extract VERDICT / CATEGORIES / CONFIDENCE / REASON from noisy model text.

    Returns None on parse failure (missing verdict or confidence).
    A base-model refusal is treated as implicit UNSAFE, not parse failure.
    """
    if not raw or not str(raw).strip():
        return None

    text = _strip_code_fences(str(raw))
    fields: dict[str, str] = {}
    for match in re.finditer(
        r"(?im)^\s*(?:\*\*)?(VERDICT|CATEGORIES|CONFIDENCE|REASON)(?:\*\*)?\s*[:\-]\s*(.+?)\s*$",
        text,
    ):
        key = match.group(1).upper()
        value = match.group(2).strip().strip("`").strip()
        fields[key] = value

    verdict_raw = fields.get("VERDICT", "").upper()
    if "UNSAFE" in verdict_raw:
        is_safe = False
    elif re.search(r"\bSAFE\b", verdict_raw):
        is_safe = True
    elif _REFUSAL_RE.search(text):
        return ParsedClassification(
            is_safe=False,
            categories=[],
            confidence=_REFUSAL_CONFIDENCE,
            reason="model_refusal",
        )
    else:
        return None

    conf_match = re.search(r"(\d+(?:\.\d+)?)", fields.get("CONFIDENCE", ""))
    if not conf_match:
        if _REFUSAL_RE.search(text):
            return ParsedClassification(
                is_safe=False,
                categories=[],
                confidence=_REFUSAL_CONFIDENCE,
                reason="model_refusal",
            )
        return None
    confidence = float(conf_match.group(1))
    if confidence > 1.0 and confidence <= 100.0:
        confidence /= 100.0
    confidence = min(1.0, max(0.0, confidence))

    categories = _parse_categories(fields.get("CATEGORIES", "none"))
    reason = fields.get("REASON", "").strip()
    return ParsedClassification(
        is_safe=is_safe,
        categories=categories,
        confidence=confidence,
        reason=reason,
    )


class RoutingStep(NamedTuple):
    rule: str
    fired: bool
    detail: str


class RoutingResult(NamedTuple):
    action: Action
    escalation_reason: Optional[str]
    steps: tuple[RoutingStep, ...]
    winning_rule: str


def explain_route(
    text: str,
    parsed: ParsedClassification,
    policy: DecisionPolicy | None = None,
) -> RoutingResult:
    """Route a classification and record which rules were evaluated.

    Short-circuits: rules after the winner are not evaluated. That is the
    real control flow, not a reconstructed post-hoc explanation.
    """
    policy = policy or DecisionPolicy()
    categories = parsed.categories
    confidence = parsed.confidence
    steps: list[RoutingStep] = []

    matched = [
        cat
        for cat in categories
        if cat in policy.always_escalate_categories
        and not (cat == "EXPLOITATION" and _ignore_exploitation_label(text))
    ]
    if matched:
        steps.append(
            RoutingStep("high_severity_category", True, ",".join(matched))
        )
        return RoutingResult(
            "ESCALATE",
            "policy: high-severity category",
            tuple(steps),
            "high_severity_category",
        )
    if any(cat == "EXPLOITATION" for cat in categories) and _ignore_exploitation_label(
        text
    ):
        skip_detail = "EXPLOITATION ignored (competing clear-unsafe cues)"
    else:
        skip_detail = "none" if not categories else ",".join(categories)
    steps.append(RoutingStep("high_severity_category", False, skip_detail))

    if confidence < policy.confidence_threshold:
        detail = f"{confidence:.2f} < {policy.confidence_threshold:.2f}"
        steps.append(RoutingStep("low_confidence", True, detail))
        return RoutingResult(
            "ESCALATE",
            f"low confidence ({detail})",
            tuple(steps),
            "low_confidence",
        )
    steps.append(
        RoutingStep(
            "low_confidence",
            False,
            f"{confidence:.2f} >= {policy.confidence_threshold:.2f}",
        )
    )

    domain_rules = (
        (
            "medical_decision_support",
            policy.always_escalate_if_medical_advice,
            looks_like_medical_advice,
            "policy: medical decision support",
        ),
        (
            "legal_regulatory",
            policy.always_escalate_if_legal_interpretation,
            looks_like_legal_question,
            "policy: legal / regulatory interpretation",
        ),
        (
            "hr_sensitive",
            policy.always_escalate_if_hr_sensitive,
            looks_like_hr_sensitive,
            "policy: high-stakes HR communication",
        ),
        (
            "workplace_complaint",
            policy.always_escalate_if_workplace_complaint,
            looks_like_workplace_complaint,
            "policy: workplace complaint gray area",
        ),
        (
            "operational_pii",
            policy.always_escalate_if_operational_pii,
            looks_like_operational_pii,
            "policy: operational PII / live identifiers",
        ),
        (
            "regulatory_avoidance",
            policy.always_escalate_if_regulatory_avoidance,
            looks_like_regulatory_avoidance,
            "policy: regulatory-avoidance request",
        ),
    )
    for rule, enabled, matcher, reason in domain_rules:
        if not enabled:
            continue
        if matcher(text):
            steps.append(RoutingStep(rule, True, "matched"))
            return RoutingResult("ESCALATE", reason, tuple(steps), rule)
        steps.append(RoutingStep(rule, False, "no match"))

    action: Action = "ALLOW" if parsed.is_safe else "BLOCK"
    detail = "SAFE → ALLOW" if parsed.is_safe else "UNSAFE → BLOCK"
    steps.append(RoutingStep("classification", True, detail))
    return RoutingResult(action, None, tuple(steps), "classification")


def route_decision(
    text: str,
    parsed: ParsedClassification,
    policy: DecisionPolicy | None = None,
) -> tuple[Action, Optional[str]]:
    """Map a parsed classification onto a system action.

    Order matches the v1 brief: high-severity category, then low confidence,
    then domain policy rules, else ALLOW/BLOCK from the classification.
    """
    result = explain_route(text, parsed, policy)
    return result.action, result.escalation_reason


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if not text.startswith("```"):
        return text
    lines = text.split("\n")
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines)


def _parse_categories(raw: str) -> List[str]:
    if not raw or raw.strip().lower() in _NONE_CATEGORY_TOKENS:
        return []
    # Models copy "none|VIOLENCE,..." from schema-style prompts; drop the none prefix.
    cleaned = re.sub(r"(?i)^\s*none\s*[|:/]\s*", "", raw.strip())
    found: list[str] = []
    for part in re.split(r"[,;/]", cleaned):
        token = re.sub(r"[^A-Z0-9_]", "", part.strip().upper().replace(" ", "_"))
        if token in KNOWN_CATEGORIES and token not in found:
            found.append(token)
    # Full taxonomy dump is a schema echo, not a real multi-label prediction.
    if len(found) >= _SCHEMA_ECHO_MIN:
        return []
    return found
