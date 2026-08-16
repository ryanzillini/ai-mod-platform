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


def route_decision(
    text: str,
    parsed: ParsedClassification,
    policy: DecisionPolicy | None = None,
) -> tuple[Action, Optional[str]]:
    """Map a parsed classification onto a system action.

    Order matches the v1 brief: high-severity category, then low confidence,
    then domain policy rules, else ALLOW/BLOCK from the classification.
    """
    policy = policy or DecisionPolicy()
    categories = parsed.categories
    confidence = parsed.confidence

    if any(cat in policy.always_escalate_categories for cat in categories):
        return "ESCALATE", "policy: high-severity category"

    if confidence < policy.confidence_threshold:
        return (
            "ESCALATE",
            f"low confidence ({confidence:.2f} < {policy.confidence_threshold:.2f})",
        )

    if policy.always_escalate_if_medical_advice and looks_like_medical_advice(text):
        return "ESCALATE", "policy: medical decision support"

    if policy.always_escalate_if_legal_interpretation and looks_like_legal_question(text):
        return "ESCALATE", "policy: legal / regulatory interpretation"

    if policy.always_escalate_if_hr_sensitive and looks_like_hr_sensitive(text):
        return "ESCALATE", "policy: high-stakes HR communication"

    if policy.always_escalate_if_workplace_complaint and looks_like_workplace_complaint(
        text
    ):
        return "ESCALATE", "policy: workplace complaint gray area"

    action: Action = "ALLOW" if parsed.is_safe else "BLOCK"
    return action, None


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
