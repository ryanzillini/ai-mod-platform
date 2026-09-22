"""Separable decision policy: parse model output, apply routing rules.

The SLM only classifies (SAFE/UNSAFE + confidence). Detectors only report
hits. This module maps classification + hits onto ALLOW / BLOCK / ESCALATE.
Policy can force escalation even when the model is highly confident.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Literal, NamedTuple, Optional

from src.detectors import DetectorResult, detector_hit, run_detectors

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
    detectors: tuple[DetectorResult, ...] = ()


def _ignore_noisy_exploitation(detectors: tuple[DetectorResult, ...]) -> bool:
    """Policy combines two sensors: competing BLOCK cues, not person-targeted."""
    return detector_hit(detectors, "competing_clear_block") and not detector_hit(
        detectors, "exploitation"
    )


def explain_route(
    text: str,
    parsed: ParsedClassification,
    policy: DecisionPolicy | None = None,
    detector_results: tuple[DetectorResult, ...] | None = None,
) -> RoutingResult:
    """Route a classification and record which rules were evaluated.

    Short-circuits: rules after the winner are not evaluated. That is the
    real control flow, not a reconstructed post-hoc explanation.

    Detectors always run (stage 1) even when a later routing rule wins.
    They do not choose ALLOW / BLOCK / ESCALATE.
    """
    policy = policy or DecisionPolicy()
    detectors = (
        detector_results if detector_results is not None else run_detectors(text)
    )
    categories = parsed.categories
    confidence = parsed.confidence
    steps: list[RoutingStep] = []
    ignore_exploitation = _ignore_noisy_exploitation(detectors)

    matched = [
        cat
        for cat in categories
        if cat in policy.always_escalate_categories
        and not (cat == "EXPLOITATION" and ignore_exploitation)
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
            detectors,
        )
    if any(cat == "EXPLOITATION" for cat in categories) and ignore_exploitation:
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
            detectors,
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
            "medical_decision_support",
            "policy: medical decision support",
        ),
        (
            "legal_regulatory",
            policy.always_escalate_if_legal_interpretation,
            "legal_regulatory",
            "policy: legal / regulatory interpretation",
        ),
        (
            "hr_sensitive",
            policy.always_escalate_if_hr_sensitive,
            "hr_sensitive",
            "policy: high-stakes HR communication",
        ),
        (
            "workplace_complaint",
            policy.always_escalate_if_workplace_complaint,
            "workplace_complaint",
            "policy: workplace complaint gray area",
        ),
        (
            "operational_pii",
            policy.always_escalate_if_operational_pii,
            "operational_pii",
            "policy: operational PII / live identifiers",
        ),
        (
            "regulatory_avoidance",
            policy.always_escalate_if_regulatory_avoidance,
            "regulatory_avoidance",
            "policy: regulatory-avoidance request",
        ),
    )
    for rule, enabled, detector_id, reason in domain_rules:
        if not enabled:
            continue
        if detector_hit(detectors, detector_id):
            steps.append(RoutingStep(rule, True, "matched"))
            return RoutingResult(
                "ESCALATE", reason, tuple(steps), rule, detectors
            )
        steps.append(RoutingStep(rule, False, "no match"))

    action: Action = "ALLOW" if parsed.is_safe else "BLOCK"
    if action == "ALLOW" and any(not item.ok for item in detectors):
        steps.append(
            RoutingStep("detector_error", True, "detector exception; refuse ALLOW")
        )
        return RoutingResult(
            "ESCALATE",
            "detector error (fail closed)",
            tuple(steps),
            "detector_error",
            detectors,
        )
    detail = "SAFE → ALLOW" if parsed.is_safe else "UNSAFE → BLOCK"
    steps.append(RoutingStep("classification", True, detail))
    return RoutingResult(action, None, tuple(steps), "classification", detectors)


def route_decision(
    text: str,
    parsed: ParsedClassification,
    policy: DecisionPolicy | None = None,
    detector_results: tuple[DetectorResult, ...] | None = None,
) -> tuple[Action, Optional[str]]:
    """Map a parsed classification onto a system action.

    Order matches the v1 brief: high-severity category, then low confidence,
    then domain policy rules, else ALLOW/BLOCK from the classification.
    """
    result = explain_route(text, parsed, policy, detector_results)
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
