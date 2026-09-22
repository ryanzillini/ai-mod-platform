"""Deterministic text detectors. They report hits; they never choose an action.

Policy (`explain_route`) combines hits + scores + model labels into
ALLOW / BLOCK / ESCALATE. A detector returning an action is a contract bug.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Sequence


@dataclass(frozen=True)
class DetectorResult:
    detector_id: str
    hit: bool
    detail: str
    ok: bool = True


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
# identify a competing clear-BLOCK category so policy can ignore the noisy label.
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


def looks_like_competing_clear_block(text: str) -> bool:
    """Clear-BLOCK how-to cues that compete with a noisy EXPLOITATION label."""
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


DetectorFn = Callable[[str], bool]

# Stage-1 order is for traces. Policy short-circuit order is separate.
DETECTORS: tuple[tuple[str, DetectorFn], ...] = (
    ("medical_decision_support", looks_like_medical_advice),
    ("legal_regulatory", looks_like_legal_question),
    ("hr_sensitive", looks_like_hr_sensitive),
    ("workplace_complaint", looks_like_workplace_complaint),
    ("operational_pii", looks_like_operational_pii),
    ("regulatory_avoidance", looks_like_regulatory_avoidance),
    ("exploitation", looks_like_exploitation),
    ("competing_clear_block", looks_like_competing_clear_block),
)


def run_detectors(
    text: str,
    specs: Sequence[tuple[str, DetectorFn]] | None = None,
) -> tuple[DetectorResult, ...]:
    """Run every detector. Exceptions become an error span, not a crash.

    `hit` is False on error so policy does not treat a broken sensor as a match.
    Policy must still refuse ALLOW when any span is not `ok`.
    """
    results: list[DetectorResult] = []
    for detector_id, fn in specs or DETECTORS:
        try:
            hit = bool(fn(text))
            results.append(
                DetectorResult(
                    detector_id=detector_id,
                    hit=hit,
                    detail="matched" if hit else "no match",
                    ok=True,
                )
            )
        except Exception as exc:
            results.append(
                DetectorResult(
                    detector_id=detector_id,
                    hit=False,
                    detail=f"error: {type(exc).__name__}",
                    ok=False,
                )
            )
    return tuple(results)


def detector_hit(results: Sequence[DetectorResult], detector_id: str) -> bool:
    """True only when that detector ran, succeeded, and matched."""
    for result in results:
        if result.detector_id == detector_id:
            return bool(result.ok and result.hit)
    return False
