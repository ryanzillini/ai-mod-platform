"""Unit tests for token-distribution confidence math and dual routing.

No model load required.
"""

from src.policy import DecisionPolicy
from src.slm_engine import (
    LocalSLMEngine,
    TokenDistributionConfidence,
    confidence_from_binary_mass,
    first_continuation_token,
)


def test_normalized_confidence_from_masses():
    assert confidence_from_binary_mass(0.8, 0.2) == 0.8
    assert confidence_from_binary_mass(0.2, 0.8) == 0.8
    assert confidence_from_binary_mass(0.5, 0.5) == 0.5
    assert abs(confidence_from_binary_mass(0.09, 0.01) - 0.9) < 1e-12
    assert confidence_from_binary_mass(0.0, 0.0) == 0.0


def test_first_continuation_token_requires_stable_prefix():
    class _Tok:
        bos_token = None

        def encode(self, text, add_special_tokens=True):
            table = {
                "VERDICT:": [1, 2, 3],
                "VERDICT: SAFE": [1, 2, 3, 10],
                "VERDICT: UNSAFE": [1, 2, 3, 11, 12],
                "VERDICT: ": [1, 2, 3, 9],
            }
            return table[text]

    tok = _Tok()
    assert first_continuation_token(tok, "", "VERDICT:", " SAFE") == 10
    assert first_continuation_token(tok, "", "VERDICT:", " UNSAFE") == 11
    # Trailing space is not a BPE-stable prefix of "VERDICT: SAFE".
    assert first_continuation_token(tok, "", "VERDICT: ", "SAFE") is None


def test_dual_routing_only_confidence_source_changes():
    engine = LocalSLMEngine.__new__(LocalSLMEngine)
    engine.policy = DecisionPolicy()
    raw = (
        "VERDICT: UNSAFE\n"
        "CATEGORIES: HARASSMENT\n"
        "CONFIDENCE: 0.95\n"
        "REASON: targeted attack\n"
    )
    text = "Insult this person until they quit."
    high = TokenDistributionConfidence(
        score=0.95, p_safe=0.02, p_unsafe=0.90, decision_mass=0.92, margin=0.88
    )
    low = TokenDistributionConfidence(
        score=0.55, p_safe=0.40, p_unsafe=0.50, decision_mass=0.90, margin=0.10
    )

    with_self = LocalSLMEngine._decide(engine, text, raw, 10.0, high)
    assert with_self.action == "BLOCK"
    assert with_self.action_if_self_report == "BLOCK"
    assert with_self.action_if_computed == "BLOCK"
    assert with_self.confidence_source == "computed"
    assert with_self.confidence_score == 0.95
    assert with_self.self_reported_confidence == 0.95
    assert with_self.computed_confidence == 0.95

    mixed = LocalSLMEngine._decide(engine, text, raw, 10.0, low)
    assert mixed.action_if_self_report == "BLOCK"
    assert mixed.action_if_computed == "ESCALATE"
    assert mixed.action == "ESCALATE"
    assert mixed.confidence_score == 0.55
    assert mixed.confidence_source == "computed"
    assert mixed.self_reported_confidence == 0.95
    assert "low confidence" in (mixed.escalation_reason or "")
    assert mixed.escalation_reason_self_report is None


def test_missing_computed_falls_back_to_self_report():
    engine = LocalSLMEngine.__new__(LocalSLMEngine)
    engine.policy = DecisionPolicy()
    raw = (
        "VERDICT: UNSAFE\n"
        "CATEGORIES: HARASSMENT\n"
        "CONFIDENCE: 0.95\n"
        "REASON: targeted attack\n"
    )
    decision = LocalSLMEngine._decide(engine, "Insult this person until they quit.", raw, 10.0)
    assert decision.confidence_source == "self_report_fallback"
    assert decision.action == "BLOCK"
    assert decision.confidence_score == 0.95
    assert decision.self_reported_confidence == 0.95
    assert decision.computed_confidence is None


def test_parse_failure_escalates_both_paths_even_with_computed_score():
    engine = LocalSLMEngine.__new__(LocalSLMEngine)
    engine.policy = DecisionPolicy()
    computed = TokenDistributionConfidence(
        score=0.99, p_safe=0.01, p_unsafe=0.90, decision_mass=0.91, margin=0.89
    )
    decision = LocalSLMEngine._decide(
        engine, "hello", "not a structured response", 12.3, computed
    )
    assert decision.action == "ESCALATE"
    assert decision.action_if_self_report == "ESCALATE"
    assert decision.action_if_computed == "ESCALATE"
    assert decision.escalation_reason == "parse_failure"
    assert decision.confidence_source == "parse_failure"
    assert decision.confidence_score == 0.0
    assert decision.computed_confidence == 0.99
