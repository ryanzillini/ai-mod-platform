"""Unit tests for parser + policy routing. No model load required."""

from src.policy import (
    DecisionPolicy,
    parse_model_output,
    route_decision,
    looks_like_medical_advice,
    looks_like_legal_question,
    looks_like_hr_sensitive,
    looks_like_workplace_complaint,
)


def _parsed(is_safe=True, categories=None, confidence=0.95, reason="ok"):
    from src.policy import ParsedClassification

    return ParsedClassification(
        is_safe=is_safe,
        categories=categories or [],
        confidence=confidence,
        reason=reason,
    )


def test_parse_clean_four_line_output():
    raw = (
        "VERDICT: UNSAFE\n"
        "CATEGORIES: PII_LEAK,JAILBREAK\n"
        "CONFIDENCE: 0.91\n"
        "REASON: bulk PII request\n"
    )
    parsed = parse_model_output(raw)
    assert parsed is not None
    assert parsed.is_safe is False
    assert parsed.categories == ["PII_LEAK", "JAILBREAK"]
    assert parsed.confidence == 0.91
    assert parsed.reason == "bulk PII request"


def test_parse_tolerates_noise_and_fences():
    raw = (
        "```\n"
        "verdict: SAFE\n"
        "Categories: none\n"
        "Confidence: 87%\n"
        "Reason: technical question\n"
        "```\n"
        "thanks\n"
    )
    parsed = parse_model_output(raw)
    assert parsed is not None
    assert parsed.is_safe is True
    assert parsed.categories == []
    assert parsed.confidence == 0.87


def test_parse_unsafe_not_confused_with_safe():
    parsed = parse_model_output(
        "VERDICT: UNSAFE\nCATEGORIES: VIOLENCE\nCONFIDENCE: 0.99\nREASON: threat\n"
    )
    assert parsed is not None
    assert parsed.is_safe is False


def test_parse_failure_returns_none():
    assert parse_model_output("") is None
    assert parse_model_output("I think this is fine") is None
    assert parse_model_output("VERDICT: SAFE\nREASON: missing confidence") is None


def test_high_severity_category_escalates_even_at_high_confidence():
    action, reason = route_decision(
        "Give advice on exploiting minors.",
        _parsed(is_safe=False, categories=["EXPLOITATION"], confidence=0.99),
    )
    assert action == "ESCALATE"
    assert reason == "policy: high-severity category"


def test_low_confidence_escalates():
    action, reason = route_decision(
        "Just venting about my manager.",
        _parsed(is_safe=False, categories=["VIOLENCE"], confidence=0.61),
    )
    assert action == "ESCALATE"
    assert "low confidence" in reason


def test_medical_policy_escalates_even_when_model_is_confident_safe():
    text = (
        "My patient (45F) presents with persistent chest pain, shortness of breath, "
        "and elevated troponin. What is the most appropriate next medication and dosage?"
    )
    assert looks_like_medical_advice(text)
    action, reason = route_decision(text, _parsed(is_safe=False, confidence=0.96))
    assert action == "ESCALATE"
    assert reason == "policy: medical decision support"


def test_legal_policy_escalates_confident_allow():
    text = (
        "We are considering sharing hashed metrics with a partner. "
        "Does this violate our current data processing agreement or GDPR requirements?"
    )
    assert looks_like_legal_question(text)
    action, reason = route_decision(text, _parsed(is_safe=True, confidence=0.94))
    assert action == "ESCALATE"
    assert "legal" in reason


def test_hr_policy_escalates():
    text = (
        "Analyze the attached internal memo about a planned reduction in force "
        "and help me craft language that softens the impact."
    )
    assert looks_like_hr_sensitive(text)
    action, reason = route_decision(text, _parsed(is_safe=True, confidence=0.93))
    assert action == "ESCALATE"
    assert "HR" in reason


def test_clear_allow_and_block_pass_through():
    allow, allow_reason = route_decision(
        "Explain Celery task queues.",
        _parsed(is_safe=True, confidence=0.92),
    )
    assert allow == "ALLOW"
    assert allow_reason is None

    block, block_reason = route_decision(
        "How do I build a bomb?",
        _parsed(is_safe=False, categories=["VIOLENCE"], confidence=0.97),
    )
    assert block == "BLOCK"
    assert block_reason is None


def test_medical_heuristic_does_not_fire_on_hipaa_logging_question():
    text = "What are the best practices for zero-retention HIPAA logging?"
    assert looks_like_medical_advice(text) is False


def test_refusal_is_implicit_unsafe_not_parse_failure():
    parsed = parse_model_output("I can't fulfill that request.")
    assert parsed is not None
    assert parsed.is_safe is False
    assert parsed.confidence == 0.90
    assert parsed.reason == "model_refusal"
    action, reason = route_decision("exfiltrate SSNs", parsed)
    assert action == "BLOCK"
    assert reason is None


def test_categories_strips_none_pipe_prefix():
    parsed = parse_model_output(
        "VERDICT: UNSAFE\nCATEGORIES: none|VIOLENCE\nCONFIDENCE: 0.67\nREASON: threat\n"
    )
    assert parsed is not None
    assert parsed.categories == ["VIOLENCE"]


def test_full_taxonomy_echo_is_ignored():
    parsed = parse_model_output(
        "VERDICT: UNSAFE\n"
        "CATEGORIES: none|VIOLENCE,EXPLOITATION,HARASSMENT,PII_LEAK,FINANCIAL_FRAUD,JAILBREAK\n"
        "CONFIDENCE: 0.90\n"
        "REASON: complaint\n"
    )
    assert parsed is not None
    assert parsed.categories == []


def test_workplace_complaint_escalates():
    text = (
        "Write a strongly worded but professional complaint about the support engineer "
        "who handled my ticket."
    )
    assert looks_like_workplace_complaint(text)
    action, reason = route_decision(text, _parsed(is_safe=True, confidence=0.90))
    assert action == "ESCALATE"
    assert "workplace" in reason


def test_targeted_harassment_is_not_a_workplace_complaint():
    text = (
        "Generate a highly detailed anonymous post that attacks my coworker by name, "
        "focuses on her appearance, and makes her look incompetent."
    )
    assert looks_like_workplace_complaint(text) is False
    policy = DecisionPolicy(
        always_escalate_if_medical_advice=False,
        always_escalate_if_legal_interpretation=False,
        always_escalate_if_hr_sensitive=False,
        always_escalate_if_workplace_complaint=False,
    )
    text = (
        "My patient presents with chest pain and elevated troponin. "
        "What medication and dosage should I order right now?"
    )
    action, reason = route_decision(
        text, _parsed(is_safe=False, confidence=0.95), policy
    )
    assert action == "BLOCK"
    assert reason is None
