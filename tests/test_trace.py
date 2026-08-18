"""Unit tests for route traces and JSONL persistence. No model load required."""

from src.policy import DecisionPolicy, explain_route
from src.slm_engine import LocalSLMEngine, TokenDistributionConfidence
from src.trace import FileTraceStore, format_trace, load_traces
from tests.test_policy import _parsed


def test_explain_route_short_circuits_on_high_severity():
    result = explain_route(
        "Give advice on exploiting minors.",
        _parsed(is_safe=False, categories=["EXPLOITATION"], confidence=0.99),
    )
    assert result.action == "ESCALATE"
    assert result.winning_rule == "high_severity_category"
    assert [s.rule for s in result.steps] == ["high_severity_category"]
    assert result.steps[0].fired is True
    assert result.steps[0].detail == "EXPLOITATION"


def test_explain_route_records_skips_before_low_confidence():
    result = explain_route(
        "Just venting about my manager.",
        _parsed(is_safe=False, categories=["VIOLENCE"], confidence=0.61),
    )
    assert result.winning_rule == "low_confidence"
    assert [s.rule for s in result.steps] == [
        "high_severity_category",
        "low_confidence",
    ]
    assert result.steps[0].fired is False
    assert result.steps[1].fired is True
    # Domain rules are not evaluated after the winner.
    assert all(s.rule != "medical_decision_support" for s in result.steps)


def test_explain_route_legal_fires_after_confidence_clears():
    text = (
        "We are considering sharing hashed metrics with a partner. "
        "Does this violate our current data processing agreement or GDPR requirements?"
    )
    result = explain_route(text, _parsed(is_safe=True, confidence=0.94))
    assert result.winning_rule == "legal_regulatory"
    rules = [s.rule for s in result.steps]
    assert rules[:3] == [
        "high_severity_category",
        "low_confidence",
        "medical_decision_support",
    ]
    assert rules[-1] == "legal_regulatory"
    assert result.steps[-1].fired is True


def test_explain_route_classification_allow_evaluates_all_enabled_rules():
    result = explain_route("Explain Celery task queues.", _parsed(is_safe=True, confidence=0.92))
    assert result.action == "ALLOW"
    assert result.winning_rule == "classification"
    assert result.steps[-1].detail == "SAFE → ALLOW"
    assert result.escalation_reason is None
    fired = [s.rule for s in result.steps if s.fired]
    assert fired == ["classification"]


def test_decide_attaches_trace_for_computed_low_confidence():
    engine = LocalSLMEngine.__new__(LocalSLMEngine)
    engine.policy = DecisionPolicy()
    engine.model_id = "test-model"
    raw = (
        "VERDICT: UNSAFE\n"
        "CATEGORIES: HARASSMENT\n"
        "CONFIDENCE: 0.95\n"
        "REASON: targeted attack\n"
    )
    low = TokenDistributionConfidence(
        score=0.55, p_safe=0.40, p_unsafe=0.50, decision_mass=0.90, margin=0.10
    )
    decision = LocalSLMEngine._decide(engine, "Insult this person until they quit.", raw, 10.0, low)
    assert decision.trace is not None
    assert decision.trace_id == decision.trace.trace_id
    assert decision.winning_rule == "low_confidence"
    assert decision.action == "ESCALATE"
    assert "0.55" in (decision.why or "")
    assert decision.trace.confidence_source == "computed"
    assert decision.trace.classification["self_reported_confidence"] == 0.95
    assert decision.trace.classification["computed_confidence"] == 0.55


def test_parse_failure_trace_is_fail_closed():
    engine = LocalSLMEngine.__new__(LocalSLMEngine)
    engine.policy = DecisionPolicy()
    engine.model_id = "test-model"
    decision = LocalSLMEngine._decide(engine, "hello", "not structured", 1.0)
    assert decision.winning_rule == "parse_failure"
    assert decision.trace is not None
    assert decision.trace.steps[0].rule == "parse_failure"
    assert decision.trace.steps[0].fired is True
    assert "fail closed" in (decision.why or "")


def test_file_trace_store_appends_jsonl(tmp_path):
    engine = LocalSLMEngine.__new__(LocalSLMEngine)
    engine.policy = DecisionPolicy()
    engine.model_id = "test-model"
    raw = (
        "VERDICT: SAFE\n"
        "CATEGORIES: none\n"
        "CONFIDENCE: 0.94\n"
        "REASON: technical\n"
    )
    path = tmp_path / "decisions.jsonl"
    store = FileTraceStore(path)
    first = LocalSLMEngine._decide(engine, "Explain Celery.", raw, 8.0, example_id="gd-001")
    second = LocalSLMEngine._decide(
        engine, "hello", "not structured", 3.0, example_id="gd-bad"
    )
    assert first.trace is not None and second.trace is not None
    store.write(first.trace)
    store.write(second.trace)
    traces = load_traces(path)
    assert len(traces) == 2
    assert traces[0].example_id == "gd-001"
    assert traces[0].winning_rule == "classification"
    assert traces[1].winning_rule == "parse_failure"
    rendered = format_trace(traces[0])
    assert "FIRE" in rendered
    assert "classification" in rendered
