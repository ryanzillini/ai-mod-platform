"""LangGraph checkpointer + interrupt/HITL tests. No model load."""

from src.classifier import ReplayClassifier, format_structured_output
from src.graph import build_decision_graph, resume_review
from src.policy import DecisionPolicy


def _classifier(record: dict, example_id: str = "x") -> ReplayClassifier:
    return ReplayClassifier({example_id: record})


def test_allow_completes_without_interrupt():
    graph = build_decision_graph(
        classifier=_classifier(
            {
                "is_safe": True,
                "categories": [],
                "confidence": 0.95,
                "reason": "technical",
            }
        )
    )
    config = {"configurable": {"thread_id": "allow-1"}}
    result = graph.invoke(
        {"input_text": "Explain Celery task queues.", "example_id": "x"},
        config,
    )
    assert "__interrupt__" not in result
    assert result["action"] == "ALLOW"
    assert result["final_action"] == "ALLOW"
    assert result["winning_rule"] == "classification"


def test_escalate_interrupts_and_human_can_override():
    text = (
        "My patient (45F) presents with persistent chest pain, shortness of breath, "
        "and elevated troponin. What is the most appropriate next medication and dosage?"
    )
    graph = build_decision_graph(
        classifier=_classifier(
            {
                "is_safe": False,
                "categories": [],
                "confidence": 0.96,
                "reason": "clinical",
            }
        )
    )
    config = {"configurable": {"thread_id": "hitl-1"}}
    paused = graph.invoke({"input_text": text, "example_id": "x"}, config)
    assert paused["action"] == "ESCALATE"
    assert paused.get("final_action") is None
    interrupts = paused["__interrupt__"]
    assert interrupts
    payload = interrupts[0].value
    assert payload["proposed_action"] == "ESCALATE"
    assert payload["winning_rule"] == "medical_decision_support"

    snapshot = graph.get_state(config)
    assert snapshot.next == ("review",)

    resumed = resume_review(
        graph, config, {"action": "BLOCK", "note": "clinician blocked"}
    )
    assert resumed["final_action"] == "BLOCK"
    assert resumed["human_decision"] == "BLOCK"
    assert resumed["human_note"] == "clinician blocked"
    assert graph.get_state(config).next == ()


def test_invalid_human_action_fails_closed():
    text = (
        "Does this violate our current data processing agreement or GDPR requirements?"
    )
    graph = build_decision_graph(
        classifier=_classifier(
            {
                "is_safe": True,
                "categories": [],
                "confidence": 0.94,
                "reason": "legal",
            }
        )
    )
    config = {"configurable": {"thread_id": "hitl-bad"}}
    graph.invoke({"input_text": text, "example_id": "x"}, config)
    resumed = resume_review(graph, config, {"action": "SHIP_IT", "note": "typo"})
    assert resumed["final_action"] == "ESCALATE"
    assert "fail closed" in resumed["human_note"]


def test_checkpointer_is_required_for_resume_across_invokes():
    from langgraph.checkpoint.memory import InMemorySaver

    saver = InMemorySaver()
    record = {
        "is_safe": True,
        "categories": [],
        "confidence": 0.92,
        "reason": "operational pii",
    }
    text = (
        "Draft an internal email that includes the client's full account number "
        "4521-8832-0199 and their date of birth so the ops team can process "
        "the wire transfer today."
    )
    graph = build_decision_graph(
        classifier=_classifier(record),
        checkpointer=saver,
        policy=DecisionPolicy(),
    )
    config = {"configurable": {"thread_id": "ckpt-1"}}
    graph.invoke({"input_text": text, "example_id": "x"}, config)
    # Same compiled graph + same thread_id resumes from the saver.
    resumed = resume_review(graph, config, {"action": "ESCALATE", "note": "keep hold"})
    assert resumed["final_action"] == "ESCALATE"


def test_structured_output_helper_is_parseable():
    from src.policy import parse_model_output

    raw = format_structured_output(
        is_safe=False, categories=["PII_LEAK"], confidence=0.91, reason="bulk"
    )
    parsed = parse_model_output(raw)
    assert parsed is not None
    assert parsed.is_safe is False
    assert parsed.categories == ["PII_LEAK"]
