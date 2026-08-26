"""LangGraph wrapper around the existing policy + trace path.

Nodes: classify → route → review → END

review calls interrupt() when the proposed action is ESCALATE. That is the
fail-safe / HITL path. ALLOW and BLOCK complete without a human.

A checkpointer is required for interrupt/resume. Tests and CI use
InMemorySaver (process-local). A paused thread does not survive process
restart here — production would swap in a durable saver (SQLite/Postgres).
"""

from __future__ import annotations

from typing import Any, Optional, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from src.classifier import Classifier, ClassifyOutput
from src.policy import DecisionPolicy
from src.slm_engine import LocalSLMEngine, TokenDistributionConfidence

VALID_ACTIONS = frozenset({"ALLOW", "BLOCK", "ESCALATE"})


class DecisionState(TypedDict, total=False):
    input_text: str
    example_id: Optional[str]
    raw_model_output: str
    inference_time_ms: float
    computed: Optional[dict]
    action: str
    policy_verdict: str
    confidence_score: float
    confidence_source: str
    winning_rule: Optional[str]
    why: Optional[str]
    escalation_reason: Optional[str]
    human_decision: Optional[str]
    human_note: Optional[str]
    final_action: Optional[str]


def _as_computed(payload: Optional[dict]) -> Optional[TokenDistributionConfidence]:
    if not payload:
        return None
    return TokenDistributionConfidence(
        score=float(payload["score"]),
        p_safe=float(payload["p_safe"]),
        p_unsafe=float(payload["p_unsafe"]),
        decision_mass=float(payload["decision_mass"]),
        margin=float(payload["margin"]),
    )


def _computed_payload(
    computed: Optional[TokenDistributionConfidence],
) -> Optional[dict]:
    if computed is None:
        return None
    return {
        "score": computed.score,
        "p_safe": computed.p_safe,
        "p_unsafe": computed.p_unsafe,
        "decision_mass": computed.decision_mass,
        "margin": computed.margin,
    }


def build_decision_graph(
    *,
    classifier: Classifier,
    policy: Optional[DecisionPolicy] = None,
    checkpointer: Any = None,
    model_id: str = "replay",
):
    """Compile classify → route → review with a checkpointer."""
    policy = policy or DecisionPolicy()
    if checkpointer is None:
        checkpointer = InMemorySaver()

    def classify_node(state: DecisionState) -> dict:
        result: ClassifyOutput = classifier.classify(
            state["input_text"], state.get("example_id")
        )
        return {
            "raw_model_output": result.raw_model_output,
            "inference_time_ms": result.inference_time_ms,
            "computed": _computed_payload(result.computed),
        }

    def route_node(state: DecisionState) -> dict:
        engine = LocalSLMEngine.__new__(LocalSLMEngine)
        engine.policy = policy
        engine.model_id = model_id
        decision = LocalSLMEngine._decide(
            engine,
            state["input_text"],
            state.get("raw_model_output") or "",
            float(state.get("inference_time_ms") or 0.0),
            _as_computed(state.get("computed")),
            state.get("example_id"),
        )
        return {
            "action": decision.action,
            "policy_verdict": decision.policy_verdict,
            "confidence_score": decision.confidence_score,
            "confidence_source": decision.confidence_source,
            "winning_rule": decision.winning_rule,
            "why": decision.why,
            "escalation_reason": decision.escalation_reason,
        }

    def review_node(state: DecisionState) -> dict:
        proposed = state.get("action") or "ESCALATE"
        if proposed != "ESCALATE":
            return {
                "human_decision": None,
                "human_note": None,
                "final_action": proposed,
            }
        decision = interrupt(
            {
                "proposed_action": "ESCALATE",
                "example_id": state.get("example_id"),
                "input_text": state.get("input_text"),
                "policy_verdict": state.get("policy_verdict"),
                "winning_rule": state.get("winning_rule"),
                "why": state.get("why"),
                "escalation_reason": state.get("escalation_reason"),
            }
        )
        if isinstance(decision, dict):
            human_action = str(decision.get("action") or "ESCALATE")
            note = str(decision.get("note") or "")
        else:
            human_action = str(decision)
            note = ""
        if human_action not in VALID_ACTIONS:
            human_action = "ESCALATE"
            note = (note + " invalid human action; fail closed").strip()
        return {
            "human_decision": human_action,
            "human_note": note,
            "final_action": human_action,
        }

    graph = StateGraph(DecisionState)
    graph.add_node("classify", classify_node)
    graph.add_node("route", route_node)
    graph.add_node("review", review_node)
    graph.add_edge(START, "classify")
    graph.add_edge("classify", "route")
    graph.add_edge("route", "review")
    graph.add_edge("review", END)
    return graph.compile(checkpointer=checkpointer)


def invoke_until_interrupt_or_end(graph, payload: dict, config: dict) -> dict:
    """Run once. Caller inspects `__interrupt__` or `final_action`."""
    return graph.invoke(payload, config)


def resume_review(graph, config: dict, decision: dict) -> dict:
    """Resume a paused ESCALATE thread with a human decision."""
    return graph.invoke(Command(resume=decision), config)
