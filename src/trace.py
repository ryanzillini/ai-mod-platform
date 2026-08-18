"""Local decision traces: why this action, persisted as JSONL.

Not OpenTelemetry. One record per decision, append-only, grep-able.
Production would redact input_text and pin a policy/model version.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Literal, Optional

from src.policy import DecisionPolicy, RoutingResult, RoutingStep
from pydantic import BaseModel

Action = Literal["ALLOW", "BLOCK", "ESCALATE"]
PolicyVerdict = Literal["ALLOW", "BLOCK"]
ConfidenceSource = Literal["computed", "self_report_fallback", "parse_failure"]


class TraceStep(BaseModel):
    rule: str
    fired: bool
    detail: str


class DecisionTrace(BaseModel):
    trace_id: str
    timestamp_utc: str
    model_id: str
    example_id: Optional[str] = None
    input_text: str
    policy: dict
    classification: dict
    confidence_source: ConfidenceSource
    steps: List[TraceStep]
    winning_rule: str
    why: str
    action: Action
    policy_verdict: PolicyVerdict
    inference_time_ms: float
    raw_model_output: str = ""


class FileTraceStore:
    """Append-only JSONL store. One line per decision."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, trace: DecisionTrace) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(trace.model_dump_json() + "\n")


def policy_snapshot(policy: DecisionPolicy) -> dict:
    return {
        "confidence_threshold": policy.confidence_threshold,
        "always_escalate_categories": sorted(policy.always_escalate_categories),
        "always_escalate_if_medical_advice": policy.always_escalate_if_medical_advice,
        "always_escalate_if_legal_interpretation": policy.always_escalate_if_legal_interpretation,
        "always_escalate_if_hr_sensitive": policy.always_escalate_if_hr_sensitive,
        "always_escalate_if_workplace_complaint": policy.always_escalate_if_workplace_complaint,
        "always_escalate_if_operational_pii": policy.always_escalate_if_operational_pii,
    }


def parse_failure_result() -> RoutingResult:
    step = RoutingStep("parse_failure", True, "unparseable model output")
    return RoutingResult("ESCALATE", "parse_failure", (step,), "parse_failure")


def build_trace(
    *,
    input_text: str,
    model_id: str,
    policy: DecisionPolicy,
    routing: RoutingResult,
    policy_verdict: PolicyVerdict,
    confidence_source: ConfidenceSource,
    classification: dict,
    inference_time_ms: float,
    raw_model_output: str = "",
    example_id: Optional[str] = None,
) -> DecisionTrace:
    why = _why(routing)
    return DecisionTrace(
        trace_id=uuid.uuid4().hex[:12],
        timestamp_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        model_id=model_id,
        example_id=example_id,
        input_text=input_text,
        policy=policy_snapshot(policy),
        classification=classification,
        confidence_source=confidence_source,
        steps=[TraceStep(rule=s.rule, fired=s.fired, detail=s.detail) for s in routing.steps],
        winning_rule=routing.winning_rule,
        why=why,
        action=routing.action,
        policy_verdict=policy_verdict,
        inference_time_ms=inference_time_ms,
        raw_model_output=raw_model_output,
    )


def format_trace(trace: DecisionTrace) -> str:
    lines = [
        f" TRACE {trace.example_id or trace.trace_id}  action={trace.action}  "
        f"winning_rule={trace.winning_rule}",
        f"   why: {trace.why}",
        f"   source={trace.confidence_source}  "
        f"computed={_fmt(trace.classification.get('computed_confidence'))}  "
        f"self={_fmt(trace.classification.get('self_reported_confidence'))}",
        "   steps:",
    ]
    for step in trace.steps:
        mark = "FIRE" if step.fired else "skip"
        lines.append(f"     - {step.rule:<28} {mark}  {step.detail}")
    return "\n".join(lines)


def load_traces(path: Path) -> list[DecisionTrace]:
    traces = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                traces.append(DecisionTrace.model_validate(json.loads(line)))
    return traces


def _why(routing: RoutingResult) -> str:
    if routing.winning_rule == "parse_failure":
        return "ESCALATE because model output was unparseable (fail closed)"
    if routing.winning_rule == "classification":
        return f"{routing.action} from {routing.steps[-1].detail}"
    reason = routing.escalation_reason or routing.steps[-1].detail
    return f"{routing.action} because {reason}"


def _fmt(value) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}"
