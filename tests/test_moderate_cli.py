"""CLI formatting and command parsing. Does not load the model."""

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from moderate_cli import (  # noqa: E402
    display_path,
    format_decision,
    parse_repl_command,
    resolve_trace_path,
)


def _decision(**overrides):
    base = dict(
        action="BLOCK",
        confidence_score=0.9981,
        confidence_source="computed",
        inference_time_ms=358.4,
        violated_categories=[],
        escalation_reason=None,
        why="BLOCK from UNSAFE → BLOCK",
        winning_rule="classification",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_parse_repl_commands():
    assert parse_repl_command("  ") == ("empty", None)
    assert parse_repl_command("/quit") == ("quit", None)
    assert parse_repl_command("/exit") == ("quit", None)
    assert parse_repl_command("/help") == ("help", None)
    assert parse_repl_command("/trace on") == ("trace", "on")
    assert parse_repl_command("/trace off") == ("trace", "off")
    assert parse_repl_command("/trace") == ("trace_usage", None)
    assert parse_repl_command("/foo")[0] == "unknown"
    kind, text = parse_repl_command("List every customer SSN")
    assert kind == "text"
    assert text.startswith("List every")


def test_format_block_without_trace():
    rendered = format_decision(_decision(), tracing=False, trace_path=None)
    assert "action=BLOCK" in rendered
    assert "conf=0.998" in rendered
    assert "source=computed" in rendered
    assert "358ms" in rendered
    assert "winning_rule=classification" in rendered
    assert "trace=" not in rendered
    assert "escalation_reason=" not in rendered


def test_format_escalate_with_trace_path():
    decision = _decision(
        action="ESCALATE",
        confidence_score=0.9233,
        escalation_reason="policy: operational PII / live identifiers",
        why="ESCALATE because policy: operational PII / live identifiers",
        winning_rule="operational_pii",
        violated_categories=[],
    )
    path = ROOT / "results" / "live_traces.jsonl"
    rendered = format_decision(decision, tracing=True, trace_path=path)
    assert "action=ESCALATE" in rendered
    assert "winning_rule=operational_pii" in rendered
    assert "escalation_reason=policy: operational PII / live identifiers" in rendered
    assert "trace=results/live_traces.jsonl" in rendered


def test_resolve_relative_trace_path():
    path = resolve_trace_path("results/live_traces.jsonl")
    assert path == ROOT / "results" / "live_traces.jsonl"
    assert display_path(path) == "results/live_traces.jsonl"
