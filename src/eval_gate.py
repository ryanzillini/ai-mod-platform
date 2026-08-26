"""Golden-set routing judge used as the CI gate.

This does not load MLX. It replays an explicit classification per example,
runs the LangGraph (classify → route → review), and scores:

- predicted proposed action vs expected_system_action
- ESCALATE cases must interrupt (HITL fail-safe)
- ALLOW/BLOCK cases must not interrupt

A policy-flag flip or a removed interrupt fails this gate. Live 3B model
accuracy is a separate Mac-only job (scripts/run_golden_baseline.py).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional
from uuid import uuid4

from src.classifier import ReplayClassifier
from src.graph import build_decision_graph
from src.policy import DecisionPolicy

ROOT = Path(__file__).resolve().parents[1]
GOLDEN_PATH = ROOT / "data" / "golden_dataset.json"
REPLAY_PATH = ROOT / "data" / "ci_replay_classifications.json"


@dataclass
class EvalRow:
    example_id: str
    difficulty: str
    expected_action: str
    predicted_action: Optional[str]
    interrupted: bool
    winning_rule: Optional[str]
    why: Optional[str]
    match: bool
    interrupt_ok: bool


@dataclass
class EvalReport:
    n: int
    hits: int
    action_misses: list[str]
    interrupt_misses: list[str]
    rows: list[EvalRow] = field(default_factory=list)
    dataset_version: str = ""
    notes: str = ""

    @property
    def passed(self) -> bool:
        return not self.action_misses and not self.interrupt_misses

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "n": self.n,
            "hits": self.hits,
            "action_accuracy": (self.hits / self.n) if self.n else 0.0,
            "action_misses": self.action_misses,
            "interrupt_misses": self.interrupt_misses,
            "dataset_version": self.dataset_version,
            "notes": self.notes,
            "rows": [asdict(row) for row in self.rows],
        }


def load_golden(path: Path = GOLDEN_PATH) -> tuple[str, list[dict]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return str(data["version"]), list(data["examples"])


def load_replay_records(path: Path = REPLAY_PATH) -> dict[str, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return dict(data["records"])


def replay_classifier_for_golden(
    examples: list[dict],
    records: Optional[dict[str, dict]] = None,
) -> ReplayClassifier:
    records = records if records is not None else load_replay_records()
    missing = [ex["id"] for ex in examples if ex["id"] not in records]
    if missing:
        raise KeyError(f"Replay fixture missing golden ids: {missing}")
    by_text = {ex["input_text"]: records[ex["id"]] for ex in examples}
    return ReplayClassifier(records, by_text)


def run_routing_eval(
    *,
    policy: Optional[DecisionPolicy] = None,
    examples: Optional[list[dict]] = None,
    records: Optional[dict[str, dict]] = None,
    dataset_version: str = "",
) -> EvalReport:
    if examples is None:
        dataset_version, examples = load_golden()
    classifier = replay_classifier_for_golden(examples, records)
    graph = build_decision_graph(classifier=classifier, policy=policy or DecisionPolicy())

    rows: list[EvalRow] = []
    action_misses: list[str] = []
    interrupt_misses: list[str] = []

    for ex in examples:
        config = {"configurable": {"thread_id": f"{ex['id']}-{uuid4().hex[:8]}"}}
        result = graph.invoke(
            {"input_text": ex["input_text"], "example_id": ex["id"]},
            config,
        )
        interrupted = bool(result.get("__interrupt__"))
        predicted = result.get("action")
        expected = ex["expected_system_action"]
        match = predicted == expected
        if expected == "ESCALATE":
            interrupt_ok = interrupted
        else:
            interrupt_ok = not interrupted
        row = EvalRow(
            example_id=ex["id"],
            difficulty=ex.get("difficulty", ""),
            expected_action=expected,
            predicted_action=predicted,
            interrupted=interrupted,
            winning_rule=result.get("winning_rule"),
            why=result.get("why"),
            match=match,
            interrupt_ok=interrupt_ok,
        )
        rows.append(row)
        if not match:
            action_misses.append(ex["id"])
        if not interrupt_ok:
            interrupt_misses.append(ex["id"])

    hits = sum(1 for row in rows if row.match)
    return EvalReport(
        n=len(rows),
        hits=hits,
        action_misses=action_misses,
        interrupt_misses=interrupt_misses,
        rows=rows,
        dataset_version=dataset_version,
        notes=(
            "Routing contract on replay classifications. "
            "Does not measure live MLX model accuracy."
        ),
    )
