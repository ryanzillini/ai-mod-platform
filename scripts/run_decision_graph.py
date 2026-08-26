#!/usr/bin/env python3
"""Run one golden example through the LangGraph, including HITL resume.

Default path uses the CI replay classifier (no MLX). On a Mac with mlx-lm:

    python scripts/run_decision_graph.py --example gd-015 --mlx

ESCALATE pauses and waits for a human action on stdin.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval_gate import load_golden, load_replay_records, replay_classifier_for_golden
from src.graph import build_decision_graph, resume_review
from src.observability import configure_langsmith


def _load_mlx_classifier():
    from src.classifier import MlxClassifier
    from src.slm_engine import LocalSLMEngine

    return MlxClassifier(LocalSLMEngine())


def main() -> int:
    parser = argparse.ArgumentParser(description="Decision graph dry-run (HITL on ESCALATE)")
    parser.add_argument("--example", default="gd-015", help="Golden example id")
    parser.add_argument(
        "--action",
        default="",
        help="Human resume action if the graph interrupts (ALLOW|BLOCK|ESCALATE)",
    )
    parser.add_argument("--note", default="cli review", help="Human resume note")
    parser.add_argument(
        "--mlx",
        action="store_true",
        help="Use LocalSLMEngine (Apple Silicon). Default is replay.",
    )
    args = parser.parse_args()

    configure_langsmith()
    version, examples = load_golden()
    by_id = {ex["id"]: ex for ex in examples}
    if args.example not in by_id:
        print(f"Unknown example {args.example}. Dataset v{version} ids: {sorted(by_id)}")
        return 2
    example = by_id[args.example]

    if args.mlx:
        classifier = _load_mlx_classifier()
        model_id = "mlx"
    else:
        classifier = replay_classifier_for_golden(examples, load_replay_records())
        model_id = "replay"

    graph = build_decision_graph(classifier=classifier, model_id=model_id)
    config = {"configurable": {"thread_id": f"cli-{args.example}"}}
    print(f"[*] example={args.example} classifier={model_id}")
    print(f"[*] input: {example['input_text'][:160]}")
    result = graph.invoke(
        {"input_text": example["input_text"], "example_id": args.example},
        config,
    )

    if result.get("__interrupt__"):
        payload = result["__interrupt__"][0].value
        print("[!] INTERRUPT — fail-safe paused for human review")
        print(json.dumps(payload, indent=2))
        action = args.action.strip().upper()
        if not action:
            action = input("Human action [ALLOW|BLOCK|ESCALATE]: ").strip().upper()
        result = resume_review(
            graph, config, {"action": action, "note": args.note}
        )

    print("[✓] final")
    print(
        json.dumps(
            {
                "proposed_action": result.get("action"),
                "final_action": result.get("final_action"),
                "winning_rule": result.get("winning_rule"),
                "why": result.get("why"),
                "human_decision": result.get("human_decision"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
