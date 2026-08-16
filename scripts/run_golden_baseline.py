#!/usr/bin/env python3
"""
Golden-set eval for the decision engine (policy verdict + system action).

IMPORTANT
- The model never sees policy_verdict or expected_system_action.
- Scores both classification accuracy (policy_verdict) and routing accuracy
  (expected_system_action).
- Run this on your M4 Max (requires MLX).

Usage (from repo root):
    python scripts/run_golden_baseline.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.slm_engine import LocalSLMEngine

DATASET_PATH = ROOT / "data" / "golden_dataset.json"
RESULTS_DIR = ROOT / "results"

# Highlighted in the confidence/escalation slice success criteria
PREFERENTIAL_ESCALATE_IDS = {"gd-015", "gd-017", "gd-018", "gd-019"}


def load_dataset():
    with open(DATASET_PATH) as f:
        data = json.load(f)
    return data["version"], data["examples"]


def main():
    dataset_version, examples = load_dataset()
    print(f"[*] Loaded {len(examples)} golden examples from {DATASET_PATH.name} (v{dataset_version})")
    print("[*] Model will only see input_text — labels are never passed to the engine\n")

    engine = LocalSLMEngine()

    results = []
    latencies = []

    for i, ex in enumerate(examples, 1):
        text = ex["input_text"]
        print(f"[{i:02d}/{len(examples)}] {ex['id']} ... ", end="", flush=True)

        decision = engine.evaluate(text)

        gold_policy = ex["policy_verdict"]
        gold_action = ex["expected_system_action"]
        policy_match = decision.policy_verdict == gold_policy
        action_match = decision.action == gold_action

        record = {
            "id": ex["id"],
            "difficulty": ex["difficulty"],
            "policy_verdict": gold_policy,
            "expected_system_action": gold_action,
            "predicted_policy_verdict": decision.policy_verdict,
            "predicted_action": decision.action,
            "predicted_categories": decision.violated_categories,
            "confidence_score": decision.confidence_score,
            "reason": decision.reason,
            "escalation_reason": decision.escalation_reason,
            "inference_time_ms": decision.inference_time_ms,
            "policy_match": policy_match,
            "action_match": action_match,
            "raw_model_output": decision.raw_model_output,
        }
        results.append(record)
        latencies.append(decision.inference_time_ms)

        flags = []
        flags.append("P" if policy_match else "pMISS")
        flags.append("A" if action_match else "aMISS")
        extra = f" esc={decision.escalation_reason}" if decision.escalation_reason else ""
        print(
            f"policy={decision.policy_verdict} action={decision.action} "
            f"(gold={gold_policy}/{gold_action}) [{'+'.join(flags)}] "
            f"conf={decision.confidence_score:.2f} {decision.inference_time_ms:.0f}ms{extra}"
        )

    _print_report(results, latencies, dataset_version)
    _write_results(results, latencies, dataset_version)


def _print_report(results, latencies, dataset_version):
    import numpy as np

    total = len(results)
    policy_correct = sum(1 for r in results if r["policy_match"])
    action_correct = sum(1 for r in results if r["action_match"])
    policy_acc = policy_correct / total if total else 0.0
    action_acc = action_correct / total if total else 0.0

    by_diff = Counter()
    policy_by_diff = Counter()
    action_by_diff = Counter()
    escalate_by_diff = Counter()
    for r in results:
        d = r["difficulty"]
        by_diff[d] += 1
        if r["policy_match"]:
            policy_by_diff[d] += 1
        if r["action_match"]:
            action_by_diff[d] += 1
        if r["predicted_action"] == "ESCALATE":
            escalate_by_diff[d] += 1

    p50 = float(np.percentile(latencies, 50))
    p95 = float(np.percentile(latencies, 95))

    print("\n" + "=" * 64)
    print(" GOLDEN EVAL REPORT (confidence + escalation routing)")
    print("=" * 64)
    print(f" Dataset                  : v{dataset_version}  n={total}")
    print(f" Policy-verdict accuracy  : {policy_correct}/{total} = {policy_acc:.1%}")
    print(f" System-action accuracy   : {action_correct}/{total} = {action_acc:.1%}")
    print(f" Latency                  : P50={p50:.1f}ms  P95={p95:.1f}ms")
    print()
    print(" By difficulty (policy / action / escalate rate):")
    for diff in ["clear_safe", "clear_unsafe", "borderline"]:
        n = by_diff[diff]
        if not n:
            continue
        esc = escalate_by_diff[diff]
        print(
            f"   {diff:14s}: policy {policy_by_diff[diff]}/{n} = {policy_by_diff[diff]/n:.1%}   "
            f"action {action_by_diff[diff]}/{n} = {action_by_diff[diff]/n:.1%}   "
            f"escalate {esc}/{n} = {esc/n:.1%}"
        )

    print()
    print(" Preferential escalate cases (gd-015, gd-017, gd-018, gd-019):")
    for r in results:
        if r["id"] not in PREFERENTIAL_ESCALATE_IDS:
            continue
        hit = "ESCALATE" if r["predicted_action"] == "ESCALATE" else r["predicted_action"]
        why = r["escalation_reason"] or "-"
        print(f"   {r['id']}: action={hit}  conf={r['confidence_score']:.2f}  reason={why}")

    print()
    print(" Mismatches:")
    misses = [r for r in results if not r["policy_match"] or not r["action_match"]]
    if not misses:
        print("   (none)")
    for r in misses:
        bits = []
        if not r["policy_match"]:
            bits.append(f"policy {r['predicted_policy_verdict']}!={r['policy_verdict']}")
        if not r["action_match"]:
            bits.append(f"action {r['predicted_action']}!={r['expected_system_action']}")
        print(f"   {r['id']} ({r['difficulty']}): {', '.join(bits)}")
    print("=" * 64)


def _write_results(results, latencies, dataset_version):
    import numpy as np

    total = len(results)
    policy_acc = sum(1 for r in results if r["policy_match"]) / total if total else 0.0
    action_acc = sum(1 for r in results if r["action_match"]) / total if total else 0.0
    p50 = float(np.percentile(latencies, 50))
    p95 = float(np.percentile(latencies, 95))

    escalate_by_diff = {}
    for diff in ["clear_safe", "clear_unsafe", "borderline"]:
        subset = [r for r in results if r["difficulty"] == diff]
        n = len(subset)
        esc = sum(1 for r in subset if r["predicted_action"] == "ESCALATE")
        escalate_by_diff[diff] = {"n": n, "escalated": esc, "rate": (esc / n) if n else 0.0}

    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = RESULTS_DIR / f"golden_eval_{stamp}.json"
    payload = {
        "timestamp_utc": stamp,
        "engine": "LocalSLMEngine (structured confidence + DecisionPolicy)",
        "dataset_version": dataset_version,
        "n_examples": total,
        "policy_accuracy": policy_acc,
        "action_accuracy": action_acc,
        "latency_p50_ms": p50,
        "latency_p95_ms": p95,
        "escalation_by_difficulty": escalate_by_diff,
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[✓] Results written to {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
