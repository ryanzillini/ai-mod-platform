#!/usr/bin/env python3
"""
Baseline run of the Day-1 LocalSLMEngine against the golden dataset.

IMPORTANT
- This is a *baseline* only. The current engine is still binary (ALLOW/BLOCK)
  with hardcoded confidence. It cannot produce ESCALATE yet.
- The model never sees policy_verdict or expected_system_action.
- Run this on your M4 Max (requires MLX). This environment cannot execute it.

Usage (from repo root on your machine):
    python scripts/run_golden_baseline.py
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is on sys.path when running as a script
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.slm_engine import LocalSLMEngine

DATASET_PATH = ROOT / "data" / "golden_dataset.json"
RESULTS_DIR = ROOT / "results"


def load_dataset():
    with open(DATASET_PATH) as f:
        data = json.load(f)
    return data["examples"]


def main():
    examples = load_dataset()
    print(f"[*] Loaded {len(examples)} golden examples from {DATASET_PATH.name}")
    print("[*] Model will only see input_text — labels are never passed to the engine\n")

    engine = LocalSLMEngine()

    results = []
    latencies = []

    for i, ex in enumerate(examples, 1):
        text = ex["input_text"]
        print(f"[{i:02d}/{len(examples)}] {ex['id']} ... ", end="", flush=True)

        verdict = engine.evaluate(text)

        # Map engine output to policy_verdict space for comparison
        predicted = verdict.verdict  # ALLOW or BLOCK
        gold = ex["policy_verdict"]
        match = predicted == gold

        record = {
            "id": ex["id"],
            "difficulty": ex["difficulty"],
            "policy_verdict": gold,
            "expected_system_action": ex["expected_system_action"],
            "predicted_verdict": predicted,
            "predicted_categories": verdict.violated_categories,
            "confidence_score": verdict.confidence_score,
            "inference_time_ms": verdict.inference_time_ms,
            "policy_match": match,
            # Note: we cannot yet score expected_system_action because
            # the engine has no ESCALATE path.
        }
        results.append(record)
        latencies.append(verdict.inference_time_ms)

        status = "OK" if match else "MISS"
        print(f"{predicted} (gold={gold}) [{status}] {verdict.inference_time_ms:.0f}ms")

    # ---- Summary ----
    total = len(results)
    correct = sum(1 for r in results if r["policy_match"])
    accuracy = correct / total if total else 0.0

    by_difficulty = Counter()
    correct_by_difficulty = Counter()
    for r in results:
        by_difficulty[r["difficulty"]] += 1
        if r["policy_match"]:
            correct_by_difficulty[r["difficulty"]] += 1

    import numpy as np
    p50 = float(np.percentile(latencies, 50))
    p95 = float(np.percentile(latencies, 95))

    print("\n" + "=" * 60)
    print(" BASELINE REPORT (Day-1 binary engine)")
    print("=" * 60)
    print(f" Policy-verdict accuracy : {correct}/{total} = {accuracy:.1%}")
    print(f" Latency                 : P50={p50:.1f}ms  P95={p95:.1f}ms")
    print()
    print(" By difficulty:")
    for diff in ["clear_safe", "clear_unsafe", "borderline"]:
        n = by_difficulty[diff]
        c = correct_by_difficulty[diff]
        if n:
            print(f"   {diff:14s}: {c}/{n} = {c/n:.1%}")
    print()
    print(" NOTE: expected_system_action (ESCALATE) cannot be scored yet.")
    print("       Confidence is still hardcoded. This is a baseline only.")
    print("=" * 60)

    # Persist results
    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = RESULTS_DIR / f"baseline_{stamp}.json"
    payload = {
        "timestamp_utc": stamp,
        "engine": "LocalSLMEngine (Day-1 binary)",
        "dataset_version": "0.2.0",
        "n_examples": total,
        "policy_accuracy": accuracy,
        "latency_p50_ms": p50,
        "latency_p95_ms": p95,
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[✓] Results written to {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
