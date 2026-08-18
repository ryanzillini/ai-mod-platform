#!/usr/bin/env python3
"""
Golden-set eval for the decision engine (policy verdict + system action).

Live routing uses token-distribution confidence. Self-reported CONFIDENCE is
recorded for audit only and does not drive the action.

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
from statistics import mean, pstdev

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.slm_engine import LocalSLMEngine
from src.trace import FileTraceStore, format_trace

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

    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    trace_path = RESULTS_DIR / f"golden_traces_{stamp}.jsonl"
    engine = LocalSLMEngine(trace_store=FileTraceStore(trace_path))

    results = []
    latencies = []
    traces = []

    for i, ex in enumerate(examples, 1):
        text = ex["input_text"]
        print(f"[{i:02d}/{len(examples)}] {ex['id']} ... ", end="", flush=True)

        decision = engine.evaluate(text, example_id=ex["id"])

        gold_policy = ex["policy_verdict"]
        gold_action = ex["expected_system_action"]
        policy_match = decision.policy_verdict == gold_policy
        action_live = decision.action
        action_self = decision.action_if_self_report or decision.action
        action_comp = decision.action_if_computed or decision.action
        action_match = action_live == gold_action
        action_match_self = action_self == gold_action
        action_match_comp = action_comp == gold_action
        steps = []
        if decision.trace is not None:
            traces.append(decision.trace)
            steps = [s.model_dump() for s in decision.trace.steps]

        record = {
            "id": ex["id"],
            "difficulty": ex["difficulty"],
            "policy_verdict": gold_policy,
            "expected_system_action": gold_action,
            "predicted_policy_verdict": decision.policy_verdict,
            "predicted_action": action_live,
            "confidence_source": decision.confidence_source,
            "action_if_self_report": action_self,
            "action_if_computed": action_comp,
            "predicted_categories": decision.violated_categories,
            "confidence_score": decision.confidence_score,
            "self_reported_confidence": decision.self_reported_confidence,
            "computed_confidence": decision.computed_confidence,
            "p_safe": decision.p_safe,
            "p_unsafe": decision.p_unsafe,
            "decision_mass": decision.decision_mass,
            "reason": decision.reason,
            "escalation_reason": decision.escalation_reason,
            "escalation_reason_self_report": decision.escalation_reason_self_report,
            "escalation_reason_computed": decision.escalation_reason_computed,
            "trace_id": decision.trace_id,
            "winning_rule": decision.winning_rule,
            "why": decision.why,
            "trace_steps": steps,
            "inference_time_ms": decision.inference_time_ms,
            "policy_match": policy_match,
            "action_match": action_match,
            "action_match_self_report": action_match_self,
            "action_match_computed": action_match_comp,
            "raw_model_output": decision.raw_model_output,
        }
        results.append(record)
        latencies.append(decision.inference_time_ms)

        flags = []
        flags.append("P" if policy_match else "pMISS")
        flags.append("A" if action_match else "aMISS")
        extra = f" esc={decision.escalation_reason}" if decision.escalation_reason else ""
        print(
            f"policy={decision.policy_verdict} action={action_live} "
            f"(gold={gold_policy}/{gold_action}) [{'+'.join(flags)}] "
            f"conf={decision.confidence_score:.2f} src={decision.confidence_source} "
            f"rule={decision.winning_rule} {decision.inference_time_ms:.0f}ms{extra}"
        )

    _print_report(results, latencies, dataset_version)
    _print_traces(traces, results)
    _print_comparison(results)
    _write_results(results, latencies, dataset_version, stamp, trace_path)


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
    print(" GOLDEN EVAL REPORT (computed confidence locked)")
    print("=" * 64)
    print(f" Dataset                  : v{dataset_version}  n={total}")
    print(f" Policy-verdict accuracy  : {policy_correct}/{total} = {policy_acc:.1%}")
    print(f" System-action accuracy   : {action_correct}/{total} = {action_acc:.1%}")
    print(f" Latency (gen + logprob)  : P50={p50:.1f}ms  P95={p95:.1f}ms")
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
        why = r["escalation_reason"] or "-"
        print(
            f"   {r['id']}: action={r['predicted_action']}  "
            f"conf={r['confidence_score']:.2f}  rule={r.get('winning_rule') or '-'}  "
            f"reason={why}"
        )

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


def _print_traces(traces, results):
    by_id = {t.example_id: t for t in traces if t.example_id}
    highlight = []
    for r in results:
        if r["id"] in PREFERENTIAL_ESCALATE_IDS or not r["action_match"] or not r["policy_match"]:
            highlight.append(r["id"])
    print("\n" + "=" * 64)
    print(" DECISION TRACES (preferential + mismatches)")
    print("=" * 64)
    for example_id in highlight:
        trace = by_id.get(example_id)
        if trace is None:
            continue
        print(format_trace(trace))
        print()
    print("=" * 64)


def _print_comparison(results):
    total = len(results)
    acc_self = sum(1 for r in results if r["action_match_self_report"])
    acc_comp = sum(1 for r in results if r["action_match_computed"])

    print("\n" + "=" * 88)
    print(" AUDIT: locked computed vs self-report (self-report does not drive action)")
    print("=" * 88)
    print(
        f" Action accuracy vs expected_system_action:\n"
        f"   self-report : {acc_self}/{total} = {acc_self / total:.1%}\n"
        f"   computed    : {acc_comp}/{total} = {acc_comp / total:.1%}"
    )

    print()
    print(" Side-by-side (all 20):")
    print(
        f"   {'id':<8} {'diff':<14} {'self':>5} {'comp':>5} {'mass':>5}  "
        f"{'act_self':<9} {'act_comp':<9} {'gold':<9} {'agree':<7}"
    )
    for r in results:
        self_c = r["self_reported_confidence"]
        comp_c = r["computed_confidence"]
        mass = r["decision_mass"]
        agree = "both" if r["action_if_self_report"] == r["action_if_computed"] else "DIFF"
        gold_hit_self = "✓" if r["action_match_self_report"] else "·"
        gold_hit_comp = "✓" if r["action_match_computed"] else "·"
        print(
            f"   {r['id']:<8} {r['difficulty']:<14} "
            f"{_fmt(self_c):>5} {_fmt(comp_c):>5} {_fmt(mass):>5}  "
            f"{r['action_if_self_report']:<9} {str(r['action_if_computed']):<9} "
            f"{r['expected_system_action']:<9} {agree:<7} {gold_hit_self}{gold_hit_comp}"
        )

    print()
    print(" 1. Spread (mean / std / min–max) by difficulty:")
    for diff in ["clear_safe", "clear_unsafe", "borderline"]:
        subset = [r for r in results if r["difficulty"] == diff]
        if not subset:
            continue
        print(f"   {diff}:")
        for label, key in (
            ("self-report", "self_reported_confidence"),
            ("computed   ", "computed_confidence"),
        ):
            stats = _spread([r[key] for r in subset])
            print(f"      {label}  {stats}")

    print()
    print(" 2. Action accuracy by difficulty:")
    for diff in ["clear_safe", "clear_unsafe", "borderline"]:
        subset = [r for r in results if r["difficulty"] == diff]
        if not subset:
            continue
        n = len(subset)
        s = sum(1 for r in subset if r["action_match_self_report"])
        c = sum(1 for r in subset if r["action_match_computed"])
        esc_s = sum(1 for r in subset if r["action_if_self_report"] == "ESCALATE")
        esc_c = sum(1 for r in subset if r["action_if_computed"] == "ESCALATE")
        print(
            f"   {diff:14s}: self {s}/{n} = {s/n:.1%} (esc {esc_s}/{n})   "
            f"computed {c}/{n} = {c/n:.1%} (esc {esc_c}/{n})"
        )

    print()
    print(" 3. Preferential cases (gd-015 / 017 / 018 / 019):")
    for r in results:
        if r["id"] not in PREFERENTIAL_ESCALATE_IDS:
            continue
        print(
            f"   {r['id']}: gold={r['expected_system_action']}  "
            f"self={r['action_if_self_report']}@{_fmt(r['self_reported_confidence'])} "
            f"({r['escalation_reason_self_report'] or '-'})  "
            f"computed={r['action_if_computed']}@{_fmt(r['computed_confidence'])} "
            f"({r['escalation_reason_computed'] or '-'})"
        )

    print()
    print(" 4. Honesty check (mean confidence):")
    for diff in ["clear_safe", "clear_unsafe", "borderline"]:
        subset = [r for r in results if r["difficulty"] == diff]
        if not subset:
            continue
        self_m = _mean([r["self_reported_confidence"] for r in subset])
        comp_m = _mean([r["computed_confidence"] for r in subset])
        mass_m = _mean([r["decision_mass"] for r in subset])
        print(
            f"   {diff:14s}: self-report {self_m}   computed {comp_m}   "
            f"decision_mass {mass_m}"
        )

    diffs = [
        r
        for r in results
        if r["action_if_self_report"] != r["action_if_computed"]
    ]
    print()
    print(" Action disagreements (self vs computed):")
    if not diffs:
        print("   (none — both sources produced the same action on every example)")
    for r in diffs:
        winner = []
        if r["action_match_self_report"] and not r["action_match_computed"]:
            winner.append("self matches gold")
        elif r["action_match_computed"] and not r["action_match_self_report"]:
            winner.append("computed matches gold")
        elif r["action_match_self_report"] and r["action_match_computed"]:
            winner.append("both match gold")
        else:
            winner.append("neither matches gold")
        print(
            f"   {r['id']} ({r['difficulty']}): "
            f"self={r['action_if_self_report']}@{_fmt(r['self_reported_confidence'])}  "
            f"computed={r['action_if_computed']}@{_fmt(r['computed_confidence'])}  "
            f"gold={r['expected_system_action']}  [{winner[0]}]"
        )
    print("=" * 88)


def _fmt(value) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}"


def _mean(values) -> str:
    nums = [v for v in values if v is not None]
    if not nums:
        return "n/a"
    return f"{mean(nums):.2f}"


def _spread(values) -> str:
    nums = [v for v in values if v is not None]
    if not nums:
        return "n/a"
    std = pstdev(nums) if len(nums) > 1 else 0.0
    return f"mean={mean(nums):.2f}  std={std:.2f}  range={min(nums):.2f}–{max(nums):.2f}"


def _comparison_payload(results):
    total = len(results)
    by_diff = {}
    for diff in ["clear_safe", "clear_unsafe", "borderline"]:
        subset = [r for r in results if r["difficulty"] == diff]
        n = len(subset)
        by_diff[diff] = {
            "n": n,
            "self_report": _source_stats(subset, "self_reported_confidence", "action_if_self_report", "action_match_self_report"),
            "computed": _source_stats(subset, "computed_confidence", "action_if_computed", "action_match_computed"),
        }
    preferential = [
        {
            "id": r["id"],
            "gold": r["expected_system_action"],
            "self_report": {
                "confidence": r["self_reported_confidence"],
                "action": r["action_if_self_report"],
                "escalation_reason": r["escalation_reason_self_report"],
            },
            "computed": {
                "confidence": r["computed_confidence"],
                "action": r["action_if_computed"],
                "escalation_reason": r["escalation_reason_computed"],
            },
        }
        for r in results
        if r["id"] in PREFERENTIAL_ESCALATE_IDS
    ]
    disagreements = [
        {
            "id": r["id"],
            "difficulty": r["difficulty"],
            "self_report": r["action_if_self_report"],
            "computed": r["action_if_computed"],
            "gold": r["expected_system_action"],
        }
        for r in results
        if r["action_if_self_report"] != r["action_if_computed"]
    ]
    return {
        "formula": "max(p_SAFE, p_UNSAFE) / (p_SAFE + p_UNSAFE)",
        "threshold": 0.80,
        "locked_source": "computed",
        "action_accuracy_self_report": (sum(1 for r in results if r["action_match_self_report"]) / total)
        if total
        else 0.0,
        "action_accuracy_computed": (sum(1 for r in results if r["action_match_computed"]) / total)
        if total
        else 0.0,
        "by_difficulty": by_diff,
        "preferential": preferential,
        "action_disagreements": disagreements,
    }


def _source_stats(subset, conf_key, action_key, match_key):
    n = len(subset)
    confs = [r[conf_key] for r in subset if r[conf_key] is not None]
    esc = sum(1 for r in subset if r[action_key] == "ESCALATE")
    hits = sum(1 for r in subset if r[match_key])
    return {
        "action_accuracy": (hits / n) if n else 0.0,
        "escalate_rate": (esc / n) if n else 0.0,
        "confidence_mean": mean(confs) if confs else None,
        "confidence_std": pstdev(confs) if len(confs) > 1 else 0.0,
        "confidence_min": min(confs) if confs else None,
        "confidence_max": max(confs) if confs else None,
    }


def _write_results(results, latencies, dataset_version, stamp, trace_path):
    import numpy as np

    total = len(results)
    policy_acc = sum(1 for r in results if r["policy_match"]) / total if total else 0.0
    action_acc = (
        sum(1 for r in results if r["action_match"]) / total if total else 0.0
    )
    p50 = float(np.percentile(latencies, 50))
    p95 = float(np.percentile(latencies, 95))

    escalate_by_diff = {}
    for diff in ["clear_safe", "clear_unsafe", "borderline"]:
        subset = [r for r in results if r["difficulty"] == diff]
        n = len(subset)
        esc = sum(1 for r in subset if r["predicted_action"] == "ESCALATE")
        escalate_by_diff[diff] = {"n": n, "escalated": esc, "rate": (esc / n) if n else 0.0}

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"golden_eval_{stamp}.json"
    payload = {
        "timestamp_utc": stamp,
        "engine": "LocalSLMEngine (computed confidence + decision traces)",
        "dataset_version": dataset_version,
        "n_examples": total,
        "policy_accuracy": policy_acc,
        "action_accuracy": action_acc,
        "latency_p50_ms": p50,
        "latency_p95_ms": p95,
        "escalation_by_difficulty": escalate_by_diff,
        "traces_path": str(trace_path.relative_to(ROOT)),
        "comparison": _comparison_payload(results),
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[✓] Results written to {out_path.relative_to(ROOT)}")
    print(f"[✓] Traces written to {trace_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
