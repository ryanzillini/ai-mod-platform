#!/usr/bin/env python3
"""CI routing gate against the golden set. No MLX. No LangSmith key required."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval_gate import run_routing_eval
from src.observability import configure_langsmith, export_eval_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Golden-set LangGraph routing gate")
    parser.add_argument(
        "--export-langsmith",
        action="store_true",
        help="Best-effort LangSmith export when LANGSMITH_API_KEY is set",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full report JSON after the human-readable summary",
    )
    args = parser.parse_args()

    tracing = configure_langsmith()
    print("[*] Golden routing eval (replay classifier + LangGraph)")
    print(f"[*] LangSmith tracing: {'on' if tracing else 'off (no LANGSMITH_API_KEY)'}")

    report = run_routing_eval()
    print()
    print("=" * 64)
    print(" CI EVAL GATE")
    print("=" * 64)
    print(f" Dataset           : v{report.dataset_version}  n={report.n}")
    print(f" Action matches    : {report.hits}/{report.n}")
    print(f" Action misses     : {report.action_misses or '(none)'}")
    print(f" Interrupt misses  : {report.interrupt_misses or '(none)'}")
    print(f" Result            : {'PASS' if report.passed else 'FAIL'}")
    print()
    print(" Rows:")
    for row in report.rows:
        mark = "OK" if row.match and row.interrupt_ok else "MISS"
        pause = "interrupt" if row.interrupted else "pass-through"
        print(
            f"   {row.example_id:<8} {mark:<4}  "
            f"pred={row.predicted_action:<8} gold={row.expected_action:<8}  "
            f"{pause:<12}  rule={row.winning_rule or '-'}"
        )
    print("=" * 64)
    print(report.notes)

    if args.export_langsmith:
        url = export_eval_report(report)
        if url:
            print(f"[✓] LangSmith export attempted: {url}")
        else:
            print("[*] LangSmith export skipped (no key, or export failed)")

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))

    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
