#!/usr/bin/env python3
"""Live CLI for the local decision agent. Demo-oriented, not a product.

Same evaluate() path as golden eval: computed confidence + policy + traces.

Usage (from repo root):
    python scripts/moderate_cli.py
    python scripts/moderate_cli.py "post text here"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_TRACE_PATH = ROOT / "results" / "live_traces.jsonl"
HELP_TEXT = """The engine is already loaded. It waits at the prompt.

  Type or paste a post, then press Enter to submit.
  /quit              exit
  /trace on|off      persist JSONL traces (FileTraceStore)
  /help              this text
"""


def parse_repl_command(raw: str) -> tuple[str, Optional[str]]:
    text = raw.strip()
    if not text:
        return ("empty", None)
    if not text.startswith("/"):
        return ("text", text)
    parts = text.split()
    cmd = parts[0].lower()
    if cmd in {"/quit", "/exit", "/q"}:
        return ("quit", None)
    if cmd == "/help":
        return ("help", None)
    if cmd == "/trace":
        arg = parts[1].lower() if len(parts) > 1 else ""
        if arg in {"on", "off"}:
            return ("trace", arg)
        return ("trace_usage", None)
    return ("unknown", cmd)


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def format_decision(decision, *, tracing: bool, trace_path: Optional[Path]) -> str:
    categories = list(getattr(decision, "violated_categories", None) or [])
    conf = float(decision.confidence_score)
    latency = float(decision.inference_time_ms)
    lines = [
        f"action={decision.action}  conf={conf:.3f}  "
        f"source={decision.confidence_source}  {latency:.0f}ms",
        f"categories={categories}",
    ]
    if decision.escalation_reason:
        lines.append(f"escalation_reason={decision.escalation_reason}")
    if decision.why:
        lines.append(f"why={decision.why}")
    if decision.winning_rule:
        lines.append(f"winning_rule={decision.winning_rule}")
    if tracing and trace_path is not None:
        lines.append(f"trace={display_path(trace_path)}")
    return "\n".join(lines)


def resolve_trace_path(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = ROOT / path
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Live ALLOW / BLOCK / ESCALATE from the local decision engine.",
    )
    parser.add_argument(
        "text",
        nargs="?",
        help="Post text (omit for an interactive REPL)",
    )
    parser.add_argument(
        "--trace-path",
        default=str(DEFAULT_TRACE_PATH),
        help="JSONL path for FileTraceStore (default: results/live_traces.jsonl)",
    )
    parser.add_argument(
        "--no-trace",
        action="store_true",
        help="Do not persist traces",
    )
    return parser


def attach_store(engine, tracing: bool, trace_path: Path):
    if not tracing:
        engine.trace_store = None
        return
    from src.trace import FileTraceStore

    engine.trace_store = FileTraceStore(trace_path)


def run_once(engine, text: str, *, tracing: bool, trace_path: Path) -> None:
    decision = engine.evaluate(text)
    print(format_decision(decision, tracing=tracing, trace_path=trace_path))


def repl(engine, *, tracing: bool, trace_path: Path) -> None:
    print("[✓] Waiting for input. Type a post and press Enter to submit.")
    print("    Commands: /quit  /help  /trace on|off", flush=True)
    processed = 0
    while True:
        try:
            raw = input("\npost> ")
        except (EOFError, KeyboardInterrupt):
            print()
            if processed == 0 and not sys.stdin.isatty():
                print(
                    "This CLI waits in a terminal. Run:\n"
                    "  python scripts/moderate_cli.py\n"
                    "Then type a post and press Enter.",
                    file=sys.stderr,
                )
            return
        kind, payload = parse_repl_command(raw)
        if kind == "empty":
            continue
        if kind == "quit":
            return
        if kind == "help":
            print(HELP_TEXT, end="")
            continue
        if kind == "trace_usage":
            print("usage: /trace on|off")
            continue
        if kind == "unknown":
            print(f"unknown command {payload}. Try /help")
            continue
        if kind == "trace":
            tracing = payload == "on"
            attach_store(engine, tracing, trace_path)
            state = "on" if tracing else "off"
            where = f" → {display_path(trace_path)}" if tracing else ""
            print(f"trace {state}{where}")
            continue
        print("[*] Processing...", flush=True)
        run_once(engine, payload, tracing=tracing, trace_path=trace_path)
        processed += 1
        print("[*] Waiting for next post.", flush=True)


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    trace_path = resolve_trace_path(args.trace_path)
    tracing = not args.no_trace

    from src.slm_engine import LocalSLMEngine

    engine = LocalSLMEngine()
    attach_store(engine, tracing, trace_path)

    if args.text is not None:
        run_once(engine, args.text, tracing=tracing, trace_path=trace_path)
        return 0
    repl(engine, tracing=tracing, trace_path=trace_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
