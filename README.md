# AI Moderation Platform

Local decision agent for text moderation. A small model proposes SAFE or UNSAFE. Policy turns that proposal into **ALLOW**, **BLOCK**, or **ESCALATE**, and every decision records why.

This repo is interview evidence for forward-deployed and applied AI work: latency, fail-closed routing, human escalation, and an evaluation set the model never sees. It runs on Apple Silicon with MLX. It is a single-tenant, text-only slice.

Design trade-offs live in [docs/architecture.md](docs/architecture.md).

## Decision path

```
post
  → Llama-3.2-3B (MLX) emits VERDICT / CATEGORIES / CONFIDENCE / REASON
  → token-distribution confidence replaces the model's self-reported score
  → detectors report hits on the user text (they do not choose an action)
  → explain_route() returns ALLOW, BLOCK, or ESCALATE
  → optional JSONL trace
```

Routing after a successful parse:

1. High-severity category (`EXPLOITATION`) escalates, unless detectors show a noisy label on a clear block.
2. Computed confidence below 0.80 escalates.
3. Enabled domain hits escalate: medical, legal, HR, workplace complaint, operational PII, regulatory avoidance.
4. Otherwise SAFE becomes ALLOW and UNSAFE becomes BLOCK. A detector exception refuses ALLOW.

Self-reported `CONFIDENCE` is stored for audit. It does not change the action. The 3B model often emits `0.00` to mean "definitely unsafe."

## Requirements

- Apple Silicon (MLX)
- Python 3.11
- Network on first model load (`mlx-community/Llama-3.2-3B-Instruct-4bit` from Hugging Face)

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install "mlx-lm>=0.31" "pydantic>=2" "pytest>=8"
```

Unit tests import MLX, so they need the same machine. They do not load model weights.

## Moderate a post

From the repo root, with the venv active:

```bash
python scripts/moderate_cli.py
```

The model loads once, then the CLI waits at `post>`. Type a post and press Enter. One-shot:

```bash
python scripts/moderate_cli.py "post text here"
```

Traces append to `results/live_traces.jsonl` unless you pass `--no-trace`. In the REPL, `/trace on|off` toggles that file. `/quit` exits.

Same `evaluate()` path as the golden-set runner.

## Tests

```bash
pytest -q
```

Policy, detectors, confidence, and trace tests do not load the model.

## Golden evaluation

`data/golden_dataset.json` (v0.2, 20 examples) has two labels:

| Label | Question | Values |
|---|---|---|
| `policy_verdict` | Is the content allowed? | ALLOW or BLOCK |
| `expected_system_action` | What should the agent output? | ALLOW, BLOCK, or ESCALATE |

The runner passes only `input_text` into the engine.

```bash
python scripts/run_golden_baseline.py
```

Latest locked run ([results/golden_eval_20260826T031240Z.json](results/golden_eval_20260826T031240Z.json)):

| Metric | Result |
|---|---|
| System action | 20/20 |
| Policy verdict | 17/20 |
| Clear-safe escalations | 0/5 |
| Clear-unsafe escalations | 0/6 |
| Borderline escalations | 9/9 |
| Latency | P50 ~468 ms, P95 ~486 ms |

The three policy-verdict misses are rows the classifier calls SAFE while gold says BLOCK. The routed action on those rows is still ESCALATE.

## Layout

| Path | Role |
|---|---|
| `src/slm_engine.py` | Load, generate, computed confidence, `evaluate()` |
| `src/policy.py` | Parse, `DecisionPolicy`, `explain_route()` |
| `src/detectors.py` | Keyword sensors. Hits only. |
| `src/trace.py` | `DecisionTrace` and append-only JSONL |
| `scripts/moderate_cli.py` | Live REPL |
| `scripts/run_golden_baseline.py` | Golden-set scorecard |
| `data/golden_dataset.json` | Dual-label examples |
| `docs/architecture.md` | Why the routing and confidence choices look like this |

## Limits

- One 3B model. Category labels are noisy. Domain sensors are keywords.
- Golden set is 20 synthetic rows. There is no production traffic.
- Traces store the full post text in a local JSONL file.
- Computed confidence is a second prefill of the same prompt.
- No hosted trace UI, prompt version id, or review queue yet.
