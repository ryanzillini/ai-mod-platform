# Decision agent — LangGraph fail-safe + CI eval

Public technical artifact for Ryan Zillini. It is **not a product**, not a traffic claim, and not a restaged Khoros/IgniteTech demo. Production moderation work stays on the resume. This repo is the inspectable proof that a small agent can have a real graph, a real human-in-the-loop pause, and an eval that fails CI when routing regresses.

## Two-minute read

**What the graph does.** `classify → route → review`. Classify is either a local MLX SLM (Mac) or a replay classifier (CI/Linux). Route is the existing separable policy: high-severity category, low confidence, then medical / legal / HR / workplace / operational-PII heuristics, else ALLOW or BLOCK. Review is a LangGraph `interrupt()` when the proposed action is `ESCALATE`.

**Where the fail-safe is.** ESCALATE does not auto-complete. The graph checkpoints and waits. A human resumes with `ALLOW`, `BLOCK`, or `ESCALATE`. Invalid resume values fail closed to `ESCALATE`. ALLOW/BLOCK do not pause.

**How the eval gates CI.** GitHub Actions installs the package (no `LANGSMITH_API_KEY`, no MLX) and runs `pytest` plus `python scripts/run_ci_eval.py`. The gate replays an explicit classification for each of the 20 golden examples, scores `expected_system_action`, and requires every gold `ESCALATE` to interrupt. Disabling the medical rule makes `gd-015` fail the gate. That is the regression the CI job is built to catch.

```
input text
    │
    ▼
classify   MLX on a Mac, replay fixture on CI
    │
    ▼
route      DecisionPolicy  →  ALLOW | BLOCK | ESCALATE
    │
    ▼
review     interrupt() only if ESCALATE, then human resume
    │
    ▼
final action
```

## Cloud vs local (honest miss)

| Path | What actually runs |
|---|---|
| This Linux CI / Cloud agent | Policy + LangGraph + `InMemorySaver` + replay classifications. Tests and the eval gate. |
| Ryan's Mac (Apple Silicon) | Same graph can wrap `LocalSLMEngine` (`--mlx`). Live golden accuracy is still `scripts/run_golden_baseline.py`. Last recorded live run: 17/20 action (known misses gd-007 / gd-010 category over-fire, gd-016 certain-UNSAFE vs intent-ESCALATE). |
| LangSmith | Off unless `LANGSMITH_API_KEY` is set. The repo must, and does, run without it. |
| Checkpointer | `InMemorySaver` is real and required for interrupt/resume. It is process-local. A paused HITL thread does **not** survive process restart. Production would swap in SQLite/Postgres; that swap is not shipped here. |

## Run it

```bash
python -m pip install -e ".[dev]"
pytest -q
python scripts/run_ci_eval.py
python scripts/run_decision_graph.py --example gd-015 --action BLOCK
```

Mac-only live model (not used in CI):

```bash
python -m pip install -e ".[mlx]"
python scripts/run_golden_baseline.py
python scripts/run_decision_graph.py --example gd-015 --mlx
```

Optional LangSmith (local or a private CI secret — never required):

```bash
export LANGSMITH_API_KEY=...
export LANGSMITH_PROJECT=ai-mod-platform
python scripts/run_ci_eval.py --export-langsmith
```

Tracing is enabled only when the key is present. Export is best-effort and must not break the local gate if the network or API fails.

## What a hiring manager or buyer should open

1. `src/graph.py` — three nodes, `interrupt()` on ESCALATE, compiled with a checkpointer.
2. `src/policy.py` — classification vs action. Policy can force ESCALATE when the classifier is sure.
3. `tests/test_eval_gate.py` — 20/20 replay gate, plus mutations that fail on purpose.
4. `.github/workflows/ci.yml` — the gate is the merge check.
5. `data/golden_dataset.json` — dual labels (`policy_verdict`, `expected_system_action`). The model never sees them.

## What this is not

- Not production volume, not a cost model, not a customer count.
- Not a multi-tenant platform, not OpenTelemetry, not an RL loop.
- Live model quality is a 20-row synthetic set on a 3B local SLM. The CI gate measures **routing + HITL**, not that 3B score.

The 2-week consulting offer this artifact supports is: take an agent already mid-build and install evals, fail-safes, and HITL that a merge queue can enforce. This repo is the public shape of that work, not a claim that it is already running in a bank or a hospital.
