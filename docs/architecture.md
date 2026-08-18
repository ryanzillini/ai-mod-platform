# Decision Agent System – Architecture Notes (Living Document)

**Last updated:** 2026-08-18  
**Status:** Decision tracing in place (structured why + JSONL persist)

## Purpose of this system

This is not a product. It is high-signal technical evidence for Forward Deployed / Applied AI Engineering conversations (JPMorgan, Palantir, Epic, NASA-adjacent, etc.).

Every design choice must be defensible in a 30+ minute technical deep-dive and map back to real production concerns: latency, cost, reliability, auditability, human oversight, and constrained environments.

## Current state

- Local SLM engine (`src/slm_engine.py`) using MLX + Llama-3.2-3B-Instruct-4bit
- Structured generation still emits VERDICT / CATEGORIES / CONFIDENCE / REASON. Self-reported `CONFIDENCE` is audit-only.
- Token-distribution confidence is locked as the routing source: one constrained `generate_step` after a BPE-stable `VERDICT:` prefix, scoring P(SAFE) vs P(UNSAFE)
- Three-way action space: ALLOW / BLOCK / ESCALATE
- Separable policy layer (`src/policy.py`) that can force escalation even at high confidence
- Decision traces (`src/trace.py`): ordered short-circuiting rule log on every verdict, persisted as JSONL when a `FileTraceStore` is attached
- Golden dataset: `data/golden_dataset.json` (20 examples, v0.2)
  - `policy_verdict`: binary ground truth of the content (ALLOW | BLOCK)
  - `expected_system_action`: what the agent should output (ALLOW | BLOCK | ESCALATE)
  - Model never sees either label during evaluation
- Eval runner: `scripts/run_golden_baseline.py` scores policy-verdict and system-action agreement on the locked (computed) path, and still prints self-report as an audit comparison
- Latest locked-path eval (`results/golden_eval_20260818T042323Z.json`) with traces (`results/golden_traces_20260818T042323Z.jsonl`): policy 17/20 (85%), action 16/20 (80%). Every row `confidence_source=computed`. P50 ~464ms / P95 ~482ms.

## Design decisions & trade-offs

### Why local-first (MLX on Apple Silicon)

- Deterministic latency and cost characteristics for demos and evaluation
- No network dependency or third-party model drift during interviews
- Forces thinking about resource constraints that also exist in regulated on-prem / air-gapped environments
- Trade-off: smaller model capacity vs cloud frontier models. Acceptable for v1 classification task; larger models or hybrid routing can be added later if evaluation shows need.

### Action space (ALLOW / BLOCK / ESCALATE)

Binary decisions are insufficient for production. Real systems need a third path for:

- Low model confidence
- High-stakes domains (clinical advice, legal interpretation, sensitive HR)
- Highest-severity categories (EXPLOITATION) even when the classifier is sure

Dropping low-confidence requests (or silently defaulting them to ALLOW/BLOCK) hides uncertainty. Escalation makes uncertainty an explicit, auditable action.

### How confidence is obtained (locked)

Two scores are still computed on every `evaluate()` call. **Only the token-distribution score drives `action`.**

1. **Computed (source of truth)** — `compute_decision_confidence()` prefills the same chat prompt with `VERDICT:` (no trailing space) and reads the next-token distribution via `mlx_lm.generate.generate_step`. Llama-3 BPE merges the space into the label (` SAFE` vs ` UNS`+`AFE`), so scoring after a trailing space is not a valid prefix. Formula:

   `computed_confidence = max(p_SAFE, p_UNSAFE) / (p_SAFE + p_UNSAFE)`

   That is certainty of the binary choice, not mass over the full vocab. `decision_mass = p_SAFE + p_UNSAFE` is logged so we can see when the model is about to refuse instead of classify. `confidence_score` on `DecisionVerdict` is this value. `confidence_source` is `"computed"`.

2. **Self-reported (audit only)** — parse `CONFIDENCE: 0.XX` from the structured generation. Kept on the verdict as `self_reported_confidence` and as a shadow `action_if_self_report`. The 3B model still emits `0.00` to mean "definitely unsafe," which is why it is not allowed to change the action.

If the logprob pass is unavailable (unit tests of `_decide` without a model), routing falls back to the parsed self-report and `confidence_source="self_report_fallback"`. Production `evaluate()` always measures the distribution.

Parse failure is treated as confidence 0.0 → `ESCALATE` with `escalation_reason="parse_failure"` and `confidence_source="parse_failure"` (fail closed on the action, not auto-ALLOW). A computed score is still recorded, but it cannot invent ALLOW/BLOCK without a parsed verdict.

Llama-3.2-3B often refuses to emit the schema on high-severity prompts (`I can't fulfill that request`). That is treated as an implicit UNSAFE classification; computed confidence is still measured at the `VERDICT:` position and used for routing.

Trade-off: the logprob pass is a second prefill of the same prompt (~one extra prompt-processing cost per request). A later slice can fold it into a single `stream_generate` pass.

### Comparison outcome (2026-08-16)

Same 20 examples, same policy, threshold 0.80. Formula: `max(p_SAFE, p_UNSAFE) / (p_SAFE + p_UNSAFE)`.

| | Self-report | Computed |
|---|---|---|
| Action accuracy | 17/20 (85%) | 16/20 (80%) |
| Clear-safe escalate | 0/5 | 0/5 |
| Clear-unsafe escalate | 3/6 | 2/6 |
| Borderline escalate | 9/9 | 7/9 |
| Mean conf clear-unsafe | 0.56 (dishonest `0.00`) | 1.00 |
| Mean conf borderline | 0.50 | 0.86 (range 0.58–1.00) |

Self-report's extra action hit is the `CONFIDENCE: 0.00` bug accidentally forcing ESCALATE on gd-013 / gd-016. Computed is high on every clear case, and actually lower on the gray examples where the two classes compete (gd-012 0.58, gd-014 0.76, gd-015 0.69, gd-017 0.78). Remaining computed misses are classification (gd-013 SAFE on PII) or policy (gd-007/010 `EXPLOITATION` echo; gd-016 certain-UNSAFE where gold wants ESCALATE for intent). **Computed is locked as the routing source.** Self-report remains on the verdict for audit.

### Policy can override a confident model

Routing order after a successful parse:

1. High-severity category (currently `EXPLOITATION`) → ESCALATE
2. Confidence below threshold (default 0.80) → ESCALATE
3. Domain heuristics: medical decision support, legal/regulatory interpretation, high-stakes HR → ESCALATE
4. Else ALLOW or BLOCK from the model's SAFE/UNSAFE verdict

So a clinical dosage question can be classified `UNSAFE` at 0.96 confidence and still become `ESCALATE` because policy forbids auto-answering. That split — **classification vs action** — is the point of the two labels on the golden set.

Domain heuristics are keyword-based on purpose. They are explicit, testable, and easy to swap per tenant later. They are not a second classifier.

### Policy separation

Policy (thresholds, always-escalate categories, domain rules) lives in `DecisionPolicy`, not in the prompt. This enables:

- Different policies for a bank vs a healthcare system vs a government deployment
- A clear audit story: "the model said SAFE at 0.94; policy Y mapped it to ESCALATE because this is legal interpretation"
- Unit tests that do not load the model

### Decision tracing

Every `evaluate()` builds a `DecisionTrace` from the actual routing control flow, not a reconstructed story.

- Rules are recorded in evaluation order. After a rule fires, later rules are **not** marked skipped — they were never evaluated. That matches the code.
- `winning_rule` + `why` are the interview one-liner. `steps` is the evidence.
- Persistence is optional `FileTraceStore` (append-only JSONL). The decision does not fail if a trace write fails.
- v1 stores full `input_text`. In a regulated environment that would be hashed or redacted; the schema already has a place for a policy snapshot and model id.

This is deliberately not OpenTelemetry. One JSON object per decision is enough to answer "why this action" in a deep-dive, and it stays local-first.

### Evaluation

The golden runner reports:

- `policy_match`: predicted `policy_verdict` vs gold
- `action_match`: live `action` (computed confidence) vs `expected_system_action`
- Shadow `action_match_self_report` for audit
- Escalation rate on `clear_safe` / `clear_unsafe` / `borderline`
- Preferential cases gd-015 / 017 / 018 / 019
- Decision traces for preferential cases and mismatches (`winning_rule`, ordered steps)
- Latency P50 / P95 (generation + logprob prefill)

False ALLOW on high-severity remains the most important failure mode to watch.

## What would change in a regulated environment

| Concern              | Current v1                          | Bank / Healthcare / Gov adaptation                  |
|----------------------|-------------------------------------|-----------------------------------------------------|
| Model hosting        | Local MLX                           | Often on-prem or VPC, approved model list           |
| Audit trail          | JSONL decision traces + raw output  | Immutable log + policy version + model version      |
| Human oversight      | ESCALATE action + reason            | Mandatory for certain categories + SLAs             |
| Data retention       | None yet                            | Strict, often zero-retention or short TTL           |
| Evaluation           | Golden set                          | Continuous monitoring + red-team + bias audits      |
| Latency              | Local P95 target                    | Same or tighter, plus fallback paths                |

## Known limitations (be honest in interviews)

- Self-reported confidence is miscalibrated (the 3B model still emits `0.00` to mean "definitely unsafe"). It is audit-only and does not drive routing.
- Computed confidence is a first-token SAFE vs UNSAFE score and adds a second prompt prefill
- Category labels are noisy (`EXPLOITATION` on fraud/explosives), so always-escalate-on-category can over-escalate clear BLOCKs
- Domain escalation rules are keyword heuristics, not classifiers
- Traces store full input text locally (not redacted); JSONL is not an immutable production log
- Single small model; no ensemble or cascade
- Golden set is still small (20) and synthetic
- No production traffic or online evaluation loop

## Next thin vertical slices (in order)

1. Golden dataset ← done (v0.2)
2. Real confidence estimation that can drive ESCALATE ← done (token-distribution locked)
3. Escalation path + structured decision state ← done (`DecisionVerdict`)
4. Minimal evaluation harness against the golden set ← done (dual accuracy)
5. Lock the winning confidence source from the golden-set head-to-head ← done
6. Decision tracing (why this action, persist traces) ← done

Resist: multi-tenant, full OpenTelemetry, BYOK, vector caches, RL, packaging polish. Remaining high-signal gaps: EXPLOITATION category over-escalate (gd-007/010), PII operational false ALLOW (gd-013), certain-UNSAFE vs intent-ESCALATE (gd-016).
