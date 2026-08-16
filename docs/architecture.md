# Decision Agent System – Architecture Notes (Living Document)

**Last updated:** 2026-08-16  
**Status:** Confidence estimation + escalation routing in place (v1)

## Purpose of this system

This is not a product. It is high-signal technical evidence for Forward Deployed / Applied AI Engineering conversations (JPMorgan, Palantir, Epic, NASA-adjacent, etc.).

Every design choice must be defensible in a 30+ minute technical deep-dive and map back to real production concerns: latency, cost, reliability, auditability, human oversight, and constrained environments.

## Current state

- Local SLM engine (`src/slm_engine.py`) using MLX + Llama-3.2-3B-Instruct-4bit
- Structured self-reported confidence (VERDICT / CATEGORIES / CONFIDENCE / REASON)
- Three-way action space: ALLOW / BLOCK / ESCALATE
- Separable policy layer (`src/policy.py`) that can force escalation even at high confidence
- Golden dataset: `data/golden_dataset.json` (20 examples, v0.2)
  - `policy_verdict`: binary ground truth of the content (ALLOW | BLOCK)
  - `expected_system_action`: what the agent should output (ALLOW | BLOCK | ESCALATE)
  - Model never sees either label during evaluation
- Eval runner: `scripts/run_golden_baseline.py` scores both policy-verdict agreement and system-action agreement, plus escalation rate on borderline vs clear cases
- Latest golden eval (`results/golden_eval_20260816T064446Z.json`): policy 17/20 (85%), action 17/20 (85%). Borderline escalation 9/9. Clear-safe escalation 0/5. P50 ~324ms / P95 ~342ms.

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

### How confidence is obtained (v1)

The model must emit a parseable `CONFIDENCE: 0.XX` field. That float is clamped to [0, 1] and becomes `confidence_score`.

This is **not** logprobs. Self-reported confidence is miscalibrated, but it is:

- measurable on the golden set
- cheap (no extra samples, no `mlx_lm.generate` logprob plumbing)
- enough for confidence to actually change the action

Parse failure is treated as confidence 0.0 → `ESCALATE` with `escalation_reason="parse_failure"` (fail closed on the action, not auto-ALLOW).

Llama-3.2-3B often refuses to emit the schema on high-severity prompts (`I can't fulfill that request`). That is treated as an implicit UNSAFE classification at confidence 0.90 → typically BLOCK — not as a parse failure. The model already decided the content is disallowed; we recover a structured decision from the refusal.

Later refinements (not in this slice): token logprobs, or averaging multiple samples.

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

### Evaluation

The golden runner reports:

- `policy_match`: predicted `policy_verdict` vs gold
- `action_match`: predicted `action` vs `expected_system_action`
- Escalation rate on `clear_safe` / `clear_unsafe` / `borderline`
- Latency P50 / P95

False ALLOW on high-severity remains the most important failure mode to watch.

## What would change in a regulated environment

| Concern              | Current v1                          | Bank / Healthcare / Gov adaptation                  |
|----------------------|-------------------------------------|-----------------------------------------------------|
| Model hosting        | Local MLX                           | Often on-prem or VPC, approved model list           |
| Audit trail          | Decision object + raw model output  | Immutable log + policy version + model version      |
| Human oversight      | ESCALATE action + reason            | Mandatory for certain categories + SLAs             |
| Data retention       | None yet                            | Strict, often zero-retention or short TTL           |
| Evaluation           | Golden set                          | Continuous monitoring + red-team + bias audits      |
| Latency              | Local P95 target                    | Same or tighter, plus fallback paths                |

## Known limitations (be honest in interviews)

- Self-reported confidence is miscalibrated (the 3B model still emits `0.00` to mean "definitely unsafe," which then forces ESCALATE)
- Category labels are noisy (`EXPLOITATION` on fraud/explosives), so always-escalate-on-category can over-escalate clear BLOCKs
- Domain escalation rules are keyword heuristics, not classifiers
- No decision tracing store beyond the verdict object
- Single small model; no ensemble or cascade
- Golden set is still small (20) and synthetic
- No production traffic or online evaluation loop

## Next thin vertical slices (in order)

1. Golden dataset ← done (v0.2)
2. Real confidence estimation that can drive ESCALATE ← done (v1 structured self-report)
3. Escalation path + structured decision state ← done (`DecisionVerdict`)
4. Minimal evaluation harness against the golden set ← done (dual accuracy)
5. Decision tracing (why this action, persist traces)
6. Stronger confidence signal (logprobs and/or self-consistency) if golden-set calibration is weak

Resist: multi-tenant, full OpenTelemetry, BYOK, vector caches, RL, packaging polish — until tracing + a stronger confidence signal are evaluated.
