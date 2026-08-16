# Decision Agent System – Architecture Notes (Living Document)

**Last updated:** 2026-08-16  
**Status:** Day 1 complete + Golden dataset v0.2 (dual labels locked)

## Purpose of this system

This is not a product. It is high-signal technical evidence for Forward Deployed / Applied AI Engineering conversations (JPMorgan, Palantir, Epic, NASA-adjacent, etc.).

Every design choice must be defensible in a 30+ minute technical deep-dive and map back to real production concerns: latency, cost, reliability, auditability, human oversight, and constrained environments.

## Current state (Day 1 + Dataset)

- Local SLM engine (`src/slm_engine.py`) using MLX + Llama-3.2-3B-Instruct-4bit
- Basic binary classification (ALLOW / BLOCK) with category extraction
- Latency: P95 ~118ms on M4 Max (validated in `tests/test_day1.py`)
- Confidence currently hardcoded (0.95 / 0.98) — decorative only
- Golden dataset: `data/golden_dataset.json` (20 examples, v0.2)
  - `policy_verdict`: binary ground truth of the content (ALLOW | BLOCK)
  - `expected_system_action`: what the agent should output (ALLOW | BLOCK | ESCALATE)
  - Model never sees either label during evaluation

## Design decisions & trade-offs

### Why local-first (MLX on Apple Silicon)

- Deterministic latency and cost characteristics for demos and evaluation
- No network dependency or third-party model drift during interviews
- Forces thinking about resource constraints that also exist in regulated on-prem / air-gapped environments
- Trade-off: smaller model capacity vs cloud frontier models. Acceptable for v1 classification task; larger models or hybrid routing can be added later if evaluation shows need.

### Action space (ALLOW / BLOCK / ESCALATE)

Binary decisions are insufficient for production. Real systems need a third path for:

- Low model confidence
- High-stakes domains (financial PII, clinical advice, legal interpretation)
- Ambiguous intent (threat language + disclaimer, fiction framing of crimes, etc.)

The golden dataset deliberately over-weights borderline cases (9/20) because this is where the interesting engineering and interview discussion lives.

### Confidence must drive routing

A static `confidence_score` field is weak. Next priority after the golden set is to make confidence estimation real enough that it actually changes the action (especially forcing ESCALATE).

Possible approaches under consideration (to be evaluated, not yet implemented):

- Log-prob / token probability from the SLM
- Self-consistency / multiple samples
- Explicit uncertainty phrasing in the model output + parsing
- Simple heuristic based on category severity + output length / hedging language

Whatever is chosen must be measurable on the golden set.

### Policy separation

Policy (what is allowed, what is high-stakes, thresholds) must be separable from the model engine. This enables:

- Different policies for different tenants or environments later
- Clear audit story ("the model said X, policy Y mapped it to ESCALATE")
- Interview discussion of how you would adapt this for a bank vs a healthcare system vs a government deployment

### Evaluation harness (next after confidence)

Must report at minimum:

- Agreement with expected_action
- Confusion matrix / FP / FN (especially false ALLOW on high-severity)
- Latency distribution
- Escalation rate on borderline vs clear cases

This becomes both the development tool and the interview artifact.

## What would change in a regulated environment

| Concern              | Current v1                          | Bank / Healthcare / Gov adaptation                  |
|----------------------|-------------------------------------|-----------------------------------------------------|
| Model hosting        | Local MLX                           | Often on-prem or VPC, approved model list           |
| Audit trail          | Decision object only                | Immutable log + policy version + model version      |
| Human oversight      | Planned escalation path             | Mandatory for certain categories + SLAs             |
| Data retention       | None yet                            | Strict, often zero-retention or short TTL           |
| Evaluation           | Golden set                          | Continuous monitoring + red-team + bias audits      |
| Latency              | <200ms P95 target                   | Same or tighter, plus fallback paths                |

## Known limitations (be honest in interviews)

- Confidence is not yet real
- No formal policy object yet
- No decision tracing beyond the verdict fields
- Single small model; no ensemble or cascade
- Golden set is still small (20) and synthetic
- No production traffic or online evaluation loop

## Next thin vertical slices (in order)

1. **Golden dataset** ← done (v0.2)
2. Real confidence estimation that can drive ESCALATE
3. Escalation path + structured decision state that includes reason + policy reference
4. Minimal evaluation harness against the golden set
5. Decision tracing (why this action)
6. Separable policy layer

Resist: multi-tenant, full OpenTelemetry, BYOK, vector caches, RL, packaging polish — until the core decision loop + eval are strong.
