# FDE Decision Agent System – Agent / Cursor Rules

## Primary goal
High-signal decision agent for Forward Deployed / Applied AI Engineering interview evidence (JPMorgan, Palantir, Epic, NASA-adjacent, etc.).
Depth and defensibility over feature breadth.

This is supporting evidence alongside real production experience (Khoros multi-LLM moderation platform). It is not a product.

## Hard constraints (do not violate)
- Local-first (MLX / Ollama on Apple Silicon)
- Text-only for v1
- No multi-tenant, BYOK, full OpenTelemetry, vector caches, or RL loops in v1
- Prefer working code + evaluation results over diagrams
- Every major component must be explainable in an interview

## Current priority order
1. Golden dataset (done – dual labels: policy_verdict + expected_system_action)
2. Real confidence estimation that drives routing (done – token-distribution locked; self-report is audit-only)
3. Explicit ESCALATE path (low confidence OR policy-forced: EXPLOITATION, medical/legal/HR, workplace complaint, operational PII)
4. Decision tracing (done – structured why + JSONL)
5. Minimal evaluation harness against golden set
6. Separable policy layer

## Implementation rules
- Move in thin vertical slices only
- Confidence must actually change the action (not decorative)
- Model must never see golden-set labels at inference time
- Prefer a clear DecisionVerdict + policy object over scattered flags
- Document trade-offs in docs/architecture.md as you go
- When confidence is low or policy requires it, emit ESCALATE instead of forcing ALLOW/BLOCK

## Explicit non-goals for v1
Multi-tenant isolation, full observability stack, continuous learning / RL, packaging polish, cloud deployment scaffolding, vector caches.

## Evaluation hygiene
- Golden set lives in data/golden_dataset.json
- policy_verdict = ground truth of the content (ALLOW | BLOCK)
- expected_system_action = what the agent should output (ALLOW | BLOCK | ESCALATE)
- Never pass either label to the model during inference
