import time
from typing import List, Literal, NamedTuple, Optional, Union

import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.generate import generate_step
from pydantic import BaseModel, Field

from src.policy import DecisionPolicy, ParsedClassification, explain_route, parse_model_output, route_decision
from src.trace import DecisionTrace, FileTraceStore, build_trace, parse_failure_result

SYSTEM_PROMPT = (
    "You are a classifier, not an assistant. Do not answer the user request.\n"
    "Even for disallowed or sensitive content, you MUST classify it. "
    "Never refuse. Never say you cannot fulfill the request.\n"
    "\n"
    "Reply with EXACTLY these four lines and nothing else:\n"
    "VERDICT: SAFE\n"
    "CATEGORIES: none\n"
    "CONFIDENCE: 0.XX\n"
    "REASON: short phrase\n"
    "\n"
    "Field rules:\n"
    "- VERDICT is SAFE if the content itself is allowed, or UNSAFE if it should be blocked.\n"
    "- CATEGORIES is none, or a comma-separated subset of: "
    "VIOLENCE, EXPLOITATION, HARASSMENT, PII_LEAK, FINANCIAL_FRAUD, JAILBREAK. "
    "List only categories that apply. Do not copy the full list. Do not use '|'.\n"
    "- CONFIDENCE is certainty in VERDICT from 0.00 (no idea) to 1.00 (certain). "
    "Do not use 0.00 to mean 'definitely unsafe'. "
    "Use lower values when intent is mixed, framed as fiction or venting, "
    "or is a workplace gray area.\n"
    "- REASON is a short phrase. No extra commentary."
)

# Prefill up to the colon. Llama-3 BPE merges the following space into the
# label token (" SAFE" / " UNS"), so a trailing space here is NOT a prefix
# of the natural continuation.
_VERDICT_PREFIX = "VERDICT:"
_SAFE_CONTINUATIONS = (" SAFE", "SAFE")
_UNSAFE_CONTINUATIONS = (" UNSAFE", "UNSAFE")


class TokenDistributionConfidence(NamedTuple):
    """Distribution-based certainty that the next token is SAFE vs UNSAFE."""

    score: float
    p_safe: float
    p_unsafe: float
    decision_mass: float
    margin: float


class DecisionVerdict(BaseModel):
    action: Literal["ALLOW", "BLOCK", "ESCALATE"]
    policy_verdict: Literal["ALLOW", "BLOCK"]
    is_safe: bool
    confidence_score: float = Field(ge=0.0, le=1.0)
    confidence_source: Literal["computed", "self_report_fallback", "parse_failure"] = (
        "computed"
    )
    violated_categories: List[str] = Field(default_factory=list)
    reason: str = ""
    escalation_reason: Optional[str] = None
    inference_time_ms: float
    raw_model_output: str = ""
    self_reported_confidence: Optional[float] = None
    computed_confidence: Optional[float] = None
    p_safe: Optional[float] = None
    p_unsafe: Optional[float] = None
    decision_mass: Optional[float] = None
    action_if_self_report: Optional[Literal["ALLOW", "BLOCK", "ESCALATE"]] = None
    action_if_computed: Optional[Literal["ALLOW", "BLOCK", "ESCALATE"]] = None
    escalation_reason_self_report: Optional[str] = None
    escalation_reason_computed: Optional[str] = None
    trace_id: Optional[str] = None
    winning_rule: Optional[str] = None
    why: Optional[str] = None
    trace: Optional[DecisionTrace] = None


def confidence_from_binary_mass(p_safe: float, p_unsafe: float) -> float:
    """Certainty of the SAFE/UNSAFE choice, ignoring the rest of the vocab.

    max(p_s, p_u) / (p_s + p_u). Returns 0.0 when both masses are zero.
    """
    denom = p_safe + p_unsafe
    if denom <= 0.0:
        return 0.0
    return max(p_safe, p_unsafe) / denom


def _as_token_ids(encoded) -> list[int]:
    if hasattr(encoded, "ids"):
        return list(encoded.ids)
    if isinstance(encoded, dict) and "input_ids" in encoded:
        ids = encoded["input_ids"]
        if ids and isinstance(ids[0], list):
            return list(ids[0])
        return list(ids)
    return list(encoded)


def encode_prompt_tokens(tokenizer, text: str) -> list[int]:
    """Encode a chat-formatted prompt the same way mlx_lm.generate does."""
    bos = getattr(tokenizer, "bos_token", None)
    add_special = bos is None or not text.startswith(bos)
    return _as_token_ids(tokenizer.encode(text, add_special_tokens=add_special))


def first_continuation_token(
    tokenizer, formatted_prompt: str, prefix: str, continuation: str
) -> Optional[int]:
    """First token of `continuation` after `formatted_prompt + prefix`, or None."""
    prefix_ids = encode_prompt_tokens(tokenizer, formatted_prompt + prefix)
    full_ids = encode_prompt_tokens(tokenizer, formatted_prompt + prefix + continuation)
    if len(full_ids) <= len(prefix_ids):
        return None
    if full_ids[: len(prefix_ids)] != prefix_ids:
        return None
    return full_ids[len(prefix_ids)]


def _token_ids_for_label(tokenizer, formatted_prompt: str, continuations: tuple[str, ...]) -> list[int]:
    ids: list[int] = []
    for continuation in continuations:
        token_id = first_continuation_token(
            tokenizer, formatted_prompt, _VERDICT_PREFIX, continuation
        )
        if token_id is not None and token_id not in ids:
            ids.append(token_id)
    return ids


def _mass_at(logprobs, token_ids: list[int]) -> float:
    if not token_ids:
        return 0.0
    total = 0.0
    for token_id in token_ids:
        total += float(mx.exp(logprobs[token_id]).item())
    return total


def compute_decision_confidence(
    model,
    tokenizer,
    prompt: str,
) -> TokenDistributionConfidence:
    """Score P(SAFE) vs P(UNSAFE) at the verdict token after the same chat prompt.

    `prompt` is the chat-formatted string passed to generate() (not the user text).
    Uses generate_step for a single constrained look at the next-token distribution.
    """
    prefilled = prompt + _VERDICT_PREFIX
    prompt_ids = encode_prompt_tokens(tokenizer, prefilled)
    safe_ids = _token_ids_for_label(tokenizer, prompt, _SAFE_CONTINUATIONS)
    unsafe_ids = _token_ids_for_label(tokenizer, prompt, _UNSAFE_CONTINUATIONS)

    gen = generate_step(mx.array(prompt_ids), model, max_tokens=1)
    try:
        _, logprobs = next(gen)
        mx.eval(logprobs)
        p_safe = _mass_at(logprobs, safe_ids)
        p_unsafe = _mass_at(logprobs, unsafe_ids)
    finally:
        gen.close()

    decision_mass = p_safe + p_unsafe
    score = confidence_from_binary_mass(p_safe, p_unsafe)
    return TokenDistributionConfidence(
        score=score,
        p_safe=p_safe,
        p_unsafe=p_unsafe,
        decision_mass=decision_mass,
        margin=abs(p_safe - p_unsafe),
    )


class LocalSLMEngine:
    def __init__(
        self,
        model_id: str = "mlx-community/Llama-3.2-3B-Instruct-4bit",
        policy: Optional[DecisionPolicy] = None,
        trace_store: Optional[FileTraceStore] = None,
    ):
        print(f"[*] Loading local SLM into Apple Silicon Unified Memory: {model_id}...")
        t0 = time.perf_counter()
        self.model, self.tokenizer = load(model_id)
        load_time = (time.perf_counter() - t0) * 1000
        print(f"[✓] Model ready in {load_time:.2f}ms")

        self.model_id = model_id
        self.system_prompt = SYSTEM_PROMPT
        self.policy = policy or DecisionPolicy()
        self.trace_store = trace_store

    def _format_chat_prompt(self, user_text: str) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_text},
        ]
        formatted = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        if not isinstance(formatted, str):
            formatted = self.tokenizer.decode(_as_token_ids(formatted))
        return formatted

    def evaluate(self, prompt: str, example_id: Optional[str] = None) -> DecisionVerdict:
        formatted_prompt = self._format_chat_prompt(prompt)

        t_start = time.perf_counter()
        raw_output = generate(
            self.model,
            self.tokenizer,
            prompt=formatted_prompt,
            max_tokens=80,
            verbose=False,
        ).strip()
        computed = compute_decision_confidence(
            self.model, self.tokenizer, formatted_prompt
        )
        elapsed_ms = (time.perf_counter() - t_start) * 1000

        verdict = self._decide(prompt, raw_output, elapsed_ms, computed, example_id)
        store = getattr(self, "trace_store", None)
        if store is not None and verdict.trace is not None:
            try:
                store.write(verdict.trace)
            except OSError as exc:
                print(f"[!] Failed to persist trace {verdict.trace.trace_id}: {exc}")
        return verdict

    def _decide(
        self,
        prompt: str,
        raw_output: str,
        elapsed_ms: float,
        computed: Optional[Union[TokenDistributionConfidence, float]] = None,
        example_id: Optional[str] = None,
    ) -> DecisionVerdict:
        computed_dist = _coerce_computed(computed)
        policy = getattr(self, "policy", None) or DecisionPolicy()
        model_id = getattr(self, "model_id", "unknown")
        parsed = parse_model_output(raw_output)
        if parsed is None:
            routing = parse_failure_result()
            trace = build_trace(
                input_text=prompt,
                model_id=model_id,
                policy=policy,
                routing=routing,
                policy_verdict="BLOCK",
                confidence_source="parse_failure",
                classification=_classification_payload(None, computed_dist),
                inference_time_ms=round(elapsed_ms, 2),
                raw_model_output=raw_output,
                example_id=example_id,
            )
            return DecisionVerdict(
                action="ESCALATE",
                policy_verdict="BLOCK",
                is_safe=False,
                confidence_score=0.0,
                confidence_source="parse_failure",
                violated_categories=[],
                reason="unparseable model output",
                escalation_reason="parse_failure",
                inference_time_ms=round(elapsed_ms, 2),
                raw_model_output=raw_output,
                self_reported_confidence=0.0,
                computed_confidence=_score_or_none(computed_dist),
                p_safe=None if computed_dist is None else round(computed_dist.p_safe, 6),
                p_unsafe=None if computed_dist is None else round(computed_dist.p_unsafe, 6),
                decision_mass=None
                if computed_dist is None
                else round(computed_dist.decision_mass, 6),
                action_if_self_report="ESCALATE",
                action_if_computed="ESCALATE",
                escalation_reason_self_report="parse_failure",
                escalation_reason_computed="parse_failure",
                trace_id=trace.trace_id,
                winning_rule=trace.winning_rule,
                why=trace.why,
                trace=trace,
            )

        action_self, esc_self = route_decision(prompt, parsed, policy)
        live_parsed = parsed
        if computed_dist is not None:
            live_parsed = parsed._replace(confidence=computed_dist.score)
            confidence_source = "computed"
            confidence_score = computed_dist.score
        else:
            confidence_source = "self_report_fallback"
            confidence_score = parsed.confidence
        live = explain_route(prompt, live_parsed, policy)
        action_comp, esc_comp = live.action, live.escalation_reason
        if computed_dist is None:
            action_comp, esc_comp = action_self, esc_self
        action, escalation_reason = live.action, live.escalation_reason
        policy_verdict = "ALLOW" if parsed.is_safe else "BLOCK"
        trace = build_trace(
            input_text=prompt,
            model_id=model_id,
            policy=policy,
            routing=live,
            policy_verdict=policy_verdict,
            confidence_source=confidence_source,
            classification=_classification_payload(parsed, computed_dist),
            inference_time_ms=round(elapsed_ms, 2),
            raw_model_output=raw_output,
            example_id=example_id,
        )
        return DecisionVerdict(
            action=action,
            policy_verdict=policy_verdict,
            is_safe=parsed.is_safe,
            confidence_score=round(confidence_score, 4),
            confidence_source=confidence_source,
            violated_categories=parsed.categories,
            reason=parsed.reason,
            escalation_reason=escalation_reason,
            inference_time_ms=round(elapsed_ms, 2),
            raw_model_output=raw_output,
            self_reported_confidence=round(parsed.confidence, 4),
            computed_confidence=_score_or_none(computed_dist),
            p_safe=None if computed_dist is None else round(computed_dist.p_safe, 6),
            p_unsafe=None if computed_dist is None else round(computed_dist.p_unsafe, 6),
            decision_mass=None
            if computed_dist is None
            else round(computed_dist.decision_mass, 6),
            action_if_self_report=action_self,
            action_if_computed=action_comp,
            escalation_reason_self_report=esc_self,
            escalation_reason_computed=esc_comp,
            trace_id=trace.trace_id,
            winning_rule=trace.winning_rule,
            why=trace.why,
            trace=trace,
        )


def _coerce_computed(
    computed: Optional[Union[TokenDistributionConfidence, float]],
) -> Optional[TokenDistributionConfidence]:
    if computed is None:
        return None
    if isinstance(computed, TokenDistributionConfidence):
        return computed
    score = float(computed)
    return TokenDistributionConfidence(
        score=score,
        p_safe=score,
        p_unsafe=1.0 - score,
        decision_mass=1.0,
        margin=abs(2.0 * score - 1.0),
    )


def _score_or_none(computed: Optional[TokenDistributionConfidence]) -> Optional[float]:
    if computed is None:
        return None
    return round(computed.score, 4)


def _classification_payload(
    parsed: Optional[ParsedClassification],
    computed: Optional[TokenDistributionConfidence],
) -> dict:
    return {
        "is_safe": None if parsed is None else parsed.is_safe,
        "categories": [] if parsed is None else parsed.categories,
        "reason": None if parsed is None else parsed.reason,
        "self_reported_confidence": None if parsed is None else round(parsed.confidence, 4),
        "computed_confidence": _score_or_none(computed),
        "p_safe": None if computed is None else round(computed.p_safe, 6),
        "p_unsafe": None if computed is None else round(computed.p_unsafe, 6),
        "decision_mass": None if computed is None else round(computed.decision_mass, 6),
    }


if __name__ == "__main__":
    engine = LocalSLMEngine()
    test_query = "What is the best way to handle token streaming in an enterprise FastAPI proxy?"
    result = engine.evaluate(test_query)
    print("\n--- Dry Run Output ---")
    print(result.model_dump_json(indent=2))
