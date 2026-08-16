import time
from typing import List, Literal, Optional

from pydantic import BaseModel, Field
from mlx_lm import load, generate

from src.policy import DecisionPolicy, parse_model_output, route_decision

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


class DecisionVerdict(BaseModel):
    action: Literal["ALLOW", "BLOCK", "ESCALATE"]
    policy_verdict: Literal["ALLOW", "BLOCK"]
    is_safe: bool
    confidence_score: float = Field(ge=0.0, le=1.0)
    violated_categories: List[str] = Field(default_factory=list)
    reason: str = ""
    escalation_reason: Optional[str] = None
    inference_time_ms: float
    raw_model_output: str = ""


class LocalSLMEngine:
    def __init__(
        self,
        model_id: str = "mlx-community/Llama-3.2-3B-Instruct-4bit",
        policy: Optional[DecisionPolicy] = None,
    ):
        print(f"[*] Loading local SLM into Apple Silicon Unified Memory: {model_id}...")
        t0 = time.perf_counter()
        self.model, self.tokenizer = load(model_id)
        load_time = (time.perf_counter() - t0) * 1000
        print(f"[✓] Model ready in {load_time:.2f}ms")

        self.system_prompt = SYSTEM_PROMPT
        self.policy = policy or DecisionPolicy()

    def evaluate(self, prompt: str) -> DecisionVerdict:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]

        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
        )

        t_start = time.perf_counter()
        raw_output = generate(
            self.model,
            self.tokenizer,
            prompt=formatted_prompt,
            max_tokens=80,
            verbose=False,
        ).strip()
        elapsed_ms = (time.perf_counter() - t_start) * 1000

        return self._decide(prompt, raw_output, elapsed_ms)

    def _decide(self, prompt: str, raw_output: str, elapsed_ms: float) -> DecisionVerdict:
        parsed = parse_model_output(raw_output)
        if parsed is None:
            return DecisionVerdict(
                action="ESCALATE",
                policy_verdict="BLOCK",
                is_safe=False,
                confidence_score=0.0,
                violated_categories=[],
                reason="unparseable model output",
                escalation_reason="parse_failure",
                inference_time_ms=round(elapsed_ms, 2),
                raw_model_output=raw_output,
            )

        action, escalation_reason = route_decision(prompt, parsed, self.policy)
        policy_verdict = "ALLOW" if parsed.is_safe else "BLOCK"
        return DecisionVerdict(
            action=action,
            policy_verdict=policy_verdict,
            is_safe=parsed.is_safe,
            confidence_score=round(parsed.confidence, 4),
            violated_categories=parsed.categories,
            reason=parsed.reason,
            escalation_reason=escalation_reason,
            inference_time_ms=round(elapsed_ms, 2),
            raw_model_output=raw_output,
        )


if __name__ == "__main__":
    engine = LocalSLMEngine()
    test_query = "What is the best way to handle token streaming in an enterprise FastAPI proxy?"
    result = engine.evaluate(test_query)
    print("\n--- Dry Run Output ---")
    print(result.model_dump_json(indent=2))
