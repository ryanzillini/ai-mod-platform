import time
from typing import List
from pydantic import BaseModel, Field
from mlx_lm import load, generate

class ModerationVerdict(BaseModel):
    verdict: str = Field(description="ALLOW or BLOCK")
    is_safe: bool
    confidence_score: float = Field(ge=0.0, le=1.0)
    violated_categories: List[str] = Field(default_factory=list)
    inference_time_ms: float

class LocalSLMEngine:
    def __init__(self, model_id: str = "mlx-community/Llama-3.2-3B-Instruct-4bit"):
        print(f"[*] Loading local SLM into Apple Silicon Unified Memory: {model_id}...")
        t0 = time.perf_counter()
        self.model, self.tokenizer = load(model_id)
        load_time = (time.perf_counter() - t0) * 1000
        print(f"[✓] Model ready in {load_time:.2f}ms")

        # Enterprise Safety Guard Taxonomy
        self.system_prompt = (
            "You are an enterprise AI safety classifier. Analyze the user message.\n"
            "If unsafe, return 'UNSAFE' followed by any violated categories from: "
            "[VIOLENCE, EXPLOITATION, HARASSMENT, PII_LEAK, FINANCIAL_FRAUD, JAILBREAK].\n"
            "If safe, return 'SAFE'.\n"
            "Reply ONLY with your verdict on the first line."
        )

    def evaluate(self, prompt: str) -> ModerationVerdict:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True
        )

        t_start = time.perf_counter()
        raw_output = generate(
            self.model, 
            self.tokenizer, 
            prompt=formatted_prompt, 
            max_tokens=25,
            verbose=False
        ).strip()
        elapsed_ms = (time.perf_counter() - t_start) * 1000

        lines = raw_output.split("\n")
        first_token = lines[0].strip().upper()
        
        is_safe = "SAFE" in first_token and "UNSAFE" not in first_token
        verdict = "ALLOW" if is_safe else "BLOCK"
        
        categories = []
        if not is_safe and len(lines) > 1:
            categories = [c.strip() for c in lines[1].split(",") if c.strip()]

        confidence = 0.95 if is_safe else 0.98

        return ModerationVerdict(
            verdict=verdict,
            is_safe=is_safe,
            confidence_score=confidence,
            violated_categories=categories,
            inference_time_ms=round(elapsed_ms, 2)
        )

if __name__ == "__main__":
    engine = LocalSLMEngine()
    test_query = "What is the best way to handle token streaming in an enterprise FastAPI proxy?"
    result = engine.evaluate(test_query)
    print("\n--- Dry Run Output ---")
    print(result.model_dump_json(indent=2))
