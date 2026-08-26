"""Classifiers for the decision graph.

The live path is LocalSLMEngine on Apple Silicon (MLX). CI and Linux cannot
load that model, so they replay an explicit classification per golden id.
Golden labels are never passed to a model.
"""

from __future__ import annotations

from typing import Optional, Protocol

from src.slm_engine import TokenDistributionConfidence


class ClassifyOutput:
    """Raw model text plus the optional computed-confidence payload."""

    __slots__ = ("raw_model_output", "computed", "inference_time_ms")

    def __init__(
        self,
        raw_model_output: str,
        computed: Optional[TokenDistributionConfidence] = None,
        inference_time_ms: float = 0.0,
    ):
        self.raw_model_output = raw_model_output
        self.computed = computed
        self.inference_time_ms = inference_time_ms


class Classifier(Protocol):
    def classify(
        self, text: str, example_id: Optional[str] = None
    ) -> ClassifyOutput: ...


def format_structured_output(
    *,
    is_safe: bool,
    categories: list[str],
    confidence: float,
    reason: str,
) -> str:
    cats = ",".join(categories) if categories else "none"
    verdict = "SAFE" if is_safe else "UNSAFE"
    return (
        f"VERDICT: {verdict}\n"
        f"CATEGORIES: {cats}\n"
        f"CONFIDENCE: {confidence:.2f}\n"
        f"REASON: {reason}\n"
    )


def computed_from_score(score: float, is_safe: bool) -> TokenDistributionConfidence:
    """Stand-in distribution so replay routing uses the locked computed path."""
    score = min(1.0, max(0.0, float(score)))
    p_safe = score if is_safe else 1.0 - score
    p_unsafe = 1.0 - p_safe
    return TokenDistributionConfidence(
        score=score,
        p_safe=p_safe,
        p_unsafe=p_unsafe,
        decision_mass=1.0,
        margin=abs(p_safe - p_unsafe),
    )


class ReplayClassifier:
    """Look up a recorded classification by example_id, then exact input text."""

    def __init__(
        self,
        by_id: dict[str, dict],
        by_text: Optional[dict[str, dict]] = None,
    ):
        self.by_id = by_id
        self.by_text = by_text or {}

    def classify(
        self, text: str, example_id: Optional[str] = None
    ) -> ClassifyOutput:
        record = None
        if example_id and example_id in self.by_id:
            record = self.by_id[example_id]
        elif text in self.by_text:
            record = self.by_text[text]
        if record is None:
            raise KeyError(
                "No replay classification for "
                f"{example_id or text[:48]!r}. CI uses data/ci_replay_classifications.json."
            )
        if record.get("parse_failed"):
            return ClassifyOutput(record.get("raw_model_output") or "not a structured response")
        is_safe = bool(record["is_safe"])
        categories = list(record.get("categories") or [])
        confidence = float(record["confidence"])
        reason = str(record.get("reason") or "replay")
        raw = format_structured_output(
            is_safe=is_safe,
            categories=categories,
            confidence=confidence,
            reason=reason,
        )
        return ClassifyOutput(raw, computed_from_score(confidence, is_safe), 0.0)


class MlxClassifier:
    """Thin classify-only adapter over LocalSLMEngine. Apple Silicon only."""

    def __init__(self, engine):
        self.engine = engine

    def classify(
        self, text: str, example_id: Optional[str] = None
    ) -> ClassifyOutput:
        import time

        from src.slm_engine import _mlx, compute_decision_confidence

        formatted = self.engine._format_chat_prompt(text)
        t0 = time.perf_counter()
        _mx, _load, generate, _step = _mlx()
        raw = generate(
            self.engine.model,
            self.engine.tokenizer,
            prompt=formatted,
            max_tokens=80,
            verbose=False,
        ).strip()
        computed = compute_decision_confidence(
            self.engine.model, self.engine.tokenizer, formatted
        )
        elapsed = (time.perf_counter() - t0) * 1000
        return ClassifyOutput(raw, computed, elapsed)
