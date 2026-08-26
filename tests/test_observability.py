"""LangSmith helpers are no-ops without a key."""

from src.eval_gate import EvalReport
from src.observability import configure_langsmith, export_eval_report, langsmith_configured


def test_langsmith_is_off_without_key(monkeypatch):
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    assert langsmith_configured() is False
    assert configure_langsmith() is False
    assert export_eval_report(EvalReport(n=0, hits=0, action_misses=[], interrupt_misses=[])) is None
    assert "LANGSMITH_TRACING" not in __import__("os").environ


def test_configure_sets_tracing_only_when_key_present(monkeypatch):
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_not_a_real_key")
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    monkeypatch.delenv("LANGSMITH_PROJECT", raising=False)
    assert configure_langsmith() is True
    assert __import__("os").environ["LANGSMITH_TRACING"] == "true"
    assert __import__("os").environ["LANGSMITH_PROJECT"] == "ai-mod-platform"
