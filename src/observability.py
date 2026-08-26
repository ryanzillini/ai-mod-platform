"""Optional LangSmith tracing / eval export.

Enabled only when LANGSMITH_API_KEY is set. Missing key, missing package, or a
failed export must not break local runs or CI.
"""

from __future__ import annotations

import os
from typing import Any, Optional

DEFAULT_PROJECT = "ai-mod-platform"


def langsmith_configured() -> bool:
    return bool(os.environ.get("LANGSMITH_API_KEY"))


def configure_langsmith() -> bool:
    """Turn on LangSmith tracing when an API key is present.

    Does not set LANGSMITH_TRACING when the key is absent, so CI stays offline.
    """
    if not langsmith_configured():
        return False
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_PROJECT", DEFAULT_PROJECT)
    return True


def export_eval_report(report: Any) -> Optional[str]:
    """Best-effort eval export. Returns a project URL or None.

    Never raises. A missing key, a network error, or an unexpected Client
    API is a skip — the local CI gate is the source of truth.
    """
    if not langsmith_configured():
        return None
    configure_langsmith()
    payload = report.to_dict() if hasattr(report, "to_dict") else dict(report)
    try:
        from langsmith import Client
    except ImportError:
        print("[!] langsmith package not installed; export skipped")
        return None
    try:
        client = Client()
        project = os.environ.get("LANGSMITH_PROJECT", DEFAULT_PROJECT)
        client.create_run(
            name="golden_ci_eval",
            run_type="chain",
            inputs={
                "dataset": "data/golden_dataset.json",
                "n": payload.get("n"),
            },
            outputs=payload,
            project_name=project,
        )
        return f"https://smith.langchain.com/o/default/projects?p={project}"
    except Exception as exc:  # noqa: BLE001 — export is optional
        print(f"[!] LangSmith export skipped: {exc}")
        return None
