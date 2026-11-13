"""Thread-safe, opt-in API metadata logging. Never stores prompts or keys."""

import json
import os
import threading
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from functools import partial
from datetime import datetime, timezone
from pathlib import Path

_lock = threading.Lock()
_event_context = ContextVar("simplemem_api_event_context", default={})


def current_api_event_context():
    return dict(_event_context.get())


@contextmanager
def api_event_context(**fields):
    """Attach trial/question/condition IDs to this task's API events."""
    token = _event_context.set({**_event_context.get(), **fields})
    try:
        yield
    finally:
        _event_context.reset(token)


def submit_with_context(executor, function, *args, **kwargs):
    """Use a separate context snapshot for every executor submission."""
    return executor.submit(copy_context().run, partial(function, *args, **kwargs))


def record_api_event(**event):
    path = os.getenv("SIMPLEMEM_API_LOG")
    if not path:
        return
    row = {"timestamp": datetime.now(timezone.utc).isoformat(), **_event_context.get(), **event}
    with _lock:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def error_metadata(error):
    return {"error_type": type(error).__name__,
            "status_code": getattr(error, "status_code", None)}


def record_pipeline_fallback(stage, error):
    record_api_event(kind="pipeline_fallback", fallback_stage=stage,
                     status="degraded", **error_metadata(error))
