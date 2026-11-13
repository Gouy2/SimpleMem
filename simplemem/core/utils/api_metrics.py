"""Thread-safe, opt-in API metadata logging. Never stores prompts or keys."""

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

_lock = threading.Lock()


def record_api_event(**event):
    path = os.getenv("SIMPLEMEM_API_LOG")
    if not path:
        return
    row = {"timestamp": datetime.now(timezone.utc).isoformat(), **event}
    with _lock:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def error_metadata(error):
    return {"error_type": type(error).__name__,
            "status_code": getattr(error, "status_code", None)}
