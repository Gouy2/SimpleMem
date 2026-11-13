"""Atomic checkpoints and source/config identity for recoverable evaluation."""

import hashlib
import json
import os
import tempfile
from pathlib import Path

from experiments.bootstrap import REPO_ROOT


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False, encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
        temporary = stream.name
    os.replace(temporary, path)


def read_json(path):
    return json.loads(Path(path).read_text())


def content_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def source_identity():
    files = [REPO_ROOT / name for name in ("main.py", "requirements-api.in", "requirements-api.txt")]
    files += sorted((REPO_ROOT / "simplemem/core").rglob("*.py"))
    files += sorted((REPO_ROOT / "experiments").glob("*.py"))
    hashes = {str(path.relative_to(REPO_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in files}
    return {"fingerprint": content_hash(hashes), "files": hashes}


def ensure_manifest(path, expected):
    if Path(path).exists():
        saved = read_json(path)
        for key in ("design", "source", "environment", "effective_settings"):
            if saved[key] != expected[key]:
                raise RuntimeError(f"Resume refused: {key} changed")
        return saved
    atomic_json(path, expected)
    return expected


def job_id(trial, sample_index, qa_index, condition):
    return f"t{trial}-s{sample_index}-q{qa_index}-{condition}"


def expected_jobs(design):
    return {job_id(trial, q["sample_index"], q["qa_index"], condition)
            for trial in range(design["trials"]) for q in design["questions"]
            for condition in design["conditions"]}


def read_events(path):
    if not Path(path).exists():
        return []
    # Called only after a task has joined, with no further writes for that job.
    # Ignore an incomplete last line while other tasks may still be appending.
    lines = Path(path).read_text().splitlines(keepends=True)
    return [json.loads(line) for line in lines if line.endswith("\n")]
