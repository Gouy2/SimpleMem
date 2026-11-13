"""Offline environment checks; never calls a model or prints credentials."""

import argparse
import importlib.metadata
import os
import platform
import sys
import tempfile

from experiments.bootstrap import load_environment


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file")
    parser.add_argument("--require-keys", action="store_true")
    args = parser.parse_args()
    path = load_environment(args.env_file)
    print(f"Python: {platform.python_version()} ({platform.machine()})")
    if sys.version_info < (3, 10):
        raise RuntimeError("Python >= 3.10 is required")
    print(f"Environment file: {path} ({'exists' if path.is_file() else 'absent'})")

    missing = []
    for name in ("DEEPSEEK_API_KEY", "OPENROUTER_API_KEY"):
        present = bool(os.getenv(name, "").strip())
        print(f"{name}: {'set' if present else 'missing'}")
        if not present:
            missing.append(name)

    for package in (
        "openai", "numpy", "pydantic", "lancedb", "pyarrow", "pylance",
        "dateparser", "python-dotenv", "tantivy", "tqdm",
    ):
        print(f"{package}=={importlib.metadata.version(package)}")

    # Import the actual author pipeline without initializing local models.
    from main import SimpleMemSystem
    from simplemem.core.settings import settings
    from simplemem.core.database.vector_store_backend import (
        LanceDBVectorStoreBackend,
        VectorStoreRecord,
    )

    if settings._user_config is not None:
        raise RuntimeError(
            "A top-level config.py is overriding environment settings. "
            "Remove it from this experiment's import path before proceeding."
        )
    assert settings.OPENAI_BASE_URL == os.environ["OPENAI_BASE_URL"]
    assert settings.LLM_MODEL == os.environ["LLM_MODEL"]
    print(f"Text pipeline import: OK ({SimpleMemSystem.__name__})")

    # Small synthetic vectors test the real database and all three retrieval
    # paths. They are an environment fixture, not an embedding/QA benchmark.
    with tempfile.TemporaryDirectory(prefix="simplemem-preflight-") as folder:
        store = LanceDBVectorStoreBackend(folder, "preflight", vector_dimension=3)
        records = [
            VectorStoreRecord(
                "coffee", [1.0, 0.0, 0.0],
                {"lossless_restatement": "Alice likes coffee in Shanghai",
                 "keywords": ["coffee"], "timestamp": "2026-11-13T09:00:00",
                 "location": "Shanghai", "persons": ["Alice"],
                 "entities": [], "topic": "preferences"},
            ),
            VectorStoreRecord(
                "hiking", [0.0, 1.0, 0.0],
                {"lossless_restatement": "Bob enjoys hiking in Beijing",
                 "keywords": ["hiking"], "timestamp": "2026-11-13T10:00:00",
                 "location": "Beijing", "persons": ["Bob"],
                 "entities": [], "topic": "hobbies"},
            ),
        ]
        store.insert(records)
        assert store.count() == 2
        assert store.semantic_search([1.0, 0.0, 0.0], top_k=1)[0].entry_id == "coffee"
        assert store.keyword_search(["coffee"], top_k=1)[0].entry_id == "coffee"
        assert store.structured_search(persons=["Bob"])[0].entry_id == "hiking"
    print("LanceDB insert + semantic/keyword/structured retrieval: OK")
    print("No external API calls made. API adapters and dataset are not checked here.")
    if args.require_keys and missing:
        print("Missing credentials: " + ", ".join(missing))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
