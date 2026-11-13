"""Small paid API check using synthetic facts, not a benchmark result."""

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from experiments.bootstrap import REPO_ROOT, load_environment


def main():
    load_environment()
    if any(not os.getenv(key, "").strip() for key in ("DEEPSEEK_API_KEY", "OPENROUTER_API_KEY")):
        raise RuntimeError("Fill both provider API keys in .env first")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    log_path = REPO_ROOT / ".runs" / f"api-smoke-{run_id}.jsonl"
    os.environ["SIMPLEMEM_API_LOG"] = str(log_path)

    import numpy as np
    from main import SimpleMemSystem
    from simplemem.core.models.memory_entry import MemoryEntry
    from simplemem.core.settings import settings

    if settings._user_config is not None:
        raise RuntimeError("Remove top-level config.py; it overrides experiment settings")
    with tempfile.TemporaryDirectory(prefix="simplemem-api-smoke-") as folder:
        system = SimpleMemSystem(db_path=folder, use_streaming=False,
                                 enable_thinking=False, enable_parallel_processing=False)
        entries = [
            MemoryEntry(lossless_restatement="Alice drinks coffee in Shanghai.",
                        keywords=["coffee", "Shanghai"], persons=["Alice"], location="Shanghai"),
            MemoryEntry(lossless_restatement="Bob enjoys hiking in Beijing.",
                        keywords=["hiking", "Beijing"], persons=["Bob"], location="Beijing"),
        ]
        system.vector_store.add_entries(entries)
        query = "In which city does Alice drink coffee?"
        found = system.vector_store.semantic_search(query, top_k=1)
        if not found or found[0].entry_id != entries[0].entry_id:
            raise RuntimeError("Semantic retrieval did not return the expected synthetic fact")
        # Repeating an identical query must use the local cache, not another API call.
        before = len(log_path.read_text().splitlines())
        vector = system.embedding_model.encode_single(query, is_query=True)
        np.testing.assert_allclose(np.linalg.norm(vector), 1.0, atol=1e-5)
        new_events = [json.loads(line) for line in log_path.read_text().splitlines()[before:]]
        if any(row["kind"] == "embedding" for row in new_events):
            raise RuntimeError("Repeated query missed embedding cache")
        answer = system.llm_client.chat_completion(
            [{"role": "system", "content": "Answer using only the supplied fact. Return JSON with a city field."},
             {"role": "user", "content": found[0].lossless_restatement + "\n" + query}],
            temperature=0.0, response_format={"type": "json_object"},
        )
        if system.llm_client.extract_json(answer).get("city", "").lower() != "shanghai":
            raise RuntimeError("Chat output did not preserve the supplied fact")

    events = [json.loads(line) for line in log_path.read_text().splitlines()]
    summary = {
        "purpose": "API connectivity and adapter validation; not a LoCoMo result",
        "run_id": run_id, "status": "passed",
        "chat_model": settings.LLM_MODEL, "thinking": False,
        "embedding_model": settings.EMBEDDING_MODEL,
        "embedding_dimension": len(vector), "embedding_normalized": True,
        "cache_verified": True, "synthetic_retrieval_correct": True,
        "chat_json_correct": True,
        "api_events": [row for row in events if row["kind"] in ("chat", "embedding")],
        "cache_hits": sum(row.get("cache_hits", 0) for row in events),
    }
    target = REPO_ROOT / "experiments/results/api_smoke.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Summary: {target}")


if __name__ == "__main__":
    main()
