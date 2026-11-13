"""Freeze deterministic, category-balanced questions before formal evaluation."""

import json
import random
from collections import Counter

from experiments.bootstrap import REPO_ROOT
from experiments.locomo import CATEGORIES, DATA_REVISION, DATA_SHA256, load_dataset


def make_design(samples):
    rng = random.Random(5660)
    sample_indices = [0, 1]  # First two official conversations, not performance-selected.
    excluded = {(0, 0)}  # Used only for pilot/debugging.
    questions = []
    for category in (1, 2, 3, 4):
        pool = [(i, j) for i in sample_indices for j, qa in enumerate(samples[i]["qa"])
                if qa["category"] == category and (i, j) not in excluded]
        for i, j in sorted(rng.sample(pool, 6)):
            questions.append({"sample_index": i, "sample_id": samples[i]["sample_id"],
                              "qa_index": j, "category": category})
    return {
        "design_version": 1, "status": "frozen_before_formal_runs",
        "dataset_commit": DATA_REVISION, "dataset_sha256": DATA_SHA256,
        "selection_seed": 5660, "sample_indices": sample_indices,
        "pilot_exclusions": [{"sample_index": 0, "qa_index": 0}],
        "questions_per_category": 6, "category_names": CATEGORIES,
        "questions": questions, "trials": 3,
        "conditions": {"baseline": {"enable_reflection": True},
                       "ablation": {"enable_reflection": False}},
        "condition_order": [["baseline", "ablation"], ["ablation", "baseline"],
                            ["baseline", "ablation"]],
        "settings": {
            "LLM_MODEL": "deepseek-flash", "EMBEDDING_BACKEND": "openrouter",
            "EMBEDDING_MODEL": "qwen/qwen3-embedding-8b", "EMBEDDING_DIMENSION": 1024,
            "ENABLE_THINKING": False, "USE_STREAMING": False,
            "WINDOW_SIZE": 40, "OVERLAP_SIZE": 2, "SEMANTIC_TOP_K": 25,
            "KEYWORD_TOP_K": 5, "STRUCTURED_TOP_K": 5,
            "ENABLE_PLANNING": True, "MAX_REFLECTION_ROUNDS": 2,
            "MAX_PARALLEL_WORKERS": 4, "MAX_RETRIEVAL_WORKERS": 4,
            "MAX_TOKENS": 16384, "API_TIMEOUT": 120.0,
        },
        "question_workers": 2,
        "repeat_scope": "independent memory build per trial and conversation; same memory for both conditions",
        "metric": "upstream set-token F1; category-macro and question-mean reported separately",
        "failure_policy": "preserve failed rows with F1=0 in all-question mean; report failures separately",
        "cache_policy": "shared persistent embedding cache, actual API/cache usage reported; not cold-cache timing",
    }


def main():
    design = make_design(load_dataset())
    target = REPO_ROOT / "experiments/configs/design.json"
    serialized = json.dumps(design, indent=2) + "\n"
    if target.exists() and target.read_text() != serialized:
        raise RuntimeError("A different design already exists; version it explicitly instead of overwriting")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(serialized)
    print(f"Frozen {len(design['questions'])} questions over two full conversations, three paired trials.")
    print("Category counts:", dict(Counter(q["category"] for q in design["questions"])))
    print("Conversation counts:", dict(Counter(q["sample_id"] for q in design["questions"])))
    print(target)


if __name__ == "__main__":
    main()
