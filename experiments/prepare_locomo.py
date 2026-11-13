"""Download, validate and describe the official data without running a model."""

import json
from collections import Counter

import httpx

from experiments.bootstrap import REPO_ROOT
from experiments.locomo import DATA_PATH, DATA_REVISION, DATA_SHA256, DATA_URL, load_dataset


def main():
    if not DATA_PATH.exists():
        response = httpx.get(DATA_URL, timeout=60, follow_redirects=True)
        response.raise_for_status()
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        DATA_PATH.write_bytes(response.content)
    samples = load_dataset()
    manifest = {"source_url": DATA_URL, "source_commit": DATA_REVISION,
                "sha256": DATA_SHA256, "sample_count": len(samples), "samples": []}
    for index, sample in enumerate(samples):
        sessions = [value for key, value in sample["conversation"].items()
                    if key.startswith("session_") and isinstance(value, list)]
        manifest["samples"].append({
            "index": index, "sample_id": sample["sample_id"],
            "sessions": len(sessions), "turns": sum(map(len, sessions)),
            "questions": len(sample["qa"]),
            "category_counts": dict(Counter(qa["category"] for qa in sample["qa"])),
        })
    target = REPO_ROOT / "experiments/configs/data_manifest.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Validated {len(samples)} conversations; manifest: {target}")


if __name__ == "__main__":
    main()
