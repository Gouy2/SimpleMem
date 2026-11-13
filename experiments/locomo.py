"""Pinned official LoCoMo data and lightweight upstream-compatible scoring."""

import hashlib
import json
from pathlib import Path

from experiments.bootstrap import REPO_ROOT

DATA_REVISION = "3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376"
DATA_URL = f"https://raw.githubusercontent.com/snap-research/locomo/{DATA_REVISION}/data/locomo10.json"
DATA_SHA256 = "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4"
DATA_PATH = REPO_ROOT / ".cache/data/locomo10.json"
CATEGORIES = {1: "multi_hop", 2: "temporal", 3: "open_domain", 4: "single_hop"}


def load_dataset(path=DATA_PATH):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != DATA_SHA256:
        raise ValueError("LoCoMo checksum differs from the pinned official data")
    return json.loads(raw)


def conversation_to_dialogues(sample):
    # Only conversation turns are visible to the memory pipeline. Never pass
    # QA answers, evidence labels, observations or supplied summaries to it.
    from simplemem.core.models.memory_entry import Dialogue
    conversation = sample["conversation"]
    session_ids = sorted(int(key.split("_")[1]) for key, value in conversation.items()
                         if key.startswith("session_") and isinstance(value, list))
    dialogues = []
    for session_id in session_ids:
        session = f"session_{session_id}"
        timestamp = conversation[f"{session}_date_time"]
        for turn in conversation[session]:
            text = turn.get("text", "")
            if "img_url" in turn and "blip_caption" in turn:
                text = f"[Image: {turn['blip_caption']}]" + (f" {text}" if text else "")
            dialogues.append(Dialogue(dialogue_id=len(dialogues) + 1,
                                      speaker=turn["speaker"], content=text, timestamp=timestamp))
    return dialogues


def simple_tokenize(text):
    # Exact normalization used in upstream test_locomo10.py.
    return str(text).lower().replace('.', ' ').replace(',', ' ').replace('!', ' ').replace('?', ' ').split()


def score_answer(prediction, reference):
    if not prediction or not reference:
        return {"exact_match": 0, "f1": 0.0}
    prediction, reference = str(prediction).strip(), str(reference).strip()
    pred, ref = set(simple_tokenize(prediction)), set(simple_tokenize(reference))
    overlap = len(pred & ref)
    precision = overlap / len(pred) if pred else 0.0
    recall = overlap / len(ref) if ref else 0.0
    return {"exact_match": int(prediction.lower() == reference.lower()),
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}


def memory_digest(entries):
    # Sort by ID so backend row order does not affect the read-only check.
    payload = sorted((entry.model_dump() for entry in entries), key=lambda item: item["entry_id"])
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
