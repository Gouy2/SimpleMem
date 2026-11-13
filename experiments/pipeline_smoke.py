"""Pilot a complete official conversation; do not treat this as a final trial."""

import argparse
import contextlib
import functools
import json
import os
import threading
import time
from collections import Counter
from datetime import datetime, timezone

from experiments.bootstrap import REPO_ROOT, load_environment
from experiments.locomo import DATA_SHA256, conversation_to_dialogues, load_dataset, memory_digest, score_answer
from experiments.usage import summarize_usage
from simplemem.core.utils.api_metrics import current_api_event_context


def finalize_usage(result, run_dir):
    """Can also enrich a completed pilot offline without repeating API calls."""
    events = [json.loads(line) for line in (run_dir / "api.jsonl").read_text().splitlines()]
    qa_event_count = sum(len(condition["api_events"]) for condition in result["conditions"])
    build_events = events[:len(events) - qa_event_count]
    result["build_api_usage"] = summarize_usage(build_events)
    for condition in result["conditions"]:
        condition["usage_summary"] = summarize_usage(condition["api_events"])
    result["total_api_usage"] = summarize_usage(events)


class PipelineTrace:
    """Observe existing methods without changing their prompts or return values."""
    def __init__(self):
        self.events = []
        self.lock = threading.Lock()

    def wrap(self, obj, method, stage):
        original = getattr(obj, method)

        @functools.wraps(original)
        def traced(*args, **kwargs):
            started = time.perf_counter()
            event = {**current_api_event_context(), "stage": stage}
            if stage == "memory_window":
                event["dialogue_ids"] = [d.dialogue_id for d in args[0]]
            try:
                value = original(*args, **kwargs)
                event["status"] = "ok"
                if isinstance(value, list):
                    event["result_count"] = len(value)
                if stage == "completeness_check":
                    event["result"] = value
                return value
            except Exception as error:
                event.update(status="error", error_type=type(error).__name__)
                raise
            finally:
                event["elapsed_seconds"] = time.perf_counter() - started
                with self.lock:
                    self.events.append(event)
        setattr(obj, method, traced)

    def attach(self, system):
        self.wrap(system.memory_builder, "_generate_memory_entries_worker", "memory_window")
        self.wrap(system.memory_builder, "_generate_memory_entries", "memory_window")
        for method, stage in [
            ("_analyze_information_requirements", "planning"),
            ("_generate_targeted_queries", "query_generation"),
            ("_retrieve_with_intelligent_reflection", "reflection"),
            ("_analyze_information_completeness", "completeness_check"),
        ]:
            self.wrap(system.hybrid_retriever, method, stage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--qa-index", type=int, default=0)
    args = parser.parse_args()
    load_environment()
    # Memory extraction emits a full JSON array. Use a larger output allowance
    # for BOTH conditions to avoid truncating 40-turn windows; disclose it.
    os.environ["MAX_TOKENS"] = "16384"
    os.environ["API_TIMEOUT"] = "120"
    os.environ["USE_STREAMING"] = "false"
    os.environ["ENABLE_THINKING"] = "false"
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = REPO_ROOT / ".runs" / f"pipeline-pilot-{run_id}"
    run_dir.mkdir(parents=True)
    os.environ["SIMPLEMEM_API_LOG"] = str(run_dir / "api.jsonl")
    from main import SimpleMemSystem
    from simplemem.core.settings import settings
    if settings._user_config is not None:
        raise RuntimeError("Top-level config.py overrides experiment settings")
    sample = load_dataset()[args.sample_index]
    qa = sample["qa"][args.qa_index]
    if qa["category"] not in (1, 2, 3, 4):
        raise ValueError("Pilot only supports categories 1-4")
    dialogues = conversation_to_dialogues(sample)
    result = {"purpose": "development pilot; excluded from formal question subset",
              "run_id": run_id, "dataset_sha256": DATA_SHA256,
              "sample_index": args.sample_index, "sample_id": sample["sample_id"],
              "qa_index": args.qa_index, "category": qa["category"],
              "question": qa["question"], "reference": str(qa["answer"]),
              "dialogue_turns": len(dialogues), "conditions": []}
    trace = PipelineTrace()
    with (run_dir / "pipeline.log").open("w") as log, contextlib.redirect_stdout(log):
        system = SimpleMemSystem(
            db_path=str(run_dir / "memory"), enable_thinking=False, use_streaming=False,
            enable_parallel_processing=True, max_parallel_workers=4,
            enable_parallel_retrieval=True, max_retrieval_workers=4,
            enable_planning=True, enable_reflection=True, max_reflection_rounds=2)
        trace.attach(system)
        started = time.perf_counter()
        system.add_dialogues(dialogues)
        system.finalize()
        result["memory_build_seconds"] = time.perf_counter() - started
        memories = system.get_all_memories()
        result["memory_count"] = len(memories)
        windows = [event for event in trace.events if event["stage"] == "memory_window"]
        covered = {i for event in windows if event.get("result_count", 0) for i in event["dialogue_ids"]}
        result["covered_turns"] = len(covered)
        if covered != {d.dialogue_id for d in dialogues}:
            raise RuntimeError("Some conversation windows produced no memories; inspect pilot logs")
        digest = memory_digest(memories)
        result["memory_digest"] = digest
        result["build_trace"] = list(trace.events)
        for reflection in (True, False):
            system.hybrid_retriever.enable_reflection = reflection
            trace_start = len(trace.events)
            api_start = len((run_dir / "api.jsonl").read_text().splitlines())
            started = time.perf_counter()
            contexts = system.hybrid_retriever.retrieve(qa["question"])
            answer = system.answer_generator.generate_answer(qa["question"], contexts)
            elapsed = time.perf_counter() - started
            events = [json.loads(line) for line in (run_dir / "api.jsonl").read_text().splitlines()[api_start:]]
            stage_trace = trace.events[trace_start:]
            reflected = any(e["stage"] == "reflection" for e in stage_trace)
            if reflected != reflection:
                raise RuntimeError("Reflection switch did not control the expected method")
            if memory_digest(system.get_all_memories()) != digest:
                raise RuntimeError("Question answering mutated the memory store")
            result["conditions"].append({
                "reflection": reflection, "answer": answer, **score_answer(answer, qa["answer"]),
                "elapsed_seconds": elapsed, "contexts": len(contexts),
                "context_ids": [entry.entry_id for entry in contexts],
                "trace": stage_trace, "api_events": events,
                "api_calls": dict(Counter(e["kind"] for e in events if e["kind"] in ("chat", "embedding"))),
            })
    result["settings"] = {key: getattr(settings, key) for key in (
        "LLM_MODEL", "EMBEDDING_MODEL", "EMBEDDING_DIMENSION", "WINDOW_SIZE", "OVERLAP_SIZE",
        "SEMANTIC_TOP_K", "KEYWORD_TOP_K", "STRUCTURED_TOP_K", "MAX_TOKENS", "API_TIMEOUT")}
    result["status"] = "passed"
    finalize_usage(result, run_dir)
    target = REPO_ROOT / "experiments/results/pipeline_smoke.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "sample_id", "dialogue_turns", "covered_turns",
                                                 "memory_count", "memory_build_seconds")}, indent=2))
    for condition in result["conditions"]:
        print({key: condition[key] for key in ("reflection", "answer", "f1", "elapsed_seconds", "api_calls")})
    print(f"Result: {target}\nRaw logs: {run_dir}")


if __name__ == "__main__":
    main()
