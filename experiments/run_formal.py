"""Execute the frozen paired experiment with atomic, validated checkpoints."""

import argparse
import contextlib
import fcntl
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from experiments.bootstrap import REPO_ROOT, load_environment
from experiments.locomo import conversation_to_dialogues, load_dataset, memory_digest, score_answer
from experiments.run_state import atomic_json, content_hash, ensure_manifest, job_id, read_events, read_json, source_identity
from experiments.summarize_formal import export_results
from experiments.usage import summarize_usage


def tagged_events(path, identity):
    return [event for event in read_events(path) if event.get("job_id") == identity]


def configure(design):
    load_environment()
    for key, value in design["settings"].items():
        os.environ[key] = str(value).lower() if isinstance(value, bool) else str(value)
    if not all(os.getenv(key, "").strip() for key in ("DEEPSEEK_API_KEY", "OPENROUTER_API_KEY")):
        raise RuntimeError("Both API keys are required")
    from simplemem.core.settings import settings
    if settings._user_config is not None:
        raise RuntimeError("Top-level config.py overrides frozen settings")
    return {key: getattr(settings, key) for key in (
        *design["settings"], "OPENAI_BASE_URL", "EMBEDDING_BASE_URL", "EMBEDDING_QUERY_INSTRUCTION",
        "EMBEDDING_BATCH_SIZE", "USE_JSON_FORMAT")}


def make_system(db_path, design):
    from main import SimpleMemSystem
    s = design["settings"]
    return SimpleMemSystem(
        db_path=str(db_path), enable_thinking=False, use_streaming=False,
        enable_parallel_processing=True, max_parallel_workers=s["MAX_PARALLEL_WORKERS"],
        enable_parallel_retrieval=True, max_retrieval_workers=s["MAX_RETRIEVAL_WORKERS"],
        enable_planning=True, enable_reflection=True, max_reflection_rounds=s["MAX_REFLECTION_ROUNDS"])


def prepare_memory(run_dir, manifest, design, trial, sample_index, sample):
    from experiments.pipeline_smoke import PipelineTrace
    from simplemem.core.utils.api_metrics import api_event_context
    identity = f"build-t{trial}-s{sample_index}"
    directory = run_dir / "builds" / identity
    checkpoint = directory / "build.json"
    resumed = checkpoint.exists()
    if resumed:
        build = read_json(checkpoint)
        if build["status"] != "complete":
            raise RuntimeError(f"Incomplete memory build preserved at {directory}; inspect it before explicitly starting a new run")
    elif (directory / "memory").exists():
        raise RuntimeError(f"Incomplete uncheckpointed memory at {directory}; refusing silent rebuild")
    directory.mkdir(parents=True, exist_ok=True)
    system = make_system(directory / "memory", design)
    trace = PipelineTrace()
    trace.attach(system)
    if resumed:
        if (build["source_fingerprint"] != manifest["source"]["fingerprint"]
                or memory_digest(system.get_all_memories()) != build["memory_digest"]):
            raise RuntimeError("Resumed memory differs from its checkpoint")
        return system, trace, build
    started = time.perf_counter()
    build = {"trial": trial, "sample_index": sample_index, "sample_id": sample["sample_id"],
             "status": "building", "source_fingerprint": manifest["source"]["fingerprint"]}
    atomic_json(checkpoint, build)
    try:
        dialogues = conversation_to_dialogues(sample)
        with api_event_context(run_id=manifest["run_id"], phase="build", job_id=identity,
                               trial=trial, sample_index=sample_index, sample_id=sample["sample_id"]):
            system.add_dialogues(dialogues)
            system.finalize()
        events = tagged_events(run_dir / "api.jsonl", identity)
        windows = [event for event in trace.events if event["stage"] == "memory_window"]
        covered = {i for event in windows if event.get("result_count", 0) for i in event["dialogue_ids"]}
        if covered != {d.dialogue_id for d in dialogues} or any(e["kind"] == "pipeline_fallback" for e in events):
            raise RuntimeError("Memory build had failed windows or fallback paths")
        memories = system.get_all_memories()
        build.update(status="complete", memory_count=len(memories), memory_digest=memory_digest(memories),
                     dialogue_turns=len(dialogues), covered_turns=len(covered), trace=trace.events[:],
                     elapsed_seconds=time.perf_counter() - started, usage_summary=summarize_usage(events))
        atomic_json(checkpoint, build)
        return system, trace, build
    except Exception as error:
        build.update(status="failed", error_type=type(error).__name__, elapsed_seconds=time.perf_counter() - started)
        atomic_json(checkpoint, build)
        raise


def evaluate_question(system, trace, build, run_dir, manifest, trial, selected, qa, condition):
    from simplemem.core.utils.api_metrics import api_event_context
    identity = job_id(trial, selected["sample_index"], selected["qa_index"], condition)
    path = run_dir / "rows" / f"{identity}.json"
    if path.exists():
        row = read_json(path)
        if row["memory_digest"] != build["memory_digest"] or row["source_fingerprint"] != manifest["source"]["fingerprint"]:
            raise RuntimeError("Saved row does not match memory/source checkpoint")
        return row
    row = {**selected, "job_id": identity, "trial": trial, "condition": condition,
           "question": qa["question"], "reference": str(qa["answer"]),
           "memory_digest": build["memory_digest"], "source_fingerprint": manifest["source"]["fingerprint"],
           "status": "ok", "answer": "", "contexts": [], "f1": 0.0, "exact_match": 0}
    started = time.perf_counter()
    with api_event_context(run_id=manifest["run_id"], phase="qa", job_id=identity,
                           trial=trial, sample_index=selected["sample_index"], sample_id=selected["sample_id"],
                           qa_index=selected["qa_index"], condition=condition):
        try:
            contexts = system.hybrid_retriever.retrieve(qa["question"])
            answer = system.answer_generator.generate_answer(qa["question"], contexts)
            row.update(answer=answer, contexts=[entry.model_dump() for entry in contexts],
                       context_ids=[entry.entry_id for entry in contexts], **score_answer(answer, qa["answer"]))
        except Exception as error:
            row.update(status="error", error_type=type(error).__name__)
    row["elapsed_seconds"] = time.perf_counter() - started
    row["api_events"] = tagged_events(run_dir / "api.jsonl", identity)
    with trace.lock:
        row["trace"] = [event for event in trace.events if event.get("job_id") == identity]
    row["fallbacks"] = [event for event in row["api_events"] if event["kind"] == "pipeline_fallback"]
    row["raw_score"] = {"f1": row["f1"], "exact_match": row["exact_match"]}
    if row["fallbacks"]:
        row.update(status="fallback", f1=0.0, exact_match=0)
    expected_reflection = manifest["design"]["conditions"][condition]["enable_reflection"]
    reflected = any(event["stage"] == "reflection" for event in row["trace"])
    if row["status"] == "ok" and reflected != expected_reflection:
        row.update(status="error", error_type="ReflectionSwitchMismatch", f1=0.0, exact_match=0)
    if memory_digest(system.get_all_memories()) != build["memory_digest"]:
        row.update(status="error", error_type="MemoryMutation", f1=0.0, exact_match=0)
    row["usage_summary"] = summarize_usage(row["api_events"])
    row["usage_by_operation"] = {
        operation: summarize_usage([e for e in row["api_events"] if e.get("operation") == operation])
        for operation in sorted({e.get("operation", "unlabeled") for e in row["api_events"]})}
    atomic_json(path, row)
    return row


def execute(run_dir, design, manifest, console):
    from simplemem.core.utils.api_metrics import submit_with_context
    samples = load_dataset()
    api_path = run_dir / "api.jsonl"
    os.environ["SIMPLEMEM_API_LOG"] = str(api_path)
    for trial in range(design["trials"]):
        for sample_index in design["sample_indices"]:
            sample = samples[sample_index]
            print(f"Trial {trial + 1}/3 {sample['sample_id']}: preparing memory", file=console, flush=True)
            system, trace, build = prepare_memory(run_dir, manifest, design, trial, sample_index, sample)
            print(f"  memory: {build['memory_count']} entries, {build['elapsed_seconds']:.1f}s", file=console, flush=True)
            selected = [q for q in design["questions"] if q["sample_index"] == sample_index]
            for condition in design["condition_order"][trial]:
                # Finish all futures before mutating the shared reflection flag.
                system.hybrid_retriever.enable_reflection = design["conditions"][condition]["enable_reflection"]
                batch_path = run_dir / "batches" / f"t{trial}-s{sample_index}-{condition}.json"
                missing = [q for q in selected if not (run_dir / "rows" / f"{job_id(trial, sample_index, q['qa_index'], condition)}.json").exists()]
                started = time.perf_counter()
                if missing:
                    with ThreadPoolExecutor(max_workers=design["question_workers"]) as pool:
                        futures = [submit_with_context(pool, evaluate_question, system, trace, build, run_dir,
                                                       manifest, trial, q, sample["qa"][q["qa_index"]], condition)
                                   for q in missing]
                        for future in as_completed(futures):
                            row = future.result()
                            print(f"  {row['job_id']}: {row['status']} F1={row['f1']:.3f}", file=console, flush=True)
                            if row.get("error_type") in ("MemoryMutation", "ReflectionSwitchMismatch"):
                                raise RuntimeError("Experiment invariant failed; raw state preserved")
                    batch = read_json(batch_path) if batch_path.exists() else {"trial": trial, "sample_index": sample_index,
                                                                              "condition": condition, "segments": []}
                    batch["segments"].append({"completed_in_segment": len(missing),
                                               "wall_seconds": time.perf_counter() - started})
                    atomic_json(batch_path, batch)
                # Validate even rows skipped on resume and verify the store stayed read-only.
                for q in selected:
                    saved = read_json(run_dir / "rows" / f"{job_id(trial, sample_index, q['qa_index'], condition)}.json")
                    if saved["memory_digest"] != build["memory_digest"] or saved["source_fingerprint"] != manifest["source"]["fingerprint"]:
                        raise RuntimeError("Saved row identity mismatch")
                if memory_digest(system.get_all_memories()) != build["memory_digest"]:
                    raise RuntimeError("Memory mutation between conditions")
            export_results(run_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", help="Existing run directory to resume; otherwise create a fresh run")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if args.summarize_only:
        if not args.run_dir:
            parser.error("--summarize-only requires --run-dir")
        print(json.dumps(export_results(Path(args.run_dir).resolve()), indent=2))
        return
    design = read_json(REPO_ROOT / "experiments/configs/design.json")
    from experiments.freeze_design import make_design
    if content_hash(design) != content_hash(make_design(load_dataset())):
        raise RuntimeError("Design differs from the preregistered deterministic selection")
    effective = configure(design)
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = Path(args.run_dir).resolve() if args.run_dir else REPO_ROOT / ".runs" / f"formal-{now}"
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = ensure_manifest(run_dir / "manifest.json", {
            "run_id": run_dir.name, "started_utc": now, "design": design,
            "source": source_identity(), "effective_settings": effective,
            "environment": {"python": platform.python_version(), "machine": platform.machine(),
                            "packages": sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions())},
            "git_head_at_start": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).decode().strip(),
        })
        print(f"Run directory: {run_dir}", flush=True)
        console = sys.stdout
        with (run_dir / "pipeline.log").open("a", buffering=1) as log, contextlib.redirect_stdout(log):
            try:
                execute(run_dir, design, manifest, console)
            finally:
                export_results(run_dir)
        summary = read_json(REPO_ROOT / "experiments/results/formal_summary.json")
        print(json.dumps({k: summary[k] for k in ("status", "completed_rows", "failed_rows", "conditions",
                                                  "paired_delta_ablation_minus_baseline")}, indent=2))


if __name__ == "__main__":
    main()
