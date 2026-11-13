"""Validate scheduled rows and summarize paired trials without extra API calls."""

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

from experiments.bootstrap import REPO_ROOT
from experiments.run_state import atomic_json, expected_jobs, read_events, read_json
from experiments.usage import summarize_usage


def distribution(values):
    return {"mean": statistics.mean(values),
            "sample_variance": statistics.variance(values) if len(values) > 1 else 0.0,
            "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "values": values}


def summarize_rows(design, rows):
    identities = [row["job_id"] for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("Duplicate formal result identity")
    expected = expected_jobs(design)
    if set(identities) - expected:
        raise ValueError("Unexpected result outside frozen design")
    for row in rows:
        expected_id = f"t{row['trial']}-s{row['sample_index']}-q{row['qa_index']}-{row['condition']}"
        if row["job_id"] != expected_id:
            raise ValueError("Result fields do not match identity")
        source = next(q for q in design["questions"]
                      if (q["sample_index"], q["qa_index"]) == (row["sample_index"], row["qa_index"]))
        if row["category"] != source["category"] or not 0 <= row["f1"] <= 1:
            raise ValueError("Invalid category or F1 in result")
    summary = {"status": "complete" if set(identities) == expected else "partial",
               "scheduled_rows": len(expected), "completed_rows": len(rows),
               "missing_jobs": sorted(expected - set(identities)),
               "failed_rows": sum(row["status"] != "ok" for row in rows),
               "trials": [], "conditions": {}}
    if summary["status"] != "complete":
        return summary
    for trial in range(design["trials"]):
        trial_row = {"trial": trial}
        for condition in design["conditions"]:
            selected = [r for r in rows if r["trial"] == trial and r["condition"] == condition]
            categories = {str(category): statistics.mean(r["f1"] for r in selected if r["category"] == category)
                          for category in (1, 2, 3, 4)}
            trial_row[condition] = {
                "category_f1": categories,
                "category_macro_f1": statistics.mean(categories.values()),
                "question_mean_f1": statistics.mean(r["f1"] for r in selected),
                "exact_match": statistics.mean(r["exact_match"] for r in selected),
                "failed_rows": sum(r["status"] != "ok" for r in selected),
                "mean_qa_seconds": statistics.mean(r["elapsed_seconds"] for r in selected),
                "chat_calls_per_question": statistics.mean(r["usage_summary"]["api_attempts"].get("chat", 0) for r in selected),
                "chat_tokens_per_question": statistics.mean(
                    r["usage_summary"]["chat_input_tokens"] + r["usage_summary"]["chat_output_tokens"] for r in selected),
            }
        trial_row["paired_delta_ablation_minus_baseline"] = (
            trial_row["ablation"]["category_macro_f1"] - trial_row["baseline"]["category_macro_f1"])
        summary["trials"].append(trial_row)
    for condition in design["conditions"]:
        summary["conditions"][condition] = {
            metric: distribution([trial[condition][metric] for trial in summary["trials"]])
            for metric in ("category_macro_f1", "question_mean_f1", "exact_match", "mean_qa_seconds",
                           "chat_calls_per_question", "chat_tokens_per_question")}
        summary["conditions"][condition]["category_f1"] = {
            str(category): distribution([trial[condition]["category_f1"][str(category)] for trial in summary["trials"]])
            for category in (1, 2, 3, 4)}
    summary["paired_delta_ablation_minus_baseline"] = distribution(
        [trial["paired_delta_ablation_minus_baseline"] for trial in summary["trials"]])
    return summary


def export_results(run_dir):
    run_dir = Path(run_dir)
    manifest = read_json(run_dir / "manifest.json")
    rows = [read_json(path) for path in sorted((run_dir / "rows").glob("*.json"))]
    summary = summarize_rows(manifest["design"], rows)
    events = read_events(run_dir / "api.jsonl")
    summary["run_id"] = manifest["run_id"]
    summary["source_fingerprint"] = manifest["source"]["fingerprint"]
    summary["total_recorded_api_usage"] = summarize_usage(events)
    builds = [read_json(path) for path in sorted((run_dir / "builds").glob("*/build.json"))]
    summary["builds"] = [{k: build[k] for k in ("trial", "sample_index", "memory_count", "elapsed_seconds", "usage_summary")}
                          for build in builds if build["status"] == "complete"]
    summary["condition_batches"] = [read_json(path) for path in sorted((run_dir / "batches").glob("*.json"))]
    if summary["status"] == "complete" and len(summary["builds"]) != manifest["design"]["trials"] * len(manifest["design"]["sample_indices"]):
        raise ValueError("Result rows complete, but memory builds are incomplete")
    output = REPO_ROOT / "experiments/results"
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "formal_summary.json", summary)
    atomic_json(output / "formal_manifest.json", manifest)
    # Raw contexts and individual request metadata stay in .runs; the committed
    # table includes answers, scores, IDs and measured resource summaries.
    public_rows = [{k: v for k, v in row.items() if k not in ("contexts", "api_events", "trace")}
                   for row in sorted(rows, key=lambda r: r["job_id"])]
    with (output / "formal_rows.jsonl").open("w") as stream:
        for row in public_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    columns = ["job_id", "trial", "sample_id", "qa_index", "category", "condition", "status",
               "question", "reference", "answer", "f1", "exact_match", "elapsed_seconds"]
    with (output / "formal_rows.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(public_rows)
    return summary
