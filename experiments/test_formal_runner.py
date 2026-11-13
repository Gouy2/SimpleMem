"""Offline tests for failed rows, resume guards and paired aggregation."""

import copy
import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

from experiments.locomo import memory_digest
from experiments.run_formal import evaluate_question
from experiments.run_state import content_hash, ensure_manifest, expected_jobs, job_id, read_events
from experiments.summarize_formal import summarize_rows
from simplemem.core.models.memory_entry import MemoryEntry
from simplemem.core.utils.api_metrics import current_api_event_context, record_api_event, record_pipeline_fallback


def small_design():
    return {"trials": 3, "conditions": {"baseline": {"enable_reflection": True},
                                       "ablation": {"enable_reflection": False}},
            "questions": [{"sample_index": 0, "qa_index": c, "category": c} for c in (1, 2, 3, 4)]}


class FormalRunnerTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name)
        env = patch.dict(os.environ, {"SIMPLEMEM_API_LOG": str(self.path / "api.jsonl")})
        env.start()
        self.addCleanup(env.stop)

    def test_resume_refuses_changed_source_design_or_environment(self):
        expected = {"design": {"question": 1}, "source": {"hash": "a"},
                    "environment": {"python": "3.12"}, "effective_settings": {"model": "a"}}
        target = self.path / "manifest.json"
        ensure_manifest(target, expected)
        self.assertEqual(ensure_manifest(target, expected), expected)
        for key in expected:
            changed = copy.deepcopy(expected)
            changed[key] = {"changed": True}
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                ensure_manifest(target, changed)

    def test_design_identity_survives_json_object_key_conversion(self):
        value = {"category_names": {1: "multi_hop", 2: "temporal"}}
        self.assertEqual(content_hash(value), content_hash(json.loads(json.dumps(value))))

    def fake_system(self, trace, fallback=False):
        entry = MemoryEntry(entry_id="fixed", lossless_restatement="gold", keywords=[])
        def retrieve(query):
            if system.hybrid_retriever.enable_reflection:
                trace.events.append({**current_api_event_context(), "stage": "reflection", "status": "ok"})
            return [entry]
        def answer(query, contexts):
            record_api_event(kind="chat", status="ok", usage={"prompt_tokens": 2, "completion_tokens": 1})
            if fallback:
                record_pipeline_fallback("query_analysis", ValueError("test"))
            return "gold"
        system = NS(hybrid_retriever=NS(enable_reflection=True, retrieve=retrieve),
                    answer_generator=NS(generate_answer=answer), get_all_memories=lambda: [entry])
        return system

    def evaluate(self, fallback=False):
        trace = NS(events=[], lock=threading.Lock())
        system = self.fake_system(trace, fallback)
        build = {"memory_digest": memory_digest(system.get_all_memories())}
        manifest = {"run_id": "test", "source": {"fingerprint": "unit"}, "design": small_design()}
        selected = {"sample_index": 0, "sample_id": "unit", "qa_index": 1, "category": 1}
        args = (system, trace, build, self.path, manifest, 0, selected,
                {"question": "question", "answer": "gold"}, "baseline")
        return evaluate_question(*args), args

    def test_success_row_resume_does_not_repeat_requests(self):
        row, args = self.evaluate()
        self.assertEqual(row["f1"], 1.0)
        self.assertEqual(row["status"], "ok")
        before = (self.path / "api.jsonl").read_text()
        args[0].hybrid_retriever.retrieve = lambda q: self.fail("Repeated billable task")
        self.assertEqual(evaluate_question(*args), row)
        self.assertEqual((self.path / "api.jsonl").read_text(), before)

    def test_swallowed_fallback_preserves_raw_answer_but_scores_failure_zero(self):
        row, _ = self.evaluate(fallback=True)
        self.assertEqual(row["answer"], "gold")
        self.assertEqual(row["status"], "fallback")
        self.assertEqual(row["raw_score"]["f1"], 1.0)
        self.assertEqual(row["f1"], 0.0)
        self.assertEqual(len(row["fallbacks"]), 1)

    def test_resume_rejects_changed_memory(self):
        _, args = self.evaluate()
        args[2]["memory_digest"] = "changed"
        with self.assertRaises(RuntimeError):
            evaluate_question(*args)

    def rows(self):
        rows = []
        for trial in range(3):
            for question in small_design()["questions"]:
                for condition in ("baseline", "ablation"):
                    rows.append({**question, "job_id": job_id(trial, 0, question["qa_index"], condition),
                                 "trial": trial, "condition": condition, "status": "ok",
                                 "f1": .5 + .1 * trial - (.1 if condition == "ablation" else 0),
                                 "exact_match": 0, "elapsed_seconds": 2.0,
                                 "usage_summary": {"api_attempts": {"chat": 1},
                                                   "chat_input_tokens": 2, "chat_output_tokens": 1}})
        return rows

    def test_three_trial_variance_and_paired_delta(self):
        summary = summarize_rows(small_design(), self.rows())
        self.assertEqual(summary["status"], "complete")
        self.assertEqual(summary["completed_rows"], 24)
        self.assertAlmostEqual(summary["conditions"]["baseline"]["category_macro_f1"]["mean"], .6)
        self.assertAlmostEqual(summary["conditions"]["baseline"]["category_macro_f1"]["sample_variance"], .01)
        self.assertAlmostEqual(summary["paired_delta_ablation_minus_baseline"]["mean"], -.1)

    def test_missing_duplicate_and_failed_rows(self):
        rows = self.rows()
        partial = summarize_rows(small_design(), rows[:-1])
        self.assertEqual(partial["status"], "partial")
        self.assertEqual(len(partial["missing_jobs"]), 1)
        self.assertEqual(partial["conditions"], {})
        with self.assertRaises(ValueError):
            summarize_rows(small_design(), rows + rows[:1])
        rows[0].update(status="error", f1=0.0)
        summary = summarize_rows(small_design(), rows)
        self.assertEqual(summary["failed_rows"], 1)
        self.assertAlmostEqual(summary["trials"][0]["baseline"]["question_mean_f1"], .375)

    def test_event_reader_ignores_only_incomplete_trailing_line(self):
        path = self.path / "log.jsonl"
        path.write_text('{"kind":"ok"}\n{"incomplete":')
        self.assertEqual(read_events(path), [{"kind": "ok"}])


if __name__ == "__main__":
    unittest.main()
