"""Verify scoring against author code without importing heavy metric models."""

import ast
import unittest

from experiments.bootstrap import REPO_ROOT
from experiments.locomo import conversation_to_dialogues, load_dataset, score_answer
from experiments.freeze_design import make_design


class LoCoMoHelperTests(unittest.TestCase):
    def test_scores_match_author_implementation(self):
        tree = ast.parse((REPO_ROOT / "test_locomo10.py").read_text())
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name in ("simple_tokenize", "calculate_metrics")]
        self.assertEqual(len(functions), 2)
        # Execute only these pure functions. Stub unrelated metrics, which are
        # intentionally outside this experiment, to avoid local model downloads.
        scope = {"Dict": dict,
                 "calculate_rouge_scores": lambda *a: {},
                 "calculate_bleu_scores": lambda *a: {},
                 "calculate_bert_scores": lambda *a: {},
                 "calculate_meteor_score": lambda *a: 0.0,
                 "calculate_sentence_similarity": lambda *a: 0.0}
        exec(compile(ast.Module(body=functions, type_ignores=[]), "upstream_metrics", "exec"), scope)
        for prediction, reference in [
            ("coffee coffee tea", "coffee tea"), ("7 May, 2023!", "7 May 2023"),
            ("DOG", "dog"), ("hello-world", "hello world"), ("", "x"),
            ("x", None), ("", ""), ("other", "reference"), (123, 123),
        ]:
            with self.subTest(prediction=prediction, reference=reference):
                expected = scope["calculate_metrics"](prediction, reference)
                actual = score_answer(prediction, reference)
                self.assertEqual(actual, {key: expected[key] for key in ("exact_match", "f1")})

    def test_full_conversation_conversion_uses_no_qa_or_summary_fields(self):
        sample = load_dataset()[0]
        # Converting only the conversation must work even when privileged
        # evaluation annotations are absent.
        dialogues = conversation_to_dialogues({"conversation": sample["conversation"]})
        self.assertEqual(len(dialogues), 419)
        self.assertEqual([d.dialogue_id for d in dialogues], list(range(1, 420)))
        self.assertEqual(dialogues[0].timestamp, sample["conversation"]["session_1_date_time"])

    def test_frozen_selection_is_balanced_unique_and_excludes_pilot(self):
        from collections import Counter
        samples = load_dataset()
        design = make_design(samples)
        questions = design["questions"]
        self.assertEqual(len(questions), 24)
        self.assertEqual(Counter(q["category"] for q in questions), {1: 6, 2: 6, 3: 6, 4: 6})
        identities = {(q["sample_index"], q["qa_index"]) for q in questions}
        self.assertEqual(len(identities), 24)
        self.assertNotIn((0, 0), identities)
        self.assertEqual(design, make_design(samples))
        for question in questions:
            source = samples[question["sample_index"]]["qa"][question["qa_index"]]
            self.assertEqual(source["category"], question["category"])


if __name__ == "__main__":
    unittest.main()
