"""Offline regression tests for billable API boundaries."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np

from simplemem.core.utils.api_embedding import APIEmbeddingModel
from simplemem.core.utils.llm_client import LLMClient


class APIAdapterTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.env = patch.dict(os.environ, {
            "OPENROUTER_API_KEY": "test-only", "OPENAI_API_KEY": "test-only",
            "OPENAI_BASE_URL": "https://api.deepseek.com", "LLM_MODEL": "deepseek-flash",
            "EMBEDDING_MODEL": "qwen/qwen3-embedding-8b", "EMBEDDING_DIMENSION": "2",
            "EMBEDDING_CACHE_PATH": str(Path(self.folder.name) / "cache.sqlite3"),
            "SIMPLEMEM_API_LOG": str(Path(self.folder.name) / "calls.jsonl"),
            "USE_STREAMING": "false", "ENABLE_THINKING": "false",
        })
        self.env.start()
        self.addCleanup(self.env.stop)

    def response(self, vectors, indices=None):
        indices = indices if indices is not None else list(range(len(vectors)))
        return NS(model="qwen/qwen3-embedding-8b", usage=NS(model_dump=lambda: {"prompt_tokens": 5}),
                  data=[NS(index=i, embedding=v) for i, v in zip(indices, vectors)])

    def test_batch_reordering_deduplication_normalization_and_cache(self):
        with patch("simplemem.core.utils.api_embedding.OpenAI") as sdk:
            request = sdk.return_value.embeddings.create
            request.return_value = self.response([[0, 4], [3, 0]], [1, 0])
            model = APIEmbeddingModel()
            np.testing.assert_allclose(model.encode(["a", "b", "a"]), [[1, 0], [0, 1], [1, 0]])
            self.assertEqual(request.call_args.kwargs["input"], ["a", "b"])
            # A fresh adapter must reuse the on-disk cache.
            np.testing.assert_allclose(APIEmbeddingModel().encode(["b", "a"]), [[0, 1], [1, 0]])
            self.assertEqual(request.call_count, 1)

    def test_query_prompt_uses_separate_cache_entry(self):
        with patch("simplemem.core.utils.api_embedding.OpenAI") as sdk:
            request = sdk.return_value.embeddings.create
            request.return_value = self.response([[3, 4]])
            model = APIEmbeddingModel()
            model.encode_documents(["coffee"])
            model.encode_query(["coffee"])
            self.assertEqual(request.call_count, 2)
            self.assertTrue(request.call_args.kwargs["input"][0].startswith("Instruct:"))
            self.assertEqual(model.encode([]).shape, (0, 2))

    def test_invalid_vectors_or_indices_are_not_cached(self):
        with patch("simplemem.core.utils.api_embedding.OpenAI") as sdk:
            request = sdk.return_value.embeddings.create
            model = APIEmbeddingModel()
            for vectors, indices in [([[0, 0]], [0]), ([[1, 2, 3]], [0]),
                                     ([[1, float('nan')]], [0]), ([[1, 0]], [1])]:
                request.return_value = self.response(vectors, indices)
                with self.subTest(vectors=vectors), self.assertRaises(ValueError):
                    model.encode(["invalid"])
            self.assertEqual(request.call_count, 4)

    def test_deepseek_parameters_usage_and_empty_output(self):
        with patch("simplemem.core.utils.llm_client.OpenAI") as sdk:
            response = NS(model="deepseek-flash", id="test-request",
                          usage=NS(model_dump=lambda: {"prompt_tokens": 10, "completion_tokens": 3}),
                          choices=[NS(message=NS(content='{"ok": true}'), finish_reason="stop")])
            request = sdk.return_value.chat.completions.create
            request.return_value = response
            client = LLMClient()
            self.assertEqual(client.chat_completion([{"role": "user", "content": "test"}]), '{"ok": true}')
            self.assertEqual(request.call_args.kwargs["extra_body"], {"thinking": {"type": "disabled"}})
            self.assertEqual(request.call_args.kwargs["max_tokens"], 4096)
            self.assertEqual(sdk.call_args.kwargs["max_retries"], 0)
            events = Path(os.environ["SIMPLEMEM_API_LOG"]).read_text()
            self.assertNotIn("test-only", events)
            self.assertEqual(json.loads(events.splitlines()[0])["usage"]["prompt_tokens"], 10)
            response.choices[0].message.content = ""
            with self.assertRaises(ValueError):
                client.chat_completion([{"role": "user", "content": "test"}])
            self.assertEqual(request.call_count, 2)

    def test_thinking_mode_omits_ineffective_temperature(self):
        with patch("simplemem.core.utils.llm_client.OpenAI") as sdk:
            sdk.return_value.chat.completions.create.return_value = NS(
                model="deepseek-flash", id="test", usage=None,
                choices=[NS(message=NS(content="ok"), finish_reason="stop")])
            LLMClient(enable_thinking=True).chat_completion([])
            kwargs = sdk.return_value.chat.completions.create.call_args.kwargs
            self.assertNotIn("temperature", kwargs)
            self.assertEqual(kwargs["extra_body"]["thinking"]["type"], "enabled")

    def test_streaming_usage_chunk_without_choices(self):
        with patch("simplemem.core.utils.llm_client.OpenAI") as sdk:
            sdk.return_value.chat.completions.create.return_value = iter([
                NS(model="deepseek-flash", id="stream", usage=None,
                   choices=[NS(delta=NS(content="ok"), finish_reason="stop")]),
                NS(model="deepseek-flash", id="stream", choices=[],
                   usage=NS(model_dump=lambda: {"prompt_tokens": 5})),
            ])
            self.assertEqual(LLMClient(use_streaming=True).chat_completion([]), "ok")
            event = json.loads(Path(os.environ["SIMPLEMEM_API_LOG"]).read_text().splitlines()[0])
            self.assertEqual(event["usage"]["prompt_tokens"], 5)
            self.assertTrue(sdk.return_value.chat.completions.create.call_args.kwargs["stream_options"]["include_usage"])

    def test_truncated_chat_is_rejected_without_retry(self):
        with patch("simplemem.core.utils.llm_client.OpenAI") as sdk:
            request = sdk.return_value.chat.completions.create
            request.return_value = NS(model="deepseek-flash", id="test", usage=None,
                                     choices=[NS(message=NS(content="partial"), finish_reason="length")])
            with self.assertRaises(ValueError):
                LLMClient().chat_completion([])
            self.assertEqual(request.call_count, 1)


if __name__ == "__main__":
    unittest.main()
