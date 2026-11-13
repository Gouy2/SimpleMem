"""OpenRouter Qwen embeddings with strict validation and a persistent cache."""

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path

import numpy as np
from openai import OpenAI

from simplemem.core.settings import settings as config
from simplemem.core.utils.api_metrics import error_metadata, record_api_event


class APIEmbeddingModel:
    def __init__(self, model_name=None):
        self.model_name = model_name or config.EMBEDDING_MODEL
        if not self.model_name.startswith("qwen/"):
            raise ValueError("This experiment requires an explicit qwen/ embedding model")
        self.dimension = int(config.EMBEDDING_DIMENSION)
        self.model_type = "openrouter_api"
        self.supports_query_prompt = True
        self.base_url = config.EMBEDDING_BASE_URL
        self.query_instruction = config.EMBEDDING_QUERY_INSTRUCTION
        if not config.OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY is missing")
        self.client = OpenAI(
            api_key=config.OPENROUTER_API_KEY, base_url=self.base_url,
            timeout=config.API_TIMEOUT, max_retries=0,
        )
        self.cache_path = Path(config.EMBEDDING_CACHE_PATH)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with sqlite3.connect(self.cache_path) as db:
            db.execute("CREATE TABLE IF NOT EXISTS embeddings (key TEXT PRIMARY KEY, vector BLOB NOT NULL)")

    def _key(self, text):
        payload = [self.base_url, self.model_name, self.dimension, text, "float32-l2-v1"]
        return hashlib.sha256(json.dumps(payload).encode()).hexdigest()

    def encode(self, texts, is_query=False):
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Embedding inputs must be nonempty strings")
        inputs = [f"Instruct: {self.query_instruction}\nQuery: {text}" if is_query else text
                  for text in texts]
        keys = [self._key(text) for text in inputs]
        vectors = {}
        with self._lock, sqlite3.connect(self.cache_path) as db:
            for key in set(keys):
                row = db.execute("SELECT vector FROM embeddings WHERE key=?", (key,)).fetchone()
                if row:
                    vector = np.frombuffer(row[0], dtype=np.float32).copy()
                    if vector.shape != (self.dimension,) or not np.isfinite(vector).all():
                        raise ValueError("Invalid cached embedding")
                    vectors[key] = vector
        # Preserve first occurrence order and avoid duplicate billable inputs.
        missing = list(dict.fromkeys(key for key in keys if key not in vectors))
        by_key = dict(zip(keys, inputs))
        batch_size = int(config.EMBEDDING_BATCH_SIZE)
        if batch_size < 1:
            raise ValueError("EMBEDDING_BATCH_SIZE must be positive")
        hits = len(keys) - sum(key not in vectors for key in keys)
        if hits:
            record_api_event(kind="embedding_cache", model=self.model_name, cache_hits=hits)
        for offset in range(0, len(missing), batch_size):
            batch_keys = missing[offset:offset + batch_size]
            batch = [by_key[key] for key in batch_keys]
            response = self._request(batch)
            ordered = sorted(response.data, key=lambda item: item.index)
            if [item.index for item in ordered] != list(range(len(batch))):
                raise ValueError("Embedding response has missing or duplicate indices")
            array = np.asarray([item.embedding for item in ordered], dtype=np.float32)
            if array.shape != (len(batch), self.dimension) or not np.isfinite(array).all():
                raise ValueError(f"Expected embeddings {(len(batch), self.dimension)}, got {array.shape}")
            norms = np.linalg.norm(array, axis=1, keepdims=True)
            if np.any(norms == 0):
                raise ValueError("Embedding response contains zero vectors")
            array /= norms
            with self._lock, sqlite3.connect(self.cache_path) as db:
                for key, vector in zip(batch_keys, array):
                    vectors[key] = vector
                    db.execute("INSERT OR REPLACE INTO embeddings VALUES (?, ?)", (key, vector.tobytes()))
        return np.stack([vectors[key] for key in keys])

    def _request(self, batch):
        for attempt in range(1, 4):
            started = time.perf_counter()
            try:
                response = self.client.embeddings.create(
                    model=self.model_name, input=batch, dimensions=self.dimension,
                    encoding_format="float",
                )
                record_api_event(
                    kind="embedding", status="ok", model=self.model_name,
                    response_model=response.model, request_id=response.id if hasattr(response, "id") else None,
                    inputs=len(batch), dimensions=self.dimension, attempt=attempt,
                    elapsed_seconds=time.perf_counter() - started,
                    usage=response.usage.model_dump() if response.usage else None,
                )
                return response
            except Exception as error:
                record_api_event(kind="embedding", status="error", model=self.model_name,
                                 attempt=attempt, elapsed_seconds=time.perf_counter() - started,
                                 **error_metadata(error))
                status = getattr(error, "status_code", None)
                if attempt == 3 or (status is not None and status != 429 and status < 500):
                    raise
                time.sleep(2 ** (attempt - 1))

    def encode_single(self, text, is_query=False):
        return self.encode([text], is_query=is_query)[0]

    def encode_query(self, queries):
        return self.encode(queries, is_query=True)

    def encode_documents(self, documents):
        return self.encode(documents, is_query=False)
