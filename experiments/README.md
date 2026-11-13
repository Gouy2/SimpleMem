# API-backed text-memory reproduction

This experiment uses the author's text-memory pipeline with DeepSeek chat and
Qwen embeddings through OpenRouter. The planned main ablation only disables
retrieval reflection. Both conditions will share the same API/model adaptations.
These adaptations differ from the original paper's setup.

## Install (Python 3.12)

Run commands from the repository root:

```bash
uv venv --python python3.12 .venv
uv pip sync --python .venv/bin/python -r requirements-api.txt
cp experiments/.env.example .env
```

Fill `DEEPSEEK_API_KEY` and `OPENROUTER_API_KEY` in `.env`. Keep the file local.
The `openai` package is the compatible API client, not the model provider.
This minimal installation intentionally omits local model inference and the
upstream script's BERTScore/SentenceTransformers metrics. It is not an installation
of all optional SimpleMem features.

## Stage 1: verify API adapters

```bash
.venv/bin/python -m unittest experiments.test_api_adapters -v
.venv/bin/python -m experiments.preflight --require-keys
.venv/bin/python -m experiments.api_smoke
```

The first two commands are offline. The last makes small billable requests using
two synthetic facts; it verifies API connectivity, embedding shape/normalization,
semantic retrieval, persistent cache use and a structured chat response.
It does **not** reproduce a LoCoMo benchmark result or validate memory construction,
retrieval planning or reflection. Those stages require a separate dataset run.

The successful API check is saved in `experiments/results/api_smoke.json`.
Its token usage and timings describe this small check only. A repeated run can
reuse embeddings and therefore issue fewer billable requests.

## Configuration and measurement

- Load `experiments.bootstrap.load_environment()` before importing the pipeline.
  It reads the repository `.env` and maps DeepSeek variables to the upstream
  `OPENAI_API_KEY`, `OPENAI_BASE_URL` and `LLM_MODEL` settings.
- Do not create top-level `config.py`; it overrides environment settings.
- `EMBEDDING_BACKEND=openrouter` enables the API adapter; the upstream local
  embedding path remains the default outside the experiment bootstrap.
- The experiment uses `qwen/qwen3-embedding-8b`, requesting 1024 dimensions.
  Responses are reordered by index and L2-normalized. Missing/duplicate indices,
  unexpected dimensions, zero vectors and nonfinite values fail validation.
  Query strings use Qwen's documented `Instruct: ...\nQuery: ...` format; documents
  are embedded as-is. Both ablation conditions use this same convention.
- Cache identity includes endpoint, model, requested dimension, actual input and
  normalization version. Embeddings persist in `.cache/embeddings.sqlite3`.
  No alternative embedding model is silently selected.
- DeepSeek's thinking parameter is explicitly sent as disabled for this experiment.
  SDK-internal retries are disabled; application retries are bounded and recorded.
  Default request timeout is 60 seconds and maximum output is 4096 tokens.
  Empty/truncated chat responses fail rather than being scored as valid answers.
- Set `SIMPLEMEM_API_LOG` to enable JSONL request metadata. Logs contain model IDs,
  usage, latency, completion status and safe error metadata, without prompts,
  API keys, chat content or reasoning content. Raw run logs live in ignored `.runs/`.
- Usage from OpenRouter includes reported cost when supplied. DeepSeek usage is
  recorded as tokens/cache hits; no unverified dollar estimate is fabricated.

References: [DeepSeek thinking mode](https://api-docs.deepseek.com/guides/thinking_mode/),
[OpenRouter embeddings](https://openrouter.ai/docs/api/api-reference/embeddings/submit-an-embedding-request),
[Qwen embedding model card](https://huggingface.co/Qwen/Qwen3-Embedding-8B).

## Remaining work

Freeze a LoCoMo subset, finish the lightweight evaluator, verify the full memory
pipeline, and run reflection-on/off comparisons. No formal benchmark results are
available yet.
