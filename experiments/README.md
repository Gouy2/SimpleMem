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
- `api_event_context` adds task IDs to request events; `submit_with_context`
  propagates them through nested thread pools. Use tags to attribute concurrent
  costs. Invalid chat responses are counted once, retaining their billable usage.
- Usage from OpenRouter includes reported cost when supplied. DeepSeek usage is
  recorded as tokens/cache hits; no unverified dollar estimate is fabricated.

References: [DeepSeek thinking mode](https://api-docs.deepseek.com/guides/thinking_mode/),
[OpenRouter embeddings](https://openrouter.ai/docs/api/api-reference/embeddings/submit-an-embedding-request),
[Qwen embedding model card](https://huggingface.co/Qwen/Qwen3-Embedding-8B).

## Stage 2: pinned data and full-conversation pilot

```bash
.venv/bin/python -m experiments.prepare_locomo
.venv/bin/python -m unittest experiments.test_api_adapters experiments.test_locomo_helpers -v
.venv/bin/python -m experiments.pipeline_smoke
.venv/bin/python -m experiments.freeze_design
```

Data preparation downloads only when needed and always validates the pinned
SHA-256. The complete data is cached locally, not committed. The pipeline pilot
builds memory from all 419 turns of the first official conversation, answers one
question under both reflection settings, verifies method traces and read-only
memory, and saves `results/pipeline_smoke.json`. It makes billable API calls and
can take several minutes; no need to rerun it merely to read the saved evidence.

`configs/design.json` fixes 24 questions (six per category), two full conversations
and three independent paired trials. The pilot question is excluded. See
[DESIGN.md](DESIGN.md) for the original-paper reference, scoring definition,
model/code differences, sampling and cost-measurement limitations.

## Stage 3: formal paired evaluation (macOS/Linux)

```bash
.venv/bin/python -m unittest experiments.test_api_adapters experiments.test_locomo_helpers experiments.test_formal_runner -v
.venv/bin/python -u -m experiments.run_formal
```

The runner prints a new `.runs/formal-<id>` directory. It makes billable requests
for the frozen design: three independent builds for each of two conversations,
and 24 questions under each reflection condition in each trial (144 QA rows).
It uses an advisory process lock and atomic per-build/per-question checkpoints.
The code/dependency content fingerprint, environment, effective settings, data
version and frozen selection are stored before any model calls.

To resume a particular run, keep its source and settings unchanged:

```bash
.venv/bin/python -u -m experiments.run_formal --run-dir .runs/formal-<id>
```

Completed rows, including failed ones, are not repeated. Incomplete memory builds
are preserved and stop automatic recovery; inspect them rather than silently
rebuilding or deleting data. Source/settings/environment mismatch also refuses
resume. Do not edit the experiment's Python code during a live run.

Offline re-export (no model calls):

```bash
.venv/bin/python -m experiments.run_formal --run-dir .runs/formal-<id> --summarize-only
```

Outputs in `experiments/results/`:

- `formal_manifest.json`: source hashes, runtime dependencies, frozen protocol.
- `formal_rows.csv` and `formal_rows.jsonl`: all scheduled QA identities, answers,
  references, scores, status, latency and measured usage.
- `formal_summary.json`: per-trial/per-category results, sample variance/std over
  three trials, paired ablation-minus-baseline differences, construction and
  condition-batch records, and recorded API usage.

Partial runs export `status=partial` without full aggregate claims. Missing or
duplicate identities are checked. Internal upstream fallback is instrumented
without changing fallback behavior: such rows get primary F1=0 but retain the
actual answer/raw score and fallback metadata. Recovered request retries retain
their actual answer score and all recorded attempts. Read-only memory and actual
reflection-method invocation are verified. All condition futures finish before
the shared system's reflection setting changes.

Detailed contexts, traces, API metadata and databases stay in ignored `.runs/`.
Use task tags for concurrent attribution; do not slice the global log by line
number. Batch wall time, per-question latency and summed request time are distinct.

## Remaining work

All 144 formal rows and six independent memory builds are complete and audited.
See [RESULTS.md](results/RESULTS.md) for tables, per-trial differences and limitations.
Actual resume of the completed run made zero additional API calls. The remaining
deliverables are the brief report and recorded presentation.
