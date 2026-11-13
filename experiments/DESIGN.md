# LoCoMo subset reproduction and reflection ablation

## Target and comparison boundary

Reproduce the released SimpleMem text pipeline's ability to answer questions from
long conversation memory. Measure F1 on a fixed official LoCoMo subset, then
disable only retrieval reflection and quantify the quality/resource difference.

The [original SimpleMem paper, v1](https://arxiv.org/html/2601.02553v1), Table 1,
reports GPT-4.1-mini F1 of 43.46 (multi-hop), 58.62 (temporal), 19.76 (open domain)
and 51.12 (single hop), with Average F1 43.24. These are reference figures, not
matched-condition predictions for our subset. Our DeepSeek chat model, API Qwen
embedding, two-conversation subset and current released code differ.

The paper's Table 6 specifies 10-turn windows/5-turn stride, while this code's
defaults are 40-turn windows/2-turn overlap. We retain the code defaults and
report this discrepancy. Disabling the repository's reflection loop is our own
ablation; it is not identical to the paper's “w/o Adaptive Pruning” experiment.

## Data and leakage prevention

- Official source: [snap-research/locomo](https://github.com/snap-research/locomo),
  pinned at `3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376`.
- The data manifest stores URL, commit and SHA-256. Raw data lives in ignored
  `.cache/data/locomo10.json`; `prepare_locomo` downloads/checks it reproducibly.
- Use the first two official conversations (`conv-26`, `conv-30`) in full, totaling
  788 turns. Six questions per category (1-4) are sampled with local RNG seed 5660.
  Category 3 questions in this subset all come from `conv-26`, since `conv-30`
  contains none. This limits generalizability.
- The pilot question `conv-26` QA index 0 is excluded before formal selection.
  Category 5 is excluded because the upstream evaluator already disables
  reflection for those adversarial questions.
- Only raw conversation text, released image captions and session timestamps
  enter memory construction, following the upstream text conversion. Answers,
  evidence labels, observations and supplied summaries do not enter prompts.
- `configs/design.json` freezes question indices and experimental settings before
  formal runs. No changing questions in response to their scores.

## Metric

The upstream `test_locomo10.py` uses sets of normalized tokens, not token-count
multisets. Our lightweight F1 exactly preserves that implementation: lowercase;
replace `. , ! ?` with spaces; split; deduplicate; compute overlap precision/recall
and harmonic mean. Empty prediction/reference scores zero. Exact match is also
recorded. A regression test compares against the original functions loaded via
AST, without loading unrelated local scoring models.

Report category means and their macro average, plus question mean. The balanced
subset makes the two overall averages equal when all scheduled rows are retained.
For three trials, report sample variance and standard deviation over trial scores,
and paired baseline/ablation differences. Variation among questions is distinct
from variation among trials. API failures are retained as zero-scoring scheduled
rows and counted separately; this convention will be explicit in the report.

## Protocol and resource measurement

- Three trials, each with an independent memory build for each full conversation.
  Within a trial/conversation, both conditions use the exact same read-only store.
- Both conditions keep planning enabled and share the same prompts, top-k,
  embedding instruction and models. Only `enable_reflection` changes.
- Reverse condition order in the middle trial. Use two question workers, four
  memory-build workers and up to four retrieval workers.
- Both conditions use explicit non-thinking/non-streaming chat requests,
  16384 maximum output tokens and 120-second timeout. The larger allowance is a
  shared extraction safeguard, not the main ablation. Actual decoding temperatures
  remain the upstream per-stage values and are logged on requests.
- The dataset sampling seed does not imply server-side LLM determinism. The API
  supplies no controlled model seed in this experiment.
- Save original answers, scores, context IDs, timings, API usage, errors, code and
  data versions. Preserve a memory digest before/after QA to verify no mutation.
- Separate construction costs from QA costs. Embedding cache is shared/warm;
  report actual cache hits and network usage. Do not describe timings as cold-cache
  or sum concurrent request durations as wall-clock runtime.
- No dollar estimate for DeepSeek is invented from tokens; OpenRouter's returned
  embedding costs are recorded when available.

## Pre-run measurement audit

The completed pilot had no invalid responses and executed conditions serially.
Before formal concurrency, logging was hardened so an invalid chat response counts
as one request, and task metadata propagates through nested retrieval/build worker
pools. Formal rows must be attributed by trial/sample/question/condition tags,
never by slicing a shared log by line number. Run each condition as a separate
batch if sharing one mutable system instance. Record upstream fallback paths and
failed windows explicitly; a function returning normally is not sufficient proof
of successful retrieval. The frozen question list and treatment are unchanged.

The paper's token-cost accounting has not been established as equivalent to our
sum over all planning/reflection/answer calls. Do not divide those numbers to claim
a reproduced efficiency multiplier. Likewise, more or fewer memory records do not
by themselves establish lossless compression or a token compression ratio.

## Development pilot (not a formal trial)

The complete `conv-26` conversation (419 turns, 19 sessions) produced 439 memories
in 126.77 seconds. QA index 0 was answered correctly with reflection enabled and
disabled (F1=1.0 for both). Instrumentation verified the reflection method ran only
when enabled, and QA did not modify memory. One question cannot establish a
quality or latency benefit. See `results/pipeline_smoke.json` for evidence.
