# Voice RAG over MSMARCO-XI

Speak a question in Hindi, Bengali, Tamil or English. The system transcribes it,
retrieves from a multi-strategy chunk index built over
[`ai4bharat/MSMARCO-XI`](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI),
verifies the answer is actually supported by the retrieved passages, and answers
— or declines and says why.

```
voice ──► Sarvam Saaras ──► input rails ──► hybrid retrieval ──► answer ──► grounding check ──► response
          (STT)             safety /        dense + BM25          extractive   abstain if        + citations
                            injection       + learned rerank      or Claude    unsupported       + per-stage timings
```

Every response carries its own per-stage latency breakdown and the passage it
drew from, with the quoted span highlighted in its source context.

---

## Quick start

```bash
uv sync
cp .env.example .env            # add SARVAM_API_KEY for voice input

uv run python -m vrag.download           # fetch MSMARCO-XI shards (~1.4GB)
uv run python -m vrag.build_index        # chunk, embed, index, fit reranker
uv run python scripts/smoke.py           # end-to-end sanity check
uv run python -m vrag.bench --n 400      # P50 / P70 / P100
uv run uvicorn vrag.server:app --reload  # http://localhost:8000
```

Runs fully offline apart from speech-to-text. Without `SARVAM_API_KEY` the text
box works and everything downstream is unchanged.

---

## The dataset

MSMARCO-XI is MS MARCO translated into 14 Indic languages. Each row is a query
with ten candidate passages in both English and the target language, plus
`is_selected` relevance labels and a gold answer.

Two consequences shaped the whole build:

- **The relevance labels are free ground truth.** Retrieval quality (Recall@k,
  MRR) and the reranker's training signal both come from `is_selected` rather
  than from vibes.
- **The gold passage is often in a *different language* than the query.** A
  Tamil question can have its answer only in an English passage, so retrieval
  has to work cross-lingually rather than falling back to same-language matching.

The HF dataset viewer is broken for this repo (one ~1.2GB parquet row group
exceeds their limit), so `vrag.download` fetches the raw shards directly and
`vrag.corpus` streams record batches rather than materialising a shard.

---

## Chunking — seven strategies, not seven tunings

The requirement was a chunking approach with real thought behind it. Each
strategy here has a *different failure mode*, which is the only reason to run
more than one:

| strategy | unit | idea |
|---|---|---|
| `fixed` | token window + overlap | the baseline; cuts mid-sentence by design |
| `recursive` | structural descent | paragraph → sentence → clause → word; only cuts mid-sentence as a last resort |
| `sentence_window` | 1 sentence, ±1 context | match narrow, read wide — a sharp vector, but neighbours for the reader |
| `parent_child` | ~45-token child, 200-token parent | siblings resolve to one identical context, so duplicates collapse |
| `metadata_aware` | routed by `query_type` | NUMERIC gets 60 tokens (a figure must not be buried), DESCRIPTION gets 160 |
| `proposition` | atomic clause | highest-precision unit; undersized fragments merge so a clause never loses its subject |
| `semantic` | embedding-boundary | cuts where meaning shifts, at a percentile of *this passage's own* distances |

Segmentation is script-aware: Devanagari/Bengali danda (`।`, `॥`), Urdu (`؟`,
`۔`), not just the Latin full stop.

**Deduplication turns overlap into signal.** Seven strategies over one passage
produce near-identical spans; indexing all of them would inflate the index and
let one passage occupy every slot in the top-k. The registry deduplicates on
normalised content and records *every* strategy that produced each survivor —
text that several independent strategies agree is a coherent unit is a better
retrieval target, exposed to the reranker as a `provenance` feature.

Measured over 17,917 passages (Hindi + Bengali + Tamil + their English sources):

| strategy | produced | unique | duplicate |
|---|---:|---:|---:|
| `proposition` | 59,776 | 14,942 | 75.0% |
| `sentence_window` | 58,085 | 49,648 | 14.5% |
| `parent_child` | 51,628 | 46,749 | 9.5% |
| `metadata_aware` | 30,038 | 11,489 | 61.8% |
| `semantic` | 29,608 | 10,067 | 66.0% |
| `fixed` | 25,495 | 25,150 | 1.4% |
| `recursive` | 20,975 | 8,542 | 59.3% |
| **total** | **275,605** | **166,587** | **39.6%** |

**Read that table carefully — "unique" is order-dependent.** Strategies run in
list order and whoever gets to a shared span first claims it, so `fixed`
scoring 1.4% duplicate means only that it ran first, not that it is the
strongest strategy. The meaningful figures are the union (166,587 chunks, 9.3
per passage) and the 39.6% collapse rate, which is what the provenance feature
is built from.

Chunking runs at ~110 passages/s including the semantic strategy's embedding
pass. That is offline and amortised; only the query-time re-chunk is inside the
latency budget.

---

## Retrieval

Three stages, each measured against the others in `vrag.eval_retrieval`:

1. **Recall** — dense ANN (usearch HNSW, int8) and BM25 run independently. They
   fail on *different* queries: dense handles paraphrase and cross-script
   matching, BM25 nails the rare entities and numbers that pooled vectors wash out.
2. **Fusion** — Reciprocal Rank Fusion. RRF combines by **rank, not score**,
   which matters here specifically: dense cosine magnitudes are not comparable
   across language pairs, so any score-level blend would silently down-weight
   every cross-lingual hit.
3. **Rerank** — a 7-feature linear model fitted by pairwise logistic regression
   on the gold labels, with score features z-scored *within the candidate set*
   so one weight vector is valid across every language.

The reranker's features are deliberately limited to things computable from the
query and chunk text alone. `is_selected` **is** the retrieval label; using it
as a ranking feature would leak the answer into the retriever and make every
number in this README fiction.

### The encoder decision

The first build used static embeddings (model2vec) for speed — ~0.03ms per
query. On full-corpus retrieval they collapsed: **R@1 = 0.045**, and a depth-500
probe found the gold chunk for only **39%** of queries. The ceiling was the
encoder, not the ranking on top of it — a static model averages subword vectors,
so a Tamil transliteration of "CHIPSA" shares nothing with the Latin string and
short acronym queries are unmatchable in principle.

`multilingual-e5-small`, int8-quantised to ONNX, costs **1.3ms** per query and
fixes it. Measured head-to-head on 155 queries per language:

| | static P@1 | e5 P@1 | static MRR | e5 MRR |
|---|---|---|---|---|
| Hindi | 0.245 | **0.394** | 0.467 | **0.576** |
| Tamil | 0.232 | **0.239** | 0.441 | **0.451** |

Tamil barely benefits — e5's Tamil coverage is genuinely weaker. That is
reported per language rather than averaged away, and it is why the abstention
rail matters: on Tamil the system declines more often, which is the correct
behaviour when retrieval is weak.

---

## The harness

`vrag.harness` is what makes this a pipeline rather than five chained calls:

- **Per-stage latency budgets.** Exceeding one is a recorded outcome, not
  something discovered with a stopwatch around the whole request.
- **Per-stage degradation.** STT fails over between providers and then to text
  input; the LLM answerer falls back to the extractive one; a guardrail that
  *errors* fails closed. A request loses capability instead of dying.
- **Typed failure.** Every outcome is a `Decision`, so "refused as unsafe",
  "abstained as ungrounded" and "crashed" stay distinguishable — collapsing them
  into a 500 would make the guardrail numbers meaningless.
- **Retries keyed on error type.** Retrying a bad API key is pointless; retrying
  malformed audio triples the latency of a request that was always going to fail.
  Only genuinely transient errors get another attempt.
- **Circuit breakers per dependency.** When a provider goes down, the breaker
  opens and the pipeline degrades in milliseconds instead of spending the full
  timeout on every request.
- **A tool registry.** Stages are named, described, individually invocable, and
  every call is timed — the same instrumentation the benchmark reads, so the
  reported numbers cannot drift from production behaviour.

---

## Guardrails

**Input rails** (before retrieval — a refusal should never pay for a search):
structural validation, a narrow unsafe-content list, prompt-injection patterns,
and PII flagging. Injection is *refused*, not stripped: sanitising and
proceeding means guessing which part of the input was the real question.

Unicode normalisation is load-bearing. NFKC folding closes the fullwidth bypass
(`ｉｇｎｏｒｅ　ｐｒｅｖｉｏｕｓ`), and invisible characters are handled in two classes —
zero-width space and bidi overrides become a *space* (deleting them would weld
words together and match nothing), while ZWJ/ZWNJ are **preserved**, because
they are linguistically load-bearing in Devanagari, Bengali and Tamil. They are
stripped only between two ASCII letters, where they can only be evasion.

**Output rails** (after generation): two distinct failure modes, checked
separately.

- *No evidence.* Detected from the **shape** of the score distribution, never an
  absolute threshold — cosine magnitudes differ systematically by language, so a
  constant floor tuned on Hindi rejects valid Tamil traffic. What generalises is
  the margin between the top hit and the body of the candidate set.
- *Ungrounded answer.* IDF-weighted containment of the answer's content tokens
  in the passages it cites. Weighting is the whole point: unweighted containment
  is ~1.0 for any fluent sentence because function words dominate, and the rare
  tokens — names, numbers, technical terms — are exactly what a hallucination
  invents.

The extractive path is grounded by construction (the answer is a substring of a
retrieved passage). The check still runs, so the LLM path is held to the same
bar rather than trusted because it sounds fluent.

---

## Answer generation, and the 200ms question

The stated target is under 200ms for "chunking + vector DB retrieval +
everything through to final output". Two things in that sentence deserve a
straight answer rather than a favourable reading:

**An LLM call cannot fit in 200ms.** A single API round trip is ~1-2s. So the
default path is **extractive**: it selects the best-supporting span from the
retrieved evidence using already-computed vectors. It is grounded by
construction, sub-millisecond, and abstains when nothing scores well. The LLM
path exists behind the same contract for fluency, and is benchmarked and
reported **separately** rather than blended into one headline number.

**Speech-to-text is a network call.** Sarvam takes ~650ms for a short clip, and
no third-party ASR fits in 200ms. So the benchmark reports two totals:
`pipeline_ms` (chunking, retrieval, guardrails, generation — the locally
computed path the budget can meaningfully constrain) and `total_ms` (adds the
STT round trip). Both are in the table below.

Query-time chunking is included in `pipeline_ms`: retrieved context is
re-split into sentence units before span selection, because quoting a whole
parent block back at the user is not an answer. Corpus chunking is offline and
amortised, and reported as build throughput instead.

---

## Results

> Numbers below are produced by `uv run python -m vrag.bench --n 400` on
> held-out queries the reranker never saw, and written to
> `bench/results/latest.json`. Warm-up runs are excluded.

*(populated from the benchmark run — see `bench/results/latest.json`)*

---

## Layout

```
src/vrag/
  corpus.py            MSMARCO-XI streaming ingestion + gold labels
  chunking/            base · strategies (6) · semantic · registry (dedup + provenance)
  embed.py             int8 ONNX e5, length-sorted batching
  index/               dense (HNSW) · lexical (BM25) · hybrid (RRF + rerank) · store
  guardrails/          input_rails · output_rails
  harness/             contracts · policy (retry/breaker) · tools · orchestrator
  answer/              extractive (fast path) · llm (Claude, same contract)
  build_index.py       offline build
  bench.py             P50/P70/P100 + retrieval quality + guardrail decisions
  eval_retrieval.py    ablation: dense vs BM25 vs fusion vs rerank
  server.py            FastAPI
web/index.html         mic UI with live per-stage timings and citation highlighting
tests/                 chunk offsets, guardrail bypasses, retry/breaker semantics
deploy/                Hugging Face Space packaging
```

## License

MIT.
