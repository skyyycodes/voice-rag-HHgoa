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
| `fixed` | 40-token window, 12 overlap | the baseline; cuts mid-sentence by design |
| `recursive` | structural descent, ≤45 tokens | paragraph → sentence → clause → word; only cuts mid-sentence as a last resort |
| `sentence_window` | 1 sentence, ±1 context | match narrow, read wide — a sharp vector, but neighbours for the reader |
| `parent_child` | 18-token child, 60-token parent | siblings resolve to one identical context, so duplicates collapse |
| `metadata_aware` | routed by `query_type` | NUMERIC gets 22 tokens (a figure must not be buried), DESCRIPTION gets 55 |
| `proposition` | atomic clause | highest-precision unit; undersized fragments merge so a clause never loses its subject |
| `semantic` | embedding-boundary, ≤60 tokens | cuts where meaning shifts, at a percentile of *this passage's own* distances |

Segmentation is script-aware: Devanagari/Bengali danda (`।`, `॥`), Urdu (`؟`,
`۔`), not just the Latin full stop.

### Those sizes are measured, not conventional

Chunk-size defaults in the wild assume document-scale text — 200–500 tokens.
This corpus is not documents. Measured over it, MS MARCO passages run
**48 tokens at the median, 116 at p99, 3 sentences typical**.

The first build used conventional ceilings (120–200 tokens), which meant
**99–100% of passages fit whole under every one of them**. Four of the seven
strategies were therefore returning the same thing — the entire passage — and
collapsing into each other at dedup. `recursive` bottomed out at **96.9%
duplicate, contributing 535 unique chunks out of 17k passages**. "Seven
strategies" was true on paper and false in the index.

Retuning every ceiling to the measured distribution (roughly 3× smaller) is
what makes each strategy actually cut where its own rule says it should:

| strategy | unique before | unique after |
|---|---:|---:|
| `recursive` | 535 | **13,550** |
| `parent_child` | 19,132 | **34,182** |
| `fixed` | 18,412 | 23,703 |

A related trap worth naming: `default_chunkers()` passed sizes as constructor
arguments, silently shadowing the tuned dataclass defaults. The retune had no
effect on three strategies until those literals were removed — the real
configuration lived in two places and the wrong one won.

**Deduplication turns overlap into signal.** Seven strategies over one passage
produce near-identical spans; indexing all of them would inflate the index and
let one passage occupy every slot in the top-k. The registry deduplicates on
normalised content and records *every* strategy that produced each survivor —
text that several independent strategies agree is a coherent unit is a better
retrieval target, exposed to the reranker as a `provenance` feature.

Measured over the shipped index — 11,960 passages (Hindi + Bengali + Tamil and
their English sources):

| strategy | produced | unique | duplicate |
|---|---:|---:|---:|
| `parent_child` | 35,864 | 34,182 | 4.7% |
| `sentence_window` | 32,770 | 27,876 | 14.9% |
| `fixed` | 24,103 | 23,703 | 1.7% |
| `recursive` | 17,669 | 13,550 | 23.3% |
| `metadata_aware` | 24,022 | 11,645 | 51.5% |
| `proposition` | 38,456 | 9,928 | 74.2% |
| `semantic` | 21,291 | 6,488 | 69.5% |
| **total** | **194,175** | **127,372** | **34.4%** |

**Read that table carefully — "unique" is order-dependent.** Strategies run in
list order and whoever reaches a shared span first claims it, so `fixed`
scoring 1.7% duplicate means only that it ran early, not that it is the
strongest strategy. The meaningful figures are the union (127,372 chunks, 10.6
per passage) and the 34.4% collapse rate, which is what the provenance feature
is built from. `proposition` and `semantic` show high duplicate rates because
their units genuinely coincide with sentences much of the time — that is the
agreement signal, not waste.

Chunking runs at ~90 passages/s including the semantic strategy's embedding
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

### The bug that silently broke every non-English language

Python's `\w` matches Unicode *letters* but not combining *marks*. Indic
scripts write vowels as marks attached to consonants, so a plain `\w+` breaks
at every matra:

```
tokenize("कॉर्पोरेशन क्या है?")  ->  ['क','र','प','र','शन','क','य','ह']
tokenize("சிப்சா என்றால் என்ன")   ->  ['ச','ப','ச','என','ற','ல','என','ன']
```

Words were being shredded into consonant fragments. Nothing errored. The
damage ran through everything lexical — BM25, corpus IDF, the reranker's
`coverage` and `phrase` features, the grounding check, extractive span
scoring, and chunk sizing — for **three of the four languages in the corpus**.

It surfaced only because a Hindi question that the corpus demonstrably covers
came back as `abstain_no_evidence`. Symptoms that had looked like separate
problems turned out to be this one bug:

- the reranker fitting `coverage = -1.398` (penalising chunks containing the
  query's rare terms) and scoring **0.501 held-out — exactly chance**;
- Indic chunks sized by fragment counts, so every ceiling tripped ~3× early;
- the extractive answerer unable to score any Indic evidence.

The fix is a character class that includes the Indic blocks and Latin
diacritics. Regression tests now cover all three scripts, and the reranker
went from 0.501 to 0.561 held-out on the next build.

### Answering in the language the question was asked in

MSMARCO-XI translates every passage into every language, so near-identical
translations sit adjacent in embedding space and retrieval picks among them
arbitrarily. Measured: only **22%** of queries got a same-language top hit.

This is a product decision, not something to learn — `is_selected` says nothing
about what language the reader speaks — so it is applied as an explicit
preference over ranked candidates (`script_detect.py`, Unicode-block counting,
microseconds) rather than as a reranker feature. The bonus is deliberately
small: a correct answer in the wrong language still beats a wrong answer in the
right one.

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

All numbers below come from `uv run python -m vrag.bench` and
`vrag.eval_retrieval` on **165 held-out queries the reranker never saw**,
against a **127,372-chunk** index (11,960 passages · Hindi, Bengali, Tamil, and
their English sources). Warm-up runs excluded. Raw JSON in
`bench/results/latest.json`.

### Latency — the 200ms target

```
P50    5.44 ms     P70   22.97 ms     P95   58.91 ms     P100  170.75 ms
within 200ms budget: 100.0% of queries      57.0 queries/sec, single process
```

| stage | P50 | P70 | P95 | P100 |
|---|---:|---:|---:|---:|
| retrieve | 4.75 | 5.08 | 5.67 | 6.12 |
| generate | 0.14 | 18.21 | 54.24 | 166.46 |
| guard_input | 0.03 | 0.03 | 0.04 | 0.19 |
| guard_evidence | 0.02 | 0.02 | 0.02 | 0.03 |
| guard_output | 0.03 | 0.04 | 0.05 | 0.08 |

Retrieval is flat and predictable. The entire tail lives in `generate`: when a
question's evidence is only available in another language, lexical span scoring
scores zero and the answerer falls back to encoding candidate sentences. That
path costs ~50-170ms and fires on roughly a third of queries.

**Speech-to-text is excluded from these figures and reported separately.**
Measured end-to-end on the live API with a spoken Hindi question:

```
POST /api/voice   (spoken: "कॉर्पोरेशन क्या है?")
  transcript  : कॉर्पोरेशन क्या है?     detected hi-IN via Sarvam
  answer      : एक निगम एक कंपनी या लोगों का समूह है ...   cited chunk: hin
  pipeline_ms : 23.3          total_ms: 660.9
  stages      : transcribe 637.5 | retrieve 22.3 | generate 0.6 | guards 0.5
```

Sarvam is 96% of wall-clock. No third-party ASR fits in 200ms, so folding it
into one headline number would hide which part of the system the budget
actually constrains.

### Retrieval quality

| configuration | R@1 | R@5 | MRR@5 | p50 |
|---|---:|---:|---:|---:|
| dense only (ANN) | 0.109 | 0.291 | 0.170 | 1.8 ms |
| BM25 only | 0.061 | 0.188 | 0.106 | 1.7 ms |
| hybrid + RRF fusion | 0.097 | 0.236 | 0.147 | 4.6 ms |
| hybrid + RRF + learned rerank | 0.103 | 0.297 | 0.176 | 4.6 ms |

Two things this table says that a summary would hide. **RRF fusion alone is
worse than dense alone** — adding BM25 by rank costs recall on this corpus, and
only the rerank stage recovers it. And the full stack beats dense-only by
**+3.8% MRR for +2.8ms**, which is a real but modest return on the whole hybrid
layer.

The reranker is fitted per build and **only shipped if it generalises**:
held-out pairwise accuracy must clear 0.55 or the build discards it and falls
back to RRF-only ordering. It scored 0.561 here and shipped. On the previous
build — before the Indic tokenisation fix — it scored **0.501, exactly chance**,
and would have been rejected.

### Answering in the asker's language

MSMARCO-XI translates every passage into every language, so near-identical
translations sit next to each other in embedding space and retrieval picks
among them arbitrarily. Measured, only **22%** of queries got a same-language
top hit — a Hindi speaker was routinely answered in Bengali.

A small explicit preference (applied after ranking, not learned — relevance
labels say nothing about what language the reader speaks) fixes it:

| | before | after |
|---|---:|---:|
| answered in the query's language | 22% | **56%** |
| fell back to English | 20% | 36% |
| answered in an unrelated third language | 58% | **8%** |

It also cut P50 latency **5.8x (31.5ms → 5.4ms)**: same-language evidence has
lexical overlap, so the expensive cross-lingual fallback stops firing.

### Guardrails

The smoke suite exercises every outcome; all six probes behave correctly —
in-domain English and Hindi answer with citations, unsafe/injection/too-short
are refused in under 0.05ms, and the personal-scope query is declined.

**The evidence-margin rail does not work, and the calibration measured it.**
Fitting thresholds against in-domain vs out-of-domain populations gives
**0.0% out-of-domain catch rate**: out-of-domain queries score a *higher*
median margin (2.957) than in-domain ones (1.979). The signal is inverted, not
merely weak. Reported as measured rather than tuned until it looked good.

What does work is the **personal-scope input rail** — "what did I have for
breakfast" retrieves passages about breakfast, so the topic matches and only
the scope is impossible; no similarity signal can catch that. Verified against
real traffic: **1 false rejection in 263 held-out MS MARCO queries**, while
correctly allowing "how much does my dog need to eat", "what is my credit
score", and "my heart rate is 120 what does that mean".

### The weakest number

**The system answers 98.8% of queries but cites a gold-labelled passage only
5.5% of the time.** Some of that gap is measurement — MS MARCO marks roughly
one passage in ten as gold, so a correct citation of an unlabelled passage
counts as a miss — but not all of it. R@5 is 0.297, meaning the gold chunk is
in the top five for 30% of queries, so the answerer is picking a non-gold
candidate more often than it should. It is confident far more often than it is
verifiably right, and the abstention rails are not catching the difference.

That is the first thing to fix, ahead of any further latency work.

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
