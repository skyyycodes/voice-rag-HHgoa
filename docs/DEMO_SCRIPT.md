# Demo script

Two videos. Both are short. The failure mode to avoid is narrating features —
show three things the viewer does not expect, and let the numbers on screen do
the talking.

---

## Video 2 — the product demo (~2 min)

Open with the thing working, not with an explanation.

### Beat 1 (0:00–0:25) — it works, in the user's own language

Hold the mic button and speak, in Hindi: **"कॉर्पोरेशन क्या है?"**

On screen, without cutting away:
- the transcript appears — *exactly* the words spoken
- the answer comes back **in Hindi**, not English
- the cited passage is highlighted in its source context
- the timing strip reads roughly `retrieve 117ms · generate 0.4ms · guards 0.1ms`

Say only: *"Spoken Hindi, answered in Hindi, from a 127,000-chunk index, in
about 120 milliseconds of retrieval — on free Hugging Face hardware, not my
laptop. Speech-to-text is the other two to three seconds, and the table breaks
it out so you can see which part is ours."*

Do not round the STT number down. It measured 2068ms, 3031ms and 3060ms across
three takes on the Space; it is Sarvam's network hop and it varies. Quote the
range. A number on screen that disagrees with the number you said costs you
every other number in the video.

Do not explain the architecture yet. Let it land.

### Beat 2 (0:25–1:00) — it knows when to refuse

Click through the probe buttons. Four in a row, no typing:

| click | what the viewer sees |
|---|---|
| Prompt injection | `REJECT_MALFORMED` in 0.0ms |
| Unsafe request | `REFUSE_UNSAFE` in 0.0ms |
| Personal / unknowable | `REJECT_OFF_TOPIC` — "contains no information about the person asking" |
| **Says "I" — answered** | **`ANSWER`** — "how much water should I drink a day" |
| **Says "my" — answered** | **`ANSWER`** — "what is my credit score based on" |

Those last two are what matter and what nobody else will show.
Say: *"A guardrail that refuses everything is easy. This one is calibrated:
first-person phrasing does not trigger it. The rail keys on whether the corpus
can answer, not on how the question is worded."*

Then, live and unscripted, **hold the mic and say "Hello, hello. What's up?
How are you?"** It comes back `REJECT_OFF_TOPIC · rails: chitchat` in **0.1ms**
— refused before retrieval runs at all.

Say: *"We only found this by deploying it. Before this rail, that greeting was
answered — with a grounding score of 1.00 — from a passage that happens to
begin 'Hello there, when user logs in to Citrix web Interface.' BM25 matched
the literal word 'hello' and the cross-encoder rated the overlap highly. No
score threshold fixes that, because the passage really does contain the word.
It needed a different layer: greetings are a class of input, so a query is
refused only when every token in it is conversational filler. That change
declines zero of our 165 held-out queries."*

### Beat 3 (1:00–1:35) — the numbers are checkable

Click **⇄ A/B the reranker**. The same query runs twice, live, with the
cross-encoder off and on. The table shows both latencies, both citations, and
whether the ranking actually changed.

Then click **🔍 Explain ranking**: every candidate with which retriever found
it, its dense and BM25 ranks, the cross-encoder's absolute relevance logit, and
which of the seven chunking strategies produced the text. Logits below the
abstention floor render in amber.

Say: *"Every claim in the README is a button in this UI. The ablation is not a
screenshot — it runs on whatever you type."*

### Beat 4 (1:35–2:00) — the budget is a scheduling input

Point at the **Rerank depth** tile while asking two questions of different
lengths. It moves.

Say: *"The 200ms target is not something we measured afterwards. The pipeline
reads the clock and decides how deep to rerank with the time it has left. On a
slower machine it reranks shallower and still lands inside the budget — it
times itself at startup rather than trusting a constant from my laptop."*

Close on the benchmark — 400 queries, `uv run python -m vrag.bench --n 400`:
**P50 95.7ms · P70 100.0ms · P95 102.6ms · P100 115.5ms · 100% within 200ms.**

And then the honest half: *"That benchmark is from my machine. On the free
Hugging Face box you are looking at right now, it runs 117 to 138 milliseconds.
Slower, still inside budget — because the reranker measures the hardware at
startup instead of trusting a constant I wrote down."*

---

## Video 1 — team / process (90s)

This one is about *how*, and the strongest material is the bugs, not the build.

### Beat 1 (0:00–0:35) — the bug most Indic RAG has right now

Run it on camera:

```bash
uv run python scripts/why_indic_rag_breaks.py
```

Show the output for Hindi, Bengali, Tamil:

```
कॉर्पोरेशन क्या है
  \w+  ->  8 tokens   ['क','र','प','र','शन','क','य','ह']
  fixed ->  3 tokens   ['कॉर्पोरेशन','क्या','है']
```

Say: *"Python's `\\w` matches Unicode letters but not combining marks. Indic
scripts write vowels as marks, so the standard tokenising idiom shreds every
word into consonant fragments. Nothing errors. It quietly broke BM25, IDF,
grounding and chunk sizing for three of our four languages — and we only found
it because one Hindi question the corpus definitely covers came back as
'no evidence'."*

Then the receipt: *"After the fix, our reranker went from 0.501 held-out
accuracy — exactly chance — to 0.561."*

### Beat 2 (0:35–1:05) — we measured instead of assuming

Three decisions, each one where the obvious answer was wrong:

- **Chunk sizes.** Conventional defaults are 200–500 tokens. Our passages
  measure 48 at the median, so 99% fit whole under every ceiling and four of
  seven strategies were returning identical output. `recursive` contributed
  535 unique chunks out of 17,000 passages. Retuned to the measured
  distribution: 13,550.
- **Deeper reranking.** Tried it. R@5 improved, R@1 got *worse*, latency
  doubled, budget broke. Rejected our own change and wrote why into the config.
- **The build refuses to ship a bad model.** If the reranker cannot clear 0.55
  held-out accuracy it is discarded for plain rank fusion, automatically.
- **Deploying found a bug testing never would.** Locally we always typed
  questions. On the Space we said hello — and got a confident, grounded answer
  about Citrix login errors, because that passage starts with "Hello there".
  Fixed as an input rail, not a threshold, and verified to decline 0 of 165
  held-out queries.

### Beat 3 (1:05–1:30) — what we are still honest about

*"R@5 is 0.30. Our out-of-domain rail catches 50%, not 90%. Both are in the
README with the measurement, because a system that reports only its wins isn't
one you can trust the rest of the numbers from."*

---

## Setup checklist

```bash
uv run uvicorn vrag.server:app --port 8000      # warm it first
open http://localhost:8000
```

Recording on the deployed Space instead (recommended — the live link is what
gets submitted): open it **five minutes early**. Free Spaces sleep, and waking
one re-downloads 236MB of ONNX before it serves anything.

- Fire one query before recording — the first request pays ONNX warm-up.
- Check `/api/health` shows the chunk count you expect.
- `SARVAM_API_KEY` must be set for Beat 1 of the demo video.
- Record at a window width that keeps the timing tiles on one row.
