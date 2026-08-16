"""Extractive answer synthesis — the sub-200ms default path.

Selects the best-supporting sentence span from the retrieved evidence rather
than generating text. Three properties follow from that, and they are the
reason this is the default rather than a fallback:

* It is grounded by construction. The answer is a substring of a retrieved
  passage, so it cannot state a fact the corpus does not contain. The output
  guardrail still runs, but it has nothing to catch.
* It is fast. No model call — scoring a handful of sentences is pure numpy over
  already-computed vectors, well under a millisecond.
* It is honest about coverage. When no sentence scores well the module returns
  nothing and the abstention rail fires, instead of smoothing over a bad
  retrieval with fluent prose.

The cost is register: answers read as quoted source text, not as conversational
replies. The LLM path exists for when that matters, behind the same contract.
"""

from __future__ import annotations

import numpy as np

from ..chunking.base import split_sentences
from ..config import Settings, settings
from ..harness.contracts import Answer, Citation
from ..index.hybrid import Candidate
from ..index.lexical import tokenize

# MS MARCO query types map to what a good answer span looks like.
_TERSE_TYPES = {"NUMERIC", "ENTITY", "PERSON", "LOCATION"}


def _span_score(
    q_tokens: set[str],
    q_idf: dict[str, float],
    sentence: str,
    position: int,
) -> float:
    """Score one candidate sentence against the query."""
    s_tokens = tokenize(sentence)
    if not s_tokens:
        return 0.0
    s_set = set(s_tokens)

    overlap = sum(q_idf.get(t, 1.0) for t in (q_tokens & s_set))
    total = sum(q_idf.get(t, 1.0) for t in q_tokens) or 1.0
    coverage = overlap / total

    # Prefer sentences that are informative but not sprawling. Very short
    # fragments rarely contain a full answer; very long ones bury it.
    length = len(s_tokens)
    length_fit = 1.0 if 8 <= length <= 45 else (0.75 if length < 8 else 0.85)

    # Leading sentences of a passage are more often definitional, which is what
    # DESCRIPTION queries want.
    position_prior = 1.0 - 0.06 * min(position, 4)

    return coverage * length_fit * position_prior


def _semantic_span_scores(query: str, sentences: list[str], cfg: Settings) -> np.ndarray:
    """Cosine similarity between the query and each candidate sentence.

    The lexical scorer cannot rank a passage written in a different language
    from the query — token overlap is exactly zero. That is not an edge case
    here: cross-lingual retrieval is the point of the multilingual encoder, so
    a Hindi question routinely retrieves the English passage that answers it,
    and a purely lexical answerer then finds no span and abstains on a question
    the corpus demonstrably covers.

    Encoding the handful of candidate sentences costs a few milliseconds
    against a 200ms budget, and it is the same encoder that retrieved them.

    The returned score is *rescaled*, not a raw cosine. e5 similarities are
    compressed into a narrow high band — two unrelated sentences still score
    ~0.78 — so treating a raw cosine as a relevance score means every query
    finds something and the abstention rail never fires. Mapping
    `[floor, 1] -> [0, 1]` and clipping below restores a usable zero: a span
    that is merely in the same language, not on the same topic, scores 0 and
    the pipeline abstains.
    """
    from ..embed import Kind, get_encoder

    encoder = get_encoder(cfg)
    q = encoder.encode([query], Kind.QUERY, 1)[0]
    s = encoder.encode(sentences, Kind.PASSAGE, cfg.embed_batch)
    cos = s @ q
    floor = cfg.semantic_span_floor
    return np.clip((cos - floor) / max(1e-6, 1.0 - floor), 0.0, 1.0)


def answer_extractive(
    query: str,
    candidates: list[Candidate],
    idf: dict[str, float] | None = None,
    query_type: str = "DESCRIPTION",
    cfg: Settings = settings,
    remaining_ms: float | None = None,
) -> Answer:
    """Pick the best supporting span across the retrieved candidates."""
    if not candidates:
        return Answer(text="", citations=[], mode="extractive")

    idf = idf or {}
    q_tokens = set(tokenize(query))
    q_idf = {t: idf.get(t, 1.0) for t in q_tokens}

    # Re-chunk every retrieved context at query time. Retrieval returns parent
    # blocks for the small-to-big strategies, and quoting a whole parent block
    # back at the user is not an answer.
    # (candidate, retrieval rank, sentence, position within its passage, start, end)
    spans: list[tuple[Candidate, int, str, int, int, int]] = []
    for rank, cand in enumerate(candidates):
        cursor = 0
        for position, sentence in enumerate(split_sentences(cand.context)):
            start = cand.context.find(sentence, cursor)
            if start < 0:
                start = cursor
            cursor = start + len(sentence)
            spans.append((cand, rank, sentence, position, start, start + len(sentence)))

    if not spans:
        return Answer(text="", citations=[], mode="extractive")

    lexical = np.array(
        [_span_score(q_tokens, q_idf, sentence, position)
         for _, _, sentence, position, _, _ in spans],
        dtype=np.float32,
    )

    # Fall back to semantics only when the lexical signal is absent — which in
    # practice means the evidence is in a different language from the question.
    # Always blending would cost the encode on every request and would let a
    # topically-similar sentence outrank one that literally contains the answer.
    # The cross-lingual fallback is the single most expensive thing in this
    # module (up to ~157ms at P100) and it is what pushed end-to-end latency
    # past the 200ms bar. It only runs when there is measurably enough budget
    # left, so the deadline is enforced rather than merely reported.
    affordable = remaining_ms is None or remaining_ms >= cfg.semantic_span_min_budget_ms
    semantic = None
    if float(lexical.max()) <= 0.0 and affordable:
        try:
            # Cap the work: encoding every sentence of every candidate is the
            # one part of this path that scales with retrieval depth, and the
            # answer is almost always in a top-ranked passage anyway.
            budget = cfg.semantic_span_limit
            scored = _semantic_span_scores(
                query, [sentence for _, _, sentence, _, _, _ in spans[:budget]], cfg
            )
            semantic = np.zeros(len(spans), dtype=np.float32)
            semantic[: len(scored)] = scored
        except Exception:
            # Answering lexically-unmatched evidence is a bonus, not a
            # guarantee; if the encoder is unavailable the abstention rail
            # fires, which is the correct conservative outcome.
            semantic = None

    best: tuple[float, Candidate, str, int, int] | None = None
    for i, (cand, rank, sentence, _position, start, end) in enumerate(spans):
        # Retrieval rank is strong evidence, and this weighting used to be far
        # too weak to express that. With a 1.0-1.15 multiplier the span score
        # dominated, and the answerer cited rank-0 through rank-4 at almost
        # uniform rates (35/29/31/28/41 over 165 queries) — it was effectively
        # ignoring the ranking that the entire retrieval stack exists to
        # produce. Geometric decay makes rank the primary term: a rank-4
        # sentence must be dramatically better in isolation to displace rank 0.
        #
        # Taken from enumerate rather than `candidates.index(cand)` — the latter
        # is O(n^2) and invokes the dataclass `__eq__`, which compares the numpy
        # feature vectors and raises on an ambiguous truth value.
        rank_weight = cfg.answer_rank_decay ** rank
        base = float(lexical[i]) if semantic is None else float(semantic[i])
        score = base * rank_weight
        if best is None or score > best[0]:
            best = (score, cand, sentence, start, end)

    if best is None or best[0] <= 0.0:
        return Answer(text="", citations=[], mode="extractive")

    _, cand, sentence, start, end = best

    # For terse query types, a neighbouring sentence often completes the
    # answer ("It was founded in 1897." needs the sentence naming the subject).
    text = sentence
    if query_type not in _TERSE_TYPES:
        text = _expand(cand.context, start, end, q_tokens, q_idf)

    citation = Citation(
        chunk_id=cand.idx,
        text=cand.text,
        context=cand.context,
        lang=cand.lang,
        strategy=cand.query_type,
        score=float(cand.score),
        span_start=start,
        span_end=end,
    )
    return Answer(text=text.strip(), citations=[citation], mode="extractive")


def _expand(
    context: str, start: int, end: int, q_tokens: set[str], q_idf: dict[str, float]
) -> str:
    """Append the following sentence when it adds query-relevant content.

    Guarded rather than unconditional: always returning two sentences doubles
    the answer length for no gain on questions the first sentence already
    answered, and dilutes the grounding score.
    """
    tail = context[end:].strip()
    if not tail:
        return context[start:end]
    following = split_sentences(tail)
    if not following:
        return context[start:end]

    extra = following[0]
    extra_tokens = set(tokenize(extra))
    gain = sum(q_idf.get(t, 1.0) for t in (q_tokens & extra_tokens))
    total = sum(q_idf.get(t, 1.0) for t in q_tokens) or 1.0
    if gain / total < 0.15 or len(extra) > 300:
        return context[start:end]
    return context[start:end] + " " + extra
