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
    n_sentences: int,
) -> float:
    """Score one candidate sentence against the query."""
    s_tokens = tokenize(sentence)
    if not s_tokens:
        return 0.0
    s_set = set(s_tokens)

    overlap = sum(q_idf.get(t, 1.0) for t in q_set_intersect(q_tokens, s_set))
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


def q_set_intersect(a: set[str], b: set[str]) -> set[str]:
    return a & b


def answer_extractive(
    query: str,
    candidates: list[Candidate],
    idf: dict[str, float] | None = None,
    query_type: str = "DESCRIPTION",
    cfg: Settings = settings,
) -> Answer:
    """Pick the best supporting span across the retrieved candidates."""
    if not candidates:
        return Answer(text="", citations=[], mode="extractive")

    idf = idf or {}
    q_tokens = set(tokenize(query))
    q_idf = {t: idf.get(t, 1.0) for t in q_tokens}

    best: tuple[float, Candidate, str, int, int] | None = None
    for cand in candidates:
        # Re-chunk the retrieved context at query time. Retrieval returns
        # parent blocks for the small-to-big strategies, and quoting a whole
        # parent block back at the user is not an answer.
        sentences = split_sentences(cand.context)
        cursor = 0
        for position, sentence in enumerate(sentences):
            start = cand.context.find(sentence, cursor)
            if start < 0:
                start = cursor
            cursor = start + len(sentence)

            score = _span_score(q_tokens, q_idf, sentence, position, len(sentences))
            # Retrieval rank is evidence too: a great sentence inside a poorly
            # matched passage is usually a coincidence of shared vocabulary.
            score *= 1.0 + 0.15 * _rank_bonus(candidates, cand)
            if best is None or score > best[0]:
                best = (score, cand, sentence, start, start + len(sentence))

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


def _rank_bonus(candidates: list[Candidate], cand: Candidate) -> float:
    """1.0 for the top-ranked candidate, decaying to 0."""
    try:
        rank = candidates.index(cand)
    except ValueError:
        return 0.0
    return max(0.0, 1.0 - rank / max(1, len(candidates)))


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
