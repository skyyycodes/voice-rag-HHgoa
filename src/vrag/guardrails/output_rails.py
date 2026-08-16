"""Output guardrails: refusing to answer from weak or absent evidence.

Two distinct failure modes, checked separately because they need different
signals:

*No evidence.* The corpus simply does not cover the question. Detected from the
shape of the retrieval score distribution, never from an absolute threshold —
that was measured to be unworkable here. Cosine magnitudes differ systematically
by language pair, so a constant floor that abstains correctly on Tamil rejects
half of the valid Hindi traffic. What generalises is the *margin*: when the
corpus contains an answer, the top hit stands clear of the rest; when it does
not, retrieval returns a flat spread of equally-mediocre matches.

*Ungrounded answer.* Evidence was retrieved but the answer does not follow from
it. On the extractive path this is nearly impossible by construction, and that
is the point — the check still runs, so the LLM path is held to exactly the
same bar rather than being trusted because it sounds fluent.

Grounding is measured lexically (IDF-weighted content-token containment)
instead of by an NLI model. An NLI cross-encoder is ~50ms per pair and would
consume a quarter of the budget to verify a span that was copied verbatim from
the source. The lexical check is strict in the direction that matters: it
cannot be fooled by fluent text that cites nothing.
"""

from __future__ import annotations

import math

import numpy as np

from ..config import Settings, settings
from ..harness.contracts import Answer, Decision, GuardVerdict
from ..index.hybrid import Candidate
from ..index.lexical import tokenize


def evidence_verdict(
    candidates: list[Candidate], cfg: Settings = settings
) -> GuardVerdict:
    """Decide whether retrieval found anything worth answering from."""
    if not candidates:
        return GuardVerdict(
            allowed=False,
            decision=Decision.ABSTAIN_NO_EVIDENCE,
            reason="Nothing in the indexed corpus matches this question.",
            triggered=["empty_candidates"],
        )

    # Prefer the cross-encoder's raw logit when it ran. It is the only
    # absolute relevance signal available — everything else in the pipeline is
    # relative to the other candidates for the same query, which is exactly the
    # wrong frame for "is any of this relevant at all?".
    #
    # The margin heuristic below was measured at a 0.0% out-of-domain catch
    # rate, and worse, it is *inverted*: out-of-domain queries score a HIGHER
    # median margin (2.957) than in-domain ones (1.979), because retrieval
    # surfaces one weak match against noise while a covered question surfaces
    # several comparable ones. Relative spread cannot answer this question.
    logits = [c.rerank_logit for c in candidates if c.rerank_logit > float("-inf")]
    if logits:
        best_logit = max(logits)
        if best_logit < cfg.ood_rerank_floor:
            return GuardVerdict(
                allowed=False,
                decision=Decision.REJECT_OFF_TOPIC,
                reason=(
                    "Nothing in the indexed corpus is relevant enough to this "
                    "question to answer from."
                ),
                triggered=["low_rerank_relevance"],
                scores={"rerank_logit": best_logit},
            )

    scores = np.array([c.score for c in candidates], dtype=np.float32)
    top = float(scores[0])

    # Margin between the best hit and the body of the candidate set, in units
    # of the set's own spread. Scale-free, so it transfers across languages.
    if len(scores) >= 3:
        rest = scores[1:]
        spread = float(rest.std()) or 1e-6
        margin = (top - float(rest.mean())) / spread
    else:
        margin = float("inf")

    # Does the top hit share any rare vocabulary with the query at all? A hit
    # with high cosine but zero lexical overlap is usually a topical neighbour
    # rather than an answer.
    dense_top = float(candidates[0].dense_score)

    scores_out = {"top_score": top, "margin": margin, "dense_top": dense_top}

    if margin < cfg.ood_margin_floor and dense_top < cfg.ood_dense_floor:
        return GuardVerdict(
            allowed=False,
            decision=Decision.REJECT_OFF_TOPIC,
            reason=(
                "This question does not appear to be covered by the indexed "
                "corpus, so answering would mean guessing."
            ),
            triggered=["low_margin", "low_dense"],
            scores=scores_out,
        )

    return GuardVerdict(allowed=True, decision=Decision.ANSWER, scores=scores_out)


def grounding_score(
    answer: str, contexts: list[str], idf: dict[str, float] | None = None
) -> float:
    """IDF-weighted share of the answer's content tokens present in the evidence.

    Weighting by IDF is what makes this meaningful. Unweighted containment is
    near 1.0 for any fluent sentence, because function words dominate and every
    passage contains them. The rare tokens — names, numbers, technical terms —
    are the ones a hallucination invents, and IDF is exactly the weighting that
    makes those decisive.
    """
    answer_tokens = tokenize(answer)
    if not answer_tokens:
        return 0.0

    evidence = set()
    for context in contexts:
        evidence.update(tokenize(context))

    idf = idf or {}
    total = 0.0
    covered = 0.0
    for token in set(answer_tokens):
        # Unseen tokens are treated as rare: an invented word must not be
        # cheap to leave uncovered.
        weight = idf.get(token, math.log(1000.0))
        total += weight
        if token in evidence:
            covered += weight

    return covered / total if total else 0.0


def check_answer(
    answer: Answer,
    candidates: list[Candidate],
    idf: dict[str, float] | None = None,
    cfg: Settings = settings,
) -> GuardVerdict:
    """Verify the answer is supported by the evidence it claims to cite."""
    if not answer.text.strip():
        return GuardVerdict(
            allowed=False,
            decision=Decision.ABSTAIN_NO_EVIDENCE,
            reason="No answer could be extracted from the retrieved evidence.",
            triggered=["empty_answer"],
        )

    # Ground against what was actually cited, not the whole candidate pool.
    # Scoring against every retrieved chunk would let an answer drawn from
    # nowhere pass because some unrelated chunk happened to share its words.
    cited_idx = {c.chunk_id for c in answer.citations}
    contexts = [c.context for c in candidates if c.idx in cited_idx] or [
        c.context for c in candidates
    ]

    score = grounding_score(answer.text, contexts, idf)
    if score < cfg.grounding_floor:
        return GuardVerdict(
            allowed=False,
            decision=Decision.ABSTAIN_UNGROUNDED,
            reason=(
                "A draft answer was produced but could not be verified against "
                "the retrieved passages, so it was withheld."
            ),
            triggered=["low_grounding"],
            scores={"grounding": score},
        )

    if not answer.citations:
        # An answer with no citation cannot be checked by the user, which
        # defeats the purpose of a retrieval system.
        return GuardVerdict(
            allowed=False,
            decision=Decision.ABSTAIN_UNGROUNDED,
            reason="The answer carried no citation and was withheld.",
            triggered=["no_citation"],
            scores={"grounding": score},
        )

    return GuardVerdict(
        allowed=True, decision=Decision.ANSWER, scores={"grounding": score}
    )
