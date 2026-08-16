"""Hybrid retrieval: dense recall + lexical recall, fused, then reranked.

Dense retrieval alone is not good enough to answer from on this corpus, which
is the entire justification for this layer. `vrag.eval_retrieval` ablates each
stage against the others so the claim is measured rather than asserted.

Three stages:

1. *Recall* — dense ANN and BM25 run independently over the full index. They
   fail on different queries: dense handles paraphrase and cross-script
   matching, BM25 handles the rare entities and numbers that a pooled vector
   washes out.

2. *Fusion* — Reciprocal Rank Fusion. RRF combines by rank, not score, which
   matters enormously here: dense cosine magnitudes are not comparable across
   language pairs, so any score-level blend would silently down-weight every
   cross-lingual hit. Ranks are immune to that.

3. *Rerank* — two stages. First a linear model over cheap features, with all
   score features z-scored *within the candidate set*; that normalisation is
   what makes one weight vector valid across every language. Then a
   cross-encoder rescores the head of the list (see `cross_encoder.py`), which
   is where most of the ranking quality comes from: it lifts MRR over
   dense-only from +3.8% to +17%.

The cross-encoder is affordable only because its depth is chosen from the
*budget still remaining* on the request, and because it scores the short
indexed chunk rather than the parent context — roughly 5x cheaper.

4. *Language preference* — an explicit, measured nudge toward answering in the
   language the question was asked in. Applied last, and scoped to candidates
   that share a scoring scale.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config import Settings, settings
from ..embed import encode_query
from ..script_detect import detect_script
from .cross_encoder import depth_for_budget, get_reranker
from .dense import DenseIndex
from .lexical import LexicalIndex, tokenize
from .store import ChunkStore

# Feature order is fixed; the fitted weight vector is indexed by it.
#
# Every feature is computable from the query string and the chunk text alone.
# Deliberately absent: the corpus `is_selected` flag and the chunk's
# `query_type`. Both are MS MARCO gold annotations — `is_selected` *is* the
# retrieval label. Using either as a ranking feature would leak the answer
# into the retriever and make every recall number reported here fiction.
FEATURES = (
    "dense_z",
    "lexical_z",
    "rrf",
    "coverage",
    "phrase",
    "provenance",
    "length",
)


@dataclass(slots=True)
class Candidate:
    idx: int
    text: str
    context: str
    lang: str
    query_type: str
    dense_score: float = 0.0
    lexical_score: float = 0.0
    dense_rank: int = -1
    lexical_rank: int = -1
    rrf: float = 0.0
    score: float = 0.0
    # True once the cross-encoder has rescored this candidate. Its scores live
    # on a different scale from the feature model's, and anything that reasons
    # about score *magnitudes* has to know which scale it is looking at.
    reranked: bool = False
    # The cross-encoder's *raw* logit, before normalisation into the ranking
    # band. Normalising is right for ordering and wrong for judging: the raw
    # value is an absolute relevance statement ("this passage answers this
    # question") and is the only signal in the pipeline that separates
    # in-domain from out-of-domain traffic. The abstention rail reads it.
    rerank_logit: float = float("-inf")
    features: np.ndarray = field(default_factory=lambda: np.zeros(len(FEATURES), dtype=np.float32))


def _zscore(values: np.ndarray) -> np.ndarray:
    """Standardise within the candidate set.

    This is the fix for cross-language score incomparability: we never compare
    a raw cosine to a global constant, only to the other candidates retrieved
    for this same query, which share its language.
    """
    if len(values) < 2:
        return np.zeros_like(values)
    std = float(values.std())
    if std < 1e-9:
        return np.zeros_like(values)
    return (values - float(values.mean())) / std


class HybridRetriever:
    def __init__(
        self,
        store: ChunkStore,
        dense: DenseIndex,
        lexical: LexicalIndex,
        weights: np.ndarray | None = None,
        idf: dict[str, float] | None = None,
        cfg: Settings = settings,
    ) -> None:
        self.store = store
        self.dense = dense
        self.lexical = lexical
        self.cfg = cfg
        # Hand-set priors, overwritten by `fit_weights` when gold labels exist.
        self.weights = (
            weights
            if weights is not None
            else np.array([1.0, 0.8, 2.0, 1.2, 0.8, 0.3, -0.1], dtype=np.float32)
        )
        self.idf = idf or {}
        # Depth used by the most recent `retrieve`, for telemetry.
        self.last_rerank_depth = 0

    # -- retrieval ---------------------------------------------------------
    def retrieve(
        self,
        query: str,
        k: int | None = None,
        remaining_ms: float | None = None,
    ) -> list[Candidate]:
        """Retrieve the top `k` chunks.

        `remaining_ms` is how much of the request's latency budget is left. It
        controls how deeply the cross-encoder reranks — the pipeline spends
        spare budget on quality rather than banking it. Omitted means "assume
        the full budget", which is what offline evaluation wants.
        """
        cfg = self.cfg
        k = k or cfg.final_k

        qvec = encode_query(query, cfg)
        d_idx, d_score = self.dense.search(qvec, cfg.dense_candidates, cfg)
        l_idx, l_score = self.lexical.search(query, cfg.lexical_candidates)

        pool: dict[int, Candidate] = {}
        for rank, (i, s) in enumerate(zip(d_idx.tolist(), d_score.tolist())):
            pool[i] = self._new_candidate(i)
            pool[i].dense_score = s
            pool[i].dense_rank = rank
        for rank, (i, s) in enumerate(zip(l_idx.tolist(), l_score.tolist())):
            cand = pool.get(i)
            if cand is None:
                cand = pool[i] = self._new_candidate(i)
            cand.lexical_score = s
            cand.lexical_rank = rank

        if not pool:
            return []

        candidates = list(pool.values())
        for cand in candidates:
            # A document missing from one retriever's list gets that
            # retriever's worst possible rank, not a zero — absence is weak
            # evidence against, not neutral.
            d_r = cand.dense_rank if cand.dense_rank >= 0 else cfg.dense_candidates
            l_r = cand.lexical_rank if cand.lexical_rank >= 0 else cfg.lexical_candidates
            cand.rrf = 1.0 / (cfg.rrf_k + d_r + 1) + 1.0 / (cfg.rrf_k + l_r + 1)

        candidates.sort(key=lambda c: c.rrf, reverse=True)
        candidates = candidates[: cfg.fusion_candidates]

        self._score(query, candidates)
        candidates.sort(key=lambda c: c.score, reverse=True)
        self._cross_encode(query, candidates, remaining_ms)
        self._prefer_query_language(query, candidates)
        candidates.sort(key=lambda c: c.score, reverse=True)
        return self._diversify(candidates, k)

    def _cross_encode(
        self, query: str, candidates: list[Candidate], remaining_ms: float | None
    ) -> None:
        """Rescore the head of the list with the cross-encoder.

        Only the head: the cheap stages have already ordered the pool, so the
        expensive model is spent where it can change the answer. Scores are
        *replaced*, not blended — the cross-encoder reads the pair jointly and
        is strictly better informed than the feature model, so averaging the
        two would drag its judgement back toward the weaker signal.
        """
        cfg = self.cfg
        self.last_rerank_depth = 0
        if not cfg.rerank_enabled or not candidates:
            return

        budget = cfg.budget_total_ms if remaining_ms is None else remaining_ms
        depth = depth_for_budget(budget, cfg)
        if depth <= 1:
            return

        self.last_rerank_depth = depth
        head = candidates[:depth]
        try:
            # Score the indexed chunk, not its parent context. The chunk is
            # the unit retrieval matched, and it is ~5x cheaper: parent blocks
            # cost 288ms for 20 pairs against 57ms for the chunks themselves,
            # because cross-encoder cost scales with pair length.
            scores = get_reranker(cfg).score(query, [c.text for c in head])
        except Exception:
            # Reranking is an enhancement, not a dependency. If the model is
            # unavailable the fused ordering still stands.
            return

        # Put *every* candidate on one comparable 0-1 scale rather than
        # shifting the reranked head into a separate band with a magic offset.
        #
        # The offset version left two score populations in one list — the head
        # near 105, the tail near 0.4 — and 37% of final candidate sets ended up
        # containing both. Anything downstream that reasons about score
        # magnitudes then computed on a bimodal distribution: the abstention
        # rail's margin divides by the standard deviation of the "rest", which
        # was dominated by the artificial gap rather than by any relevance
        # signal. Normalising into adjacent bands keeps the ordering guarantee
        # (reranked always above unreranked) while making magnitudes mean
        # something.
        head_scores = np.asarray(scores, dtype=np.float32)
        lo, hi = float(head_scores.min()), float(head_scores.max())
        span = (hi - lo) or 1.0
        for cand, raw in zip(head, head_scores):
            cand.rerank_logit = float(raw)
            cand.score = 0.5 + 0.5 * (float(raw) - lo) / span
            cand.reranked = True

        tail = candidates[depth:]
        if tail:
            tail_scores = np.array([c.score for c in tail], dtype=np.float32)
            t_lo, t_hi = float(tail_scores.min()), float(tail_scores.max())
            t_span = (t_hi - t_lo) or 1.0
            for cand, raw in zip(tail, tail_scores):
                cand.score = 0.45 * (float(raw) - t_lo) / t_span

    def _prefer_query_language(self, query: str, candidates: list[Candidate]) -> None:
        """Nudge same-script candidates above their translations.

        Applied after scoring rather than as a reranker feature: which language
        the reader wants is a product decision, and the relevance labels carry
        no signal about it. The bonus is small because it is only meant to
        break ties between translations of the same passage — it must not
        promote a weakly-matching same-language chunk over a strongly-matching
        foreign one, since a correct answer in the wrong language still beats a
        wrong answer in the right one.
        """
        want = detect_script(query)
        if not want or len(candidates) < 2:
            return

        # The bonus is expressed as a fraction of the candidate set's own score
        # spread, not as an absolute number. Two different scorers feed this:
        # the linear model produces scores around ±2, the cross-encoder
        # produces logits around ±11. A fixed bonus tuned for one is either
        # overwhelming or invisible on the other — when the cross-encoder was
        # added, a fixed +0.25 silently stopped having any effect and Hindi
        # questions went back to being answered in Bengali.
        # Measure the spread over candidates that share a scoring scale. When
        # the cross-encoder has run, the full list spans two scales — reranked
        # candidates sit ~100 above the rest because of the ordering offset —
        # and the resulting "spread" is that artificial gap, not signal. Using
        # it produced a 16.1-point language bonus on top of a cross-encoder
        # whose entire discriminative range was 4.4 points, so language silently
        # overrode relevance for every reranked query.
        pool = [c for c in candidates if c.reranked] or candidates
        if len(pool) < 2:
            return
        scores = np.array([c.score for c in pool], dtype=np.float32)
        spread = float(scores.max() - scores.min())
        if spread < 1e-6:
            spread = 1.0

        # Only candidates on the same scale may be nudged past one another.
        candidates = pool

        same = self.cfg.same_language_bonus * spread
        english = self.cfg.english_fallback_bonus * spread
        for cand in candidates:
            if cand.lang == want:
                cand.score += same
            elif cand.lang == "eng":
                # English is the corpus's source language and the most common
                # second language of its readers, so it is the preferred
                # fallback when the query's own language is unavailable.
                cand.score += english

    def _new_candidate(self, i: int) -> Candidate:
        s = self.store
        return Candidate(
            idx=i,
            text=s.text[i],
            context=s.context[i],
            lang=str(s.lang[i]),
            query_type=str(s.query_type[i]),
        )

    # -- reranking ---------------------------------------------------------
    def _score(self, query: str, candidates: list[Candidate]) -> None:
        feats = self.features(query, candidates)
        scores = feats @ self.weights
        for cand, f, s in zip(candidates, feats, scores):
            cand.features = f
            cand.score = float(s)

    def features(self, query: str, candidates: list[Candidate]) -> np.ndarray:
        """Feature matrix, shape (n_candidates, len(FEATURES))."""
        q_tokens = tokenize(query)
        q_set = set(q_tokens)
        q_idf = {t: self.idf.get(t, 1.0) for t in q_set}
        idf_total = sum(q_idf.values()) or 1.0

        dense = _zscore(np.array([c.dense_score for c in candidates], dtype=np.float32))
        lexical = _zscore(np.array([c.lexical_score for c in candidates], dtype=np.float32))
        rrf = np.array([c.rrf for c in candidates], dtype=np.float32)
        rrf = rrf / (rrf.max() or 1.0)

        out = np.zeros((len(candidates), len(FEATURES)), dtype=np.float32)
        for row, cand in enumerate(candidates):
            c_tokens = tokenize(cand.text)
            c_set = set(c_tokens)

            # IDF-weighted share of the query's terms the chunk actually
            # contains. Rewards covering the *rare* terms, not just many terms.
            coverage = sum(q_idf[t] for t in q_set & c_set) / idf_total
            phrase = _longest_run(q_tokens, c_tokens) / max(1, len(q_tokens))
            n_strat = int(self.store.n_strategies[cand.idx])
            length = math.log1p(len(c_tokens)) / 6.0

            out[row] = (
                dense[row],
                lexical[row],
                rrf[row],
                coverage,
                phrase,
                0.05 * min(3, n_strat - 1),
                length,
            )
        return out

    def _diversify(self, candidates: list[Candidate], k: int) -> list[Candidate]:
        """One chunk per source passage.

        Seven chunking strategies over one passage produce many overlapping
        spans. Without this, a single passage can occupy every slot in the
        top-k, which starves the answer stage of corroborating evidence and
        makes the grounding check meaningless — it would be verifying an answer
        against the one passage it came from, restated seven ways.
        """
        seen: set[str] = set()
        out: list[Candidate] = []
        for cand in candidates:
            pid = str(self.store.passage_id[cand.idx])
            if pid in seen:
                continue
            seen.add(pid)
            out.append(cand)
            if len(out) >= k:
                break
        return out

    # -- persistence -------------------------------------------------------
    def save_weights(self, path: Path) -> None:
        np.save(path, self.weights)

    @classmethod
    def load(cls, cfg: Settings = settings) -> HybridRetriever:
        store = ChunkStore.load(cfg.chunk_store)
        dense = DenseIndex.load(cfg.vector_store, cfg)
        lexical = LexicalIndex.load(cfg.lexical_store)
        weights_path = cfg.index_dir / "weights.npy"
        weights = np.load(weights_path) if weights_path.exists() else None
        idf_path = cfg.index_dir / "idf.npz"
        idf = {}
        if idf_path.exists():
            data = np.load(idf_path, allow_pickle=True)
            idf = dict(zip(data["keys"].tolist(), data["vals"].tolist()))
        return cls(store, dense, lexical, weights, idf, cfg)


def _longest_run(query_tokens: list[str], chunk_tokens: list[str]) -> int:
    """Longest contiguous run of query tokens appearing in the chunk.

    Distinguishes a chunk that happens to contain the query's words scattered
    about from one that contains the actual phrase — the difference between
    "bank of the river" and "river bank".
    """
    if not query_tokens or not chunk_tokens:
        return 0
    positions: dict[str, list[int]] = {}
    for i, t in enumerate(chunk_tokens):
        positions.setdefault(t, []).append(i)

    best = 0
    for start in range(len(query_tokens)):
        for pos in positions.get(query_tokens[start], ()):
            run = 0
            while (
                start + run < len(query_tokens)
                and pos + run < len(chunk_tokens)
                and query_tokens[start + run] == chunk_tokens[pos + run]
            ):
                run += 1
            best = max(best, run)
    return best


def compute_idf(texts: list[str]) -> dict[str, float]:
    """Corpus IDF over our own tokeniser, for the coverage feature."""
    from collections import Counter

    df: Counter = Counter()
    for text in texts:
        df.update(set(tokenize(text)))
    n = len(texts) or 1
    return {t: math.log(1.0 + n / (1.0 + c)) for t, c in df.items()}
