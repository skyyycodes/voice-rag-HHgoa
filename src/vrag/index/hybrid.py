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

3. *Rerank* — a linear model over cheap features, with all score features
   z-scored *within the candidate set*. That normalisation is what makes a
   single set of weights valid across all 14 languages.

No cross-encoder anywhere: the whole stage must fit in a ~120ms retrieval
budget, and a cross-encoder over 40 candidates costs more than that on its own.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config import Settings, settings
from ..embed import encode_query
from ..script_detect import detect_script
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

    # -- retrieval ---------------------------------------------------------
    def retrieve(self, query: str, k: int | None = None) -> list[Candidate]:
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
        self._prefer_query_language(query, candidates)
        candidates.sort(key=lambda c: c.score, reverse=True)
        return self._diversify(candidates, k)

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
        if not want:
            return
        for cand in candidates:
            if cand.lang == want:
                cand.score += self.cfg.same_language_bonus
            elif cand.lang == "eng":
                # English is the corpus's source language and the most common
                # second language of its readers, so it is the preferred
                # fallback when the query's own language is unavailable.
                cand.score += self.cfg.english_fallback_bonus

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
