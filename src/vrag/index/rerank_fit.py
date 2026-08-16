"""Fit the reranker's linear weights on gold relevance labels.

This is a small learning-to-rank step, not a heuristic tune. For each training
query we retrieve a candidate set, label candidates by whether they came from a
gold `is_selected` passage, and fit weights by pairwise logistic regression:
every (relevant, non-relevant) pair inside a query should be ordered correctly.

Pairwise rather than pointwise because ranking is what we actually care about,
and because it is naturally invariant to the per-query score scaling that the
multilingual corpus forces on us.

Implemented in ~40 lines of numpy rather than pulling in scikit-learn: the
model is seven weights, and a dependency that size is not worth it for a
container we are trying to keep small.
"""

from __future__ import annotations

import numpy as np

from ..config import Settings, settings
from .hybrid import FEATURES, HybridRetriever


def build_pairs(
    retriever: HybridRetriever,
    queries: list,
    gold_chunks: dict[int, set[int]],
    cfg: Settings = settings,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (positive_features, negative_features) aligned as pairs."""
    pos_rows: list[np.ndarray] = []
    neg_rows: list[np.ndarray] = []

    for q in queries:
        gold = gold_chunks.get(q.query_id)
        if not gold:
            continue
        candidates = retriever.retrieve(q.query, k=cfg.fusion_candidates)
        if len(candidates) < 2:
            continue
        feats = retriever.features(q.query, candidates)
        labels = np.array([1 if c.idx in gold else 0 for c in candidates])
        if labels.sum() == 0 or labels.sum() == len(labels):
            # No contrast in this candidate set — nothing to learn from.
            continue
        pos = feats[labels == 1]
        neg = feats[labels == 0]
        # Cap pairs per query so long candidate lists do not dominate the fit.
        for p in pos[:4]:
            for n in neg[:8]:
                pos_rows.append(p)
                neg_rows.append(n)

    if not pos_rows:
        return np.zeros((0, len(FEATURES)), np.float32), np.zeros((0, len(FEATURES)), np.float32)
    return np.array(pos_rows, np.float32), np.array(neg_rows, np.float32)


def fit(
    pos: np.ndarray,
    neg: np.ndarray,
    epochs: int = 300,
    lr: float = 0.15,
    l2: float = 1e-3,
    seed: int = 0,
) -> np.ndarray:
    """Pairwise logistic regression on feature differences.

    For a pair (p, n) we want w·p > w·n, i.e. sigmoid(w·(p-n)) -> 1. That makes
    the whole thing plain logistic regression on the difference vectors, with
    an implicit label of 1 everywhere.
    """
    if len(pos) == 0:
        return np.array([1.0, 0.8, 2.0, 1.2, 0.8, 0.3, -0.1], dtype=np.float32)

    diff = pos - neg
    rng = np.random.default_rng(seed)
    w = rng.normal(0, 0.01, size=diff.shape[1])

    for _ in range(epochs):
        margin = diff @ w
        # sigmoid(-margin) is the per-pair gradient weight: pairs already
        # ordered correctly contribute almost nothing.
        grad_w = -(diff * _sigmoid(-margin)[:, None]).mean(axis=0) + l2 * w
        w -= lr * grad_w

    return w.astype(np.float32)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    # Branch on sign to avoid overflow in exp for large negative inputs.
    out = np.empty_like(x)
    pos_mask = x >= 0
    out[pos_mask] = 1.0 / (1.0 + np.exp(-x[pos_mask]))
    e = np.exp(x[~pos_mask])
    out[~pos_mask] = e / (1.0 + e)
    return out


def pair_accuracy(w: np.ndarray, pos: np.ndarray, neg: np.ndarray) -> float:
    """Fraction of (relevant, non-relevant) pairs ordered correctly."""
    if len(pos) == 0:
        return float("nan")
    return float(((pos - neg) @ w > 0).mean())


def report(w: np.ndarray) -> str:
    order = np.argsort(-np.abs(w))
    lines = ["fitted reranker weights:"]
    for i in order:
        lines.append(f"  {FEATURES[i]:<12} {w[i]:+.3f}")
    return "\n".join(lines)
