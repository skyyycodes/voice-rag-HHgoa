"""Retrieval quality ablation.

Answers the only question that matters about the retrieval layer: does each
piece of machinery actually earn its latency? Every configuration is scored on
the same held-out queries the reranker never saw, with gold labels from
MS MARCO's `is_selected`.

    uv run python -m vrag.eval_retrieval
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

import numpy as np

from .config import settings
from .index.hybrid import HybridRetriever


@dataclass(slots=True)
class Metrics:
    recall_at_1: float
    recall_at_5: float
    mrr: float
    latency_p50: float
    latency_p95: float
    n: int


def _evaluate(retriever, queries, mode: str, k: int = 5) -> Metrics:
    hits1 = hits5 = 0
    rr = 0.0
    lat: list[float] = []

    for q in queries:
        gold = set(q["gold_chunks"])
        if not gold:
            continue
        t = time.perf_counter()
        results = _retrieve(retriever, q["query"], mode, k)
        lat.append((time.perf_counter() - t) * 1000)

        ranked = [c.idx for c in results]
        if ranked and ranked[0] in gold:
            hits1 += 1
        if gold & set(ranked):
            hits5 += 1
        for rank, idx in enumerate(ranked, 1):
            if idx in gold:
                rr += 1.0 / rank
                break

    n = len(lat) or 1
    return Metrics(
        recall_at_1=hits1 / n,
        recall_at_5=hits5 / n,
        mrr=rr / n,
        latency_p50=float(np.percentile(lat, 50)) if lat else 0.0,
        latency_p95=float(np.percentile(lat, 95)) if lat else 0.0,
        n=n,
    )


def _retrieve(retriever: HybridRetriever, query: str, mode: str, k: int):
    """Isolate one stage of the pipeline at a time."""
    cfg = retriever.cfg
    if mode == "dense":
        from .embed import encode_query

        idx, _ = retriever.dense.search(encode_query(query, cfg), k * 4, cfg)
        return _to_candidates(retriever, idx.tolist(), k)
    if mode == "bm25":
        idx, _ = retriever.lexical.search(query, k * 4)
        return _to_candidates(retriever, idx.tolist(), k)
    if mode == "rrf":
        # Fusion but no learned reranking: keep RRF order.
        saved = retriever.weights
        retriever.weights = np.zeros_like(saved)
        retriever.weights[2] = 1.0  # rrf feature only
        out = retriever.retrieve(query, k)
        retriever.weights = saved
        return out
    return retriever.retrieve(query, k)  # full stack


def _to_candidates(retriever: HybridRetriever, idx: list[int], k: int):
    """Apply the same one-chunk-per-passage rule the full path uses, so the
    ablation compares ranking quality rather than duplicate handling."""
    seen: set[str] = set()
    out = []
    for i in idx:
        pid = str(retriever.store.passage_id[i])
        if pid in seen:
            continue
        seen.add(pid)
        out.append(retriever._new_candidate(i))
        if len(out) >= k:
            break
    return out


def main() -> None:
    cfg = settings
    queries = json.loads((cfg.index_dir / "eval_queries.json").read_text(encoding="utf-8"))
    retriever = HybridRetriever.load(cfg)
    print(f"held-out queries: {len(queries)}   chunks: {len(retriever.store):,}\n")

    rows = []
    for mode, label in [
        ("dense", "dense only (static ANN)"),
        ("bm25", "BM25 only"),
        ("rrf", "hybrid + RRF fusion"),
        ("full", "hybrid + RRF + learned rerank"),
    ]:
        m = _evaluate(retriever, queries, mode)
        rows.append((label, m))

    header = f"{'configuration':<32}{'R@1':>7}{'R@5':>8}{'MRR':>8}{'p50 ms':>9}{'p95 ms':>9}"
    print(header)
    print("-" * len(header))
    for label, m in rows:
        print(
            f"{label:<32}{m.recall_at_1:>7.3f}{m.recall_at_5:>8.3f}"
            f"{m.mrr:>8.3f}{m.latency_p50:>9.1f}{m.latency_p95:>9.1f}"
        )

    base, best = rows[0][1], rows[-1][1]
    if base.mrr:
        print(f"\nfull stack vs dense-only: MRR {base.mrr:.3f} -> {best.mrr:.3f} "
              f"({100 * (best.mrr / base.mrr - 1):+.1f}%)")


if __name__ == "__main__":
    main()
