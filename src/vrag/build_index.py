"""Offline index build.

    uv run python -m vrag.build_index [--langs hin,ben,tam] [--rows 12000]

Produces everything the query path needs: chunk store, dense ANN index, BM25
index, corpus IDF, fitted reranker weights, and a manifest. Nothing here runs
at query time.

Reranker weights are fitted on a *training* slice of queries and the report
prints held-out numbers, so the retrieval quality claimed in the README is
measured on queries the weights never saw.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .chunking.registry import chunk_passages, default_chunkers
from .config import settings
from .corpus import load
from .embed import encode, get_model
from .index.dense import DenseIndex
from .index.hybrid import HybridRetriever, compute_idf
from .index.lexical import LexicalIndex
from .index.rerank_fit import build_pairs, fit, pair_accuracy, report
from .index.store import ChunkStore


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", default=",".join(settings.languages))
    ap.add_argument("--rows", type=int, default=settings.max_rows_per_lang)
    ap.add_argument("--train-queries", type=int, default=600)
    args = ap.parse_args()

    cfg = settings
    cfg.languages = tuple(args.langs.split(","))
    cfg.max_rows_per_lang = args.rows
    cfg.index_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    print(f"[1/7] loading corpus: {cfg.languages} x {cfg.max_rows_per_lang} rows")
    passages, queries = load(cfg)
    print(f"      {len(passages):,} passages, {len(queries):,} labelled queries")

    print("[2/7] loading embedder")
    get_model(cfg)

    print("[3/7] chunking (7 strategies)")
    t = time.perf_counter()
    chunks, stats = chunk_passages(
        passages, chunkers=default_chunkers(encode=lambda xs: encode(xs, cfg))
    )
    print(stats.report())
    print(f"      {time.perf_counter() - t:.1f}s")

    print("[4/7] embedding chunks")
    t = time.perf_counter()
    vectors = encode([c.text for c in chunks], cfg)
    print(f"      {len(vectors):,} x {vectors.shape[1]}d in {time.perf_counter() - t:.1f}s")

    print("[5/7] building indexes")
    t = time.perf_counter()
    store = ChunkStore.from_chunks(chunks)
    dense = DenseIndex.build(vectors, cfg)
    lexical = LexicalIndex.build(store.text)
    idf = compute_idf(store.text)
    print(f"      {time.perf_counter() - t:.1f}s")

    # Map each query's gold passages to the chunk rows derived from them.
    chunks_by_passage: dict[str, list[int]] = {}
    for i, pid in enumerate(store.passage_id):
        chunks_by_passage.setdefault(str(pid), []).append(i)
    gold_chunks = {
        q.query_id: {i for pid in q.gold_passage_ids for i in chunks_by_passage.get(pid, ())}
        for q in queries
    }

    print("[6/7] fitting reranker on gold labels")
    retriever = HybridRetriever(store, dense, lexical, None, idf, cfg)
    rng = np.random.default_rng(0)
    shuffled = list(queries)
    rng.shuffle(shuffled)
    train = shuffled[: args.train_queries]
    held_out = shuffled[args.train_queries : args.train_queries + 400]

    pos, neg = build_pairs(retriever, train, gold_chunks, cfg)
    weights = fit(pos, neg)
    retriever.weights = weights
    print(report(weights))
    print(f"      train pairs: {len(pos):,}  train pair-acc: {pair_accuracy(weights, pos, neg):.3f}")

    if held_out:
        h_pos, h_neg = build_pairs(retriever, held_out, gold_chunks, cfg)
        print(f"      held-out pair-acc: {pair_accuracy(weights, h_pos, h_neg):.3f}")

    print("[7/7] saving")
    store.save(cfg.chunk_store)
    dense.save(cfg.vector_store)
    lexical.save(cfg.lexical_store)
    np.save(cfg.index_dir / "weights.npy", weights)
    np.savez_compressed(
        cfg.index_dir / "idf.npz",
        keys=np.array(list(idf.keys()), dtype=object),
        vals=np.array(list(idf.values()), dtype=np.float32),
    )
    # Eval queries are held separately so the benchmark never scores itself on
    # the queries the reranker was fitted on.
    eval_path = cfg.index_dir / "eval_queries.json"
    eval_path.write_text(
        json.dumps(
            [
                {
                    "query_id": q.query_id,
                    "lang": q.lang,
                    "query_type": q.query_type,
                    "query": q.query,
                    "answer": q.answer,
                    "gold_chunks": sorted(gold_chunks.get(q.query_id, ())),
                }
                for q in held_out
                if gold_chunks.get(q.query_id)
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    manifest = {
        "languages": list(cfg.languages),
        "rows_per_lang": cfg.max_rows_per_lang,
        "passages": len(passages),
        "chunks": len(chunks),
        "dim": int(vectors.shape[1]),
        "model": cfg.static_model,
        "quantize": cfg.embed_quantize,
        "strategies": sorted({s for c in chunks for s in c.strategies}),
        "chunk_stats": {k: v for k, v in stats.produced.items()},
        "build_seconds": round(time.perf_counter() - t0, 1),
    }
    cfg.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    size = sum(f.stat().st_size for f in Path(cfg.index_dir).rglob("*") if f.is_file())
    print(f"\ndone in {time.perf_counter() - t0:.1f}s — index {size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
