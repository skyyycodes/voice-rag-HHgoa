"""Latency and quality benchmark.

    uv run python -m vrag.bench [--n 500] [--mode extractive|llm] [--warmup 30]

Reports P50 / P70 / P100 end-to-end and per stage, plus retrieval quality and
guardrail behaviour, over held-out queries the reranker never saw.

Three decisions that keep the numbers honest:

*Warm-up runs are discarded.* The first requests pay ONNX graph optimisation,
lazy index page-ins and Python import costs. Including them makes P100 a
measure of process startup rather than of the pipeline.

*P100 is reported, not trimmed.* P100 is the worst single observation, so it is
the number most sensitive to an unlucky OS scheduling event — which is exactly
why it belongs in the report. P99 is shown beside it so the tail's shape is
visible rather than resting on one sample.

*Two totals are always reported.* `pipeline_ms` covers the locally-computed
path — chunking, retrieval, guardrails, generation — which is what the 200ms
target can meaningfully constrain. `total_ms` adds the speech-to-text network
round trip. No third-party HTTP call fits in 200ms, and averaging the two into
one headline number would obscure that rather than address it.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .config import settings
from .harness.contracts import Decision, QueryRequest
from .harness.orchestrator import Pipeline
from .index.hybrid import HybridRetriever

PERCENTILES = (50, 70, 90, 95, 99, 100)


def percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {f"p{p}": 0.0 for p in PERCENTILES}
    arr = np.array(values, dtype=np.float64)
    return {f"p{p}": round(float(np.percentile(arr, p)), 2) for p in PERCENTILES}


async def run_bench(n: int, mode: str, warmup: int) -> dict:
    cfg = settings
    retriever = HybridRetriever.load(cfg)
    queries = json.loads((cfg.index_dir / "eval_queries.json").read_text(encoding="utf-8"))
    if not queries:
        raise SystemExit("No eval queries. Run `python -m vrag.build_index` first.")

    llm = None
    if mode == "llm":
        from .answer.llm import LLMAnswerer, is_configured

        if not is_configured(cfg):
            raise SystemExit("mode=llm needs ANTHROPIC_API_KEY in .env")
        llm = LLMAnswerer(cfg)

    pipeline = Pipeline(retriever, transcribers=[], cfg=cfg, llm_answerer=llm)

    # Warm-up: touch every code path so measurement excludes first-call costs.
    for q in queries[: min(warmup, len(queries))]:
        await pipeline.run(QueryRequest(text=q["query"], answer_mode=mode))

    sample = queries[:n]
    e2e: list[float] = []
    pipe: list[float] = []
    stages: dict[str, list[float]] = defaultdict(list)
    decisions: Counter = Counter()
    by_lang: dict[str, list[float]] = defaultdict(list)

    hits1 = hits5 = 0
    rr = 0.0
    answered_with_gold = 0
    scored = 0

    wall_start = time.perf_counter()
    for q in sample:
        response = await pipeline.run(QueryRequest(text=q["query"], answer_mode=mode))

        e2e.append(response.total_ms)
        pipe.append(response.pipeline_ms)
        by_lang[q["lang"]].append(response.pipeline_ms)
        decisions[response.decision.value] += 1
        for timing in response.timings:
            stages[timing.stage].append(timing.ms)

        gold = set(q["gold_chunks"])
        if gold:
            scored += 1
            ranked = [c.chunk_id for c in response.citations] or []
            # Citations only carry what the answer used, so re-run retrieval
            # to score ranking independently of what generation selected.
            retrieved = [c.idx for c in retriever.retrieve(q["query"], cfg.final_k)]
            if retrieved and retrieved[0] in gold:
                hits1 += 1
            if gold & set(retrieved):
                hits5 += 1
            for rank, idx in enumerate(retrieved, 1):
                if idx in gold:
                    rr += 1.0 / rank
                    break
            if response.decision == Decision.ANSWER and set(ranked) & gold:
                answered_with_gold += 1

    wall = time.perf_counter() - wall_start
    scored = scored or 1

    return {
        "config": {
            "mode": mode,
            "queries": len(sample),
            "warmup": warmup,
            "chunks": len(retriever.store),
            "final_k": cfg.final_k,
            "encoder": cfg.encoder_model,
        },
        "latency_pipeline_ms": percentiles(pipe),
        "latency_end_to_end_ms": percentiles(e2e),
        "latency_by_stage_ms": {k: percentiles(v) for k, v in sorted(stages.items())},
        "latency_by_language_ms": {k: percentiles(v) for k, v in sorted(by_lang.items())},
        "throughput_qps": round(len(sample) / wall, 1),
        "budget": {
            "target_ms": cfg.budget_total_ms,
            "within_budget_pct": round(
                100.0 * sum(1 for v in pipe if v <= cfg.budget_total_ms) / max(1, len(pipe)), 1
            ),
        },
        "retrieval": {
            "recall_at_1": round(hits1 / scored, 4),
            "recall_at_5": round(hits5 / scored, 4),
            "mrr_at_5": round(rr / scored, 4),
        },
        "decisions": dict(decisions),
        "answered_and_cited_gold_pct": round(100.0 * answered_with_gold / scored, 1),
    }


def render(report: dict) -> str:
    cfg = report["config"]
    lines = [
        "=" * 68,
        f"  Voice-RAG benchmark — {cfg['queries']} held-out queries, mode={cfg['mode']}",
        f"  {cfg['chunks']:,} chunks · {cfg['encoder']}",
        "=" * 68,
        "",
        "LATENCY — pipeline (chunk + retrieve + guard + generate)",
    ]
    p = report["latency_pipeline_ms"]
    lines.append(
        f"  P50 {p['p50']:>7.2f} ms   P70 {p['p70']:>7.2f} ms   "
        f"P95 {p['p95']:>7.2f} ms   P100 {p['p100']:>7.2f} ms"
    )
    budget = report["budget"]
    lines += [
        f"  within {budget['target_ms']:.0f}ms budget: {budget['within_budget_pct']}% of queries",
        f"  throughput: {report['throughput_qps']} queries/sec (single process)",
        "",
        "PER STAGE (ms)",
        f"  {'stage':<18}{'P50':>9}{'P70':>9}{'P95':>9}{'P100':>9}",
        "  " + "-" * 54,
    ]
    for stage, s in report["latency_by_stage_ms"].items():
        lines.append(f"  {stage:<18}{s['p50']:>9.2f}{s['p70']:>9.2f}{s['p95']:>9.2f}{s['p100']:>9.2f}")

    lines += ["", "PER LANGUAGE — pipeline ms", f"  {'lang':<8}{'P50':>9}{'P70':>9}{'P100':>9}", "  " + "-" * 35]
    for lang, s in report["latency_by_language_ms"].items():
        lines.append(f"  {lang:<8}{s['p50']:>9.2f}{s['p70']:>9.2f}{s['p100']:>9.2f}")

    r = report["retrieval"]
    lines += [
        "",
        "RETRIEVAL QUALITY (gold labels from MS MARCO is_selected)",
        f"  Recall@1 {r['recall_at_1']:.3f}   Recall@5 {r['recall_at_5']:.3f}   MRR@5 {r['mrr_at_5']:.3f}",
        "",
        "GUARDRAIL DECISIONS",
    ]
    total = sum(report["decisions"].values()) or 1
    for decision, count in sorted(report["decisions"].items(), key=lambda kv: -kv[1]):
        lines.append(f"  {decision:<26}{count:>6}  ({100 * count / total:>5.1f}%)")

    lines += ["", f"answered AND cited a gold passage: {report['answered_and_cited_gold_pct']}%", ""]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--mode", default="extractive", choices=["extractive", "llm"])
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--out", default="bench/results/latest.json")
    args = ap.parse_args()

    report = asyncio.run(run_bench(args.n, args.mode, args.warmup))
    print(render(report))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
