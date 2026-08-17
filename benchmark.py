"""Measure end-to-end pipeline latency against the 200ms budget.

    uv run python benchmark.py [n_queries]

Standalone and dependency-light on purpose — it prints one table you can paste
into a chat. `python -m vrag.bench` is the full harness (per-language splits,
retrieval quality, guardrail decision counts, JSON report); this is the quick
number.

What is measured is the *pipeline*: guardrails, retrieval, reranking and answer
generation. Speech-to-text is deliberately excluded — it is a network call to
Sarvam (~2s, and not ours to optimise), and the 200ms target is for the local
pipeline. The UI reports both separately for the same reason.

Queries come from the held-out evaluation set, not from a list written by hand:
they are real MS MARCO questions in Hindi, Bengali and Tamil, which is what the
index actually contains.
"""

from __future__ import annotations

import asyncio
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from vrag.config import settings  # noqa: E402
from vrag.harness.contracts import QueryRequest  # noqa: E402
from vrag.harness.orchestrator import Pipeline  # noqa: E402
from vrag.index.hybrid import HybridRetriever  # noqa: E402

BUDGET_MS = settings.budget_total_ms


def percentile(values: list[float], pct: float) -> float:
    values = sorted(values)
    k = (len(values) - 1) * (pct / 100)
    f, c = int(k), min(int(k) + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (k - f) * (values[c] - values[f])


def load_queries() -> list[tuple[str, str]]:
    path = settings.index_dir / "eval_queries.json"
    if not path.exists():
        raise SystemExit(f"No evaluation queries at {path}. Run: uv run python -m vrag.build_index")
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [(r["query"], r["lang"]) for r in rows if r.get("query")]


async def run(n: int, warmup: int) -> dict:
    retriever = HybridRetriever.load(settings)
    pipeline = Pipeline(retriever, transcribers=[], cfg=settings)

    # The first requests pay ONNX graph warm-up and page-cache misses for the
    # memory-mapped index. Timing them would measure startup, not steady state.
    print(f"Warming up ({warmup} queries: ONNX sessions, index paging)...", flush=True)
    queries = load_queries()
    for i in range(warmup):
        await pipeline.run(QueryRequest(text=queries[i % len(queries)][0]))

    print(f"Running {n} queries...", flush=True)
    total: list[float] = []
    stages: dict[str, list[float]] = {}
    by_lang: dict[str, list[float]] = {}
    decisions: Counter[str] = Counter()

    for i in range(n):
        query, lang = queries[i % len(queries)]
        response = await pipeline.run(QueryRequest(text=query))
        total.append(response.pipeline_ms)
        by_lang.setdefault(lang, []).append(response.pipeline_ms)
        decisions[response.decision.value] += 1
        for timing in response.timings:
            stages.setdefault(timing.stage, []).append(timing.ms)

    return {"total": total, "stages": stages, "by_lang": by_lang, "decisions": decisions}


def render(results: dict, n: int) -> str:
    total = results["total"]
    cols = ("avg", "p50", "p70", "p95", "p99", "p100")

    def row(name: str, values: list[float]) -> str:
        cells = [
            statistics.mean(values),
            percentile(values, 50),
            percentile(values, 70),
            percentile(values, 95),
            percentile(values, 99),
            percentile(values, 100),
        ]
        return f"{name:<16}" + "".join(f"{c:>9.2f}" for c in cells)

    header = f"{'stage':<16}" + "".join(f"{c:>9}" for c in cols) + "   (ms)"
    lines = [
        "",
        f"Voice RAG — pipeline latency over {n} queries",
        f"index: {settings.index_dir}  |  budget: {BUDGET_MS:.0f} ms",
        "",
        header,
        "-" * len(header),
    ]
    for stage in ("guard_input", "retrieve", "guard_evidence", "generate", "guard_output"):
        if stage in results["stages"]:
            lines.append(row(stage, results["stages"][stage]))
    lines += ["-" * len(header), row("PIPELINE TOTAL", total), ""]

    lines.append("per language (pipeline ms)")
    for lang, values in sorted(results["by_lang"].items()):
        lines.append(
            f"  {lang:<6} n={len(values):<5} p50 {percentile(values, 50):>7.2f}"
            f"   p100 {percentile(values, 100):>7.2f}"
        )

    lines += ["", "decisions"]
    for decision, count in results["decisions"].most_common():
        lines.append(f"  {decision:<22} {count:>5}  ({100 * count / len(total):.1f}%)")

    within = sum(1 for v in total if v <= BUDGET_MS)
    lines += [
        "",
        f"within {BUDGET_MS:.0f} ms budget: {within}/{len(total)} = {100 * within / len(total):.1f}%",
        f"P50 {percentile(total, 50):.2f} ms   P70 {percentile(total, 70):.2f} ms   "
        f"P100 {percentile(total, 100):.2f} ms",
    ]
    return "\n".join(lines)


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    results = asyncio.run(run(n, warmup=min(30, max(5, n // 10))))
    print(render(results, n))

    # Gate on the worst case, not the median. The claim being defended is that
    # *every* request lands inside the budget, so a P95 gate would pass a build
    # that breaches it on one query in twenty.
    worst = percentile(results["total"], 100)
    if worst <= BUDGET_MS:
        print(f"\nPASS: worst case {worst:.2f} ms is within the {BUDGET_MS:.0f} ms budget")
    else:
        print(f"\nFAIL: worst case {worst:.2f} ms exceeds the {BUDGET_MS:.0f} ms budget")
        sys.exit(1)


if __name__ == "__main__":
    main()
