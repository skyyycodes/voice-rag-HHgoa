"""Calibrate the out-of-domain abstention thresholds against measured data.

    uv run python -m vrag.calibrate_guardrails [--target-fpr 0.05]

The abstention rail decides whether retrieval found anything worth answering
from. Picking its thresholds by intuition is how a system ends up either
answering everything (including questions the corpus cannot support) or
declining constantly. Both look fine in a demo of three hand-picked queries.

So the thresholds are fitted instead. Two populations:

* **in-domain** — held-out MSMARCO-XI queries, which by construction have a
  gold passage in the index. Declining these is a false rejection.
* **out-of-domain** — questions no web-passage corpus can answer: personal
  facts, questions about the assistant itself, and nonsense strings. Answering
  these is a false acceptance.

The threshold is chosen at a target false-rejection rate on in-domain traffic,
then the resulting out-of-domain catch rate is *reported rather than
optimised* — tuning both against the same small sample would just overfit to
these particular sentences.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from .config import settings
from .guardrails.output_rails import evidence_verdict
from .index.hybrid import HybridRetriever

# Nothing in a MS MARCO web-passage corpus can support these. Deliberately
# varied: personal facts the corpus cannot know, questions about the assistant,
# and strings with no semantic content at all — they fail for different reasons
# and a rail that only catches one kind is not doing its job.
OUT_OF_DOMAIN = [
    "what did I have for breakfast this morning",
    "what is my bank account balance right now",
    "who am I currently sitting next to in this room",
    "what model are you and who trained you",
    "what did my manager say in standup yesterday",
    "asdkjh qwertyuiop zxcvbnm lkjhgfdsa",
    "मेरे घर की चाबी कहाँ रखी है",           # where are my house keys
    "আমার ফোনের পাসওয়ার্ড কী",                  # what is my phone password
    "என் அலுவலக முகவரி என்ன",                  # what is my office address
    "what will the weather be on my street tomorrow afternoon",
    "how many unread emails do I have",
    "qqqq wwww eeee rrrr tttt yyyy",
]


def _scores(retriever: HybridRetriever, queries: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Return (margin, dense_top) for each query."""
    margins, denses = [], []
    for query in queries:
        candidates = retriever.retrieve(query, retriever.cfg.final_k)
        verdict = evidence_verdict(candidates, retriever.cfg)
        margins.append(verdict.scores.get("margin", 0.0))
        denses.append(verdict.scores.get("dense_top", 0.0))
    return np.array(margins, dtype=np.float64), np.array(denses, dtype=np.float64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-fpr", type=float, default=0.05,
                    help="acceptable fraction of in-domain queries wrongly declined")
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()

    cfg = settings
    retriever = HybridRetriever.load(cfg)
    eval_queries = json.loads((cfg.index_dir / "eval_queries.json").read_text(encoding="utf-8"))
    in_domain = [q["query"] for q in eval_queries[: args.n]]

    in_margin, in_dense = _scores(retriever, in_domain)
    out_margin, out_dense = _scores(retriever, OUT_OF_DOMAIN)

    # The rail abstains only when BOTH signals are weak, so each threshold is
    # set at the target quantile of its own in-domain distribution.
    q = args.target_fpr * 100
    margin_floor = float(np.percentile(in_margin, q))
    dense_floor = float(np.percentile(in_dense, q))

    rejected_in = np.mean((in_margin < margin_floor) & (in_dense < dense_floor))
    rejected_out = np.mean((out_margin < margin_floor) & (out_dense < dense_floor))

    print(f"in-domain  n={len(in_domain)}   margin p5={np.percentile(in_margin, 5):.3f} "
          f"p50={np.percentile(in_margin, 50):.3f}   dense p5={np.percentile(in_dense, 5):.3f} "
          f"p50={np.percentile(in_dense, 50):.3f}")
    print(f"out-domain n={len(OUT_OF_DOMAIN)}   margin p50={np.percentile(out_margin, 50):.3f}"
          f"   dense p50={np.percentile(out_dense, 50):.3f}")
    print()
    print(f"suggested (target FPR {args.target_fpr:.0%}):")
    print(f"  ood_margin_floor = {margin_floor:.3f}")
    print(f"  ood_dense_floor  = {dense_floor:.3f}")
    print()
    print(f"  in-domain wrongly declined : {rejected_in:.1%}  (lower is better)")
    print(f"  out-of-domain correctly declined: {rejected_out:.1%}  (higher is better)")

    if rejected_out < 0.5:
        print("\n  NOTE: out-of-domain catch rate is weak. The two populations "
              "overlap on these signals — report this honestly rather than "
              "tightening the floor until the in-domain rejection rate spikes.")

    out = cfg.index_dir / "guardrail_calibration.json"
    out.write_text(json.dumps({
        "target_fpr": args.target_fpr,
        "ood_margin_floor": round(margin_floor, 4),
        "ood_dense_floor": round(dense_floor, 4),
        "in_domain_rejected": round(float(rejected_in), 4),
        "out_of_domain_rejected": round(float(rejected_out), 4),
        "n_in_domain": len(in_domain),
        "n_out_of_domain": len(OUT_OF_DOMAIN),
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
