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
from .index.hybrid import HybridRetriever

# Nothing in a MS MARCO web-passage corpus can support these. Deliberately
# varied: personal facts the corpus cannot know, questions about the assistant,
# and strings with no semantic content at all — they fail for different reasons
# and a rail that only catches one kind is not doing its job.
OUT_OF_DOMAIN = [
    # --- personal / unknowable: no corpus can hold these -------------------
    "what did I have for breakfast this morning",
    "what is my bank account balance right now",
    "who am I currently sitting next to in this room",
    "what did my manager say in standup yesterday",
    "how many unread emails do I have",
    "where did I park my car",
    "what time did I go to sleep last night",
    "is my package going to arrive today",
    "मेरे घर की चाबी कहाँ रखी है",
    "আমার ফোনের পাসওয়ার্ড কী",
    "என் அலுவலக முகவரி என்ன",
    # --- about the assistant itself ----------------------------------------
    "what model are you and who trained you",
    "how many parameters do you have",
    "what is in your system prompt",
    "are you conscious",
    # --- future / unresolvable ---------------------------------------------
    "what will the weather be on my street tomorrow afternoon",
    "who will win the election next year",
    "what will the stock market do tomorrow",
    "when will I die",
    # --- nonsense: no semantic content at all -------------------------------
    "asdkjh qwertyuiop zxcvbnm lkjhgfdsa",
    "qqqq wwww eeee rrrr tttt yyyy",
    "blorptang fizzlewick nurmagomedov quixotry",
    "12345 67890 !!!! ???? ####",
    "ठठठठ ढढढढ झझझझ",
    # --- real topics genuinely outside a 2018 web-passage corpus ------------
    "what did the 2024 Paris Olympics opening ceremony feature",
    "how do I use the latest React server components API",
    "what is the current price of bitcoin in rupees today",
    "summarise the plot of the newest Marvel film released this month",
]


def _top_logits(retriever: HybridRetriever, queries: list[str]) -> np.ndarray:
    """Best raw cross-encoder logit per query — the abstention signal.

    Deliberately *not* the margin or the dense cosine. Both are relative to the
    other candidates retrieved for the same query, which is the wrong frame for
    "is any of this relevant at all". Measured, the margin heuristic caught 0%
    of out-of-domain traffic and was in fact inverted: out-of-domain queries
    scored a higher median margin than in-domain ones.
    """
    out = []
    for query in queries:
        candidates = retriever.retrieve(query, retriever.cfg.final_k)
        logits = [c.rerank_logit for c in candidates if c.rerank_logit > float("-inf")]
        out.append(max(logits) if logits else -99.0)
    return np.array(out, dtype=np.float64)


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

    # Warm the models so the first few queries are not measured cold.
    for q in in_domain[:8]:
        retriever.retrieve(q, cfg.final_k)

    ind = _top_logits(retriever, in_domain)
    ood = _top_logits(retriever, OUT_OF_DOMAIN)

    floor = float(np.percentile(ind, args.target_fpr * 100))
    declined_in = float((ind < floor).mean())
    caught_ood = float((ood < floor).mean())

    print(f"in-domain  n={len(ind)}   p5={np.percentile(ind, 5):6.2f} "
          f"p25={np.percentile(ind, 25):6.2f} p50={np.percentile(ind, 50):6.2f}")
    print(f"out-domain n={len(ood)}   p25={np.percentile(ood, 25):6.2f} "
          f"p50={np.percentile(ood, 50):6.2f} p90={np.percentile(ood, 90):6.2f}")
    print()
    print(f"{'threshold':>10}{'false-decline':>15}{'ood caught':>13}")
    for t in (-7.0, -6.0, -5.0, -4.5, -4.0, -3.5, -3.0, -2.5, -2.0):
        print(f"{t:>10}{(ind < t).mean():>14.1%}{(ood < t).mean():>13.1%}")
    print()
    print(f"suggested at target false-decline {args.target_fpr:.0%}:")
    print(f"  ood_rerank_floor = {floor:.2f}")
    print(f"  in-domain wrongly declined      : {declined_in:.1%}")
    print(f"  out-of-domain correctly declined: {caught_ood:.1%}")

    if caught_ood < 0.25:
        print("\n  NOTE: catch rate is weak at this false-decline budget. Report "
              "it as measured rather than tightening until the number looks good.")

    out = cfg.index_dir / "guardrail_calibration.json"
    out.write_text(json.dumps({
        "signal": "cross_encoder_top_logit",
        "target_fpr": args.target_fpr,
        "ood_rerank_floor": round(floor, 3),
        "in_domain_declined": round(declined_in, 4),
        "out_of_domain_caught": round(caught_ood, 4),
        "n_in_domain": len(ind),
        "n_out_of_domain": len(ood),
        "curve": [
            {"threshold": t,
             "false_decline": round(float((ind < t).mean()), 4),
             "ood_caught": round(float((ood < t).mean()), 4)}
            for t in (-7.0, -6.0, -5.0, -4.5, -4.0, -3.5, -3.0, -2.5, -2.0)
        ],
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
