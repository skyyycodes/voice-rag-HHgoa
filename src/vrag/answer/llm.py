"""LLM answer path (Claude), behind the same contract as the extractive path.

This exists for answer *fluency*, not capability: the extractive path already
finds the right span, but it reads as quoted source text. What it cannot do is
fuse two passages into one sentence or answer in the user's language when the
evidence is in another.

It cannot meet the 200ms target and is not pretended to — a single API round
trip is ~1-2s. It is benchmarked and reported separately, and the harness
falls back to the extractive answer whenever this path is slow, erroring, or
ungrounded.

Three things make it safe to put an LLM behind a grounding guarantee:

* Structured output. `messages.parse` validates against a Pydantic schema, so
  the answer, its citations and its own abstention flag arrive as typed fields
  rather than prose to be regex'd out.
* The same output rail. The generated answer goes through the identical
  grounding check as the extractive one — fluency does not buy an exemption.
* Explicit abstention. The schema has an `answerable` field, so "the passages
  do not say" is a first-class response instead of something the model has to
  be talked out of inventing.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

from pydantic import BaseModel, Field

from ..config import Settings, settings
from ..harness.contracts import Answer, Citation
from ..index.hybrid import Candidate

SYSTEM = """You answer strictly from the numbered passages supplied by the user.

Rules:
- Use only facts stated in the passages. Never add outside knowledge.
- If the passages do not contain the answer, set answerable=false and leave
  answer empty. Do not guess, and do not answer a question that was not asked.
- Cite the passage numbers you actually used in `cited`.
- Answer in the same language as the question.
- Two sentences maximum. No preamble, no restating the question."""


class LLMAnswer(BaseModel):
    """Schema the model is constrained to. Field order matters: `answerable`
    is declared first so the model commits to whether it can answer before it
    starts composing, rather than rationalising a decision it already made."""

    answerable: bool = Field(description="True only if the passages contain the answer.")
    answer: str = Field(default="", description="The answer, or empty if not answerable.")
    cited: list[int] = Field(
        default_factory=list, description="1-based numbers of passages actually used."
    )


def _build_prompt(query: str, candidates: Sequence[Candidate]) -> str:
    blocks = [
        f"[{i}] {c.context.strip()}" for i, c in enumerate(candidates, start=1)
    ]
    return "PASSAGES:\n" + "\n\n".join(blocks) + f"\n\nQUESTION: {query}"


class LLMAnswerer:
    def __init__(self, cfg: Settings = settings) -> None:
        self.cfg = cfg
        self._client = None

    def _get_client(self):
        if self._client is None:
            from anthropic import AsyncAnthropic

            self._client = AsyncAnthropic(
                api_key=self.cfg.anthropic_api_key or None,
                timeout=self.cfg.llm_timeout_s,
                # The harness owns retry policy for every other stage; letting
                # the SDK also retry would silently multiply the wall-clock
                # budget by (max_retries + 1).
                max_retries=0,
            )
        return self._client

    async def answer(
        self, query: str, candidates: list[Candidate], cfg: Settings | None = None
    ) -> Answer:
        cfg = cfg or self.cfg
        if not candidates:
            return Answer(text="", citations=[], mode="llm")

        client = self._get_client()
        response = await client.messages.parse(
            model=cfg.llm_model,
            # Thinking is on by default on Opus 5 and shares this budget with
            # the response text, so a tight cap sized to the answer alone would
            # truncate mid-sentence.
            max_tokens=2048,
            # Lowest effort that still reasons: this is span selection over
            # supplied text, not open-ended analysis, and effort is the main
            # latency dial. Note `temperature` is not accepted on this model.
            output_config={"effort": "low"},
            system=SYSTEM,
            messages=[{"role": "user", "content": _build_prompt(query, candidates)}],
            output_format=LLMAnswer,
        )

        # A safety classifier can decline with HTTP 200 and empty content.
        # Reading `.parsed_output` first would raise on an attribute that was
        # never populated, turning a policy decision into a crash.
        if response.stop_reason == "refusal":
            return Answer(text="", citations=[], mode="llm")

        parsed = response.parsed_output
        if parsed is None or not parsed.answerable or not parsed.answer.strip():
            # The model declined for lack of evidence — exactly the outcome the
            # abstention rail wants, reached without a fabricated answer.
            return Answer(text="", citations=[], mode="llm")

        citations = [
            Citation(
                chunk_id=candidates[n - 1].idx,
                text=candidates[n - 1].text,
                context=candidates[n - 1].context,
                lang=candidates[n - 1].lang,
                strategy=candidates[n - 1].query_type,
                score=float(candidates[n - 1].score),
            )
            for n in parsed.cited
            if 1 <= n <= len(candidates)
        ]
        # A cited-nothing answer would sail past the citation rail on a
        # technicality; attribute it to the top candidate so the grounding
        # check has something concrete to verify against.
        if not citations:
            top = candidates[0]
            citations = [
                Citation(
                    chunk_id=top.idx,
                    text=top.text,
                    context=top.context,
                    lang=top.lang,
                    strategy=top.query_type,
                    score=float(top.score),
                )
            ]

        return Answer(text=parsed.answer.strip(), citations=citations, mode="llm")

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.close()


def is_configured(cfg: Settings = settings) -> bool:
    """Whether the LLM path can run at all. The harness checks this before
    selecting the mode so a missing key degrades to extractive rather than
    failing the request."""
    import os

    return bool(cfg.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY"))


__all__ = ["LLMAnswer", "LLMAnswerer", "asyncio", "is_configured"]
