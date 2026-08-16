"""Typed contracts for every stage boundary.

The pipeline is a sequence of stages that each transform a typed object into
another typed object. Making those boundaries explicit is what separates a
harness from a script: a stage can be retried, timed, swapped, mocked or
skipped precisely because its input and output are declared, and a
malformed hand-off fails at the boundary that produced it rather than three
stages later inside a string format call.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Stage(str, Enum):
    TRANSCRIBE = "transcribe"
    GUARD_INPUT = "guard_input"
    RETRIEVE = "retrieve"
    GENERATE = "generate"
    GUARD_OUTPUT = "guard_output"


class Decision(str, Enum):
    """What the pipeline concluded. Anything other than ANSWER means the
    system declined, and the reason is carried alongside so the UI can explain
    itself instead of showing a blank box."""

    ANSWER = "answer"
    ABSTAIN_NO_EVIDENCE = "abstain_no_evidence"
    ABSTAIN_UNGROUNDED = "abstain_ungrounded"
    REFUSE_UNSAFE = "refuse_unsafe"
    REJECT_OFF_TOPIC = "reject_off_topic"
    REJECT_MALFORMED = "reject_malformed"
    ERROR = "error"


class QueryRequest(BaseModel):
    """Entry point. Exactly one of `audio` or `text` is expected."""

    text: str | None = None
    audio_b64: str | None = None
    audio_format: str = "wav"
    language: str | None = None
    # Lets the demo UI force the LLM path without a server restart.
    answer_mode: str | None = None
    top_k: int | None = None


class Citation(BaseModel):
    chunk_id: int
    text: str
    context: str
    lang: str
    strategy: str
    score: float
    # Character span of the quoted answer inside `context`, for highlighting.
    span_start: int = -1
    span_end: int = -1


class GuardVerdict(BaseModel):
    allowed: bool
    decision: Decision = Decision.ANSWER
    reason: str = ""
    # Named rules that fired, so a refusal is auditable rather than opaque.
    triggered: list[str] = Field(default_factory=list)
    scores: dict[str, float] = Field(default_factory=dict)


class Answer(BaseModel):
    text: str
    citations: list[Citation] = Field(default_factory=list)
    mode: str = "extractive"
    grounding_score: float = 0.0


class StageTiming(BaseModel):
    stage: str
    ms: float
    attempts: int = 1
    ok: bool = True
    note: str = ""


class QueryResponse(BaseModel):
    decision: Decision
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    transcript: str | None = None
    detected_language: str | None = None
    stt_provider: str | None = None
    mode: str = "extractive"
    grounding_score: float = 0.0
    reason: str = ""
    triggered: list[str] = Field(default_factory=list)
    timings: list[StageTiming] = Field(default_factory=list)
    total_ms: float = 0.0
    # Latency excluding the STT network call. The <200ms target applies to the
    # locally-computed pipeline; a third-party HTTP round trip cannot fit in it
    # and is reported separately rather than hidden.
    pipeline_ms: float = 0.0
    budget_exceeded: bool = False
    degraded: list[str] = Field(default_factory=list)

    def timing_map(self) -> dict[str, float]:
        return {t.stage: t.ms for t in self.timings}


class ToolCall(BaseModel):
    """A recorded invocation of a registered tool."""

    name: str
    args: dict[str, Any] = Field(default_factory=dict)
    ms: float = 0.0
    ok: bool = True
    error: str = ""
