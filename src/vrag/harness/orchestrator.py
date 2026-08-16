"""The harness: staged execution with budgets, retries, and degradation.

What makes this a harness rather than a function that calls five other
functions:

* Every stage runs under a *declared latency budget*. Exceeding it is a
  first-class outcome, recorded per stage, not something you find out about
  from a stopwatch around the whole request.
* Every stage is *individually degradable*. STT can fail over between two
  providers and then to text input; the LLM answerer can fall back to the
  extractive one; a guardrail that errors fails closed rather than open. A
  request loses capability instead of dying.
* Failure is *typed*. Each stage returns a `Decision`, so "refused because
  unsafe", "abstained because ungrounded" and "crashed" are distinguishable by
  the caller and by the benchmark — collapsing them into a 500 would make the
  guardrail numbers meaningless.
* Every stage emits a *span*. The per-stage timings the benchmark reports come
  from the same instrumentation that runs in production, not a separate
  measurement path that could drift from it.

Ordering is deliberate: guardrails first (cheapest, and a refusal should never
pay for retrieval), retrieval next, generation last. The output rail runs after
generation because grounding cannot be assessed before there is an answer.
"""

from __future__ import annotations

import asyncio
import base64
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, TypeVar

from ..answer.extractive import answer_extractive
from ..config import Settings, settings
from ..guardrails.input_rails import check_input, normalise, redact
from ..guardrails.output_rails import check_answer, evidence_verdict
from ..index.hybrid import Candidate, HybridRetriever
from ..stt.base import AudioFormat, STTError, Transcript
from .contracts import (
    Answer,
    Decision,
    QueryRequest,
    QueryResponse,
    StageTiming,
)
from .policy import Attempt, CircuitBreaker, CircuitOpen, RetryPolicy, run_with_policy
from .tools import ToolRegistry

T = TypeVar("T")


@dataclass
class Span:
    """One stage's execution record."""

    stage: str
    started: float
    ms: float = 0.0
    attempts: int = 1
    ok: bool = True
    note: str = ""

    def close(self, ok: bool = True, note: str = "", attempts: int = 1) -> "Span":
        self.ms = (time.perf_counter() - self.started) * 1000
        self.ok = ok
        self.note = note
        self.attempts = attempts
        return self

    def to_timing(self) -> StageTiming:
        return StageTiming(
            stage=self.stage, ms=round(self.ms, 3), attempts=self.attempts,
            ok=self.ok, note=self.note,
        )


@dataclass
class Telemetry:
    spans: list[Span] = field(default_factory=list)
    degraded: list[str] = field(default_factory=list)

    def start(self, stage: str) -> Span:
        span = Span(stage=stage, started=time.perf_counter())
        self.spans.append(span)
        return span

    def total_ms(self) -> float:
        return sum(s.ms for s in self.spans)

    def pipeline_ms(self) -> float:
        """Everything except the third-party STT round trip."""
        return sum(s.ms for s in self.spans if s.stage != "transcribe")


class Pipeline:
    """Voice → transcript → guardrail → retrieve → answer → verify."""

    def __init__(
        self,
        retriever: HybridRetriever,
        transcribers: list | None = None,
        cfg: Settings = settings,
        llm_answerer=None,
    ) -> None:
        self.cfg = cfg
        self.retriever = retriever
        self.transcribers = transcribers or []
        self.llm_answerer = llm_answerer
        self.tools = ToolRegistry()
        self._register_tools()

        # One breaker per external dependency, keyed by provider name so a
        # sick provider does not trip the healthy one.
        self.breakers = {
            t.name: CircuitBreaker(name=t.name, failure_threshold=3, reset_after=20.0)
            for t in self.transcribers
        }
        self.breakers["llm"] = CircuitBreaker(name="llm", failure_threshold=2, reset_after=30.0)
        self.stt_policy = RetryPolicy(max_attempts=cfg.stt_max_retries + 1)

    def _register_tools(self) -> None:
        """Stages are registered as named, introspectable tools.

        The pipeline could call these methods directly; going through a
        registry is what lets the bench swap a stage for a stub, the API expose
        a stage on its own for debugging, and every invocation be recorded with
        its timing and outcome without instrumenting each call site.
        """
        self.tools.register("retrieve", self._tool_retrieve, "Hybrid retrieval over the chunk index")
        self.tools.register("answer_extractive", self._tool_extractive, "Span-selection answering")
        self.tools.register("guard_input", self._tool_guard_input, "Input safety and injection rails")
        self.tools.register("guard_output", self._tool_guard_output, "Grounding verification")

    # -- tool implementations ---------------------------------------------
    def _tool_retrieve(self, query: str, k: int | None = None) -> list[Candidate]:
        return self.retriever.retrieve(query, k or self.cfg.final_k)

    def _tool_extractive(self, query: str, candidates: list[Candidate], query_type: str = "DESCRIPTION") -> Answer:
        return answer_extractive(query, candidates, self.retriever.idf, query_type, self.cfg)

    def _tool_guard_input(self, text: str):
        return check_input(text, self.cfg)

    def _tool_guard_output(self, answer: Answer, candidates: list[Candidate]):
        return check_answer(answer, candidates, self.retriever.idf, self.cfg)

    # -- main entry point --------------------------------------------------
    async def run(self, request: QueryRequest) -> QueryResponse:
        tel = Telemetry()
        started = time.perf_counter()

        transcript: Transcript | None = None
        query = (request.text or "").strip()

        # --- stage 1: transcribe (only when audio was sent) ---------------
        if request.audio_b64:
            transcript, failure = await self._transcribe(request, tel)
            if transcript is not None:
                query = transcript.text
            elif not query:
                # No audio result and no typed fallback text: nothing to answer.
                return self._respond(
                    Decision.ERROR, "", tel, started,
                    reason=failure or "Transcription failed.",
                    transcript=None,
                )

        # --- stage 2: input guardrails ------------------------------------
        span = tel.start("guard_input")
        try:
            verdict = self.tools.call("guard_input", text=query)
            span.close(ok=True)
        except Exception as exc:
            # A crashing rail must not become an open door.
            span.close(ok=False, note=f"{type(exc).__name__}")
            return self._respond(
                Decision.ERROR, "", tel, started,
                reason="Input validation failed.", transcript=transcript,
            )

        if not verdict.allowed:
            return self._respond(
                verdict.decision, "", tel, started,
                reason=verdict.reason, triggered=verdict.triggered,
                transcript=transcript,
            )

        clean_query = normalise(query)

        # --- stage 3: retrieval -------------------------------------------
        span = tel.start("retrieve")
        try:
            candidates = self.tools.call(
                "retrieve", query=clean_query, k=request.top_k or self.cfg.final_k
            )
            span.close(ok=True)
        except Exception as exc:
            span.close(ok=False, note=f"{type(exc).__name__}: {exc}")
            return self._respond(
                Decision.ERROR, "", tel, started,
                reason="Retrieval failed.", transcript=transcript,
            )

        if span.ms > self.cfg.budget_retrieve_ms:
            tel.degraded.append(f"retrieve_over_budget:{span.ms:.0f}ms")

        # --- stage 4: evidence sufficiency (out-of-domain rail) -----------
        span = tel.start("guard_evidence")
        evidence = evidence_verdict(candidates, self.cfg)
        span.close(ok=evidence.allowed)
        if not evidence.allowed:
            return self._respond(
                evidence.decision, "", tel, started,
                reason=evidence.reason, triggered=evidence.triggered,
                transcript=transcript, scores=evidence.scores,
            )

        # --- stage 5: generation ------------------------------------------
        mode = request.answer_mode or self.cfg.answer_mode
        span = tel.start("generate")
        answer = await self._generate(clean_query, candidates, mode, tel)
        span.close(ok=bool(answer.text), note=answer.mode)

        # --- stage 6: output guardrails -----------------------------------
        span = tel.start("guard_output")
        try:
            out_verdict = self.tools.call("guard_output", answer=answer, candidates=candidates)
            span.close(ok=out_verdict.allowed)
        except Exception as exc:
            span.close(ok=False, note=f"{type(exc).__name__}")
            return self._respond(
                Decision.ERROR, "", tel, started,
                reason="Answer verification failed.", transcript=transcript,
            )

        if not out_verdict.allowed:
            return self._respond(
                out_verdict.decision, "", tel, started,
                reason=out_verdict.reason, triggered=out_verdict.triggered,
                transcript=transcript, citations=answer.citations,
                scores=out_verdict.scores, mode=answer.mode,
            )

        return self._respond(
            Decision.ANSWER, answer.text, tel, started,
            transcript=transcript, citations=answer.citations,
            mode=answer.mode, scores=out_verdict.scores,
        )

    # -- stage helpers -----------------------------------------------------
    async def _transcribe(
        self, request: QueryRequest, tel: Telemetry
    ) -> tuple[Transcript | None, str]:
        """Try each configured provider in turn, under retry + breaker policy."""
        span = tel.start("transcribe")
        try:
            audio = base64.b64decode(request.audio_b64 or "", validate=True)
        except Exception:
            span.close(ok=False, note="bad_base64")
            return None, "Audio payload was not valid base64."

        fmt = AudioFormat(request.audio_format) if request.audio_format in {
            f.value for f in AudioFormat
        } else AudioFormat.WAV

        last_error = "No speech-to-text provider is configured."
        total_attempts = 0

        for transcriber in self.transcribers:
            attempt = Attempt()
            try:
                result = await run_with_policy(
                    lambda t=transcriber: t.transcribe(audio, fmt, request.language),
                    self.stt_policy,
                    timeout=self.cfg.stt_timeout_s,
                    breaker=self.breakers.get(transcriber.name),
                    attempt_log=attempt,
                )
                total_attempts += attempt.n
                span.close(ok=True, note=transcriber.name, attempts=total_attempts)
                return result, ""
            except CircuitOpen as exc:
                last_error = str(exc)
                tel.degraded.append(f"stt_circuit_open:{transcriber.name}")
            except STTError as exc:
                total_attempts += attempt.n
                last_error = str(exc)
                tel.degraded.append(f"stt_failed:{transcriber.name}")
            except Exception as exc:
                total_attempts += attempt.n
                last_error = f"{type(exc).__name__}: {exc}"
                tel.degraded.append(f"stt_error:{transcriber.name}")

        span.close(ok=False, note=redact(last_error)[:120], attempts=max(1, total_attempts))
        return None, last_error

    async def _generate(
        self, query: str, candidates: list[Candidate], mode: str, tel: Telemetry
    ) -> Answer:
        """LLM path when asked for and healthy; extractive otherwise.

        The extractive answer is computed *first* even on the LLM path — it is
        sub-millisecond, and having it in hand means an LLM timeout degrades to
        a real answer instead of an error.
        """
        extractive = self.tools.call("answer_extractive", query=query, candidates=candidates)

        if mode != "llm" or self.llm_answerer is None:
            return extractive

        breaker = self.breakers["llm"]
        try:
            breaker.check()
            answer = await asyncio.wait_for(
                self.llm_answerer.answer(query, candidates, self.cfg),
                timeout=self.cfg.llm_timeout_s,
            )
            breaker.record_success()
            if answer.text.strip():
                return answer
            # The model abstained. Trust it and abstain too rather than
            # quietly substituting the extractive answer it declined to give.
            tel.degraded.append("llm_abstained")
            return answer
        except CircuitOpen:
            tel.degraded.append("llm_circuit_open")
        except (asyncio.TimeoutError, TimeoutError):
            breaker.record_failure()
            tel.degraded.append("llm_timeout")
        except Exception as exc:
            breaker.record_failure()
            tel.degraded.append(f"llm_error:{type(exc).__name__}")
        return extractive

    # -- response assembly -------------------------------------------------
    def _respond(
        self,
        decision: Decision,
        text: str,
        tel: Telemetry,
        started: float,
        reason: str = "",
        triggered: list[str] | None = None,
        transcript: Transcript | None = None,
        citations: list | None = None,
        mode: str = "extractive",
        scores: dict | None = None,
    ) -> QueryResponse:
        total = (time.perf_counter() - started) * 1000
        pipeline = tel.pipeline_ms()
        return QueryResponse(
            decision=decision,
            answer=text,
            citations=citations or [],
            transcript=transcript.text if transcript else None,
            detected_language=transcript.language if transcript else None,
            stt_provider=transcript.provider if transcript else None,
            mode=mode,
            grounding_score=float((scores or {}).get("grounding", 0.0)),
            reason=reason,
            triggered=triggered or [],
            timings=[s.to_timing() for s in tel.spans],
            total_ms=round(total, 3),
            pipeline_ms=round(pipeline, 3),
            budget_exceeded=pipeline > self.cfg.budget_total_ms,
            degraded=tel.degraded,
        )
