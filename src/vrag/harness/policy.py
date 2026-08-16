"""Execution policy: retries, timeouts, circuit breaking.

Retry logic keyed on *error type*, not on a blanket except. Retrying a bad API
key is pointless and retrying malformed audio is worse than pointless — it
triples the latency of a request that was always going to fail. Only genuinely
transient conditions get another attempt.

The circuit breaker exists for the case that actually bites in a live demo: the
STT provider goes down, every request spends its full timeout budget before
failing, and the whole app appears hung. After a threshold of consecutive
failures the breaker opens and subsequent calls fail instantly, so the pipeline
degrades to text input in milliseconds instead of stalling.
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, TypeVar

from ..stt.base import STTAuthError, STTBadAudio, STTRateLimited, STTTransient

T = TypeVar("T")

# Only these justify another attempt. Everything else fails fast.
RETRYABLE = (STTTransient, STTRateLimited, asyncio.TimeoutError, TimeoutError)
# Explicitly non-retryable, listed so the intent is documented rather than
# implied by omission.
FATAL = (STTAuthError, STTBadAudio)


class CircuitOpen(Exception):
    """The breaker is open; the call was not attempted."""


@dataclass
class CircuitBreaker:
    """Per-dependency breaker.

    `failure_threshold` consecutive failures opens it for `reset_after`
    seconds. The first call after that window is allowed through as a probe;
    if it succeeds the breaker closes, if it fails the window restarts.
    """

    name: str
    failure_threshold: int = 3
    reset_after: float = 20.0
    _failures: int = 0
    _opened_at: float = 0.0

    @property
    def is_open(self) -> bool:
        if self._failures < self.failure_threshold:
            return False
        if time.monotonic() - self._opened_at >= self.reset_after:
            return False  # half-open: allow one probe through
        return True

    def record_success(self) -> None:
        self._failures = 0
        self._opened_at = 0.0

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self.failure_threshold:
            self._opened_at = time.monotonic()

    def check(self) -> None:
        if self.is_open:
            remaining = self.reset_after - (time.monotonic() - self._opened_at)
            raise CircuitOpen(f"{self.name} circuit open for another {remaining:.1f}s")


@dataclass
class RetryPolicy:
    max_attempts: int = 3
    base_delay: float = 0.25
    max_delay: float = 2.0
    # Jitter prevents a burst of simultaneous clients from retrying in lockstep
    # and re-hammering a service that is already struggling.
    jitter: float = 0.3

    def delay_for(self, attempt: int, retry_after: float | None = None) -> float:
        if retry_after is not None:
            return min(retry_after, self.max_delay)
        raw = min(self.base_delay * (2 ** (attempt - 1)), self.max_delay)
        return raw * (1 + random.uniform(-self.jitter, self.jitter))


@dataclass
class Attempt:
    n: int = 0
    errors: list[str] = field(default_factory=list)


async def run_with_policy(
    fn: Callable[[], Awaitable[T]],
    policy: RetryPolicy,
    timeout: float,
    breaker: CircuitBreaker | None = None,
    attempt_log: Attempt | None = None,
) -> T:
    """Execute `fn` under timeout, retry and circuit-breaker policy."""
    log = attempt_log or Attempt()
    last: Exception | None = None

    for attempt in range(1, policy.max_attempts + 1):
        log.n = attempt
        if breaker is not None:
            breaker.check()  # raises CircuitOpen without consuming an attempt
        try:
            result = await asyncio.wait_for(fn(), timeout=timeout)
            if breaker is not None:
                breaker.record_success()
            return result
        except FATAL as exc:
            # Record against the breaker? No: a caller sending bad audio is not
            # evidence the provider is unhealthy, and counting it would trip
            # the breaker on user error.
            log.errors.append(f"{type(exc).__name__}: {exc}")
            raise
        except RETRYABLE as exc:
            last = exc
            log.errors.append(f"{type(exc).__name__}: {exc}")
            if breaker is not None:
                breaker.record_failure()
            if attempt < policy.max_attempts:
                retry_after = getattr(exc, "retry_after", None)
                await asyncio.sleep(policy.delay_for(attempt, retry_after))
        except Exception as exc:
            last = exc
            log.errors.append(f"{type(exc).__name__}: {exc}")
            if breaker is not None:
                breaker.record_failure()
            raise

    assert last is not None
    raise last
