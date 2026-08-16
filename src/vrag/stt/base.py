"""Speech-to-text abstraction.

STT is the one stage that must call a third-party network service, which makes
it the least reliable thing in the pipeline and the reason the harness needs
real error handling rather than a try/except. Everything provider-specific is
confined behind `Transcriber` so the orchestrator only ever sees a
`Transcript` or a typed failure.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol


class STTError(Exception):
    """Base for transcription failures the harness knows how to handle."""


class STTAuthError(STTError):
    """Bad or missing credentials. Never worth retrying."""


class STTRateLimited(STTError):
    """Provider throttled us. Retry with backoff."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class STTTransient(STTError):
    """Timeout or 5xx. Retry."""


class STTBadAudio(STTError):
    """Empty, truncated, or unsupported audio. Retrying will not help."""


class AudioFormat(str, Enum):
    WAV = "wav"
    MP3 = "mp3"
    WEBM = "webm"
    OGG = "ogg"
    FLAC = "flac"


@dataclass(slots=True)
class Transcript:
    text: str
    language: str | None = None
    # Providers differ on whether they return confidence at all; None means
    # "not reported", which the guardrails treat differently from "low".
    confidence: float | None = None
    provider: str = ""
    duration_ms: float = 0.0
    request_id: str | None = None


class Transcriber(Protocol):
    name: str

    async def transcribe(
        self, audio: bytes, fmt: AudioFormat = AudioFormat.WAV, language: str | None = None
    ) -> Transcript: ...


# Minimum plausible payload for a real utterance. Browsers occasionally emit a
# header-only blob when the user taps the mic and releases immediately; sending
# that to a paid API wastes a call and returns an empty string anyway.
MIN_AUDIO_BYTES = 1024
