"""ElevenLabs Scribe STT.

The task allows either Sarvam or ElevenLabs. Sarvam is the configured default
because the corpus is Indic; Scribe is kept as a live second provider so the
harness has something real to fail over to when Sarvam is down or throttled,
rather than a fallback that exists only on paper.

Note the different auth header (`xi-api-key`) and the different response shape:
Scribe returns `language_probability` consistently, where Sarvam only returns
it on auto-detect.
"""

from __future__ import annotations

import time

import httpx

from ..config import Settings, settings
from .base import (
    MIN_AUDIO_BYTES,
    AudioFormat,
    STTAuthError,
    STTBadAudio,
    STTError,
    STTRateLimited,
    STTTransient,
    Transcript,
)

ENDPOINT = "https://api.elevenlabs.io/v1/speech-to-text"

# Scribe expects ISO-639-3; MSMARCO-XI shard prefixes already are, mostly.
LANG_MAP = {
    "hin": "hin", "ben": "ben", "tam": "tam", "tel": "tel", "kan": "kan",
    "mal": "mal", "mar": "mar", "guj": "guj", "pan": "pan", "ori": "ori",
    "asm": "asm", "urd": "urd", "nep": "nep", "san": "san", "eng": "eng",
}

_MIME = {
    AudioFormat.WAV: "audio/wav",
    AudioFormat.MP3: "audio/mpeg",
    AudioFormat.WEBM: "audio/webm",
    AudioFormat.OGG: "audio/ogg",
    AudioFormat.FLAC: "audio/flac",
}


class ElevenLabsTranscriber:
    name = "elevenlabs"

    def __init__(self, cfg: Settings = settings, client: httpx.AsyncClient | None = None) -> None:
        self.cfg = cfg
        self._client = client or httpx.AsyncClient(timeout=cfg.stt_timeout_s)
        self._owns_client = client is None

    async def transcribe(
        self,
        audio: bytes,
        fmt: AudioFormat = AudioFormat.WAV,
        language: str | None = None,
    ) -> Transcript:
        if not self.cfg.elevenlabs_api_key:
            raise STTAuthError("ELEVENLABS_API_KEY is not set")
        if len(audio) < MIN_AUDIO_BYTES:
            raise STTBadAudio(f"audio too small ({len(audio)} bytes)")

        data = {"model_id": self.cfg.elevenlabs_model}
        code = LANG_MAP.get(language or "")
        if code:
            data["language_code"] = code

        started = time.perf_counter()
        try:
            response = await self._client.post(
                ENDPOINT,
                headers={"xi-api-key": self.cfg.elevenlabs_api_key},
                files={"file": (f"audio.{fmt.value}", audio, _MIME.get(fmt, "audio/wav"))},
                data=data,
            )
        except httpx.TimeoutException as exc:
            raise STTTransient(f"elevenlabs timeout after {self.cfg.stt_timeout_s}s") from exc
        except httpx.HTTPError as exc:
            raise STTTransient(f"elevenlabs transport error: {exc}") from exc

        elapsed = (time.perf_counter() - started) * 1000
        self._raise_for_status(response)

        body = response.json()
        text = (body.get("text") or "").strip()
        if not text:
            raise STTBadAudio("no speech detected in audio")

        return Transcript(
            text=text,
            language=body.get("language_code"),
            confidence=body.get("language_probability"),
            provider=self.name,
            duration_ms=elapsed,
        )

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        if response.status_code == 200:
            return
        try:
            detail = response.json().get("detail", {})
            message = (
                detail.get("message") if isinstance(detail, dict) else str(detail)
            ) or response.text[:200]
        except Exception:
            message = response.text[:200]

        if response.status_code in (401, 403):
            raise STTAuthError(f"elevenlabs auth failed: {message}")
        if response.status_code == 429:
            retry_after = response.headers.get("retry-after")
            raise STTRateLimited(
                f"elevenlabs rate limited: {message}",
                float(retry_after) if retry_after else None,
            )
        if response.status_code in (400, 422):
            raise STTBadAudio(f"elevenlabs rejected audio: {message}")
        if response.status_code >= 500:
            raise STTTransient(f"elevenlabs {response.status_code}: {message}")
        raise STTError(f"elevenlabs {response.status_code}: {message}")

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
