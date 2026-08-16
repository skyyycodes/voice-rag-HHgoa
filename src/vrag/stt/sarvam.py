"""Sarvam Saaras STT.

Contract verified against the live API rather than transcribed from docs — the
published model list and the response body both differ from what the reference
page implies. Observed behaviour:

    200  {"request_id", "transcript", "language_code", "language_probability"?}
    400  {"error": {"message", "code": "invalid_request_error", "request_id"}}
    403  {"error": {"message", "code": "invalid_api_key_error", "request_id"}}

`language_probability` is present on auto-detect but absent when an explicit
`language_code` is supplied, so downstream code must treat missing confidence
as "unknown", never as zero.

Chosen for this build because MSMARCO-XI is 14 Indic languages and Saaras is
trained on exactly that set; `language_code=unknown` auto-detects, which is
what the demo uses so a speaker can switch languages without touching the UI.
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

ENDPOINT = "https://api.sarvam.ai/speech-to-text"

# MSMARCO-XI shard prefix -> Sarvam BCP-47 code.
LANG_MAP = {
    "hin": "hi-IN", "ben": "bn-IN", "tam": "ta-IN", "tel": "te-IN",
    "kan": "kn-IN", "mal": "ml-IN", "mar": "mr-IN", "guj": "gu-IN",
    "pan": "pa-IN", "ori": "od-IN", "asm": "as-IN", "urd": "ur-IN",
    "nep": "ne-IN", "san": "sa-IN", "eng": "en-IN",
}

_MIME = {
    AudioFormat.WAV: "audio/wav",
    AudioFormat.MP3: "audio/mpeg",
    AudioFormat.WEBM: "audio/webm",
    AudioFormat.OGG: "audio/ogg",
    AudioFormat.FLAC: "audio/flac",
}


class SarvamTranscriber:
    name = "sarvam"

    def __init__(self, cfg: Settings = settings, client: httpx.AsyncClient | None = None) -> None:
        self.cfg = cfg
        # Reusing one client keeps the TLS session warm. A fresh connection per
        # request adds a full handshake — around 100ms — to every utterance.
        self._client = client or httpx.AsyncClient(timeout=cfg.stt_timeout_s)
        self._owns_client = client is None

    async def transcribe(
        self,
        audio: bytes,
        fmt: AudioFormat = AudioFormat.WAV,
        language: str | None = None,
    ) -> Transcript:
        if not self.cfg.sarvam_api_key:
            raise STTAuthError("SARVAM_API_KEY is not set")
        if len(audio) < MIN_AUDIO_BYTES:
            # Browsers emit a header-only blob on an accidental mic tap.
            raise STTBadAudio(f"audio too small ({len(audio)} bytes)")

        code = LANG_MAP.get(language or "", language) or "unknown"
        started = time.perf_counter()
        try:
            response = await self._client.post(
                ENDPOINT,
                headers={"api-subscription-key": self.cfg.sarvam_api_key},
                files={"file": (f"audio.{fmt.value}", audio, _MIME.get(fmt, "audio/wav"))},
                data={"model": self.cfg.sarvam_model, "language_code": code},
            )
        except httpx.TimeoutException as exc:
            raise STTTransient(f"sarvam timeout after {self.cfg.stt_timeout_s}s") from exc
        except httpx.HTTPError as exc:
            raise STTTransient(f"sarvam transport error: {exc}") from exc

        elapsed = (time.perf_counter() - started) * 1000
        self._raise_for_status(response)

        body = response.json()
        text = (body.get("transcript") or "").strip()
        if not text:
            # A 200 with an empty transcript means silence or noise. That is a
            # user-facing outcome, not an error to retry.
            raise STTBadAudio("no speech detected in audio")

        return Transcript(
            text=text,
            language=body.get("language_code"),
            confidence=body.get("language_probability"),
            provider=self.name,
            duration_ms=elapsed,
            request_id=body.get("request_id"),
        )

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        if response.status_code == 200:
            return
        try:
            detail = response.json().get("error", {})
            message = detail.get("message", response.text[:200])
            code = detail.get("code", "")
        except Exception:
            message, code = response.text[:200], ""

        if response.status_code in (401, 403) or code == "invalid_api_key_error":
            raise STTAuthError(f"sarvam auth failed: {message}")
        if response.status_code == 429:
            retry_after = response.headers.get("retry-after")
            raise STTRateLimited(
                f"sarvam rate limited: {message}",
                float(retry_after) if retry_after else None,
            )
        if response.status_code == 400:
            raise STTBadAudio(f"sarvam rejected audio: {message}")
        if response.status_code >= 500:
            raise STTTransient(f"sarvam {response.status_code}: {message}")
        raise STTError(f"sarvam {response.status_code}: {message}")

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
