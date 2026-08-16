"""STT provider selection with failover."""

from __future__ import annotations

from ..config import Settings, settings
from .base import (
    AudioFormat,
    STTAuthError,
    STTBadAudio,
    STTError,
    STTRateLimited,
    STTTransient,
    Transcriber,
    Transcript,
)
from .elevenlabs import ElevenLabsTranscriber
from .sarvam import SarvamTranscriber

__all__ = [
    "AudioFormat",
    "ElevenLabsTranscriber",
    "STTAuthError",
    "STTBadAudio",
    "STTError",
    "STTRateLimited",
    "STTTransient",
    "SarvamTranscriber",
    "Transcriber",
    "Transcript",
    "build_transcribers",
]

_PROVIDERS = {"sarvam": SarvamTranscriber, "elevenlabs": ElevenLabsTranscriber}


def build_transcribers(cfg: Settings = settings) -> list[Transcriber]:
    """Configured provider first, then any other provider that has a key.

    Returning a list rather than one transcriber is what lets the harness fail
    over. A provider without credentials is omitted entirely — an unusable
    fallback in the chain would just burn a retry budget on a guaranteed
    STTAuthError.
    """
    keys = {"sarvam": cfg.sarvam_api_key, "elevenlabs": cfg.elevenlabs_api_key}
    ordered = [cfg.stt_provider] + [p for p in _PROVIDERS if p != cfg.stt_provider]
    return [_PROVIDERS[name](cfg) for name in ordered if name in _PROVIDERS and keys.get(name)]
