"""Semantic chunking — cut where the meaning actually changes.

Every strategy in `strategies.py` decides boundaries from surface form: token
counts, punctuation, structure. This one decides from meaning. It embeds each
sentence, walks the sequence measuring cosine distance between neighbours, and
cuts at the distance peaks — the points where the passage changes subject.

The distance threshold is a *percentile of this passage's own* distances, not a
global constant. A tightly-focused passage has uniformly low neighbour
distances and should be cut rarely; a passage that lists ten unrelated facts
has uniformly high ones and should be cut often. A fixed threshold gets both
cases wrong.

Cost is why this is viable at all: with static embeddings, sentence encoding is
a hash lookup and an average rather than a transformer forward pass, so
semantic chunking over the whole corpus stays within the build budget.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .base import Chunk, ChunkMeta, make_chunk
from .strategies import _WORD_SPAN, _sentence_spans

Encoder = Callable[[list[str]], np.ndarray]


@dataclass(slots=True)
class SemanticChunker:
    """Embedding-boundary splitting with a per-passage adaptive threshold."""

    encode: Encoder
    # Cut at neighbour distances above this percentile of the passage's own
    # distance distribution.
    percentile: float = 78.0
    # A cut is only honoured if the distance also clears this floor, so a
    # uniformly coherent passage is not sliced just because some boundary
    # happens to be its local maximum.
    min_distance: float = 0.08
    max_tokens: int = 180
    min_tokens: int = 20
    # Smooth distances over a ±1 sentence window. A single odd sentence
    # shouldn't trigger a cut; a genuine topic shift moves several in a row.
    smooth: bool = True
    name: str = "semantic"
    _buf: list = field(default_factory=list, repr=False)

    def split(self, text: str, meta: ChunkMeta) -> list[Chunk]:
        spans = _sentence_spans(text)
        if len(spans) < 2:
            body = text.strip()
            if len(_WORD_SPAN.findall(body)) < self.min_tokens:
                return []
            return [make_chunk(body, body, meta, self.name, 0, len(text))]

        sentences = [s[2] for s in spans]
        vecs = self.encode(sentences)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vecs = vecs / norms

        # Cosine distance between consecutive sentences.
        dists = 1.0 - np.sum(vecs[:-1] * vecs[1:], axis=1)

        if self.smooth and len(dists) >= 3:
            kernel = np.array([0.25, 0.5, 0.25])
            padded = np.pad(dists, 1, mode="edge")
            dists = np.convolve(padded, kernel, mode="valid")

        threshold = max(float(np.percentile(dists, self.percentile)), self.min_distance)

        # Walk sentences, cutting at boundaries that clear the threshold or
        # when the buffer would overflow the token ceiling.
        out: list[Chunk] = []
        start_idx = 0
        buf_tokens = len(_WORD_SPAN.findall(sentences[0]))

        for i in range(len(dists)):
            next_tokens = len(_WORD_SPAN.findall(sentences[i + 1]))
            over_budget = buf_tokens + next_tokens > self.max_tokens
            topic_shift = dists[i] >= threshold and buf_tokens >= self.min_tokens

            if over_budget or topic_shift:
                chunk = self._emit(text, spans, start_idx, i, meta)
                if chunk:
                    out.append(chunk)
                start_idx = i + 1
                buf_tokens = next_tokens
            else:
                buf_tokens += next_tokens

        tail = self._emit(text, spans, start_idx, len(spans) - 1, meta)
        if tail:
            out.append(tail)
        return out

    def _emit(
        self,
        text: str,
        spans: list[tuple[int, int, str]],
        lo: int,
        hi: int,
        meta: ChunkMeta,
    ) -> Chunk | None:
        start, end = spans[lo][0], spans[hi][1]
        body = text[start:end].strip()
        if len(_WORD_SPAN.findall(body)) < self.min_tokens:
            return None
        return make_chunk(body, body, meta, self.name, start, end)
