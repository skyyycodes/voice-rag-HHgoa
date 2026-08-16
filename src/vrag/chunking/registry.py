"""Multi-strategy fan-out.

Running seven strategies over the same passage produces overlapping output —
a short passage yields near-identical text from `recursive`, `metadata_aware`
and `semantic`. Indexing all of it would inflate the index ~4x and, worse,
let one passage occupy every slot in the top-k because the same text was
retrieved seven times under seven ids.

So the registry deduplicates on normalised content. The surviving chunk records
*every* strategy that produced it in `strategies`, which turns duplication into
signal: text that several independent strategies agree is a coherent unit is
usually a good retrieval target, and `provenance_boost` exposes that as a
rankable feature.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np

from ..corpus import Passage
from .base import Chunk, ChunkMeta, content_hash
from .semantic import SemanticChunker
from .strategies import (
    FixedTokenChunker,
    MetadataAwareChunker,
    ParentChildChunker,
    PropositionChunker,
    RecursiveChunker,
    SentenceWindowChunker,
)

_DEDUP_NORM = re.compile(r"[^\w]+", re.UNICODE)


def _dedup_key(text: str) -> str:
    """Normalise away punctuation, case, and the metadata prefix so that
    cosmetically different renderings of the same span collapse together."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"^\[[A-Z_]+\|\w+\]\s*", "", text)
    return _DEDUP_NORM.sub("", text.lower())


def default_chunkers(encode: Callable[[list[str]], np.ndarray] | None = None) -> list:
    """The full strategy set.

    `encode` is optional so the deterministic strategies stay usable (and
    testable) without loading an embedding model.
    """
    chunkers: list = [
        FixedTokenChunker(size=120, overlap=30),
        RecursiveChunker(max_tokens=140),
        SentenceWindowChunker(window=1),
        ParentChildChunker(parent_tokens=200, child_tokens=45),
        MetadataAwareChunker(),
        PropositionChunker(),
    ]
    if encode is not None:
        chunkers.append(SemanticChunker(encode=encode))
    return chunkers


@dataclass(slots=True)
class ChunkStats:
    """Per-strategy accounting, reported by the build so the chunking layer is
    inspectable rather than a black box."""

    produced: Counter
    survived: Counter
    total_chunks: int
    total_passages: int

    def report(self) -> str:
        lines = [
            f"{'strategy':<16} {'produced':>9} {'unique':>9} {'dup%':>7}",
            "-" * 44,
        ]
        for name, produced in self.produced.most_common():
            kept = self.survived[name]
            dup = 100.0 * (1 - kept / produced) if produced else 0.0
            lines.append(f"{name:<16} {produced:>9,} {kept:>9,} {dup:>6.1f}%")
        lines.append("-" * 44)
        lines.append(
            f"{'TOTAL':<16} {sum(self.produced.values()):>9,} "
            f"{self.total_chunks:>9,} "
            f"{100.0 * (1 - self.total_chunks / max(1, sum(self.produced.values()))):>6.1f}%"
        )
        lines.append(
            f"\n{self.total_passages:,} passages -> {self.total_chunks:,} chunks "
            f"({self.total_chunks / max(1, self.total_passages):.1f} per passage)"
        )
        return "\n".join(lines)


def chunk_passages(
    passages: Sequence[Passage],
    chunkers: Iterable | None = None,
    encode: Callable[[list[str]], np.ndarray] | None = None,
) -> tuple[list[Chunk], ChunkStats]:
    """Fan every passage through every strategy, then deduplicate."""
    chunkers = list(chunkers) if chunkers is not None else default_chunkers(encode)

    # Let embedding-dependent chunkers batch their encoding across the whole
    # corpus before the per-passage loop starts. Encoding six sentences at a
    # time, once per passage, was the dominant cost of the entire build.
    for chunker in chunkers:
        prime = getattr(chunker, "prime", None)
        if callable(prime):
            prime(passages)

    produced: Counter = Counter()
    survived: Counter = Counter()
    by_key: dict[str, Chunk] = {}

    for passage in passages:
        meta = ChunkMeta(
            passage_id=passage.passage_id,
            lang=passage.lang,
            query_type=passage.query_type,
            is_selected=passage.is_selected,
        )
        for chunker in chunkers:
            try:
                chunks = chunker.split(passage.text, meta)
            except Exception:
                # A malformed passage must never abort a corpus-wide build.
                continue
            produced[chunker.name] += len(chunks)
            for chunk in chunks:
                key = _dedup_key(chunk.text)
                if len(key) < 12:
                    continue
                existing = by_key.get(key)
                if existing is None:
                    by_key[key] = chunk
                    survived[chunker.name] += 1
                else:
                    existing.strategies.add(chunk.strategy)
                    # Keep the richest available context. A chunk that one
                    # strategy emitted bare and another emitted with a parent
                    # window should retain the window.
                    if len(chunk.context) > len(existing.context):
                        existing.context = chunk.context

    chunks = list(by_key.values())
    stats = ChunkStats(
        produced=produced,
        survived=survived,
        total_chunks=len(chunks),
        total_passages=len(passages),
    )
    return chunks, stats


def provenance_boost(chunk: Chunk) -> float:
    """Multi-strategy agreement as a ranking feature.

    A span that only the fixed-window chunker produced is probably an artifact
    of where the window happened to land. A span that `semantic`, `recursive`
    and `sentence_window` independently chose is a real semantic unit. Scaled
    to a modest 1.0–1.15 so it breaks ties without overriding relevance.
    """
    return 1.0 + 0.05 * min(3, len(chunk.strategies) - 1)


def chunk_query_context(text: str, encode: Callable[[list[str]], np.ndarray] | None = None) -> list[str]:
    """Query-time re-chunking of a retrieved passage.

    Retrieval returns chunk `context`, which for parent-child and
    sentence-window strategies is deliberately wide. Before answer extraction
    we re-split that context into sentence units so the answerer can select a
    tight span rather than quoting the whole parent block. This is the one
    chunking operation that runs inside the latency budget, so it stays
    rule-based.
    """
    from .base import split_sentences

    return split_sentences(text)
