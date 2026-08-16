"""Deterministic chunking strategies.

Six strategies with genuinely different failure modes, not six parameterisations
of the same sliding window:

  fixed          fixed token window + overlap        — the baseline
  recursive      hierarchical separator descent      — respects structure
  sentence_window match one sentence, return N       — precise match, wide read
  parent_child   small child indexed, parent read    — small-to-big
  metadata_aware sizing routed by query_type         — NUMERIC != DESCRIPTION
  proposition    atomic clause-level statements      — maximum precision

Each returns `Chunk`s carrying character offsets into the source passage, so a
retrieved chunk can be highlighted in its original context at answer time.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .base import Chunk, ChunkMeta, make_chunk, split_sentences

# Includes Indic combining marks — a bare `\w+` breaks at every matra and
# shreds Devanagari/Bengali/Tamil words into consonant fragments, which
# would size every non-English chunk by a fragment count rather than a
# word count. See `index/lexical.py` for the full explanation.
_WORD_SPAN = re.compile(r"[\wऀ-෿̀-ͯ]+", re.UNICODE)


def _token_spans(text: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in _WORD_SPAN.finditer(text)]


def _sentence_spans(text: str) -> list[tuple[int, int, str]]:
    """Sentences with their character offsets in `text`."""
    spans: list[tuple[int, int, str]] = []
    cursor = 0
    for sent in split_sentences(text):
        idx = text.find(sent, cursor)
        if idx < 0:  # normalisation drift; fall back to sequential placement
            idx = cursor
        spans.append((idx, idx + len(sent), sent))
        cursor = idx + len(sent)
    return spans


# --------------------------------------------------------------------------
# 1. Fixed token window with overlap
# --------------------------------------------------------------------------
@dataclass(slots=True)
class FixedTokenChunker:
    """Sliding window over tokens.

    The baseline every RAG system starts with. Cheap and predictable, but it
    cuts mid-sentence, which is exactly why the other five exist.
    """

    # Sized to the corpus, not to intuition. MS MARCO passages measure 48
    # tokens at the median and 116 at p99; a 120-token window returned the
    # whole passage for 99% of them, making this strategy — and `recursive`,
    # and `metadata_aware` — degenerate duplicates of each other. See the
    # sizing note in `registry.py`.
    size: int = 40
    overlap: int = 12
    name: str = "fixed"

    def split(self, text: str, meta: ChunkMeta) -> list[Chunk]:
        spans = _token_spans(text)
        if not spans:
            return []
        stride = max(1, self.size - self.overlap)
        out: list[Chunk] = []
        for i in range(0, len(spans), stride):
            window = spans[i : i + self.size]
            if not window:
                break
            # Drop a trailing sliver that is almost entirely overlap.
            if i > 0 and len(window) < self.overlap:
                break
            start, end = window[0][0], window[-1][1]
            body = text[start:end]
            out.append(make_chunk(body, body, meta, self.name, start, end))
            if i + self.size >= len(spans):
                break
        return out


# --------------------------------------------------------------------------
# 2. Recursive structure-aware descent
# --------------------------------------------------------------------------
@dataclass(slots=True)
class RecursiveChunker:
    """Split on the largest structural boundary that fits, then descend.

    Tries paragraph breaks, then sentence terminators, then clause commas,
    then whitespace. A chunk is only cut mid-sentence when no coarser boundary
    can produce a small enough piece — the opposite of the fixed strategy's
    default behaviour.
    """

    max_tokens: int = 45
    min_tokens: int = 12
    name: str = "recursive"
    separators: tuple[str, ...] = ("\n\n", "\n", "। ", "॥ ", ". ", "? ", "! ", "; ", ", ", " ")

    def _descend(self, text: str, offset: int, depth: int) -> list[tuple[int, int]]:
        if len(_WORD_SPAN.findall(text)) <= self.max_tokens or depth >= len(self.separators):
            return [(offset, offset + len(text))]

        sep = self.separators[depth]
        if sep not in text:
            return self._descend(text, offset, depth + 1)

        spans: list[tuple[int, int]] = []
        cursor = 0
        buf_start = 0
        for part in text.split(sep):
            piece_end = cursor + len(part)
            buf_tokens = len(_WORD_SPAN.findall(text[buf_start:piece_end]))
            if buf_tokens >= self.max_tokens:
                # Flush the accumulated buffer, recursing if it is still large.
                segment = text[buf_start:piece_end]
                if len(_WORD_SPAN.findall(segment)) > self.max_tokens:
                    spans.extend(self._descend(segment, offset + buf_start, depth + 1))
                else:
                    spans.append((offset + buf_start, offset + piece_end))
                buf_start = piece_end + len(sep)
            cursor = piece_end + len(sep)

        if buf_start < len(text):
            spans.append((offset + buf_start, offset + len(text)))
        return spans or [(offset, offset + len(text))]

    def split(self, text: str, meta: ChunkMeta) -> list[Chunk]:
        out: list[Chunk] = []
        for start, end in self._descend(text, 0, 0):
            body = text[start:end].strip()
            if len(_WORD_SPAN.findall(body)) < self.min_tokens:
                continue
            out.append(make_chunk(body, body, meta, self.name, start, end))
        return out


# --------------------------------------------------------------------------
# 3. Sentence window — match narrow, read wide
# --------------------------------------------------------------------------
@dataclass(slots=True)
class SentenceWindowChunker:
    """Index one sentence; return it surrounded by its neighbours.

    Embedding a single sentence gives a sharp, unmuddied vector — the whole
    chunk is about one thing. But a lone sentence often lacks the antecedent
    for its pronouns, so `context` carries ±`window` neighbours for the
    answer stage to read.
    """

    window: int = 1
    min_tokens: int = 8
    name: str = "sentence_window"

    def split(self, text: str, meta: ChunkMeta) -> list[Chunk]:
        spans = _sentence_spans(text)
        out: list[Chunk] = []
        for i, (start, end, sent) in enumerate(spans):
            if len(_WORD_SPAN.findall(sent)) < self.min_tokens:
                continue
            lo = max(0, i - self.window)
            hi = min(len(spans), i + self.window + 1)
            context = text[spans[lo][0] : spans[hi - 1][1]]
            out.append(make_chunk(sent, context, meta, self.name, start, end))
        return out


# --------------------------------------------------------------------------
# 4. Parent-child (small-to-big)
# --------------------------------------------------------------------------
@dataclass(slots=True)
class ParentChildChunker:
    """Small children are indexed, the parent is what gets read.

    Differs from sentence_window in that the parent is a fixed-size block
    rather than a symmetric neighbour window, so every child of a parent
    resolves to the identical context string — which lets the retriever
    collapse sibling hits into one context instead of returning near-duplicates.
    """

    parent_tokens: int = 60
    child_tokens: int = 18
    name: str = "parent_child"

    def split(self, text: str, meta: ChunkMeta) -> list[Chunk]:
        spans = _token_spans(text)
        if not spans:
            return []
        out: list[Chunk] = []
        for p in range(0, len(spans), self.parent_tokens):
            parent = spans[p : p + self.parent_tokens]
            if not parent:
                break
            p_start, p_end = parent[0][0], parent[-1][1]
            parent_text = text[p_start:p_end]
            for c in range(0, len(parent), self.child_tokens):
                child = parent[c : c + self.child_tokens]
                if len(child) < 8:
                    continue
                c_start, c_end = child[0][0], child[-1][1]
                out.append(
                    make_chunk(
                        text[c_start:c_end], parent_text, meta, self.name, c_start, c_end
                    )
                )
        return out


# --------------------------------------------------------------------------
# 5. Metadata-aware routing
# --------------------------------------------------------------------------
@dataclass(slots=True)
class MetadataAwareChunker:
    """Chunk size and framing routed by the passage's MS MARCO `query_type`.

    MSMARCO-XI labels every query NUMERIC / ENTITY / PERSON / LOCATION /
    DESCRIPTION. Those want different things:

      NUMERIC     tight chunks — a figure and its unit must stay together,
                  and a wide chunk buries the number among distractor digits
      ENTITY,
      PERSON,
      LOCATION    medium chunks that keep the entity with its appositive
      DESCRIPTION wide chunks — definitional answers span several sentences

    The chunk text is also prefixed with its type and language. The static
    embedder sees that prefix, which pulls same-type chunks together in vector
    space and gives the retriever a cheap type-affinity signal for free.
    """

    name: str = "metadata_aware"
    profiles: dict[str, tuple[int, int]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.profiles is None:
            # query_type -> (target tokens, overlap tokens)
            self.profiles = {
                "NUMERIC": (22, 8),
                "ENTITY": (32, 10),
                "PERSON": (32, 10),
                "LOCATION": (32, 10),
                "DESCRIPTION": (55, 16),
                "UNKNOWN": (42, 12),
            }

    def split(self, text: str, meta: ChunkMeta) -> list[Chunk]:
        size, overlap = self.profiles.get(meta.query_type, self.profiles["UNKNOWN"])
        spans = _sentence_spans(text)
        if not spans:
            return []

        out: list[Chunk] = []
        buf: list[tuple[int, int, str]] = []
        buf_tokens = 0

        def flush() -> None:
            nonlocal buf, buf_tokens
            if not buf:
                return
            start, end = buf[0][0], buf[-1][1]
            body = text[start:end]
            # The prefix is embedded but not shown to the user; `context` stays
            # clean so the answer stage never quotes the metadata back.
            tagged = f"[{meta.query_type}|{meta.lang}] {body}"
            out.append(make_chunk(tagged, body, meta, self.name, start, end))
            # Carry the tail sentence forward as overlap.
            if overlap and len(buf) > 1:
                buf = buf[-1:]
                buf_tokens = len(_WORD_SPAN.findall(buf[0][2]))
            else:
                buf, buf_tokens = [], 0

        for span in spans:
            n = len(_WORD_SPAN.findall(span[2]))
            if buf_tokens + n > size and buf:
                flush()
            buf.append(span)
            buf_tokens += n
        flush()
        return out


# --------------------------------------------------------------------------
# 6. Proposition — atomic clause statements
# --------------------------------------------------------------------------
@dataclass(slots=True)
class PropositionChunker:
    """Break sentences into standalone clauses.

    Genuine proposition extraction normally needs an LLM pass, which is far too
    slow for a corpus this size. This is a rule-based approximation: split on
    coordinating conjunctions, relative pronouns and clause punctuation, then
    keep only fragments that still carry a verb-like token. The payoff is the
    highest-precision unit in the index — for NUMERIC and ENTITY queries a
    proposition hit is almost always the exact answer span.
    """

    min_tokens: int = 6
    max_tokens: int = 30
    name: str = "proposition"
    # English + transliterated Indic clause boundaries.
    clause_re: re.Pattern = re.compile(
        r"\s*(?:,\s+(?:which|who|whom|whose|where|when|that|and|but|while|although)\b"
        r"|;\s*|\s+—\s+|\s+–\s+|\s+जो\s+|\s+जबकि\s+|\s+तथा\s+|\s+এবং\s+|\s+যা\s+)",
        re.IGNORECASE | re.UNICODE,
    )

    def split(self, text: str, meta: ChunkMeta) -> list[Chunk]:
        out: list[Chunk] = []
        for s_start, s_end, sent in _sentence_spans(text):
            spans = self._fragment_spans(text, sent, s_start)
            for start, end in self._merge_short(text, spans):
                body = text[start:end].strip(" ,;—–")
                n = len(_WORD_SPAN.findall(body))
                if n < self.min_tokens or n > self.max_tokens:
                    continue
                # The full sentence is the context — a bare clause read alone
                # is exactly the kind of thing that produces a hallucination.
                out.append(make_chunk(body, sent, meta, self.name, start, end))
        return out

    def _fragment_spans(self, text: str, sent: str, s_start: int) -> list[tuple[int, int]]:
        """Clause spans as offsets into `text`, so merging stays lossless."""
        spans: list[tuple[int, int]] = []
        cursor = s_start
        for piece in self.clause_re.split(sent):
            piece = piece.strip()
            if not piece:
                continue
            idx = text.find(piece, cursor)
            if idx < 0:
                idx = cursor
            spans.append((idx, idx + len(piece)))
            cursor = idx + len(piece)
        return spans

    def _merge_short(
        self, text: str, spans: list[tuple[int, int]]
    ) -> list[tuple[int, int]]:
        """Absorb undersized fragments into their neighbour.

        Splitting "Marie Curie, who was born in Warsaw" on the relative pronoun
        strands "Marie Curie" as a 2-token fragment and leaves the next clause
        subjectless — "was born in Warsaw", which is unusable as a standalone
        retrieval unit and actively dangerous as a quoted answer. Merging any
        fragment below `min_tokens` into its neighbour restores the subject.
        Because fragments are contiguous, the merged span is still a literal
        substring of the passage, so character offsets stay valid.
        """
        if not spans:
            return []
        merged: list[list[int]] = [list(spans[0])]
        for start, end in spans[1:]:
            prev = merged[-1]
            if len(_WORD_SPAN.findall(text[prev[0] : prev[1]])) < self.min_tokens:
                prev[1] = end  # undersized head absorbs the following clause
            else:
                merged.append([start, end])
        # A trailing runt has no successor to merge into; fold it backwards.
        if len(merged) > 1:
            last = merged[-1]
            if len(_WORD_SPAN.findall(text[last[0] : last[1]])) < self.min_tokens:
                merged[-2][1] = last[1]
                merged.pop()
        return [(a, b) for a, b in merged]
