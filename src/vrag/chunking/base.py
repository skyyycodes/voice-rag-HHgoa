"""Chunk primitives and multilingual segmentation.

MSMARCO-XI spans 14 Indic scripts, so segmentation cannot assume a Latin
full stop. Devanagari/Bengali/Gujarati use the danda (।) and double danda (॥),
Urdu uses the Arabic question mark (؟) and full stop (۔). We segment on the
union, and we never split inside a decimal, an abbreviation, or a digit group
— MS MARCO is full of NUMERIC queries where "3.5 million" must survive intact.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Protocol

# Sentence terminators across the scripts present in MSMARCO-XI.
_TERMINATORS = "।॥؟۔?!."
# Split after a terminator followed by whitespace, but not after a single
# capital letter (U.S.A.) or a common abbreviation.
#
# There is deliberately no decimal guard here. Requiring whitespace after the
# terminator already protects "3.5" — the period in a decimal is never followed
# by a space. An explicit `(?<!\d[.])` guard looks like it protects decimals but
# actually blocks every sentence ending in a number, so "...founded in 1867. She
# won..." silently stopped splitting. MS MARCO is full of NUMERIC passages, so
# that failure was both common and invisible.
_ABBREV = r"(?<!\b[A-Z])(?<!\bMr)(?<!\bMrs)(?<!\bDr)(?<!\bSt)(?<!\bNo)(?<!\bvs)(?<!\bInc)(?<!\bJr)"
_SENT_SPLIT = re.compile(rf"(?<=[{_TERMINATORS}])" + _ABBREV + r"\s+")

_WORD = re.compile(r"\w+", re.UNICODE)


def split_sentences(text: str) -> list[str]:
    """Script-aware sentence segmentation."""
    parts = [s.strip() for s in _SENT_SPLIT.split(text) if s and s.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def tokenize(text: str) -> list[str]:
    """Unicode word tokens. Used for length accounting and lexical overlap.

    Deliberately not a BPE tokenizer: this runs over millions of chunks at
    build time and a regex is ~50x faster, while "how many words" is the
    property chunk sizing actually cares about.
    """
    return _WORD.findall(text.lower())


def token_len(text: str) -> int:
    return len(_WORD.findall(text))


def content_hash(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=8).hexdigest()


@dataclass(slots=True)
class Chunk:
    """A retrievable unit.

    `text` is what gets embedded and matched. `context` is what gets handed to
    answer generation — for the sentence-window and parent-child strategies
    these deliberately differ: match narrow, read wide.
    """

    chunk_id: str
    text: str
    context: str
    passage_id: str
    lang: str
    query_type: str
    strategy: str
    # Character span of `text` inside the source passage. Lets the extractive
    # answerer point at an exact offset for citation highlighting.
    start: int
    end: int
    is_selected: bool = False
    strategies: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        if not self.strategies:
            self.strategies = {self.strategy}


class Chunker(Protocol):
    """Every strategy implements this. The registry fans out across them."""

    name: str

    def split(self, text: str, meta: ChunkMeta) -> list[Chunk]: ...


@dataclass(slots=True)
class ChunkMeta:
    passage_id: str
    lang: str
    query_type: str
    is_selected: bool = False


def make_chunk(
    text: str,
    context: str,
    meta: ChunkMeta,
    strategy: str,
    start: int,
    end: int,
) -> Chunk:
    return Chunk(
        chunk_id=f"{strategy[:2]}{content_hash(text + strategy)}",
        text=text,
        context=context,
        passage_id=meta.passage_id,
        lang=meta.lang,
        query_type=meta.query_type,
        strategy=strategy,
        start=start,
        end=end,
        is_selected=meta.is_selected,
    )
