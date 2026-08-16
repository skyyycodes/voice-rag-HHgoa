"""MSMARCO-XI ingestion.

The dataset ships one parquet per Indic language, each a *single* ~1.2GB row
group, so we stream record batches with a projected column set rather than
calling `read_table` and pinning the whole thing in memory.

Each row is one query with ten candidate passages in both English and the
target language, plus `is_selected` relevance labels. We fan that out into
passage records and keep the labels, which gives us free retrieval ground
truth (recall@k / MRR) and gold answers for grounding evaluation.
"""

from __future__ import annotations

import hashlib
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import pyarrow.parquet as pq

from .config import Settings, settings

_COLUMNS = [
    "target_lang",
    "query_id",
    "query_type",
    "query",
    "Eng_Query",
    "Answer",
    "Eng_Answer",
    "passages",
]


@dataclass(slots=True)
class Passage:
    """One retrievable passage before chunking."""

    passage_id: str
    text: str
    lang: str
    query_id: int
    query_type: str
    is_selected: bool
    # Passage position within its original result list. Low rank correlates
    # with relevance in MS MARCO, so retrieval can use it as a mild prior.
    rank: int


@dataclass(slots=True)
class QueryRecord:
    """A held-out evaluation query with its gold answer and gold passages."""

    query_id: int
    lang: str
    query_type: str
    query: str  # in the target Indic language — what the user actually speaks
    eng_query: str
    answer: str
    eng_answer: str
    gold_passage_ids: list[str] = field(default_factory=list)


def _norm(text: str) -> str:
    """NFC-normalise and collapse whitespace.

    Indic scripts have multiple valid encodings for the same grapheme; without
    NFC the tokenizer and BM25 see different strings for identical text.
    """
    return " ".join(unicodedata.normalize("NFC", text).split())


def _passage_id(text: str) -> str:
    """Content-addressed id so the same passage across languages/queries
    collapses to one entry."""
    return hashlib.blake2b(text.encode("utf-8"), digest_size=8).hexdigest()


def shard_path(lang: str, cfg: Settings = settings) -> Path:
    return cfg.raw_dir / f"{lang}val.parquet"


def iter_rows(lang: str, limit: int, cfg: Settings = settings) -> Iterator[dict]:
    path = shard_path(lang, cfg)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing shard {path}. Run `python -m vrag.download {lang}` first."
        )
    seen = 0
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=512, columns=_COLUMNS):
        for row in batch.to_pylist():
            yield row
            seen += 1
            if seen >= limit:
                return


def load(
    cfg: Settings = settings,
) -> tuple[list[Passage], list[QueryRecord]]:
    """Return deduplicated passages plus the query records that reference them."""
    passages: dict[str, Passage] = {}
    queries: list[QueryRecord] = []

    for lang in cfg.languages:
        for row in iter_rows(lang, cfg.max_rows_per_lang, cfg):
            p = row["passages"] or {}
            translated = p.get("Translated_passages") or []
            english = p.get("English_passages") or []
            selected = p.get("is_selected") or []
            if not translated:
                continue

            gold: list[str] = []
            for rank, text in enumerate(translated):
                text = _norm(text or "")
                # Very short fragments are translation artifacts, not answers.
                if len(text) < 40:
                    continue
                pid = _passage_id(text)
                is_sel = bool(selected[rank]) if rank < len(selected) else False
                if is_sel:
                    gold.append(pid)
                if pid not in passages:
                    passages[pid] = Passage(
                        passage_id=pid,
                        text=text,
                        lang=lang,
                        query_id=row["query_id"],
                        query_type=(row["query_type"] or "UNKNOWN").upper(),
                        is_selected=is_sel,
                        rank=rank,
                    )
                elif is_sel:
                    # A passage selected for any query is a known-good answer
                    # span; keep the strongest label we have seen.
                    passages[pid].is_selected = True

            # Index the English source passages too. The static embedder is
            # multilingual, so an English passage stays reachable from a Hindi
            # query — this is what lets the system answer cross-lingually.
            for rank, text in enumerate(english):
                text = _norm(text or "")
                if len(text) < 40:
                    continue
                pid = _passage_id(text)
                is_sel = bool(selected[rank]) if rank < len(selected) else False
                if is_sel:
                    gold.append(pid)
                if pid not in passages:
                    passages[pid] = Passage(
                        passage_id=pid,
                        text=text,
                        lang="eng",
                        query_id=row["query_id"],
                        query_type=(row["query_type"] or "UNKNOWN").upper(),
                        is_selected=is_sel,
                        rank=rank,
                    )
                elif is_sel:
                    passages[pid].is_selected = True

            query = _norm(row["query"] or "")
            if not query or not gold:
                # No gold passage means the row is useless as an eval example.
                continue
            queries.append(
                QueryRecord(
                    query_id=row["query_id"],
                    lang=lang,
                    query_type=(row["query_type"] or "UNKNOWN").upper(),
                    query=query,
                    eng_query=_norm(row["Eng_Query"] or ""),
                    answer=_norm(row["Answer"] or ""),
                    eng_answer=_norm(row["Eng_Answer"] or ""),
                    gold_passage_ids=gold,
                )
            )

    return list(passages.values()), queries
