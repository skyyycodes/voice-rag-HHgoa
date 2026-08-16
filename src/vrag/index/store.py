"""Chunk persistence.

Chunks are held as parallel numpy/list columns rather than a list of dataclass
objects. With ~1M chunks, a list of Python objects costs seconds to unpickle
and hundreds of MB of interpreter overhead; columnar arrays memory-map in
milliseconds. Retrieval only ever touches a handful of rows by integer index,
so a columnar layout is also what the access pattern wants.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc

from ..chunking.base import Chunk


@dataclass(slots=True)
class ChunkStore:
    text: list[str]  # embedded/matched surface
    context: list[str]  # what the answer stage reads
    passage_id: np.ndarray
    lang: np.ndarray
    query_type: np.ndarray
    strategies: list[frozenset]
    is_selected: np.ndarray
    n_strategies: np.ndarray  # denormalised for the reranker's hot path

    @classmethod
    def from_chunks(cls, chunks: Sequence[Chunk]) -> "ChunkStore":
        return cls(
            text=[c.text for c in chunks],
            context=[c.context for c in chunks],
            passage_id=np.array([c.passage_id for c in chunks], dtype=object),
            lang=np.array([c.lang for c in chunks], dtype=object),
            query_type=np.array([c.query_type for c in chunks], dtype=object),
            strategies=[frozenset(c.strategies) for c in chunks],
            is_selected=np.array([c.is_selected for c in chunks], dtype=bool),
            n_strategies=np.array([len(c.strategies) for c in chunks], dtype=np.int8),
        )

    def __len__(self) -> int:
        return len(self.text)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.table(
            {
                "text": pa.array(self.text, type=pa.large_string()),
                "context": pa.array(self.context, type=pa.large_string()),
                "passage_id": pa.array([str(x) for x in self.passage_id]),
                "lang": pa.array([str(x) for x in self.lang]),
                "query_type": pa.array([str(x) for x in self.query_type]),
                # Sorted join keeps the round-trip stable and diffable.
                "strategies": pa.array(["|".join(sorted(s)) for s in self.strategies]),
                "is_selected": pa.array(self.is_selected),
            }
        )
        with ipc.new_file(path, table.schema) as writer:
            writer.write_table(table)

    @classmethod
    def load(cls, path: Path) -> "ChunkStore":
        with ipc.open_file(path) as reader:
            table = reader.read_all()
        strategies = [frozenset(s.split("|")) for s in table["strategies"].to_pylist()]
        return cls(
            text=table["text"].to_pylist(),
            context=table["context"].to_pylist(),
            passage_id=np.array(table["passage_id"].to_pylist(), dtype=object),
            lang=np.array(table["lang"].to_pylist(), dtype=object),
            query_type=np.array(table["query_type"].to_pylist(), dtype=object),
            strategies=strategies,
            is_selected=np.array(table["is_selected"].to_pylist(), dtype=bool),
            n_strategies=np.array([len(s) for s in strategies], dtype=np.int8),
        )
