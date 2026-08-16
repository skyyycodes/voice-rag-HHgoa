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

# Chunk text compresses ~3x with zstd and decompresses fast enough that load
# time still beats reading the uncompressed file off disk.
_IPC_OPTS = ipc.IpcWriteOptions(compression="zstd")


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
        """Two space optimisations, both material at corpus scale.

        *Context deduplication.* parent_child emits the identical parent block
        as the context of every one of its children, so contexts are stored
        once in a side table and referenced by index. On a 78k-chunk build that
        is 21.6MB of context text down to 13.1MB.

        *Dictionary encoding.* `lang`, `query_type` and `strategies` have
        cardinality in the single digits across the whole corpus; stored as
        raw strings they cost more than the chunk text they describe.
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        uniq_context: dict[str, int] = {}
        ctx_ref: list[int] = []
        for c in self.context:
            i = uniq_context.get(c)
            if i is None:
                i = len(uniq_context)
                uniq_context[c] = i
            ctx_ref.append(i)

        table = pa.table(
            {
                "text": pa.array(self.text, type=pa.large_string()),
                "context_ref": pa.array(ctx_ref, type=pa.int32()),
                "passage_id": pa.array([str(x) for x in self.passage_id]).dictionary_encode(),
                "lang": pa.array([str(x) for x in self.lang]).dictionary_encode(),
                "query_type": pa.array([str(x) for x in self.query_type]).dictionary_encode(),
                # Sorted join keeps the round-trip stable and diffable.
                "strategies": pa.array(
                    ["|".join(sorted(s)) for s in self.strategies]
                ).dictionary_encode(),
                "is_selected": pa.array(self.is_selected),
            }
        )
        with ipc.new_file(path, table.schema, options=_IPC_OPTS) as writer:
            writer.write_table(table)

        ctx_table = pa.table(
            {"context": pa.array(list(uniq_context.keys()), type=pa.large_string())}
        )
        with ipc.new_file(
            _context_path(path), ctx_table.schema, options=_IPC_OPTS
        ) as writer:
            writer.write_table(ctx_table)

    @classmethod
    def load(cls, path: Path) -> "ChunkStore":
        with ipc.open_file(path) as reader:
            table = reader.read_all()
        with ipc.open_file(_context_path(path)) as reader:
            contexts = reader.read_all()["context"].to_pylist()

        ctx_ref = table["context_ref"].to_numpy()
        strategies = [
            frozenset(s.split("|")) for s in table["strategies"].to_pylist()
        ]
        return cls(
            text=table["text"].to_pylist(),
            context=[contexts[i] for i in ctx_ref],
            passage_id=np.array(table["passage_id"].to_pylist(), dtype=object),
            lang=np.array(table["lang"].to_pylist(), dtype=object),
            query_type=np.array(table["query_type"].to_pylist(), dtype=object),
            strategies=strategies,
            is_selected=np.array(table["is_selected"].to_pylist(), dtype=bool),
            n_strategies=np.array([len(s) for s in strategies], dtype=np.int8),
        )


def _context_path(path: Path) -> Path:
    return path.with_name(path.stem + "_contexts" + path.suffix)
