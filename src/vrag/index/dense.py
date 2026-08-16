"""Dense ANN index (usearch HNSW).

Two decisions here are driven by deployment constraints rather than by recall:

*Scalar quantisation to int8.* Seven chunking strategies over three language
shards produce well over a million chunks. At 256 float32 dimensions that is
~1.4GB of vectors, which will not fit a free-tier container alongside the
model. usearch quantises to int8 internally, cutting that ~4x to ~350MB while
costing very little recall — the vectors are L2-normalised, so their components
already sit in a narrow, well-conditioned range, which is the case scalar
quantisation handles best.

*`exact=True` fallback for small indexes.* Below a few thousand vectors HNSW's
graph structure is pure overhead and its approximation actually hurts recall;
a brute-force scan is both faster and exact. The threshold matters because the
test suite and CI build tiny indexes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from usearch.index import Index

from ..config import Settings, settings

# Below this many vectors, brute force beats the graph on both speed and recall.
_EXACT_THRESHOLD = 4096


class DenseIndex:
    def __init__(self, index: Index, count: int) -> None:
        self._index = index
        self._count = count

    @classmethod
    def build(cls, vectors: np.ndarray, cfg: Settings = settings) -> DenseIndex:
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)
        index = Index(
            ndim=vectors.shape[1],
            metric="cos",
            dtype="i8",  # scalar quantisation; see module docstring
            connectivity=cfg.hnsw_connectivity,
            expansion_add=cfg.hnsw_expansion_add,
            expansion_search=cfg.hnsw_expansion_search,
        )
        index.add(np.arange(len(vectors), dtype=np.int64), vectors, log=False)
        return cls(index, len(vectors))

    def search(
        self, query: np.ndarray, k: int, cfg: Settings = settings
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (chunk_indices, similarities) with similarity in [0, 1]."""
        if query.ndim == 1:
            query = query.reshape(1, -1)
        matches = self._index.search(
            query.astype(np.float32),
            min(k, self._count),
            exact=self._count < _EXACT_THRESHOLD,
            log=False,
        )
        keys = np.asarray(matches.keys, dtype=np.int64).ravel()
        # usearch returns cosine *distance*; the rest of the pipeline reasons
        # in similarity, and fusion assumes higher-is-better.
        sims = 1.0 - np.asarray(matches.distances, dtype=np.float32).ravel()
        return keys, sims

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._index.save(str(path))

    @classmethod
    def load(cls, path: Path, cfg: Settings = settings) -> DenseIndex:
        index = Index.restore(str(path), view=False)
        return cls(index, len(index))

    def __len__(self) -> int:
        return self._count
