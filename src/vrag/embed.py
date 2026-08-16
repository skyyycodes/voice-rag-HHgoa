"""Embedding tier.

The 200ms budget is what dictates the model choice here. A transformer
bi-encoder costs 10-30ms per query on CPU before any search happens, and on a
cold free-tier container closer to 100ms. A *static* embedder (model2vec) has
no forward pass at all: token vectors are looked up from a table and pooled, so
encoding a query is a gather plus a mean — tens of microseconds, and flat in
model size.

The trade is quality: static embeddings lose word order and context. We buy
most of that back at the retrieval layer instead, with BM25 fusion and a
lexical-overlap reranker, which together cost far less than a transformer
forward pass.

`potion-multilingual-128M` is distilled from a multilingual teacher, so Hindi,
Bengali and Tamil queries land in the same space as the English passages they
should match. That is what makes cross-lingual retrieval work without a
translation hop.
"""

from __future__ import annotations

import threading
from typing import Sequence

import numpy as np

from .config import Settings, settings

_model = None
_lock = threading.Lock()


def get_model(cfg: Settings = settings):
    """Process-wide singleton. Loading is slow and the model is read-only.

    Two non-obvious arguments:

    `force_download=False` — model2vec defaults this to *True*, so every call
    re-fetches the entire ~1GB repo (including a 512MB ONNX export we never
    use) even when the cache is warm. Left at the default it adds minutes to
    every cold start and silently re-downloads on a deployed Space.

    `quantize_to` — the released model is float32, and its embedding matrix is
    ~500k vocab x 256 dims = 512MB, which dominates container memory. int8
    scalar quantisation cuts that to ~128MB for a negligible retrieval-quality
    cost, since we L2-normalise immediately afterwards anyway.
    """
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                from model2vec import StaticModel

                source = cfg.local_model_dir if cfg.local_model_dir else cfg.static_model
                kwargs = {"force_download": False}
                if cfg.embed_quantize:
                    kwargs["quantize_to"] = cfg.embed_quantize
                _model = StaticModel.from_pretrained(source, **kwargs)
    return _model


def encode(texts: Sequence[str], cfg: Settings = settings) -> np.ndarray:
    """Encode to L2-normalised float32.

    Normalising here means every downstream similarity is a plain dot product,
    which is what both usearch's cosine metric and the reranker assume.
    """
    model = get_model(cfg)
    vecs = model.encode(
        list(texts), batch_size=cfg.embed_batch, show_progress_bar=False
    ).astype(np.float32)
    return l2_normalise(vecs)


def l2_normalise(vecs: np.ndarray) -> np.ndarray:
    if vecs.ndim == 1:
        vecs = vecs.reshape(1, -1)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    # A zero vector means the text tokenised to nothing (punctuation only).
    # Leave it at zero rather than dividing by zero; it will simply never match.
    np.maximum(norms, 1e-12, out=norms)
    return vecs / norms


def encode_one(text: str, cfg: Settings = settings) -> np.ndarray:
    """Single-query path. Kept separate so the hot path skips list overhead."""
    return encode([text], cfg)[0]


def dim(cfg: Settings = settings) -> int:
    return int(get_model(cfg).dim)
