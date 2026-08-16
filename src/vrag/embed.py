"""Embedding tier.

Started on static embeddings (model2vec) purely for speed, then measured them:
on full-corpus retrieval over 418k chunks they collapsed to R@1 = 0.045, and a
depth-500 probe found the gold chunk for only 39% of queries. The ceiling was
the encoder, not the ranking on top of it — a static model averages subword
vectors, so a Tamil transliteration of "CHIPSA" shares nothing with the Latin
string, and short acronym queries are unmatchable in principle.

`multilingual-e5-small`, int8-quantised to ONNX, fixes that for +1.3ms per
query — trivial against a 200ms budget. Measured head-to-head on 155 queries
per language:

    Hindi   P@1 0.245 -> 0.394   MRR 0.467 -> 0.576
    Tamil   P@1 0.232 -> 0.239   MRR 0.441 -> 0.451

Tamil barely benefits; e5's Tamil coverage is genuinely weaker, and the
per-language numbers are reported rather than averaged away.

E5 is *asymmetric*: it was trained with "query: " and "passage: " prefixes and
loses substantial accuracy without them, which is why this module never exposes
an untyped `encode`. Getting the prefix wrong is a silent quality regression,
so `Kind` makes the caller state which side they are on.
"""

from __future__ import annotations

import threading
from enum import Enum
from typing import Sequence

import numpy as np

from .config import Settings, settings


class Kind(str, Enum):
    QUERY = "query: "
    PASSAGE = "passage: "


class OnnxEncoder:
    """int8 ONNX sentence encoder with mean pooling."""

    def __init__(self, cfg: Settings = settings) -> None:
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        model_path = (
            cfg.local_encoder_path
            or hf_hub_download(cfg.encoder_model, cfg.encoder_onnx_file)
        )
        tok_path = (
            cfg.local_tokenizer_path
            or hf_hub_download(cfg.encoder_model, cfg.encoder_tokenizer_file)
        )

        self.tokenizer = Tokenizer.from_file(tok_path)
        self.tokenizer.enable_truncation(cfg.encoder_max_tokens)

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = cfg.onnx_threads
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            model_path, opts, providers=["CPUExecutionProvider"]
        )
        self._input_names = {i.name for i in self.session.get_inputs()}
        self.dim = int(self.session.get_outputs()[0].shape[-1])

    def encode(self, texts: Sequence[str], kind: Kind, batch_size: int = 64) -> np.ndarray:
        """Encode a batch, sorting by length first.

        Every sequence in a batch is padded to the longest member, and
        attention is quadratic in that padded length. With chunks arriving in
        corpus order, one 400-token chunk drags 63 short ones up to its length
        and most of the compute is spent on padding. Sorting by length groups
        similar sizes together, then the original order is restored — on this
        corpus that is the difference between a ~50-minute index build and a
        few minutes.

        Only worth the bookkeeping in bulk; a single query skips it.
        """
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        if len(texts) <= batch_size:
            return self._encode_batch(list(texts), kind)

        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        sorted_texts = [texts[i] for i in order]

        encoded = np.vstack(
            [
                self._encode_batch(sorted_texts[i : i + batch_size], kind)
                for i in range(0, len(sorted_texts), batch_size)
            ]
        )

        # Scatter back so row i corresponds to texts[i]; callers index these
        # rows against the chunk store, so a permuted result would silently
        # attach every vector to the wrong chunk.
        out = np.empty_like(encoded)
        out[np.asarray(order)] = encoded
        return out

    def _encode_batch(self, texts: list[str], kind: Kind) -> np.ndarray:
        encoded = self.tokenizer.encode_batch([kind.value + t for t in texts])
        # Pad to the longest sequence in *this* batch rather than the model
        # maximum: attention is quadratic in length, and MS MARCO chunks are
        # short, so batch-local padding is several times faster.
        width = max(len(e.ids) for e in encoded)
        ids = np.zeros((len(encoded), width), dtype=np.int64)
        mask = np.zeros((len(encoded), width), dtype=np.int64)
        for i, e in enumerate(encoded):
            ids[i, : len(e.ids)] = e.ids
            mask[i, : len(e.ids)] = e.attention_mask

        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self._input_names:
            feed["token_type_ids"] = np.zeros_like(ids)

        hidden = self.session.run(None, feed)[0]
        # Mean pooling over non-padding tokens — what e5 was trained with.
        m = mask[..., None].astype(np.float32)
        pooled = (hidden * m).sum(axis=1) / np.maximum(m.sum(axis=1), 1e-9)
        return l2_normalise(pooled.astype(np.float32))


_encoder: OnnxEncoder | None = None
_lock = threading.Lock()


def get_encoder(cfg: Settings = settings) -> OnnxEncoder:
    """Process-wide singleton; the ONNX session is thread-safe for inference."""
    global _encoder
    if _encoder is None:
        with _lock:
            if _encoder is None:
                _encoder = OnnxEncoder(cfg)
    return _encoder


def l2_normalise(vecs: np.ndarray) -> np.ndarray:
    if vecs.ndim == 1:
        vecs = vecs.reshape(1, -1)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    # A zero vector means the text tokenised to nothing; leave it at zero
    # rather than dividing by zero — it simply never matches.
    np.maximum(norms, 1e-12, out=norms)
    return vecs / norms


def encode_passages(texts: Sequence[str], cfg: Settings = settings) -> np.ndarray:
    return get_encoder(cfg).encode(texts, Kind.PASSAGE, cfg.embed_batch)


def encode_query(text: str, cfg: Settings = settings) -> np.ndarray:
    return get_encoder(cfg).encode([text], Kind.QUERY, 1)[0]


def encode_queries(texts: Sequence[str], cfg: Settings = settings) -> np.ndarray:
    return get_encoder(cfg).encode(texts, Kind.QUERY, cfg.embed_batch)


def dim(cfg: Settings = settings) -> int:
    return get_encoder(cfg).dim
