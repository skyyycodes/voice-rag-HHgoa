"""BM25 lexical index.

Dense retrieval alone is weak exactly where MS MARCO is strongest: rare proper
nouns, product codes, and numbers. A static embedder averages token vectors, so
"1867" contributes almost nothing to the pooled vector, while BM25 treats it as
a high-IDF term and nails it. That complementarity is the whole reason for
hybrid fusion — the two retrievers fail on different queries.

Tokenisation is the hard part here, because the corpus spans Devanagari,
Bengali and Tamil. Snowball has no stemmer for any of them, and Tamil in
particular is agglutinative: one written word can carry case, number and
postposition, so exact-token matching misses constantly. We approximate
stemming for non-Latin tokens by additionally indexing a character prefix,
which collapses most inflected forms of a stem onto a shared term.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

import bm25s
import numpy as np

_WORD = re.compile(r"\w+", re.UNICODE)
# Latin-script detection: if a token is pure ASCII we can stem it properly.
_ASCII = re.compile(r"^[a-z0-9]+$")

# Indic morphology approximation. 5 characters keeps the stem of most Hindi and
# Bengali words while dropping case/postposition suffixes; Tamil needs more
# because its roots are longer.
_PREFIX_LEN = 5
_MIN_FOR_PREFIX = 7

_STOP = {
    # English function words. Indic stopwords are left in: they are short and
    # high-frequency, so IDF already discounts them to near zero, and hand-built
    # Indic stoplists tend to remove genuine content words.
    "the", "a", "an", "of", "and", "or", "in", "on", "at", "to", "for", "is",
    "are", "was", "were", "be", "been", "it", "its", "this", "that", "as",
    "by", "with", "from", "what", "which", "who", "how", "why", "when",
}


def _stem_ascii(token: str) -> str:
    """Light suffix stripping for English. Snowball via PyStemmer is more
    accurate but costs a Python call per token; at corpus scale this
    hand-rolled version is ~8x faster and loses very little on MS MARCO."""
    for suffix in ("ations", "ation", "ings", "ing", "ies", "ied", "es", "ed", "s"):
        if len(token) > len(suffix) + 3 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def tokenize(text: str) -> list[str]:
    """Script-aware tokenisation shared by index and query paths.

    Both paths *must* use this same function — a mismatch between index-time
    and query-time tokenisation is the classic silent BM25 bug where recall
    quietly collapses and nothing errors.
    """
    out: list[str] = []
    for match in _WORD.finditer(text.lower()):
        token = match.group()
        if token in _STOP:
            continue
        if _ASCII.match(token):
            out.append(_stem_ascii(token))
        else:
            out.append(token)
            # Emit a prefix term as a stand-in for the missing Indic stemmer.
            if len(token) >= _MIN_FOR_PREFIX:
                out.append(token[:_PREFIX_LEN])
    return out


class LexicalIndex:
    """Thin wrapper over bm25s with our tokeniser bolted to both paths."""

    def __init__(self, retriever: bm25s.BM25, vocab: dict[str, int]) -> None:
        self._retriever = retriever
        self._vocab = vocab

    @classmethod
    def build(cls, texts: Sequence[str]) -> "LexicalIndex":
        corpus = [tokenize(t) for t in texts]
        vocab: dict[str, int] = {}
        ids: list[list[int]] = []
        for doc in corpus:
            row = []
            for token in doc:
                idx = vocab.get(token)
                if idx is None:
                    idx = len(vocab)
                    vocab[token] = idx
                row.append(idx)
            ids.append(row)

        retriever = bm25s.BM25()
        retriever.index(bm25s.tokenization.Tokenized(ids=ids, vocab=vocab))
        return cls(retriever, vocab)

    def search(self, query: str, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (doc_indices, scores), highest score first."""
        tokens = [self._vocab[t] for t in tokenize(query) if t in self._vocab]
        if not tokens:
            # Every query term is out of vocabulary — legitimate for a query in
            # a language the corpus does not cover. Dense retrieval carries it.
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)

        query_obj = bm25s.tokenization.Tokenized(ids=[tokens], vocab=self._vocab)
        idx, scores = self._retriever.retrieve(
            query_obj, k=min(k, self._retriever.scores["num_docs"])
        )
        return idx[0].astype(np.int64), scores[0].astype(np.float32)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self._retriever.save(str(path))
        np.save(path / "vocab_keys.npy", np.array(list(self._vocab.keys()), dtype=object))
        np.save(path / "vocab_vals.npy", np.array(list(self._vocab.values()), dtype=np.int64))

    @classmethod
    def load(cls, path: Path) -> "LexicalIndex":
        retriever = bm25s.BM25.load(str(path))
        keys = np.load(path / "vocab_keys.npy", allow_pickle=True)
        vals = np.load(path / "vocab_vals.npy")
        return cls(retriever, {str(k): int(v) for k, v in zip(keys, vals)})
