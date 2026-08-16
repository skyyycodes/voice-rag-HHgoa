"""Cross-encoder reranking, sized to the latency budget that is left.

The linear reranker scores query and chunk *separately* and combines cheap
features. A cross-encoder reads the pair jointly, so it can judge whether this
passage answers this question rather than whether they look alike. Measured on
this corpus the difference is stark — on a Hindi query it scores the relevant
passage +3.91, a same-topic distractor -2.80, and an unrelated passage -7.23,
where the linear model sat at 0.561 pairwise accuracy (barely above chance).

It also reranks *across* languages: a Hindi question against an English
passage scores +5.06 relevant / -6.13 irrelevant. That matters because gold
evidence in this corpus is frequently in another language than the question.

`mmarco-mMiniLMv2` is trained on multilingual MS MARCO — the exact task and
data distribution this system runs on — and int8-quantises to 118MB.

The reason this is affordable at all: retrieval measures 5.4ms P50 against a
200ms budget. Rather than bank 36x headroom nobody benefits from, the pipeline
spends it, and `depth_for_budget` decides how much to spend per request from
the time actually remaining. A request that has already burned its budget on a
slow transcription reranks shallowly or not at all; a fast one reranks deep.
The budget stops being a number in a report and becomes something the system
schedules against.
"""

from __future__ import annotations

import threading

import numpy as np

from ..config import Settings, settings

# Measured *in-pipeline*, not standalone: two ONNX sessions share the machine,
# so a pair costs ~5.5ms here against ~3.5ms for the reranker running alone.
# Budgeting with the standalone figure would consistently overshoot.
_MS_PER_PAIR = 5.5
# Below this many milliseconds of remaining budget, skip reranking entirely
# rather than half-run it and blow the deadline.
_MIN_VIABLE_MS = 12.0


class CrossEncoderReranker:
    def __init__(self, cfg: Settings = settings) -> None:
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        model_path = cfg.local_rerank_path or hf_hub_download(
            cfg.rerank_model, cfg.rerank_onnx_file
        )
        # Note the tokenizer lives at the repo root for this model, not under
        # onnx/ as it does for the bi-encoder.
        tok_path = cfg.local_rerank_tokenizer or hf_hub_download(
            cfg.rerank_model, cfg.rerank_tokenizer_file
        )

        self.tokenizer = Tokenizer.from_file(tok_path)
        self.tokenizer.enable_truncation(cfg.rerank_max_tokens)

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = cfg.onnx_threads
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Two ONNX sessions live in this process (bi-encoder + cross-encoder).
        # By default each spins its threads while waiting for work, so on a
        # 4-performance-core machine they busy-wait against each other and the
        # cross-encoder ran ~4x slower in-pipeline than standalone. Yielding
        # instead of spinning removes that interference.
        opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self.session = ort.InferenceSession(
            model_path, opts, providers=["CPUExecutionProvider"]
        )
        self._input_names = {i.name for i in self.session.get_inputs()}

    def score(self, query: str, docs: list[str]) -> np.ndarray:
        """Relevance logit per (query, doc) pair. Higher is more relevant."""
        if not docs:
            return np.zeros(0, dtype=np.float32)

        encoded = self.tokenizer.encode_batch([(query, d) for d in docs])
        width = max(len(e.ids) for e in encoded)
        ids = np.zeros((len(encoded), width), dtype=np.int64)
        mask = np.zeros((len(encoded), width), dtype=np.int64)
        for i, e in enumerate(encoded):
            ids[i, : len(e.ids)] = e.ids
            mask[i, : len(e.ids)] = e.attention_mask

        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self._input_names:
            feed["token_type_ids"] = np.zeros_like(ids)
        return self.session.run(None, feed)[0].ravel().astype(np.float32)


def depth_for_budget(remaining_ms: float, cfg: Settings = settings) -> int:
    """How many candidates can be reranked in the time that is left.

    Returns 0 when there is not enough budget to be worth starting — a
    half-finished rerank costs the latency and delivers none of the quality.
    """
    if remaining_ms < _MIN_VIABLE_MS:
        return 0
    # Leave a margin for generation and the output rails.
    usable = remaining_ms - cfg.budget_generate_ms - cfg.budget_guardrail_out_ms
    if usable < _MIN_VIABLE_MS:
        return 0
    return int(min(cfg.rerank_depth, max(0, usable / _MS_PER_PAIR)))


_reranker: CrossEncoderReranker | None = None
_lock = threading.Lock()


def get_reranker(cfg: Settings = settings) -> CrossEncoderReranker:
    global _reranker
    if _reranker is None:
        with _lock:
            if _reranker is None:
                _reranker = CrossEncoderReranker(cfg)
    return _reranker
