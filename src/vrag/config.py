"""Central configuration. Everything tunable lives here, nothing is hardcoded downstream."""

from __future__ import annotations

from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    # Every setting is namespaced `VRAG_*`. Without a prefix these bind to bare
    # names like `LANGUAGES` and `ONNX_THREADS`, which collide with unrelated
    # environment variables — and, worse, made every `VRAG_*` override in the
    # Dockerfile and docs silently do nothing.
    #
    # The third-party credentials are the deliberate exception: they accept
    # their conventional unprefixed names too, since that is what the providers
    # document and what a deploy platform's secret UI will be set to.
    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        env_prefix="VRAG_",
        extra="ignore",
    )

    # ---- paths -------------------------------------------------------------
    data_dir: Path = REPO_ROOT / "data"
    raw_dir: Path = REPO_ROOT / "data" / "raw"
    index_dir: Path = REPO_ROOT / "data" / "index"

    # ---- corpus ------------------------------------------------------------
    # Which MSMARCO-XI language shards to ingest. Each is a validation parquet.
    languages: tuple[str, ...] = ("hin", "ben", "tam")
    # Cap rows per shard so the index fits in a free-tier container.
    max_rows_per_lang: int = 12_000

    # ---- embeddings --------------------------------------------------------
    # int8-quantised ONNX export: 118MB on disk, ~1.3ms per query on CPU.
    # The fp32 export is 470MB and measurably no better for this corpus.
    encoder_model: str = "intfloat/multilingual-e5-small"
    encoder_onnx_file: str = "onnx/model_qint8_avx512_vnni.onnx"
    encoder_tokenizer_file: str = "onnx/tokenizer.json"
    encoder_max_tokens: int = 512
    # Max padded tokens (rows x width) per inference batch. Work and peak
    # memory scale with this product, not with row count — batching by rows
    # alone made Indic batches, which tokenise far longer than English, both
    # slow and multi-gigabyte.
    encoder_token_budget: int = 8192
    # Set these to ship the model inside the image so container boot never
    # depends on the Hub being reachable.
    local_encoder_path: str = ""
    local_tokenizer_path: str = ""
    # Free-tier Spaces give 2 vCPU; oversubscribing threads makes it slower.
    onnx_threads: int = 4
    # Row ceiling; the token budget above is what actually binds on long text.
    embed_batch: int = 256

    # ---- retrieval ---------------------------------------------------------
    dense_candidates: int = 60  # ANN top-k before fusion
    lexical_candidates: int = 60  # BM25 top-k before fusion
    fusion_candidates: int = 40  # survivors of RRF, fed to the reranker
    final_k: int = 5  # chunks handed to answer generation
    rrf_k: float = 60.0  # RRF damping constant
    # Held-out pairwise accuracy the learned reranker must beat to be shipped.
    # Below this it is not better than the RRF ordering it would replace, and
    # the build falls back rather than reordering results by noise.
    rerank_min_held_out_acc: float = 0.55
    # Answer in the language the question was asked in. Expressed as a
    # *fraction of the candidate set score spread*, so it behaves identically
    # whether the linear model (~±2) or the cross-encoder (~±11) produced the
    # scores. Small on purpose: it breaks ties between translations of the same
    # passage without letting a weak same-language match beat a strong foreign
    # one.
    same_language_bonus: float = 0.15
    english_fallback_bonus: float = 0.06

    # HNSW build/search knobs. `ef_search` is the main latency/recall dial.
    hnsw_connectivity: int = 16
    hnsw_expansion_add: int = 128
    hnsw_expansion_search: int = 64

    # ---- cross-encoder rerank ----------------------------------------------
    # Trained on multilingual MS MARCO — this system's exact task. Reads the
    # (query, chunk) pair jointly, so it judges whether a passage *answers* the
    # question rather than whether the two look alike.
    rerank_enabled: bool = True
    rerank_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    rerank_onnx_file: str = "onnx/model_qint8_avx512_vnni.onnx"
    # This model keeps its tokenizer at the repo root, unlike the bi-encoder.
    rerank_tokenizer_file: str = "tokenizer.json"
    rerank_max_tokens: int = 128
    # Ceiling on pairs scored per query; the *actual* depth is chosen per
    # request from the budget still remaining. 8 is where the measured
    # quality-vs-budget curve peaks on MRR while keeping *every* percentile
    # (P100 139.8ms) inside the 200ms bar — deeper buys R@5 but breaks it.
    # See bench/results/quality_vs_budget.json.
    rerank_depth: int = 8
    local_rerank_path: str = ""
    local_rerank_tokenizer: str = ""
    # Lifts cross-encoder logits (roughly -11..+11) clear of the linear
    # feature scores so a reranked candidate never sorts below an unreranked
    # one purely because the two scales differ.
    rerank_score_offset: float = 100.0

    # ---- latency budget (milliseconds, per stage) --------------------------
    # The harness enforces these; a stage that blows its budget degrades
    # instead of blocking the pipeline.
    budget_total_ms: float = 200.0
    budget_guardrail_in_ms: float = 10.0
    budget_retrieve_ms: float = 120.0
    budget_generate_ms: float = 40.0
    budget_guardrail_out_ms: float = 20.0

    # ---- guardrails --------------------------------------------------------
    # Out-of-domain detection is margin-based, not an absolute score floor:
    # cosine magnitudes differ systematically by language, so a constant floor
    # calibrated on Hindi rejects valid Tamil traffic. Both must be low to
    # abstain. Calibrated by `vrag.calibrate_guardrails`.
    ood_margin_floor: float = 1.0
    ood_dense_floor: float = 0.80
    # Minimum lexical+semantic overlap between answer and cited chunk.
    grounding_floor: float = 0.45
    # Cosine below which a cross-lingual span counts as unrelated. e5
    # similarities are compressed into a narrow high band, so this is ~0.78
    # rather than the ~0.3 an uncalibrated intuition would suggest.
    semantic_span_floor: float = 0.82
    # Max sentences encoded on the cross-lingual span path.
    semantic_span_limit: int = 24
    # Per-rank multiplier when choosing which retrieved passage to quote.
    # Geometric so retrieval order dominates span score; at 0.72 a rank-3
    # sentence needs to be ~4.6x better in isolation to win.
    answer_rank_decay: float = 0.60
    min_query_chars: int = 3
    max_query_chars: int = 512

    # ---- speech to text ----------------------------------------------------
    stt_provider: str = "sarvam"  # sarvam | elevenlabs
    sarvam_api_key: str = Field(
        default="", validation_alias=AliasChoices("VRAG_SARVAM_API_KEY", "SARVAM_API_KEY")
    )
    sarvam_model: str = "saaras:v3"  # verified against the live API
    elevenlabs_api_key: str = Field(
        default="", validation_alias=AliasChoices("VRAG_ELEVENLABS_API_KEY", "ELEVENLABS_API_KEY")
    )
    elevenlabs_model: str = "scribe_v1"
    stt_timeout_s: float = 12.0
    stt_max_retries: int = 2

    # ---- answer generation -------------------------------------------------
    # "extractive" is the default fast path and the one benchmarked against the
    # 200ms target. "llm" is available through the same harness contract.
    answer_mode: str = "extractive"  # extractive | llm
    anthropic_api_key: str = Field(
        default="", validation_alias=AliasChoices("VRAG_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY")
    )
    llm_model: str = "claude-opus-5"
    llm_timeout_s: float = 20.0

    @property
    def chunk_store(self) -> Path:
        return self.index_dir / "chunks.arrow"

    @property
    def vector_store(self) -> Path:
        return self.index_dir / "dense.usearch"

    @property
    def lexical_store(self) -> Path:
        return self.index_dir / "bm25"

    @property
    def manifest(self) -> Path:
        return self.index_dir / "manifest.json"


settings = Settings()
