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
    # Set these to ship the model inside the image so container boot never
    # depends on the Hub being reachable.
    local_encoder_path: str = ""
    local_tokenizer_path: str = ""
    # Free-tier Spaces give 2 vCPU; oversubscribing threads makes it slower.
    onnx_threads: int = 4
    embed_batch: int = 64

    # ---- retrieval ---------------------------------------------------------
    dense_candidates: int = 60  # ANN top-k before fusion
    lexical_candidates: int = 60  # BM25 top-k before fusion
    fusion_candidates: int = 40  # survivors of RRF, fed to the reranker
    final_k: int = 5  # chunks handed to answer generation
    rrf_k: float = 60.0  # RRF damping constant

    # HNSW build/search knobs. `ef_search` is the main latency/recall dial.
    hnsw_connectivity: int = 16
    hnsw_expansion_add: int = 128
    hnsw_expansion_search: int = 64

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
