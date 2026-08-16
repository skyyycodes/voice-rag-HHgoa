"""Central configuration. Everything tunable lives here, nothing is hardcoded downstream."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
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
    # Tier A: static token embeddings. No transformer forward pass at query
    # time, which is what makes the sub-200ms budget achievable.
    static_model: str = "minishlab/potion-multilingual-128M"
    embed_batch: int = 2048

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
    # Below this max fused score the query is treated as out-of-domain.
    ood_score_floor: float = 0.34
    # Minimum lexical+semantic overlap between answer and cited chunk.
    grounding_floor: float = 0.45
    min_query_chars: int = 3
    max_query_chars: int = 512

    # ---- speech to text ----------------------------------------------------
    stt_provider: str = "sarvam"  # sarvam | elevenlabs
    sarvam_api_key: str = ""
    sarvam_model: str = "saarika:v2.5"
    elevenlabs_api_key: str = ""
    elevenlabs_model: str = "scribe_v1"
    stt_timeout_s: float = 12.0
    stt_max_retries: int = 2

    # ---- answer generation -------------------------------------------------
    # "extractive" is the default fast path and the one benchmarked against the
    # 200ms target. "llm" is available through the same harness contract.
    answer_mode: str = "extractive"  # extractive | llm
    anthropic_api_key: str = ""
    llm_model: str = "claude-sonnet-5"
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
