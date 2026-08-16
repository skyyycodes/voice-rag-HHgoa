"""FastAPI server.

Loads the index once at startup — the model session, ANN graph, BM25 tables and
chunk store are all read-only after construction, so a single warm process
serves every request with no per-request load cost. That is a hard requirement
for the latency target, not an optimisation: reloading anything per request
would cost seconds.

The pipeline is warmed with a throwaway query during startup so the first real
user does not pay ONNX graph optimisation.
"""

from __future__ import annotations

import base64
import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import settings
from .harness.contracts import QueryRequest, QueryResponse
from .harness.orchestrator import Pipeline
from .index.hybrid import HybridRetriever
from .stt import build_transcribers

WEB_DIR = Path(__file__).resolve().parents[2] / "web"

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = settings
    if not cfg.chunk_store.exists():
        raise RuntimeError(
            f"No index at {cfg.index_dir}. Run `python -m vrag.build_index` first."
        )

    retriever = HybridRetriever.load(cfg)
    transcribers = build_transcribers(cfg)

    llm = None
    from .answer.llm import LLMAnswerer, is_configured

    if is_configured(cfg):
        llm = LLMAnswerer(cfg)

    pipeline = Pipeline(retriever, transcribers=transcribers, cfg=cfg, llm_answerer=llm)

    # Warm the ONNX session and page in the index before serving.
    await pipeline.run(QueryRequest(text="warmup query"))

    state["pipeline"] = pipeline
    state["manifest"] = (
        json.loads(cfg.manifest.read_text(encoding="utf-8")) if cfg.manifest.exists() else {}
    )
    state["stt_providers"] = [t.name for t in transcribers]
    state["llm_available"] = llm is not None
    yield
    if llm is not None:
        await llm.aclose()


app = FastAPI(title="Voice RAG — MSMARCO-XI", version="1.0.0", lifespan=lifespan)


@app.get("/api/health")
async def health() -> dict:
    pipeline: Pipeline = state["pipeline"]
    return {
        "status": "ok",
        "chunks": len(pipeline.retriever.store),
        "manifest": state.get("manifest", {}),
        "stt_providers": state.get("stt_providers", []),
        "llm_available": state.get("llm_available", False),
        "tools": pipeline.tools.describe(),
        "budget_ms": settings.budget_total_ms,
    }


@app.post("/api/ask", response_model=QueryResponse)
async def ask(request: QueryRequest) -> QueryResponse:
    """Text-in. The voice path funnels here after transcription."""
    if not request.text and not request.audio_b64:
        raise HTTPException(status_code=400, detail="Provide `text` or `audio_b64`.")
    return await state["pipeline"].run(request)


@app.post("/api/voice", response_model=QueryResponse)
async def voice(
    audio: UploadFile = File(...),
    language: str | None = Form(default=None),
    answer_mode: str | None = Form(default=None),
) -> QueryResponse:
    """Multipart audio upload — what the browser mic posts to.

    Base64 through the JSON endpoint would inflate the payload ~33% and add an
    encode/decode step on the critical path for no benefit.
    """
    # Cap the read. This endpoint is public on a deployed Space and
    # `await audio.read()` with no limit will happily pull a multi-gigabyte
    # body into memory. Reading one byte past the limit is enough to detect
    # and reject an oversized upload without buffering the rest.
    limit = settings.max_audio_bytes
    raw = await audio.read(limit + 1)
    if not raw:
        raise HTTPException(status_code=400, detail="Empty audio upload.")
    if len(raw) > limit:
        raise HTTPException(
            status_code=413,
            detail=f"Audio exceeds {limit // 1_000_000}MB. Send a shorter clip.",
        )

    suffix = (audio.filename or "audio.wav").rsplit(".", 1)[-1].lower()
    fmt = suffix if suffix in {"wav", "mp3", "webm", "ogg", "flac"} else "webm"

    return await state["pipeline"].run(
        QueryRequest(
            audio_b64=base64.b64encode(raw).decode("ascii"),
            audio_format=fmt,
            language=language,
            answer_mode=answer_mode,
        )
    )


@app.get("/api/bench")
async def bench_results() -> JSONResponse:
    """Serve the last benchmark run so the live demo can show real numbers
    rather than a screenshot of them."""
    path = Path("bench/results/latest.json")
    if not path.exists():
        return JSONResponse({"error": "no benchmark run yet"}, status_code=404)
    return JSONResponse(json.loads(path.read_text(encoding="utf-8")))


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    page = WEB_DIR / "index.html"
    if not page.exists():
        return HTMLResponse("<h1>Voice RAG</h1><p>Web UI not built.</p>")
    return HTMLResponse(page.read_text(encoding="utf-8"))


if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
