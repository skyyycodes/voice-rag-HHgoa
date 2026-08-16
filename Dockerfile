# Hugging Face Spaces (Docker SDK) — also runs anywhere else Docker does.
FROM python:3.12-slim

# Spaces runs containers as UID 1000. Writing as root here and serving as 1000
# would leave the HF cache and index unreadable at runtime.
RUN useradd -m -u 1000 app
WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/.cache/huggingface \
    # Free Spaces give 2 vCPU; more ONNX threads than cores is slower, not
    # faster. The reranker times itself at startup and picks its depth from the
    # measured cost, so the 200ms budget holds here without retuning.
    VRAG_ONNX_THREADS=2 \
    PORT=7860

COPY --chown=app:app pyproject.toml README.md ./
COPY --chown=app:app src ./src

RUN pip install --no-cache-dir . && \
    mkdir -p /app/.cache/huggingface && chown -R app:app /app/.cache

# Bake BOTH models into the image so container boot never depends on the Hub
# being reachable. A cold Space downloading ~240MB before its first response is
# the difference between a working demo link and a timeout — and the retrieval
# quality claims depend on the cross-encoder actually being present, not
# silently skipped by the "reranking is optional" fallback.
RUN python -c "\
from huggingface_hub import hf_hub_download as d; \
[d('intfloat/multilingual-e5-small', f) for f in \
 ('onnx/model_qint8_avx512_vnni.onnx','onnx/tokenizer.json','onnx/config.json')]; \
[d('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1', f) for f in \
 ('onnx/model_qint8_avx512_vnni.onnx','tokenizer.json')]" && \
    chown -R app:app /app/.cache

# The prebuilt index. ~300MB of vectors + BM25 + chunk store; copied last so
# code edits don't invalidate the layer above.
COPY --chown=app:app data/index ./data/index
COPY --chown=app:app web ./web

USER app
EXPOSE 7860

# One worker on purpose: the ONNX session, ANN graph and chunk store are all
# loaded per process, so a second worker doubles memory for no throughput gain
# on a 2-vCPU box.
CMD ["uvicorn", "vrag.server:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
