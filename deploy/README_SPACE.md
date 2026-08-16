---
title: Voice RAG — MSMARCO-XI
emoji: 🎙️
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: 6.24.0
# The Spaces builder defaults to 3.10, where numpy>=2.5 and pyarrow>=25 have no
# wheels. pyproject declares >=3.12 and every benchmark was run on 3.12.14 —
# building on anything older silently swaps the numeric stack under the numbers.
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
short_description: Voice-in RAG over Indic MS MARCO, sub-200ms retrieval
---

# Voice RAG — MSMARCO-XI

Speak a question in Hindi, Bengali, Tamil or English. The pipeline transcribes
it (Sarvam Saaras), retrieves from a multi-strategy chunk index over
[ai4bharat/MSMARCO-XI](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI),
verifies the answer is grounded in the retrieved passages, and answers — or
declines, with a reason.

Every response shows its own per-stage latency breakdown and cites the passage
it drew from, with the quoted span highlighted in context.

**Set `SARVAM_API_KEY` as a Space secret** to enable the microphone. Without it
the text box still works and exercises everything downstream of transcription.
`ANTHROPIC_API_KEY` is optional and enables the fluent LLM answer path.

Latency shown per request is measured in-process — there is no HTTP hop
inflating or hiding it.

See the [GitHub repository](https://github.com/) for architecture notes,
the chunking-strategy ablation, and the full latency benchmark.
