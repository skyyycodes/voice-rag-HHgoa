---
title: Voice RAG — MSMARCO-XI
emoji: 🎙️
colorFrom: green
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Voice-in RAG over 14-language Indic MS MARCO, sub-200ms retrieval
---

# Voice RAG — MSMARCO-XI

Speak a question in Hindi, Bengali, Tamil or English. The pipeline transcribes
it (Sarvam Saaras), retrieves from a multi-strategy chunk index over
[ai4bharat/MSMARCO-XI](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI),
verifies the answer is grounded in the retrieved passages, and answers — or
declines, with a reason.

Every response shows its own per-stage latency breakdown and cites the passage
it drew from, with the quoted span highlighted in context.

**Set `SARVAM_API_KEY` as a Space secret** to enable voice input. Without it the
text box still works and everything downstream of transcription is unchanged.
`ANTHROPIC_API_KEY` is optional and enables the fluent LLM answer path.

See the [GitHub repository](https://github.com/) for architecture notes,
the chunking-strategy ablation, and the full latency benchmark.
