"""Input guardrails: what the system refuses to even attempt.

Ordered cheapest-first and short-circuiting, because these run inside the
latency budget on every request. Structural checks cost microseconds; the
retrieval-dependent out-of-domain check is deliberately *not* here — it needs
the retrieved evidence, so it lives in the output rails where the evidence
already exists and costs nothing extra.

Deliberately not an LLM classifier. A safety call to a model would add
several hundred milliseconds and a second failure mode to protect a corpus of
MS MARCO web passages. Rules are auditable, instant, and their failure mode is
a false refusal rather than a silent leak.
"""

from __future__ import annotations

import re
import unicodedata

from ..config import Settings, settings
from ..harness.contracts import Decision, GuardVerdict

# Categories where a web-passage corpus has no business answering, regardless
# of what retrieval turns up. Kept narrow: over-broad safety lists on a general
# QA corpus mostly produce false refusals on legitimate medical or historical
# questions, which is its own kind of failure.
_UNSAFE_PATTERNS: list[tuple[str, re.Pattern]] = [
    (
        "weapons_synthesis",
        re.compile(
            r"\b(how\s+(to|do\s+i)\s+(make|build|synthes\w*|construct)\s+"
            r"(a\s+)?(bomb|explosive|nerve\s+agent|sarin|ricin|meth\w*|napalm))",
            re.IGNORECASE,
        ),
    ),
    (
        "self_harm",
        re.compile(r"\b(how\s+to\s+(kill|hurt|harm)\s+(myself|yourself)|commit\s+suicide)\b", re.IGNORECASE),
    ),
    (
        "csam",
        re.compile(r"\b(child|minor|underage)\s+(porn\w*|sexual|nude)", re.IGNORECASE),
    ),
    (
        "targeted_violence",
        re.compile(r"\bhow\s+to\s+(kill|murder|poison)\s+(?!a\s+(mockingbird|process|bug))\w+", re.IGNORECASE),
    ),
]

# Prompt injection. The extractive path cannot be hijacked by instructions in a
# query, but the LLM path can, and the same rail must cover both — a guardrail
# that only holds for the default configuration is not a guardrail.
_INJECTION_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("ignore_instructions", re.compile(
        r"\b(ignore|disregard|forget)\s+(all\s+|any\s+|your\s+|the\s+)*"
        r"(previous|prior|above|earlier|system)\s+(instruction|prompt|rule|direction)", re.IGNORECASE)),
    ("role_override", re.compile(
        r"\b(you\s+are\s+now|act\s+as|pretend\s+to\s+be|from\s+now\s+on\s+you)\b", re.IGNORECASE)),
    ("prompt_exfiltration", re.compile(
        r"\b(reveal|show|print|repeat|output)\s+(me\s+)?(your\s+)?"
        r"(system\s+prompt|instructions|initial\s+prompt|rules)\b", re.IGNORECASE)),
    ("delimiter_injection", re.compile(r"(<\|[a-z_]+\|>|\[/?INST\]|###\s*(system|assistant):)", re.IGNORECASE)),
]

# PII we should not echo back into logs or an answer.
_PII_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("credit_card", re.compile(r"\b(?:\d[ -]*?){13,16}\b")),
    ("aadhaar", re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b")),
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]{2,}\b")),
    ("phone_in", re.compile(r"\b(?:\+91[\s-]?)?[6-9]\d{9}\b")),
]

# Invisible characters split into two classes, because a blanket strip is wrong
# for this corpus.
#
# These carry no linguistic content in any script — zero-width space, BOM, and
# the bidi overrides used to make a string render differently from how it is
# processed. Replaced with a *space* rather than deleted: an attacker separating
# words with U+200B produces "ignore​all​previous", and deleting the
# separators yields "ignoreallprevious", which matches nothing. Substituting a
# space restores the word boundaries the patterns key on.
_INVISIBLE_TO_SPACE = re.compile(r"[​  ﻿‪-‮⁦-⁩]")

# ZWJ and ZWNJ are different: they are *linguistically load-bearing* in
# Devanagari, Bengali and Tamil, where they control conjunct formation. Deleting
# them globally would corrupt legitimate Indic queries — which is most of this
# corpus. They are only removed between two ASCII letters, a position where they
# have no valid role and can only be filter evasion.
_ZW_JOINER_IN_LATIN = re.compile(r"(?<=[A-Za-z])[‌‍]+(?=[A-Za-z])")


def normalise(text: str) -> str:
    """Canonicalise before matching.

    Without NFKC a filter is trivially bypassed with fullwidth or mathematical
    alphanumerics — "ｉｇｎｏｒｅ　ｐｒｅｖｉｏｕｓ" does not match an ASCII pattern but
    reads identically to a human and tokenises close enough for a model.
    """
    text = unicodedata.normalize("NFKC", text)
    text = _ZW_JOINER_IN_LATIN.sub("", text)
    text = _INVISIBLE_TO_SPACE.sub(" ", text)
    return " ".join(text.split())


def check_input(text: str, cfg: Settings = settings) -> GuardVerdict:
    """Run the input rails. Returns the first blocking verdict, or allow."""
    cleaned = normalise(text)
    triggered: list[str] = []

    if len(cleaned) < cfg.min_query_chars:
        return GuardVerdict(
            allowed=False,
            decision=Decision.REJECT_MALFORMED,
            reason="Query is too short to act on.",
            triggered=["too_short"],
        )
    if len(cleaned) > cfg.max_query_chars:
        return GuardVerdict(
            allowed=False,
            decision=Decision.REJECT_MALFORMED,
            reason=f"Query exceeds {cfg.max_query_chars} characters.",
            triggered=["too_long"],
        )

    for name, pattern in _UNSAFE_PATTERNS:
        if pattern.search(cleaned):
            return GuardVerdict(
                allowed=False,
                decision=Decision.REFUSE_UNSAFE,
                reason="This request falls outside what this system will answer.",
                triggered=[name],
            )

    for name, pattern in _INJECTION_PATTERNS:
        if pattern.search(cleaned):
            # Injection is refused rather than stripped. Sanitising and
            # proceeding means guessing which part of the input was the real
            # question, and guessing wrong answers something nobody asked.
            return GuardVerdict(
                allowed=False,
                decision=Decision.REJECT_MALFORMED,
                reason="The query contains instruction-injection patterns.",
                triggered=[name],
            )

    for name, pattern in _PII_PATTERNS:
        if pattern.search(cleaned):
            triggered.append(f"pii:{name}")

    # PII is flagged, not blocked: the query is still answerable, but the
    # harness redacts before logging so the transcript never persists it.
    return GuardVerdict(allowed=True, decision=Decision.ANSWER, triggered=triggered)


def redact(text: str) -> str:
    """Mask PII for logs and telemetry."""
    for name, pattern in _PII_PATTERNS:
        text = pattern.sub(f"[{name}]", text)
    return text
