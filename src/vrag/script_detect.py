"""Script detection, used to answer in the language the question was asked in.

MSMARCO-XI translates every passage into every target language, so for almost
any query a same-language answer exists in the index. Retrieval does not
naturally prefer it: the translations are near-identical in embedding space, so
the ordering among them is effectively arbitrary, and measured on held-out
queries only 22% of questions got a same-language top hit. A Hindi speaker
asking a Hindi question was routinely answered in Bengali.

That is a product decision, not something to learn from relevance labels —
`is_selected` says nothing about what language the reader speaks. So it is
applied as an explicit preference over the ranked candidates rather than as a
reranker feature.

Detection is by Unicode block over the query's characters. These scripts have
disjoint ranges, so a count of which block dominates is exact — no model, no
library, and microseconds.
"""

from __future__ import annotations

# MSMARCO-XI shard code -> the Unicode block its script occupies.
_BLOCKS: tuple[tuple[str, int, int], ...] = (
    ("hin", 0x0900, 0x097F),  # Devanagari — Hindi, Marathi, Nepali, Sanskrit
    ("ben", 0x0980, 0x09FF),  # Bengali — also Assamese
    ("pan", 0x0A00, 0x0A7F),  # Gurmukhi
    ("guj", 0x0A80, 0x0AFF),
    ("ori", 0x0B00, 0x0B7F),
    ("tam", 0x0B80, 0x0BFF),
    ("tel", 0x0C00, 0x0C7F),
    ("kan", 0x0C80, 0x0CFF),
    ("mal", 0x0D00, 0x0D7F),
    ("urd", 0x0600, 0x06FF),  # Arabic script
)


def detect_script(text: str) -> str:
    """Return the MSMARCO-XI language code whose script dominates `text`.

    Falls back to "eng" for Latin script or when no Indic characters are
    present. Ties and mixed-script input resolve to whichever block has the
    most characters, which handles the common case of an Indic question
    containing a Latin acronym or digit.
    """
    counts: dict[str, int] = {}
    latin = 0
    for ch in text:
        code = ord(ch)
        if 0x41 <= code <= 0x7A:
            latin += 1
            continue
        for name, lo, hi in _BLOCKS:
            if lo <= code <= hi:
                counts[name] = counts.get(name, 0) + 1
                break

    if not counts:
        return "eng"
    best = max(counts, key=lambda k: counts[k])
    # A handful of Indic characters inside otherwise-Latin text (a
    # transliterated name, say) should not flip the whole query's language.
    return best if counts[best] >= latin else "eng"
