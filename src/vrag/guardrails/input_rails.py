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
            # `(a\s+)?` alone missed "how do i build **an** explosive" — caught by
            # the red-team suite on its first run, not by any hand-written test.
            r"\b(how\s+(to|do\s+i)\s+(make|build|synthes\w*|construct)\s+"
            r"(a|an|the)?\s*(bomb|explosive|nerve\s+agent|sarin|ricin|meth\w*|napalm))",
            re.IGNORECASE,
        ),
    ),
    (
        "self_harm",
        # Covers the *question* form. Stated intent is a separate rule below,
        # because the two need different responses.
        re.compile(r"\b(how\s+to\s+(kill|hurt|harm)\s+(myself|yourself)|commit\s+suicide)\b", re.IGNORECASE),
    ),
    (
        "self_harm_intent",
        # Someone saying they intend to hurt themselves, rather than asking how.
        #
        # Found on the deployed page: "i will kill myself" was ANSWERED, citing
        # a passage about cardio workouts. `self_harm` only matched the "how
        # to …" phrasing, and the threat rule below deliberately excludes
        # "myself" because its subject is other people.
        #
        # This is the one rail whose refusal message is not a refusal — see
        # _UNSAFE_REASONS. Getting the wording right matters more here than
        # anywhere else in the file.
        re.compile(
            r"\bi\s*(?:'?m\s+gonna|'?m\s+going\s+to|am\s+going\s+to|'?ll|will|"
            r"wanna|want\s+to|gonna|feel\s+like|am\s+thinking\s+of|"
            r"'?m\s+thinking\s+(?:of|about))\s+"
            r"(?:kill(?:ing)?|hurt(?:ing)?|harm(?:ing)?|cut(?:ting)?)\s+myself\b"
            r"|\bi\s+(?:want\s+to|wanna)\s+die\b(?!\s+my\s+hair)"
            r"|\bi\s+(?:want\s+to|wanna)\s+end\s+(?:my\s+life|it\s+all)\b"
            r"|\bi\s+(?:do\s?n[o']?t|dont)\s+want\s+to\s+(?:live|be\s+alive)\b"
            r"|\bthinking\s+about\s+(?:suicide|killing\s+myself|ending\s+my\s+life)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "self_harm_method",
        # Dosage-to-harm questions. Unlike the rails above these read as
        # ordinary medical queries, and the corpus *does* hold medical
        # passages — so the evidence rail will not decline them the way it
        # declines bomb-making, where the corpus simply has nothing to say.
        #
        # Narrow on purpose: "paracetamol dosage for a child" and "what is the
        # maximum daily dose of ibuprofen" are normal questions and stay
        # answerable. Only lethal/fatal/overdose framings match.
        re.compile(
            r"\b(?:lethal|fatal|deadly|toxic)\s+(?:dose|dosage|amount|quantity)\b"
            r"|\bhow\s+(?:much|many)\s+\w+\s+(?:to|would)\s+(?:kill|overdose)\b"
            r"|\bhow\s+(?:much|many)\s+\w+\s+(?:is|are)\s+(?:lethal|fatal|deadly)\b"
            r"|\boverdose\s+(?:on|amount|how\s+much)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "csam",
        re.compile(r"\b(child|minor|underage)\s+(porn\w*|sexual|nude)", re.IGNORECASE),
    ),
    (
        "targeted_violence",
        # Keyed on the *object* being a person, not on the verb.
        #
        # The previous rule was `how to (kill|murder|poison) <anything>` with a
        # three-word exemption list, which refused "how to kill weeds in my
        # lawn" and "how to kill bacteria in water" — ordinary gardening and
        # hygiene questions this corpus can answer. Enumerating harmless
        # objects is unwinnable; enumerating people is small and stable.
        re.compile(
            r"\bhow\s+(?:to|do\s+i)\s+(?:kill|murder|poison|stab|strangle|behead)\s+"
            r"(?:someone|somebody|a\s+person|people|persons|him|her|them|"
            r"a\s+human|humans|my\s+(?:wife|husband|partner|ex|boss|neighbou?r|"
            r"teacher|classmate|co-?worker|colleague|sister|brother|mother|father|"
            r"mom|dad|parents|friend|roommate|family))\b",
            re.IGNORECASE,
        ),
    ),
    (
        "violent_threat",
        # A stated intention to hurt a named party. Found by testing the
        # deployed page: "i wanna kill you" was ANSWERED — with song lyrics,
        # because the corpus contains a passage about wanting to scream. Every
        # violence rule above requires the phrasing "how to …", so a direct
        # threat matched nothing at all.
        #
        # Anchored on first person + intent + harm verb + a person object, so
        # "i want to kill a process" and "i'm going to kill this bug" are
        # untouched.
        re.compile(
            r"\bi\s*(?:'?m\s+gonna|'?m\s+going\s+to|am\s+going\s+to|'?ll|will|"
            r"wanna|want\s+to|gonna)\s+"
            r"(?:kill|murder|stab|shoot|strangle|behead|hurt|harm|beat\s+up|rape)\s+"
            r"(?:you|u|him|her|them|someone|somebody|"
            r"my\s+(?:wife|husband|partner|ex|boss|neighbou?r|teacher|classmate|"
            r"co-?worker|colleague|sister|brother|mother|father|mom|dad|parents|"
            r"friend|roommate|family))\b",
            re.IGNORECASE,
        ),
    ),
    (
        "harassment",
        # "kill yourself" as an instruction to another person. The self_harm
        # rule only covers it behind "how to".
        re.compile(r"\b(?:go\s+)?(?:kill|hang|shoot)\s+your\s?self\b", re.IGNORECASE),
    ),
    # ---- Indic ------------------------------------------------------------
    #
    # Every rule above is English-only, which was a hole the size of the
    # product: this corpus is Hindi, Bengali and Tamil, and "मैं खुद को मार
    # डालूंगा" ("I will kill myself") passed straight through to retrieval.
    #
    # These match on high-signal *terms* rather than sentence grammar. Indic
    # languages are morphologically rich with free word order, so a regex that
    # tries to parse subject-intent-verb-object the way the English rules do
    # would be both unreliable and unmaintainable by anyone who cannot read the
    # output. Terms like आत्महत्या / আত্মহত্যা / தற்கொலை ("suicide") carry the
    # signal on their own.
    #
    # Deliberately narrower than the English set, and it should be read by a
    # native speaker of each script before anyone calls this coverage rather
    # than mitigation. What it must not do is let the severe cases through
    # silently, which is what it was doing.
    (
        "self_harm_indic",
        re.compile(
            # suicide (hin) · (ben) · (tam), and "kill/finish myself" phrasings
            r"आत्महत्या|खुदकुशी|आत्म\s*हत्या"
            r"|खुद\s*को\s*(?:मार|खत्म)|अपने\s*आप\s*को\s*मार"
            r"|मरना\s*चाहत|मर\s*जाना\s*चाहत|जान\s*दे\s*दूं"
            r"|আত্মহত্যা|নিজেকে\s*মের|মরতে\s*চাই|মরে\s*যেতে\s*চাই"
            r"|தற்கொலை|என்னை\s*கொல்|சாக\s*வேண்டும்|இறக்க\s*வேண்டும்"
        ),
    ),
    (
        "violence_indic",
        re.compile(
            r"तुम्हें\s*मार|तुझे\s*मार|मार\s*डालूंगा|मार\s*डालूँगा|जान\s*से\s*मार"
            r"|তোমাকে\s*মের|তোকে\s*মের|মেরে\s*ফেলব"
            r"|உன்னை\s*கொல்|கொன்று\s*விடுவேன்"
        ),
    ),
    (
        "weapons_indic",
        re.compile(
            r"बम\s*(?:कैसे|बनान|बनाय|बनाने)|विस्फोटक\s*(?:कैसे|बनान)"
            r"|বোমা\s*(?:কিভাবে|বানা|তৈরি)"
            r"|குண்டு\s*(?:எப்படி|செய்)|வெடிகுண்டு"
        ),
    ),
]

# Refusal wording, per rail.
#
# Every unsafe rail but one gets the same flat sentence, which is correct: the
# system declines and says nothing more. Self-harm is different. Someone typing
# "i will kill myself" is not making a request to be declined, and answering
# them with "this falls outside what this system will answer" is the wrong
# thing to put on a screen. The corpus still cannot help them — but a pointer
# to someone who can costs nothing and is the only responsible output here.
#
# Indian numbers first (this corpus and its users are Indic), then a directory
# that resolves anywhere else.
_SELF_HARM_REASON = (
    "This system cannot help with this, and it is not the right thing to ask a "
    "search index. If you are thinking about harming yourself, please talk to "
    "someone now — Tele-MANAS (India): 14416 or 1800-891-4416, free and open "
    "24/7. Outside India, findahelpline.com lists services by country. If you "
    "are in immediate danger, please contact your local emergency number."
)
_GENERIC_UNSAFE_REASON = "This request falls outside what this system will answer."
_UNSAFE_REASONS: dict[str, str] = {
    "self_harm": _SELF_HARM_REASON,
    "self_harm_intent": _SELF_HARM_REASON,
    "self_harm_indic": _SELF_HARM_REASON,
    "self_harm_method": _SELF_HARM_REASON,
}

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

# Questions about the asker's own history or present situation. A static
# corpus cannot answer these *however good retrieval is* — and similarity
# search makes them look answerable, because "what did I have for breakfast"
# genuinely retrieves passages about breakfast. The evidence rail cannot catch
# that: the topic matches, only the scope is impossible.
#
# Deliberately narrow. A blanket rule on "my" or "I" would reject real MS MARCO
# traffic ("how much does my dog need to eat" is a generic, answerable
# question). These patterns match only first-person *experiential* framing —
# past events the asker participated in, and present facts about the asker —
# which no web-passage corpus contains.
_PERSONAL_SCOPE: list[tuple[str, re.Pattern]] = [
    ("personal_history", re.compile(
        r"\b(what|who|where|when|why|how)\s+(did|have|had)\s+i\b", re.IGNORECASE)),
    ("personal_state", re.compile(
        r"\b(who|where)\s+am\s+i\b|\bam\s+i\s+(sitting|standing|currently)\b", re.IGNORECASE)),
    ("personal_possession", re.compile(
        r"\bmy\s+(password|bank\s+account|balance|house\s+keys?|phone|inbox|"
        r"manager|calendar|current\s+location)\b", re.IGNORECASE)),
]

# Conversational openers are a distinct class of input, not a retrieval failure.
#
# Found by testing the deployed Space: "Hello, what is up?" was ANSWERED with
# grounding 1.00, citing a passage that begins "Hello there, When user logs in
# to Citrix web Interface" — and "Hey hey hey, what's up" was answered from
# "...that's when it shows up on the balance sheet."
#
# No score threshold fixes that. BM25 matches the literal tokens, the passages
# genuinely contain them, and the cross-encoder rates the surface overlap
# highly, so the relevance floor never fires. Raising the floor high enough to
# catch these would decline real questions instead.
#
# The rule is all-or-nothing on purpose: *every* token must be conversational,
# and at least one must actually be a greeting. "what is a hello world program"
# and "how are you supposed to file taxes" both pass, because they carry content
# words these sets do not contain; a bare "a lot" is not a greeting either,
# because nothing in it is.
#
# Same character class as the lexical tokeniser — a plain `\w+` would shred
# Indic greetings into consonant fragments and match nothing.
_CHITCHAT_TOKEN = re.compile(r"[\wऀ-෿̀-ͯ]+", re.UNICODE)

# Words that signal a greeting on their own.
_GREETING_WORDS = frozenset(
    """
    hi hii hiii hello helo hey heyy heya yo yoo hiya howdy sup wassup whatsup
    hola greetings namaste namaskar salaam
    morning afternoon evening night
    thanks thank thankyou thx ty
    bye goodbye cya
    नमस्ते नमस्कार हाय हैलो
    নমস্কার হাই
    வணக்கம் ஹாய்
    """.split()
)

# Words carrying no topic of their own. Allowed *alongside* a greeting, never
# sufficient on their own — otherwise "what is it" would be refused as a
# greeting instead of being retrieved against and honestly abstained on.
_FILLER_WORDS = frozenset(
    """
    good great nice fine cool ok okay okey yes no
    how are you u ur is it going doing r s m a an lot much so very
    what whats there please welcome later up
    mate bro dude buddy friend sir maam madam
    कैसे हो आप क्या हाल है
    কেমন আছেন আছো তুমি আপনি
    எப்படி இருக்கிறீர்கள்
    """.split()
)

# Openers built entirely from filler, where the *sequence* is the greeting.
# Matched on the whole normalised token string, so "what's up" ("what s up")
# and "whats up" both land, while "what is up with bond yields" does not.
_GREETING_PHRASES = frozenset(
    {
        "how are you", "how are you doing", "how r u", "how are u",
        "what s up", "whats up", "what is up", "sup",
        "how is it going", "how s it going", "hows it going",
        "who are you", "what are you", "what can you do",
    }
)


def _is_chitchat(cleaned: str) -> bool:
    tokens = _CHITCHAT_TOKEN.findall(cleaned.lower())
    if not tokens:
        return False
    if " ".join(tokens) in _GREETING_PHRASES:
        return True
    return any(t in _GREETING_WORDS for t in tokens) and all(
        t in _GREETING_WORDS or t in _FILLER_WORDS for t in tokens
    )


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

    # Ahead of the length check on purpose. "hi" is two characters, but calling
    # it malformed and "too short to act on" is the wrong explanation for an
    # input the system understands perfectly well — and it made the same class
    # of query report two different reasons depending on how it was spelled.
    if _is_chitchat(cleaned):
        return GuardVerdict(
            allowed=False,
            decision=Decision.REJECT_OFF_TOPIC,
            reason=(
                "That is a greeting rather than a question. This system answers "
                "only from an indexed passage corpus — ask it something those "
                "passages could contain."
            ),
            triggered=["chitchat"],
        )

    # A query with no word characters at all cannot be retrieved against: both
    # BM25 and the tokeniser see an empty term list, so retrieval returns
    # essentially arbitrary passages that then score above the abstention floor.
    # Found by typing "🎉🎉🎉", which came back ANSWER with grounding 1.00 and
    # cited "Published: January 21, 1993." The length check misses it because
    # three emoji are three characters.
    if not _CHITCHAT_TOKEN.search(cleaned):
        return GuardVerdict(
            allowed=False,
            decision=Decision.REJECT_MALFORMED,
            reason="The query contains no words to search for.",
            triggered=["no_searchable_terms"],
        )

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
                reason=_UNSAFE_REASONS.get(name, _GENERIC_UNSAFE_REASON),
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

    for name, pattern in _PERSONAL_SCOPE:
        if pattern.search(cleaned):
            return GuardVerdict(
                allowed=False,
                decision=Decision.REJECT_OFF_TOPIC,
                reason=(
                    "This asks about you personally. The system can only answer "
                    "from an indexed passage corpus, which contains no "
                    "information about the person asking."
                ),
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
