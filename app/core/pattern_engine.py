"""
NewsLens — Pattern Engine (Rule-Based Primary Signal)

Bias taxonomy expanded from the HelpfulProfessor.com taxonomy (35 media bias
types) to cover all text-detectable dimensions:

  ORIGINAL (5):
    Loaded Language, Framing, Epistemic Manipulation, Anchoring, Sensationalism

  ADDED (3):
    False Balance   — artificial both-sidesism that hides scientific consensus
    Whataboutism    — deflection / "real story" distraction patterns
    In-Group Framing — us-vs-them tribalism; "real people" vs "the elites"

Word-boundary regex approach (P0a) is preserved — prevents false positives
like 'panic' matching inside 'Hispanic'.
"""

import re
import html
from config import BIAS_CATEGORIES

# Lever 3 — negation suppression
# If any of these tokens appear within 3 words BEFORE a matched phrase,
# the match is discarded (e.g. "not a disaster", "never a crisis").
_NEGATION_WORDS: frozenset[str] = frozenset({
    "not", "no", "never", "neither", "nor",
    "hardly", "barely", "scarcely", "rarely",
    "without", "lack", "lacking", "lacks",
    "absent", "deny", "denies", "denied", "refute", "refutes",
})

# ── Pattern Database ──────────────────────────────────────────────────────────

BIAS_PATTERNS: dict = {
    "Loaded Language": {
        "positive_triggers": [
            "triumph", "victory", "heroic", "brave", "savior", "champion",
            "landmark", "historic", "breakthrough", "stunning", "remarkable",
            "massive", "unprecedented", "game-changing", "extraordinary",
            "glorious", "magnificent", "legendary", "patriotic", "noble",
        ],
        "negative_triggers": [
            "disaster", "crisis", "catastrophe", "outrageous",
            "scandal", "devastating", "alarming", "horrific", "terrifying",
            "failed", "collapsing", "desperate", "chaos", "destruction",
            "disgraceful", "shameful", "corrupt", "criminal", "toxic",
            "radical", "extreme", "dangerous agenda", "assault on",
        ],
        "fear_words": [
            "warning", "threat", "dangerous", "concern grows",
            "experts fear", "fears grow", "panic", "anxiety", "peril",
            "dire warning", "imminent threat", "clear and present danger",
        ],
        "weight": 1.6,
    },
    "Framing": {
        "victim_frame":    ["victim", "suffering", "innocent", "helpless", "targeted", "marginalized", "left behind", "forgotten"],
        "threat_frame":    ["threat", "attack", "invasion", "under siege", "at risk", "under assault", "existential"],
        "conflict_frame":  ["clash", "battle", "war", "fight", "vs.", "against", "struggle", "showdown", "confrontation", "standoff"],
        "opportunity_frame": ["opportunity", "potential", "promising", "bright future", "turning point", "silver lining"],
        "moral_panic":     ["moral decay", "degeneracy", "erosion of values", "society is collapsing", "children at risk", "corrupting youth", "family values under"],
        "weight": 1.4,
    },
    "Epistemic Manipulation": {
        "hedging": [
            "some say", "many believe", "critics argue", "experts suggest",
            "it is said", "reportedly", "allegedly", "supposedly",
            "might be", "could be", "may be", "appears to be",
            "is believed to", "is thought to", "rumored to",
        ],
        "authority_appeal": [
            "experts say", "scientists warn", "officials claim",
            "sources indicate", "according to reports", "insiders reveal",
            "anonymous sources", "sources close to", "high-level sources",
            "experts are warning", "top officials say",
        ],
        "universal_quantifiers": [
            "everyone knows", "nobody disputes", "all agree",
            "it is well known", "undoubtedly", "clearly", "obviously",
            "no reasonable person", "any honest observer",
        ],
        "manufactured_consensus": [
            "the science is settled", "there is no debate", "proven beyond doubt",
            "widely accepted that", "universally agreed", "beyond question",
        ],
        "weight": 1.0,
    },
    "Anchoring": {
        "anchor_words": [
            "only", "just", "mere", "as much as", "as little as",
            "up to", "down to", "starts at", "maximum",
            "at least", "no more than", "nearly", "almost",
        ],
        "weight": 0.8,
    },
    "Sensationalism": {
        "urgency_words": [
            "breaking", "urgent", "emergency", "exclusive", "bombshell",
            "explosive", "shocking revelation", "you won't believe",
            "developing story", "must see", "happening now",
        ],
        "hyperbole": [
            "worst ever", "best ever", "most dangerous", "totally",
            "absolutely", "completely", "utterly", "unbelievable",
            "mind-blowing", "jaw-dropping", "once in a generation",
            "unlike anything", "never before seen", "all-time",
        ],
        "doom_language": [
            "apocalyptic", "existential threat", "end of", "collapse of",
            "destroy", "obliterate", "annihilate", "wipe out",
            "extinction", "total collapse", "irreversible damage",
        ],
        "clickbait": [
            "the truth about", "what they don't want you to know",
            "here's why", "this is why", "the real reason",
            "will shock you", "changed everything", "before it's too late",
            "mainstream media won't", "they don't want you to see",
        ],
        "weight": 1.1,
    },

    # ── NEW CATEGORY 1 ────────────────────────────────────────────────────────
    "False Balance": {
        "bothsidesism": [
            "both sides", "on both sides", "from both perspectives",
            "the other side argues", "equally valid", "a matter of opinion",
            "reasonable people disagree", "it depends who you ask",
            "no clear answer", "the debate continues", "the jury is still out",
            "we may never know", "some scientists say", "experts are divided",
            "scientists disagree", "the science is unclear",
        ],
        "false_equivalence": [
            "just as guilty", "both are wrong", "no better than",
            "equally responsible", "same as", "no different from",
            "comparable to", "no worse than",
        ],
        "downplaying_consensus": [
            "so-called consensus", "alleged consensus", "disputed science",
            "contested claim", "controversial theory", "not everyone agrees",
            "despite what scientists say", "questioning the narrative",
            "challenging the official", "the establishment claims",
        ],
        "weight": 1.2,
    },

    # ── NEW CATEGORY 2 ────────────────────────────────────────────────────────
    "Whataboutism": {
        "deflection_phrases": [
            "but what about", "what about when", "and yet nobody",
            "why don't they talk about", "where was the outrage",
            "nobody mentioned", "conveniently ignored", "the media ignored",
            "no one covered", "they never reported on",
        ],
        "hidden_truth_framing": [
            "the real story", "the story they don't want", "the untold story",
            "what the media won't tell you", "what they're hiding",
            "the truth the media ignores", "buried by the press",
            "suppressed by mainstream", "the story being buried",
            "the narrative they push", "what they don't report",
        ],
        "blame_shifting": [
            "actually the fault of", "really caused by", "the true culprit",
            "who's really responsible", "the real villain", "follow the money",
            "look who benefits", "who stands to gain",
        ],
        "weight": 1.1,
    },

    # ── NEW CATEGORY 3 ────────────────────────────────────────────────────────
    "In-Group Framing": {
        "us_vs_them": [
            "real Americans", "real people", "ordinary people vs",
            "the elites", "the establishment", "the globalists",
            "the deep state", "the swamp", "the ruling class",
            "working people", "everyday citizens", "true patriots",
            "the silent majority", "the forgotten people",
        ],
        "out_group_demonization": [
            "they want to destroy", "they are coming for", "they hate",
            "their agenda", "the radical agenda", "the socialist agenda",
            "the far-left", "the far-right", "radical left", "radical right",
            "the liberal mob", "the conservative machine",
        ],
        "tribal_identity": [
            "one of us", "our values", "their values", "our way of life",
            "their ideology", "us and them", "us versus them",
            "stand with us", "against us", "betrayed by",
        ],
        "weight": 1.3,
    },
}

# ── Political Lean Lexicons ───────────────────────────────────────────────────
# Each phrase votes for a lean direction; overall score = (R - L) / (R + L + C + ε)

LEFT_LEAN_LEXICON: list[str] = [
    # Economic left
    "corporate greed", "wealth inequality", "income inequality", "billionaire",
    "billionaires", "tax the rich", "workers rights", "worker exploitation",
    "living wage", "universal healthcare", "medicare for all", "student debt",
    "housing crisis", "affordable housing", "union", "unionize",
    # Social left
    "systemic racism", "structural racism", "white privilege", "implicit bias",
    "marginalized communities", "marginalized groups", "oppressed", "equity",
    "reproductive rights", "abortion rights", "lgbtq rights", "trans rights",
    "climate justice", "environmental justice", "social justice",
    "defund the police", "police brutality", "racial justice",
    # Framing signals
    "progressive movement", "the resistance", "the left",
    "socialist", "socialism", "democratic socialism",
    "people over profit", "corporate media", "corporate interests",
    "working class", "the poor", "poverty trap",
]

RIGHT_LEAN_LEXICON: list[str] = [
    # Economic right
    "free market", "lower taxes", "tax cuts", "deregulation", "big government",
    "government overreach", "government spending", "fiscal responsibility",
    "small business owner", "job creator", "entrepreneurship",
    # Social right
    "traditional values", "family values", "religious freedom", "pro-life",
    "second amendment", "gun rights", "border security", "illegal immigration",
    "law and order", "tough on crime", "patriotism", "american exceptionalism",
    # Framing signals
    "deep state", "the swamp", "radical left", "far left", "socialist agenda",
    "woke ideology", "cancel culture", "mainstream media bias", "fake news",
    "real Americans", "silent majority", "globalists", "open borders",
    "america first", "make america great", "establishment elites",
    "conservative values", "the left wants", "leftist agenda",
]

CENTER_LEAN_LEXICON: list[str] = [
    # US bipartisan / moderate framing
    "bipartisan", "both parties", "across the aisle", "moderate",
    "independent voters", "swing voters", "common ground",
    # Evidence-based / objective reporting
    "according to data", "the statistics show", "researchers found",
    "officials confirmed", "the report states", "data indicates",
    "according to officials", "said in a statement", "official statement",
    "according to the report", "confirmed by", "announced that",
    # International diplomatic / neutral journalism
    "peace talks", "ceasefire", "negotiations", "diplomatic talks",
    "diplomatic efforts", "mediation", "mediators", "bilateral talks",
    "both delegations", "all parties", "both sides agreed",
    "peace process", "diplomatic solution", "multilateral",
    "international community", "humanitarian aid", "peacekeeping",
    "security council", "united nations", "UN resolution",
    "foreign minister", "foreign ministry", "state department",
    # Measured sourcing language typical of international wire / broadcast journalism
    "sources say", "sources told", "sources close to the",
    "according to sources", "as reported by", "citing sources",
    "government spokesperson", "press secretary", "official spokesperson",
    "prime minister said", "president said", "minister said",
    # Balance and factual reporting markers
    "fact-check", "independent investigation", "neutral observers",
    "non-partisan", "impartial", "objective analysis",
]


# Supplementary regex patterns for Anchoring (numerical)
ANCHORING_REGEX = [
    r"\$\d[\d,]*\.?\d*",
    r"\b\d+(?:\.\d+)?%",
    r"\b(?:up|down)\s+(?:to|from)\s+\$?\d+",
    r"\b(?:as much|as little)\s+as\s+\$?\d+",
    r"\b(?:maximum|minimum)\s+of\s+\$?\d+",
]

# Per-category highlight colours
SPAN_COLORS: dict[str, str] = {
    "Loaded Language":        "#ff6b6b",
    "Framing":                "#ffd166",
    "Epistemic Manipulation": "#06d6a0",
    "Anchoring":              "#38bdf8",
    "Sensationalism":         "#f97316",
    "False Balance":          "#a78bfa",
    "Whataboutism":           "#fb7185",
    "In-Group Framing":       "#34d399",
}


# ── Engine ────────────────────────────────────────────────────────────────────

class PatternEngine:
    """
    Rule-based cognitive bias detector.

    Pre-compiles one regex per phrase at construction time for speed.
    Uses \\b word-boundary matching to prevent false-substring hits.
    Multi-word phrases use lookahead/lookbehind on their outer boundaries.
    """

    def __init__(self) -> None:
        self._compiled: dict[str, re.Pattern] = {}
        for cat_cfg in BIAS_PATTERNS.values():
            for key, val in cat_cfg.items():
                if key == "weight" or not isinstance(val, list):
                    continue
                for phrase in val:
                    if phrase not in self._compiled:
                        escaped = re.escape(phrase.lower())
                        if " " in phrase:
                            pattern = r"(?<![\w])" + escaped + r"(?![\w])"
                        else:
                            pattern = r"\b" + escaped + r"\b"
                        self._compiled[phrase] = re.compile(pattern, re.IGNORECASE)

        # Pre-compile lean lexicons separately
        self._lean_compiled: dict[str, tuple[re.Pattern, str]] = {}
        for phrase in LEFT_LEAN_LEXICON:
            self._lean_compiled[phrase] = (self._compile_phrase(phrase), "left")
        for phrase in RIGHT_LEAN_LEXICON:
            self._lean_compiled[phrase] = (self._compile_phrase(phrase), "right")
        for phrase in CENTER_LEAN_LEXICON:
            self._lean_compiled[phrase] = (self._compile_phrase(phrase), "center")

    @staticmethod
    def _compile_phrase(phrase: str) -> re.Pattern:
        escaped = re.escape(phrase.lower())
        if " " in phrase:
            return re.compile(r"(?<![\w])" + escaped + r"(?![\w])", re.IGNORECASE)
        return re.compile(r"\b" + escaped + r"\b", re.IGNORECASE)

    # ── internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _is_negated(text: str, match_start: int, window: int = 3) -> bool:
        """
        Returns True if a negation word appears within `window` tokens
        immediately before the matched phrase.

        Strategy: slice the text to the left of the match, tokenise the last
        `window` words, check membership in _NEGATION_WORDS.
        """
        prefix = text[:match_start].lower()
        tokens = re.findall(r"\b\w+\b", prefix)
        nearby = tokens[-window:] if len(tokens) >= window else tokens
        return bool(frozenset(nearby) & _NEGATION_WORDS)

    def _scan(self, text: str, wordlist: list) -> list:
        findings = []
        seen: set[tuple] = set()
        for phrase in wordlist:
            pat = self._compiled.get(phrase)
            if not pat:
                continue
            for m in pat.finditer(text):
                if self._is_negated(text, m.start()):
                    continue                       # suppress negated match
                span = (m.start(), m.end())
                if span not in seen:
                    seen.add(span)
                    findings.append({"phrase": phrase, "start": m.start(), "end": m.end()})
        return findings

    # ── public API ────────────────────────────────────────────────────────────

    def analyze(self, text: str) -> dict:
        """
        Returns:
            categories: per-category findings
            spans:      list of (start, end, category) for HTML highlighting
        """
        results: dict = {}
        span_map: list = []

        for cat, cfg in BIAS_PATTERNS.items():
            cat_findings: list = []
            for key, val in cfg.items():
                if key == "weight":
                    continue
                if isinstance(val, list):
                    for f in self._scan(text, val):
                        f["sub_type"] = key
                        cat_findings.append(f)
                        span_map.append((f["start"], f["end"], cat))

            # Anchoring: numerical regex supplements
            if cat == "Anchoring":
                for pattern in ANCHORING_REGEX:
                    for m in re.finditer(pattern, text, re.IGNORECASE):
                        if not any(s == m.start() and e == m.end() for s, e, _ in span_map):
                            f = {"phrase": m.group(), "start": m.start(), "end": m.end(), "sub_type": "numerical_anchor"}
                            cat_findings.append(f)
                            span_map.append((m.start(), m.end(), cat))

            results[cat] = {
                "count":    len(cat_findings),
                "findings": cat_findings[:5],
            }

        return {"categories": results, "spans": span_map}

    def score(self, pattern_results: dict) -> dict:
        """
        Calibrated scoring with diminishing returns per category.
        Returns per-category scores [0,1] and overall [0,100].
        """
        total = 0.0
        max_total = 0.0
        cat_scores: dict[str, float] = {}

        for cat, cfg in BIAS_PATTERNS.items():
            weight = cfg["weight"]
            count  = pattern_results["categories"][cat]["count"]
            max_total += weight * 10

            cat_raw = sum(weight * (0.6 ** i) for i in range(count)) * 10
            cat_scores[cat] = round(min(cat_raw / (weight * 10), 1.0), 3)
            total += min(cat_raw, weight * 10)

        overall = round(min(total / max_total * 100, 100), 1)
        return {"overall": overall, "categories": cat_scores}

    def build_highlighted_html(self, text: str, spans: list) -> str:
        """Annotate text with colour-coded <mark> tags for each bias span."""
        if not spans:
            return html.escape(text)

        sorted_spans = sorted(set(spans), key=lambda s: s[0])
        parts: list[str] = []
        cursor = 0

        for start, end, cat in sorted_spans:
            if start < cursor:
                continue
            parts.append(html.escape(text[cursor:start]))
            color = SPAN_COLORS.get(cat, "#aaa")
            parts.append(
                f'<mark class="bias-span" data-cat="{cat}" '
                f'style="background:{color}22;border-bottom:2px solid {color};'
                f'border-radius:3px;padding:1px 2px;" title="{cat}">'
                f'{html.escape(text[start:end])}</mark>'
            )
            cursor = end

        parts.append(html.escape(text[cursor:]))
        return "".join(parts)

    def detect_lean(self, text: str) -> dict:
        """
        Rule-based political lean detection via lexicon voting.

        Counts left/right/center keyword hits (negation-suppressed),
        then computes a normalised confidence score.

        Returns:
            label        — "left" | "center" | "right" | "unknown"
            confidence   — float [0, 1]
            distribution — {"left": float, "center": float, "right": float}
            matched      — list of (phrase, lean) that fired
        """
        votes: dict[str, int] = {"left": 0, "center": 0, "right": 0}
        matched: list[tuple[str, str]] = []

        for phrase, (pat, lean) in self._lean_compiled.items():
            for m in pat.finditer(text):
                if not self._is_negated(text, m.start()):
                    votes[lean] += 1
                    matched.append((phrase, lean))

        total = sum(votes.values())
        if total == 0:
            return {
                "label": "unknown",
                "confidence": 0.0,
                "distribution": {"left": 0.333, "center": 0.333, "right": 0.333},
                "matched": [],
            }

        dist = {k: round(v / total, 3) for k, v in votes.items()}
        best_label = max(votes, key=lambda k: votes[k])
        best_count = votes[best_label]

        # Confidence: winner's share, scaled by how many signals fired
        # (more signals = more reliable — cap at 1.0)
        raw_conf = (best_count / total) * min(1.0, total / 4)
        confidence = round(min(raw_conf, 1.0), 3)

        # Require at least 1 signal and >40% share to make a call
        if total < 1 or (best_count / total) < 0.40:
            best_label = "unknown"
            confidence = 0.0

        return {
            "label":        best_label,
            "confidence":   confidence,
            "distribution": dist,
            "matched":      matched[:5],   # top 5 for debugging
        }

    def get_category_explanations(self, pattern_results: dict) -> dict[str, list[str]]:
        """Top-3 matched phrases per category, for the UI cards."""
        return {
            cat: [f["phrase"] for f in data["findings"]][:3]
            for cat, data in pattern_results["categories"].items()
        }
