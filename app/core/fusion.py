"""
NewsLens — Score Fusion

Combines the pattern engine (primary, rule-based) and the RoBERTa model
(supplementary, learned) into a single coherent analysis result.

When RoBERTa is unavailable the result is pattern-only — same quality as
the existing CogniBias app, but the architecture is ready to upgrade
transparently once the model is trained.

Weight rationale:
  60% pattern engine  — always present, interpretable, fast
  40% RoBERTa         — adds learned signal when available
"""

from __future__ import annotations

import re
import time
import hashlib
from typing import Optional

from app.core.pattern_engine import PatternEngine
from app.core.model import model_singleton

_pattern_engine = PatternEngine()   # shared singleton, stateless after init

# Known-outlet lean reference (AllSides / Ad Fontes Media consensus ratings)
# Used as a weak fallback signal when neither RoBERTa nor rules fire.
_OUTLET_LEAN: dict[str, tuple[str, float]] = {
    # (label, confidence)
    "aljazeera":         ("center", 0.18),
    "al jazeera":        ("center", 0.18),
    "bbc":               ("center", 0.20),
    "reuters":           ("center", 0.25),
    "ap news":           ("center", 0.25),
    "associated press":  ("center", 0.25),
    "bloomberg":         ("center", 0.20),
    "npr":               ("center", 0.18),
    "pbs":               ("center", 0.18),
    "nytimes":           ("left",   0.22),
    "new york times":    ("left",   0.22),
    "washington post":   ("left",   0.20),
    "guardian":          ("left",   0.22),
    "the guardian":      ("left",   0.22),
    "msnbc":             ("left",   0.28),
    "huffpost":          ("left",   0.25),
    "huffington post":   ("left",   0.25),
    "vox":               ("left",   0.25),
    "mother jones":      ("left",   0.28),
    "fox news":          ("right",  0.28),
    "foxnews":           ("right",  0.28),
    "breitbart":         ("right",  0.35),
    "daily wire":        ("right",  0.30),
    "newsmax":           ("right",  0.30),
    "the federalist":    ("right",  0.30),
    "wall street journal": ("right", 0.18),
    "wsj":               ("right",  0.18),
}


def _severity(score: float) -> tuple[str, str]:
    if score < 33:
        return "Low", "green"
    if score < 66:
        return "Medium", "orange"
    return "High", "red"


def analyze(text: str, source_url: Optional[str] = None, outlet_name: Optional[str] = None, article_title: Optional[str] = None) -> dict:
    """
    Full analysis pipeline.

    Args:
        text:         Clean article / headline text.
        source_url:   Original URL if scraped (for DB storage).
        outlet_name:  Outlet display name if known.
        article_title: Article headline if extracted by scraper.

    Returns a dict that maps directly to AnalysisResponse schema.
    """
    t0 = time.perf_counter()

    # 1. Pattern engine
    pat_raw    = _pattern_engine.analyze(text)
    pat_scored = _pattern_engine.score(pat_raw)
    overall    = pat_scored["overall"]          # 0–100
    cat_scores = pat_scored["categories"]       # dict[str, float 0-1]

    # 2. RoBERTa (optional) — dynamic fusion weights (Lever 2)
    model_out = model_singleton.predict(text)
    if model_out is not None:
        intensity  = model_out["bias_intensity"]   # 0–1
        confidence = model_out.get("lean_confidence", 0.5)
        word_count = len(text.split())

        # Dynamic weights: trust RoBERTa more on long text + high confidence
        if word_count < 50:
            w_pat, w_rob = 0.80, 0.20   # short text — patterns more reliable
        elif word_count > 300 or confidence > 0.85:
            w_pat, w_rob = 0.45, 0.55   # long/confident — RoBERTa leads
        else:
            w_pat, w_rob = 0.60, 0.40   # default balanced

        overall      = round(w_pat * overall + w_rob * intensity * 100, 1)
        roberta_used = True
    else:
        roberta_used = False

    # 3. Political lean — confidence-weighted blend of RoBERTa + pattern signals
    pat_lean = _pattern_engine.detect_lean(text)

    rob_label = model_out["political_lean"]   if model_out else "unknown"
    rob_conf  = model_out["lean_confidence"]  if model_out else 0.0
    rob_dist  = model_out.get("lean_distribution", {"left": 0.333, "center": 0.333, "right": 0.333}) if model_out else {"left": 0.333, "center": 0.333, "right": 0.333}

    pat_label = pat_lean["label"]
    pat_conf  = pat_lean["confidence"]
    pat_dist  = pat_lean["distribution"]

    # Outlet-based fallback signal (weak prior when other signals are absent)
    outlet_label, outlet_conf = "unknown", 0.0
    if outlet_name:
        key = outlet_name.lower().strip()
        if key in _OUTLET_LEAN:
            outlet_label, outlet_conf = _OUTLET_LEAN[key]
        else:
            # Partial match (e.g. "Aljazeera" → "aljazeera")
            for k, (lbl, cf) in _OUTLET_LEAN.items():
                if k in key or key in k:
                    outlet_label, outlet_conf = lbl, cf
                    break

    total_signal = rob_conf + pat_conf + outlet_conf

    if total_signal == 0:
        # No signal at all — cannot determine lean
        political_lean    = "unknown"
        lean_confidence   = 0.0
        lean_distribution = {"left": 0.333, "center": 0.333, "right": 0.333}
    else:
        # Weighted blend of RoBERTa + pattern + outlet signals
        w_rob    = rob_conf    / total_signal
        w_pat    = pat_conf    / total_signal
        w_outlet = outlet_conf / total_signal

        # Build outlet distribution (1.0 on winner class)
        out_dist = {"left": 0.0, "center": 0.0, "right": 0.0}
        if outlet_label != "unknown":
            out_dist[outlet_label] = 1.0

        blended: dict[str, float] = {}
        for cls in ("left", "center", "right"):
            r_val = rob_dist.get(cls, 0.333) if rob_label != "unknown" else 0.333
            p_val = pat_dist.get(cls, 0.333) if pat_label != "unknown" else 0.333
            o_val = out_dist[cls]
            blended[cls] = round(w_rob * r_val + w_pat * p_val + w_outlet * o_val, 3)

        political_lean = max(blended, key=lambda k: blended[k])
        # Scale confidence by total signal strength — a pure outlet-lookup
        # should not report 1.0 confidence; cap it at the raw signal sum.
        lean_confidence   = round(blended[political_lean] * min(1.0, total_signal), 3)
        lean_distribution = blended

    # 4. Severity
    severity, sev_color = _severity(overall)

    # 5. Annotation
    highlighted_html = _pattern_engine.build_highlighted_html(text, pat_raw["spans"])
    explanations     = _pattern_engine.get_category_explanations(pat_raw)
    pattern_count    = sum(v["count"] for v in pat_raw["categories"].values())

    # 6. Text hash (for DB deduplication)
    text_hash = hashlib.sha256(text.encode()).hexdigest()

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

    return {
        "success":             True,
        "text":                text,
        "text_hash":           text_hash,
        "bias_score":          overall,
        "severity":            severity,
        "severity_color":      sev_color,
        "political_lean":      political_lean,
        "lean_confidence":     lean_confidence,
        "lean_distribution":   lean_distribution,
        "category_scores":     cat_scores,
        "category_explanations": explanations,
        "highlighted_html":    highlighted_html,
        "pattern_match_count": pattern_count,
        "roberta_used":        roberta_used,
        "processing_ms":       elapsed_ms,
        "source_url":          source_url,
        "outlet_name":         outlet_name,
        "article_title":       article_title,
    }
