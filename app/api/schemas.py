"""
NewsLens — Pydantic Request / Response Schemas

Pydantic v2. All API input is validated here before it touches business logic.
Output schemas document the exact shape callers can depend on.
"""

from __future__ import annotations

from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field, field_validator, HttpUrl


# ── Requests ──────────────────────────────────────────────────────────────────

class AnalyzeTextRequest(BaseModel):
    text: str = Field(..., min_length=10, max_length=10_000, description="Article or headline text to analyse.")

    @field_validator("text")
    @classmethod
    def strip_and_check(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("text must not be blank.")
        return v


class AnalyzeURLRequest(BaseModel):
    url: str = Field(..., description="Full article URL (must start with https://).")

    @field_validator("url")
    @classmethod
    def must_be_http(cls, v: str) -> str:
        v = v.strip()
        if not v.startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")
        return v


class BatchAnalyzeRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=10, description="1–10 texts to analyse in a single call.")


# ── Lean sub-model ────────────────────────────────────────────────────────────

class PoliticalLean(BaseModel):
    label:        str              # "left" | "center" | "right" | "unknown"
    confidence:   float            # 0.0–1.0
    distribution: dict[str, float] # {"left": p, "center": p, "right": p}


# ── Response ──────────────────────────────────────────────────────────────────

class AnalysisResponse(BaseModel):
    success:               bool
    text:                  str
    bias_score:            float          # 0–100
    severity:              str            # "Low" | "Medium" | "High"
    severity_color:        str            # "green" | "orange" | "red"
    political_lean:        PoliticalLean
    category_scores:       dict[str, float]     # each 0–1
    category_explanations: dict[str, list[str]] # matched phrases per category
    highlighted_html:      str
    pattern_match_count:   int
    roberta_used:          bool
    processing_ms:         float
    timestamp:             str
    source_url:            Optional[str] = None
    outlet_name:           Optional[str] = None
    article_title:         Optional[str] = None


class BatchAnalysisResponse(BaseModel):
    success: bool
    count:   int
    results: list[AnalysisResponse | dict]   # dict used for per-item errors


class HealthResponse(BaseModel):
    status:        str
    service:       str
    version:       str
    roberta_status: str
    roberta_enabled: bool
    uptime_since:  str
    timestamp:     str


class OutletSummary(BaseModel):
    id:              int
    name:            str
    domain:          str
    avg_bias_score:  float
    article_count:   int
    dominant_lean:   str
    lean_distribution: dict[str, int]


class HistoryItem(BaseModel):
    text_preview: str
    bias_score:   float
    severity:     str
    severity_color: str
    political_lean: str
    timestamp:    str
    outlet_name:  Optional[str] = None
