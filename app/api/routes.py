"""
NewsLens — API Routes

All endpoints live here. Business logic is delegated to core/ modules;
routes only handle HTTP concerns (validation, error wrapping, DB persistence).
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict, deque
from datetime import datetime
from typing import Optional
from functools import partial

from fastapi import APIRouter, Request, Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.api.schemas import (
    AnalyzeTextRequest, AnalyzeURLRequest, BatchAnalyzeRequest,
    AnalysisResponse, BatchAnalysisResponse, HealthResponse,
    OutletSummary, HistoryItem, PoliticalLean,
)
from app.core import preprocessor, fusion
from app.core.scraper import scrape_article
from app.core.model import model_singleton
from app.db.database import db_session
from app.db.models import Article, Outlet
from config import MAX_BATCH_SIZE

logger = logging.getLogger(__name__)
router = APIRouter()

# ── In-memory analytics (reset on restart; use DB for persistence) ────────────
_analytics: dict = {
    "total_requests":      0,
    "total_analyzed":      0,
    "errors":              0,
    "severity_dist":       defaultdict(int),
    "category_hits":       defaultdict(int),
    "recent_history":      deque(maxlen=25),
    "start_time":          datetime.utcnow().isoformat(),
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_lean(result: dict) -> PoliticalLean:
    return PoliticalLean(
        label=result["political_lean"],
        confidence=result["lean_confidence"],
        distribution=result.get("lean_distribution", {"left": 0.0, "center": 0.0, "right": 0.0}),
    )


def _to_response(result: dict) -> AnalysisResponse:
    return AnalysisResponse(
        **{k: v for k, v in result.items()
           if k not in ("text_hash", "political_lean", "lean_confidence",
                        "lean_distribution", "outlet_name", "source_url",
                        "article_title")},
        political_lean=_build_lean(result),
        timestamp=datetime.utcnow().isoformat(),
        source_url=result.get("source_url"),
        outlet_name=result.get("outlet_name"),
        article_title=result.get("article_title"),
    )


def _record(result: dict) -> None:
    """Update in-memory analytics after a successful analysis."""
    _analytics["total_analyzed"] += 1
    _analytics["severity_dist"][result["severity"]] += 1
    for cat, score in result["category_scores"].items():
        if score > 0.3:
            _analytics["category_hits"][cat] += 1
    _analytics["recent_history"].appendleft({
        "text_preview":  result["text"][:90] + ("…" if len(result["text"]) > 90 else ""),
        "bias_score":    result["bias_score"],
        "severity":      result["severity"],
        "severity_color": result["severity_color"],
        "political_lean": result["political_lean"],
        "outlet_name":   result.get("outlet_name"),
        "timestamp":     datetime.utcnow().isoformat(),
    })


def _persist_article(result: dict, db: Session) -> None:
    """Save article to DB and update outlet stats if a URL was provided."""
    # Skip if already stored (same text hash)
    if db.query(Article).filter_by(text_hash=result["text_hash"]).first():
        return

    outlet_id: Optional[int] = None
    domain = None
    if result.get("source_url") and result.get("outlet_name"):
        from urllib.parse import urlparse
        domain = urlparse(result["source_url"]).netloc.lower().removeprefix("www.")
        outlet = db.query(Outlet).filter_by(domain=domain).first()
        if not outlet:
            outlet = Outlet(name=result["outlet_name"], domain=domain)
            db.add(outlet)
            db.flush()
        outlet.update_stats(result["bias_score"], result["political_lean"])
        outlet_id = outlet.id

    article = Article(
        url=result.get("source_url"),
        text_hash=result["text_hash"],
        title=result.get("article_title"),
        body_preview=result["text"][:300],
        bias_score=result["bias_score"],
        severity=result["severity"],
        political_lean=result["political_lean"],
        lean_confidence=result["lean_confidence"],
        pattern_match_count=result["pattern_match_count"],
        roberta_used=int(result["roberta_used"]),
        outlet_id=outlet_id,
    )
    db.add(article)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/analyze", response_model=AnalysisResponse, summary="Analyse text for bias")
async def analyze_text(
    body: AnalyzeTextRequest,
    request: Request,
    db: Session = Depends(db_session),
):
    _analytics["total_requests"] += 1
    clean = preprocessor.clean(body.text)
    ok, err = preprocessor.validate(clean)
    if not ok:
        raise HTTPException(status_code=400, detail=err)

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, partial(fusion.analyze, clean))

    _record(result)
    try:
        _persist_article(result, db)
    except Exception as e:
        logger.warning("DB persist failed (non-fatal): %s", e)

    return _to_response(result)


@router.post("/analyze/url", response_model=AnalysisResponse, summary="Scrape a URL and analyse")
async def analyze_url(
    body: AnalyzeURLRequest,
    db: Session = Depends(db_session),
):
    _analytics["total_requests"] += 1

    scraped = await scrape_article(body.url)
    if scraped["error"] and not scraped["text"]:
        raise HTTPException(status_code=422, detail=scraped["error"])

    text = scraped["text"] or ""
    if scraped.get("title") and len(text) < 100:
        # Very short body — prepend headline so there's enough signal
        text = (scraped["title"] + ". " + text).strip()

    clean = preprocessor.clean(text)
    ok, err = preprocessor.validate(clean)
    if not ok:
        raise HTTPException(status_code=422, detail=f"Scraped content unusable: {err}")

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        partial(
            fusion.analyze,
            clean,
            scraped["url"],
            scraped["outlet_name"],
            scraped.get("title"),
        ),
    )

    _record(result)
    try:
        _persist_article(result, db)
    except Exception as e:
        logger.warning("DB persist failed (non-fatal): %s", e)

    return _to_response(result)


@router.post("/analyze/batch", response_model=BatchAnalysisResponse, summary="Analyse up to 10 texts")
async def analyze_batch(
    body: BatchAnalyzeRequest,
    db: Session = Depends(db_session),
):
    _analytics["total_requests"] += 1
    results = []
    loop = asyncio.get_event_loop()

    for raw_text in body.texts[:MAX_BATCH_SIZE]:
        clean = preprocessor.clean(str(raw_text))
        ok, err = preprocessor.validate(clean)
        if not ok:
            results.append({"success": False, "error": err})
            continue
        try:
            r = await loop.run_in_executor(None, partial(fusion.analyze, clean))
            _record(r)
            try:
                _persist_article(r, db)
            except Exception:
                pass
            results.append(_to_response(r))
        except Exception as e:
            _analytics["errors"] += 1
            results.append({"success": False, "error": str(e)})

    return BatchAnalysisResponse(success=True, count=len(results), results=results)


@router.get("/health", response_model=HealthResponse, summary="Service health")
async def health():
    return HealthResponse(
        status="healthy",
        service="NewsLens Bias Detector",
        version="1.0.0",
        roberta_status=model_singleton.status,
        roberta_enabled=model_singleton._loaded,
        uptime_since=_analytics["start_time"],
        timestamp=datetime.utcnow().isoformat(),
    )


@router.get("/stats", summary="Feature list and scoring info")
async def stats():
    return {
        "service":    "NewsLens Bias Detector",
        "version":    "1.0.0",
        "categories": {
            "cognitive_bias": [
                "Loaded Language", "Framing", "Epistemic Manipulation",
                "Anchoring", "Sensationalism",
            ],
            "political_lean": ["left", "center", "right"],
        },
        "scoring": {
            "method":           "Pattern engine (60%) + RoBERTa (40% when trained)",
            "word_matching":    "Regex word-boundary — no false substring matches",
            "roberta_status":   model_singleton.status,
        },
        "features": [
            "Text and URL (scrape-and-analyse) inputs",
            "Political lean detection (left / center / right)",
            "5 cognitive bias categories",
            "Per-phrase HTML annotation with colour coding",
            "Outlet leaderboard (URL analyses)",
            "Batch analysis (up to 10 texts)",
            "Persistent SQLite storage",
        ],
    }


@router.get("/analytics", summary="Usage statistics")
async def analytics():
    return {
        "total_requests":      _analytics["total_requests"],
        "total_analyzed":      _analytics["total_analyzed"],
        "errors":              _analytics["errors"],
        "severity_distribution": dict(_analytics["severity_dist"]),
        "top_categories": dict(
            sorted(_analytics["category_hits"].items(), key=lambda x: -x[1])
        ),
        "running_since":       _analytics["start_time"],
    }


@router.get("/history", summary="Last 25 analyses")
async def get_history():
    return {
        "success": True,
        "history": list(_analytics["recent_history"]),
        "count":   len(_analytics["recent_history"]),
    }


@router.delete("/history", summary="Clear analysis history")
async def clear_history():
    _analytics["recent_history"].clear()
    return {"success": True, "message": "History cleared."}


@router.get("/outlets", summary="Source outlet leaderboard")
async def get_outlets(db: Session = Depends(db_session)):
    outlets = (
        db.query(Outlet)
        .filter(Outlet.article_count > 0)
        .order_by(Outlet.avg_bias_score.desc())
        .limit(20)
        .all()
    )
    return {
        "success": True,
        "outlets": [
            OutletSummary(
                id=o.id,
                name=o.name,
                domain=o.domain,
                avg_bias_score=o.avg_bias_score,
                article_count=o.article_count,
                dominant_lean=o.dominant_lean(),
                lean_distribution=o.lean_distribution or {},
            )
            for o in outlets
        ],
    }
