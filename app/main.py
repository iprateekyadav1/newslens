"""
NewsLens — FastAPI Application Entry Point

Startup order:
  1. create_tables()      — ensure DB schema exists
  2. model_singleton.load() — load RoBERTa if checkpoint present
  3. mount /static         — serve CSS/JS
  4. mount /               — serve dashboard HTML
  5. mount /api            — REST endpoints
"""

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from config import (
    ALLOWED_ORIGINS, LOG_LEVEL,
    TEMPLATES_DIR, STATIC_DIR,
    RATE_ANALYZE, RATE_BATCH,
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("newslens")


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ───────────────────────────────────────────────────────────────
    logger.info("=" * 55)
    logger.info("  NewsLens Bias Detector  v1.0.0")
    logger.info("=" * 55)

    from app.db.database import create_tables
    create_tables()
    logger.info("Database tables ready.")

    from app.core.model import model_singleton
    model_singleton.load()

    yield

    # ── Shutdown ──────────────────────────────────────────────────────────────
    logger.info("NewsLens shutting down.")


# ── Rate limiter ──────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="NewsLens Bias Detector",
    description=(
        "Detects cognitive bias and political lean in news articles. "
        "Pattern engine + RoBERTa ensemble. URL scraping supported."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "Authorization"],
)


# ── Static files ──────────────────────────────────────────────────────────────
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Dashboard (SPA catch-all) ─────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def dashboard():
    index = TEMPLATES_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return JSONResponse({"message": "NewsLens API running. See /api/docs."})


# ── API router ────────────────────────────────────────────────────────────────
from app.api.routes import router as api_router   # noqa: E402 (after app creation)
app.include_router(api_router, prefix="/api")


# ── Dev entrypoint ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=5001,
        reload=True,
        log_level=LOG_LEVEL.lower(),
    )
