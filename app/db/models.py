"""
NewsLens — SQLAlchemy ORM Models

Articles: every analysed piece of text (from paste or URL scrape).
Outlets:  per-domain aggregate statistics (source leaderboard).

Outlet stats are updated in-place after each URL analysis so the
leaderboard reflects a running average — no separate aggregation job needed.
"""

import json
from datetime import datetime

from sqlalchemy import (
    Column, Integer, String, Float, DateTime,
    Text, ForeignKey, JSON, UniqueConstraint,
)
from sqlalchemy.orm import relationship

from app.db.database import Base


class Outlet(Base):
    __tablename__ = "outlets"

    id            = Column(Integer, primary_key=True, index=True)
    name          = Column(String(100), nullable=False)
    domain        = Column(String(100), unique=True, nullable=False, index=True)
    avg_bias_score = Column(Float, default=0.0)
    article_count = Column(Integer, default=0)
    # Running distribution of political leans across all articles
    lean_distribution = Column(
        JSON,
        default=lambda: {"left": 0, "center": 0, "right": 0, "unknown": 0},
    )
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    articles = relationship("Article", back_populates="outlet", lazy="dynamic")

    def update_stats(self, bias_score: float, lean: str) -> None:
        """Incrementally update running average after a new article is analysed."""
        n = self.article_count
        self.avg_bias_score = round(
            (self.avg_bias_score * n + bias_score) / (n + 1), 2
        )
        self.article_count += 1
        dist = dict(self.lean_distribution or {})
        dist[lean] = dist.get(lean, 0) + 1
        self.lean_distribution = dist
        self.updated_at = datetime.utcnow()

    def dominant_lean(self) -> str:
        """Return the most-seen political lean for this outlet."""
        dist = self.lean_distribution or {}
        # Exclude 'unknown' from dominant lean calculation
        known = {k: v for k, v in dist.items() if k != "unknown"}
        if not known or max(known.values(), default=0) == 0:
            return "unknown"
        return max(known, key=known.get)


class Article(Base):
    __tablename__ = "articles"

    id            = Column(Integer, primary_key=True, index=True)
    url           = Column(String(2000), unique=True, nullable=True, index=True)
    text_hash     = Column(String(64), unique=True, nullable=False, index=True)
    title         = Column(String(500), nullable=True)
    body_preview  = Column(String(300), nullable=True)   # first 300 chars
    bias_score    = Column(Float, nullable=False)
    severity      = Column(String(10), nullable=False)
    political_lean = Column(String(20), default="unknown")
    lean_confidence = Column(Float, default=0.0)
    pattern_match_count = Column(Integer, default=0)
    roberta_used  = Column(Integer, default=0)           # 0/1 boolean (SQLite compat)
    outlet_id     = Column(Integer, ForeignKey("outlets.id"), nullable=True)
    analyzed_at   = Column(DateTime, default=datetime.utcnow, index=True)

    outlet = relationship("Outlet", back_populates="articles")
