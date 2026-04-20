"""
NewsLens — Central configuration.
All tuneable values live here; env vars override defaults via .env.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR        = Path(__file__).parent
MODELS_DIR      = BASE_DIR / "models"
DATA_DIR        = BASE_DIR / "data"
TEMPLATES_DIR   = BASE_DIR / "templates"
STATIC_DIR      = BASE_DIR / "static"
LOGS_DIR        = BASE_DIR / "logs"

# ── Model ─────────────────────────────────────────────────────────────────────
ROBERTA_BASE      = os.getenv("ROBERTA_BASE", "roberta-base")
MODEL_CHECKPOINT  = str(MODELS_DIR / "newslens_best.pt")
MODEL_MAX_LENGTH  = 512
# True only when a trained checkpoint file actually exists
MODEL_ENABLED     = Path(MODEL_CHECKPOINT).exists()

BIAS_CATEGORIES = [
    "Loaded Language",
    "Framing",
    "Epistemic Manipulation",
    "Anchoring",
    "Sensationalism",
    "False Balance",
    "Whataboutism",
    "In-Group Framing",
]

LEAN_LABELS = ["left", "center", "right"]   # index 0,1,2

# ── Lever 4: Temperature Scaling ──────────────────────────────────────────────
# T > 1 softens overconfident lean predictions; T = 1 is identity (no change).
# After fitting calibration on a held-out set, update this value in .env or here.
# Fit with: python ml/calibrate_temperature.py
_TEMPERATURE_FILE = BASE_DIR / "models" / "temperature.json"
if _TEMPERATURE_FILE.exists():
    import json as _json
    LEAN_TEMPERATURE: float = float(_json.loads(_TEMPERATURE_FILE.read_text()).get("T", 1.3))
else:
    LEAN_TEMPERATURE: float = float(os.getenv("LEAN_TEMPERATURE", "1.3"))
# 1.3 is a reasonable prior for RoBERTa fine-tuned on imbalanced text classes

# ── Database ──────────────────────────────────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR}/newslens.db")

# ── API ───────────────────────────────────────────────────────────────────────
ALLOWED_ORIGINS   = os.getenv("ALLOWED_ORIGINS", "*").split(",")
MAX_TEXT_LENGTH   = int(os.getenv("MAX_TEXT_LENGTH", "10000"))
MAX_BATCH_SIZE    = int(os.getenv("MAX_BATCH_SIZE", "10"))
RATE_ANALYZE      = os.getenv("RATE_ANALYZE", "30/minute")
RATE_BATCH        = os.getenv("RATE_BATCH",   "5/minute")

# ── Scraper ───────────────────────────────────────────────────────────────────
SCRAPER_TIMEOUT    = int(os.getenv("SCRAPER_TIMEOUT", "15"))
SCRAPER_USER_AGENT = "NewsLens-BiasDetector/1.0 (+https://github.com/educational)"

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
