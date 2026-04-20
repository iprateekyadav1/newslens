# NewsLens — AI Media Bias Detector

Detects **cognitive bias** and **political lean** in news articles.  
Paste text or drop a URL — the scraper fetches and analyses automatically.

---

## Architecture

```
Input: text  OR  article URL
            ↓
     [Scraper: trafilatura]        ← URL path only
            ↓
      [Preprocessor]               ← strip HTML, normalise whitespace
         ↙          ↘
[Pattern Engine]   [RoBERTa-base]
 (always on)       (after training)
         ↘          ↙
      [Score Fusion]
   60% pattern + 40% RoBERTa
            ↓
Output:
  bias_score        0–100
  severity          Low / Medium / High
  political_lean    left / center / right + confidence
  category_scores   5 cognitive bias categories (0–1)
  highlighted_html  annotated text with colour-coded spans
  outlet_record     saved to SQLite if URL was provided
```

### Bias Categories

| # | Category | Signal |
|---|---|---|
| 1 | **Loaded Language** | Emotionally charged words ("devastating", "heroic") |
| 2 | **Framing** | Victim / threat / conflict / opportunity framing |
| 3 | **Epistemic Manipulation** | False authority, hedging ("some say", "reportedly") |
| 4 | **Anchoring** | Numerical framing ($499 → $99, percentage anchors) |
| 5 | **Sensationalism** | Urgency and doom language ("BREAKING", "apocalyptic") |
| + | **Political Lean** | Left / Center / Right — from RoBERTa transformer head |

---

## Quick Start

### Option A — Windows (double-click)

```
start.bat
```

The script creates a virtual environment, installs dependencies, and starts the server.

### Option B — Manual

```bash
# 1. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / Mac

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy environment file
copy .env.example .env

# 4. Start server
python -m uvicorn app.main:app --host 0.0.0.0 --port 5001 --reload
```

Open: **http://localhost:5001**

> **Without a trained model**, the server runs in pattern-engine-only mode.  
> This is fully functional — political lean will show "Undetermined" until you train.

---

## Train the RoBERTa Model

```bash
# Activate your virtual environment first, then:
python ml/train.py

# Options:
python ml/train.py --epochs 10 --batch_size 8 --lr 2e-5 --patience 5
```

Training uses **synthetic data** (~400 examples) by default.  
For significantly better results, add the real MBIC dataset:

1. Download `mbic.csv` from https://github.com/Media-Bias-Group/Media-Bias-Identification
2. Place it at `data/raw/mbic.csv`
3. Re-run `python ml/train.py`

The best checkpoint is saved to `models/newslens_best.pt` automatically.  
On next server start, RoBERTa loads automatically and contributes 40% of the final score.

---

## API Reference

Base URL: `http://localhost:5001`  
Interactive docs: `http://localhost:5001/api/docs`

### POST `/api/analyze`
Analyse text for bias.

```bash
curl -X POST http://localhost:5001/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Experts warn of terrifying new crisis threatening millions."}'
```

Response:
```json
{
  "success": true,
  "bias_score": 68.4,
  "severity": "High",
  "severity_color": "red",
  "political_lean": {
    "label": "right",
    "confidence": 0.71,
    "distribution": {"left": 0.11, "center": 0.18, "right": 0.71}
  },
  "category_scores": {
    "Loaded Language": 0.82,
    "Framing": 0.55,
    "Epistemic Manipulation": 0.71,
    "Anchoring": 0.0,
    "Sensationalism": 0.60
  },
  "highlighted_html": "...",
  "pattern_match_count": 5,
  "roberta_used": false,
  "processing_ms": 12.4,
  "timestamp": "2026-04-08T10:00:00"
}
```

### POST `/api/analyze/url`
Scrape and analyse a news article URL.

```bash
curl -X POST http://localhost:5001/api/analyze/url \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.reuters.com/article/..."}'
```

### POST `/api/analyze/batch`
Analyse up to 10 texts in one call.

```bash
curl -X POST http://localhost:5001/api/analyze/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Headline one...", "Headline two..."]}'
```

### GET `/api/outlets`
Source outlet leaderboard (populated by URL analyses).

### GET `/api/history`
Last 25 analyses.

### GET `/api/health`
Service status and model load state.

---

## Project Structure

```
newslens/
├── app/
│   ├── main.py              FastAPI entry point, lifespan, CORS
│   ├── api/
│   │   ├── routes.py        All API endpoints
│   │   └── schemas.py       Pydantic request/response schemas
│   ├── core/
│   │   ├── pattern_engine.py  Rule-based bias detection (primary signal)
│   │   ├── model.py           RoBERTa singleton + inference
│   │   ├── fusion.py          Score fusion (60% pattern + 40% RoBERTa)
│   │   ├── preprocessor.py    Text cleaning and validation
│   │   └── scraper.py         trafilatura URL scraper
│   └── db/
│       ├── database.py        SQLAlchemy engine + session factory
│       └── models.py          Article and Outlet ORM models
├── ml/
│   ├── architecture.py      NewsLensClassifier (RoBERTa + attention pooling)
│   ├── dataset.py           Synthetic corpus + MBIC loader + DataLoader
│   └── train.py             Training loop, multi-task loss, checkpointing
├── templates/
│   └── index.html           Dashboard (dark/light mode, URL tab, lean indicator)
├── static/
│   ├── css/style.css        Design system
│   └── js/main.js           Analysis, rendering, drawers (DOM-safe, no innerHTML)
├── models/                  Trained checkpoints (newslens_best.pt after training)
├── data/
│   └── raw/                 Place mbic.csv here for real-data training
├── config.py                Central configuration (env var overrides)
├── requirements.txt
├── .env.example
├── Dockerfile               Multi-stage, non-root user, gunicorn + uvicorn
└── start.bat                Windows one-click launcher
```

---

## Docker

```bash
docker build -t newslens .
docker run -p 5001:5001 newslens
```

---

## Comparison: NewsLens vs CogniBias

| Dimension | CogniBias (existing) | NewsLens (this project) |
|---|---|---|
| Framework | Flask | FastAPI (async, auto-docs) |
| ML model | BERT on 20 examples (disabled) | RoBERTa MTL (trainable) |
| Political lean | ✗ | ✓ Left / Center / Right |
| URL scraping | ✗ | ✓ trafilatura |
| Persistent storage | In-memory | SQLite (Article + Outlet ORM) |
| Outlet leaderboard | ✗ | ✓ |
| Pattern engine | ✓ (ported) | ✓ (refined, same quality) |
| Bias categories | 5 (original) | 5 (refined: +Sensationalism, +Epistemic Manipulation) |
| Frontend | ✓ (excellent) | ✓ (extended: URL tab, lean indicator) |
| Project structure | Flat | Modular (app/core, app/api, app/db, ml/) |

---

## Model Card

**Architecture:** RoBERTa-base (125M params) · 8 layers frozen · 4 layers fine-tuned  
**Heads:** Multi-label bias (5 classes, BCE) · Political lean (3-class, CE) · Bias intensity (regression, MSE)  
**Pooling:** Learned attention pooling over all tokens (not CLS-only)  
**Default training data:** Synthetic corpus (~400 examples, deterministic)  
**Target metrics (with MBIC):** Bias F1 ≥ 0.72, Lean Acc ≥ 75%  
**Without model:** Pattern engine only — political lean shows "Undetermined"

**Limitations:**
- Synthetic-only training produces weak political lean detection
- MBIC has ~1,700 labelled sentences — still limited for a generalizable model
- Political lean detection is inherently subjective and context-dependent
- The 85% precision shown in hero sections of other projects is aspirational; honest benchmarks are lower
