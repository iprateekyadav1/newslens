# Design Spec: Bias Explainer — LLM Fine-tuning Project

**Date:** 2026-04-20  
**Author:** Prateek Yadav  
**Status:** Approved  
**Project type:** Data Science Portfolio — LLM Fine-tuning  

---

## Overview

Fine-tune `Qwen2.5-1.5B-Instruct` using QLoRA to generate natural language
explanations of media bias. The model takes a news excerpt + bias scores from
the existing NewsLens pattern engine and produces a journalist-readable
paragraph explaining *why* the text is biased and *how* the bias manifests.

The fine-tuned model is served as a standalone FastAPI microservice
(`bias-explainer`) that NewsLens calls after its existing analysis. If the
explainer is unavailable, NewsLens degrades gracefully — the explanation field
returns `null` and all other functionality continues normally.

---

## Architecture

```
NewsLens (existing)                bias-explainer (new)
──────────────────────             ──────────────────────────────
POST /api/analyze
      │
 Pattern Engine ──► bias_score ──► POST /explain
 + category_scores                       │
                                   Qwen2.5-1.5B (fine-tuned)
                                         │
                                   explanation: str
                                         │
◄────────────────────────────────────────┘
Response includes:
  "explanation": "This article uses loaded language..."
```

**Three independent pieces:**

| Piece | Purpose | Runs on |
|---|---|---|
| `data-pipeline/` | Generate training pairs from NewsLens datasets | Local CPU |
| `training/` | QLoRA fine-tune on Qwen2.5-1.5B | Kaggle T4 GPU |
| `bias-explainer/` | FastAPI microservice serving fine-tuned model | Render (Docker) |

---

## Data Pipeline

**Source data:** `mbic.csv` and `qbias_articles.csv` (already in NewsLens `data/raw/`)

### Step 1 — Weak label generation (~800 pairs, free)
Run each article excerpt through the NewsLens pattern engine to get
`category_scores`. Convert scores into template-based explanations:

```
Input:       "The catastrophic failure of government policy has left millions suffering."
Categories:  Loaded Language: 0.91, Framing (victim): 0.74, Sensationalism: 0.62

Template →   "This text exhibits high loaded language ('catastrophic', 'suffering'),
              victim framing that positions the subject as harmed, and sensationalist
              urgency. Combined bias score: 78/100."
```

### Step 2 — Gold polishing (~150 pairs, ~$0.50–$1.50 API cost)
Take the 150 highest-confidence weak examples and send to Claude/GPT-4o API
to rewrite as fluent, journalist-quality explanations.

### Step 3 — Dataset assembly
```
data/
├── raw/           ← mbic.csv, qbias_articles.csv
├── weak/          ← 800 template-generated pairs (JSONL)
├── gold/          ← 150 API-polished pairs (JSONL)
└── final/
    ├── train.jsonl    ← 850 examples
    ├── val.jsonl      ← 70 examples
    └── test.jsonl     ← 30 examples
```

**JSONL record schema:**
```json
{
  "text": "The catastrophic failure...",
  "bias_score": 78.4,
  "categories": {"Loaded Language": 0.91, "Framing": 0.74},
  "explanation": "This article uses emotionally charged language...",
  "source": "gold"
}
```

---

## Model & Fine-tuning

**Base model:** `Qwen/Qwen2.5-1.5B-Instruct`  
- 1.5B parameters, instruction-tuned
- Fits on Kaggle T4 (15GB VRAM) with 4-bit quantization
- Fallback: `Qwen/Qwen2.5-0.5B-Instruct` for local CPU development

**Method:** QLoRA (Quantized Low-Rank Adaptation)
```
Full model weights  →  frozen, stored in INT4
LoRA adapters       →  ~3M trainable params injected into attention layers
Computation         →  BF16
VRAM required       →  ~6GB  (well within T4's 15GB)
```

**Training stack:**
| Library | Role |
|---|---|
| `transformers` | Load Qwen2.5, tokenizer |
| `peft` | LoRA adapter injection |
| `bitsandbytes` | 4-bit quantization |
| `trl` | `SFTTrainer` supervised fine-tuning loop |
| `datasets` | Load JSONL training files |
| `wandb` | Loss curves, training metrics |

**Prompt format (ChatML):**
```
<|im_start|>system
You are a media bias analyst. Given a news excerpt and its bias scores,
write a clear natural language explanation of the bias present.
<|im_end|>
<|im_start|>user
Text: "{excerpt}"
Bias Score: {score}/100
Categories: {category}: {value}, ...
<|im_end|>
<|im_start|>assistant
{explanation}
<|im_end|>
```

**Hyperparameters:**
```
Epochs:             3
Batch size:         4 (gradient accumulation steps = 4 → effective batch = 16)
Learning rate:      2e-4 with cosine decay
LoRA rank:          r=16, alpha=32
LoRA target:        q_proj, v_proj attention layers
Max token length:   512
Optimizer:          paged_adamw_8bit
```

**Output:** LoRA adapter weights (~25MB) pushed to HuggingFace Hub.
The microservice loads base model + adapter at startup.

---

## bias-explainer Microservice

**Startup sequence:**
1. Download `Qwen2.5-1.5B-Instruct` base from HuggingFace Hub
2. Load LoRA adapter from HuggingFace Hub
3. Merge adapter into base model
4. Ready to serve (cold start ~30s)

**API:**
```
POST /explain
Request:
{
  "text": "string",
  "bias_score": float,
  "categories": { "Category Name": float, ... }
}

Response 200:
{
  "explanation": "string",
  "model_version": "qwen2.5-1.5b-newslens-v1",
  "generation_ms": int
}

Response 503:  model not loaded yet (during cold start)
```

**Project structure:**
```
bias-explainer/
├── app/
│   ├── main.py          FastAPI app, lifespan model load
│   ├── model.py         Model singleton (base + LoRA merge)
│   ├── schemas.py       Pydantic request/response
│   └── generator.py     Prompt builder + inference
├── Dockerfile
├── requirements.txt
└── .env.example         HF_MODEL_ID, HF_ADAPTER_ID, MAX_NEW_TOKENS
```

**Graceful degradation in NewsLens** (`app/api/routes.py`):
```python
explanation = None
try:
    resp = httpx.post(EXPLAINER_URL + "/explain", json=payload, timeout=15)
    if resp.status_code == 200:
        explanation = resp.json()["explanation"]
except Exception:
    pass
```

**Deployment:** Render free tier (Docker), same pattern as NewsLens.

---

## Evaluation

**Automatic metrics** (on `test.jsonl`, 30 examples):

| Metric | What it measures |
|---|---|
| `ROUGE-L` | Token overlap vs reference explanation |
| `BERTScore F1` | Semantic similarity vs reference |
| Perplexity | Model confidence on test set |

**Before vs After comparison** (key portfolio visual):
```
Base model (no fine-tune):
  "The text contains some negative words and may show bias."

Fine-tuned model:
  "This excerpt employs victim framing by positioning citizens as passive
   recipients of government failure. The phrase 'left millions suffering'
   amplifies emotional impact through loaded language, while 'catastrophic'
   anchors the reader's perception before any evidence is presented."
```

**Training curves:** W&B dashboard — training loss + validation loss over 3 epochs.

**Human eval:** Rate 20 random test explanations on a 1–3 scale:
- 1 = generic / incorrect
- 2 = correct but vague  
- 3 = specific, accurate, useful

---

## Full Project Structure

```
fine-tune-bias-explainer/          ← new GitHub repo
├── data-pipeline/
│   ├── generate_weak_labels.py    ← pattern engine → template explanations
│   ├── polish_with_api.py         ← Claude/GPT-4o API polishing
│   ├── assemble_dataset.py        ← merge + train/val/test split
│   └── requirements.txt
├── training/
│   ├── train_qwen_qlora.ipynb     ← Kaggle notebook (main artifact)
│   └── requirements.txt
├── bias-explainer/
│   ├── app/
│   │   ├── main.py
│   │   ├── model.py
│   │   ├── schemas.py
│   │   └── generator.py
│   ├── Dockerfile
│   ├── requirements.txt
│   └── .env.example
├── evaluation/
│   ├── evaluate_metrics.py        ← ROUGE, BERTScore, perplexity
│   └── before_after_compare.py    ← side-by-side output comparison
├── docs/
│   └── model_card.md
└── README.md
```

---

## Constraints & Limitations

- Synthetic/template-generated training data limits explanation diversity
- 30-example test set is small — metrics are indicative, not statistically robust
- Render free tier cold starts (~30s) make real-time use feel slow; acceptable for portfolio demos
- Model trained on English-only news; does not generalize to other languages
- Political lean explanation quality limited by weak label quality for lean category

---

## Success Criteria

- [ ] 850+ training pairs generated and committed to repo
- [ ] Training loss decreases over 3 epochs (no divergence)
- [ ] ROUGE-L score improves vs base model on test set
- [ ] Before/after qualitative comparison shows clear improvement
- [ ] `bias-explainer` service deployed and reachable from live NewsLens
- [ ] W&B training run publicly shareable
- [ ] HuggingFace Hub model card published
