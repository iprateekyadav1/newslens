"""
NewsLens — RoBERTa Inference Singleton

Loads the trained NewsLensClassifier once at FastAPI startup.
If no checkpoint exists, predict() returns None and the pattern engine
acts as the sole signal (graceful degradation — same behaviour as CogniBias P0b).

torch.compile() is applied when available (PyTorch >= 2.0) for 20-40%
latency reduction on repeated inference without changing outputs.
"""

import logging
from typing import Optional

import torch

from config import (
    MODEL_CHECKPOINT,
    MODEL_ENABLED,
    ROBERTA_BASE,
    MODEL_MAX_LENGTH,
    LEAN_LABELS,
    LEAN_TEMPERATURE,
)

logger = logging.getLogger(__name__)


class _ModelSingleton:
    """
    Module-level singleton loaded once during FastAPI lifespan startup.
    All request handlers call predict() — no model state is mutated per request.
    """

    _instance: Optional["_ModelSingleton"] = None

    def __new__(cls) -> "_ModelSingleton":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialised = False
            cls._instance._loaded      = False
            cls._instance._error: Optional[str] = None
        return cls._instance

    # ── Startup ───────────────────────────────────────────────────────────────

    def load(self) -> None:
        if self._initialised:
            return

        if not MODEL_ENABLED:
            logger.info(
                "No checkpoint at %s — RoBERTa disabled. "
                "Pattern engine is the sole signal until ml/train.py is run.",
                MODEL_CHECKPOINT,
            )
            self._initialised = True
            return

        try:
            from transformers import RobertaTokenizerFast
            from ml.architecture import NewsLensClassifier

            logger.info("Loading RoBERTa tokenizer …")
            self.tokenizer = RobertaTokenizerFast.from_pretrained(ROBERTA_BASE)

            logger.info("Loading NewsLens checkpoint from %s …", MODEL_CHECKPOINT)
            self.model = NewsLensClassifier(model_name=ROBERTA_BASE)
            state = torch.load(MODEL_CHECKPOINT, map_location="cpu", weights_only=True)
            self.model.load_state_dict(state)

            # model.train(False) is identical to model.eval() — sets training=False
            self.model.train(False)

            # torch.compile requires a C++ compiler on Windows — disabled for CPU-only runs
            logger.info("Running in eager mode (no torch.compile on CPU/Windows).")

            self._loaded = True
            logger.info("RoBERTa model ready.")

            # Warm-up: prime caches (non-fatal if it fails)
            try:
                dummy = self.tokenizer(
                    "This is a warm-up sentence for cache priming.",
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=MODEL_MAX_LENGTH,
                )
                with torch.no_grad():
                    for _ in range(2):
                        self.model(dummy["input_ids"], dummy["attention_mask"])
                logger.info("Warm-up complete.")
            except Exception as warm_exc:
                logger.warning("Warm-up skipped: %s", warm_exc)

        except Exception as exc:
            self._error = str(exc)
            logger.error("Failed to load model: %s", exc)

        self._initialised = True

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, text: str) -> Optional[dict]:
        """
        Returns None when the model is unavailable (not trained yet or load failed).

        On success returns:
          bias_intensity    float [0,1]   — overall bias strength from transformer
          political_lean    str           — "left" | "center" | "right" | "unknown"
          lean_confidence   float [0,1]
          lean_distribution dict          — softmax probs for all three classes
          top_tokens        list[str]     — 5 highest-attention tokens (for debugging)
        """
        if not self._loaded:
            return None

        try:
            enc = self.tokenizer(
                text,
                max_length=MODEL_MAX_LENGTH,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            with torch.no_grad():
                out = self.model(enc["input_ids"], enc["attention_mask"])

            intensity  = float(out["intensity"][0])

            # Lever 4 — temperature scaling: soften overconfident lean predictions
            scaled_logits = out["lean_logits"][0] / LEAN_TEMPERATURE
            lean_probs = torch.softmax(scaled_logits, dim=0).tolist()
            lean_idx   = int(torch.argmax(scaled_logits).item())
            lean_label = LEAN_LABELS[lean_idx]
            lean_conf  = float(lean_probs[lean_idx])

            # Suppress lean output when the model is uncertain
            if lean_conf < 0.35:
                lean_label = "unknown"
                lean_conf  = 0.0

            # Extract top attention tokens (skip RoBERTa special tokens)
            attn = out["attention_weights"][0].cpu().tolist()
            tokens = self.tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
            special = {"<s>", "</s>", "<pad>"}
            pairs = [
                (tokens[i].lstrip("\u0120"), attn[i])   # Ġ prefix → empty
                for i in range(len(tokens))
                if tokens[i] not in special and len(tokens[i].lstrip("\u0120")) > 1
            ]
            top_tokens = [t for t, _ in sorted(pairs, key=lambda x: -x[1])[:5]]

            return {
                "bias_intensity":    round(intensity, 3),
                "political_lean":    lean_label,
                "lean_confidence":   round(lean_conf, 3),
                "lean_distribution": {
                    lbl: round(float(p), 3)
                    for lbl, p in zip(LEAN_LABELS, lean_probs)
                },
                "top_tokens": top_tokens,
            }

        except Exception as exc:
            logger.error("RoBERTa inference error: %s", exc)
            return None

    # ── Status ────────────────────────────────────────────────────────────────

    @property
    def status(self) -> str:
        if not self._initialised:
            return "not initialised"
        if self._loaded:
            return "loaded"
        if self._error:
            return f"error: {self._error}"
        return "disabled — no checkpoint (run ml/train.py)"


# Import this singleton everywhere; never instantiate _ModelSingleton directly
model_singleton = _ModelSingleton()
