"""
NewsLens — Training Pipeline

Usage:
    python ml/train.py
    python ml/train.py --epochs 10 --batch_size 8 --lr 2e-5

What it does:
  1. Loads RoBERTa-base tokenizer + NewsLensClassifier
  2. Builds train/val DataLoaders from synthetic + MBIC (if available)
  3. Trains with multi-task loss:
       L = w_bias * BCE_bias  +  w_lean * CE_lean  +  w_int * MSE_intensity
  4. AdamW + linear warmup (10% of steps) + cosine decay
  5. Early stopping (patience=5 on val F1)
  6. Saves best checkpoint to models/newslens_best.pt

Target metrics on MBIC+synthetic:  F1 >= 0.72,  Acc >= 75%
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from transformers import RobertaTokenizerFast, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import ROBERTA_BASE, MODELS_DIR, BIAS_CATEGORIES, LEAN_LABELS
from ml.architecture import NewsLensClassifier
from ml.dataset import build_loaders

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("train")

MODELS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH = str(MODELS_DIR / "newslens_best.pt")


# ── Multi-task loss ───────────────────────────────────────────────────────────

class MultiTaskLoss(nn.Module):

    def __init__(
        self,
        class_weights: torch.Tensor,
        w_bias: float = 1.0,
        w_lean: float = 2.0,
        w_int:  float = 0.3,
    ) -> None:
        super().__init__()
        self.bce    = nn.BCEWithLogitsLoss(pos_weight=class_weights)
        self.ce     = nn.CrossEntropyLoss()
        self.mse    = nn.MSELoss()
        self.w_bias = w_bias
        self.w_lean = w_lean
        self.w_int  = w_int

    def forward(self, outputs: dict, batch: dict) -> tuple[torch.Tensor, dict]:
        l_bias = self.bce(outputs["bias_logits"], batch["bias_labels"])
        l_lean = self.ce(outputs["lean_logits"],  batch["lean_label"])
        l_int  = self.mse(outputs["intensity"],   batch["bias_intensity"])
        total  = self.w_bias * l_bias + self.w_lean * l_lean + self.w_int * l_int
        return total, {"bias": l_bias.item(), "lean": l_lean.item(), "intensity": l_int.item()}


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model: nn.Module, loader, loss_fn: MultiTaskLoss, device: torch.device) -> dict:
    # model.train(False) sets inference mode (disables dropout / batch-norm updates)
    model.train(False)
    total_loss  = 0.0
    all_bias_true, all_bias_pred = [], []
    all_lean_true, all_lean_pred = [], []

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out   = model(batch["input_ids"], batch["attention_mask"])
        loss, _ = loss_fn(out, batch)
        total_loss += loss.item()

        bias_pred = (torch.sigmoid(out["bias_logits"]) > 0.5).cpu().numpy()
        bias_true = batch["bias_labels"].cpu().numpy().astype(int)
        all_bias_true.append(bias_true)
        all_bias_pred.append(bias_pred)

        lean_pred = torch.argmax(out["lean_logits"], dim=1).cpu().numpy()
        lean_true = batch["lean_label"].cpu().numpy()
        all_lean_true.append(lean_true)
        all_lean_pred.append(lean_pred)

    model.train(True)

    bias_true_all = np.vstack(all_bias_true)
    bias_pred_all = np.vstack(all_bias_pred)
    lean_true_all = np.concatenate(all_lean_true)
    lean_pred_all = np.concatenate(all_lean_pred)

    return {
        "val_loss":    round(total_loss / len(loader), 4),
        "bias_f1":     round(f1_score(bias_true_all, bias_pred_all, average="macro", zero_division=0), 4),
        "lean_f1":     round(f1_score(lean_true_all, lean_pred_all, average="macro", zero_division=0), 4),
        "lean_acc":    round(accuracy_score(lean_true_all, lean_pred_all), 4),
        "combined_f1": round((
            f1_score(bias_true_all, bias_pred_all, average="macro", zero_division=0) +
            f1_score(lean_true_all, lean_pred_all, average="macro", zero_division=0)
        ) / 2, 4),
    }


# ── Training loop ─────────────────────────────────────────────────────────────

def train(
    epochs:        int   = 8,
    batch_size:    int   = 6,
    lr:            float = 2e-5,
    weight_decay:  float = 0.01,
    patience:      int   = 5,
    warmup_ratio:  float = 0.10,
    max_length:    int   = 256,
    freeze_layers: int   = 8,
    max_samples:   int   = 6000,
    log_every:     int   = 50,
    resume:        bool  = False,
    w_lean:        float = 2.0,
) -> None:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    logger.info(
        "batch_size=%d  max_length=%d  freeze_layers=%d  max_samples=%d  resume=%s",
        batch_size, max_length, freeze_layers, max_samples, resume,
    )

    logger.info("Loading RoBERTa tokenizer (%s) …", ROBERTA_BASE)
    tokenizer = RobertaTokenizerFast.from_pretrained(ROBERTA_BASE)

    train_loader, val_loader, class_weights = build_loaders(
        tokenizer, val_split=0.15, batch_size=batch_size,
        max_length=max_length, max_samples=max_samples,
    )
    class_weights = class_weights.to(device)
    logger.info("Train batches: %d  |  Val batches: %d", len(train_loader), len(val_loader))

    model = NewsLensClassifier(model_name=ROBERTA_BASE, freeze_layers=freeze_layers).to(device)

    if resume and Path(CHECKPOINT_PATH).exists():
        try:
            state = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
            model.load_state_dict(state)
            logger.info("Resumed from checkpoint: %s", CHECKPOINT_PATH)
        except Exception as exc:
            logger.warning("Could not resume checkpoint (%s) — starting fresh.", exc)

    logger.info(
        "Trainable params: %s / %s total",
        f"{model.trainable_params():,}", f"{model.total_params():,}",
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=weight_decay,
    )
    total_steps  = len(train_loader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    loss_fn = MultiTaskLoss(class_weights, w_lean=w_lean)
    logger.info("Loss weights — bias=1.0  lean=%.1f  intensity=0.3", w_lean)

    best_f1    = 0.0
    no_improve = 0

    logger.info("\n%s", "=" * 68)
    logger.info(" Epoch | Train Loss | Val Loss | Bias F1 | Lean F1 | Lean Acc")
    logger.info("%s", "=" * 68)

    for epoch in range(1, epochs + 1):
        model.train(True)
        total_train_loss = 0.0
        t0 = time.time()

        for step, batch in enumerate(train_loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            out  = model(batch["input_ids"], batch["attention_mask"])
            loss, _ = loss_fn(out, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_train_loss += loss.item()
            if log_every > 0 and step % log_every == 0:
                logger.info(
                    "   ep%d  step %d/%d  loss=%.4f",
                    epoch, step, len(train_loader),
                    total_train_loss / step,
                )

        train_loss  = round(total_train_loss / len(train_loader), 4)
        val_metrics = evaluate(model, val_loader, loss_fn, device)
        elapsed     = round(time.time() - t0, 1)

        logger.info(
            "   %2d  |   %.4f   |  %.4f  |  %.4f  |  %.4f  |  %.4f   [%.1fs]",
            epoch, train_loss, val_metrics["val_loss"],
            val_metrics["bias_f1"], val_metrics["lean_f1"],
            val_metrics["lean_acc"], elapsed,
        )

        if val_metrics["combined_f1"] > best_f1:
            best_f1    = val_metrics["combined_f1"]
            no_improve = 0
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            logger.info("   [+] New best F1=%.4f - saved to %s", best_f1, CHECKPOINT_PATH)
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info("Early stopping — no improvement for %d epochs.", patience)
                break

    logger.info("=" * 68)
    logger.info("Training complete.  Best combined F1: %.4f", best_f1)
    logger.info("Checkpoint: %s", CHECKPOINT_PATH)
    logger.info(
        "\nNext steps:\n"
        "  1. uvicorn app.main:app --port 5001 --reload\n"
        "  2. RoBERTa is auto-detected and contributes 40%% of the final bias score.\n"
        "  3. For better accuracy: add MBIC data to data/raw/mbic.csv and retrain."
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NewsLens RoBERTa model.")
    parser.add_argument("--epochs",        type=int,   default=6)
    parser.add_argument("--batch_size",    type=int,   default=4)
    parser.add_argument("--lr",            type=float, default=2e-5)
    parser.add_argument("--patience",      type=int,   default=5)
    parser.add_argument("--max_length",    type=int,   default=256,
                        help="Token sequence length (256 saves ~4x attention RAM vs 512)")
    parser.add_argument("--freeze_layers", type=int,   default=10,
                        help="Freeze first N RoBERTa layers (10 = only last 2 + heads train)")
    parser.add_argument("--max_samples",   type=int,   default=2000,
                        help="Stratified cap on total training examples (0 = no cap)")
    parser.add_argument("--log_every",     type=int,   default=50,
                        help="Log progress every N batches within an epoch (0 = off)")
    parser.add_argument("--resume",        action="store_true",
                        help="Resume training from existing checkpoint (if compatible)")
    parser.add_argument("--w_lean",        type=float, default=2.0,
                        help="Loss weight for lean head (default 2.0 — prioritizes lean over bias)")
    args = parser.parse_args()
    train(
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        patience=args.patience, max_length=args.max_length,
        freeze_layers=args.freeze_layers, max_samples=args.max_samples,
        log_every=args.log_every, resume=args.resume, w_lean=args.w_lean,
    )
