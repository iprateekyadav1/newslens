"""
NewsLens — Temperature Scaling Calibration (Lever 4)

Fits a single scalar temperature T that minimises NLL on the validation set.
Run AFTER training completes:

    python ml/calibrate_temperature.py

Saves  models/temperature.json  → {"T": <value>}
The config.py and model.py pick this up automatically on next server start.

Reference: Guo et al., "On Calibration of Modern Neural Networks" (ICML 2017)
"""

import json
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import optim

from config import MODEL_CHECKPOINT, ROBERTA_BASE, MODEL_MAX_LENGTH, LEAN_LABELS
from ml.dataset import build_loaders

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = Path(MODEL_CHECKPOINT).parent
TEMPERATURE_FILE = MODELS_DIR / "temperature.json"


def collect_logits(model, tokenizer, val_loader, device):
    """Forward-pass all validation examples; return stacked logits + labels."""
    all_logits, all_labels = [], []
    model.train(False)
    with torch.no_grad():
        for batch in val_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            lean_labels    = batch["lean_label"].to(device)       # -1 = unknown

            out = model(input_ids, attention_mask)
            # only use examples that have a ground-truth lean label
            mask = lean_labels >= 0
            if mask.any():
                all_logits.append(out["lean_logits"][mask].cpu())
                all_labels.append(lean_labels[mask].cpu())

    if not all_logits:
        raise RuntimeError("No labelled lean examples found in validation set.")

    return torch.cat(all_logits), torch.cat(all_labels)


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Optimise temperature T by minimising cross-entropy NLL."""
    T = torch.nn.Parameter(torch.tensor([1.0]))
    optimiser = optim.LBFGS([T], lr=0.1, max_iter=100)

    def _closure():
        optimiser.zero_grad()
        loss = F.cross_entropy(logits / T, labels)
        loss.backward()
        return loss

    optimiser.step(_closure)
    return float(T.item())


def ece_score(logits: torch.Tensor, labels: torch.Tensor, t: float = 1.0) -> float:
    """Expected Calibration Error — 10-bucket version."""
    probs = torch.softmax(logits / t, dim=1)
    conf, pred = probs.max(dim=1)
    correct = pred.eq(labels).float()
    ece_val = 0.0
    for i in range(10):
        lo, hi = i / 10, (i + 1) / 10
        mask = (conf >= lo) & (conf < hi)
        if mask.sum() > 0:
            bucket_acc  = correct[mask].mean()
            bucket_conf = conf[mask].mean()
            ece_val += float(mask.float().mean() * (bucket_acc - bucket_conf).abs())
    return ece_val


def main():
    checkpoint = Path(MODEL_CHECKPOINT)
    if not checkpoint.exists():
        logger.error("No checkpoint at %s — run ml/train.py first.", checkpoint)
        return

    device = torch.device("cpu")

    logger.info("Loading model …")
    from transformers import RobertaTokenizerFast
    from ml.architecture import NewsLensClassifier

    tokenizer = RobertaTokenizerFast.from_pretrained(ROBERTA_BASE)
    model = NewsLensClassifier(model_name=ROBERTA_BASE)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)

    logger.info("Building validation loader …")
    _, val_loader = build_loaders(tokenizer=tokenizer, batch_size=16)

    logger.info("Collecting validation logits …")
    logits, labels = collect_logits(model, tokenizer, val_loader, device)
    logger.info("Collected %d labelled examples.", len(labels))

    logger.info("Fitting temperature …")
    T = fit_temperature(logits, labels)
    T = max(0.5, min(T, 4.0))   # clamp to sensible range
    logger.info("Optimal temperature T = %.4f", T)

    ece_before = ece_score(logits, labels, t=1.0)
    ece_after  = ece_score(logits, labels, t=T)
    logger.info("ECE before scaling: %.4f", ece_before)
    logger.info("ECE after  scaling: %.4f", ece_after)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    TEMPERATURE_FILE.write_text(json.dumps({"T": round(T, 4)}))
    logger.info("Saved → %s", TEMPERATURE_FILE)


if __name__ == "__main__":
    main()
