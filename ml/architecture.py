"""
NewsLens — Neural Architecture

NewsLensClassifier:
  Backbone : RoBERTa-base (12 transformer layers, hidden=768)
  Freezing : Embeddings + layers 0–7 frozen; layers 8–11 trained
  Pooling  : Learned attention pooling over all token embeddings
             (outperforms CLS-only for span-level bias tasks)
  Heads    :
    bias_head      — multi-label sigmoid, 5 cognitive bias categories
    lean_head      — 3-class softmax (left / center / right)
    intensity_head — scalar regression [0,1] overall bias strength
"""

import torch
import torch.nn as nn
from transformers import RobertaModel

from config import BIAS_CATEGORIES, LEAN_LABELS


class AttentionPooling(nn.Module):
    """
    Weighted mean-pool: learns a scalar score per token, softmax-normalises
    over non-padding positions, then computes weighted sum of hidden states.
    Returns (pooled_vec, attention_weights) so we can visualise which tokens
    drove the prediction.
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.score = nn.Linear(hidden_size, 1, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,   # [B, T, H]
        attention_mask: torch.Tensor,  # [B, T]  1=real, 0=pad
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.score(hidden_states).squeeze(-1)           # [B, T]
        scores = scores.masked_fill(attention_mask == 0, -1e9)   # mask padding
        weights = torch.softmax(scores, dim=-1)                  # [B, T]
        pooled  = torch.bmm(weights.unsqueeze(1), hidden_states).squeeze(1)  # [B, H]
        return pooled, weights


class NewsLensClassifier(nn.Module):

    NUM_BIAS  = len(BIAS_CATEGORIES)   # 5
    NUM_LEAN  = len(LEAN_LABELS)       # 3

    def __init__(
        self,
        model_name:    str   = "roberta-base",
        freeze_layers: int   = 8,
        dropout_bias:  float = 0.30,
        dropout_lean:  float = 0.30,
    ) -> None:
        super().__init__()

        self.roberta = RobertaModel.from_pretrained(model_name)
        H = self.roberta.config.hidden_size   # 768

        # Gradient checkpointing: recompute activations during backprop instead
        # of storing them — ~60% less activation RAM at ~30% extra compute cost.
        self.roberta.gradient_checkpointing_enable()

        # ── Freeze embeddings + early layers ─────────────────────────────────
        for p in self.roberta.embeddings.parameters():
            p.requires_grad = False
        for i, layer in enumerate(self.roberta.encoder.layer):
            if i < freeze_layers:
                for p in layer.parameters():
                    p.requires_grad = False

        # ── Pooler ───────────────────────────────────────────────────────────
        self.pooler = AttentionPooling(H)

        # ── Head 1 : multi-label cognitive bias classifier ───────────────────
        self.bias_head = nn.Sequential(
            nn.Linear(H, 256),
            nn.ReLU(),
            nn.Dropout(dropout_bias),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout_bias * 0.67),
            nn.Linear(64, self.NUM_BIAS),
            # No activation — raw logits; BCEWithLogitsLoss handles sigmoid
        )

        # ── Head 2 : political lean (3-class) ────────────────────────────────
        self.lean_head = nn.Sequential(
            nn.Linear(H, 128),
            nn.ReLU(),
            nn.Dropout(dropout_lean),
            nn.Linear(128, self.NUM_LEAN),
            # No activation — raw logits; CrossEntropyLoss handles softmax
        )

        # ── Head 3 : overall bias intensity (regression [0,1]) ───────────────
        self.intensity_head = nn.Sequential(
            nn.Linear(H, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:

        hidden = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state                          # [B, T, H]

        pooled, attn_weights = self.pooler(hidden, attention_mask)

        return {
            "bias_logits":        self.bias_head(pooled),       # [B, 5]
            "lean_logits":        self.lean_head(pooled),       # [B, 3]
            "intensity":          self.intensity_head(pooled).squeeze(-1),  # [B]
            "attention_weights":  attn_weights,                 # [B, T]
            "pooled":             pooled,                       # [B, H]
        }

    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
