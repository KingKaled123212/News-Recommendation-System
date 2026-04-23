"""
model.py
========
Full NRMS (Neural News Recommendation with Multi-Head Self-Attention) model.

Reference:
    Wu et al. "Neural News Recommendation with Multi-Head Self-Attention."
    EMNLP 2019.  https://aclanthology.org/D19-1671/

The model has two components that share a single NewsEncoder:
    1. NewsEncoder  — title words → news vector
    2. UserEncoder  — history of news vectors → user vector

Prediction:
    score(user, candidate) = dot(user_vec, candidate_vec)

Training:
    CrossEntropyLoss over (1 positive + K negative) candidates per impression.
"""

import numpy as np
import torch
import torch.nn as nn

from news_encoder import NewsEncoder
from user_encoder import UserEncoder


class NRMSModel(nn.Module):
    """
    End-to-end NRMS recommendation model.

    Args:
        embedding_matrix (np.ndarray): GloVe embedding matrix
                                        shape (vocab_size, embed_dim).
        num_heads (int)  : Number of attention heads for both encoders.
        head_dim  (int)  : Per-head dimensionality; news_dim = num_heads * head_dim.
        dropout   (float): Dropout rate applied inside both encoders.
    """

    def __init__(self, embedding_matrix: np.ndarray,
                 num_heads: int = 16, head_dim: int = 16,
                 dropout: float = 0.2):
        super().__init__()
        news_dim = num_heads * head_dim

        # Shared encoder — used for BOTH history articles and candidate articles
        self.news_encoder = NewsEncoder(embedding_matrix, num_heads,
                                        head_dim, dropout)
        self.user_encoder = UserEncoder(news_dim, num_heads, dropout)

    # ------------------------------------------------------------------
    def encode_news(self, title_ids: torch.Tensor) -> torch.Tensor:
        """
        Public helper: encode a batch of titles.

        Args:
            title_ids : (B, T) or (N, T)

        Returns:
            news_vecs : (B, D) or (N, D)
        """
        return self.news_encoder(title_ids)

    # ------------------------------------------------------------------
    def forward(self, history_ids: torch.Tensor,
                candidate_ids: torch.Tensor,
                hist_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            history_ids   : (B, H, T) — word indices for clicked history
            candidate_ids : (B, C, T) — word indices for candidate articles
            hist_mask     : (B, H)    — 1=real history article, 0=padding

        Returns:
            scores : (B, C) — unnormalised click scores for each candidate
        """
        B, H, T = history_ids.shape
        _, C, _ = candidate_ids.shape

        # ── Encode history articles (flatten → encode → reshape) ──────
        hist_vecs = self.news_encoder(
            history_ids.view(-1, T)              # (B*H, T)
        ).view(B, H, -1)                         # (B, H, D)

        # ── Encode candidate articles ──────────────────────────────────
        cand_vecs = self.news_encoder(
            candidate_ids.view(-1, T)            # (B*C, T)
        ).view(B, C, -1)                         # (B, C, D)

        # ── Aggregate history into user representation ─────────────────
        user_vec = self.user_encoder(hist_vecs, hist_mask)   # (B, D)

        # ── Click scores via dot product ───────────────────────────────
        scores = torch.bmm(
            cand_vecs,                           # (B, C, D)
            user_vec.unsqueeze(-1)               # (B, D, 1)
        ).squeeze(-1)                            # (B, C)

        return scores

    # ------------------------------------------------------------------
    @property
    def news_dim(self) -> int:
        """Dimensionality of news/user vectors."""
        return self.user_encoder.additive_attn.proj.in_features
