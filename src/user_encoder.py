"""
user_encoder.py
===============
UserEncoder module: aggregates a sequence of clicked-news vectors
into a single user representation vector.

Architecture:
    Clicked News Vectors  (B, H, D)   — from NewsEncoder
        → Multi-Head Self-Attention   — models inter-article dependencies
        → LayerNorm + Dropout
        → Additive Attention          — weighted aggregation into one vector
        → User Vector  (B, D)
"""

import torch
import torch.nn as nn

from news_encoder import AdditiveAttention


class UserEncoder(nn.Module):
    """
    Aggregates a user's click history (sequence of news vectors) into a
    single user representation using multi-head self-attention followed
    by additive attention pooling.

    The NewsEncoder is shared between article encoding and history encoding,
    so the UserEncoder only needs to process already-encoded news vectors.

    Args:
        news_dim  (int)  : Dimensionality of news vectors (= num_heads * head_dim).
        num_heads (int)  : Number of attention heads (must divide news_dim evenly).
        dropout   (float): Dropout probability.
    """

    def __init__(self, news_dim: int, num_heads: int = 16,
                 dropout: float = 0.2):
        super().__init__()

        self.mha = nn.MultiheadAttention(
            embed_dim=news_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm          = nn.LayerNorm(news_dim)
        self.dropout       = nn.Dropout(dropout)
        self.additive_attn = AdditiveAttention(news_dim)

    def forward(self, hist_vecs: torch.Tensor,
                hist_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            hist_vecs : (B, H, D) — sequence of encoded clicked-news vectors
            hist_mask : (B, H)    — binary mask; 1 = real article, 0 = padding

        Returns:
            user_vec  : (B, D)   — aggregated user representation
        """
        # key_padding_mask expects True where positions should be IGNORED
        key_pad = (hist_mask == 0) if hist_mask is not None else None

        x, _ = self.mha(hist_vecs, hist_vecs, hist_vecs,
                        key_padding_mask=key_pad)          # (B, H, D)
        x = self.norm(self.dropout(x))

        user_vec = self.additive_attn(x, hist_mask)        # (B, D)
        return user_vec
