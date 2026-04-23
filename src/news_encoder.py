"""
news_encoder.py
===============
NewsEncoder module: converts a news article title (sequence of word IDs)
into a fixed-size dense vector representation.

Architecture:
    Word Embedding (GloVe, frozen=False)
        → Linear Projection (embed_dim → num_heads * head_dim)
        → Multi-Head Self-Attention
        → LayerNorm + Dropout
        → Additive Attention (learned query)
        → News Vector  (num_heads * head_dim,)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdditiveAttention(nn.Module):
    """
    Collapses a sequence of vectors into a single context vector using
    a trainable additive (Bahdanau-style) attention mechanism.

    Given input x of shape (B, T, D), computes:
        e_t     = tanh(W * x_t + b)
        alpha_t = softmax(v^T * e_t)
        output  = sum_t(alpha_t * x_t)

    Args:
        dim        (int): Input feature dimension D.
        hidden_dim (int): Projection hidden size (default 200).
    """

    def __init__(self, dim: int, hidden_dim: int = 200):
        super().__init__()
        self.proj  = nn.Linear(dim, hidden_dim)
        self.query = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x    : (B, T, D) — input sequence
            mask : (B, T)    — optional binary mask; 0 positions are ignored

        Returns:
            (B, D) — weighted sum of input vectors
        """
        e      = torch.tanh(self.proj(x))           # (B, T, H)
        scores = self.query(e).squeeze(-1)           # (B, T)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        weights = F.softmax(scores, dim=-1)          # (B, T)
        out     = torch.bmm(weights.unsqueeze(1), x).squeeze(1)  # (B, D)
        return out


class NewsEncoder(nn.Module):
    """
    Encodes a news article title into a dense vector.

    Args:
        embedding_matrix (np.ndarray): Pre-trained word embeddings
                                        shape (vocab_size, embed_dim).
        num_heads (int): Number of attention heads (default 16).
        head_dim  (int): Dimension per head; output dim = num_heads * head_dim.
        dropout   (float): Dropout probability applied after embedding and MHA.
    """

    def __init__(self, embedding_matrix, num_heads: int = 16,
                 head_dim: int = 16, dropout: float = 0.2):
        super().__init__()
        embed_dim = embedding_matrix.shape[1]
        attn_dim  = num_heads * head_dim

        # Word embedding — initialised from GloVe, fine-tuned during training
        self.word_embed = nn.Embedding.from_pretrained(
            torch.FloatTensor(embedding_matrix),
            freeze=False,
            padding_idx=0,
        )
        # Project from GloVe dim (300) to attention dim (256)
        self.proj = nn.Linear(embed_dim, attn_dim)

        # Multi-head self-attention over title words
        self.mha = nn.MultiheadAttention(
            embed_dim=attn_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm          = nn.LayerNorm(attn_dim)
        self.dropout       = nn.Dropout(dropout)
        self.additive_attn = AdditiveAttention(attn_dim)

    def forward(self, title_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            title_ids : (B, T) — word index sequences

        Returns:
            news_vec  : (B, attn_dim) — article representation
        """
        pad_mask = (title_ids == 0)                              # True = ignore

        x = self.dropout(self.word_embed(title_ids))             # (B, T, E)
        x = self.proj(x)                                         # (B, T, D)
        x, _ = self.mha(x, x, x, key_padding_mask=pad_mask)     # (B, T, D)
        x = self.norm(self.dropout(x))

        news_vec = self.additive_attn(x)                         # (B, D)
        return news_vec
