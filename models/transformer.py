"""
transformer.py
--------------
Transformer Decoder Block and causal masking utilities.

BB8 uses a **decoder-only** architecture (GPT-style) where every block
consists of:

    1. Pre-LayerNorm → Masked Multi-Head Self-Attention → Residual Add
    2. Pre-LayerNorm → Feed-Forward Network             → Residual Add

Pre-LN (normalise *before* the sub-layer) is used instead of the original
Post-LN from Vaswani et al. (2017).  Pre-LN provides more stable gradient
flow, which enables training deeper models without special warmup schedules.

Residual connections
--------------------
Each sub-layer's output is added to its input:

    x = x + SubLayer(LayerNorm(x))

This creates a "highway" that lets gradients flow directly through the
network even when many layers are stacked.

Causal mask
-----------
To prevent position t from attending to positions > t (the future),
attention scores for those positions are set to −∞ before softmax.
The resulting lower-triangular mask is called the *causal* or *autoregressive*
mask.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .attention import MultiHeadAttention
from .feed_forward import FeedForward


def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Create a lower-triangular causal attention mask.

    Returns a boolean-style float mask where:
        1 → position is allowed to attend
        0 → position is blocked (will become -∞ in attention)

    Shape: (1, 1, seq_len, seq_len) — broadcastable over batch and heads.

    Example (seq_len=4):
        [[1, 0, 0, 0],
         [1, 1, 0, 0],
         [1, 1, 1, 0],
         [1, 1, 1, 1]]
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)


class TransformerDecoderBlock(nn.Module):
    """
    A single GPT-style Transformer Decoder block.

    Sub-layers
    ----------
    1. Masked multi-head self-attention (causal)
    2. Position-wise feed-forward network

    Both sub-layers use:
    - Pre-layer normalisation (LayerNorm applied *before* the sub-layer)
    - Residual (skip) connection around the sub-layer
    - Dropout on the sub-layer output

    Parameters
    ----------
    d_model    : int   — embedding dimension
    num_heads  : int   — number of attention heads
    d_ff       : int   — feed-forward inner dimension
    dropout    : float — dropout probability
    activation : str   — FFN activation ('gelu' or 'relu')
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "gelu",
    ) -> None:
        super().__init__()

        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout, activation)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x    : (batch, seq_len, d_model)
        mask : optional (1, 1, seq_len, seq_len)  — causal mask

        Returns
        -------
        x            : (batch, seq_len, d_model)  — updated representations
        attn_weights : (batch, heads, seq_len, seq_len)
        """
        # --- Sub-layer 1: Masked Self-Attention ---
        normed = self.norm1(x)
        attn_out, attn_weights = self.attention(normed, normed, normed, mask)
        x = x + self.dropout(attn_out)

        # --- Sub-layer 2: Feed-Forward ---
        normed = self.norm2(x)
        ff_out = self.feed_forward(normed)
        x = x + self.dropout(ff_out)

        return x, attn_weights
