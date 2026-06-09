"""
attention.py
------------
Self-attention mechanisms for BB8.

The attention mechanism is the core innovation of the Transformer.  It lets
every position in a sequence directly attend to every other position,
capturing long-range dependencies that RNNs struggle with.

Components
----------
ScaledDotProductAttention
    Single-head attention.  Computes:
        Attention(Q, K, V) = softmax(QKᵀ / √dₖ) · V

MultiHeadAttention
    Runs h attention heads in parallel on projected subspaces, then
    concatenates and projects back.
        MultiHead(Q,K,V) = Concat(head₁,…,headₕ) · Wᴼ
        headᵢ           = Attention(Q·Wᵢᴾ, K·Wᵢᴷ, V·Wᵢⱽ)

    Different heads can learn to attend to different types of relationships
    (syntactic structure, coreference, positional proximity, …).

Why the √dₖ scale?
-------------------
Without scaling, the dot products grow large for high-dimensional vectors,
pushing softmax into near-zero gradient regions.  Dividing by √dₖ keeps
variance ≈ 1 regardless of dimensionality.

Causal masking
--------------
Autoregressive generation requires that position t cannot attend to any
position > t (the future).  This is enforced by filling the upper-triangular
portion of the attention score matrix with -∞ before softmax.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """
    Scaled dot-product attention.

        Attention(Q, K, V) = softmax( Q Kᵀ / √dₖ ) · V

    Parameters
    ----------
    dropout : float
        Dropout applied to attention weights after softmax.
    """

    def __init__(self, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        query : (batch, heads, seq_q, d_k)
        key   : (batch, heads, seq_k, d_k)
        value : (batch, heads, seq_k, d_v)
        mask  : optional (batch, 1, seq_q, seq_k)
                Positions where mask == 0 are blocked (filled with -∞).

        Returns
        -------
        output       : (batch, heads, seq_q, d_v)
        attn_weights : (batch, heads, seq_q, seq_k)   — useful for visualisation
        """
        d_k = query.size(-1)

        # Raw scores: (batch, heads, seq_q, seq_k)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        # Apply causal / padding mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # Softmax over key dimension
        attn_weights = F.softmax(scores, dim=-1)
        # Replace NaN that appears when a whole row is -inf (padding rows)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, value)
        return output, attn_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention layer.

    Projects queries, keys and values *h* times into lower-dimensional
    subspaces (d_k = d_model / h per head), runs attention in parallel,
    then concatenates and projects the results.

    Parameters
    ----------
    d_model   : int   — total embedding dimension
    num_heads : int   — number of parallel attention heads
    dropout   : float — dropout on attention weights and output projection
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0, (
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        )

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # dimension per head

        # Linear projections — no bias on Q/K/V (common practice)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        for proj in (self.W_q, self.W_k, self.W_v, self.W_o):
            nn.init.normal_(proj.weight, std=0.02)
        nn.init.zeros_(self.W_o.bias)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, seq, d_model) → (batch, heads, seq, d_k)"""
        batch, seq, _ = x.size()
        return (
            x.view(batch, seq, self.num_heads, self.d_k)
            .transpose(1, 2)
        )

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, heads, seq, d_k) → (batch, seq, d_model)"""
        batch, _, seq, _ = x.size()
        return (
            x.transpose(1, 2)
            .contiguous()
            .view(batch, seq, self.d_model)
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        query : (batch, seq_q, d_model)
        key   : (batch, seq_k, d_model)
        value : (batch, seq_k, d_model)
        mask  : optional (batch, 1, seq_q, seq_k)

        Returns
        -------
        output       : (batch, seq_q, d_model)
        attn_weights : (batch, heads, seq_q, seq_k)
        """
        Q = self._split_heads(self.W_q(query))   # (B, H, T, d_k)
        K = self._split_heads(self.W_k(key))
        V = self._split_heads(self.W_v(value))

        x, attn_weights = self.attention(Q, K, V, mask)

        x = self._merge_heads(x)          # (B, T, d_model)
        output = self.W_o(x)
        return output, attn_weights
