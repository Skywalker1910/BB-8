"""
embeddings.py
-------------
Token and positional embedding layers for BB8.

Every Transformer needs two types of embeddings:

1. Token Embeddings
   Map discrete token IDs → continuous dense vectors of size d_model.
   These are *learned* parameters — the model discovers a geometric
   representation of the vocabulary during training.

2. Positional Embeddings
   The self-attention mechanism is permutation-invariant: it has no built-in
   notion of order.  Positional embeddings inject position information by
   adding a position-specific vector to each token embedding.

   Two variants are provided:

   SinusoidalPositionalEncoding
       Fixed (not learned).  Uses sine/cosine functions at different
       frequencies as described in "Attention Is All You Need" (Vaswani et
       al., 2017).  Advantage: generalises to sequence lengths not seen
       during training.

   LearnedPositionalEncoding
       Positions 0…max_seq_len each have their own learned embedding vector.
       Used by GPT-2/GPT-3.  Slightly better in practice but cannot
       generalise beyond max_seq_len.

BB8 uses LearnedPositionalEncoding by default (GPT-style).
"""

import math

import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """
    Learnable token embedding table.

    Maps integer token IDs to dense vectors of dimension *d_model*.
    Embeddings are scaled by √d_model following the original Transformer
    paper, which helps keep the initial embedding magnitudes in a reasonable
    range relative to the positional encodings.

    Parameters
    ----------
    vocab_size : int
        Number of tokens in the vocabulary.
    d_model : int
        Embedding dimension (must match the rest of the model).
    """

    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor  shape (batch, seq_len)
            Integer token IDs.

        Returns
        -------
        torch.Tensor  shape (batch, seq_len, d_model)
        """
        return self.embedding(x) * math.sqrt(self.d_model)


# ---------------------------------------------------------------------------
# Sinusoidal (fixed) positional encoding
# ---------------------------------------------------------------------------

class SinusoidalPositionalEncoding(nn.Module):
    """
    Fixed sinusoidal positional encoding from Vaswani et al. (2017).

    The encoding for position *pos* and dimension *i* is:

        PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

    The resulting matrix is pre-computed and stored as a non-trainable
    buffer.  A dropout layer is applied after the addition.

    Parameters
    ----------
    d_model : int
        Embedding dimension.
    max_seq_len : int
        Maximum sequence length to pre-compute encodings for.
    dropout : float
        Dropout probability applied after adding positional encodings.
    """

    def __init__(
        self,
        d_model: int,
        max_seq_len: int = 5_000,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Pre-compute the (max_seq_len, d_model) encoding matrix
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)  # (T, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10_000.0) / d_model)
        )  # (d_model/2,)

        pe[:, 0::2] = torch.sin(position * div_term)  # even dims
        pe[:, 1::2] = torch.cos(position * div_term)  # odd dims
        pe = pe.unsqueeze(0)  # (1, T, d_model)

        # Register as buffer: saved in state_dict but not a trainable param
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to token embeddings.

        Parameters
        ----------
        x : torch.Tensor  shape (batch, seq_len, d_model)

        Returns
        -------
        torch.Tensor  shape (batch, seq_len, d_model)
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Learned positional encoding
# ---------------------------------------------------------------------------

class LearnedPositionalEncoding(nn.Module):
    """
    Learned positional embedding (GPT-style).

    Each position index 0…max_seq_len-1 has its own embedding vector that
    is learned end-to-end during training.

    Parameters
    ----------
    d_model : int
        Embedding dimension.
    max_seq_len : int
        Maximum number of positions supported.
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        d_model: int,
        max_seq_len: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        nn.init.normal_(self.pos_embedding.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add learned positional embeddings.

        Parameters
        ----------
        x : torch.Tensor  shape (batch, seq_len, d_model)

        Returns
        -------
        torch.Tensor  shape (batch, seq_len, d_model)
        """
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)  # (1, T)
        x = x + self.pos_embedding(positions)
        return self.dropout(x)
