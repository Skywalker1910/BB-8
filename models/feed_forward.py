"""
feed_forward.py
---------------
Position-wise Feed-Forward Network (FFN) used in each Transformer block.

The FFN applies the same two-layer MLP independently to each position:

    FFN(x) = activation(x · W₁ + b₁) · W₂ + b₂

Key design choices
------------------
- The inner (hidden) dimension d_ff is typically 4 × d_model, creating an
  "expand-then-contract" bottleneck that lets the FFN learn rich
  non-linear transformations.
- GELU activation is used instead of ReLU.  GELU is smoother and has been
  shown to work better in Transformer-based language models (GPT-2/GPT-3).
- Dropout is applied between the two linear layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    """
    Position-wise two-layer feed-forward network.

    Parameters
    ----------
    d_model    : int   — input and output dimension
    d_ff       : int   — hidden (inner) dimension; typically 4 × d_model
    dropout    : float — dropout probability between the two linear layers
    activation : str   — 'gelu' (default) or 'relu'
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "gelu",
    ) -> None:
        super().__init__()

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

        if activation == "gelu":
            self._activation = F.gelu
        elif activation == "relu":
            self._activation = F.relu
        else:
            raise ValueError(
                f"Unsupported activation '{activation}'. Choose 'gelu' or 'relu'."
            )

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.linear1.weight, std=0.02)
        nn.init.normal_(self.linear2.weight, std=0.02)
        nn.init.zeros_(self.linear1.bias)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor  shape (batch, seq_len, d_model)

        Returns
        -------
        torch.Tensor  shape (batch, seq_len, d_model)
        """
        x = self.linear1(x)
        x = self._activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
