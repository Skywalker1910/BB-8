"""
language_model.py
-----------------
BB8LM — the complete GPT-style causal language model.

Architecture summary
--------------------
Input token IDs
    ↓
Token Embedding  (vocab_size → d_model)      — learnable
    +
Positional Embedding  (position → d_model)   — learnable
    ↓
N × TransformerDecoderBlock
    [LayerNorm → Masked MHA → Residual]
    [LayerNorm → FFN         → Residual]
    ↓
Final LayerNorm
    ↓
LM Head  (d_model → vocab_size)             — weight-tied to token embedding

Training objective
------------------
Next-token prediction (causal language modelling):

    L = - Σₜ log P(xₜ | x₁, …, xₜ₋₁)

This is a cross-entropy loss computed over all positions simultaneously.

Weight tying
------------
The LM head weight matrix is shared with the token embedding matrix.
This reduces parameters by vocab_size × d_model and provides better
generalisation — the embedding and unembedding spaces are forced to align.
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .embeddings import LearnedPositionalEncoding, TokenEmbedding
from .transformer import TransformerDecoderBlock, create_causal_mask


class BB8LM(nn.Module):
    """
    BB8 Language Model.

    A decoder-only Transformer trained for next-token prediction.

    Parameters
    ----------
    vocab_size  : int   — size of the tokenizer vocabulary
    d_model     : int   — embedding dimension (default 256)
    num_layers  : int   — number of stacked decoder blocks (default 4)
    num_heads   : int   — attention heads per block (default 8)
    d_ff        : int   — feed-forward inner dimension (default 1024)
    max_seq_len : int   — maximum context length (default 512)
    dropout     : float — dropout probability (default 0.1)
    activation  : str   — FFN activation: 'gelu' or 'relu'
    tie_weights : bool  — share token-embedding and LM-head weights
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        d_ff: int = 1024,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        activation: str = "gelu",
        tie_weights: bool = True,
    ) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len

        # --- Embedding layers ---
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.position_embedding = LearnedPositionalEncoding(
            d_model, max_seq_len, dropout
        )

        # --- Stacked Transformer blocks ---
        self.blocks = nn.ModuleList(
            [
                TransformerDecoderBlock(d_model, num_heads, d_ff, dropout, activation)
                for _ in range(num_layers)
            ]
        )

        # --- Final normalisation ---
        self.norm = nn.LayerNorm(d_model)

        # --- Language model head ---
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        if tie_weights:
            # Share weights between token embedding and LM head
            self.lm_head.weight = self.token_embedding.embedding.weight

        # Weight initialisation
        self.apply(self._init_weights)
        # Scale residual projections by 1/√(2 * num_layers) — GPT-2 trick
        for name, param in self.named_parameters():
            if name.endswith("W_o.weight") or name.endswith("linear2.weight"):
                nn.init.normal_(
                    param, mean=0.0, std=0.02 / (2 * num_layers) ** 0.5
                )

        total = self.count_parameters()
        print(
            f"BB8LM ready  |  layers={num_layers}  heads={num_heads}"
            f"  d_model={d_model}  d_ff={d_ff}  params={total:,}"
        )

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        return_attn_weights: bool = False,
    ) -> Dict:
        """
        Forward pass.

        Parameters
        ----------
        input_ids          : (batch, seq_len)   — integer token IDs
        targets            : (batch, seq_len)   — shifted token IDs for loss
        return_attn_weights: bool               — include per-layer attention maps

        Returns
        -------
        dict with keys:
            'logits'       : (batch, seq_len, vocab_size)
            'loss'         : scalar tensor (only if targets provided)
            'attn_weights' : list of (batch, heads, seq, seq) tensors
                             (only if return_attn_weights=True)
        """
        _, seq_len = input_ids.shape
        device = input_ids.device

        # Embed tokens and positions
        x = self.token_embedding(input_ids)       # (B, T, d_model)
        x = self.position_embedding(x)            # (B, T, d_model)

        # Causal mask prevents attending to future positions
        mask = create_causal_mask(seq_len, device)

        # Pass through each Transformer block
        all_attn: List[torch.Tensor] = []
        for block in self.blocks:
            x, attn = block(x, mask)
            if return_attn_weights:
                all_attn.append(attn.detach())

        # Final LayerNorm + project to vocabulary
        x = self.norm(x)
        logits = self.lm_head(x)                  # (B, T, vocab_size)

        result: Dict = {"logits": logits}

        # Cross-entropy loss over all positions
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=-1,
            )
            result["loss"] = loss

        if return_attn_weights:
            result["attn_weights"] = all_attn

        return result

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_config(self) -> dict:
        """Return a serialisable dictionary of hyperparameters."""
        return {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "d_ff": self.d_ff,
            "max_seq_len": self.max_seq_len,
        }
