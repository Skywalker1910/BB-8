"""
test_attention.py
-----------------
Unit tests for attention mechanisms and positional encodings.

Run with:
    pytest tests/test_attention.py -v
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.attention import MultiHeadAttention, ScaledDotProductAttention
from models.embeddings import (
    LearnedPositionalEncoding,
    SinusoidalPositionalEncoding,
    TokenEmbedding,
)
from models.transformer import TransformerDecoderBlock, create_causal_mask


# ---------------------------------------------------------------------------
# ScaledDotProductAttention
# ---------------------------------------------------------------------------

class TestScaledDotProductAttention:
    def test_output_shape(self):
        attn = ScaledDotProductAttention()
        B, H, T, dk = 2, 4, 10, 16
        Q = torch.randn(B, H, T, dk)
        K = torch.randn(B, H, T, dk)
        V = torch.randn(B, H, T, dk)
        out, weights = attn(Q, K, V)
        assert out.shape == (B, H, T, dk)
        assert weights.shape == (B, H, T, T)

    def test_attention_weights_sum_to_one(self):
        attn = ScaledDotProductAttention()
        Q = torch.randn(1, 1, 6, 8)
        K = torch.randn(1, 1, 6, 8)
        V = torch.randn(1, 1, 6, 8)
        _, weights = attn(Q, K, V)
        row_sums = weights.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    def test_causal_mask_blocks_future(self):
        """With a causal mask, position 0 should not attend to position 1."""
        attn = ScaledDotProductAttention()
        T = 4
        mask = create_causal_mask(T, device=torch.device("cpu"))
        Q = torch.randn(1, 1, T, 8)
        K = torch.randn(1, 1, T, 8)
        V = torch.ones(1, 1, T, 8)
        _, weights = attn(Q, K, V, mask=mask)
        # Position 0 should have zero weight on positions 1, 2, 3
        assert weights[0, 0, 0, 1].item() == pytest.approx(0.0, abs=1e-5)
        assert weights[0, 0, 0, 2].item() == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# MultiHeadAttention
# ---------------------------------------------------------------------------

class TestMultiHeadAttention:
    def test_output_shape(self):
        mha = MultiHeadAttention(d_model=64, num_heads=4)
        B, T, D = 2, 10, 64
        x = torch.randn(B, T, D)
        out, weights = mha(x, x, x)
        assert out.shape == (B, T, D)
        assert weights.shape == (B, 4, T, T)

    def test_d_model_not_divisible_raises(self):
        with pytest.raises(AssertionError):
            MultiHeadAttention(d_model=65, num_heads=4)

    def test_output_different_from_input(self):
        mha = MultiHeadAttention(d_model=32, num_heads=2)
        x = torch.randn(1, 5, 32)
        out, _ = mha(x, x, x)
        # Output should generally differ from input
        assert not torch.allclose(out, x)


# ---------------------------------------------------------------------------
# Causal mask
# ---------------------------------------------------------------------------

class TestCausalMask:
    def test_shape(self):
        mask = create_causal_mask(8, device=torch.device("cpu"))
        assert mask.shape == (1, 1, 8, 8)

    def test_lower_triangular(self):
        T = 5
        mask = create_causal_mask(T, device=torch.device("cpu")).squeeze()
        for i in range(T):
            for j in range(T):
                expected = 1.0 if j <= i else 0.0
                assert mask[i, j].item() == expected


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

class TestEmbeddings:
    def test_token_embedding_shape(self):
        emb = TokenEmbedding(vocab_size=100, d_model=32)
        x = torch.randint(0, 100, (2, 16))
        out = emb(x)
        assert out.shape == (2, 16, 32)

    def test_sinusoidal_shape(self):
        pe = SinusoidalPositionalEncoding(d_model=32, max_seq_len=64)
        x = torch.randn(2, 20, 32)
        out = pe(x)
        assert out.shape == (2, 20, 32)

    def test_learned_positional_shape(self):
        pe = LearnedPositionalEncoding(d_model=32, max_seq_len=64)
        x = torch.randn(2, 20, 32)
        out = pe(x)
        assert out.shape == (2, 20, 32)


# ---------------------------------------------------------------------------
# TransformerDecoderBlock
# ---------------------------------------------------------------------------

class TestTransformerDecoderBlock:
    def test_output_shape(self):
        block = TransformerDecoderBlock(d_model=64, num_heads=4, d_ff=128)
        x = torch.randn(2, 10, 64)
        mask = create_causal_mask(10, device=torch.device("cpu"))
        out, weights = block(x, mask)
        assert out.shape == (2, 10, 64)

    def test_residual_connection_preserves_shape(self):
        block = TransformerDecoderBlock(d_model=32, num_heads=2, d_ff=64)
        x = torch.randn(1, 8, 32)
        out, _ = block(x)
        assert out.shape == x.shape
