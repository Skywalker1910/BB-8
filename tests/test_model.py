"""
test_model.py
-------------
Integration tests for BB8LM — forward pass, loss computation, and generation.

Run with:
    pytest tests/test_model.py -v
"""

import os
import sys
import tempfile

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.language_model import BB8LM
from tokenizer import CharTokenizer
from datasets.text_dataset import TextDataset, create_train_val_split, split_text
from inference.generator import TextGenerator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VOCAB_SIZE = 100
D_MODEL = 32
NUM_LAYERS = 2
NUM_HEADS = 2
D_FF = 64
MAX_SEQ = 32


@pytest.fixture
def tiny_model():
    return BB8LM(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        max_seq_len=MAX_SEQ,
        dropout=0.0,
    )


@pytest.fixture
def tiny_tokenizer():
    tok = CharTokenizer()
    text = "abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ.,!?"
    tok.train([text])
    return tok


# ---------------------------------------------------------------------------
# BB8LM tests
# ---------------------------------------------------------------------------

class TestBB8LM:
    def test_forward_logits_shape(self, tiny_model):
        x = torch.randint(0, VOCAB_SIZE, (2, MAX_SEQ))
        output = tiny_model(x)
        assert "logits" in output
        assert output["logits"].shape == (2, MAX_SEQ, VOCAB_SIZE)

    def test_forward_with_targets_returns_loss(self, tiny_model):
        x = torch.randint(0, VOCAB_SIZE, (2, MAX_SEQ))
        y = torch.randint(0, VOCAB_SIZE, (2, MAX_SEQ))
        output = tiny_model(x, targets=y)
        assert "loss" in output
        assert output["loss"].item() > 0.0

    def test_loss_is_scalar(self, tiny_model):
        x = torch.randint(0, VOCAB_SIZE, (2, MAX_SEQ))
        y = torch.randint(0, VOCAB_SIZE, (2, MAX_SEQ))
        output = tiny_model(x, targets=y)
        assert output["loss"].ndim == 0  # scalar

    def test_backward_does_not_raise(self, tiny_model):
        x = torch.randint(0, VOCAB_SIZE, (2, MAX_SEQ))
        y = torch.randint(0, VOCAB_SIZE, (2, MAX_SEQ))
        output = tiny_model(x, targets=y)
        output["loss"].backward()  # must not raise

    def test_attention_weights_returned(self, tiny_model):
        x = torch.randint(0, VOCAB_SIZE, (1, 10))
        output = tiny_model(x, return_attn_weights=True)
        assert "attn_weights" in output
        assert len(output["attn_weights"]) == NUM_LAYERS

    def test_count_parameters_positive(self, tiny_model):
        assert tiny_model.count_parameters() > 0

    def test_get_config_returns_dict(self, tiny_model):
        cfg = tiny_model.get_config()
        assert cfg["vocab_size"] == VOCAB_SIZE
        assert cfg["d_model"] == D_MODEL
        assert cfg["num_layers"] == NUM_LAYERS
        assert cfg["dropout"] == 0.0
        assert cfg["activation"] == "gelu"
        assert cfg["tie_weights"] is True

    def test_save_and_load_checkpoint(self, tiny_model):
        x = torch.randint(0, VOCAB_SIZE, (1, 10))
        original_logits = tiny_model(x)["logits"].detach().clone()

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            torch.save({"model_state_dict": tiny_model.state_dict()}, path)
            ckpt = torch.load(path, map_location="cpu")
            tiny_model.load_state_dict(ckpt["model_state_dict"])
            restored_logits = tiny_model(x)["logits"].detach()
            assert torch.allclose(original_logits, restored_logits)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# TextDataset tests
# ---------------------------------------------------------------------------

class TestTextDataset:
    def test_dataset_len(self, tiny_tokenizer):
        text = "hello world " * 50
        ds = TextDataset(text, tiny_tokenizer, seq_len=16)
        assert len(ds) > 0

    def test_item_shapes(self, tiny_tokenizer):
        text = "hello world " * 50
        ds = TextDataset(text, tiny_tokenizer, seq_len=16)
        x, y = ds[0]
        assert x.shape == (16,)
        assert y.shape == (16,)

    def test_target_is_input_shifted_by_one(self, tiny_tokenizer):
        text = "abcdefghijklmnopqrstuvwxyz" * 10
        ds = TextDataset(text, tiny_tokenizer, seq_len=10)
        x, y = ds[0]
        # y should equal x shifted left (x[1:] == y[:-1])
        assert torch.all(x[1:] == y[:-1])

    def test_stride_reduces_overlapping_samples(self, tiny_tokenizer):
        text = "hello world " * 50
        dense = TextDataset(text, tiny_tokenizer, seq_len=16, stride=1)
        sparse = TextDataset(text, tiny_tokenizer, seq_len=16, stride=8)
        assert len(sparse) < len(dense)
        assert torch.equal(sparse[1][0], dense[8][0])

    def test_raw_text_split_is_disjoint(self):
        train_text, val_text = split_text("abcdefghij", val_fraction=0.2)
        assert train_text == "abcdefgh"
        assert val_text == "ij"
        assert train_text + val_text == "abcdefghij"

    def test_raw_text_split_can_preserve_record_boundaries(self):
        text = "first\n\nsecond\n\nthird\n\nfourth"
        train_text, val_text = split_text(
            text,
            val_fraction=0.4,
            record_separator="\n\n",
        )
        assert train_text.endswith("\n\n")
        assert not val_text.startswith("\n")
        assert train_text + val_text == text

    def test_train_val_split(self, tiny_tokenizer):
        text = "hello world " * 200
        ds = TextDataset(text, tiny_tokenizer, seq_len=16)
        train_ds, val_ds = create_train_val_split(ds, val_fraction=0.1)
        total = len(train_ds) + len(val_ds)
        assert total == len(ds)


# ---------------------------------------------------------------------------
# TextGenerator tests
# ---------------------------------------------------------------------------

class TestTextGenerator:
    def test_greedy_generation_returns_string(self, tiny_model, tiny_tokenizer):
        gen = TextGenerator(tiny_model, tiny_tokenizer)
        result = gen.generate("hello", max_new_tokens=10, strategy="greedy")
        assert isinstance(result, str)

    def test_top_p_generation_returns_string(self, tiny_model, tiny_tokenizer):
        gen = TextGenerator(tiny_model, tiny_tokenizer)
        result = gen.generate("hello", max_new_tokens=10, strategy="top_p")
        assert isinstance(result, str)

    def test_generated_longer_than_prompt(self, tiny_model, tiny_tokenizer):
        gen = TextGenerator(tiny_model, tiny_tokenizer)
        prompt = "ab"
        result = gen.generate(prompt, max_new_tokens=20, strategy="temperature")
        assert len(result) >= len(prompt)
