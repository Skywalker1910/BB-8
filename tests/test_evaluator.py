"""Tests for tokenizer-normalized language-model metrics."""

import math

import pytest
import torch
from torch.utils.data import DataLoader

from datasets.text_dataset import TextDataset
from evaluation.evaluator import Evaluator
from models.language_model import BB8LM
from tokenizer import CharTokenizer


def test_character_dataset_reports_bits_per_token_and_character():
    text = "a small evaluation corpus. " * 20
    tokenizer = CharTokenizer()
    tokenizer.train([text], vocab_size=100)
    dataset = TextDataset(text, tokenizer, seq_len=16, stride=16)
    loader = DataLoader(dataset, batch_size=4)
    model = BB8LM(
        vocab_size=tokenizer.get_vocab_size(),
        d_model=16,
        num_layers=1,
        num_heads=2,
        d_ff=32,
        max_seq_len=16,
        dropout=0.0,
    )

    metrics = Evaluator(model, tokenizer, device="cpu").evaluate_dataset(loader)

    assert metrics["bits_per_token"] == pytest.approx(
        metrics["loss"] / math.log(2)
    )
    assert metrics["bits_per_char"] == pytest.approx(
        metrics["bits_per_token"] * dataset.n_tokens / dataset.n_characters
    )
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_evaluate_text_normalizes_bits_by_source_characters():
    text = "hello evaluator"
    tokenizer = CharTokenizer()
    tokenizer.train([text], vocab_size=100)
    model = BB8LM(
        vocab_size=tokenizer.get_vocab_size(),
        d_model=16,
        num_layers=1,
        num_heads=2,
        d_ff=32,
        max_seq_len=16,
        dropout=0.0,
    )

    metrics = Evaluator(model, tokenizer, device="cpu").evaluate_text(text)

    assert metrics["bits_per_char"] == pytest.approx(metrics["bits_per_token"])
