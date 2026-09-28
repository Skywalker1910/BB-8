"""Tests for loading a complete inference bundle."""

import json

import torch

from inference.model_loader import load_model_bundle
from models.language_model import BB8LM
from tokenizer.char_tokenizer import CharTokenizer


def test_load_model_bundle(tmp_path):
    tokenizer = CharTokenizer()
    tokenizer.train(["abc abc"])
    tokenizer.save(str(tmp_path / "tokenizer.json"))

    model = BB8LM(
        vocab_size=tokenizer.get_vocab_size(),
        d_model=16,
        num_layers=1,
        num_heads=2,
        d_ff=32,
        max_seq_len=16,
        dropout=0.0,
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model.get_config(),
        },
        tmp_path / "model.pt",
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps({"model_name": "test-model", "tokenizer_type": "char"}),
        encoding="utf-8",
    )

    bundle = load_model_bundle(tmp_path, device="cpu")

    assert bundle.name == "test-model"
    assert bundle.device == "cpu"
    assert bundle.generator.tokenizer.get_vocab_size() == tokenizer.get_vocab_size()
    assert bundle.generator.generate("a", max_new_tokens=2, strategy="greedy")

