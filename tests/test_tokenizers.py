"""
test_tokenizers.py
------------------
Unit tests for all three BB8 tokenizers.

Run with:
    pytest tests/test_tokenizers.py -v
"""

import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenizer import BPETokenizer, CharTokenizer, WordTokenizer

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

TRAINING_TEXTS = [
    "Hello world! This is a test of the tokenizer.",
    "The quick brown fox jumps over the lazy dog.",
    "To be or not to be, that is the question.",
    "All that glitters is not gold.",
    "A journey of a thousand miles begins with a single step.",
]

SAMPLE_TEXT = "Hello world!"


# ---------------------------------------------------------------------------
# CharTokenizer tests
# ---------------------------------------------------------------------------

class TestCharTokenizer:
    def test_train_creates_vocab(self):
        tok = CharTokenizer()
        tok.train(TRAINING_TEXTS)
        assert tok.get_vocab_size() > 4  # at least special tokens + some chars

    def test_encode_length_equals_input_length(self):
        tok = CharTokenizer()
        tok.train(TRAINING_TEXTS)
        ids = tok.encode(SAMPLE_TEXT)
        assert len(ids) == len(SAMPLE_TEXT)

    def test_decode_roundtrip(self):
        tok = CharTokenizer()
        tok.train(TRAINING_TEXTS)
        ids = tok.encode(SAMPLE_TEXT)
        decoded = tok.decode(ids)
        assert decoded == SAMPLE_TEXT

    def test_special_tokens_are_first(self):
        tok = CharTokenizer()
        tok.train(TRAINING_TEXTS)
        assert tok.vocab["<PAD>"] == 0
        assert tok.vocab["<UNK>"] == 1
        assert tok.vocab["<BOS>"] == 2
        assert tok.vocab["<EOS>"] == 3

    def test_bos_eos_added_when_requested(self):
        tok = CharTokenizer()
        tok.train(TRAINING_TEXTS)
        ids = tok.encode("Hi", add_special_tokens=True)
        assert ids[0] == tok.vocab["<BOS>"]
        assert ids[-1] == tok.vocab["<EOS>"]

    def test_unknown_char_maps_to_unk(self):
        tok = CharTokenizer()
        tok.train(["abc"])
        ids = tok.encode("xyz")
        unk_id = tok.vocab["<UNK>"]
        assert all(i == unk_id for i in ids)

    def test_save_load_roundtrip(self):
        tok = CharTokenizer()
        tok.train(TRAINING_TEXTS)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            tok.save(path)
            tok2 = CharTokenizer()
            tok2.load(path)
            assert tok2.vocab == tok.vocab
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# WordTokenizer tests
# ---------------------------------------------------------------------------

class TestWordTokenizer:
    def test_train_creates_vocab(self):
        tok = WordTokenizer()
        tok.train(TRAINING_TEXTS, vocab_size=200)
        assert tok.get_vocab_size() > 4

    def test_vocab_size_respected(self):
        tok = WordTokenizer()
        tok.train(TRAINING_TEXTS, vocab_size=20)
        assert tok.get_vocab_size() <= 20

    def test_encode_returns_list_of_ints(self):
        tok = WordTokenizer()
        tok.train(TRAINING_TEXTS)
        ids = tok.encode(SAMPLE_TEXT)
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)

    def test_decode_returns_string(self):
        tok = WordTokenizer()
        tok.train(TRAINING_TEXTS)
        ids = tok.encode(SAMPLE_TEXT)
        decoded = tok.decode(ids)
        assert isinstance(decoded, str)
        assert len(decoded) > 0

    def test_oov_word_maps_to_unk(self):
        tok = WordTokenizer()
        tok.train(["apple banana cherry"], vocab_size=10)
        ids = tok.encode("xyznotaword")
        assert tok.vocab["<UNK>"] in ids


# ---------------------------------------------------------------------------
# BPETokenizer tests
# ---------------------------------------------------------------------------

class TestBPETokenizer:
    def test_train_creates_vocab(self):
        tok = BPETokenizer()
        tok.train(TRAINING_TEXTS, vocab_size=100)
        assert tok.get_vocab_size() > 4

    def test_vocab_size_does_not_exceed_target(self):
        tok = BPETokenizer()
        tok.train(TRAINING_TEXTS, vocab_size=100)
        assert tok.get_vocab_size() <= 100

    def test_encode_returns_list_of_ints(self):
        tok = BPETokenizer()
        tok.train(TRAINING_TEXTS, vocab_size=100)
        ids = tok.encode(SAMPLE_TEXT)
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)

    def test_decode_returns_string(self):
        tok = BPETokenizer()
        tok.train(TRAINING_TEXTS, vocab_size=100)
        ids = tok.encode("hello world")
        decoded = tok.decode(ids)
        assert isinstance(decoded, str)

    def test_merges_are_learned(self):
        tok = BPETokenizer()
        tok.train(TRAINING_TEXTS, vocab_size=100)
        assert len(tok.merges) > 0

    def test_special_tokens_present(self):
        tok = BPETokenizer()
        tok.train(TRAINING_TEXTS, vocab_size=50)
        for special in ("<PAD>", "<UNK>", "<BOS>", "<EOS>"):
            assert special in tok.vocab

    def test_save_load_roundtrip(self):
        tok = BPETokenizer()
        tok.train(TRAINING_TEXTS, vocab_size=80)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            tok.save(path)
            tok2 = BPETokenizer()
            tok2.load(path)
            assert tok2.vocab == tok.vocab
            assert tok2.merges == tok.merges
            assert tok2.encode(SAMPLE_TEXT) == tok.encode(SAMPLE_TEXT)
        finally:
            os.unlink(path)

    def test_merge_matches_complete_symbols_only(self):
        word_vocab = {"xa b </w>": 1}
        merged = BPETokenizer._apply_merge(("a", "b"), word_vocab)
        assert merged == word_vocab
