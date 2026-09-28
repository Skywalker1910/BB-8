"""
base_tokenizer.py
-----------------
Abstract base class that all BB8 tokenizers inherit from.

Defines the interface every tokenizer must implement:
  - train(texts, vocab_size)
  - encode(text) -> List[int]
  - decode(token_ids) -> str
  - save / load vocab to/from JSON
"""

import json
from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class BaseTokenizer(ABC):
    """
    Abstract base class for all BB8 tokenizers.

    All tokenizers share a common vocabulary format:
        vocab : Dict[str, int]   token_string -> token_id
        inverse_vocab : Dict[int, str]   token_id -> token_string

    Four special tokens are reserved at fixed low IDs:
        <PAD>  – padding                         id = 0
        <UNK>  – unknown / out-of-vocabulary     id = 1
        <BOS>  – beginning-of-sequence           id = 2
        <EOS>  – end-of-sequence                 id = 3
    """

    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    BOS_TOKEN = "<BOS>"
    EOS_TOKEN = "<EOS>"
    SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]

    def __init__(self) -> None:
        self.vocab: Dict[str, int] = {}
        self.inverse_vocab: Dict[int, str] = {}

        # Convenience aliases so subclasses can use self.unk_token etc.
        self.pad_token = self.PAD_TOKEN
        self.unk_token = self.UNK_TOKEN
        self.bos_token = self.BOS_TOKEN
        self.eos_token = self.EOS_TOKEN
        self.special_tokens = self.SPECIAL_TOKENS

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def train(self, texts: List[str], vocab_size: int) -> None:
        """
        Build the vocabulary from a list of training texts.

        Args:
            texts:      List of raw text strings used to derive the vocab.
            vocab_size: Maximum target vocabulary size.
        """

    @abstractmethod
    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """
        Convert a text string to a list of token IDs.

        Args:
            text:               Input string.
            add_special_tokens: If True, prepend <BOS> and append <EOS>.

        Returns:
            List of integer token IDs.
        """

    @abstractmethod
    def decode(self, token_ids: List[int]) -> str:
        """
        Convert a list of token IDs back to a text string.

        Special tokens are stripped from the output.

        Args:
            token_ids: List of integer token IDs.

        Returns:
            Decoded text string.
        """

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def get_vocab_size(self) -> int:
        """Return the number of tokens in the vocabulary."""
        return len(self.vocab)

    def _build_inverse_vocab(self) -> None:
        """Rebuild inverse_vocab from vocab. Call after modifying vocab."""
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

    def _init_special_tokens(self) -> None:
        """Populate vocab with the four special tokens at IDs 0-3."""
        for idx, token in enumerate(self.SPECIAL_TOKENS):
            self.vocab[token] = idx
        self._build_inverse_vocab()

    def save(self, path: str) -> None:
        """Persist the vocabulary to a JSON file."""
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"vocab": self.vocab}, fh, ensure_ascii=False, indent=2)
        print(f"Tokenizer saved -> {path}  (vocab_size={self.get_vocab_size()})")

    def load(self, path: str) -> None:
        """Restore vocabulary from a JSON file produced by save()."""
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        self.vocab = data["vocab"]
        self._build_inverse_vocab()
        print(f"Tokenizer loaded <- {path}  (vocab_size={self.get_vocab_size()})")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(vocab_size={self.get_vocab_size()})"
