"""
char_tokenizer.py
-----------------
Character-level tokenizer.

Each unique character becomes one token.  This gives the smallest possible
vocabulary (typically 60-100 tokens for English text) but produces the
longest token sequences.

Advantages
----------
- No out-of-vocabulary problem — every possible character is representable.
- Tiny vocabulary → small embedding table.

Disadvantages
-------------
- Sequences are very long (one token per character).
- The model must learn to compose meaning from individual characters.
- High perplexity scores compared to subword methods.
"""

from typing import List

from .base_tokenizer import BaseTokenizer


class CharTokenizer(BaseTokenizer):
    """
    Character-level tokenizer.

    Training simply collects every unique character that appears in the
    training corpus.  Encoding maps each character to its ID; decoding
    joins characters back into a string.

    Example
    -------
    >>> tok = CharTokenizer()
    >>> tok.train(["Hello, World!"])
    >>> tok.encode("Hello")
    [8, 9, 10, 10, 13]          # IDs will vary
    >>> tok.decode([8, 9, 10, 10, 13])
    'Hello'
    """

    def train(self, texts: List[str], vocab_size: int = None) -> None:  # noqa: ARG002
        """
        Build vocabulary from all unique characters in *texts*.

        The ``vocab_size`` parameter is accepted for API compatibility but
        is ignored — the vocabulary always contains every character seen.
        """
        unique_chars: set[str] = set()
        for text in texts:
            unique_chars.update(text)

        # Deterministic ordering: special tokens first, then sorted characters
        tokens = self.SPECIAL_TOKENS + sorted(unique_chars)
        self.vocab = {token: idx for idx, token in enumerate(tokens)}
        self._build_inverse_vocab()

        print(
            f"CharTokenizer trained  |  vocab_size={self.get_vocab_size()}"
        )

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """
        Encode *text* to a list of character-level token IDs.

        Unknown characters are mapped to the <UNK> token.
        """
        unk_id = self.vocab[self.unk_token]
        ids = [self.vocab.get(ch, unk_id) for ch in text]

        if add_special_tokens:
            ids = [self.vocab[self.bos_token]] + ids + [self.vocab[self.eos_token]]

        return ids

    def decode(self, token_ids: List[int]) -> str:
        """
        Decode a list of token IDs back to a string.

        Special tokens (<PAD>, <UNK>, <BOS>, <EOS>) are stripped.
        """
        special_ids = {self.vocab[t] for t in self.special_tokens if t in self.vocab}
        chars = [
            self.inverse_vocab[idx]
            for idx in token_ids
            if idx not in special_ids and idx in self.inverse_vocab
        ]
        return "".join(chars)
