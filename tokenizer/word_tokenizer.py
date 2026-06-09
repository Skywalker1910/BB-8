"""
word_tokenizer.py
-----------------
Word-level tokenizer.

Splits text on whitespace and punctuation using a simple regex, then builds
a vocabulary of the most frequent words up to *vocab_size*.

Advantages
----------
- Shorter sequences than character-level.
- Directly interpretable tokens.

Disadvantages
-------------
- Large vocabulary required to cover a diverse corpus.
- Out-of-vocabulary (OOV) problem: rare or novel words map to <UNK>.
- Cannot represent morphological variants (run / running / ran all differ).
"""

import re
from collections import Counter
from typing import List

from .base_tokenizer import BaseTokenizer


class WordTokenizer(BaseTokenizer):
    """
    Word-level tokenizer with frequency-based vocabulary truncation.

    Tokenisation is done with the regex ``\\w+|[^\\w\\s]``, which keeps
    whole words and isolates punctuation as separate tokens.  All text is
    lower-cased before processing to reduce vocabulary fragmentation.

    Example
    -------
    >>> tok = WordTokenizer()
    >>> tok.train(["The quick brown fox."], vocab_size=100)
    >>> tok.encode("The quick fox")
    [4, 5, 7]                   # IDs will vary
    """

    # Splits on whole words OR single non-word, non-space characters
    _TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]")

    def _tokenize(self, text: str) -> List[str]:
        """Split *text* into word/punctuation tokens (lower-cased)."""
        return self._TOKEN_PATTERN.findall(text.lower())

    def train(self, texts: List[str], vocab_size: int = 10_000) -> None:
        """
        Build a frequency-ranked vocabulary from *texts*.

        The *vocab_size* most common tokens are kept; the rest will map to
        <UNK> at encode time.
        """
        counter: Counter = Counter()
        for text in texts:
            counter.update(self._tokenize(text))

        # Reserve space for the four special tokens
        n_words = vocab_size - len(self.SPECIAL_TOKENS)
        most_common = [word for word, _ in counter.most_common(n_words)]

        tokens = self.SPECIAL_TOKENS + most_common
        self.vocab = {token: idx for idx, token in enumerate(tokens)}
        self._build_inverse_vocab()

        oov_count = sum(1 for _, c in counter.items() if _ not in self.vocab)
        print(
            f"WordTokenizer trained  |  vocab_size={self.get_vocab_size()}"
            f"  |  OOV types={oov_count}"
        )

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """
        Tokenise *text* and return a list of token IDs.

        Words not in the vocabulary are replaced with <UNK>.
        """
        unk_id = self.vocab[self.unk_token]
        ids = [self.vocab.get(tok, unk_id) for tok in self._tokenize(text)]

        if add_special_tokens:
            ids = [self.vocab[self.bos_token]] + ids + [self.vocab[self.eos_token]]

        return ids

    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs back to a whitespace-joined string.

        Special tokens are stripped.  Note that the original spacing around
        punctuation cannot be perfectly recovered.
        """
        special_ids = {self.vocab[t] for t in self.special_tokens if t in self.vocab}
        tokens = [
            self.inverse_vocab[idx]
            for idx in token_ids
            if idx not in special_ids and idx in self.inverse_vocab
        ]
        return " ".join(tokens)
