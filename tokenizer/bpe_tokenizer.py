"""
bpe_tokenizer.py
----------------
Byte Pair Encoding (BPE) tokenizer — implemented from scratch.

BPE was introduced for NLP in:
    Sennrich et al. (2016). "Neural Machine Translation of Rare Words with
    Subword Units."  https://arxiv.org/abs/1508.07909

It is used (with modifications) in GPT-2, GPT-3, RoBERTa, and most modern
large language models.

Algorithm
---------
1. Start with a character-level vocabulary (each character is a token).
2. Count the frequency of every adjacent pair of tokens in the training
   corpus.
3. Merge the most frequent pair into a single new token.
4. Repeat steps 2–3 until the vocabulary reaches the desired size.

The resulting vocabulary contains frequent subword units (morphemes,
syllables, common word fragments) which balances vocabulary size against
sequence length better than either character-level or word-level approaches.

Advantages
----------
- Handles unseen words gracefully (falls back to character pieces).
- Controllable vocabulary size.
- Good compression ratio compared to character-level.

Disadvantages
-------------
- Training is O(V * corpus_size) — slower than simple tokenizers.
- Merge order must be saved and applied consistently at encode time.
"""

import json
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

from .base_tokenizer import BaseTokenizer


class BPETokenizer(BaseTokenizer):
    """
    BPE tokenizer built from scratch.

    Attributes
    ----------
    merges : List[Tuple[str, str]]
        Ordered list of merge rules learned during training.
    merge_rules : Dict[Tuple[str, str], str]
        Fast lookup: pair → merged token string.

    Example
    -------
    >>> tok = BPETokenizer()
    >>> tok.train(["low lower newest widest"], vocab_size=50)
    >>> tok.encode("lowest")
    [...]
    """

    # End-of-word marker: appended to each word so the tokenizer knows
    # where word boundaries are.  Adopted from the original BPE paper.
    EOW = "</w>"

    def __init__(self) -> None:
        super().__init__()
        self.merges: List[Tuple[str, str]] = []
        self.merge_rules: Dict[Tuple[str, str], str] = {}
        self._word_cache: Dict[str, List[str]] = {}

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------

    def _build_word_vocab(self, texts: List[str]) -> Dict[str, int]:
        """
        Convert raw texts into a word frequency table where each word is
        represented as a space-separated sequence of characters + EOW.

        E.g.  "low"  →  "l o w </w>"   with its frequency count.
        """
        counter: Counter = Counter()
        for text in texts:
            for word in text.strip().split():
                segmented = " ".join(list(word)) + " " + self.EOW
                counter[segmented] += 1
        return dict(counter)

    @staticmethod
    def _get_pair_counts(
        word_vocab: Dict[str, int]
    ) -> Dict[Tuple[str, str], int]:
        """Count all adjacent symbol pairs across the vocabulary."""
        pairs: Dict[Tuple[str, str], int] = defaultdict(int)
        for word, freq in word_vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs

    @staticmethod
    def _apply_merge(
        pair: Tuple[str, str], word_vocab: Dict[str, int]
    ) -> Dict[str, int]:
        """Replace every occurrence of *pair* in *word_vocab* with a merged token."""
        new_vocab: Dict[str, int] = {}
        for word, freq in word_vocab.items():
            symbols = word.split()
            merged_symbols: List[str] = []
            i = 0
            while i < len(symbols):
                if (
                    i < len(symbols) - 1
                    and symbols[i] == pair[0]
                    and symbols[i + 1] == pair[1]
                ):
                    merged_symbols.append("".join(pair))
                    i += 2
                else:
                    merged_symbols.append(symbols[i])
                    i += 1
            merged_word = " ".join(merged_symbols)
            new_vocab[merged_word] = new_vocab.get(merged_word, 0) + freq
        return new_vocab

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, texts: List[str], vocab_size: int = 5_000) -> None:
        """
        Learn BPE merge rules from *texts*.

        The algorithm starts with a character vocabulary and iteratively
        merges the most frequent adjacent pair until *vocab_size* is reached.
        """
        print(f"BPETokenizer: starting training  |  target vocab_size={vocab_size}")

        word_vocab = self._build_word_vocab(texts)

        # Collect all initial characters (+ EOW marker)
        init_chars: set = set()
        for word in word_vocab:
            init_chars.update(word.split())

        # Seed the vocabulary: special tokens + sorted characters
        base_tokens = self.SPECIAL_TOKENS + sorted(init_chars)
        self.vocab = {tok: idx for idx, tok in enumerate(base_tokens)}
        self._build_inverse_vocab()

        # How many merge operations do we need?
        num_merges = vocab_size - len(self.vocab)
        self.merges = []
        self.merge_rules = {}
        self._word_cache = {}

        for i in range(num_merges):
            pairs = self._get_pair_counts(word_vocab)
            if not pairs:
                break  # corpus fully merged

            best_pair = max(pairs, key=pairs.get)
            word_vocab = self._apply_merge(best_pair, word_vocab)

            merged_token = "".join(best_pair)
            self.merges.append(best_pair)
            self.merge_rules[best_pair] = merged_token

            if merged_token not in self.vocab:
                self.vocab[merged_token] = len(self.vocab)
                self.inverse_vocab[self.vocab[merged_token]] = merged_token

            if (i + 1) % 500 == 0:
                print(
                    f"  merges={i + 1}/{num_merges}"
                    f"  vocab_size={len(self.vocab)}"
                )

        print(
            f"BPETokenizer trained   |  vocab_size={self.get_vocab_size()}"
            f"  |  merges learned={len(self.merges)}"
        )

    def _tokenize_word(self, word: str) -> List[str]:
        """
        Apply the learned merge rules to a single word.

        The word is first split into individual characters with an EOW
        marker appended, then merges are applied left-to-right in the
        order they were learned.
        """
        symbols = list(word) + [self.EOW]

        for merge_pair in self.merges:
            merged = self.merge_rules[merge_pair]
            i = 0
            new_symbols: List[str] = []
            while i < len(symbols):
                if (
                    i < len(symbols) - 1
                    and symbols[i] == merge_pair[0]
                    and symbols[i + 1] == merge_pair[1]
                ):
                    new_symbols.append(merged)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols

        return symbols

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """
        Encode *text* into BPE token IDs.

        Unknown subword pieces fall back to the <UNK> token ID.
        """
        unk_id = self.vocab[self.unk_token]
        ids: List[int] = []

        if add_special_tokens:
            ids.append(self.vocab[self.bos_token])

        # Repeated words are common in language-model corpora. Caching their
        # segmentation avoids replaying every learned merge for every
        # occurrence, which makes larger instruction datasets practical.
        for word in text.strip().split():
            pieces = self._word_cache.get(word)
            if pieces is None:
                pieces = self._tokenize_word(word)
                self._word_cache[word] = pieces
            for piece in pieces:
                ids.append(self.vocab.get(piece, unk_id))

        if add_special_tokens:
            ids.append(self.vocab[self.eos_token])

        return ids

    def decode(self, token_ids: List[int]) -> str:
        """
        Decode BPE token IDs back to a text string.

        EOW markers are replaced with spaces; special tokens are stripped.
        """
        special_ids = {self.vocab[t] for t in self.special_tokens if t in self.vocab}
        pieces = [
            self.inverse_vocab[idx]
            for idx in token_ids
            if idx not in special_ids and idx in self.inverse_vocab
        ]
        text = "".join(pieces)
        # EOW marks word boundaries — replace with spaces then strip edges
        text = text.replace(self.EOW, " ").strip()
        return text

    def save(self, path: str) -> None:
        """Persist the vocabulary and ordered BPE merge rules."""
        data = {
            "vocab": self.vocab,
            "merges": [list(pair) for pair in self.merges],
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
        print(f"Tokenizer saved -> {path}  (vocab_size={self.get_vocab_size()})")

    def load(self, path: str) -> None:
        """Restore the vocabulary and ordered BPE merge rules."""
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        self.vocab = data["vocab"]
        self._build_inverse_vocab()
        self.merges = [tuple(pair) for pair in data.get("merges", [])]
        self.merge_rules = {pair: "".join(pair) for pair in self.merges}
        self._word_cache = {}
        print(f"Tokenizer loaded <- {path}  (vocab_size={self.get_vocab_size()})")
