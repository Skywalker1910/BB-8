"""
text_dataset.py
---------------
PyTorch Dataset wrappers for language-model training.

Language modelling uses a **next-token prediction** objective:
    Given tokens x_1, x_2, ..., x_T  predict  x_2, x_3, ..., x_{T+1}

Each dataset item is therefore a (input, target) pair where target is the
input sequence shifted one position to the right.

Classes
-------
TextDataset
    Constructed from an in-memory string.
FileTextDataset
    Convenience subclass that reads from a file path.

Utilities
---------
split_text(text, val_fraction)
    Split raw text before tokenizer training to avoid validation leakage.
create_train_val_split(dataset, val_fraction)
    Backward-compatible sequential split for an existing dataset.
"""

import os
from typing import Tuple

import torch
from torch.utils.data import Dataset, Subset


class TextDataset(Dataset):
    """
    Character or token-level dataset for language modelling.

    The full text is tokenised once at construction time. Indexing returns
    overlapping windows of length *seq_len* as (input_ids, target_ids)
    pairs.

    Parameters
    ----------
    text : str
        Raw training text.
    tokenizer : BaseTokenizer
        A trained tokenizer with an ``encode`` method.
    seq_len : int
        Context window length (number of input tokens per sample).
    stride : int
        Number of tokens between the start of adjacent samples. A stride of
        one creates maximally overlapping windows; larger values train faster.

    Notes
    -----
    The number of samples is ``len(token_ids) - seq_len``.  Each sample
    starts one position later than the previous, so samples heavily overlap.
    This is standard practice for character/token-level LM training.
    """

    def __init__(
        self,
        text: str,
        tokenizer,
        seq_len: int = 256,
        stride: int = 1,
    ) -> None:
        if seq_len < 1:
            raise ValueError("seq_len must be positive")
        if stride < 1:
            raise ValueError("stride must be positive")
        self.seq_len = seq_len
        self.stride = stride
        self.tokenizer = tokenizer
        self.n_characters = len(text)

        # Encode the entire corpus into a flat list of token IDs
        self.token_ids: list = tokenizer.encode(text)
        self.n_tokens = len(self.token_ids)

        print(
            f"TextDataset  |  tokens={self.n_tokens:,}"
            f"  seq_len={seq_len}"
            f"  stride={stride}"
            f"  samples={len(self):,}"
        )

    def __len__(self) -> int:
        # Each sample needs seq_len tokens for input + 1 for the target.
        if self.n_tokens <= self.seq_len:
            return 0
        return 1 + (self.n_tokens - self.seq_len - 1) // self.stride

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a (input, target) pair.

        input  = token_ids[idx : idx + seq_len]
        target = token_ids[idx + 1 : idx + seq_len + 1]
        """
        start = idx * self.stride
        chunk = self.token_ids[start : start + self.seq_len + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


class FileTextDataset(TextDataset):
    """
    TextDataset loaded directly from a UTF-8 text file.

    Parameters
    ----------
    file_path : str
        Path to the text file.
    tokenizer : BaseTokenizer
        A trained tokenizer.
    seq_len : int
        Context window length.
    """

    def __init__(
        self,
        file_path: str,
        tokenizer,
        seq_len: int = 256,
        stride: int = 1,
    ) -> None:
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Training data not found: {file_path}")
        with open(file_path, "r", encoding="utf-8") as fh:
            text = fh.read()
        file_size_mb = os.path.getsize(file_path) / (1024 ** 2)
        print(
            f"Loaded '{file_path}'  |  {len(text):,} chars  ({file_size_mb:.2f} MB)"
        )
        super().__init__(text, tokenizer, seq_len, stride)


def split_text(
    text: str,
    val_fraction: float = 0.1,
    record_separator: str | None = None,
) -> Tuple[str, str]:
    """Sequentially split raw text before fitting the tokenizer.

    Fitting the tokenizer on only ``train_text`` prevents BPE merges and word
    vocabulary entries from learning information from the validation corpus.
    When supplied, ``record_separator`` moves the boundary to the end of a
    complete record so instruction/response pairs are not divided across sets.
    """

    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")
    split_index = int(len(text) * (1.0 - val_fraction))
    if record_separator:
        next_boundary = text.find(record_separator, split_index)
        if next_boundary >= 0:
            split_index = next_boundary + len(record_separator)
        else:
            previous_boundary = text.rfind(record_separator, 0, split_index)
            if previous_boundary >= 0:
                split_index = previous_boundary + len(record_separator)
    if split_index < 2 or len(text) - split_index < 2:
        raise ValueError("Text is too short for the requested train/validation split")
    return text[:split_index], text[split_index:]


def create_train_val_split(
    dataset: TextDataset, val_fraction: float = 0.1
) -> Tuple[Subset, Subset]:
    """
    Split *dataset* into training and validation subsets.

    The split is **sequential** (not random) to preserve temporal order.
    Validation data comes from the end of the corpus.

    Parameters
    ----------
    dataset : TextDataset
        The full dataset.
    val_fraction : float
        Fraction of samples assigned to validation (default 0.1 = 10 %).

    Returns
    -------
    train_dataset, val_dataset : (Subset, Subset)
    """
    n = len(dataset)
    val_size = max(1, int(n * val_fraction))
    train_size = n - val_size

    train_ds = Subset(dataset, range(train_size))
    val_ds = Subset(dataset, range(train_size, n))

    print(
        f"Split  |  train={train_size:,} samples  val={val_size:,} samples"
        f"  ({val_fraction:.0%} val)"
    )
    return train_ds, val_ds
