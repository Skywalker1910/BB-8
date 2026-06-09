"""
losses.py
---------
Loss function utilities for BB8.
"""

import math

import torch
import torch.nn.functional as F


def cross_entropy_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -1,
) -> torch.Tensor:
    """
    Compute cross-entropy loss for language modelling.

    Parameters
    ----------
    logits       : (batch, seq_len, vocab_size)
    targets      : (batch, seq_len)
    ignore_index : token ID to ignore in the loss (e.g. padding)

    Returns
    -------
    Scalar loss tensor.
    """
    vocab_size = logits.size(-1)
    return F.cross_entropy(
        logits.view(-1, vocab_size),
        targets.view(-1),
        ignore_index=ignore_index,
    )


def perplexity(loss: float) -> float:
    """
    Compute perplexity from cross-entropy loss (in nats).

        PPL = exp(loss)

    Perplexity is the exponentiation of the average negative log-likelihood.
    A perplexity of k means the model is as uncertain as if it were choosing
    uniformly among k options at each step.  Lower is better.
    """
    return math.exp(loss)


def bits_per_character(loss: float) -> float:
    """
    Convert cross-entropy loss from nats to bits per character.

        BPC = loss / log(2)

    BPC is a common metric for character-level language models.
    A BPC of 1.0 means the model uses on average 1 bit per character, which
    corresponds to a random binary process.  State-of-the-art models achieve
    < 1.0 BPC on standard benchmarks.
    """
    return loss / math.log(2)
