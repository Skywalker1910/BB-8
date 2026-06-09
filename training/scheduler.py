"""
scheduler.py
------------
Learning rate schedulers for BB8.

Why schedule the learning rate?
--------------------------------
Using a constant learning rate is rarely optimal:
- Too high → unstable early training (large random gradients)
- Too low  → slow convergence

A warmup + cosine decay schedule is the standard recipe for Transformer
language models (GPT-2, GPT-3, LLaMA all use it):

    Phase 1 — Linear warmup (0 → max_lr over *warmup_steps*)
    Phase 2 — Cosine annealing (max_lr → min_lr over remaining steps)

The minimum LR is typically 10% of the peak LR.
"""

import math

import torch


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    max_steps: int,
    min_lr_ratio: float = 0.1,
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Cosine annealing schedule with linear warmup.

    Parameters
    ----------
    optimizer     : PyTorch optimiser
    warmup_steps  : number of steps for linear warmup phase
    max_steps     : total training steps (warmup + decay)
    min_lr_ratio  : floor for the LR multiplier (default 0.1 = 10% of peak)

    Returns
    -------
    LambdaLR scheduler
    """

    def lr_lambda(step: int) -> float:
        # Linear warmup
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        # Cosine decay
        progress = float(step - warmup_steps) / max(1, max_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def get_linear_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    max_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Linear warmup then linear decay to zero.

    Simpler than cosine but performs similarly for short runs.
    """

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        return max(
            0.0,
            float(max_steps - step) / max(1, max_steps - warmup_steps),
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
