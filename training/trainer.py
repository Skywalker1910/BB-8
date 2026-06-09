"""
trainer.py
----------
Training framework for BB8 Language Model.

Responsibilities
----------------
- Training loop with per-step and per-epoch logging
- Validation evaluation
- Learning rate scheduling (cosine + warmup)
- Gradient clipping (prevents exploding gradients)
- Checkpoint saving and loading
- Training history export (JSON)

Design notes
------------
AdamW is used as the optimizer.  Weight decay is applied only to weight
matrices — not to bias vectors or LayerNorm parameters — following the
common practice in GPT-style models.  This is implemented by creating two
separate parameter groups.

Gradient clipping to max_norm=1.0 prevents the rare but catastrophic
gradient spikes that can occur early in training.
"""

import json
import math
import os
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from .scheduler import get_cosine_schedule_with_warmup


class Trainer:
    """
    Training loop for BB8LM.

    Parameters
    ----------
    model          : nn.Module      — BB8LM instance
    train_loader   : DataLoader     — training data
    val_loader     : DataLoader     — validation data (optional)
    lr             : float          — peak learning rate (default 3e-4)
    weight_decay   : float          — L2 regularisation on weight matrices
    max_grad_norm  : float          — gradient clipping threshold
    warmup_steps   : int            — linear warmup steps
    max_steps      : int | None     — total steps; inferred if None
    checkpoint_dir : str            — directory for saving checkpoints
    log_interval   : int            — print loss every N steps
    eval_interval  : int            — evaluate on val set every N steps
    save_interval  : int            — save checkpoint every N steps
    device         : str | None     — 'cuda', 'cpu', or None (auto-detect)
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        lr: float = 3e-4,
        weight_decay: float = 0.1,
        max_grad_norm: float = 1.0,
        warmup_steps: int = 100,
        max_steps: Optional[int] = None,
        checkpoint_dir: str = "checkpoints",
        log_interval: int = 10,
        eval_interval: int = 100,
        save_interval: int = 500,
        device: Optional[str] = None,
    ) -> None:
        # Device selection
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.max_grad_norm = max_grad_norm
        self.checkpoint_dir = checkpoint_dir
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.save_interval = save_interval

        # Optimizer — separate param groups for weight decay
        self.optimizer = self._build_optimizer(lr, weight_decay)

        # Scheduler — total steps inferred from loader if not given
        if max_steps is None:
            max_steps = len(train_loader) * 10
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, warmup_steps, max_steps
        )

        # State tracking
        self.global_step: int = 0
        self.current_epoch: int = 0
        self.best_val_loss: float = float("inf")
        self.train_loss_history: List[float] = []
        self.val_loss_history: List[float] = []

        os.makedirs(checkpoint_dir, exist_ok=True)

        print(
            f"Trainer ready  |  device={self.device}"
            f"  |  peak_lr={lr}  warmup={warmup_steps}  max_steps={max_steps}"
        )

    # ------------------------------------------------------------------
    # Optimiser
    # ------------------------------------------------------------------

    def _build_optimizer(self, lr: float, weight_decay: float) -> AdamW:
        """
        Build AdamW with weight decay applied only to weight matrices.

        Bias terms and LayerNorm parameters are excluded from regularisation.
        """
        decay, no_decay = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if any(nd in name for nd in ("bias", "norm", "LayerNorm")):
                no_decay.append(param)
            else:
                decay.append(param)

        param_groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return AdamW(param_groups, lr=lr, betas=(0.9, 0.95), eps=1e-8)

    # ------------------------------------------------------------------
    # Single training step
    # ------------------------------------------------------------------

    def train_step(self, batch) -> float:
        """Perform one gradient update. Returns the scalar loss."""
        self.model.train()
        x, y = [t.to(self.device) for t in batch]

        self.optimizer.zero_grad()
        output = self.model(x, targets=y)
        loss: torch.Tensor = output["loss"]
        loss.backward()

        # Clip gradients to prevent exploding updates
        if self.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

        self.optimizer.step()
        self.scheduler.step()

        return loss.item()

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Compute loss and perplexity on the validation set."""
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for x, y in self.val_loader:
            x, y = x.to(self.device), y.to(self.device)
            output = self.model(x, targets=y)
            total_loss += output["loss"].item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        return {
            "val_loss": avg_loss,
            "perplexity": math.exp(avg_loss),
        }

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self, num_epochs: int = 10) -> Dict:
        """
        Run the full training loop.

        Parameters
        ----------
        num_epochs : int — number of passes over the training data

        Returns
        -------
        history : dict with keys 'train_loss', 'val_loss', 'perplexity', 'lr'
        """
        steps_per_epoch = len(self.train_loader)
        total_steps = steps_per_epoch * num_epochs

        print(f"\n{'='*60}")
        print(f"  Training BB8 for {num_epochs} epochs")
        print(f"  steps/epoch={steps_per_epoch}  total_steps={total_steps}")
        print(f"{'='*60}\n")

        history: Dict = {
            "train_loss": [],
            "val_loss": [],
            "perplexity": [],
            "lr": [],
        }

        for epoch in range(num_epochs):
            self.current_epoch = epoch
            epoch_loss_sum = 0.0
            epoch_steps = 0

            pbar = tqdm(
                self.train_loader,
                desc=f"Epoch {epoch + 1}/{num_epochs}",
                dynamic_ncols=True,
            )

            for batch in pbar:
                loss = self.train_step(batch)
                self.global_step += 1
                epoch_loss_sum += loss
                epoch_steps += 1
                self.train_loss_history.append(loss)

                current_lr = self.optimizer.param_groups[0]["lr"]

                # Periodic logging
                if self.global_step % self.log_interval == 0:
                    pbar.set_postfix(
                        {
                            "loss": f"{loss:.4f}",
                            "ppl": f"{math.exp(min(loss, 20)):.1f}",
                            "lr": f"{current_lr:.2e}",
                        }
                    )

                # Periodic validation
                if (
                    self.val_loader is not None
                    and self.global_step % self.eval_interval == 0
                ):
                    metrics = self.evaluate()
                    val_loss = metrics.get("val_loss", float("inf"))
                    self.val_loss_history.append(val_loss)
                    tqdm.write(
                        f"  [step {self.global_step}] "
                        f"val_loss={val_loss:.4f}  "
                        f"ppl={metrics.get('perplexity', 0):.1f}"
                    )
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.save_checkpoint("best_model.pt")

                # Periodic checkpoint
                if self.global_step % self.save_interval == 0:
                    self.save_checkpoint(f"checkpoint_step_{self.global_step}.pt")

            # End-of-epoch summary
            avg_epoch_loss = epoch_loss_sum / max(epoch_steps, 1)
            history["train_loss"].append(avg_epoch_loss)
            history["lr"].append(self.optimizer.param_groups[0]["lr"])

            epoch_msg = (
                f"Epoch {epoch + 1}/{num_epochs}  "
                f"avg_loss={avg_epoch_loss:.4f}  "
                f"ppl={math.exp(min(avg_epoch_loss, 20)):.1f}"
            )

            if self.val_loader is not None:
                metrics = self.evaluate()
                val_loss = metrics.get("val_loss", 0.0)
                ppl = metrics.get("perplexity", 0.0)
                history["val_loss"].append(val_loss)
                history["perplexity"].append(ppl)
                epoch_msg += f"  val_loss={val_loss:.4f}  val_ppl={ppl:.1f}"

            print(f"\n{epoch_msg}\n")

        # Save final checkpoint and history
        self.save_checkpoint("final_model.pt")
        self._save_history(history)

        return history

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, filename: str) -> None:
        """Save model, optimizer and scheduler state."""
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(
            {
                "global_step": self.global_step,
                "current_epoch": self.current_epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "best_val_loss": self.best_val_loss,
                "model_config": (
                    self.model.get_config()
                    if hasattr(self.model, "get_config")
                    else {}
                ),
            },
            path,
        )
        print(f"  ✓ checkpoint saved → {path}")

    def load_checkpoint(self, path: str) -> None:
        """Restore training state from a checkpoint file."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.global_step = ckpt["global_step"]
        self.current_epoch = ckpt["current_epoch"]
        self.best_val_loss = ckpt["best_val_loss"]
        print(
            f"Loaded checkpoint ← {path}"
            f"  (step={self.global_step}  epoch={self.current_epoch})"
        )

    def _save_history(self, history: Dict) -> None:
        path = os.path.join(self.checkpoint_dir, "training_history.json")
        with open(path, "w") as fh:
            json.dump(history, fh, indent=2)
        print(f"  ✓ training history → {path}")
