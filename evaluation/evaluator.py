"""
evaluator.py
------------
Evaluation framework for BB8.

Metrics
-------
Perplexity
    The primary metric for language models.  Defined as the exponentiation
    of the average cross-entropy loss:

        PPL = exp( -1/N Σ log P(xₜ | x<ₜ) )

    Intuitively, a perplexity of k means the model is as uncertain as if it
    were choosing uniformly among k equally likely options at every step.
    Lower is better.  Well-trained character-level models typically achieve
    PPL < 5 on their training domain.

Bits per Character (BPC)
    Cross-entropy measured in bits (divide nats by log(2)).
    BPC < 1.5 is a reasonable target for character-level English text.

Token Accuracy
    The fraction of positions where argmax(logits) == target.
    Useful as a sanity check but not the primary metric.

Vocabulary Coverage
    What fraction of the test corpus tokens appear in the vocabulary.
    Low coverage indicates a mismatch between training and test domains.
"""

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class Evaluator:
    """
    Evaluation wrapper for BB8LM.

    Parameters
    ----------
    model     : BB8LM — trained (or partially trained) language model
    tokenizer : BaseTokenizer — matching tokenizer
    device    : str | None
    """

    def __init__(self, model: nn.Module, tokenizer, device: Optional[str] = None) -> None:
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.tokenizer = tokenizer

    # ------------------------------------------------------------------
    # Dataset-level evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate_dataset(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Compute metrics over all batches in *dataloader*.

        Returns
        -------
        dict with keys: loss, perplexity, accuracy, bits_per_char
        """
        self.model.eval()

        total_loss = 0.0
        total_correct = 0
        total_tokens = 0
        n_batches = 0

        for x, y in dataloader:
            x, y = x.to(self.device), y.to(self.device)
            output = self.model(x, targets=y)

            total_loss += output["loss"].item()
            n_batches += 1

            # Token-level accuracy
            preds = output["logits"].argmax(dim=-1)
            valid = y != -1
            total_correct += (preds[valid] == y[valid]).sum().item()
            total_tokens += valid.sum().item()

        avg_loss = total_loss / max(n_batches, 1)
        accuracy = total_correct / max(total_tokens, 1)

        return {
            "loss": avg_loss,
            "perplexity": math.exp(avg_loss),
            "accuracy": accuracy,
            "bits_per_char": avg_loss / math.log(2),
        }

    # ------------------------------------------------------------------
    # Single-text evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate_text(self, text: str) -> Dict[str, float]:
        """
        Evaluate the model on a single text string.

        The text is split into non-overlapping chunks of *max_seq_len* tokens
        and the average loss is reported.

        Returns
        -------
        dict with keys: loss, perplexity, bits_per_char
        """
        token_ids = self.tokenizer.encode(text)
        max_len = self.model.max_seq_len

        self.model.eval()
        total_loss = 0.0
        n_chunks = 0

        for start in range(0, len(token_ids) - 1, max_len):
            chunk = token_ids[start : start + max_len + 1]
            if len(chunk) < 2:
                break

            x = torch.tensor([chunk[:-1]], dtype=torch.long, device=self.device)
            y = torch.tensor([chunk[1:]], dtype=torch.long, device=self.device)

            output = self.model(x, targets=y)
            total_loss += output["loss"].item()
            n_chunks += 1

        avg_loss = total_loss / max(n_chunks, 1)
        return {
            "loss": avg_loss,
            "perplexity": math.exp(avg_loss),
            "bits_per_char": avg_loss / math.log(2),
        }

    # ------------------------------------------------------------------
    # Vocabulary coverage
    # ------------------------------------------------------------------

    def vocabulary_coverage(self, text: str) -> Dict[str, float]:
        """
        Compute what fraction of tokens in *text* are in-vocabulary.

        Returns
        -------
        dict with keys: coverage (0-1), unk_rate (0-1), total_tokens, unk_tokens
        """
        token_ids = self.tokenizer.encode(text)
        unk_id = self.tokenizer.vocab.get(self.tokenizer.unk_token, -1)

        total = len(token_ids)
        unk_count = token_ids.count(unk_id)
        coverage = 1.0 - (unk_count / max(total, 1))

        return {
            "coverage": coverage,
            "unk_rate": unk_count / max(total, 1),
            "total_tokens": total,
            "unk_tokens": unk_count,
        }

    # ------------------------------------------------------------------
    # Summary report
    # ------------------------------------------------------------------

    def full_report(
        self,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        sample_texts: Optional[List[str]] = None,
    ) -> Dict:
        """
        Generate a comprehensive evaluation report.

        Parameters
        ----------
        train_loader  : training DataLoader (optional)
        val_loader    : validation DataLoader (optional)
        sample_texts  : list of raw text strings to evaluate individually

        Returns
        -------
        Nested dict with all computed metrics.
        """
        report: Dict = {}

        if train_loader is not None:
            print("Evaluating on training set …")
            report["train"] = self.evaluate_dataset(train_loader)
            print(
                f"  loss={report['train']['loss']:.4f}"
                f"  ppl={report['train']['perplexity']:.2f}"
                f"  acc={report['train']['accuracy']:.3f}"
            )

        if val_loader is not None:
            print("Evaluating on validation set …")
            report["val"] = self.evaluate_dataset(val_loader)
            print(
                f"  loss={report['val']['loss']:.4f}"
                f"  ppl={report['val']['perplexity']:.2f}"
                f"  acc={report['val']['accuracy']:.3f}"
            )

        if sample_texts:
            report["texts"] = {}
            for i, text in enumerate(sample_texts):
                key = f"text_{i}"
                report["texts"][key] = self.evaluate_text(text)
                cov = self.vocabulary_coverage(text)
                report["texts"][key].update(cov)

        return report
