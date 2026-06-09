"""
generator.py
------------
Text generation engine for BB8.

Four decoding strategies are implemented:

1. Greedy Decoding
   Always select the token with the highest probability.
   Deterministic; produces repetitive text; not recommended for creative
   generation.

2. Temperature Sampling
   Divide logits by *temperature* before softmax, then sample.
   - temperature < 1.0 → sharper distribution, more conservative text
   - temperature > 1.0 → flatter distribution, more creative/random text
   - temperature → 0   → equivalent to greedy

3. Top-K Sampling
   Restrict sampling to the *k* most probable tokens, renormalise, then
   sample.  Prevents the tail of the distribution (very unlikely tokens)
   from ever being selected.

4. Top-P (Nucleus) Sampling (Holtzman et al., 2020)
   Dynamically determine the smallest set of tokens whose cumulative
   probability mass exceeds *p*, then sample from that set.
   More adaptive than Top-K because the set size adjusts with the
   sharpness of the distribution.

All strategies support a *repetition_penalty* parameter that down-weights
tokens that have appeared recently, encouraging diversity.

Reference
---------
Holtzman et al. (2020). "The Curious Case of Neural Text Degeneration."
https://arxiv.org/abs/1904.09751
"""

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F


class TextGenerator:
    """
    Text generation engine wrapping a BB8LM model.

    Parameters
    ----------
    model      : BB8LM — trained language model
    tokenizer  : BaseTokenizer — matching tokenizer
    device     : str | None — 'cuda', 'cpu', or None (auto)
    """

    def __init__(self, model, tokenizer, device: Optional[str] = None) -> None:
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.model.eval()
        self.tokenizer = tokenizer

    # ------------------------------------------------------------------
    # Main generation entry point
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        strategy: str = "top_p",
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
    ) -> str:
        """
        Generate text conditioned on *prompt*.

        Parameters
        ----------
        prompt            : seed text for generation
        max_new_tokens    : how many new tokens to generate
        strategy          : 'greedy' | 'temperature' | 'top_k' | 'top_p'
        temperature       : sampling temperature (ignored for greedy)
        top_k             : K for top-k sampling
        top_p             : cumulative probability threshold for nucleus sampling
        repetition_penalty: factor by which recently generated tokens are
                            penalised (1.0 = no penalty; >1.0 discourages repetition)

        Returns
        -------
        Full text: prompt + generated continuation.
        """
        input_ids = self.tokenizer.encode(prompt)
        generated: List[int] = list(input_ids)
        max_ctx = self.model.max_seq_len
        eos_id = self.tokenizer.vocab.get(self.tokenizer.eos_token, -1)

        for _ in range(max_new_tokens):
            # Truncate context to model's maximum length
            ctx = torch.tensor(
                [generated[-max_ctx:]], dtype=torch.long, device=self.device
            )

            # Forward pass — only need the last position's logits
            logits = self.model(ctx)["logits"][:, -1, :]  # (1, vocab_size)

            # Repetition penalty: divide logit of seen tokens by penalty factor
            if repetition_penalty != 1.0:
                seen = set(generated[-64:])  # look back 64 tokens
                for token_id in seen:
                    if 0 <= token_id < logits.size(-1):
                        logits[0, token_id] = logits[0, token_id] / repetition_penalty

            # Select next token
            if strategy == "greedy":
                next_token = self._greedy(logits)
            elif strategy == "temperature":
                next_token = self._temperature_sample(logits, temperature)
            elif strategy == "top_k":
                next_token = self._top_k_sample(logits, temperature, top_k)
            elif strategy == "top_p":
                next_token = self._top_p_sample(logits, temperature, top_p)
            else:
                raise ValueError(
                    f"Unknown strategy '{strategy}'. "
                    "Choose: greedy | temperature | top_k | top_p"
                )

            generated.append(next_token)

            # Stop at end-of-sequence token
            if next_token == eos_id:
                break

        return self.tokenizer.decode(generated)

    # ------------------------------------------------------------------
    # Decoding strategies
    # ------------------------------------------------------------------

    def _greedy(self, logits: torch.Tensor) -> int:
        """Greedy: always pick the most probable token."""
        return int(logits.argmax(dim=-1).item())

    def _temperature_sample(self, logits: torch.Tensor, temperature: float) -> int:
        """Temperature scaling followed by categorical sampling."""
        logits = logits / max(temperature, 1e-5)
        probs = F.softmax(logits, dim=-1)
        return int(torch.multinomial(probs, num_samples=1).item())

    def _top_k_sample(
        self, logits: torch.Tensor, temperature: float, k: int
    ) -> int:
        """
        Top-K sampling.

        Zero out all logits except the top *k*, then sample.
        """
        logits = logits / max(temperature, 1e-5)
        k = min(k, logits.size(-1))
        top_k_logits, top_k_indices = torch.topk(logits, k=k, dim=-1)
        probs = F.softmax(top_k_logits, dim=-1)
        sampled_idx = torch.multinomial(probs, num_samples=1)
        return int(top_k_indices[0, sampled_idx].item())

    def _top_p_sample(
        self, logits: torch.Tensor, temperature: float, p: float
    ) -> int:
        """
        Nucleus (Top-P) sampling.

        Find the smallest set of tokens whose cumulative probability ≥ p,
        then sample uniformly from that set (after renormalisation).
        """
        logits = logits / max(temperature, 1e-5)
        probs = F.softmax(logits, dim=-1)  # (1, vocab)

        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Remove tokens beyond the nucleus
        # The mask is shifted right so we keep at least one token
        remove_mask = (cumulative_probs - sorted_probs) > p
        sorted_probs[remove_mask] = 0.0
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

        sampled_idx = torch.multinomial(sorted_probs, num_samples=1)
        return int(sorted_indices[0, sampled_idx].item())

    # ------------------------------------------------------------------
    # Comparison utility
    # ------------------------------------------------------------------

    def compare_strategies(
        self,
        prompt: str,
        max_new_tokens: int = 150,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
    ) -> Dict[str, str]:
        """
        Generate with all four strategies and print a side-by-side comparison.

        Returns
        -------
        dict mapping strategy name → generated text
        """
        results: Dict[str, str] = {}
        strategies = ["greedy", "temperature", "top_k", "top_p"]

        print(f"\nPrompt: {repr(prompt)}")
        print("=" * 60)

        for strat in strategies:
            text = self.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                strategy=strat,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )
            results[strat] = text
            print(f"\n[{strat.upper()}]")
            print(text)
            print("-" * 60)

        return results
