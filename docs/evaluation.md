# Evaluation

## Why Evaluate?

Training loss alone does not tell the full story:
- Training loss measures performance on seen data
- Validation loss measures generalisation to unseen data
- Perplexity converts loss into an interpretable "difficulty" score
- Text quality metrics capture the human-perceived quality of generated text

The measured results and their interpretation across all eight experiments
(v001–v008) are recorded in [BB8 Training Results](training_results.md).
The trusted from-scratch baseline is `bb8-char-small-v002` (validation loss
1.6863, perplexity 5.40, accuracy 49.92%, BPC 2.4329). The best from-scratch
model is `bb8-bpe-shakespeare-v007` (BPC 1.4237). The best chat model is
`bb8-qwen-instruct-v008` (validation perplexity 5.94).

BB8 uses the following evaluation framework (`evaluation/evaluator.py`).

---

## Perplexity

The primary language model metric.

$$\text{PPL} = \exp\left(-\frac{1}{N} \sum_{t=1}^{N} \log P(x_t \mid x_{<t})\right) = \exp(\mathcal{L})$$

### Interpretation

Perplexity equals the **geometric mean of inverse probabilities** assigned to each token.

A model with PPL = k is as uncertain as if it were choosing uniformly among k equally likely options at every step.

| PPL | Interpretation |
|---|---|
| = vocab_size | Random model (uniform distribution) |
| 50–100 | Very poor language model |
| 20–50 | Acceptable for small, domain-specific models |
| 5–20 | Good quality language model |
| < 5 | Excellent (state-of-the-art for specific domains) |

For character-level models (vocab ~65), a random baseline has PPL = 65.

### Important: PPL Is Not Comparable Across Tokenizers

Perplexity is computed per token.  Character-level tokens are much smaller than BPE tokens, so character-level PPL is not directly comparable to BPE PPL.  Use **bits per character** for cross-tokenizer comparison.

---

## Bits Per Character (BPC)

$$\text{BPT} = \frac{\mathcal{L}}{\log 2}$$

$$\text{BPC} = \text{BPT} \times
\frac{\text{number of tokens}}{\text{number of source characters}}$$

BPT converts per-token loss from nats to bits. BPC then normalises by the
token-to-character ratio, making it comparable across tokenization strategies
when measured on the same underlying corpus. The shorter formula
`loss / ln(2)` is BPC only for a character tokenizer with one token per source
character.

| BPC | Interpretation |
|---|---|
| 1.0 | Random binary process |
| 1.5 | Good character-level model |
| 1.2 | Very good |
| < 1.0 | State-of-the-art |

---

## Cross-Entropy Loss

The direct output of the training objective.  Lower is better.

$$\mathcal{L} = -\frac{1}{N} \sum_{t=1}^{N} \log P(x_t \mid x_{<t})$$

Relationship to other metrics:
- $\text{PPL} = e^{\mathcal{L}}$
- $\text{BPT} = \mathcal{L} / \ln 2$
- $\text{BPC} = \text{BPT} \times \text{tokens}/\text{characters}$

---

## Token Accuracy

The fraction of positions where the model's top prediction matches the target:

$$\text{Acc} = \frac{1}{N} \sum_{t=1}^{N} \mathbf{1}[\hat{x}_t = x_t]$$

This is a weaker metric than perplexity because it only checks the rank-1 prediction, ignoring calibration of the full distribution.

For character-level models:
- Random: Acc ≈ 1/65 ≈ 1.5%
- Good model: Acc 30–60%

---

## Vocabulary Coverage

$$\text{Coverage} = 1 - \frac{|\text{UNK tokens}|}{|\text{total tokens}|}$$

Measures the fraction of tokens in a test text that are in-vocabulary.  Low coverage indicates domain mismatch or vocabulary too small.

For character-level tokenizers: coverage is typically 100% (all characters are in-vocabulary).

For word-level tokenizers with 5k vocab on Shakespeare: coverage ≈ 80–90%.

---

## Qualitative Evaluation

Quantitative metrics don't capture everything.  Visual inspection of generated text is important:

### Signs of Good Training

- **Grammatical sentences** — subject-verb agreement, proper punctuation
- **Consistent style** — text "sounds like" Shakespeare if trained on Shakespeare
- **Coherent short-range context** — recent words are semantically related
- **Rare repetition** — with repetition_penalty

### Signs of Poor Training

- **Repetitive loops** — "the the the the …"
- **Random character sequences** — model hasn't learned word boundaries
- **Very short outputs** — model predicts `<EOS>` too early
- **Generic outputs** — model memorised most common sequences

---

## Running Evaluation

```python
from evaluation.evaluator import Evaluator

evaluator = Evaluator(model, tokenizer)

# Dataset-level metrics
metrics = evaluator.evaluate_dataset(val_loader)
print(metrics)
# {'loss': 2.3, 'perplexity': 9.97, 'accuracy': 0.42,
#  'bits_per_token': 3.32, 'bits_per_char': 1.41}

# Single text evaluation
metrics = evaluator.evaluate_text("To be or not to be.")
print(metrics)

# Vocabulary coverage
cov = evaluator.vocabulary_coverage(test_text)
print(cov)

# Full report
report = evaluator.full_report(train_loader, val_loader, sample_texts=[...])
```

---

## Storing Results

Evaluation results are stored as JSON in `outputs/<experiment_name>/results.json`:

```json
{
  "experiment_name": "experiment_01",
  "model_config": {"d_model": 128, "num_layers": 4, ...},
  "evaluation": {
    "train": {"loss": 1.8, "perplexity": 6.05, "accuracy": 0.52},
    "val":   {"loss": 2.1, "perplexity": 8.17, "accuracy": 0.46}
  },
  "training_history": {
    "train_loss": [...],
    "val_loss": [...]
  }
}
```
