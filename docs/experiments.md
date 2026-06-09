# Experiments

BB8 experiments follow a systematic progression from a baseline small model to increasingly capable configurations.  Each experiment isolates one variable to understand its effect on model quality.

---

## Experiment Design Principles

1. **One variable at a time** — change one hyperparameter per experiment to understand its isolated effect
2. **Same data, same seed** — ensures changes are due to the model, not randomness
3. **Record everything** — save config, metrics, and sample outputs
4. **Compare quantitatively** — use perplexity as the primary metric

---

## Experiment 1 — Small Baseline

**Goal:** Establish a baseline with the smallest functional model.

**Config:** `configs/small_model.yaml`

| Hyperparameter | Value |
|---|---|
| Tokenizer | Character-level |
| d_model | 64 |
| num_layers | 2 |
| num_heads | 2 |
| d_ff | 256 |
| max_seq_len | 128 |
| batch_size | 64 |
| epochs | 5 |
| learning_rate | 1e-3 |

**Expected outcome:**
- Model trains quickly (~minutes on CPU)
- Basic word and phrase patterns learned
- Perplexity: 30–50 (Shakespeare, char-level)

**Run:**
```bash
python experiments/run_experiment.py \
    --config configs/small_model.yaml \
    --data data/tiny_shakespeare.txt \
    --name exp01_baseline
```

**Results:** *(fill in after running)*

| Metric | Value |
|---|---|
| Train loss | — |
| Val loss | — |
| Perplexity | — |
| Parameters | — |

---

## Experiment 2 — More Layers

**Goal:** Understand the effect of depth on model quality.

**Change from Exp 1:** num_layers: 2 → 6

| Hyperparameter | Value |
|---|---|
| Tokenizer | Character-level |
| d_model | 64 |
| **num_layers** | **6** |
| num_heads | 2 |
| d_ff | 256 |

**Hypothesis:** More layers = better capacity to model complex patterns, but risk of overfitting.

**Results:** *(fill in after running)*

| Metric | Value |
|---|---|
| Train loss | — |
| Val loss | — |
| Perplexity | — |

---

## Experiment 3 — Larger Embeddings

**Goal:** Understand the effect of wider representations.

**Change from Exp 1:** d_model: 64 → 128, d_ff: 256 → 512

| Hyperparameter | Value |
|---|---|
| Tokenizer | Character-level |
| **d_model** | **128** |
| num_layers | 4 |
| **d_ff** | **512** |

**Hypothesis:** Wider models have more capacity per layer, which may improve quality without the stability challenges of depth.

**Config:** `configs/default.yaml`

**Results:** *(fill in after running)*

| Metric | Value |
|---|---|
| Train loss | — |
| Val loss | — |
| Perplexity | — |
| Parameters | — |

---

## Experiment 4 — BPE Tokenizer

**Goal:** Compare character-level vs BPE tokenization at the same model size.

**Change from Exp 3:** tokenizer: char → bpe (vocab_size=3000)

| Hyperparameter | Value |
|---|---|
| **Tokenizer** | **BPE (vocab_size=3000)** |
| d_model | 128 |
| num_layers | 4 |

**Hypothesis:** BPE produces shorter sequences with more semantic content per token.  The model may achieve lower perplexity (as measured per BPE token) but the comparison requires care — use BPC for fair comparison.

**Run:**
```bash
python train.py \
    --data data/tiny_shakespeare.txt \
    --tokenizer bpe --vocab-size 3000 \
    --d-model 128 --num-layers 4 --num-heads 4 \
    --checkpoint-dir checkpoints/exp04_bpe
```

**Results:** *(fill in after running)*

| Metric | Char | BPE |
|---|---|---|
| Vocab size | ~65 | 3000 |
| Tokens in corpus | — | — |
| Train loss | — | — |
| BPC | — | — |

---

## Experiment 5 — Medium Model

**Goal:** Train the largest feasible model on Tiny Shakespeare.

**Config:** `configs/medium_model.yaml`

| Hyperparameter | Value |
|---|---|
| Tokenizer | BPE (vocab_size=5000) |
| d_model | 256 |
| num_layers | 6 |
| num_heads | 8 |
| d_ff | 1024 |
| epochs | 20 |

**Hypothesis:** The medium model should show significantly better text quality and lower perplexity than the small baseline.

**Results:** *(fill in after running)*

| Metric | Value |
|---|---|
| Train loss | — |
| Val loss | — |
| Perplexity | — |
| Parameters | — |

---

## Experiment 6 — Decoding Strategy Comparison

**Goal:** Compare text quality across the four generation strategies at a fixed temperature.

**Setup:** Same trained model (Exp 5), same prompt, different decoding.

| Strategy | Temperature | Notes |
|---|---|---|
| Greedy | — | Deterministic, maximally confident |
| Temperature (0.5) | 0.5 | Conservative; less variety |
| Temperature (1.2) | 1.2 | Creative; more diverse |
| Top-K (k=50) | 0.8 | Restricts to top 50 tokens |
| Top-P (p=0.9) | 0.8 | Nucleus sampling; adaptive |

**Qualitative analysis:** *(fill in after generating)*

> Compare generated paragraphs for fluency, diversity, and coherence.

---

## Summary Table

*(Fill in as experiments are run)*

| Exp | Tokenizer | d_model | Layers | Heads | Val PPL | BPC | Params |
|---|---|---|---|---|---|---|---|
| 01 Baseline | Char | 64 | 2 | 2 | — | — | — |
| 02 Deep | Char | 64 | 6 | 2 | — | — | — |
| 03 Wide | Char | 128 | 4 | 4 | — | — | — |
| 04 BPE | BPE-3k | 128 | 4 | 4 | — | — | — |
| 05 Medium | BPE-5k | 256 | 6 | 8 | — | — | — |

---

## Lessons Learned

*(Update this section as experiments are completed)*

- [ ] Depth vs width trade-off
- [ ] BPE vs character-level comparison
- [ ] Effect of learning rate on training stability
- [ ] Role of warmup steps
- [ ] Repetition penalty effect on generation quality
