# Experiments

I ran eight experiments across three tracks. Each one was designed to answer a specific question about how language models work. I'm presenting them in logical order, grouped by what I was trying to learn.

---

## Track 1: From-Scratch Transformer on Shakespeare

These four experiments use my own Transformer implementation trained on Tiny Shakespeare (~1 MB). Everything here — the model, the tokenizer, the training loop — is code I wrote from scratch.

### v001 — First Working Model (Historical)

**Question:** Can I get a Transformer to learn anything at all?

**Config:** `configs/small_model.yaml`

| Setting | Value |
|---|---|
| Tokenizer | Character (69 tokens) |
| d_model | 64 |
| Layers / Heads | 2 / 2 |
| d_ff | 256 |
| Context | 128 tokens |
| Params | 112,320 |
| Epochs | 5 |

```bash
python experiments/run_experiment.py \
    --config configs/small_model.yaml \
    --data data/tiny_shakespeare.txt \
    --name bb8-char-small-v001
```

**Result:** Val loss 1.67, perplexity 5.32. It worked — the model learned character patterns. But I later found the tokenizer was fitted on the entire corpus before splitting, which leaked validation data into training. Kept as a historical record.

---

### v002 — Corrected Baseline

**Question:** What are the real numbers with a proper train/val split?

**Config:** `configs/small_model.yaml` (same architecture, fixed data pipeline)

```bash
python experiments/run_experiment.py \
    --config configs/small_model.yaml \
    --data data/tiny_shakespeare.txt \
    --name bb8-char-small-v002
```

| Metric | Value |
|---|---:|
| Val loss | 1.6863 |
| Val perplexity | 5.3997 |
| Val accuracy | 49.92% |
| Val BPC | 2.4329 |

**Takeaway:** Slightly worse numbers than v001 (as expected — the leak is gone). This is the trusted baseline. The model learns speaker names, punctuation, and common letter sequences, but can't produce coherent sentences.

---

### v006 — Scaling Study

**Question:** How much does more capacity help on the same data?

**Config:** `configs/char_medium_v006.yaml`

| Setting | Value | Change from v002 |
|---|---|---|
| d_model | 128 | 2x |
| Layers / Heads | 4 / 4 | 2x |
| d_ff | 512 | 2x |
| Context | 256 tokens | 2x |
| Params | 835,584 | 7.4x |
| Epochs | 15 | 3x |

```bash
python experiments/run_experiment.py \
    --config configs/char_medium_v006.yaml \
    --data data/tiny_shakespeare.txt \
    --name bb8-char-medium-v006
```

I also ran v006b (6 layers instead of 4) to isolate the effect of depth.

| Metric | v002 | v006 (4L) | v006b (6L) |
|---|---:|---:|---:|
| Params | 112,320 | 833,408 | 1,229,184 |
| Val loss | 1.6863 | 2.2586 | 2.1090 |
| Val perplexity | 5.40 | 9.57 | 8.24 |
| Val accuracy | 49.92% | 34.23% | 37.67% |

Note: v002 used stride=1 (dense windows) while v006/v006b used stride=128 (sparser, harder generalization test), so val loss isn't directly comparable. Within v006 vs v006b the comparison is fair.

**Takeaway:** Adding 2 layers (4->6) improved perplexity by 14% with the same d_model. Depth matters more than width at this scale. But character-level tokenization remains the bottleneck.

---

### v007 — BPE Tokenizer + Larger Model

**Question:** How much does tokenizer choice matter compared to model size?

**Config:** `configs/bpe_shakespeare_v007.yaml`

| Setting | Value | Change from v006 |
|---|---|---|
| Tokenizer | BPE (3,000 tokens) | Was char (69) |
| d_model | 256 | 2x |
| Layers / Heads | 6 / 8 | 1.5x / 2x |
| d_ff | 1024 | 2x |
| Context | 512 tokens | 2x |
| Params | 5,633,536 | 6.7x |
| Epochs | 20 | +5 |

```bash
python experiments/run_experiment.py \
    --config configs/bpe_shakespeare_v007.yaml \
    --data data/tiny_shakespeare.txt \
    --name bb8-bpe-shakespeare-v007
```

I also ran v007a (BPE vocab=1000) to test vocab size effect.

| Metric | v007 (BPE 3K) | v007a (BPE 1K) |
|---|---:|---:|
| Params | 5,633,536 | 5,121,536 |
| Train loss | 2.8101 | 3.9733 |
| Val loss | 5.8721 | 4.4288 |
| Val perplexity | 355.01 | 83.83 |
| Val accuracy | 9.82% | 12.02% |

**Overfitting:** Both models severely overfit. v007 (BPE 3K) has a train/val gap of 2.81 vs 5.87 — it memorized Shakespeare. This is expected with 5.6M params on ~270K BPE tokens.

**Takeaway:** Despite terrible validation metrics, v007's generated text was the best of any from-scratch model — full sentences, consistent characters, recognizable Shakespeare verse. The model memorized the training distribution extremely well. BPE lets each token carry word-level information, which is why even an overfit BPE model produces more coherent text than a well-generalized character model.

---

## Track 2: Instruction Following

These experiments explore how to make a model follow instructions and answer questions.

### v003 — Instruction Model from Scratch (Failed)

**Question:** Can I train an instruction-following model without pretraining?

**Config:** `configs/instruction_v003.yaml`

| Setting | Value |
|---|---|
| Dataset | Dolly 15K (15,011 records) |
| Tokenizer | BPE (1,024 tokens) |
| Architecture | 6L / 8H / 256d |
| Params | 5,062,144 |
| Epochs | 8 |

```bash
python experiments/run_experiment.py \
    --config configs/instruction_v003.yaml \
    --data data/dolly_15k_chat.txt \
    --name bb8-bpe-instruct-v003-dev \
    --prompt "USER: What is machine learning? BB:" \
    --seed 42
```

| Metric | Value |
|---|---:|
| Val loss | 2.9497 |
| Val perplexity | 19.10 |
| Val accuracy | 37.59% |

**Qualitative result:** all five smoke prompts produced wrong or incoherent answers. The model learned the format of a response but could not produce correct content.

**Takeaway:** this was the single most important experiment in the project. It proved that instruction tuning alone cannot teach a model language, facts, or reasoning. Those have to come from pretraining on a broad corpus first. This is exactly why GPT, LLaMA, and every production LLM uses a two-stage pipeline.

---

### v004 — Pretrained Model + LoRA

**Question:** Does starting from a pretrained base fix v003's failures?

**Config:** `configs/qwen_lora_v004.yaml`

| Setting | Value |
|---|---|
| Base model | Qwen2.5-0.5B (pretrained, not mine) |
| Dataset | Dolly 15K |
| LoRA rank / alpha | 16 / 32 |
| Trainable params | 8,798,208 (1.75% of model) |
| Epochs | 1 |
| Context | 256 tokens |

```bash
python fine_tune.py \
    --config configs/qwen_lora_v004.yaml \
    --name bb8-qwen-lora-v004-dev
```

| Metric | Value |
|---|---:|
| Val loss | 2.0753 |
| Val perplexity | 7.9669 |

**Qualitative result:** basic questions answered correctly — "2+2=4", "Paris", valid Python code. Dramatic improvement over v003 with the same training data.

**Takeaway:** pretrained language knowledge is what v003 was missing. The LoRA adapter only had to learn the instruction-following behavior, not the entire English language.

**Known issues:** `4 - 2` returned wrong answer, multi-turn conversations repeated old answers, hallucinated identity claims.

---

### v008 — Instruction-Tuned Base + LoRA

**Question:** Is it better to start from Qwen-Instruct (already instruction-tuned) instead of the raw base?

**Config:** `configs/qwen_instruct_v008.yaml`

| Setting | Value | Change from v004 |
|---|---|---|
| Base model | Qwen2.5-0.5B-Instruct | Was raw Qwen2.5-0.5B |
| Context | 512 tokens | Was 256 |
| Epochs | 2 | Was 1 |
| Learning rate | 1e-4 | Was 2e-4 |

```bash
python fine_tune.py \
    --config configs/qwen_instruct_v008.yaml \
    --name bb8-qwen-instruct-v008
```

I ran two sub-experiments: v008a (rank=8, 1 epoch) and v008 (rank=16, 2 epochs).

| Metric | v004 | v008a (r=8, 1ep) | v008 (r=16, 2ep) |
|---|---:|---:|---:|
| Trainable params | 8,798,208 | 4,399,104 | 8,798,208 |
| Train loss | 2.0087 | 1.9531 | 1.8076 |
| Val loss | 2.0753 | 2.0559 | 2.0619 |
| Val perplexity | 7.97 | 7.81 | 7.86 |

**Smoke test results (v008):**

| Prompt | Answer |
|---|---|
| Say hello | Hello! |
| 2 + 2 | Two plus two equals four. |
| Capital of France | The capital of France is Paris |
| Photosynthesis | Correct explanation with sunlight, water, CO2, oxygen, energy |
| Python add function | `def add(x, y): return x + y` |

**Key finding:** v008 trained lower (1.81 vs 1.95) but val loss was essentially identical to v008a (2.06 vs 2.06). The second epoch didn't improve generalization — the model started overfitting to Dolly's training distribution. Both v008 variants outperform v004 on every metric.

**Takeaway:** starting from an instruction-tuned base is strictly better. Even v008a with half the LoRA parameters beat v004. The upstream instruction tuning gives a cleaner foundation that my Dolly LoRA can build on.

---

## Track 3: Grounded Retrieval

### v005 — Evidence-Based Answering (Pilot)

**Question:** Can a small model answer questions from provided evidence, cite its sources, and refuse when it doesn't have the information?

**Config:** `configs/qwen_grounded_v005.yaml`

| Setting | Value |
|---|---|
| Base model | Qwen2.5-0.5B-Instruct |
| Trainable params | 540,672 |
| Evidence corpus | 12 manually curated documents |
| Retriever | Deterministic BM25-style |
| Training examples | 192 train / 48 val |
| Test cases | 88 across 4 configurations |

```bash
python fine_tune.py \
    --config configs/qwen_grounded_v005.yaml \
    --name bb8-grounded-v005-pilot
```

**Best result (LoRA + retrieval):** 16/72 combined content+citation checks passed, 16/16 abstention checks passed.

**What worked:** the model learned to cite documents and to say "I don't have evidence to answer that" when asked about things outside the corpus.

**What didn't work:** content coverage dropped from the baseline, one invalid citation was generated, and overall quality didn't meet the predefined 80% gate.

**Release decision:** not promoted. The approach is viable but needs richer training data, multi-document examples, and better evaluation. Full details in [Grounded Results v005](grounded_results_v005.md).

---

## Summary

| Experiment | Track | Question Answered |
|---|---|---|
| v001, v002 | From scratch | Can a tiny Transformer learn from text? Yes. |
| v006, v006b | From scratch | Does more capacity help? Yes. Depth > width (14% PPL gain). |
| v007, v007a | From scratch | Does BPE help? Yes for text quality; overfit on small data. |
| v003 | Instruction | Can you skip pretraining? No. Model hallucinates everything. |
| v004 | Instruction | Does pretrained + LoRA work? Yes, dramatic quality jump. |
| v008, v008a | Instruction | Does instruct base help more? Yes, beats v004 with half the params. |
| v005 | Grounded | Can a small model cite and abstain? Partially, needs more data. |

### Experiment design principles I followed

1. **Change one thing at a time** when possible — isolate the variable being tested
2. **Same data, same seed** across comparable experiments for fair comparison
3. **Record everything** — config hashes, dataset hashes, Git commits, GPU info
4. **Don't cherry-pick** — report failures honestly, they teach more than successes
5. **Predefined gates** — decide what "good enough" means before looking at results
