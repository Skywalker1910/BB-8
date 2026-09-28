# Future Work: Diagnosis, Goals, and Planned Improvements

This document outlines why the current models underperform, what the root causes are, and what the next experiments should target — in order of priority.

---

## Why the Current Models Struggle

### 1. Data is the bottleneck, not architecture

| Model | Parameters | Training tokens | Tokens per parameter |
|---|---:|---:|---:|
| v002 (char) | 112K | ~1M chars | ~9 |
| v006b (char) | 1.2M | ~1M chars | ~0.8 |
| v007 (BPE) | 5.6M | ~270K BPE tokens | ~0.05 |
| GPT-2 (reference) | 117M | ~2B tokens | ~17 |
| LLaMA-7B (reference) | 7B | ~1T tokens | ~143 |

The rule of thumb for Transformer training is at least 20 tokens per parameter (Chinchilla scaling laws). v007 has 0.05 tokens per parameter — roughly 400x too little data. This is the single biggest reason for overfitting. The model has vastly more capacity than the data can fill, so it memorizes instead of generalizing.

### 2. Stride and data windowing created unfair comparisons

v002 used stride=1, creating ~1M overlapping training windows from 1M characters. v006/v006b used stride=128, creating ~7.8K non-overlapping windows. This means:
- v002's validation perplexity (5.40) is optimistic — the model saw nearly every validation position from an adjacent training window
- v006's validation perplexity (9.57) is stricter but more honest
- The two numbers cannot be compared directly, which makes the scaling story confusing

### 3. Character tokenization limits what the model can learn

With a character vocabulary of 69 tokens, the model spends most of its capacity learning which letters follow which. A 256-character context is roughly 50 words — not enough for coherent paragraphs. Even a perfect character model would need many more layers and wider embeddings to implicitly discover word boundaries and syntax.

### 4. BPE vocabulary size interacted badly with data size

With BPE-3000 on Shakespeare, the corpus compresses to ~270K tokens. With stride=64, that's only ~4.2K training windows for a 5.6M-parameter model. The model memorizes every window perfectly (train PPL ~16) but can't generalize (val PPL ~355).

### 5. Instruction tuning from scratch can't work (by design)

v003 proved this: 15K instruction/response pairs can teach format but not knowledge. A randomly initialized model doesn't know English, doesn't know facts, and doesn't know arithmetic. Instruction data assumes all of that already exists.

### 6. LoRA on Dolly hit a ceiling

v008a and v008 achieved nearly identical validation loss (2.0559 vs 2.0619). The second epoch of v008 improved training loss but not validation — the model started fitting Dolly's specific phrasing patterns rather than learning generalizable instruction-following. Dolly 15K has limited diversity: many responses follow similar templates.

---

## Planned Improvements (Ordered by Priority)

### Experiment 1: Fix the data problem for from-scratch models

**Goal:** Train v007-class BPE model without severe overfitting.

**What to change:**
- Add WikiText-103 (~100M tokens) or OpenWebText as pretraining corpus
- Keep Shakespeare as a fine-tuning or evaluation corpus
- This gives ~18 tokens per parameter for a 5.6M model — close to the Chinchilla ratio

**Expected outcome:** Val PPL should drop dramatically (from 355 to <50) because the model has enough data to generalize. The train/val gap should narrow.

**Config changes:**
- `data.path`: point to wikitext-103 or a concatenated corpus
- `training.num_epochs`: reduce to 1-3 (more data needs fewer passes)
- `data.stride`: can use 1 since we have enough data
- Everything else stays the same

### Experiment 2: Normalize evaluation across all from-scratch models

**Goal:** Make all character models comparable by using the same stride.

**What to change:**
- Re-evaluate v002's checkpoint with stride=128 (same as v006/v006b)
- Or retrain v006/v006b with stride=1 and fewer epochs
- Report both stride=1 and stride=128 results for at least one model to quantify the difference

**Expected outcome:** A clear, apples-to-apples scaling curve showing how PPL improves from 112K → 833K → 1.2M parameters.

### Experiment 3: Two-stage training (pretrain + instruct)

**Goal:** Prove the two-stage pipeline works with our own from-scratch Transformer.

**What to change:**
1. Pretrain BPE model on WikiText-103 with next-token prediction
2. Fine-tune that checkpoint on Dolly 15K instruction data
3. Compare against v003 (instruction from scratch) using the same smoke test prompts

**Expected outcome:** The pretrained-then-finetuned model should answer basic questions correctly, unlike v003 which hallucinated everything. This would close the loop between "my from-scratch architecture" and "industry training practice."

### Experiment 4: Smarter LoRA training on Dolly

**Goal:** Reduce overfitting on Dolly by improving data quality and training efficiency.

**What to change:**
- Filter Dolly to remove low-quality, very short, or duplicate-pattern records
- Use the `full_example_v2` preprocessing (skip examples that don't fit context, instead of truncating)
- Try a lower rank (r=4) to reduce overfitting
- Add early stopping based on validation loss instead of fixed epochs
- Increase context from 512 to 768 or 1024 tokens (some Dolly responses are truncated at 512)

**Expected outcome:** Lower validation loss and better multi-turn coherence. The current v008 plateau at val loss ~2.06 may be Dolly's quality ceiling rather than a model limitation.

### Experiment 5: Better evaluation

**Goal:** Go beyond perplexity with task-specific benchmarks.

**What to add:**
- A frozen test set of 50+ diverse prompts (arithmetic, factual, coding, refusal) with expected answers
- Exact-match scoring for arithmetic and factual questions
- Human rating (1-5) for coherence and relevance on open-ended questions
- Track these metrics across every model version to show clear progression

**Expected outcome:** A dashboard that shows "v003 passes 0/50, v004 passes 23/50, v008 passes 31/50" — much more convincing than raw perplexity for interviews and portfolio.

### Experiment 6: Expand grounded retrieval training

**Goal:** Improve v005's 22.2% pass rate toward the 80% target.

**What to change:**
- Expand corpus from 12 documents to 50+ covering more diverse topics
- Add multi-document training examples (answer requires combining evidence from 2-3 sources)
- Add contradiction examples (evidence says X, question assumes Y)
- Add varied citation formats and more natural abstention phrasing
- Create a new frozen evaluation set before training

**Expected outcome:** Citation accuracy should improve significantly with richer training data. Abstention was already strong (16/16); the gap is in content quality.

---

## Root Cause Summary

| Problem | Root Cause | Fix |
|---|---|---|
| v006/v006b PPL worse than v002 | Different stride makes numbers incomparable | Standardize evaluation stride |
| v007 extreme overfitting (PPL 355) | 5.6M params / 270K tokens = 400x under-sampled | Use 100x more pretraining data |
| v003 hallucinating everything | No pretraining — can't learn language from 15K instructions | Two-stage pretrain + instruct |
| v008 plateau at val loss ~2.06 | Dolly's template diversity ceiling | Filter/expand instruction data |
| v005 only 22.2% pass rate | Narrow training templates, 192 examples | Richer, more diverse training set |
| Character models can't produce sentences | 1 token = 1 character, context too narrow | Use BPE, increase context |
| All small models hallucinate on complex questions | 500M params is not enough for broad knowledge | Known limitation — not fixable at this scale |

---

## What This Means for the Project

The from-scratch Transformer architecture is correct — it learns, it scales, and it generates text that improves predictably with more parameters and better tokenization. The problems are all about data scale, evaluation methodology, and training recipe, not the model implementation.

The most impactful next step is **Experiment 1** (more pretraining data). Everything else — better evaluation, two-stage training, LoRA improvements — depends on having a from-scratch model that doesn't overfit its training data. With WikiText-103, the 5.6M BPE model should generalize properly, and the full scaling story (112K → 833K → 1.2M → 5.6M) will show clean, progressive improvement.

For the LoRA track, **Experiment 4** (smarter Dolly training) and **Experiment 5** (task-specific evaluation) would make the v004→v008 comparison much stronger. Right now we compare raw perplexity numbers that are hard to interpret; a prompt-level pass/fail benchmark would tell a much clearer story.
