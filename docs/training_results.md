# Training Results

This page documents the results of all BB8 experiments in the order they logically build on each other — from the smallest from-scratch character model up to a pretrained instruction-tuned chat model.

---

## How I Evaluated Each Model

Every from-scratch model (v001–v007) is evaluated by my own `evaluation/evaluator.py`. After training finishes, the evaluator runs the trained model over both the training and validation DataLoaders and computes:

| Metric | What it measures | How to read it |
|---|---|---|
| **Cross-entropy loss** | Average negative log-probability the model assigns to the next correct token | Lower is better. Measures how surprised the model is by the real data |
| **Perplexity (PPL)** | `exp(loss)` — the effective number of choices the model is confused between at each step | Lower is better. PPL=5 means the model is choosing among ~5 equally likely options per token |
| **Accuracy** | Fraction of positions where the model's top prediction matches the actual next token | Higher is better. A rough sanity check — ignores the quality of the full probability distribution |
| **Bits per character (BPC)** | `loss / ln(2)` for char models, or normalized by token-to-character ratio for BPE | Lower is better. Allows fair comparison across different tokenizers on the same text |

For LoRA models (v004, v005, v008), HuggingFace Transformers computes the loss internally over assistant-only tokens (prompt tokens are masked with `labels=-100`). I run a separate validation pass after training that averages loss over all supervised validation tokens. Smoke tests (five fixed prompts) provide a qualitative check.

**Important:** perplexity is only meaningful within the same tokenizer and evaluation setup. A char PPL of 5.4 cannot be compared to a BPE PPL of 355 — they predict fundamentally different-sized tokens. I note where comparisons are valid and where they aren't.

---

## Progressive Experiment Summary

The experiments are arranged in order of increasing complexity — from the simplest character model to pretrained LoRA chat models. Each row builds on the lessons from the rows above it.

### Phase 1: From-Scratch Character Models on Shakespeare

These use my own Transformer implementation, my own tokenizers, and 1 MB of Shakespeare text. The goal was to verify the architecture works and understand scaling.

| Run | Architecture | Params | Context | Stride | Epochs | Train Loss | Val Loss | Val PPL | Val Acc |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v001 | 2L/2H/64d, char(69) | 112,320 | 128 | 1 | 5 | 1.4937 | 1.6707 | 5.32 | 50.0% |
| v002 | 2L/2H/64d, char(69) | 112,320 | 128 | 1 | 5 | 1.5084 | 1.6863 | 5.40 | 49.9% |
| v006 | 4L/4H/128d, char(69) | 833,408 | 256 | 128 | 15 | 2.2045 | 2.2586 | 9.57 | 34.2% |
| v006b | 6L/4H/128d, char(69) | 1,229,184 | 256 | 128 | 15 | 2.0325 | 2.1090 | 8.24 | 37.7% |

**How to read these results:**

- **v001 vs v002:** Same architecture, same data. v001 has slightly better numbers (PPL 5.32 vs 5.40) but its tokenizer was trained on the full corpus including validation data — a data leak. v002 is the honest baseline. Lesson: always split data before fitting the tokenizer.

- **v002 vs v006/v006b — why does PPL look worse?** v002 shows PPL 5.40 while v006 shows 9.57. That seems like a regression, but these numbers are not comparable because stride changed from 1 to 128. With stride=1, every possible window of 128 characters becomes a training example (~1M samples) so the model sees nearly every validation position during training from a slightly shifted window. With stride=128, training windows don't overlap at all (~7.8K samples), making generalization genuinely harder. The models with more parameters are producing better text — the evaluation is just stricter.

- **v006 vs v006b — depth matters:** These two are directly comparable (same stride, same data, same d_model). Adding 2 layers (4→6) dropped perplexity from 9.57 to 8.24 — a 14% improvement. Accuracy jumped from 34.2% to 37.7%. At this model scale, depth helps more than width.

- **What the character models can and can't do:** All character models produce Shakespeare-like formatting (speaker names, colons, verse breaks) and common letter patterns. None of them produce grammatically correct English. Each token is a single character, so a 256-token window covers only 256 characters — roughly 50 words. That's not enough context for coherent sentences.

### Phase 2: BPE Models on Shakespeare (Scaling the Tokenizer)

Same from-scratch architecture, same data, but switching from character-level to Byte Pair Encoding. This tests whether a smarter tokenizer helps more than more parameters.

| Run | Architecture | Params | Vocab | Context | Stride | Epochs | Train Loss | Val Loss | Val PPL | Val Acc |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v007a | 6L/8H/256d, BPE | 5,121,536 | 1,000 | 512 | 256 | 20 | 3.9733 | 4.4288 | 83.83 | 12.0% |
| v007 | 6L/8H/256d, BPE | 5,633,536 | 3,000 | 512 | 256→64 | 20 | 2.8101 | 5.8721 | 355.01 | 9.8% |

**How to read these results:**

- **Both models overfit severely.** v007 has a train/val gap of 2.81 vs 5.87. This means the model memorized the training windows rather than learning generalizable patterns. This is expected: 5.6M parameters is enormous for ~270K BPE tokens of training data (~4K windows). There simply isn't enough data for this model size.

- **Why PPL is so high (355) and accuracy so low (9.8%):** With a 3,000-token BPE vocabulary, the model has to choose the right subword from 3,000 options at each step. At 9.8% accuracy, it's getting ~1 in 10 tokens right — bad numerically, but note that random guessing would be 0.03% (1/3000). It's learning real patterns, just not generalizing to unseen positions.

- **v007a (vocab 1K) vs v007 (vocab 3K):** v007a overfit less (gap 3.97 vs 4.43) because a smaller vocabulary produces more tokens per text, creating more training windows. But the smaller vocab can't represent meaningful subwords well. Neither configuration is ideal for this data size.

- **Despite terrible metrics, v007 produces the best text.** BPE lets each token represent a subword like "thou", "King", or "death" instead of a single letter. The 512-token context covers ~2,000 characters. So even though validation PPL is 355, the generated text contains full sentences, proper character names, and recognizable verse. The model memorized Shakespeare's patterns perfectly — it just can't generalize.

- **The real lesson:** tokenizer choice matters more than model size. An overfit BPE model produces more coherent text than a well-generalized character model because BPE operates at a higher level of abstraction.

### Phase 3: Instruction Model from Scratch (The Key Failure)

Moving from Shakespeare text generation to instruction-following. Still using my own Transformer, but now trained on Dolly 15K instruction/response pairs.

| Run | Architecture | Params | Vocab | Data | Epochs | Train Loss | Val Loss | Val PPL | Val Acc |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| v003 | 6L/8H/256d, BPE | 5,062,144 | 1,024 | Dolly 15K | 8 | 2.7063 | 2.9497 | 19.10 | 37.6% |

**How to read these results:**

- **The optimization worked perfectly.** Loss decreased at every epoch (55.7→19.1 PPL), training was stable, no divergence. The engineering pipeline is correct.

- **The outputs were completely wrong.** "What is 2+2?" produced random equations. "Explain photosynthesis" repeated topic words without any explanation. The model learned what answer-shaped text looks like (format) but has no knowledge of language, facts, or reasoning (content).

- **Why this happened:** Dolly is an instruction-tuning dataset — it teaches a model how to use its knowledge, not what to know. A randomly initialized 5M-parameter model doesn't know English, doesn't know arithmetic, and doesn't know what photosynthesis is. 15K examples can't teach all of that. This is why the industry uses a two-stage pipeline: (1) pretrain on billions of tokens to learn language, then (2) fine-tune on instruction data to learn the Q&A format.

- **This experiment is the most important in the project.** It proved, with my own code and data, exactly why pretraining matters. When interviewers ask "why can't you just train on instructions?" this is the answer.

### Phase 4: Pretrained Base + LoRA (The Industry Path)

Starting from Qwen2.5-0.5B (a pretrained model with 500M parameters) and training only a small LoRA adapter on the same Dolly data that v003 used.

| Run | Base Model | LoRA Rank | Trainable Params | Data | Context | Epochs | Train Loss | Val Loss | Val PPL |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| v004 | Qwen2.5-0.5B | 16 | 8,798,208 | Dolly 15K | 256 | 1 | 2.0087 | 2.0753 | 7.97 |
| v008a | Qwen2.5-0.5B-Instruct | 8 | 4,399,104 | Dolly 15K | 512 | 1 | 1.9531 | 2.0559 | 7.81 |
| v008 | Qwen2.5-0.5B-Instruct | 16 | 8,798,208 | Dolly 15K | 512 | 2 | 1.8076 | 2.0619 | 7.86 |

**How to read these results:**

- **v004 vs v003 — same data, dramatically different results.** v003 (from scratch) can't answer a single question. v004 (LoRA on pretrained) answers "2+2=4", "Paris", and writes correct Python code. The only difference is that v004 starts from a model that already knows language. The LoRA adapter just teaches it the instruction-following format.

- **v004 val PPL of 7.97:** This is computed over the assistant tokens only (prompt tokens are masked). It means the model is choosing among ~8 equally likely options for each subword token in its responses. This is reasonable for a 500M model — production models like GPT-4 would be much lower.

- **v008a vs v004 — instruct base beats raw base:** v008a used Qwen-Instruct (already instruction-tuned by Qwen's team) as the base instead of raw Qwen. With half the LoRA parameters (rank 8 vs 16), it achieved better val loss (2.0559 vs 2.0753). The upstream instruction tuning gives the LoRA adapter a better starting point.

- **v008 vs v008a — diminishing returns on epoch 2:** v008 trained lower (1.81 vs 1.95) but val loss barely changed (2.0619 vs 2.0559). The second epoch improved training fit but didn't improve generalization — classic sign of beginning to overfit. In practice, v008a (1 epoch, half the rank) is nearly as good as v008 (2 epochs, full rank), which means the first epoch captures most of the learning.

- **Smoke test progression:**

| Prompt | v003 (scratch) | v004 (base+LoRA) | v008 (instruct+LoRA) |
|---|---|---|---|
| Say hello | Garbled | Hello, how are you today? | Hello! |
| 2 + 2 | Wrong equations | 4 | Two plus two equals four |
| Capital of France | Unrelated text | Paris | The capital of France is Paris |
| Photosynthesis | Word repetition | Correct 1 sentence | Correct detailed explanation |
| Python add function | No code produced | `add_numbers(a,b): return a+b` | `add(x,y): return x+y` |

### Phase 5: Grounded Retrieval (Citation + Abstention)

Teaching the model to answer from provided evidence, cite sources, and refuse when it doesn't have information.

| Run | Base Model | Trainable Params | Data | Train Examples | Eval Cases | Train Loss | Val Loss | Val PPL |
|---|---|---:|---|---:|---:|---:|---:|---:|
| v005 | Qwen2.5-0.5B-Instruct | 540,672 | Portfolio evidence | 192 / 48 | 88 | 0.0219 | 0.0259 | 1.03 |

**How to read these results:**

- **PPL near 1.0 doesn't mean perfection.** The training data was 192 short, templated examples. The model memorized the narrow training format perfectly (PPL 1.03 means virtually no uncertainty). But this doesn't transfer to real-world questions — the evaluation showed only 22.2% of answerable questions passed combined content+citation checks.

- **The evaluation was task-specific, not just PPL.** I tested 88 cases across 4 configurations (with/without retrieval × with/without LoRA), checking: (1) does the response contain the right keywords? (2) does it cite the correct source document? (3) does it abstain when asked about undocumented topics?

- **What worked:** the model learned to cite sources (`[architecture]`) and to say "I don't have evidence to answer that" instead of inventing answers. The baseline invented a GPA of "3.50" for a question that had no answer in the evidence.

- **What didn't:** content coverage dropped (57→46 keyword matches), and one invalid citation was produced. The training distribution was too narrow.

---

## How the Results Look Overall (Honest Assessment)

**The from-scratch models (v001–v007) demonstrate understanding, not capability.** A 112K-parameter character model can't write English. A 5M-parameter model memorizes Shakespeare but can't generalize. These are expected results at these scales — the value is in showing that I understand why each limitation exists.

**The instruction experiment (v003) is the most important result.** It proves, with real data and real training, that instruction tuning without pretraining doesn't work. This single experiment explains why the entire LLM industry uses a two-stage pipeline.

**The LoRA experiments (v004, v008) show the industry path works.** Starting from a pretrained model gives immediate instruction-following ability. Starting from an instruction-tuned model gives even better results with fewer parameters. This is how real teams ship small models on limited hardware.

**The grounded experiment (v005) shows the right approach but not ready for deployment.** Citations and abstention are the right ideas, but the training data was too narrow and the evaluation found real failures. I didn't promote it — predefined quality gates caught the problems before deployment.

**No result in this project should be compared to GPT-4 or production LLMs.** The largest model here is 500M parameters trained on 15K examples. GPT-4 is estimated at 1.8T parameters trained on trillions of tokens. The comparison isn't meaningful. What's meaningful is that my 11 experiments demonstrate the same scaling laws, the same failure modes, and the same engineering discipline that production systems require.

For a detailed analysis of root causes (why models overfit, why v003 failed, why v006 PPL looks worse than v002) and planned next experiments, see [Future Work: Diagnosis and Planned Improvements](future_work.md).
