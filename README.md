# BB8 — Transformer Language Model Built from Scratch

A GPT-style decoder-only Transformer I built from scratch in PyTorch to understand how large language models actually work — from raw matrix operations to text generation. This is a graduate-level learning project, not a production chatbot.

---

## Why I Built This

I wanted to go beyond using LLM APIs and actually understand what happens inside a Transformer. That meant implementing every component myself: the attention mechanism, the feed-forward layers, the positional encodings, the tokenizers, the training loop, and the inference pipeline. I then ran a series of eight experiments to see how each design choice — model size, tokenizer, dataset, training strategy — affects what the model learns.

The project grew into three tracks:
1. **From-scratch Transformer** — my own GPT-style model, every weight matrix written by hand
2. **LoRA fine-tuning** — adapting a pretrained Qwen model to understand the industry workflow
3. **Grounded retrieval** — teaching a model to cite evidence and say "I don't know"

---

## What I Implemented from Scratch

Every component below is my own PyTorch code, not imported from HuggingFace or any library:

| Component | File | What It Does |
|---|---|---|
| Token Embedding | `models/embeddings.py` | Maps token IDs to dense vectors |
| Positional Encoding | `models/embeddings.py` | Learned position embeddings (like GPT-2) |
| Scaled Dot-Product Attention | `models/attention.py` | Core QKV attention with causal masking |
| Multi-Head Self-Attention | `models/attention.py` | Parallel attention heads with output projection |
| Feed-Forward Network | `models/feed_forward.py` | Two-layer MLP with GELU activation |
| Transformer Decoder Block | `models/transformer.py` | Pre-LN block with residual connections |
| Causal Mask | `models/transformer.py` | Prevents attending to future tokens |
| Full Language Model | `models/language_model.py` | Stacks everything together, weight-tied LM head |
| Character Tokenizer | `tokenizer/char_tokenizer.py` | Maps individual characters to IDs |
| Word Tokenizer | `tokenizer/word_tokenizer.py` | Whitespace-based tokenization |
| BPE Tokenizer | `tokenizer/bpe_tokenizer.py` | Byte Pair Encoding with merge rules |
| Training Loop | `training/trainer.py` | AdamW, cosine LR schedule, mixed precision, checkpointing |
| LR Scheduler | `training/scheduler.py` | Cosine annealing with linear warmup |
| Evaluation | `evaluation/evaluator.py` | Perplexity, accuracy, bits per character |
| Text Generation | `inference/generator.py` | Greedy, temperature, top-k, top-p decoding |

---

## Architecture

```
Input Text
    |
    v
+-----------------+
|  Tokenizer      |  Character / Word / BPE
+--------+--------+
         |  Token IDs
         v
+-----------------+
| Token Embedding |  vocab_size -> d_model
+--------+--------+
         |
+-----------------+
|  Pos. Encoding  |  Learned positional embeddings
+--------+--------+
         |
+-----------------+  x N layers
| Transformer     |
| Decoder Block   |
|  -------------- |
|  LayerNorm      |
|  Masked MHA     |  Multi-Head Self-Attention
|  Residual Add   |
|  LayerNorm      |
|  Feed-Forward   |  GELU activation
|  Residual Add   |
+--------+--------+
         |
+-----------------+
|   LayerNorm     |
+--------+--------+
         |
+-----------------+
|   LM Head       |  d_model -> vocab_size (tied weights)
+--------+--------+
         |
    Logits / Loss
```

### Design Choices I Made and Why

| Decision | Choice | Why |
|---|---|---|
| Architecture | Decoder-only | Same as GPT-2/3 — simplest for next-token prediction |
| Positional Encoding | Learned | More flexible than sinusoidal; used in GPT-2 |
| Normalization | Pre-LN (norm before attention) | More stable gradients during training |
| Activation | GELU | Smoother than ReLU; standard in modern LLMs |
| Weight Tying | LM head shares token embedding weights | Cuts parameters without hurting quality |
| Optimizer | AdamW | Proper weight decay with bias correction |
| LR Schedule | Cosine decay with linear warmup | Prevents early instability, standard practice |
| Residual Scaling | 1/sqrt(2N) on output projections | GPT-2 trick for stable deep networks |

---

## Experiment Progression (v001 – v008)

I ran 11 experiments across four phases, each building on what I learned from the last. The progression goes from the simplest possible model to a pretrained chat system with evidence retrieval.

### Phase 1 — From-Scratch Character Models (Shakespeare, my own Transformer)

| Run | Architecture | Params | Val PPL | Key Takeaway |
|---|---|---:|---:|---|
| v001 | 2L/2H/64d, char | 112,320 | 5.32 | First working model (had data leak — historical only) |
| v002 | 2L/2H/64d, char | 112,320 | 5.40 | Fixed pipeline — trusted baseline |
| v006 | 4L/4H/128d, char | 833,408 | 9.57 | 7x params, stricter eval — scaling works |
| v006b | 6L/4H/128d, char | 1,229,184 | 8.24 | +2 layers = 14% PPL gain — depth > width |

### Phase 2 — BPE Tokenization (Shakespeare, my own Transformer + BPE tokenizer)

| Run | Architecture | Params | Val PPL | Key Takeaway |
|---|---|---:|---:|---|
| v007a | 6L/8H/256d, BPE-1K | 5,121,536 | 83.83 | Vocab too small, moderate overfitting |
| v007 | 6L/8H/256d, BPE-3K | 5,633,536 | 355.01 | Heavy overfitting, but best generated text |

### Phase 3 — Instruction Learning (Dolly 15K)

| Run | Base | Trainable Params | Val PPL | Key Takeaway |
|---|---|---:|---:|---|
| v003 | None (from scratch) | 5,062,144 | 19.10 | Can't learn language from 15K instructions |
| v004 | Qwen2.5-0.5B + LoRA | 8,798,208 | 7.97 | Pretrained knowledge is the difference |
| v008a | Qwen2.5-0.5B-Instruct + LoRA r=8 | 4,399,104 | 7.81 | Instruct base + half the params beats v004 |
| v008 | Qwen2.5-0.5B-Instruct + LoRA r=16 | 8,798,208 | 7.86 | Best chat model, all smoke tests correct |

### Phase 4 — Grounded Retrieval (Portfolio evidence)

| Run | Base | Trainable Params | Val PPL | Key Takeaway |
|---|---|---:|---:|---|
| v005 | Qwen2.5-0.5B-Instruct + LoRA | 540,672 | 1.03 | Learned citations + abstention, not promoted |

Full analysis with metric interpretation: [Training Results](docs/training_results.md)
Experiment configs and run commands: [Experiments](docs/experiments.md)

**Note on perplexity:** PPL numbers are only comparable within the same tokenizer and evaluation setup. Character PPL (v002: 5.40) and BPE PPL (v007: 355) measure different things — see the training results doc for fair cross-tokenizer comparisons.

---

## Quick Start

### 1. Install

```bash
git clone <repo-url>
cd BB8
python -m venv venv
venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

### 2. Download training data

```bash
python data/prepare_data.py --dataset tiny_shakespeare
python data/prepare_data.py --dataset dolly_15k
```

### 3. Train the from-scratch model

```bash
# Small character model (v002 baseline, ~2 min on GPU)
python experiments/run_experiment.py \
    --config configs/small_model.yaml \
    --data data/tiny_shakespeare.txt \
    --name bb8-char-small-v002

# Medium character model (v006, ~8 min on GPU)
python experiments/run_experiment.py \
    --config configs/char_medium_v006.yaml \
    --data data/tiny_shakespeare.txt \
    --name bb8-char-medium-v006

# BPE medium model (v007, ~18 min on GPU)
python experiments/run_experiment.py \
    --config configs/bpe_shakespeare_v007.yaml \
    --data data/tiny_shakespeare.txt \
    --name bb8-bpe-shakespeare-v007

# From-scratch instruction model (v003, ~14 min on GPU)
python experiments/run_experiment.py \
    --config configs/instruction_v003.yaml \
    --data data/dolly_15k_chat.txt \
    --name bb8-bpe-instruct-v003-dev \
    --prompt "USER: What is machine learning? BB:" \
    --seed 42
```

### 4. Fine-tune with LoRA (requires GPU + extra dependencies)

```bash
pip install -r requirements-finetune.txt

# Qwen base + LoRA (v004)
python fine_tune.py \
    --config configs/qwen_lora_v004.yaml \
    --name bb8-qwen-lora-v004-dev

# Qwen-Instruct + LoRA (v008)
python fine_tune.py \
    --config configs/qwen_instruct_v008.yaml \
    --name bb8-qwen-instruct-v008
```

### 5. Generate text

```bash
python generate.py \
    --model-dir checkpoints/bb8-char-medium-v006 \
    --prompt "ROMEO:" \
    --max-new-tokens 150
```

### 6. Chat

```bash
python chat.py \
    --model-dir checkpoints/bb8-qwen-instruct-v008 \
    --device cuda
```

### 7. Run tests

```bash
pytest tests/ -v
```

### 8. Interactive Token Lab

Start the local server and open `http://127.0.0.1:8000/lab` to inspect next-token logits, sampling distributions, and decoding behavior step by step.

```bash
python -m api.local_server \
    --model-dir checkpoints/bb8-qwen-instruct-v008 \
    --api-key local-test-key \
    --device cuda
```

### 9. Deploy to AWS

The inference API deploys as a CPU Lambda container behind API Gateway.
See [AWS Deployment](docs/aws_deployment.md) for full instructions.

---

## Repository Structure

```
BB8/
├── models/                 # Every layer implemented from scratch
│   ├── embeddings.py       # Token + positional embeddings
│   ├── attention.py        # Scaled dot-product & multi-head attention
│   ├── feed_forward.py     # Position-wise FFN with GELU
│   ├── transformer.py      # Decoder block + causal mask
│   └── language_model.py   # Full BB8LM: stack of blocks + LM head
├── tokenizer/              # Three tokenizers, all from scratch
│   ├── char_tokenizer.py   # Character-level
│   ├── word_tokenizer.py   # Word-level
│   └── bpe_tokenizer.py    # Byte Pair Encoding
├── training/               # Training infrastructure
│   ├── trainer.py          # Training loop, checkpointing, mixed precision
│   ├── scheduler.py        # Cosine LR with warmup
│   └── losses.py           # Loss utilities
├── evaluation/             # Evaluation metrics
│   └── evaluator.py        # Perplexity, accuracy, bits per character
├── inference/              # Text generation + serving
│   ├── generator.py        # Greedy, temperature, top-k, top-p
│   ├── model_loader.py     # Load any BB8 checkpoint for inference
│   ├── conversation.py     # Chat prompt construction
│   └── lab.py              # Token inspection engine
├── api/                    # Local and AWS Lambda HTTP inference
├── grounded/               # Evidence retrieval + citation pipeline
├── datasets/               # PyTorch Dataset classes
├── experiments/             # Experiment runner + tracking
│   ├── run_experiment.py   # Config-driven experiment runner
│   └── tracking.py         # JSONL registry with hashes and Git info
├── configs/                # YAML configs for all 8 experiments
├── data/                   # Raw data + preparation scripts
├── tests/                  # Unit tests
├── notebooks/              # Jupyter exploration notebooks
├── docs/                   # Technical documentation
├── scripts/                # Deployment packaging
├── train.py                # Quick-start training script
├── fine_tune.py            # LoRA fine-tuning script
├── generate.py             # Load checkpoint + generate text
├── chat.py                 # Interactive terminal chat
├── Dockerfile              # AWS Lambda container
└── template.yaml           # AWS SAM infrastructure
```

---

## Datasets

| Dataset | Size | Used In |
|---|---|---|
| Tiny Shakespeare | ~1 MB, complete works | v001, v002, v006, v007 |
| Databricks Dolly 15K | 15,011 instruction/response pairs, ~12 MB | v003, v004, v008 |
| Portfolio Evidence Corpus | 12 curated documents, manually authored | v005 |

---

## Sample Output

**v002 (112K param character model on Shakespeare):**
```
Prompt: "ROMEO:"
Strategy: Top-P (p=0.9, temp=0.8)
Output: "Then, I'll thou hast my words your hand ..."
```

**v007 (5.6M param BPE model on Shakespeare):**
```
Prompt: "ROMEO:"
Strategy: Top-P (p=0.9, temp=0.8)
Output: "O, she doth teach the torches to burn bright.
         It seems she hangs upon the cheek of night
         As a rich jewel in an Ethiope's ear ..."
```

**v008 (Qwen-Instruct + LoRA):**
```
Q: What is 2 + 2?
A: 4

Q: What is the capital of France?
A: Paris

Q: Write a Python function that adds two numbers.
A: def add_numbers(a, b):
       return a + b
```

---

## What I Learned

These are the main takeaways from running all eight experiments:

1. **You can't skip pretraining.** v003 tried to teach a randomly initialized model to follow instructions using only 15K Dolly examples. It learned to produce answer-shaped text but hallucinated everything. Language knowledge has to come from broad pretraining first.

2. **More parameters help, but with diminishing returns.** v006 (835K params) beat v002 (112K params) by 18% on perplexity, but still couldn't produce coherent paragraphs. The model learned better word patterns and Shakespeare formatting, but not meaning.

3. **BPE tokenization captures more structure.** v007 with BPE achieved much lower bits-per-character than the character models because each token carries more information. The generated text was noticeably more coherent.

4. **LoRA on a pretrained model is dramatically better.** v004 answered basic questions correctly on day one. My from-scratch v003 with similar parameter count couldn't answer anything. The pretrained base model already knows language; LoRA just steers it.

5. **Starting from an instruction-tuned base helps even more.** v008 (Qwen-Instruct + LoRA) handled arithmetic and multi-step questions that v004 (Qwen base + LoRA) got wrong. Upstream instruction tuning gives a better foundation to build on.

6. **Small models hallucinate.** Even v008 invents facts outside simple questions. This is expected at 500M parameters with 15K training examples. Production systems need retrieval, guardrails, and much larger models.

7. **Data integrity matters from day one.** v001 had a subtle data leak (tokenizer trained on test data). The metrics looked slightly better but were untrustworthy. v002 fixed this. I now hash every dataset and track every config.

8. **Honest evaluation is more valuable than good numbers.** Documenting failures (v003 hallucinating, v004 failing subtraction, v005 not meeting promotion gates) taught me more than the successes.

---

## Documentation

- [Training Results](docs/training_results.md) — full learning curves and analysis for all 8 experiments
- [Experiments](docs/experiments.md) — configs, run commands, and ablation design
- [Tokenization](docs/tokenization.md)
- [Attention Mechanism](docs/attention.md)
- [Transformer Architecture](docs/transformer_architecture.md)
- [Training Pipeline](docs/training_pipeline.md)
- [Evaluation](docs/evaluation.md)
- [Interactive Model Lab](docs/model_lab.md)
- [Grounded Assistant](docs/grounded_assistant.md)
- [AWS Deployment](docs/aws_deployment.md)
- [Future Work: Diagnosis and Planned Improvements](docs/future_work.md)

---

## Hardware

All experiments ran on a single NVIDIA RTX 4070 Laptop GPU (8 GB VRAM).
The character models also train on CPU in minutes.

---

## License

MIT License
