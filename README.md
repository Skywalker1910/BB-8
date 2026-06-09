# BB8 – A Transformer-Based Large Language Model

> Built from scratch in Python and PyTorch to demonstrate mastery of modern LLM engineering.

---

## Project Overview

**BB8** is a custom GPT-style Transformer decoder language model built entirely from scratch. This project is not designed to compete with production-scale models like GPT-4, Gemini, or Llama. Instead, it serves as a **portfolio-quality, Master's-level project** demonstrating deep understanding of:

- Transformer architecture and self-attention
- Tokenization strategies (Character, Word, BPE)
- Language model training pipelines
- Text generation decoding strategies
- Model evaluation and explainability

---

## Motivation

Modern LLMs are often treated as black boxes. Building one from scratch—implementing every weight matrix, every attention head, every loss computation—forces a true understanding of how these systems work at the mathematical and engineering level. BB8 exists to make that understanding visible and reproducible.

---

## Architecture

BB8 is a **decoder-only Transformer** (GPT-style), trained with a **causal language modeling** objective (predict the next token given all previous tokens).

```
Input Text
    │
    ▼
┌─────────────────┐
│  Tokenizer      │  Character / Word / BPE
└────────┬────────┘
         │  Token IDs
         ▼
┌─────────────────┐
│ Token Embedding │  vocab_size → d_model
└────────┬────────┘
         │
┌─────────────────┐
│  Pos. Encoding  │  Learned positional embeddings
└────────┬────────┘
         │
┌─────────────────┐  ┐
│  Transformer    │  │
│  Decoder Block  │  │  × N layers
│  ─────────────  │  │
│  LayerNorm      │  │
│  Masked MHA     │  │  Multi-Head Self-Attention
│  Residual Add   │  │
│  LayerNorm      │  │
│  Feed-Forward   │  │  GELU activation
│  Residual Add   │  │
└────────┬────────┘  ┘
         │
┌─────────────────┐
│   LayerNorm     │
└────────┬────────┘
         │
┌─────────────────┐
│   LM Head       │  d_model → vocab_size (tied weights)
└────────┬────────┘
         │
    Logits / Loss
```

### Key Design Choices

| Component | Choice | Reason |
|---|---|---|
| Positional Encoding | Learned | More flexible than sinusoidal; used in GPT-2/3 |
| Normalization | Pre-LN | More stable gradients than Post-LN |
| Activation | GELU | Smoother than ReLU; used in modern LLMs |
| Weight Tying | Yes | Ties LM head to token embedding; reduces parameters |
| Optimizer | AdamW | Weight decay with correct bias correction |
| LR Schedule | Cosine + Warmup | Standard in modern LLM training |

---

## Repository Structure

```
BB8/
├── data/                   # Raw training data
│   └── prepare_data.py     # Download sample datasets
├── datasets/               # PyTorch Dataset classes
│   └── text_dataset.py
├── models/                 # All model components
│   ├── embeddings.py       # Token + positional embeddings
│   ├── attention.py        # Scaled dot-product & multi-head attention
│   ├── feed_forward.py     # Position-wise FFN
│   ├── transformer.py      # Decoder block + causal mask
│   └── language_model.py   # Full BB8LM model
├── tokenizer/              # Tokenization strategies
│   ├── base_tokenizer.py   # Abstract base class
│   ├── char_tokenizer.py   # Character-level
│   ├── word_tokenizer.py   # Word-level
│   └── bpe_tokenizer.py    # Byte Pair Encoding
├── training/               # Training infrastructure
│   ├── trainer.py          # Training loop, checkpointing
│   ├── scheduler.py        # LR schedulers
│   └── losses.py           # Loss utilities
├── evaluation/             # Evaluation metrics
│   └── evaluator.py        # Perplexity, accuracy, BPC
├── inference/              # Text generation
│   └── generator.py        # Greedy, temperature, top-k, top-p
├── experiments/            # Experiment scripts
│   └── run_experiment.py   # Config-driven experiment runner
├── notebooks/              # Jupyter exploration notebooks
│   ├── 01_tokenization_demo.ipynb
│   ├── 02_attention_visualization.ipynb
│   └── 03_training_demo.ipynb
├── docs/                   # Technical documentation
│   ├── tokenization.md
│   ├── attention.md
│   ├── transformer_architecture.md
│   ├── training_pipeline.md
│   ├── evaluation.md
│   └── experiments.md
├── tests/                  # Unit tests
│   ├── test_tokenizers.py
│   ├── test_attention.py
│   └── test_model.py
├── configs/                # YAML experiment configs
│   ├── default.yaml
│   ├── small_model.yaml
│   └── medium_model.yaml
├── checkpoints/            # Saved model checkpoints
├── outputs/                # Generated text, plots, logs
├── train.py                # Quick-start training script
└── requirements.txt
```

---

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd BB8

# Create a virtual environment
python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

---

## Quick Start

### 1. Download sample training data

```bash
python data/prepare_data.py --dataset tiny_shakespeare
```

### 2. Train BB8

```bash
# Default: character-level tokenizer, small model
python train.py --data data/tiny_shakespeare.txt

# BPE tokenizer, medium model
python train.py --data data/tiny_shakespeare.txt \
    --tokenizer bpe --vocab-size 3000 \
    --d-model 256 --num-layers 6 --num-heads 8 \
    --epochs 20
```

### 3. Run a config-driven experiment

```bash
python experiments/run_experiment.py \
    --config configs/small_model.yaml \
    --data data/tiny_shakespeare.txt \
    --name experiment_01
```

### 4. Run tests

```bash
pytest tests/ -v
```

---

## Development Phases

| Phase | Topic | Status |
|---|---|---|
| 1 | Tokenization (Char / Word / BPE) | ✅ |
| 2 | Embeddings + Positional Encoding | ✅ |
| 3 | Multi-Head Self-Attention | ✅ |
| 4 | Transformer Decoder Architecture | ✅ |
| 5 | Language Model Training | ✅ |
| 6 | Text Generation Strategies | ✅ |
| 7 | Evaluation Framework | ✅ |
| 8 | Explainability & Visualization | ✅ |
| 9 | Experiments & Ablation Studies | 🔄 |

---

## Dataset Information

| Dataset | Size | Description |
|---|---|---|
| Tiny Shakespeare | ~1 MB | Complete works of Shakespeare; classic LM benchmark |

Training data is plain text. Any UTF-8 text file can be used.

---

## Training Process

### Objective

BB8 is trained with **causal language modeling** (next-token prediction):

$$\mathcal{L} = -\sum_{t=1}^{T} \log P(x_t \mid x_1, \ldots, x_{t-1})$$

### Default Hyperparameters

| Hyperparameter | Small Model | Medium Model |
|---|---|---|
| `d_model` | 128 | 256 |
| `num_layers` | 4 | 6 |
| `num_heads` | 4 | 8 |
| `d_ff` | 512 | 1024 |
| `max_seq_len` | 256 | 512 |
| `dropout` | 0.1 | 0.1 |
| `batch_size` | 32 | 16 |
| `learning_rate` | 3e-4 | 3e-4 |
| `optimizer` | AdamW | AdamW |
| `lr_schedule` | Cosine+Warmup | Cosine+Warmup |
| `grad_clip` | 1.0 | 1.0 |

---

## Results

> Results will be filled in after running experiments.

| Experiment | Tokenizer | Model Size | Train Loss | Val Loss | Perplexity |
|---|---|---|---|---|---|
| 01 – Small baseline | Char | 128d, 4L | — | — | — |
| 02 – More layers | Char | 128d, 6L | — | — | — |
| 03 – Larger embeddings | Char | 256d, 4L | — | — | — |
| 04 – BPE tokenizer | BPE (3k) | 128d, 4L | — | — | — |
| 05 – Medium model | BPE (5k) | 256d, 6L | — | — | — |

---

## Sample Generated Text

> Samples will be added after training.

```
Prompt: "To be or not to be"
Strategy: Top-P (p=0.9, temp=0.8)
Output: [...]
```

---

## Lessons Learned

> This section will be updated throughout development.

- BPE produces significantly better compression than character-level tokenization
- Pre-LN architectures train more stably than Post-LN
- Weight tying between embeddings and LM head reduces parameters with no quality loss
- Cosine LR schedule with warmup consistently outperforms constant LR

---

## Documentation

Detailed technical notes are in the `docs/` directory:

- [Tokenization](docs/tokenization.md)
- [Attention Mechanism](docs/attention.md)
- [Transformer Architecture](docs/transformer_architecture.md)
- [Training Pipeline](docs/training_pipeline.md)
- [Evaluation](docs/evaluation.md)
- [Experiments](docs/experiments.md)

---

## License

MIT License – free to use, modify, and build upon.
