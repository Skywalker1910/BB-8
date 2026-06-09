# Transformer Architecture in BB8

## Overview

BB8 implements a **decoder-only Transformer** — the same architectural family as GPT-2, GPT-3, and LLaMA.  It is trained with a **causal language modelling** (next-token prediction) objective.

This is distinct from the original encoder-decoder Transformer (Vaswani et al., 2017) used for machine translation, and from encoder-only models like BERT used for classification tasks.

---

## Full Architecture

```
Input: sequence of token IDs (batch, seq_len)

┌─────────────────────────────────┐
│         Token Embedding         │  (vocab_size → d_model)
│   + Learned Position Embedding  │  (position 0..T → d_model)
│           + Dropout             │
└────────────────┬────────────────┘
                 │
                 │  x: (batch, seq_len, d_model)
                 │
     ┌───────────┴───────────┐
     │                       │  × N layers (num_layers)
     │  ┌─────────────────┐  │
     │  │   LayerNorm(x)  │  │  Pre-LN: normalise BEFORE attention
     │  └────────┬────────┘  │
     │           │            │
     │  ┌────────▼────────┐  │
     │  │   Masked MHA    │  │  Causal self-attention
     │  └────────┬────────┘  │
     │           │ + x        │  Residual connection
     │  ┌────────▼────────┐  │
     │  │   LayerNorm(x)  │  │  Pre-LN: normalise BEFORE FFN
     │  └────────┬────────┘  │
     │           │            │
     │  ┌────────▼────────┐  │
     │  │  Feed-Forward   │  │  2-layer MLP: d_model → 4d_model → d_model
     │  └────────┬────────┘  │
     │           │ + x        │  Residual connection
     └───────────┴───────────┘
                 │
     ┌───────────▼───────────┐
     │      LayerNorm        │  Final normalisation
     └───────────┬───────────┘
                 │
     ┌───────────▼───────────┐
     │        LM Head        │  Linear: d_model → vocab_size
     │    (weight-tied)      │  Weights shared with token embedding
     └───────────┬───────────┘
                 │
           Logits (batch, seq_len, vocab_size)
           Loss   (cross-entropy, if targets provided)
```

---

## Key Design Decisions

### 1. Decoder-Only (GPT-style)

The encoder-decoder architecture from the original Transformer paper requires a source sequence (for the encoder) and a target sequence (for the decoder).  This is natural for translation but overly complex for language modelling.

Decoder-only models:
- Process a single sequence
- Use causal (masked) self-attention to prevent attending to future tokens
- Are simpler to implement and scale
- Have dominated recent language modelling research (GPT-2 through LLaMA)

### 2. Pre-LN (Layer Normalisation Before Sub-Layer)

**Original Transformer (Post-LN):**
```
x = x + SubLayer(x)
x = LayerNorm(x)
```

**GPT-2 / BB8 (Pre-LN):**
```
x = x + SubLayer(LayerNorm(x))
```

Pre-LN provides more stable gradient flow.  In Post-LN, the output of the final layer does not pass through a normalisation layer, which can lead to large gradient magnitudes early in training.  Pre-LN makes training more stable and often requires less learning rate tuning.

### 3. Residual Connections

Every sub-layer (attention, FFN) wraps its output with a residual connection:

$$x = x + \text{SubLayer}(\text{LayerNorm}(x))$$

Residual connections solve the vanishing gradient problem in deep networks by providing "gradient highways" that allow gradients to flow through many layers with minimal attenuation.  They were introduced in ResNets (He et al., 2016) and are essential in Transformers.

### 4. GELU Activation

The Feed-Forward Networks use **GELU** (Gaussian Error Linear Unit) instead of ReLU:

$$\text{GELU}(x) = x \cdot \Phi(x)$$

where $\Phi$ is the Gaussian CDF.  GELU is smoother than ReLU (no hard threshold at 0) and empirically outperforms ReLU in Transformer language models (as used in BERT, GPT-2/3, GPT-4).

### 5. Weight Tying

The LM Head weight matrix $W_{\text{head}} \in \mathbb{R}^{d\_model \times V}$ is shared with the token embedding matrix $W_e \in \mathbb{R}^{V \times d\_model}$.

Benefits:
- Saves $d\_model \times V$ parameters (significant for large vocabularies)
- Forces the model's internal representation space to align with the "unembedding" space
- Often improves generalisation, especially with small datasets

Reference: Press & Wolf (2017). "Using the Output Embedding to Improve Language Models."

### 6. Learned Positional Encoding

Positions 0, 1, …, max_seq_len−1 each have a learned embedding vector.  During training, the model learns to embed position information that is most useful for language modelling.

Alternative: Sinusoidal positional encoding (fixed, no learned parameters).  Both are implemented in `models/embeddings.py`.

---

## Component Sizes

For the **default** BB8 configuration (d_model=128, num_layers=4, num_heads=4, d_ff=512):

| Component | Parameters |
|---|---|
| Token Embedding (vocab≈70) | 70 × 128 ≈ 9K |
| Position Embedding (256 positions) | 256 × 128 ≈ 33K |
| Multi-Head Attention (per layer) | 4 × 128² ≈ 66K |
| Feed-Forward (per layer) | 2 × 128 × 512 ≈ 131K |
| LayerNorm (per layer, ×2) | 2 × 128 ≈ 256 |
| **Total (4 layers)** | **~0.8M** |

For the **medium** configuration (d_model=256, num_layers=6, num_heads=8, d_ff=1024):

| Component | Parameters |
|---|---|
| Per-layer attention | 4 × 256² ≈ 262K |
| Per-layer FFN | 2 × 256 × 1024 ≈ 524K |
| **Total (6 layers)** | **~5M** |

---

## The Feed-Forward Network

Each Transformer block contains a position-wise FFN:

$$\text{FFN}(x) = \text{GELU}(x \cdot W_1 + b_1) \cdot W_2 + b_2$$

- Input/output dimension: d_model
- Hidden dimension: d_ff = 4 × d_model (typically)
- Same weights applied independently at every position

The FFN is thought to act as a "memory" component, storing factual associations that attention queries.  Research has shown that specific neurons in the FFN store specific facts (e.g., "Paris is the capital of France").

---

## Inference: Autoregressive Generation

At generation time, the model produces one token at a time:

```
1. Encode prompt → token_ids
2. Forward pass → logits for the last position
3. Sample next_token from logits
4. Append next_token to sequence
5. Repeat from step 2 until max_tokens or <EOS>
```

The entire previously generated sequence is re-fed at each step (KV-cache optimisation is not implemented in BB8 to keep the code educational).
