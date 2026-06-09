# Attention Mechanisms in BB8

## The Core Idea

The attention mechanism allows every position in a sequence to directly communicate with every other position.  This is fundamentally different from RNNs, which propagate information step-by-step and struggle with long-range dependencies.

> *"Attention is the mechanism that lets the model decide, at each step, which parts of its context are most relevant."*

---

## Scaled Dot-Product Attention

**Implemented in:** `models/attention.py` → `ScaledDotProductAttention`

The fundamental attention operation:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

### The Three Matrices

For each input sequence, three matrices are created:

| Matrix | Name | What it represents |
|---|---|---|
| **Q** | Query | "What am I looking for?" |
| **K** | Key | "What do I contain?" |
| **V** | Value | "What do I actually return?" |

Each Q, K, V is a linear projection of the input.

### Step-by-Step Computation

```
Input: X  shape (batch, seq_len, d_model)

1. Project:
   Q = X · W_Q    (batch, seq_len, d_k)
   K = X · W_K    (batch, seq_len, d_k)
   V = X · W_V    (batch, seq_len, d_v)

2. Compute raw attention scores:
   scores = Q · Kᵀ / √d_k    (batch, seq_len, seq_len)
   
   scores[i, j] = how much position i attends to position j

3. Apply optional mask (causal or padding):
   scores[masked positions] = -∞

4. Softmax over keys:
   weights = softmax(scores, dim=-1)    (batch, seq_len, seq_len)
   
   weights[i, :] sums to 1.0
   weights[i, j] = "how much i attends to j"

5. Weighted sum of values:
   output = weights · V    (batch, seq_len, d_v)
```

### Why Scale by √d_k?

Without scaling, dot products grow proportional to d_k.  For large d_k, the scores become very large in magnitude, pushing the softmax into saturation regions where gradients are near zero.

Dividing by √d_k keeps the variance of the dot products approximately 1.0 regardless of d_k.

---

## Causal Mask

BB8 is an **autoregressive** model — at inference time, position t cannot see tokens at positions > t because they haven't been generated yet.

This is enforced during training by masking the upper triangle of the attention score matrix:

```
seq_len = 4:

Mask:              Scores after masking:
1  0  0  0         s₀₀  -∞   -∞   -∞
1  1  0  0    →    s₁₀  s₁₁  -∞   -∞
1  1  1  0         s₂₀  s₂₁  s₂₂  -∞
1  1  1  1         s₃₀  s₃₁  s₃₂  s₃₃

After softmax:  each row sums to 1,
                upper triangle = 0 (can't attend to future)
```

**Implemented in:** `models/transformer.py` → `create_causal_mask()`

---

## Multi-Head Attention

**Implemented in:** `models/attention.py` → `MultiHeadAttention`

Instead of a single attention function, multi-head attention runs *h* attention heads in **parallel**, each in a lower-dimensional subspace:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) \cdot W^O$$

$$\text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)$$

### Why Multiple Heads?

Different heads can learn to attend to different types of relationships simultaneously:

- **Head 1** might track syntactic structure
- **Head 2** might track coreference (pronouns → antecedents)
- **Head 3** might track positional proximity
- **Head 4** might track semantic roles

By running *h* heads in parallel and concatenating their outputs, the model gains a richer multi-perspective view of each token's context.

### Dimension Bookkeeping

```
d_model = 256
num_heads = 8
d_k = d_model / num_heads = 32   (dimension per head)

Q: (batch, seq, d_model)
   → split into h heads →  (batch, heads, seq, d_k)
   
After attention:  (batch, heads, seq, d_k)
   → merge heads →  (batch, seq, d_model)
   → W_O projection →  (batch, seq, d_model)
```

### Implementation

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention(dropout)

    def forward(self, query, key, value, mask=None):
        Q = self._split_heads(self.W_q(query))  # (B, H, T, d_k)
        K = self._split_heads(self.W_k(key))
        V = self._split_heads(self.W_v(value))
        x, weights = self.attention(Q, K, V, mask)
        x = self._merge_heads(x)
        return self.W_o(x), weights
```

---

## Attention Visualisation

Attention weight matrices can be visualised as heatmaps.  Each cell (i, j) shows how much position i attends to position j.

Run `notebooks/02_attention_visualization.ipynb` to produce attention maps after training.

### Interpreting Attention Maps

- **Diagonal patterns** — attending to the current position (identity)
- **Left-column patterns** — attending to the first token (often a start-of-text marker)
- **Diffuse patterns** — attending broadly across context
- **Sharp patterns** — attending to one or two specific positions

Different layers and heads show qualitatively different patterns.

---

## Parameter Count

For a single multi-head attention layer with d_model=256, num_heads=8:

| Matrix | Shape | Parameters |
|---|---|---|
| W_Q | (256, 256) | 65,536 |
| W_K | (256, 256) | 65,536 |
| W_V | (256, 256) | 65,536 |
| W_O | (256, 256) | 65,536 |
| **Total** | | **262,144** |

This is why LLM parameter counts grow so quickly with d_model.
