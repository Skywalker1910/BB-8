# Training Pipeline

## Objective: Causal Language Modelling

BB8 is trained to predict the next token given all previous tokens:

$$\mathcal{L} = -\frac{1}{T} \sum_{t=1}^{T} \log P(x_t \mid x_1, x_2, \ldots, x_{t-1})$$

This is the standard **cross-entropy loss** applied to every position in the sequence simultaneously.  PyTorch's `F.cross_entropy` handles this efficiently.

At training time, the causal mask ensures that the model cannot "cheat" by looking at future tokens — it must make each prediction using only past context.

---

## Data Preparation

```
Raw text file
    ↓  sequential raw-text split
Training text  +  Validation text
    ↓  fit CharTokenizer / WordTokenizer / BPETokenizer on training text only
Flat list of token IDs  [t₁, t₂, t₃, …, tₙ]
    ↓  TextDataset(seq_len=256, stride=1)
Overlapping windows:
    Input:  [t₁, t₂, …, t₂₅₆]
    Target: [t₂, t₃, …, t₂₅₇]
    ...
    Input:  [t₂, t₃, …, t₂₅₇]
    Target: [t₃, t₄, …, t₂₅₈]
    ...
Train DataLoader  +  Val DataLoader
```

The raw corpus is split **sequentially before tokenizer training**, not randomly.
This prevents validation-only vocabulary or BPE merge rules from leaking into
training. `stride=1` reproduces the densely overlapping baseline; larger
strides reduce training time on larger corpora.

---

## Optimizer: AdamW

AdamW (Adam with decoupled weight decay) is the standard optimizer for Transformer language models.

$$\theta_t = \theta_{t-1} - \alpha \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_{t-1} \right)$$

Key hyperparameters:

| Hyperparameter | Default | Notes |
|---|---|---|
| Learning rate α | 3×10⁻⁴ | Peak LR after warmup |
| β₁ | 0.9 | Momentum coefficient |
| β₂ | 0.95 | RMS coefficient (slightly lower than Adam's 0.999) |
| ε | 1×10⁻⁸ | Numerical stability |
| λ (weight decay) | 0.1 | Applied only to weight matrices |

### Selective Weight Decay

Weight decay (L2 regularisation) is applied **only to weight matrices**, not to:
- Bias vectors
- LayerNorm parameters (γ, β)

This follows the practice in GPT-2/GPT-3.  Applying weight decay to biases and layer norm parameters tends to hurt performance.

---

## Learning Rate Schedule: Cosine Annealing with Warmup

A constant learning rate is suboptimal:
- Too high early → unstable training (large random gradients at initialisation)
- Too low late → slow convergence

BB8 uses a two-phase schedule:

**Phase 1 — Linear Warmup (steps 0 → warmup_steps)**
$$\alpha(t) = \alpha_{\text{max}} \cdot \frac{t}{\text{warmup\_steps}}$$

**Phase 2 — Cosine Decay (steps warmup_steps → max_steps)**
$$\alpha(t) = \alpha_{\text{min}} + \frac{1}{2}(\alpha_{\text{max}} - \alpha_{\text{min}}) \left(1 + \cos\left(\pi \cdot \frac{t - \text{warmup}}{T - \text{warmup}}\right)\right)$$

Where $\alpha_{\text{min}} = 0.1 \times \alpha_{\text{max}}$ (10% of peak LR).

```
Learning rate schedule (visual):

α_max ─────────────╮
                   │╲
                   │ ╲
                   │  ╲
                   │   ╲ (cosine decay)
α_min ─────────────────────────── (floor)

      |──warmup──|──────── cosine decay ────────|
      0      warmup_steps                  max_steps
```

---

## Gradient Clipping

The gradient norm is clipped to a maximum value (default: 1.0):

```python
nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

This prevents rare but catastrophic gradient explosions that can occur during early training.  After clipping, all parameter gradients are rescaled so their L2 norm does not exceed 1.0.

---

## Training Loop

Each training step:

1. Load batch `(x, y)` from DataLoader — both shape `(batch, seq_len)`
2. `optimizer.zero_grad()`
3. Forward pass: `logits, loss = model(x, targets=y)`
4. `loss.backward()` — compute gradients
5. `clip_grad_norm_(...)` — clip gradients
6. `optimizer.step()` — update parameters
7. `scheduler.step()` — update learning rate

---

## Checkpointing

Checkpoints are saved as PyTorch `.pt` files containing:
- `model_state_dict` — all model parameters
- `optimizer_state_dict` — optimizer momentum buffers
- `scheduler_state_dict` — scheduler state
- `global_step`, `current_epoch` — training progress
- `best_val_loss` — for selecting the best checkpoint
- `model_config` — hyperparameters for reconstruction

The **best model** (lowest validation loss) is saved separately as `best_model.pt`.

---

## Monitoring

During training, track:

| Metric | Frequency | Interpretation |
|---|---|---|
| Training loss | Every step | Should decrease steadily |
| Validation loss | Every eval_interval steps | Should decrease; divergence = overfitting |
| Perplexity | Every eval_interval | exp(loss); lower is better |
| Learning rate | Every step | Follows schedule |
| Gradient norm | Every step | Should be ≤ max_grad_norm after clipping |

### Reading Loss Curves

- **Both losses decreasing** — healthy training
- **Train loss decreasing, val loss flat/increasing** — overfitting (reduce model size or add dropout)
- **Both losses not decreasing** — learning rate too low, or model too small
- **Loss spikes** — learning rate too high or data issue

---

## Perplexity as a Training Metric

$$\text{PPL} = \exp(\mathcal{L})$$

Starting perplexity for a character-level model with vocab_size=65:
- **Random model:** PPL ≈ 65 (uniform over vocabulary)
- **After some training:** PPL 20–40 (good)
- **Well-trained:** PPL 5–15 (state of the art for domain-specific data)

Perplexity is bounded below by the true entropy of the language — even a perfect model cannot achieve PPL = 1 because natural language has genuine uncertainty.

---

## LoRA Fine-Tuning Track

In addition to the from-scratch training pipeline above, BB8 includes a second
track that fine-tunes a pretrained model using LoRA (Low-Rank Adaptation).
This is handled by `fine_tune.py` and uses HuggingFace Transformers + PEFT.

The key differences from the from-scratch pipeline:

| Aspect | From-scratch (`train.py`) | LoRA (`fine_tune.py`) |
|---|---|---|
| Model | My own BB8LM implementation | Pretrained Qwen2.5-0.5B |
| Trainable params | All parameters | Only LoRA adapter (~1.75%) |
| Tokenizer | My char/word/BPE tokenizers | HuggingFace AutoTokenizer |
| Loss masking | All tokens | Assistant tokens only (`labels=-100` for prompts) |
| Dataset | Plain text windows | Instruction/response pairs (JSONL) |
| Precision | float32 or float16 | bfloat16 |

Both tracks use AdamW, cosine LR scheduling, gradient clipping, and the same
experiment tracking system (JSONL registry with dataset hashes and Git info).

The from-scratch track (v001–v007) demonstrates understanding of the
architecture. The LoRA track (v004, v005, v008) demonstrates the industry
workflow for adapting pretrained models on limited hardware.
