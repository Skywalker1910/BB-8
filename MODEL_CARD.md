---
language:
- en
license: mit
tags:
- transformers
- pytorch
- language-model
- from-scratch
- lora
- educational
- gpt
- decoder-only
datasets:
- tiny_shakespeare
- databricks/databricks-dolly-15k
pipeline_tag: text-generation
---

# BB8 — Transformer Language Model Built from Scratch

A GPT-style decoder-only Transformer implemented from scratch in PyTorch, plus LoRA fine-tuning experiments on pretrained Qwen models. This is a graduate-level learning project exploring LLM internals across 11 experiments.

## What's in This Repo

This Hugging Face repository contains all trained checkpoints, configs, and experiment results from the BB8 project. The source code lives at [github.com/Skywalker1910/BB8](https://github.com/Skywalker1910/BB8).

### Checkpoints

| Directory | Type | Parameters | Data | Val PPL | Description |
|---|---|---:|---|---:|---|
| `checkpoints/bb8-char-small-v001` | From scratch | 112K | Shakespeare | 5.32 | Historical baseline (had data leak) |
| `checkpoints/bb8-char-small-v002` | From scratch | 112K | Shakespeare | 5.40 | Corrected baseline |
| `checkpoints/bb8-char-medium-v006` | From scratch | 833K | Shakespeare | 9.57 | Scaling study (4 layers) |
| `checkpoints/bb8-char-medium-v006b` | From scratch | 1.2M | Shakespeare | 8.24 | Depth ablation (6 layers) |
| `checkpoints/bb8-bpe-shakespeare-v007a` | From scratch | 5.1M | Shakespeare | 83.83 | BPE vocab=1000 |
| `checkpoints/bb8-bpe-shakespeare-v007` | From scratch | 5.6M | Shakespeare | 355.01 | BPE vocab=3000 (best text) |
| `checkpoints/bb8-bpe-instruct-v003-dev` | From scratch | 5.1M | Dolly 15K | 19.10 | Instruction from scratch (failed) |
| `checkpoints/bb8-qwen-lora-v004-dev` | Qwen + LoRA | 8.8M trained | Dolly 15K | 7.97 | First pretrained adapter |
| `checkpoints/bb8-qwen-instruct-v008a` | Qwen-Instruct + LoRA | 4.4M trained | Dolly 15K | 7.81 | Quick test (rank=8) |
| `checkpoints/bb8-qwen-instruct-v008` | Qwen-Instruct + LoRA | 8.8M trained | Dolly 15K | 7.86 | Best chat model |
| `checkpoints/bb8-grounded-v005-pilot` | Qwen-Instruct + LoRA | 540K trained | Portfolio | 1.03 | Grounded retrieval pilot |

### Outputs

Each `outputs/<run-name>/results.json` contains the full training history, evaluation metrics, config snapshot, and generated text samples.

## Architecture (From Scratch)

The core BB8 model is a decoder-only Transformer with:
- Learned positional embeddings
- Pre-LayerNorm (normalize before attention)
- GELU activation in feed-forward layers
- Weight-tied LM head (shared with token embedding)
- Cosine LR schedule with linear warmup
- Custom character, word, and BPE tokenizers

All implemented from scratch in PyTorch — no HuggingFace model classes.

## Key Results

- **v002** (112K params): Val PPL 5.40 on Shakespeare — learns character patterns
- **v006b** (1.2M params): 14% PPL improvement over v006 — proves depth > width
- **v003** (5M params, from scratch on Dolly): All smoke tests wrong — can't skip pretraining
- **v004** (Qwen + LoRA): Same data as v003, answers correctly — pretraining matters
- **v008** (Qwen-Instruct + LoRA): Best chat model, all smoke tests correct

## Hardware

All experiments ran on a single NVIDIA RTX 4070 Laptop GPU (8 GB VRAM).

## License

MIT — code and model weights are free to use, modify, and build upon.

## Citation

```
@misc{bb8-transformer,
  author = {Aditya More},
  title = {BB8: Transformer Language Model Built from Scratch},
  year = {2026},
  url = {https://github.com/Skywalker1910/BB8}
}
```
