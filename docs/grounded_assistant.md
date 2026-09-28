# BB8 grounded assistant: a controlled engineering experiment

Completed first-run results and release decision: [v005 pilot report](grounded_results_v005.md).

## Contribution and research question

Can a small local model answer questions about documented projects, cite its
evidence, retain useful conversation context and abstain when information is
missing? Does targeted LoRA training improve this behavior beyond retrieval alone?

Original project work includes the evidence pipeline, deterministic retriever,
dataset/split design, training preprocessing, LoRA experiment, evaluation harness,
versioned artifacts, serving integration and error analysis. Qwen's pretrained
weights and upstream instruction tuning are **not** original BB8 work. The older
from-scratch Transformer remains a separate educational track.

This is a small graduate-project pilot, not a production-ready assistant or an
independently validated benchmark. No production portfolio or cloud resources
are modified by this workflow.

## 1. Task definition and acceptance criteria

Supported: questions about facts in the frozen project corpus, evidence-supported
follow-ups, concise cited responses, and an explicit insufficient-evidence answer.
Unsupported: private personal details, live status/billing, undocumented work
history, tools/actions, or claims of independent verification of portfolio results.

Before training, `data/grounded_v1/manifest.json` freezes these promotion gates:

- At least 80% of answerable cases pass the keyword + correct-citation checks.
- At least 90% of unanswerable cases use the defined abstention response.
- No citation IDs outside the supplied evidence.
- LoRA + retrieval must not regress versus the unchanged model + retrieval.
- Human claim-level review is required; passing numeric gates never auto-deploys.

These are pilot engineering criteria, not a promise of factual accuracy. The
keyword check can miss semantic errors and reject correct paraphrases. Citation
ID validity does not establish that every sentence is supported by the source.
The v1 abstention metric detects the required phrase; alternate valid refusals
can fail it, and a phrase followed by an unsupported claim would need manual
rejection. It is a formatting/behavior proxy, not a safety guarantee.

## 2. Approved public evidence

The user authorized the public portfolio and GitHub source. The corpus contains
12 short, manually curated evidence chunks: eight BB8 repository facts and four
public portfolio facts. Each chunk has an ID, title, source path/URL and source
hash. These fact-sized chunks avoid arbitrary text cuts and are independently
retrievable; a large portfolio will need a more general ingestion/chunking stage.

Public sources:

- [Aditya's portfolio](https://www.adityamore.dev/), fetched on 2026-09-27.
- [Pinned Tech-Portfolio README](https://github.com/Skywalker1910/Tech-Portfolio/blob/a17142dbbbc447b08a6072aad40972385cced616/README.md).

Public source snapshots are retained with the dataset. Portfolio-reported metrics
are labelled as reported, not independently reproduced. Content was curated by
the coding assistant; the owner still needs to review the corpus and gold labels
before using the results in public claims. No private account or résumé files
were accessed, and no claims were inferred from model responses.

## 3. Frozen splits and leakage controls

| Split | Size | Purpose |
|---|---:|---|
| Train | 192 | Synthetic evidence use, citations, abstention, follow-ups and malicious evidence |
| Validation | 48 | Different fictional project/owner groups; monitor training loss |
| Reserved evaluation | 88 | 48 direct, 12 follow-up, 12 evidence-injection and 16 unanswerable cases |

The model does not train on the real evaluation questions or answers. Synthetic
training entities, validation entities and real evaluation facts are distinct.
Training and validation share template families, so validation loss is an easier
test than generalization to real portfolio questions. The corpus is deliberately
available to retrieval during evaluation: that is the task, not leakage into
weights. This is an agent-authored development suite, not a blinded external test.
After these results are inspected, future tuning needs a new untouched final set.
The 88 cases are correlated variants around 12 fact groups, not 88 independent
knowledge domains. The modest corpus also makes retrieval much easier than a
large real-world knowledge base.

The generator refuses to overwrite an existing dataset version. File hashes are
verified before training and evaluation. For new source material, create a new
version instead of editing `grounded_v1` or regenerating its snapshots.

## 4. Retrieval and controlled comparison

`grounded/core.py` implements a small BM25-style lexical retriever with transparent
term frequencies, inverse document frequencies, length normalization, stopwords
and deterministic tie-breaking. No vector database or external model API is
required. The top three positive-scoring chunks are supplied as evidence.

Follow-up retrieval includes the last two user questions, but not model-generated
claims. Old turns and complete evidence chunks can be removed to fit the budget;
the latest question and system prompt are never silently sliced.

The 2x2 comparison keeps the system prompt, native chat template, corpus, questions,
greedy decoding, repetition penalty 1.0, 768-token input budget and 96-token output
cap fixed:

| Weights | No retrieved evidence | With retrieved evidence |
|---|---|---|
| Unchanged Qwen Instruct | `base_no_rag` | `base_rag` |
| Our LoRA adapter | `lora_no_rag` | `lora_rag` |

No-evidence conditions intentionally cannot satisfy the cited-answer contract;
their purpose is to measure abstention and unsupported answering, **not** to
measure Qwen's general knowledge. Retrieval recall, keyword content matches,
correct gold citations, invalid citations, abstention and latency are reported
separately. The keyword-only count was added as a diagnostic after observing
missing citations in the first baseline; the primary acceptance check was unchanged.
Injection cases append an adversarial instruction to evidence. They are a narrow
stress test, not a comprehensive security evaluation. Follow-up cases use controlled
history rather than a live multi-turn rollout; live serving smoke tests supplement them.

## 5. Corrected preprocessing and v005 pilot training

Historical v004 used prefix truncation that could remove the response marker.
That behavior remains explicitly available as `legacy_v1` for reproducing its
dataset preview/audit. New Dolly preprocessing defaults to `full_example_v2`:
retain a complete example or explicitly skip it, preserving the EOS token.

The new `chat_full_v1` training path uses Qwen Instruct's own template. It checks
that the assistant prefix matches, masks all prompt labels, keeps the assistant
response/end-of-turn tokens, and records every skipped example. It never inserts
a premature EOS into a cut response. All 192 pilot training examples fit.

`configs/qwen_grounded_v005.yaml` explains the reproducible run settings:

- Pinned Qwen2.5-0.5B-Instruct revision; this is a new branch from Instruct,
  not continued training of v004's base-model adapter.
- LoRA rank 8, alpha 16 on query/value projections: a small-capacity, low-memory
  initial adaptation. These are starting hypotheses, not optimized choices.
- Learning rate 5e-5, dropout 0.05, three epochs: a conservative small-data pilot.
- Batch size 2 with four-step accumulation, seed 42, BF16 on the local CUDA GPU.
- Final-epoch adapter is saved; no held-out evaluation-based checkpoint selection.

Training logs include hashes, config/source snapshots, preprocessing counts,
validation loss, runtime and adapter lineage. The adapter gets its own immutable
run name and never overwrites v004 or the external baseline.
The exact local package versions for this pilot are recorded in
`experiments/grounded_environment_20260927.json`. Public snapshot file hashes are
also recorded in `data/grounded_v1/snapshot_integrity.json` (the stored README has
Windows line endings; its fetched-text hash is separately preserved).

## 6. Reproduce and inspect

Run from the repository root with the existing fine-tuning environment and cached
Qwen Instruct revision. These commands use the local GPU, not paid cloud training:

```powershell
# Only when initially creating a new version (requires public-source network access):
python -m grounded.data

# Baseline evaluation:
python -m grounded.evaluate

# A fresh training run needs a unique name; do not overwrite the completed pilot:
python fine_tune.py --config configs/qwen_grounded_v005.yaml --name bb8-grounded-v005-pilot

# Four-way evaluation:
python -m grounded.evaluate --adapter checkpoints/bb8-grounded-v005-pilot

python -m pytest tests -q
```

Evaluations save raw prompts, retrieved/visible IDs, answers, generated token IDs,
checks, per-case latency, dataset/model provenance, source snapshots and a readable
report under `outputs/grounded/<run-id>`. Compact comparison summaries are appended
to `experiments/grounded_registry.jsonl`. The first failed offline-loading attempt
is not a completed experiment and has no result record.

For the local UI, start with `--chat-models configs/chat_models.json`, select a
**Grounded** model, and use **Portfolio evidence + citations**. Selecting a grounded
model sets the evaluated greedy/penalty/output-cap defaults. Expand the evidence
under each answer to check it. Missing or invalid citations are warned about,
not silently repaired. The application does not replace generated answers with
canned factual answers. Training learns behavior; retrieval supplies project facts.

## Remaining production work

Human source/label review, a larger and less templated training set, an untouched
final evaluation set, live conversation/security tests, latency/load testing,
cloud artifact packaging and monitoring, and explicit release approval remain.
Do not describe the experimental adapter as promoted or deployed to production.
