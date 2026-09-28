# BB8 Model Lab

The lab makes next-token inference inspectable. It uses the real local model
weights, not mock probabilities, and does not retrain or improve those weights.
The original chat interface remains available at `/chat`; the lab is `/lab`.

## Start and explore

From the repository root, using the project's Python environment:

```powershell
python -m api.local_server --model-dir checkpoints/bb8-qwen-lora-v004-dev --api-key local-test-key --device cuda
```

Open <http://127.0.0.1:8000/lab>. If you use a different key, enter it and click
**Connect / reload models**. Use `--device cpu` when CUDA is unavailable.

The lab and chat use a light theme. Each generation control has an **i** help
button: hover or focus it to read the explanation, or tap/click to keep it open.
Press Escape or click outside to dismiss it. Help describes the control and the
effect of changing it; it remains available when a parameter is inactive.
The strategy note explains which settings currently apply. The lab also explains
entropy and probability columns. Example prompt buttons load a factual question,
arithmetic, an ambiguous topic, or raw continuation for a controlled comparison;
they clear the current unsaved continuation and wait for **Inspect prompt**.

Registered models with local checkpoint directories appear in the dropdown;
the first use loads them on the server's device. Models are cached until restart.
For Qwen, install `requirements-finetune.txt` and retain its pinned base model
in the Hugging Face cache. Initial loading can download that base if missing.

1. Select a model and enter `The capital of France is` in raw continuation mode.
2. Click **Inspect prompt**. Expand the exact formatted input to see token IDs.
3. Observe the model, adjusted, and selection probabilities for the next token.
4. Change temperature or Top-P and click **Recalculate next token**. This holds
   the context fixed so the distributions can be compared.
5. **Generate one token** appends the draw. **Choose** deliberately forces a
   candidate, even a filtered-out token. Manual steps are marked in gold.
6. Click a generated token to discard its later continuation and branch from
   that point, or use **Undo token**. A run ends at EOS or 64 generated tokens.
7. Give the run a label and click **Verify & save**. Use **Experiments** for A/B
   comparison, replay, setup reuse, and JSON export.

Prompt edits and model changes reset the unsaved continuation. Save a branch
before switching away if you want to keep it. API keys stay in the page and are
not included in saved experiment JSON. Prompts and outputs *are* saved in plain
text; avoid sensitive data.

## What the numbers mean

For each generation step the server runs a model forward pass over the visible
token IDs and obtains the final position's logits (one score per vocabulary ID).
The named sampler `bb8-lab-v1` then applies:

| Stage | Operation |
|---|---|
| Model probability | Softmax of raw logits, before any decoding controls |
| Repetition penalty | For IDs already in the visible context: positive logits divided by the penalty; negative logits multiplied by it |
| Temperature | Divide penalized logits by temperature; skipped for greedy |
| Adjusted probability | Softmax after penalty and temperature, before filtering |
| Top-K | Retain the K highest adjusted scores |
| Top-P | Retain the smallest probability-ranked prefix whose cumulative mass reaches P, including the crossing token |
| Selection probability | Renormalize retained scores, then sample proportionally to probability |
| Greedy alternative | Select the highest penalized score; selection distribution is one-hot |

Only the selected strategy's K or P filter is applied; they are not combined.
Temperature-only sampling retains the full vocabulary. Ranking ties for filters
use stable ordering. Calculations and sampling use float32 on CPU after the
model forward pass, with a local generator seeded by `seed + zero-based step`.
This avoids mutating the application's global random generator.

The table includes the 15 highest raw-probability IDs, up to 15 highest nonzero
selection-probability IDs, and the selected ID. **Other tokens** accounts for the
remaining probability mass; visible rows are not renormalized to sum to 100%.
Entropy is calculated over the full distribution in bits. Lower entropy means
more concentrated predictions, not greater factual accuracy. Manual choices do
not change the displayed probability; a forced token can have selection
probability zero.

Token pieces are tokenizer representations, not always complete words or valid
standalone Unicode text. Hover for the single-token decoding. Output is decoded
from the full generated ID sequence, not concatenated single-token strings.
EOS ends generation and can be visible as a special-token piece.

### Context and prompt formatting

**Raw** mode tokenizes exactly the entered text without automatically adding
special tokens. **Instruction / chat wrapper** uses the selected model's current
BB8 conversation formatter for a single user question, including its system
instruction. It is not automatically the upstream Qwen-Instruct chat template:
v004 adapted the Qwen **base** model using the repository's instruction format.
Oversized wrapped questions are rejected by that formatter.

The lab limits the forward pass to the last N token IDs where N is your context
control, up to the checkpoint's configured limit. As new tokens arrive, older
tokens fall out. The UI shows the exact remaining context, including any loss of
instructions. It recomputes positions and the full visible sequence each step,
without a KV cache. This is a controlled rolling-window experiment, **not** the
same context-management behavior or performance as `/chat`.

The lab intentionally has a separate versioned sampler. It does not promise
bit-for-bit parity with the historical native generator or Transformers
`generate`, which have different implementation details and defaults.

## Inspect the training data

**Dataset** browses local `data/databricks-dolly-15k.jsonl`, with search, category,
and v004 train/validation filters. The manifest identifies its source and
CC BY-SA 3.0 license. The dataset is fine-tuning data for v004, not Qwen's entire
pretraining corpus. Generation does not retrieve a matching Dolly record.

The dataset hash must match v004's recorded training run. Split membership is
reconstructed by shuffling raw row indices with that run's seed and taking its
recorded validation count from the beginning (14,260 train / 751 validation).
It does not claim these are v003's splits.

Selecting a record shows raw JSON, the full formatted example, the reference
response, and the exact token/label pairs produced by the current
`fine_tune.InstructionDataset` implementation used for v004:

- Prompt prefix retained up to half of the 256-token training budget.
- Response plus EOS retained up to the remaining budget.
- Prompt labels `-100` (ignored by the loss); response labels are token IDs.
- The model internally shifts the causal labels by one position.

Warnings expose truncated prompts, missing response markers, and removed EOS.
This is a diagnostic view of an existing preprocessing limitation, not a fix.
If preprocessing code changes for a future training version, preserve a versioned
v004 reconstruction before claiming its preview is still historical.

**Inspect this training prompt** loads the full prompt/context without the target
answer. The lab's rolling inference truncation is deliberately distinguished from
the training-time prefix truncation above. Inspecting a training example is not
an independent generalization evaluation; validation examples are labelled.

## Experiment records and reproducibility

The server recomputes every saved step and checks the generated IDs against the
browser preview before accepting the record. It writes a new UUID-named JSON
file using exclusive creation under `outputs/lab_runs/`; replay creates another
record and never overwrites the original. These are **inference experiments**,
not new trained-model versions or a replacement for the training registry.

Each record includes:

- Original and formatted prompt, model name, input and generated token IDs.
- Per-step decoding settings, seed, context budget, and manual interventions.
- Candidate probability traces, entropy, visible context, output, engine version.
- Selected weight, tokenizer, and available configuration file SHA-256 hashes.
- Inference/model/tokenizer source hashes, Git commit and dirty flag.
- Original training registry record, including dataset identity where recorded.
- Python, PyTorch, relevant installed library versions, dtype, device, and GPU.

For LoRA, local adapter files are hashed and the base model's pinned repository
revision is recorded in hashed metadata. The lab does not hash every upstream
base-weight cache shard. Preserve the pinned base and local artifacts for replay.

Replay requires matching artifact/source hashes and compares every generated ID.
Hardware, library changes, and numerical nondeterminism can still change results;
a match is measured, not guaranteed. A dirty Git flag and hashes detect changes
but do not archive uncommitted source: commit or preserve it separately.
`outputs/` is Git-ignored; export or back up records you want to keep. This local
file workflow is intentionally simple, not a durable cloud model registry.

## Small studies worth reporting

Keep the checkpoint, input, prompt mode, context window, and seed fixed unless
they are the variable under study. Record a hypothesis before each comparison.

| Question | Controlled experiment |
|---|---|
| How does temperature affect uncertainty? | Compare 0.3, 0.8, and 1.5 at one unchanged context; record entropy and top-token mass |
| What does Top-P remove? | Compare P=0.5 and P=0.95 with fixed temperature; observe eligible count and renormalization |
| Why do continuations diverge? | Save a run, branch at one token, force an alternative, then keep later settings fixed |
| What happens when context fills? | Reduce N on the same sequence; inspect precisely which instruction tokens disappear |
| Did the model learn instruction following? | Use fixed held-out questions across registered models; manually score relevance and correctness, not probability alone |

For quantitative claims, use multiple fixed prompts and seeds, keep held-out
evaluation separate from examples used for tuning, and report failure cases.
Token probabilities, token counts, and perplexities are not directly comparable
across different tokenizers. The legacy v001 metrics also have a known data-leakage
caveat; see `docs/training_results.md`.

## Implementation and boundaries

- `inference/lab.py`: model forward pass and inspectable sampling mathematics.
- `api/lab.py`: registry access, dataset inspection, validation, saved runs/replay.
- `api/lab.html`, `lab.css`, `lab.js`: dependency-free local interface.
- `api/local_server.py`: static assets and local-only lab routing.
- `tests/test_lab.py`: distribution, context, RNG, manual choice, replay, and data-mask checks.

Lab API routes are POST requests under `/lab/`: `catalog`, `prepare`, `step`,
`dataset`, `example`, `runs`, `run`, `save`, and `replay`. They use the same local
`x-api-key` authentication as chat. Models must be in the registry; clients cannot
provide arbitrary checkpoint paths. Saved run IDs are validated UUID hex strings.
Requests are limited to 1 MB, prompts to 8,000 characters / 4,096 encoded tokens,
and runs to 64 generation steps. Inference requests are serialized to avoid
concurrent GPU work. All loaded models remain cached until server restart.

Keep the server bound to its default `127.0.0.1`. The default example key is not
secure. This is not a public multi-user service: there are no quotas, account
isolation, persistent job queue, or cloud storage. Lab routes are **not** added to
the Lambda handler. Cloud/public deployment needs its own security and resource
review. No new training job or cloud spend is triggered by this feature.

Run tests with `python -m pytest -q tests` and check frontend syntax with
`node --check api/lab.js`. Actual browser interaction/layout verification is a
separate check; successful Python tests alone do not establish visual correctness.

### Implementation verification (2026-09-26)

- Full Python suite: **95 passed** (`-p no:cacheprovider` avoids an existing local
  `.pytest_cache` permission warning). Frontend JavaScript syntax check passed.
- Live HTTP checks: lab/chat assets, all four registered checkpoints' next-token
  inference, all four decoding strategies, authentication, invalid IDs, and
  path-traversal rejection.
- On the RTX 4070 Laptop GPU, a saved run containing context cropping, a manual
  intervention, and different strategies replayed with identical generated IDs.
  QA records are labelled in the local Experiments list; they are not model-quality
  benchmarks. Earlier QA records may fail replay after source changes, by design.
- Dataset checks confirmed 751 validation records and the response-only loss mask.
- Automated in-app browser verification was unavailable due to a browser-tool
  connection error. Layout and interactive browser behavior still need a manual
  pass; no screenshot-based verification is claimed.
