# Chat diagnostics v1

This suite investigates why BB8 v004 sometimes answers a clear question correctly
and then repeats an earlier answer or produces an unrelated response. It contains
**80 cases**, each with an ID, hypothesis, input, and check or review rubric.
It is a development diagnostic set. Several cases deliberately reproduce prompts
already tried in the UI; scores must not be described as independent test accuracy.

## The experiments

| IDs | Cases | Question under investigation |
|---|---:|---|
| arithmetic-01–10 | 10 | Does the model handle basic operations, negatives, and terse versus explicit questions? |
| knowledge-01–10 | 10 | Can it produce factual answers when the requested answer format is explicit? |
| instructions-01–10 | 10 | Does it obey output constraints such as exact text, JSON, capitalization and bullet count? |
| grounding-01–10 | 10 | Does it use provided records correctly, including facts that conflict with a familiar association? |
| ambiguity-01–10 | 10 | Does it clarify ambiguous inputs and avoid inventing unavailable personal or live information? |
| identity-01–10 | 10 | Does it greet users, describe itself appropriately, and avoid copying or confusing the identity instruction? |
| conversation-01–10 | 10 | Does it track references, recover from incorrect history, and avoid copying previous answers? |
| context-01–10 | 10 | What happens when useful or irrelevant history exceeds the configured input window? |

The exact cases live in `evaluation/chat_suite.py`. Each run exports the complete
suite as `suite.json` and hashes its canonical JSON representation. A changed
question, expected answer, hypothesis or rubric changes that hash. Preserve v1
and create a new suite version when making substantive changes.

Grounding records are fictional and self-contained. They are not claims about
the portfolio owner. Seven conversation cases use a controlled, predefined
history; three perform real rollouts using each tested model's own previous
responses. The latter reproduce how an early mistake can influence later turns.

## Five comparison conditions

Every case runs in five conditions, producing **400 case-condition results**.
Live conversation cases contain multiple generation calls.

The optional `v004_context_first` follow-up profile puts history before the
latest instruction. Run it separately with `--profiles v004_context_first`;
it is not a serving default. See [recorded findings and fixes](chat_diagnostic_findings.md)
for the completed comparison and follow-up results.

| Profile | Weights | Input format | Selection |
|---|---|---|---|
| v004_current | Qwen base + our adapter | Current BB8 chat wrapper, including identity context | Greedy, repetition penalty 1.1 |
| v004_no_system | Same adapter | Omit the identity paragraph; retain available history | Same greedy settings |
| qwen_base | Same base with adapter disabled | Current BB8 chat wrapper | Same greedy settings |
| qwen_instruct | Official Qwen2.5-0.5B-Instruct | Its own chat template, with equivalent BB8 system instruction | Same greedy settings |
| v004_top_p | Same adapter | Current BB8 chat wrapper | Top-P 0.9, temperature 0.8, penalty 1.1 |

Comparing `v004_current` with `qwen_base` isolates the effect of enabling the
adapter under the same prompt format. It does not claim that this format is the
optimal way to use a base model. Comparing with `v004_no_system` measures the
effect of the identity paragraph and the input tokens it consumes. The Instruct
comparison changes both checkpoint and its appropriate prompt template, so it
measures the complete configuration rather than isolating one training technique.

The default input budget is 256 tokens and the output cap is 64 new tokens.
These reproduce the chat generator's distinction between input truncation and
output length; they are not the lab's rolling-window algorithm. The underlying
Qwen model can process a longer combined sequence than our configured input cap.
The current-profile formatter calls the same `build_chat_prompt` used by `/chat`.
For the alternate wrappers, old complete turns are removed until the input fits.
An oversized latest message is explicitly rejected.

All prompts, effective input IDs, generated IDs, generation settings and seed
are saved. For sampling, the seed is `run seed + full-suite case index * 10 +
rollout turn index`. One sampling run is one observation; it cannot establish
an expected sampling success rate. Repeat with additional seeds before drawing
conclusions about stochastic reliability.

## How results are judged

There are **57 automatically checked cases and 23 requiring review** per profile.
One automatic case tests input rejection; it is an API/prompt-policy check rather
than a test of language knowledge.

Automatic statuses are:

- `pass`: the specified check passed.
- `flag`: the generated output did not satisfy the check; inspect it manually.
- `error`: an unexpected runtime failure, or input rejection where not expected.
- `review`: assess the response against the case's written rubric.

Exact-answer checks normalize whitespace, case, and trailing sentence punctuation.
They deliberately reject a correct keyword embedded in an incorrect claim. They
also flag some semantically correct explanations, so **check pass rate is not
semantic answer accuracy**. Capitalization and JSON tasks use stricter checks.
Bullet and word-count checks verify format; they do not prove factual quality.

For memory cases, the expected answer depends on whether the supporting sentence
is actually present in the formatted prompt. If trimming removed it, `UNKNOWN`
is the correct response. This distinguishes a model failure from information the
application no longer provides.

For manual review, use 0 = failure, 1 = partially satisfactory, 2 = satisfactory
against the written rubric. Record a reason and distinguish factual mistakes,
irrelevance, identity confusion, unsupported claims, repetition, and missing
context. For live rollouts, inspect every turn, not only the final answer. Review
scores from one reviewer are exploratory; a stronger evaluation would use an
independent second reviewer and examine disagreements.

Report categories separately. Preserve failure examples and do not hide an
arithmetic regression behind improvements in greetings. A new frozen test set
will be needed after using these diagnostics to choose improvements. Neither this
suite nor public questions can be guaranteed absent from Qwen's pretraining data.

## Run locally

From the repository root, with the existing environment and cached checkpoints:

```powershell
.\.venv\Scripts\python.exe -m evaluation.run_chat_diagnostics --device cuda
```

All model loading is offline. The runner uses the pinned Qwen base revision from
the adapter metadata and the cached Instruct revision
`7ae557604adf67be50417f59c2c2f167def9a775`. It does not launch training, change the
serving checkpoint, or call an external generation API. The local server can
remain running, although concurrent GPU work can affect latency.

For a smaller investigation:

```powershell
.\.venv\Scripts\python.exe -m evaluation.run_chat_diagnostics --device cuda --profiles v004_current v004_no_system --case-ids arithmetic-01 conversation-08 --skip-data-audit
```

Use `--seed` for another sampling run. The selected cases, profiles and settings
are explicitly recorded; a subset must not be reported as a complete 80-case run.

Each invocation creates a unique directory under `outputs/chat_diagnostics/`:

- `manifest.json`: source, suite and artifact hashes; base revisions; settings;
  Git commit/dirty flag; libraries and hardware.
- `suite.json`: the complete case definitions used by the run.
- `results.jsonl`: flushed after each case, with actual inputs, outputs and checks.
- `training_audit.json`: optional reconstruction of v004 training truncation.
- `summary.json`: final counts, categories and timing summaries.
- `report.md`: readable results and responses grouped by case.

The compact `experiments/chat_diagnostics_registry.jsonl` records completed runs.
It is separate from the trained-model registry: changing an inference setting
does not create a new trained model. Raw outputs are Git-ignored, so preserve the
run folder separately when archiving an experiment. Hashes detect changes but do
not preserve uncommitted code; commit or archive the source as well.

Timing is measured after a warm-up, with sequential generation and batch size
one. It includes the full generation call and GPU synchronization, excludes
HTTP transport and prompt preparation, and reports actual generated token counts.
It is a local diagnostic measurement, not a cloud load test or time-to-first-token
benchmark. A length stop flags a potentially incomplete response.

## Training preprocessing audit

The audit verifies Dolly's hash against the original v004 training record and
reuses `InstructionDataset` to reconstruct every example. It counts prompt-prefix
truncation, removed end-of-text tokens, and truncation missed by the historical
counter. Prefix truncation removes the complete response marker because it lies
at the end of the formatted prompt. This identifies a data-processing issue;
whether fixing it improves chat behavior must be tested with a newly versioned
training run.

Useful follow-up work is driven by the measured differences: prompt formatting
if the identity ablation improves results; adapter/data work if disabling the
adapter improves them; conversation formatting if single questions work but
history causes failures; and context policy if required facts are removed.
