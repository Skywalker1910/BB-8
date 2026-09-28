# BB8 chat pipeline investigation — 2026-09-27

Run: `20260927T172145Z-459f670f`.
Suite: `bb8-chat-diagnostics-v1`, SHA-256
`c48263e27fd25f24e70e3d7260774be6c536572cb811e879e98a48be7b1c033e`.
See [the protocol](chat_diagnostics.md) for case definitions and scoring limits.

## Completed comparison

All 80 cases ran against each of five configurations (400 case results).
Each configuration has 57 automatic checks and 23 cases requiring qualitative
review. Automatic passes include answer-format checks and one expected rejection
of an oversized input; these numbers are **not general factual accuracy**.

| Configuration | Automatic passes / 57 | Automatic flags | Qualitative cases |
|---|---:|---:|---:|
| BB8 v004, current wrapper, greedy | 26 | 31 | 23 |
| BB8 v004, identity paragraph omitted | 31 | 26 | 23 |
| Qwen base, adapter disabled, current wrapper | 2 | 55 | 23 |
| Official Qwen Instruct, native chat template | 48 | 9 | 23 |
| BB8 v004, current wrapper, Top-P | 21 | 36 | 23 |

The base-only result measures this particular instruction wrapper, not all of
Qwen base's capabilities. Instruct changes both weights and formatting; it is
an external comparison model, not a new BB8 training run. It still made errors
on subtraction, operator precedence, translation and identity/capability claims.
Raw outputs, prompts, generated token IDs, settings, artifact hashes and runtime
information are in `outputs/chat_diagnostics/20260927T172145Z-459f670f/`.
Compact summaries are tracked in `experiments/chat_diagnostics_registry.jsonl`.

## Follow-up format experiment and implemented serving changes

Run `20260927T174344Z-9e6db3d8` tested all 80 cases with v004's history placed
before the latest instruction. It passed 29/57 automatic checks, versus 26/57
originally. Its arithmetic rollout improved from `4, 4, 4` to `4, 6, -2`, but
the final subtraction was still incorrect. This format remains an explicit
diagnostic ablation (`v004_context_first`), not the default serving format.
There are now 480 recorded case results across the two runs.

Implemented changes:

- The chat page can select the original model or the official
  `Qwen/Qwen2.5-0.5B-Instruct` baseline, pinned to revision
  `7ae557604adf67be50417f59c2c2f167def9a775`. It loads locally from the existing
  cache; it does not call an external generation API. The baseline is clearly
  labelled as external, not added to the locally trained model registry.
- Instruct uses its native role-aware chat template. Both models retain the
  same 256-token input cap for this comparison. Older complete turns are removed
  when necessary; an oversized latest message is rejected, not silently cut.
- Model selection uses operator-configured aliases, not arbitrary request paths.
  Changing models clears UI history. Model/context/reset controls are disabled
  during a request so responses cannot land in a different conversation.
- Generation reports actual output token counts (including a generated EOS) and
  `eos` versus `length` stop reasons. It does not estimate counts by retokenizing
  displayed text. Legacy generators lacking telemetry return unknown/null.
- HF sampling explicitly disables inactive Top-K/Top-P filters instead of
  inheriting potentially hidden checkpoint defaults. All checkpoint EOS IDs
  are respected.

The restarted local API was checked with the baseline and conversation history:
`2 + 2 → 4`, `3 + 3 → 6`, `4 - 2 → 2`. Each response contained two generated
tokens including EOS, rather than the requested cap of 64. This small smoke
test validates serving integration; it does not replace the 80-case comparison.
The Python regression suite passes 110 tests. Browser visual/interaction QA
remains unverified in this session; API behavior and static script syntax were
checked separately.

### Try the updated chat

Start from the repository root:

```powershell
.venv\Scripts\python.exe -m api.local_server --model-dir checkpoints\bb8-qwen-lora-v004-dev --chat-models configs\chat_models.json --api-key local-test-key --device cuda --port 8000
```

Open `http://127.0.0.1:8000/chat`, refresh the page, choose **Qwen Instruct —
external baseline**, choose **Conversation**, and initially use **Greedy**.
The selector retains the configured BB8 model by default so switching to an
external baseline is explicit. The token/dataset lab continues to show the
locally trained models; its catalog is unchanged.

No checkpoint weights were modified and no new training was started. The next
training task is to version and correct the preprocessing described below,
train a separate adapter, and evaluate it against both unchanged baselines.

## What BB8 actually generates

BB8 v004 performs local autoregressive generation. The application formats the
question and selected history, tokenizes that text, and runs Qwen2.5-0.5B with
the LoRA adapter. The model produces a score for every possible next token;
decoding chooses one token, appends it, and repeats until a stop token or output
limit. The response is then decoded and cleaned for display.

This is the same basic generation process used by modern generative language
models. There is no live Dolly lookup, external generation API, calculator, or
portfolio search in the current chat path. A correct answer emerges from the
model's learned weights and the text visible in its context. Changing the full
pipeline can therefore improve the answer even when weights stay fixed.

## Observed failures and controlled comparisons

The current wrapper puts the newest question under `### Instruction`, followed
by `### Context` containing an identity paragraph and earlier conversation.
The Dolly fine-tuning examples use the context field for supporting text. Our
chat representation consequently asks the model to interpret system identity
and dialogue as that supporting text.

With every other generation setting fixed, removing the identity paragraph
changed these outputs:

| Input | Current wrapper | Identity paragraph omitted |
|---|---|---|
| `2 + 2` | `You are BB8` | `4` |
| `3 + 3` | `You are BB8` | `6` |
| `4 * 4` | `You are BB8` | `16` |

This is evidence that the wrapper affects the failures. It is not a complete
solution: without the paragraph, `4 - 2` produced `1`, and the requested
`7 times 8` produced `64` in both configurations. Removing identity also allowed
an invented origin story when asked who the model was.

The current profile's `2 + 2` output was independently reproduced through the
running `/chat` API using the same settings. The diagnostic engine is exercising
the serving behavior, rather than the lab's separate sampler.

### Conversation copying happens before the window fills

Given controlled history containing `2 + 2 → 4`, the current model answered a
new `3 + 3` question with `The answer to 2 + 2 is 4`. The input used only 69 of
256 tokens. Similarly, after a deliberately incorrect history answer of `5`,
it repeated `5` when asked to check `2 + 2` again.

In a live three-question arithmetic rollout, it returned the first answer and
then repeated `4` for the questions that required `6` and `2`. Removing the
identity paragraph did not fix this rollout. These examples cannot be explained
by the newest question falling outside the context window; the stored input
traces retain it. Dialogue handling and learned instruction behavior need work.

### Supplied facts often help

The current configuration correctly used several fictional records: it selected
Python for Project Harbor, Ada as the builder of Project River, and Lyon as the
capital explicitly stipulated by a fictional story. It also produced correct
content for the bus count, access code and cheaper item, although explanatory
phrasing failed their strict answer-format checks.

This is useful evidence for testing portfolio retrieval as a future pipeline
improvement. It is not yet a retrieval evaluation: these records were supplied
directly in the prompt, and the application has no portfolio retrieval component.

## Measured training preprocessing issue

The audit verified the local Dolly file against v004's original dataset hash
and reconstructed all 15,011 examples with `InstructionDataset`.

| Observation | Records | Share of dataset |
|---|---:|---:|
| Prompt prefix truncated, losing the complete response marker | 3,638 | 24.2% |
| End-of-text token removed by response truncation | 1,173 | 7.8% |
| Any token truncation | 4,361 | 29.1% |
| Truncated examples missed by the historical counter | 1,296 | 8.6% |

The prompt is capped at half the 256-token training budget even when a short
response leaves spare room. The old counter only checks whether the original
combined length exceeds 256. It therefore misses examples whose prompt was cut
despite the complete example fitting in the total budget.

This is a concrete preprocessing defect and a plausible contributor to behavior.
Its effect on answer quality has not been isolated: that requires a new adapter
trained with corrected preprocessing, compared with the same baseline.

## Measurement issue found in the original API (now fixed)

The original `/chat` response reported the requested output cap as
`generated_tokens`. In the verified `2 + 2` request, it reported 64 even though
the output sequence contained five tokens including EOS. The diagnostic runner
records the actual generated IDs and their count. The updated serving API now
reports the actual count as well; older saved API measurements remain incorrect.

## Interpretation limits

These are development diagnostics built around observed weaknesses. Exact checks
can flag a semantically correct answer that violates a requested format. One
sampling seed does not estimate reliability across random draws. Qwen Instruct
uses its own appropriate template, so that comparison changes both weights and
format. Timing uses the local shared GPU, not a dedicated deployment benchmark.
Qualitative assessments are made by one AI reviewer and require independent
human review before being treated as validated capability scores.
