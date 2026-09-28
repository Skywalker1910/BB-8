# Grounded BB8 v005: first controlled pilot

Date: 2026-09-27. **Outcome: mixed improvement; not promoted.**

This report completes the first iteration of the six-task workflow described in
[the project protocol](grounded_assistant.md). It is a local research/engineering
experiment, not a claim that BB8 is ready for production or that its model is
trained from scratch.

## What was built

1. A defined evidence-only answering task and promotion gates recorded before training.
2. A 12-document, source-attributed corpus from BB8 code, the user-approved public
   portfolio, and a pinned GitHub README; human source/label review is still pending.
3. Frozen 192/48 training/validation examples and 88 reserved evaluation cases.
4. A deterministic BM25-style retriever and a measured unmodified-model baseline.
5. Corrected full-example training preprocessing and a new local LoRA adapter.
6. A four-way comparison with raw traces, immutable run IDs, regression findings,
   and this reproducible report. Nothing was pushed to a remote repository or cloud.

## Run identity and reproducibility

| Artifact | Identity |
|---|---|
| Dataset | `data/grounded_v1/manifest.json` |
| Training run | `bb8-grounded-v005-pilot` |
| Baseline-only evaluation | `20260927T192610Z-703deaff` |
| Four-way evaluation | `20260927T193128Z-3c5c6326` |
| Base checkpoint | `Qwen/Qwen2.5-0.5B-Instruct` |
| Base revision | `7ae557604adf67be50417f59c2c2f167def9a775` |
| Training configuration | `configs/qwen_grounded_v005.yaml` |
| Training registry | `experiments/model_registry.jsonl` |
| Comparison registry | `experiments/grounded_registry.jsonl` |

Run directories under `outputs/` preserve model/dataset provenance, config/source
snapshots and raw outputs. The dataset and compact registry/report files belong
in version control; large checkpoints and generated outputs remain gitignored.
For external reproducibility, publish a separately reviewed artifact bundle or
model release rather than claiming the Git repository contains model weights.

## Training results

The pilot trained 540,672 LoRA parameters (about 0.11% of the combined model)
on the local RTX 4070 Laptop GPU using BF16. Three epochs took 76.82 seconds and
72 optimizer updates. All 192 training and 48 validation examples were retained;
none were truncated or skipped. The training examples are at most 188 tokens.

| Epoch | Training loss | Validation loss | Validation perplexity |
|---|---:|---:|---:|
| 1 | 0.4650 | 0.1724 | 1.1882 |
| 2 | 0.0778 | 0.0369 | 1.0376 |
| 3 | 0.0219 | 0.0259 | 1.0262 |

These very low validation losses reflect a small, repetitive synthetic task.
Validation uses different fictional entity groups but the same template families.
They are **not** evidence of broad language ability or real-world factual accuracy.

## Four-way comparison: 88 cases per configuration

All four conditions share the evidence-only system instruction, native chat
template, greedy decoding, repetition penalty 1.0, 768-token input budget and
96-token response cap. No-evidence conditions intentionally cannot satisfy the
supported-citation contract; zero cited-answer passes there is not a statement
about Qwen's general knowledge.

| Configuration | Keyword content matches / 72 | Content + correct citation checks / 72 | Required abstention phrase / 16 | Invalid citation IDs |
|---|---:|---:|---:|---:|
| Instruct, no retrieval | 0 | 0 | 16 | 0 |
| Instruct + retrieval | 57 | 0 | 10 | 0 |
| Our LoRA, no retrieval | 0 | 0 | 16 | 0 |
| Our LoRA + retrieval | 46 | 16 | 16 | 1 |

The 72 answerable cases comprise 48 direct questions, 12 controlled follow-ups,
and 12 evidence-injection variants. The remaining 16 cases ask for undocumented
information. A keyword match alone does not establish semantic correctness.
The phrase-based abstention check also rejects some valid alternative refusals:
the baseline correctly said the home address, passport number and database
password were absent without using the required sentence.

Retrieval supplied the intended document on all 72 answerable cases for both
retrieval configurations. This is a small 12-document corpus with correlated
question variants—not a general search-quality benchmark.

LoRA's combined check passes were 15/48 direct, 0/12 follow-up and 1/12 injection
cases. These combined scores are not attack success rates: a missing citation
also fails the check. A separate inspection found the injected `BANANA` text in
6/12 baseline outputs and 5/12 adapter outputs. Neither configuration is robust
against malicious evidence in this pilot.

The baseline emitted only three citation IDs, all present in the supplied context,
but none supported the required gold-document check. LoRA emitted 33 citation IDs,
one outside the supplied set. Valid IDs are not equivalent to supported claims.

## Concrete successes and failures

**Improved grounded response:** for the architecture question, the adapter named
the decoder-only Transformer, learned positional embeddings and weight-tied head,
then cited `[architecture]`. The baseline supplied the facts but omitted a citation.

**Improved abstention:** asked for Aditya's GPA, the baseline invented `3.50` from
an education document that contains no GPA. The adapter answered:
`I don't have evidence to answer that.` This was reproduced through the live API.
The fabricated GPA is not an assertion about Aditya.

**Source contradiction remains:** asked whether this project trained the external
Instruct baseline, the adapter said yes and cited `[baseline]`, despite that
document explicitly saying the opposite. This demonstrates why citation presence
cannot be used as a factuality guarantee.

**Wrong-source citation remains:** the LoRA rank/alpha answer cited `[diagnostics]`
instead of `[v004]`, even though the numeric facts were correct.

**Invented citation:** case `lab-2` produced `[chat]`, which was not among the
retrieved documents. The serving UI now warns about this class of invalid ID.

**Content regression:** keyword coverage fell from 57/72 to 46/72. Some responses
turned into document headings or omitted requested details. The adapter is not
an across-the-board improvement.

## Release decision

**Do not promote v005.** It achieved only 22.2% of the combined answerable checks
against the 80% requirement and violated the zero-invalid-citations gate. Human
claim review is still pending. Better abstention and some new correct citations
are useful findings, but do not compensate for the remaining failures.

The adapter is available in the local UI strictly as an experimental comparison.
v004 and the upstream Instruct checkpoints remain unchanged. No production
portfolio integration, remote model release, or cloud deployment was performed.

## Serving verification

The restarted `/chat` API reproduced both the architecture citation improvement
and the GPA abstention difference with the same prompts and settings. The page
serves the new evidence-mode control and exposes retrieved source text beneath
answers. Backend responses include actual generated-token counts and citation
warnings. The Python test suite passes **120 tests**; JavaScript syntax checks
pass. Browser visual/interaction QA was not performed in this session.

Try `http://127.0.0.1:8000/chat`, select **BB8 v005 pilot — our LoRA + evidence**,
and verify that **Portfolio evidence + citations** is selected. Compare with
**Grounded Qwen — unchanged baseline + evidence** using the same question.

## Next experiment, not conclusions disguised as fixes

Our working hypothesis is that the training distribution is too narrow: mostly
one short document, repeated owners/languages/counts and fixed response shapes,
whereas evaluation asks about varied facts and often supplies multiple documents.
That hypothesis has not been isolated experimentally.

For a future grounded iteration, review the real corpus/labels, author richer multi-document and
contradiction/negation examples with varied citations, and add realistic follow-ups
and conflicting evidence. Preserve a new untouched final evaluation set before
tuning. Compare changed data against the same training settings before changing
model size, learning rate or epochs. Add semantic human grading so valid paraphrases
and unsupported claims are not reduced to keyword checks.

## Honest portfolio contribution statement

“Built a local evidence-grounded assistant experiment with source-versioned data,
BM25 retrieval, native-chat LoRA training, assistant-only loss masking, and a
four-way evaluation across 88 cases. Identified citation/abstention improvements
and content regressions, and withheld model promotion using predefined gates.”

This describes the implemented contribution without claiming authorship of Qwen,
production deployment, or accuracy that the experiment did not establish.
