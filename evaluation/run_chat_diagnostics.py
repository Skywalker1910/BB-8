"""Run the BB8 chat diagnostic matrix locally, with immutable raw run records.

Usage: python -m evaluation.run_chat_diagnostics --device cuda
All model loading is offline. No checkpoint or serving configuration is changed.
"""

import argparse
from collections import Counter, defaultdict
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
import shutil
from pathlib import Path
import statistics
import time
import uuid

import torch

from evaluation.chat_suite import SUITE_VERSION, cases, messages_for, normalize, score, validate_suite
from experiments.tracking import git_commit, git_is_dirty, runtime_info, sha256_file, utc_now
from inference.conversation import (
    INSTRUCTION_SYSTEM_CONTEXT, INSTRUCTION_TEMPLATE, build_chat_prompt,
    extract_assistant_reply, format_instruction_messages,
)
from inference.model_loader import load_model_bundle


ROOT = Path(__file__).resolve().parents[1]
INSTRUCT_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
PROFILES = {
    "v004_context_first": "Same adapter/settings; put history before the latest instruction (ablation only)",
    "v004_current": "Current adapter, current chat wrapper, greedy, penalty 1.1",
    "v004_no_system": "Same adapter and greedy settings; omit the identity paragraph",
    "qwen_base": "Same current wrapper/settings with the LoRA adapter disabled",
    "qwen_instruct": "Official Qwen Instruct checkpoint with its native chat template",
    "v004_top_p": "Same current adapter/wrapper; sample with T=0.8, P=0.9, penalty 1.1",
}


def write_json(path, data):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def artifact_hashes(directory):
    return {path.name: sha256_file(str(path)) for path in sorted(directory.iterdir())
            if path.is_file() and path.suffix in {".json", ".safetensors", ".yaml", ".txt"}}


def format_variant(messages, tokenizer, profile):
    if profile == "v004_context_first":
        return build_chat_prompt(messages, tokenizer, 1000000, "instruction_context_first")[0]
    if profile == "qwen_instruct":
        return tokenizer.apply_chat_template(
            [{"role": "system", "content": INSTRUCTION_SYSTEM_CONTEXT.removeprefix("System: ")}, *messages],
            tokenize=False, add_generation_prompt=True,
        )
    if profile == "v004_no_system":
        history = "\n".join(f"{message['role'].title()}: {message['content']}" for message in messages[:-1])
        context = "\n\n### Context:\n" + history if history else ""
        return INSTRUCTION_TEMPLATE.format(instruction=messages[-1]["content"], context=context)
    return format_instruction_messages(messages)


def prepare_prompt(messages, tokenizer, profile, max_context):
    original = format_variant(messages, tokenizer, profile)
    original_count = len(tokenizer.encode(original, add_special_tokens=False))
    if profile not in {"qwen_instruct", "v004_no_system", "v004_context_first"}:
        prompt, _ = build_chat_prompt(messages, tokenizer, max_context, "instruction")
    else:
        retained = list(messages)
        while True:
            prompt = format_variant(retained, tokenizer, profile)
            if len(tokenizer.encode(prompt, add_special_tokens=False)) <= max_context:
                break
            if len(retained) == 1:
                raise ValueError("latest message exceeds the model context window")
            retained = retained[1:]
            if retained[0]["role"] == "assistant":
                retained = retained[1:]
    return prompt, original_count


@torch.inference_mode()
def generate_turn(model, tokenizer, messages, profile, args, seed):
    prompt, original_count = prepare_prompt(messages, tokenizer, profile, args.context)
    # Mirror HfInstructionGenerator's tokenization and response extraction.
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.context).to(args.device)
    torch.manual_seed(seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.synchronize()
    eos = model.generation_config.eos_token_id if profile == "qwen_instruct" else tokenizer.eos_token_id
    options = {"max_new_tokens": args.max_new_tokens, "do_sample": profile == "v004_top_p",
               "repetition_penalty": 1.1, "pad_token_id": tokenizer.pad_token_id,
               "eos_token_id": eos, "top_k": 0, "top_p": .9 if profile == "v004_top_p" else 1.,
               "temperature": .8 if profile == "v004_top_p" else 1.}
    started = time.perf_counter()
    with model.disable_adapter() if profile == "qwen_base" else nullcontext():
        output = model.generate(**inputs, **options)
    if args.device == "cuda":
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - started) * 1000
    generated = output[0, inputs.input_ids.shape[1]:].tolist()
    decoded = tokenizer.decode(generated, skip_special_tokens=True)
    reply = extract_assistant_reply(prompt, decoded)
    eos_ids = eos if isinstance(eos, list) else [eos]
    previous = [normalize(item["content"]) for item in messages if item["role"] == "assistant"]
    return {"messages": messages, "formatted_prompt": prompt,
            "input_ids": inputs.input_ids[0].tolist(), "input_tokens": inputs.input_ids.shape[1],
            "prompt_tokens_before_trimming": original_count,
            "history_trimmed": original_count > inputs.input_ids.shape[1],
            "latest_message_present": messages[-1]["content"] in prompt,
            "generated_ids": generated, "generated_tokens": len(generated),
            "raw_output": tokenizer.decode(generated, skip_special_tokens=False),
            "decoded_output": decoded, "reply": reply,
            "stop_reason": "eos" if generated and generated[-1] in eos_ids else "length",
            "generation_options": options, "seed": seed, "latency_ms": round(elapsed, 2),
            "tokens_per_second": round(len(generated) / max(elapsed / 1000, .000001), 2),
            "history_echo": len(reply) > 12 and normalize(reply) in previous,
            "system_echo": "you are bb8" in reply.lower() or "### context:" in reply.lower()}


def run_case(model, tokenizer, case, profile, args, index):
    record = {"case_id": case["id"], "category": case["category"], "profile": profile,
              "hypothesis": case["hypothesis"], "rubric": case["rubric"], "turns": []}
    try:
        if case["turns"]:
            history = []
            for turn_index, question in enumerate(case["turns"]):
                history.append({"role": "user", "content": question})
                turn = generate_turn(model, tokenizer, list(history), profile, args, args.seed + index * 10 + turn_index)
                record["turns"].append(turn)
                history.append({"role": "assistant", "content": turn["reply"] or "[empty response]"})
        else:
            record["turns"].append(generate_turn(model, tokenizer, messages_for(case), profile, args, args.seed + index * 10))
        last = record["turns"][-1]
        record["assessment"] = score(case, last["reply"], formatted_prompt=last["formatted_prompt"])
    except ValueError as error:
        context_error = "latest message exceeds" in str(error)
        record["error"] = str(error)
        record["assessment"] = score(case, "", error="context_limit" if context_error else str(error))
    except Exception as error:
        record["error"] = f"{type(error).__name__}: {error}"
        record["assessment"] = {"status": "error", "reason": record["error"]}
    return record


def summarize(records):
    groups = defaultdict(list)
    for record in records:
        groups[record["profile"]].append(record)
    result = {}
    for profile, rows in groups.items():
        statuses = Counter(row["assessment"]["status"] for row in rows)
        turns = [turn for row in rows for turn in row["turns"]]
        latency = sorted(turn["latency_ms"] for turn in turns)
        categories = {}
        for category in sorted({row["category"] for row in rows}):
            counts = Counter(row["assessment"]["status"] for row in rows if row["category"] == category)
            categories[category] = dict(counts)
        result[profile] = {"cases": len(rows), "checks": dict(statuses), "categories": categories,
                           "median_latency_ms": statistics.median(latency) if latency else None,
                           "p95_latency_ms": latency[min(len(latency) - 1, int(.95 * len(latency)))] if latency else None,
                           "median_tokens_per_second": statistics.median(t["tokens_per_second"] for t in turns) if turns else None,
                           "generation_turns": len(turns), "length_stops": sum(t["stop_reason"] == "length" for t in turns),
                           "history_echoes": sum(t["history_echo"] for t in turns),
                           "system_echoes": sum(t["system_echo"] for t in turns),
                           "history_trimmed_turns": sum(t["history_trimmed"] for t in turns)}
    return result


def audit_training(tokenizer, model_dir):
    """Measure actual v004 prefix/response truncation, without changing training."""
    from fine_tune import InstructionDataset, format_prompt
    data_path = ROOT / "data/databricks-dolly-15k.jsonl"
    run = json.loads((ROOT / "outputs/bb8-qwen-lora-v004-dev/run.json").read_text(encoding="utf-8"))
    digest = sha256_file(str(data_path))
    if digest != run["dataset"]["sha256"]:
        raise ValueError("Dataset differs from the recorded v004 dataset")
    metadata = json.loads((model_dir / "bb8_model_config.json").read_text(encoding="utf-8"))
    limit = int(metadata["max_context_tokens"])
    counts = Counter()
    examples = []
    for index, line in enumerate(data_path.read_text(encoding="utf-8").splitlines()):
        raw = json.loads(line)
        record = {key: str(raw.get(key, "")).strip() for key in ("instruction", "context", "response")}
        prompt_count = len(tokenizer.encode(format_prompt(record), add_special_tokens=False))
        response_count = len(tokenizer.encode(record["response"], add_special_tokens=False)) + 1
        dataset = InstructionDataset([record], tokenizer, limit, preprocessing="legacy_v1")
        item = dataset[0]
        prompt_cut = prompt_count > limit // 2
        eos_cut = item["input_ids"][-1] != tokenizer.eos_token_id
        counts["records"] += 1
        counts["prompt_prefix_truncated"] += prompt_cut
        counts["eos_truncated"] += eos_cut
        counts["any_tokens_truncated"] += prompt_cut or eos_cut
        counts["historical_truncation_counter"] += dataset.truncated_examples
        counts["truncated_but_not_counted"] += (prompt_cut or eos_cut) and not dataset.truncated_examples
        if prompt_cut and len(examples) < 5:
            examples.append({"row_index": index, "instruction": record["instruction"],
                             "prompt_tokens": prompt_count, "response_plus_eos_tokens": response_count,
                             "retained_prompt_tokens": limit // 2})
    return {"dataset_sha256": digest, "max_length": limit, "counts": dict(counts),
            "examples": examples,
            "interpretation": "Cutting a prompt prefix removes the complete response marker. This is a measured preprocessing issue; its causal effect on chat quality needs a new training experiment."}


def write_report(directory, summary, records):
    lines = ["# BB8 chat diagnostic results", "",
             "Automatic checks are conservative development checks, not semantic accuracy. Read raw outputs and human rubrics.", "",
             "| Profile | Pass | Flag | Human review | Error | Median ms |", "|---|---:|---:|---:|---:|---:|"]
    for profile, stats in summary["profiles"].items():
        c = stats["checks"]
        lines.append(f"| {profile} | {c.get('pass', 0)} | {c.get('flag', 0)} | {c.get('review', 0)} | {c.get('error', 0)} | {stats['median_latency_ms']} |")
    lines += ["", "## Responses by case", ""]
    for case_id in dict.fromkeys(row["case_id"] for row in records):
        lines += [f"### {case_id}", ""]
        for row in (r for r in records if r["case_id"] == case_id):
            lines += [f"**{row['profile']}** — {row['assessment']['status']}", ""]
            for turn in row["turns"]:
                lines += ["User: " + turn["messages"][-1]["content"], "", "```text", turn["reply"], "```", ""]
            if row.get("error"):
                lines += [row["error"], ""]
    (directory / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--profiles", nargs="+", choices=list(PROFILES),
                        default=[name for name in PROFILES if name != "v004_context_first"])
    parser.add_argument("--case-ids", nargs="+", help="Optional subset for diagnosis; full suite contains 80 cases")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--context", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--skip-data-audit", action="store_true")
    args = parser.parse_args()
    if not 32 <= args.context <= 256 or not 1 <= args.max_new_tokens <= 256:
        parser.error("context must be 32–256 and max-new-tokens 1–256")
    if len(set(args.profiles)) != len(args.profiles):
        parser.error("profiles must be unique")
    os.chdir(ROOT)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    torch.set_num_threads(1)
    suite = validate_suite(cases())
    selected = [case for case in suite if not args.case_ids or case["id"] in args.case_ids]
    if not selected or args.case_ids and set(args.case_ids) - {case["id"] for case in suite}:
        parser.error("Unknown or empty case selection")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    directory = ROOT / "outputs/chat_diagnostics" / run_id
    directory.mkdir(parents=True)
    print(f"RUN_DIR={directory}", flush=True)
    model_dir = ROOT / "checkpoints/bb8-qwen-lora-v004-dev"
    bundle = load_model_bundle(model_dir, device=args.device)
    adapter, tokenizer = bundle.generator.hf_model, bundle.generator.tokenizer
    base_revision = json.loads((model_dir / "bb8_model_config.json").read_text())["base_model"]["revision"]
    base_dir = ROOT / ".cache/huggingface/models--Qwen--Qwen2.5-0.5B/snapshots" / base_revision
    instruct_dir = ROOT / ".cache/huggingface/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots" / INSTRUCT_REVISION
    sources = ["evaluation/chat_suite.py", "evaluation/run_chat_diagnostics.py", "inference/conversation.py",
               "inference/model_loader.py", "fine_tune.py", "api/handler.py"]
    manifest = {"run_id": run_id, "started_at": utc_now(), "suite": SUITE_VERSION,
                "suite_sha256": canonical_hash(suite), "selected_case_ids": [c["id"] for c in selected],
                "purpose": "development diagnosis; no claim of held-out or uncontaminated evaluation",
                "profiles": {name: PROFILES[name] for name in args.profiles}, "arguments": vars(args),
                "git_commit": git_commit(), "git_dirty": git_is_dirty(),
                "source_hashes": {name: sha256_file(str(ROOT / name)) for name in sources},
                "runtime": runtime_info(torch.device(args.device)),
                "libraries": {name: version(name) for name in ("transformers", "peft", "tokenizers")},
                "adapter_hashes": artifact_hashes(model_dir), "base_revision": base_revision,
                "base_hashes": artifact_hashes(base_dir),
                "instruct_revision": INSTRUCT_REVISION if "qwen_instruct" in args.profiles else None,
                "instruct_hashes": artifact_hashes(instruct_dir) if "qwen_instruct" in args.profiles else None,
                "adapter_generation_config": adapter.generation_config.to_dict(),
                "latency_note": "Warm-up excluded. Local shared GPU, sequential batch size 1; latency includes decoding to IDs but excludes prompt formatting and HTTP."}
    write_json(directory / "manifest.json", manifest)
    for name in sources:
        destination = directory / "source_snapshot" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
    write_json(directory / "suite.json", suite)
    if not args.skip_data_audit:
        print("Auditing historical training preprocessing...", flush=True)
        audit = audit_training(tokenizer, model_dir)
        write_json(directory / "training_audit.json", audit)
        print(json.dumps(audit["counts"]), flush=True)
    records = []
    with (directory / "results.jsonl").open("x", encoding="utf-8") as output:
        for profile in args.profiles:
            if profile == "qwen_instruct":
                from transformers import AutoModelForCausalLM, AutoTokenizer
                profile_tokenizer = AutoTokenizer.from_pretrained(instruct_dir, local_files_only=True)
                model = AutoModelForCausalLM.from_pretrained(instruct_dir, local_files_only=True,
                                                           dtype=torch.bfloat16 if args.device == "cuda" else torch.float32).to(args.device).eval()
            else:
                model, profile_tokenizer = adapter, tokenizer
            print(f"PROFILE={profile}", flush=True)
            generate_turn(model, profile_tokenizer, [{"role": "user", "content": "Hello"}], profile, args, args.seed)
            for index, case in enumerate(selected):
                record = run_case(model, profile_tokenizer, case, profile, args, suite.index(case))
                records.append(record)
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                output.flush()
                print(f"{profile} {index+1}/{len(selected)} {case['id']} {record['assessment']['status']}", flush=True)
            if profile == "qwen_instruct":
                del model
                if args.device == "cuda":
                    torch.cuda.empty_cache()
    summary = {"run_id": run_id, "completed_at": utc_now(), "suite": SUITE_VERSION,
               "suite_sha256": manifest["suite_sha256"], "profiles": summarize(records)}
    write_json(directory / "summary.json", summary)
    write_report(directory, summary, records)
    with (ROOT / "experiments/chat_diagnostics_registry.jsonl").open("a", encoding="utf-8") as registry:
        registry.write(json.dumps({**summary, "output_dir": directory.relative_to(ROOT).as_posix()}, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
