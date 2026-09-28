"""Local-only lab routes: model inspection, Dolly browsing and saved experiments."""

import json
import logging
import platform
import random
import re
import threading
import time
import uuid
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch

from api import handler
from experiments.tracking import git_commit, git_is_dirty, sha256_file, utc_now
from inference.lab import ENGINE_VERSION, inspect_step, prepare, token_info
from inference.model_loader import load_model_bundle


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "outputs" / "lab_runs"
LAB_LOCK = threading.RLock()
_bundles = {}
_loaded_hashes = {}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def catalog():
    registry = ROOT / "experiments" / "model_registry.jsonl"
    records = {}
    if registry.exists():
        for line in registry.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            name = record["run_id"]
            if re.fullmatch(r"[a-zA-Z0-9_-]+", name) and (ROOT / "checkpoints" / name).is_dir():
                records[name] = record
    return records


@lru_cache(maxsize=64)
def file_hash(path, size, mtime):
    return sha256_file(path)


def fingerprint(path):
    stat = path.stat()
    return file_hash(str(path), stat.st_size, stat.st_mtime_ns)


def bundle_for(name):
    if name not in catalog():
        raise ValueError("Choose a model from the local registry")
    if name not in _bundles:
        current = handler.get_bundle()
        _bundles[name] = current if name == current.name else load_model_bundle(
            ROOT / "checkpoints" / name, device=current.device,
        )
        _loaded_hashes[name] = identity(_bundles[name])
    if identity(_bundles[name]) != _loaded_hashes[name]:
        raise ValueError("Model files changed after loading; restart the local server")
    return _bundles[name]


def identity(bundle):
    directory = bundle.checkpoint_path.parent
    names = [bundle.checkpoint_path.name, bundle.tokenizer_path.name,
             "adapter_config.json", "bb8_model_config.json", "tokenizer_config.json",
             "special_tokens_map.json", "added_tokens.json", "merges.txt", "vocab.json",
             "config.yaml", "manifest.json"]
    return {name: fingerprint(directory / name) for name in names
            if (directory / name).is_file()}


def _current_source_hashes():
    names = [
        "inference/lab.py", "inference/conversation.py", "inference/model_loader.py",
        "api/lab.py",
    ]
    for directory in ("models", "tokenizer"):
        names.extend(path.relative_to(ROOT).as_posix() for path in (ROOT / directory).glob("*.py"))
    return {name: fingerprint(ROOT / name) for name in names}


_SOURCE_AT_IMPORT = _current_source_hashes()


def source_hashes():
    current = _current_source_hashes()
    if current != _SOURCE_AT_IMPORT:
        raise ValueError("Inference source changed on disk; restart the local server before saving or replaying")
    return current


def runtime_info(bundle):
    packages = {}
    for name in ("transformers", "peft", "tokenizers", "safetensors"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            pass
    model = bundle.generator.hf_model if bundle.backend == "hf_lora" else bundle.generator.model
    return {"python": platform.python_version(), "torch": torch.__version__,
            "device": bundle.device, "cuda": torch.version.cuda,
            "dtype": str(next(model.parameters()).dtype), "packages": packages,
            "gpu": torch.cuda.get_device_name(bundle.device) if str(bundle.device).startswith("cuda") else None}


def model_payload():
    return {"engine": ENGINE_VERSION, "default_model": handler.get_bundle().name,
            "models": [{"name": name, "record": record} for name, record in catalog().items()]}


def prepare_request(body):
    bundle = bundle_for(body.get("model"))
    prepared = prepare(bundle, body.get("prompt"), body.get("mode", "raw"))
    return {**prepared, "model": bundle.name, "backend": bundle.backend,
            "max_context_tokens": bundle.max_context_tokens,
            "artifact_hashes": identity(bundle)}


def step_request(body):
    bundle = bundle_for(body.get("model"))
    started = time.perf_counter()
    result = inspect_step(bundle, body.get("input_ids"), body.get("settings", {}),
                          int(body.get("context_window", bundle.max_context_tokens)),
                          int(body.get("step_index", 0)), body.get("forced_token_id"))
    prompt_length = int(body.get("prompt_token_count", len(body["input_ids"])))
    if not 0 <= prompt_length <= len(body["input_ids"]):
        raise ValueError("Invalid prompt token count")
    result["continuation"] = bundle.generator.tokenizer.decode(
        body["input_ids"][prompt_length:] + [result["chosen"]["id"]])
    return {**result, "model": bundle.name,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2)}


@lru_cache(maxsize=1)
def dolly():
    path = ROOT / "data" / "databricks-dolly-15k.jsonl"
    if not path.exists():
        raise ValueError("Dolly data is missing; run python data/prepare_data.py --dataset dolly_15k")
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    run = read_json(ROOT / "outputs" / "bb8-qwen-lora-v004-dev" / "run.json")
    if fingerprint(path) != run["dataset"]["sha256"]:
        raise ValueError("Dolly file differs from v004's recorded dataset; split labels would be invalid")
    indices = list(range(len(records)))
    random.Random(run["seed"]).shuffle(indices)
    validation = set(indices[:run["dataset"]["validation_records"]])
    return records, validation


def dataset_page(body):
    records, validation = dolly()
    query = str(body.get("query", "")).lower()[:200]
    category = str(body.get("category", ""))
    split = str(body.get("split", ""))
    offset = int(body.get("offset", 0))
    if offset < 0:
        raise ValueError("offset must be nonnegative")
    matches = []
    for index, record in enumerate(records):
        label = "validation" if index in validation else "train"
        if category and record.get("category") != category:
            continue
        if split and split != label:
            continue
        if query and query not in " ".join(str(v) for v in record.values()).lower():
            continue
        matches.append({"index": index, "instruction": record["instruction"],
                        "category": record.get("category", ""), "split": label})
    return {"records": matches[offset:offset + 12], "total": len(matches), "offset": offset,
            "categories": sorted({r.get("category", "") for r in records}),
            "manifest": read_json(ROOT / "data" / "manifests" / "dolly_15k.json"),
            "split_model": "bb8-qwen-lora-v004-dev"}


def dataset_example(body):
    from fine_tune import InstructionDataset, format_prompt

    records, validation = dolly()
    index = int(body.get("index", -1))
    if not 0 <= index < len(records):
        raise ValueError("Invalid dataset record index")
    raw = records[index]
    record = {key: str(raw.get(key, "")).strip() for key in ("instruction", "context", "response")}
    # Reconstruct v004 with the exact dataset code used by fine_tune.py.
    bundle = bundle_for("bb8-qwen-lora-v004-dev")
    tokenizer = bundle.generator.tokenizer
    max_length = bundle.max_context_tokens
    example = InstructionDataset([record], tokenizer, max_length, preprocessing="legacy_v1")[0]
    full_prompt = format_prompt(record)
    full_ids = tokenizer.encode(full_prompt, add_special_tokens=False)
    response_ids = tokenizer.encode(record["response"], add_special_tokens=False) + [tokenizer.eos_token_id]
    prompt_kept = sum(label == -100 for label in example["labels"])
    response_kept = len(example["labels"]) - prompt_kept
    return {"index": index, "raw": raw, "split": "validation" if index in validation else "train",
            "preview_model": bundle.name, "formatted_prompt": full_prompt,
            "formatted_example": full_prompt + record["response"],
            "tokens": [{**token_info(tokenizer, token_id), "label": label,
                        "supervised": label != -100}
                       for token_id, label in zip(example["input_ids"], example["labels"])],
            "max_length": max_length, "prompt_tokens_original": len(full_ids),
            "prompt_tokens_kept": prompt_kept, "response_tokens_original": len(response_ids),
            "response_tokens_kept": response_kept,
            "response_marker_preserved": prompt_kept == len(full_ids),
            "eos_preserved": example["input_ids"][-1] == tokenizer.eos_token_id,
            "dataset_sha256": fingerprint(ROOT / "data" / "databricks-dolly-15k.jsonl")}


def run_path(run_id):
    if not isinstance(run_id, str) or not re.fullmatch(r"[a-f0-9]{32}", run_id):
        raise ValueError("Invalid saved run ID")
    return RUNS / f"{run_id}.json"


def saved_runs():
    summaries = []
    for path in sorted(RUNS.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:100]:
        record = read_json(path)
        summaries.append({key: record[key] for key in ("id", "label", "model", "created_at", "output")})
    return {"runs": summaries}


def execute_recipe(recipe):
    bundle = bundle_for(recipe.get("model"))
    prepared = prepare(bundle, recipe.get("prompt"), recipe.get("mode", "raw"))
    schedule = recipe.get("schedule")
    if not isinstance(schedule, list) or not 1 <= len(schedule) <= 64:
        raise ValueError("Save between 1 and 64 generation steps")
    ids = list(prepared["input_ids"])
    traces = []
    for index, item in enumerate(schedule):
        if not isinstance(item, dict):
            raise ValueError("Each schedule step must be an object")
        trace = inspect_step(bundle, ids, item.get("settings", {}),
                             int(item.get("context_window", bundle.max_context_tokens)),
                             index, item.get("forced_token_id"))
        traces.append(trace)
        ids.append(trace["chosen"]["id"])
        trace["continuation"] = bundle.generator.tokenizer.decode(ids[len(prepared["input_ids"]):])
        if trace["eos"] and index != len(schedule) - 1:
            raise ValueError("A saved recipe cannot continue after EOS")
    continuation = ids[len(prepared["input_ids"]):]
    record = {
        "schema_version": 1, "engine": ENGINE_VERSION, "id": uuid.uuid4().hex,
        "created_at": utc_now(), "label": str(recipe.get("label", "Untitled experiment"))[:80],
        "model": bundle.name,
        "recipe": {"model": bundle.name, "prompt": recipe["prompt"],
                   "mode": recipe.get("mode", "raw"),
                   "schedule": [{"settings": trace["settings"],
                                 "context_window": trace["context_window"],
                                 "forced_token_id": trace["chosen"]["id"] if trace["selection_method"] == "manual" else None}
                                for trace in traces]},
        "input": prepared, "generated_ids": continuation, "traces": traces,
        "output": bundle.generator.tokenizer.decode(continuation),
        "artifact_hashes": identity(bundle), "source_hashes": source_hashes(),
        "training_run": catalog()[bundle.name],
        "runtime": runtime_info(bundle),
        "git_commit": git_commit(), "git_dirty": git_is_dirty(),
    }
    expected = recipe.get("expected_ids")
    if expected is not None:
        record["matches_preview"] = expected == continuation
    return record


def save_run(body):
    record = execute_recipe(body)
    if record.get("matches_preview") is False:
        raise ValueError("Saved recomputation differs from preview; inspect again before saving")
    RUNS.mkdir(parents=True, exist_ok=True)
    with run_path(record["id"]).open("x", encoding="utf-8") as file:
        json.dump(record, file, ensure_ascii=False, indent=2)
    return record


def replay_run(body):
    original = read_json(run_path(body.get("id")))
    bundle = bundle_for(original["model"])
    if original["artifact_hashes"] != identity(bundle) or original["source_hashes"] != source_hashes():
        raise ValueError("Model artifacts or inference source changed; restore them to replay this run")
    record = execute_recipe({**original["recipe"], "label": "Replay: " + original["label"]})
    record["replay_of"] = original["id"]
    record["matches_original"] = record["generated_ids"] == original["generated_ids"]
    with run_path(record["id"]).open("x", encoding="utf-8") as file:
        json.dump(record, file, ensure_ascii=False, indent=2)
    return record


def handle_lab(event):
    method, path, headers = handler._request_parts(event)
    if not handler._authorised(headers):
        return handler._response(401, {"error": "Invalid or missing API key"})
    if method != "POST":
        return handler._response(405, {"error": "Lab API uses POST requests"})
    routes = {
        "/lab/catalog": lambda body: model_payload(),
        "/lab/prepare": prepare_request, "/lab/step": step_request,
        "/lab/dataset": dataset_page, "/lab/example": dataset_example,
        "/lab/runs": lambda body: saved_runs(),
        "/lab/run": lambda body: read_json(run_path(body.get("id"))),
        "/lab/save": save_run, "/lab/replay": replay_run,
    }
    try:
        if path not in routes:
            return handler._response(404, {"error": "Unknown lab route"})
        with LAB_LOCK:
            return handler._response(200, routes[path](handler._parse_body(event)))
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        return handler._response(400, {"error": str(exc)})
    except FileNotFoundError:
        return handler._response(404, {"error": "Requested local artifact was not found"})
    except Exception:
        logging.exception("Local lab request failed")
        return handler._response(500, {"error": "Local model inspection failed; check the server log"})
