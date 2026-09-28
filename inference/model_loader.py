"""Utilities for loading a trained BB8 model for inference."""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import torch

from inference.generator import TextGenerator
from models.language_model import BB8LM
from tokenizer import BPETokenizer, CharTokenizer, WordTokenizer


TOKENIZERS = {
    "char": CharTokenizer,
    "word": WordTokenizer,
    "bpe": BPETokenizer,
}


@dataclass
class ModelBundle:
    """A loaded model together with the objects needed for generation."""

    name: str
    generator: TextGenerator
    device: str
    checkpoint_path: Path
    tokenizer_path: Path
    backend: str = "bb8"
    prompt_style: str = "bb8"
    max_context_tokens: int = 0


def _read_model_metadata(model_dir: Path) -> dict:
    manifest_path = model_dir / "manifest.json"
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    config_path = model_dir / "config.yaml"
    if config_path.exists():
        import yaml

        with config_path.open("r", encoding="utf-8") as fh:
            config = yaml.safe_load(fh) or {}
        return {
            "model_name": model_dir.name,
            "tokenizer_type": config.get("tokenizer", {}).get("type", "char"),
        }

    return {"model_name": model_dir.name, "tokenizer_type": "char"}


class HfInstructionGenerator:
    """Generation wrapper for pretrained LoRA instruction adapters."""

    def __init__(
        self,
        model,
        tokenizer,
        device: str,
        max_context_tokens: int,
    ) -> None:
        self.device = torch.device(device)
        self.hf_model = model.to(self.device)
        self.hf_model.eval()
        self.tokenizer = tokenizer
        self.model = SimpleNamespace(max_seq_len=max_context_tokens)

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        strategy: str = "top_p",
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        return_full_text: bool = True,
        return_details: bool = False,
    ) -> str | dict:
        do_sample = strategy != "greedy"
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.model.max_seq_len,
        ).to(self.device)
        generation_args = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "repetition_penalty": repetition_penalty,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.hf_model.generation_config.eos_token_id or self.tokenizer.eos_token_id,
        }
        if do_sample:
            generation_args["temperature"] = max(temperature, 1e-5)
            # Do not inherit hidden sampling filters from the checkpoint.
            generation_args["top_p"] = 1.0
            generation_args["top_k"] = 0
        if strategy == "top_k":
            generation_args["top_k"] = top_k
        elif strategy == "top_p":
            generation_args["top_k"] = 0
            generation_args["top_p"] = top_p
        elif strategy == "temperature":
            generation_args["top_k"] = 0
        elif strategy not in {"greedy", "temperature"}:
            raise ValueError(
                f"Unknown strategy '{strategy}'. "
                "Choose: greedy | temperature | top_k | top_p"
            )

        output = self.hf_model.generate(**inputs, **generation_args)
        if return_full_text:
            token_ids = output[0]
        else:
            token_ids = output[0, inputs["input_ids"].shape[1] :]
        text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        generated = output[0, inputs["input_ids"].shape[1]:].tolist()
        eos = generation_args["eos_token_id"]
        eos_ids = eos if isinstance(eos, list) else [eos]
        if return_details:
            return {"text": text, "generated_tokens": len(generated),
                    "stop_reason": "eos" if generated and generated[-1] in eos_ids else "length"}
        return text

    def generate_with_details(self, **kwargs) -> dict:
        return self.generate(**kwargs, return_details=True)


def _configure_system_trust_store() -> None:
    try:
        import truststore
    except ImportError:
        return
    truststore.inject_into_ssl()


def _load_hf_lora_bundle(model_dir: Path, device: Optional[str]) -> ModelBundle:
    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Pretrained adapter inference dependencies are missing. Run: "
            "pip install -r requirements-finetune.txt"
        ) from exc

    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if selected_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch cannot access a CUDA device")

    adapter_config_path = model_dir / "adapter_config.json"
    with adapter_config_path.open("r", encoding="utf-8") as handle:
        adapter_config = json.load(handle)

    metadata_path = model_dir / "bb8_model_config.json"
    metadata = {}
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)

    _configure_system_trust_store()
    dtype = torch.bfloat16 if selected_device == "cuda" else torch.float32
    base_model_id = adapter_config["base_model_name_or_path"]
    revision = metadata.get("base_model", {}).get("revision")
    cache_dir = metadata.get("base_model", {}).get("cache_dir", ".cache/huggingface")

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        revision=revision,
        cache_dir=cache_dir,
        dtype=dtype,
        trust_remote_code=False,
    )
    model = PeftModel.from_pretrained(base_model, model_dir)
    model.config.use_cache = True

    max_context_tokens = int(metadata.get("max_context_tokens", 256))
    return ModelBundle(
        name=str(metadata.get("model_name", model_dir.name)),
        generator=HfInstructionGenerator(
            model,
            tokenizer,
            selected_device,
            max_context_tokens,
        ),
        device=selected_device,
        checkpoint_path=model_dir / "adapter_model.safetensors",
        tokenizer_path=model_dir / "tokenizer.json",
        backend="hf_lora",
        prompt_style=str(metadata.get("prompt_style", "instruction")),
        max_context_tokens=max_context_tokens,
    )


def load_model_bundle(
    model_dir: str | Path,
    checkpoint_name: str = "best_model.pt",
    device: Optional[str] = None,
) -> ModelBundle:
    """Load a checkpoint and its matching tokenizer from ``model_dir``."""

    model_dir = Path(model_dir)
    metadata_path = model_dir / "bb8_model_config.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("backend") == "hf_causal_lm":
            from transformers import AutoModelForCausalLM, AutoTokenizer

            selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            source = metadata["base_model"]
            kwargs = {"revision": source["revision"], "cache_dir": source["cache_dir"],
                      "local_files_only": True, "trust_remote_code": False}
            tokenizer = AutoTokenizer.from_pretrained(source["model_id"], **kwargs)
            model = AutoModelForCausalLM.from_pretrained(
                source["model_id"], **kwargs,
                dtype=torch.bfloat16 if selected_device == "cuda" else torch.float32,
            )
            cap = int(metadata["max_context_tokens"])
            return ModelBundle(
                name=metadata["model_name"],
                generator=HfInstructionGenerator(model, tokenizer, selected_device, cap),
                device=selected_device, checkpoint_path=metadata_path, tokenizer_path=metadata_path,
                backend="hf_causal_lm", prompt_style="chat_template", max_context_tokens=cap,
            )
    if (model_dir / "adapter_config.json").exists():
        return _load_hf_lora_bundle(model_dir, device)

    checkpoint_path = model_dir / checkpoint_name
    if not checkpoint_path.exists() and checkpoint_name == "best_model.pt":
        packaged_checkpoint = model_dir / "model.pt"
        if packaged_checkpoint.exists():
            checkpoint_path = packaged_checkpoint

    tokenizer_path = model_dir / "tokenizer.json"
    for path in (checkpoint_path, tokenizer_path):
        if not path.exists():
            raise FileNotFoundError(f"Required model artifact not found: {path}")

    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if selected_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch cannot access a CUDA device")

    metadata = _read_model_metadata(model_dir)
    tokenizer_type = str(metadata.get("tokenizer_type", "char")).lower()
    tokenizer_class = TOKENIZERS.get(tokenizer_type)
    if tokenizer_class is None:
        choices = ", ".join(sorted(TOKENIZERS))
        raise ValueError(f"Unknown tokenizer type '{tokenizer_type}'. Choose: {choices}")

    checkpoint = torch.load(checkpoint_path, map_location=selected_device)
    model_config = checkpoint.get("model_config")
    if not model_config:
        raise ValueError(f"Checkpoint does not contain model_config: {checkpoint_path}")

    model = BB8LM(**model_config)
    model.load_state_dict(checkpoint["model_state_dict"])

    tokenizer = tokenizer_class()
    tokenizer.load(str(tokenizer_path))

    if tokenizer.get_vocab_size() != model.vocab_size:
        raise ValueError(
            "Tokenizer and model vocabulary sizes do not match: "
            f"{tokenizer.get_vocab_size()} != {model.vocab_size}"
        )

    thread_count = int(os.getenv("BB8_TORCH_THREADS", "1"))
    torch.set_num_threads(max(1, thread_count))

    return ModelBundle(
        name=str(metadata.get("model_name", model_dir.name)),
        generator=TextGenerator(model, tokenizer, selected_device),
        device=selected_device,
        checkpoint_path=checkpoint_path,
        tokenizer_path=tokenizer_path,
        max_context_tokens=model.max_seq_len,
    )
