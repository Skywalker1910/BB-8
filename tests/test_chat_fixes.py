"""Regression tests for native chat formatting and generation telemetry."""

import json
from types import SimpleNamespace

import pytest
import torch

from api import handler
from inference.conversation import build_chat_prompt
from inference.model_loader import HfInstructionGenerator


class Inputs(dict):
    def to(self, device):
        return self


class Tokenizer:
    pad_token_id = 0
    eos_token_id = 9

    def __call__(self, text, **kwargs):
        return Inputs(input_ids=torch.tensor([[1, 2]]))

    def decode(self, ids, **kwargs):
        return "answer"

    def encode(self, text, **kwargs):
        return list(text.encode())

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"tokenize": False, "add_generation_prompt": True}
        return "|".join(item["role"] + ":" + item["content"] for item in messages) + "|assistant:"


class Model:
    generation_config = SimpleNamespace(eos_token_id=[8, 9])

    def to(self, device):
        return self

    def eval(self):
        return self

    def generate(self, **kwargs):
        self.options = kwargs
        return torch.tensor([[1, 2, 5, 8]])


@pytest.mark.parametrize("strategy,k,p", [("temperature", 0, 1.), ("top_k", 4, 1.), ("top_p", 0, .7)])
def test_hf_actual_tokens_eos_and_no_hidden_filters(strategy, k, p):
    model = Model()
    generator = HfInstructionGenerator(model, Tokenizer(), "cpu", 256)
    result = generator.generate_with_details(prompt="hi", strategy=strategy, top_k=4, top_p=.7,
                                             max_new_tokens=64, return_full_text=False)
    assert result == {"text": "answer", "generated_tokens": 2, "stop_reason": "eos"}
    assert model.options["top_k"] == k
    assert model.options["top_p"] == p
    assert model.options["eos_token_id"] == [8, 9]
    assert generator.generate("hi", strategy="greedy") == "answer"


def test_native_template_keeps_latest_and_trims_complete_history():
    tokenizer = Tokenizer()
    latest = {"role": "user", "content": "3 + 3"}
    single, count = build_chat_prompt([latest], tokenizer, 1000, "chat_template")
    prompt, actual = build_chat_prompt([
        {"role": "user", "content": "2 + 2"},
        {"role": "assistant", "content": "4"}, latest,
    ], tokenizer, count, "chat_template")
    assert prompt == single
    assert actual == count
    assert prompt.endswith("user:3 + 3|assistant:")
    with pytest.raises(ValueError, match="latest message exceeds"):
        build_chat_prompt([latest], tokenizer, count - 1, "chat_template")


def test_model_selection_allowlist_and_api_actual_count(monkeypatch):
    generator = HfInstructionGenerator(Model(), Tokenizer(), "cpu", 1024)
    bundle = SimpleNamespace(name="external-baseline", device="cpu", generator=generator,
                             backend="hf_causal_lm", prompt_style="chat_template", max_context_tokens=1024)
    monkeypatch.setattr(handler, "get_bundle", lambda: bundle)
    monkeypatch.setattr(handler, "_chat_models", {"baseline": {"model_dir": "approved", "label": "Baseline"}})
    monkeypatch.setattr(handler, "_chat_bundles", {})
    loaded = []

    def load(path, **kwargs):
        loaded.append(path)
        return bundle

    monkeypatch.setattr(handler, "load_model_bundle", load)
    monkeypatch.setenv("BB8_API_KEY", "secret")
    event = {"rawPath": "/chat", "httpMethod": "POST", "headers": {"x-api-key": "secret"},
             "body": json.dumps({"model": "baseline", "messages": [{"role": "user", "content": "hi"}],
                                 "max_new_tokens": 64})}
    response = handler.lambda_handler(event, None)
    assert response["statusCode"] == 200
    payload = json.loads(response["body"])
    assert payload["generated_tokens"] == 2
    assert payload["stop_reason"] == "eos"
    assert handler.chat_bundle("baseline") is bundle
    assert loaded == ["approved"]
    assert handler.chat_bundle(None) is bundle
    for invalid in ("../../other", ["baseline"], "unknown"):
        with pytest.raises(ValueError, match="Unknown chat model"):
            handler.chat_bundle(invalid)
    assert handler.lambda_handler({**event, "headers": {}}, None)["statusCode"] == 401


def test_native_generator_counts_early_eos(monkeypatch):
    from inference.generator import TextGenerator
    from models.language_model import BB8LM
    from tokenizer import CharTokenizer

    tokenizer = CharTokenizer()
    tokenizer.train(["abc"])
    model = BB8LM(vocab_size=tokenizer.get_vocab_size(), d_model=16, num_layers=1,
                  num_heads=2, d_ff=32, max_seq_len=16, dropout=0.)
    generator = TextGenerator(model, tokenizer, "cpu")
    monkeypatch.setattr(generator, "_greedy", lambda logits: tokenizer.vocab[tokenizer.eos_token])
    result = generator.generate_with_details(prompt="a", strategy="greedy", max_new_tokens=64)
    assert result["generated_tokens"] == 1
    assert result["stop_reason"] == "eos"
