"""Check probability mathematics, token traces, provenance and local API bounds."""

import json
from types import SimpleNamespace

import pytest
import torch

from api import lab
from inference.lab import distributions, inspect_step, prepare, settings
from inference.generator import TextGenerator
from models.language_model import BB8LM
from tokenizer.char_tokenizer import CharTokenizer


def test_temperature_changes_entropy_without_changing_ranking():
    logits = torch.tensor([3., 1., -1.])
    _, cold, _ = distributions(logits, [], settings({"strategy": "temperature", "temperature": .5}))
    _, hot, _ = distributions(logits, [], settings({"strategy": "temperature", "temperature": 2}))
    assert cold.argmax() == hot.argmax()
    assert cold.max() > hot.max()


def test_repetition_penalty_reduces_both_positive_and_negative_seen_scores():
    raw, adjusted, _ = distributions(torch.tensor([2., -2., 0.]), [0, 1],
                                     settings({"strategy": "temperature", "temperature": 1,
                                               "repetition_penalty": 2}))
    # Compare odds to an unchanged token; normalization can change absolute mass.
    assert adjusted[0] / adjusted[2] < raw[0] / raw[2]
    assert adjusted[1] / adjusted[2] < raw[1] / raw[2]


def test_top_k_and_top_p_have_normalized_minimal_candidate_sets():
    logits = torch.log(torch.tensor([.5, .3, .15, .05]))
    _, _, k = distributions(logits, [], settings({"strategy": "top_k", "top_k": 2, "temperature": 1}))
    _, _, p = distributions(logits, [], settings({"strategy": "top_p", "top_p": .7, "temperature": 1}))
    assert torch.allclose(k, torch.tensor([.625, .375, 0., 0.]))
    assert torch.allclose(k, p)
    _, _, greedy = distributions(logits, [], settings({"strategy": "greedy"}))
    assert greedy.tolist() == [1., 0., 0., 0.]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 0, -1])
def test_invalid_temperature_rejected(value):
    with pytest.raises(ValueError):
        settings({"temperature": value})


@pytest.fixture
def bundle(tmp_path):
    tokenizer = CharTokenizer()
    tokenizer.train(["abc de"])
    model = BB8LM(vocab_size=tokenizer.get_vocab_size(), d_model=16, num_layers=1,
                  num_heads=2, d_ff=32, max_seq_len=16, dropout=0)
    checkpoint = tmp_path / "model.pt"
    torch.save(model.state_dict(), checkpoint)
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    return SimpleNamespace(name="test-model", backend="bb8", device="cpu", prompt_style="bb8",
                           generator=TextGenerator(model, tokenizer, "cpu"), max_context_tokens=16,
                           checkpoint_path=checkpoint, tokenizer_path=tmp_path / "tokenizer.json")


def test_step_trace_uses_visible_context_and_local_rng(bundle):
    ids = prepare(bundle, "abc de", "raw")["input_ids"]
    rng_before = torch.random.get_rng_state().clone()
    first = inspect_step(bundle, ids, {}, 3)
    second = inspect_step(bundle, ids, {}, 3)
    assert first["chosen"] == second["chosen"]
    assert torch.equal(rng_before, torch.random.get_rng_state())
    assert first["context_ids"] == ids[-3:]
    assert first["dropped_tokens"] == len(ids) - 3
    for key in ("raw", "adjusted", "selection"):
        assert sum(row[key] for row in first["candidates"]) + first["other_mass"][key] == pytest.approx(1, abs=1e-6)
    assert first["chosen"]["selection"] > 0


def test_manual_choice_can_override_greedy_but_is_labelled(bundle):
    ids = bundle.generator.tokenizer.encode("abc")
    original = inspect_step(bundle, ids, {"strategy": "greedy"}, 16)
    forced = (original["chosen"]["id"] + 1) % bundle.generator.model.vocab_size
    result = inspect_step(bundle, ids, {"strategy": "greedy"}, 16, forced_token_id=forced)
    assert result["chosen"]["id"] == forced
    assert result["chosen"]["selection"] == 0
    assert result["selection_method"] == "manual"


def test_bad_ids_and_context_rejected(bundle):
    for ids, window in [([], 16), ([-1], 16), ([999999], 16), ([4], 17)]:
        with pytest.raises(ValueError):
            inspect_step(bundle, ids, {}, window)


def test_hf_step_reads_real_forward_logits_without_cache(bundle):
    native = bundle.generator.model
    seen = {}
    class HfModel:
        config = SimpleNamespace(vocab_size=native.vocab_size)
        def __call__(self, input_ids, use_cache):
            seen["ids"] = input_ids.tolist()[0]
            seen["cache"] = use_cache
            return SimpleNamespace(logits=native(input_ids)["logits"])
    bundle.backend = "hf_lora"
    bundle.generator.hf_model = HfModel()
    tokenizer = bundle.generator.tokenizer
    tokenizer.eos_token_id = tokenizer.vocab[tokenizer.eos_token]
    ids = tokenizer.encode("abc")
    trace = inspect_step(bundle, ids, {"strategy": "greedy"}, 2)
    assert seen == {"ids": ids[-2:], "cache": False}
    logits = native(torch.tensor([ids[-2:]]))["logits"][0, -1]
    assert trace["chosen"]["id"] == logits.argmax().item()
    assert trace["chosen"]["raw"] == pytest.approx(logits.softmax(-1).max().item())


def test_nonfinite_logits_rejected_and_top_p_one_preserves_mass():
    with pytest.raises(ValueError, match="non-finite"):
        distributions(torch.tensor([1., float("nan")]), [], settings({}))
    _, adjusted, selection = distributions(torch.tensor([1., 2., 3.]), [],
                                           settings({"top_p": 1}))
    assert torch.allclose(adjusted, selection)


def test_saved_run_recomputes_and_replays_exact_tokens(bundle, tmp_path, monkeypatch):
    monkeypatch.setattr(lab, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(lab, "bundle_for", lambda name: bundle)
    monkeypatch.setattr(lab, "catalog", lambda: {bundle.name: {"run_id": bundle.name}})
    monkeypatch.setattr(lab, "source_hashes", lambda: {"lab": "test-source"})
    # Force a nonspecial token to avoid random EOS terminating this test.
    chosen = bundle.generator.tokenizer.encode("a")[0]
    recipe = {"model": bundle.name, "prompt": "abc", "label": "test",
              "schedule": [{"settings": {"seed": 7}, "context_window": 16, "forced_token_id": chosen}],
              "expected_ids": [chosen]}
    saved = lab.save_run(recipe)
    assert saved["matches_preview"] is True
    assert saved["output"] == "a"
    assert saved["recipe"]["mode"] == "raw"
    replayed = lab.replay_run({"id": saved["id"]})
    assert replayed["matches_original"] is True
    assert replayed["id"] != saved["id"]
    assert replayed["label"] == "Replay: test"
    assert len(lab.saved_runs()["runs"]) == 2
    recipe["expected_ids"] = [chosen + 1]
    with pytest.raises(ValueError, match="differs"):
        lab.save_run(recipe)
    bundle.checkpoint_path.write_bytes(b"different checkpoint")
    with pytest.raises(ValueError, match="changed"):
        lab.replay_run({"id": saved["id"]})


def test_local_api_requires_auth_and_blocks_run_path_traversal(monkeypatch):
    monkeypatch.setenv("BB8_API_KEY", "secret")
    event = {"rawPath": "/lab/catalog", "httpMethod": "POST", "body": "{}"}
    assert lab.handle_lab(event)["statusCode"] == 401
    event.update(rawPath="/lab/run", headers={"x-api-key": "secret"}, body=json.dumps({"id": "../other"}))
    assert lab.handle_lab(event)["statusCode"] == 400


def test_example_shows_actual_historical_mask_and_truncation(monkeypatch):
    class Tokenizer:
        eos_token_id = 0
        def encode(self, text, **kwargs): return [ord(c) for c in text]
        def decode(self, ids): return "".join(chr(i) for i in ids)
        def convert_ids_to_tokens(self, i): return chr(i)
    raw = {"instruction": "Explain", "context": "x" * 400, "response": "answer"}
    monkeypatch.setattr(lab, "dolly", lambda: ([raw], {0}))
    monkeypatch.setattr(lab, "bundle_for", lambda name: SimpleNamespace(
        name=name, max_context_tokens=256, generator=SimpleNamespace(tokenizer=Tokenizer())))
    monkeypatch.setattr(lab, "fingerprint", lambda path: "dataset-hash")
    result = lab.dataset_example({"index": 0})
    assert result["split"] == "validation"
    assert result["prompt_tokens_kept"] == 128
    assert result["response_marker_preserved"] is False
    assert result["eos_preserved"] is True
    assert all(t["label"] == -100 for t in result["tokens"][:128])
    assert all(t["supervised"] for t in result["tokens"][128:])


def test_source_changes_require_server_restart(monkeypatch):
    monkeypatch.setattr(lab, "_current_source_hashes", lambda: {"changed": "hash"})
    with pytest.raises(ValueError, match="restart"):
        lab.source_hashes()
