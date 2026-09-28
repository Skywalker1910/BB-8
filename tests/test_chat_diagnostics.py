"""Check experimental controls and scoring, without loading external models."""

from types import SimpleNamespace

import pytest

from evaluation.chat_suite import cases, messages_for, score, validate_suite
from evaluation import run_chat_diagnostics as runner
from inference.conversation import build_chat_prompt


def test_suite_has_80_unique_cases_and_rejects_changes_to_count():
    suite = validate_suite(cases())
    assert len(suite) == 80
    assert sum(case["check"] == "manual" for case in suite) == 23
    with pytest.raises(ValueError):
        validate_suite(suite[:-1])


def test_exact_grader_does_not_accept_answer_word_inside_wrong_claim():
    case = next(c for c in cases() if c["id"] == "knowledge-01")
    assert score(case, " Paris. ")["status"] == "pass"
    assert score(case, "Paris is not the capital; London is.")["status"] == "flag"
    assert score(case, "The capital is Paris.")["status"] == "flag"  # conservative format check


def test_json_and_uppercase_checks_are_not_casefolded_or_loose():
    suite = {c["id"]: c for c in cases()}
    assert score(suite["instructions-02"], "HELLO")["status"] == "pass"
    assert score(suite["instructions-02"], "hello")["status"] == "flag"
    assert score(suite["instructions-03"], '{"answer": 4}')["status"] == "pass"
    assert score(suite["instructions-03"], '{"answer": 4.0}')["status"] == "flag"
    assert score(suite["instructions-03"], 'Here is {"answer": 4}')["status"] == "flag"


def test_memory_score_depends_on_actual_visible_evidence():
    case = next(c for c in cases() if c["check"] == "retained_fact")
    assert score(case, "teal", formatted_prompt="My favorite color is teal.")["status"] == "pass"
    assert score(case, "teal", formatted_prompt="What is my favorite color?")["status"] == "flag"
    assert score(case, "UNKNOWN", formatted_prompt="What is my favorite color?")["status"] == "pass"


def test_expected_rejection_is_distinct_from_runtime_failure():
    case = next(c for c in cases() if c["check"] == "reject")
    assert score(case, "", error="context_limit")["status"] == "pass"
    assert score(case, "", error="CUDA out of memory")["status"] == "error"
    assert score(case, "an answer")["status"] == "flag"


class Tokenizer:
    def encode(self, text, **kwargs): return list(text)


def test_current_profile_uses_serving_prompt_policy_and_preserves_latest():
    case = next(c for c in cases() if c["id"] == "context-05")
    messages = messages_for(case)
    tokenizer = Tokenizer()
    actual, original_count = runner.prepare_prompt(messages, tokenizer, "v004_current", 500)
    expected, _ = build_chat_prompt(messages, tokenizer, 500, "instruction")
    assert actual == expected
    assert original_count > len(actual)
    assert case["question"] in actual
    assert "note 0:" not in actual


def test_no_system_ablation_removes_only_system_when_no_history():
    messages = [{"role": "user", "content": "What is 2 + 2?"}]
    actual, _ = runner.prepare_prompt(messages, Tokenizer(), "v004_no_system", 500)
    assert actual == "### Instruction:\nWhat is 2 + 2?\n\n### Response:\n"


def test_rollouts_use_actual_generated_history(monkeypatch):
    case = next(c for c in cases() if c["turns"])
    observed = []
    def generate(model, tokenizer, messages, profile, args, seed):
        observed.append(messages)
        return {"reply": f"actual reply {len(observed)}", "formatted_prompt": "test"}
    monkeypatch.setattr(runner, "generate_turn", generate)
    record = runner.run_case(None, None, case, "v004_current", SimpleNamespace(seed=42), 0)
    assert observed[1][1] == {"role": "assistant", "content": "actual reply 1"}
    assert record["assessment"]["status"] == "review"
    assert len(record["turns"]) == 4
