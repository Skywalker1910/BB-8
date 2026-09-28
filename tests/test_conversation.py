"""Tests for BB8 conversation formatting and prompt truncation."""

import pytest

from inference.conversation import (
    build_chat_prompt,
    extract_assistant_reply,
    format_instruction_messages,
    validate_messages,
)


class CharacterTokenizer:
    def encode(self, text):
        return [ord(char) for char in text]

    def decode(self, token_ids):
        return "".join(chr(token_id) for token_id in token_ids)


def test_build_chat_prompt_formats_roles():
    prompt, token_count = build_chat_prompt(
        [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "Continue"},
        ],
        CharacterTokenizer(),
        max_tokens=128,
    )
    assert prompt == "USER: Hello\nBB: Hi\nUSER: Continue\nBB:"
    assert token_count == len(prompt)


def test_build_chat_prompt_formats_instruction_adapter_prompt():
    prompt, token_count = build_chat_prompt(
        [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "What is 2 + 2?"},
        ],
        CharacterTokenizer(),
        max_tokens=256,
        prompt_style="instruction",
    )
    assert "### Instruction:\nWhat is 2 + 2?" in prompt
    assert "System: You are BB8" in prompt
    assert "User: Hello\nAssistant: Hi there" in prompt
    assert prompt.endswith("### Response:\n")
    assert token_count == len(prompt)


def test_build_chat_prompt_keeps_only_visible_context():
    prompt, token_count = build_chat_prompt(
        [{"role": "user", "content": "a" * 100}],
        CharacterTokenizer(),
        max_tokens=32,
    )
    assert token_count == 32
    assert len(prompt) == 32
    assert prompt.endswith("\nBB:")


def test_instruction_prompt_drops_old_turns_without_losing_latest_question():
    latest = {"role": "user", "content": "What is 2 + 2?"}
    recent_turn = [
        {"role": "user", "content": "Who is Thomas Jefferson?"},
        {"role": "assistant", "content": "He was the third US president."},
    ]
    old_turn = [
        {"role": "user", "content": "What is process mining?"},
        {"role": "assistant", "content": "An old response about process mining."},
    ]
    max_tokens = len(format_instruction_messages([*recent_turn, latest]))

    prompt, token_count = build_chat_prompt(
        [*old_turn, *recent_turn, latest],
        CharacterTokenizer(),
        max_tokens=max_tokens,
        prompt_style="instruction",
    )

    assert prompt.startswith("### Instruction:\nWhat is 2 + 2?")
    assert "System: You are BB8" in prompt
    assert "Who is Thomas Jefferson?" in prompt
    assert "He was the third US president." in prompt
    assert "process mining" not in prompt
    assert token_count == max_tokens


def test_instruction_prompt_rejects_question_that_cannot_fit():
    with pytest.raises(ValueError, match="latest message exceeds"):
        build_chat_prompt(
            [{"role": "user", "content": "x" * 300}],
            CharacterTokenizer(),
            max_tokens=256,
            prompt_style="instruction",
        )


def test_extract_reply_removes_echo_and_generated_user_turn():
    prompt = "USER: Hello\nBB8:"
    generated = prompt + " Greetings\nUSER: invented turn"
    assert extract_assistant_reply(prompt, generated) == "Greetings"


def test_extract_reply_removes_bpe_normalized_user_turn():
    generated = "A short answer. USER: invented next question BB: invented answer"
    assert extract_assistant_reply("ignored", generated) == "A short answer."


def test_extract_reply_removes_instruction_boundary():
    generated = "Four.\n### Instruction:\nWhat next?"
    assert extract_assistant_reply("ignored", generated) == "Four."


def test_validate_messages_rejects_unknown_role():
    with pytest.raises(ValueError, match="message role"):
        validate_messages([{"role": "system", "content": "Be helpful"}])
