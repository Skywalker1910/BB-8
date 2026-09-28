"""Conversation formatting helpers shared by BB8 chat clients and the API."""

from collections.abc import Mapping, Sequence
from typing import Any


ROLE_LABELS = {"user": "USER", "assistant": "BB"}
INSTRUCTION_TEMPLATE = "### Instruction:\n{instruction}{context}\n\n### Response:\n"
INSTRUCTION_SYSTEM_CONTEXT = (
    "System: You are BB8, a local educational language model created for "
    "learning and portfolio demonstrations. Answer concisely, and say when "
    "you are not sure."
)


def validate_messages(messages: Any) -> list[dict[str, str]]:
    """Validate and normalize an OpenAI-style list of chat messages."""

    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")

    normalized: list[dict[str, str]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise ValueError(f"messages[{index}] must be an object")

        role = str(message.get("role", "")).lower()
        content = message.get("content")
        if role not in ROLE_LABELS:
            raise ValueError("message role must be 'user' or 'assistant'")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"messages[{index}].content must be non-empty text")
        normalized.append({"role": role, "content": content.strip()})

    if normalized[-1]["role"] != "user":
        raise ValueError("the last message must have role 'user'")
    return normalized


def format_messages(messages: Sequence[Mapping[str, str]]) -> str:
    """Format messages as a text transcript the base language model can continue."""

    lines = [f"{ROLE_LABELS[item['role']]}: {item['content']}" for item in messages]
    return "\n".join(lines) + "\nBB:"


def format_instruction_messages(messages: Sequence[Mapping[str, str]]) -> str:
    """Format chat messages for BB8's pretrained instruction adapters."""

    last_message = messages[-1]
    context_messages = messages[:-1]
    lines = [INSTRUCTION_SYSTEM_CONTEXT]
    lines.extend(
        f"{item['role'].title()}: {item['content']}"
        for item in context_messages
    )
    context = "\n\n### Context:\n" + "\n".join(lines)
    return INSTRUCTION_TEMPLATE.format(
        instruction=last_message["content"],
        context=context,
    )


def _encode(tokenizer: Any, text: str) -> list[int]:
    try:
        return tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        return tokenizer.encode(text)


def build_chat_prompt(
    messages: Any,
    tokenizer: Any,
    max_tokens: int,
    prompt_style: str = "bb8",
) -> tuple[str, int]:
    """Build a transcript and retain only tokens visible to the model."""

    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")

    normalized = validate_messages(messages)
    if prompt_style == "bb8":
        prompt = format_messages(normalized)
    elif prompt_style in {"instruction", "chat_template", "instruction_context_first"}:
        # The instruction comes before the chat history in this training format.
        # Truncating tokens from the left would discard the newest user message.
        history = normalized[:-1]
        while True:
            retained = [*history, normalized[-1]]
            if prompt_style == "chat_template":
                prompt = tokenizer.apply_chat_template(
                    [{"role": "system", "content": INSTRUCTION_SYSTEM_CONTEXT.removeprefix("System: ")}, *retained],
                    tokenize=False, add_generation_prompt=True,
                )
            elif prompt_style == "instruction_context_first":
                context = "\n".join(f"{item['role'].title()}: {item['content']}" for item in history)
                prompt = ("### Context:\n" + INSTRUCTION_SYSTEM_CONTEXT +
                          ("\n" + context if context else "") +
                          "\n\n### Instruction:\n" + normalized[-1]["content"] + "\n\n### Response:\n")
            else:
                prompt = format_instruction_messages(retained)
            token_ids = _encode(tokenizer, prompt)
            if len(token_ids) <= max_tokens:
                return prompt, len(token_ids)
            if not history:
                raise ValueError("latest message exceeds the model context window")
            # Drop the oldest complete turn, keeping recent user/assistant pairs.
            history = history[1:]
            if history and history[0]["role"] == "assistant":
                history = history[1:]
    else:
        raise ValueError("Unknown prompt_style")

    token_ids = _encode(tokenizer, prompt)
    if len(token_ids) > max_tokens:
        token_ids = token_ids[-max_tokens:]
        prompt = tokenizer.decode(token_ids)
    return prompt, len(token_ids)


def extract_assistant_reply(prompt: str, generated_text: str) -> str:
    """Remove the echoed prompt and any model-generated next user turn."""

    reply = generated_text[len(prompt) :] if generated_text.startswith(prompt) else generated_text
    # BPE decoding normalizes whitespace, so a generated record boundary may
    # appear as either a newline or a space before the next USER label.
    boundaries = [
        position
        for marker in (
            "\nUSER:",
            " USER:",
            "\n### Instruction:",
            " ### Instruction:",
            "\n### Response:",
            " ### Response:",
            "\nUser:",
            " User:",
        )
        if (position := reply.find(marker)) >= 0
    ]
    if boundaries:
        reply = reply[: min(boundaries)]
    return reply.strip()
