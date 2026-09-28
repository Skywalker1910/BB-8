"""Tests for deterministic instruction-dataset preparation."""

import json

from data.prepare_data import format_dolly, sha256_file


def test_format_dolly_is_deterministic_and_chat_shaped(tmp_path):
    raw_path = tmp_path / "dolly.jsonl"
    output_path = tmp_path / "dolly.txt"
    records = [
        {
            "instruction": "Explain attention.",
            "context": "Transformers",
            "response": "Attention mixes relevant token information.",
            "category": "open_qa",
        },
        {
            "instruction": "Say hello—politely.",
            "context": "",
            "response": "Hello!",
            "category": "generation",
        },
    ]
    raw_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    first = format_dolly(raw_path, output_path, seed=42)
    first_hash = sha256_file(output_path)
    second = format_dolly(raw_path, output_path, seed=42)

    text = output_path.read_text(encoding="utf-8")
    assert first["records"] == 2
    assert first["categories"] == {"generation": 1, "open_qa": 1}
    assert first_hash == second["formatted_sha256"]
    assert "USER:" in text
    assert "CONTEXT:" in text
    assert "\nBB:" in text
    assert text.isascii()
