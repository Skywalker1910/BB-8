"""Unit tests for the pretrained-model fine-tuning data path."""

import json

from fine_tune import InstructionCollator, InstructionDataset, format_prompt, load_records, split_records


class TinyTokenizer:
    eos_token_id = 2
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(character) % 127 + 3 for character in text]


def test_format_prompt_separates_instruction_context_and_response():
    prompt = format_prompt(
        {"instruction": "Summarize", "context": "Some text", "response": "Done"}
    )
    assert "### Instruction:\nSummarize" in prompt
    assert "### Context:\nSome text" in prompt
    assert prompt.endswith("### Response:\n")
    assert "Done" not in prompt


def test_instruction_dataset_masks_prompt_tokens():
    dataset = InstructionDataset(
        [{"instruction": "Add", "context": "", "response": "Four"}],
        TinyTokenizer(),
        max_length=64,
    )
    example = dataset[0]
    first_supervised = next(index for index, value in enumerate(example["labels"]) if value != -100)
    assert all(value == -100 for value in example["labels"][:first_supervised])
    assert example["labels"][-1] == TinyTokenizer.eos_token_id


def test_collator_pads_labels_with_ignore_index():
    collator = InstructionCollator(pad_token_id=0)
    batch = collator(
        [
            {"input_ids": [1, 2], "attention_mask": [1, 1], "labels": [-100, 2]},
            {"input_ids": [1], "attention_mask": [1], "labels": [1]},
        ]
    )
    assert batch["input_ids"].shape == (2, 2)
    assert batch["labels"][1, 1].item() == -100


def test_records_are_deterministic_and_disjoint(tmp_path):
    path = tmp_path / "records.jsonl"
    records = [
        {"instruction": f"Question {index}", "context": "", "response": "Answer"}
        for index in range(10)
    ]
    path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
    first = load_records(path, seed=42)
    second = load_records(path, seed=42)
    train, validation = split_records(first, validation_fraction=0.2)
    assert first == second
    assert len(train) == 8
    assert len(validation) == 2
    assert {item["instruction"] for item in train}.isdisjoint(
        {item["instruction"] for item in validation}
    )
