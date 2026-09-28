"""Tests for the lightweight experiment registry."""

import json

from experiments.tracking import register_run, sha256_file


def test_sha256_file_is_stable(tmp_path):
    data_path = tmp_path / "data.txt"
    data_path.write_text("bb8", encoding="utf-8")
    assert sha256_file(str(data_path)) == sha256_file(str(data_path))


def test_register_run_writes_record_and_registry(tmp_path):
    output_dir = tmp_path / "outputs" / "test-run"
    registry_path = tmp_path / "model_registry.jsonl"
    record = {
        "run_id": "test-run",
        "artifacts": {"output_dir": str(output_dir)},
    }

    run_path = register_run(record, str(registry_path))

    with open(run_path, encoding="utf-8") as fh:
        assert json.load(fh)["run_id"] == "test-run"
    with open(registry_path, encoding="utf-8") as fh:
        assert json.loads(fh.readline())["run_id"] == "test-run"
