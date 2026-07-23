import json
import os

from typer.testing import CliRunner
from fairfuzzkv_codec.cli import app

runner = CliRunner()


def _read_manifest_record(stdout: str) -> dict:
    run_dir_line = [line for line in stdout.split("\n") if "Run Directory:" in line]
    assert run_dir_line, f"no Run Directory line in output: {stdout}"
    run_dir = run_dir_line[0].split("Run Directory: ")[1].strip()
    manifest_path = os.path.join(run_dir, "manifest.jsonl")
    assert os.path.exists(manifest_path)
    with open(manifest_path, "r") as f:
        lines = f.readlines()
    assert len(lines) == 1
    return json.loads(lines[0])


def test_capture_writes_schema_valid_manifest():
    result = runner.invoke(app, ["capture"])
    assert result.exit_code == 0
    record = _read_manifest_record(result.stdout)
    assert "config_hash" in record
    assert "timestamp" in record
    assert record["metrics"]["status"] == "captured"


def test_encode_writes_complete_manifest():
    result = runner.invoke(app, ["encode"])
    assert result.exit_code == 0
    record = _read_manifest_record(result.stdout)
    assert "config_hash" in record
    assert record["metrics"]["status"] == "encoded"
    assert record["metrics"]["exact_bytes"] > 0


def test_decode_writes_complete_manifest():
    result = runner.invoke(app, ["decode"])
    assert result.exit_code == 0
    record = _read_manifest_record(result.stdout)
    assert record["metrics"]["status"] == "decoded"
    assert record["metrics"]["exact_match"] is True


def test_evaluate_writes_measured_not_fabricated_result():
    result = runner.invoke(app, ["evaluate"])
    assert result.exit_code == 0
    record = _read_manifest_record(result.stdout)
    # Must be a real measured boolean from an actual codec round-trip,
    # not a hardcoded placeholder score.
    assert record["metrics"]["status"] == "evaluated"
    assert isinstance(record["metrics"]["exact_match"], bool)
    assert record["metrics"]["exact_match"] is True
    assert record["metrics"]["serialized_bytes"] > 0
