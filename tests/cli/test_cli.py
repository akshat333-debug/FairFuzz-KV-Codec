from typer.testing import CliRunner
from fairfuzzkv_codec.cli import app
import json
import os

runner = CliRunner()

def test_cli_manifest_generation():
    result = runner.invoke(app, ["inspect"])
    assert result.exit_code == 0
    assert "Run Directory:" in result.stdout
    
    # Extract run directory from stdout
    run_dir_line = [line for line in result.stdout.split("\n") if "Run Directory:" in line][0]
    run_dir = run_dir_line.split("Run Directory: ")[1].strip()
    
    manifest_path = os.path.join(run_dir, "manifest.jsonl")
    assert os.path.exists(manifest_path)
    
    with open(manifest_path, "r") as f:
        lines = f.readlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert "config" in record
        assert "metrics" in record
        assert record["metrics"]["status"] == "inspected"
