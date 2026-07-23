import typer
from typing import Optional
from pathlib import Path

from fairfuzzkv_codec.core.config import FairFuzzKVConfig, ModelIdentity, CodecBudget, QuantizerConfig, QuantizerType, HardwareManifest
from fairfuzzkv_codec.core.execution import ExecutionManager
from fairfuzzkv_codec.fixtures.synthetic import generate_synthetic_kv_cache
from fairfuzzkv_codec.codec.noop import NoOpCodec

app = typer.Typer(help="FairFuzzKV-Codec CLI")

def load_default_config() -> FairFuzzKVConfig:
    # A dummy default config for demo/fixture purposes
    return FairFuzzKVConfig(
        model=ModelIdentity(model_name="llama-7b-fixture", layer_count=32, head_count=32, head_dim=128),
        budget=CodecBudget(total_bits_per_element=16.0),
        quantizer=QuantizerConfig(quantizer_type=QuantizerType.NOOP),
        hardware=HardwareManifest(device="cpu", dtype="float16")
    )

def setup_execution(config_path: Optional[Path]) -> tuple[FairFuzzKVConfig, ExecutionManager, Path]:
    if config_path and config_path.exists():
        with open(config_path, "r") as f:
            config = FairFuzzKVConfig.model_validate_json(f.read())
    else:
        config = load_default_config()
        
    manager = ExecutionManager()
    run_dir = manager.create_run_directory(config)
    return config, manager, run_dir

@app.command()
def inspect(config_path: Optional[Path] = None):
    """Inspect the current configuration and execution environment."""
    config, manager, run_dir = setup_execution(config_path)
    typer.echo(f"Run Directory: {run_dir}")
    typer.echo(f"Config: {config.model_dump_json(indent=2)}")
    manager.write_manifest(run_dir, config, {"status": "inspected"})

@app.command()
def capture(config_path: Optional[Path] = None):
    """Capture raw KV-cache activations (or generate synthetic fixture)."""
    config, manager, run_dir = setup_execution(config_path)
    typer.echo(f"Run Directory: {run_dir}")
    typer.echo("Generating synthetic KV-cache fixture...")
    tensor = generate_synthetic_kv_cache(
        layers=config.model.layer_count,
        batch=1,
        heads=config.model.head_count,
        sequence=10,
        head_dim=config.model.head_dim,
        device=config.hardware.device
    )
    typer.echo(f"Captured shape: {tensor.shape}")
    manager.write_manifest(run_dir, config, {"status": "captured", "shape": list(tensor.shape)})

@app.command()
def encode(config_path: Optional[Path] = None):
    """Encode KV-cache using the selected codec."""
    config, manager, run_dir = setup_execution(config_path)
    typer.echo(f"Run Directory: {run_dir}")
    tensor = generate_synthetic_kv_cache(
        layers=config.model.layer_count, batch=1, heads=config.model.head_count, sequence=10, head_dim=config.model.head_dim, device=config.hardware.device
    )
    codec = NoOpCodec()
    byte_stream, metadata = codec.encode_prefill(tensor)
    typer.echo(f"Encoded bytes: {len(byte_stream)}")
    manager.write_manifest(run_dir, config, {"status": "encoded", "exact_bytes": metadata['exact_bytes']})

@app.command()
def decode(config_path: Optional[Path] = None):
    """Decode KV-cache from a byte stream."""
    config, manager, run_dir = setup_execution(config_path)
    typer.echo(f"Run Directory: {run_dir}")
    # Simulate encode -> decode pipeline
    tensor = generate_synthetic_kv_cache(
        layers=config.model.layer_count, batch=1, heads=config.model.head_count, sequence=10, head_dim=config.model.head_dim, device=config.hardware.device
    )
    codec = NoOpCodec()
    byte_stream, metadata = codec.encode_prefill(tensor)
    
    import torch
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = dtype_map[config.hardware.dtype]
    
    reconstructed = codec.decode(byte_stream, metadata, tuple(metadata["shape"]), dtype, config.hardware.device)
    match = torch.all(tensor == reconstructed).item()
    typer.echo(f"Decoded successfully. Exact match: {match}")
    manager.write_manifest(run_dir, config, {"status": "decoded", "exact_match": match})

@app.command()
def evaluate(config_path: Optional[Path] = None):
    """Run a NoOp round-trip evaluation on the fixture and report the measured exact-match result."""
    config, manager, run_dir = setup_execution(config_path)
    typer.echo(f"Run Directory: {run_dir}")
    typer.echo("Evaluating codec round-trip on fixture...")
    tensor = generate_synthetic_kv_cache(
        layers=config.model.layer_count, batch=1, heads=config.model.head_count, sequence=10, head_dim=config.model.head_dim, device=config.hardware.device
    )
    codec = NoOpCodec()
    byte_stream, metadata = codec.encode_prefill(tensor)

    import torch
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = dtype_map[config.hardware.dtype]
    reconstructed = codec.decode(byte_stream, metadata, tuple(metadata["shape"]), dtype, config.hardware.device)
    exact_match = torch.all(tensor == reconstructed).item()

    typer.echo(f"Exact match: {exact_match}")
    manager.write_manifest(run_dir, config, {"status": "evaluated", "exact_match": exact_match, "serialized_bytes": len(byte_stream)})

@app.command()
def dashboard(config_path: Optional[Path] = None):
    """Generate HTML dashboard summary from manifests."""
    config, manager, run_dir = setup_execution(config_path)
    typer.echo(f"Run Directory: {run_dir}")
    typer.echo("Generating dashboard...")
    html_path = run_dir / "dashboard.html"
    with open(html_path, "w") as f:
        f.write("<html><body><h1>FairFuzzKV-Codec Dashboard</h1><p>Demo successful.</p></body></html>")
    manager.write_manifest(run_dir, config, {"status": "dashboard_generated", "path": str(html_path)})

if __name__ == "__main__":
    app()
