import hashlib
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from fairfuzzkv_codec.core.config import FairFuzzKVConfig

def set_seed(seed: int) -> None:
    """Lock deterministic seeds across all randomness sources."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def compute_config_hash(config: FairFuzzKVConfig) -> str:
    """Compute a SHA-256 hash of the configuration to ensure strict provenance."""
    # Convert to JSON with sorted keys to ensure stable hashing
    config_dict = config.model_dump(mode='json')
    stable_json = json.dumps(config_dict, sort_keys=True).encode('utf-8')
    return hashlib.sha256(stable_json).hexdigest()

class ExecutionManager:
    def __init__(self, base_dir: str = "results"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
    def create_run_directory(self, config: FairFuzzKVConfig, run_name: str = "run") -> Path:
        """Create a result directory convention that prevents accidental overwrites."""
        config_hash = compute_config_hash(config)[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        run_dir = self.base_dir / f"{run_name}_{config.model.model_name}_{config_hash}_{timestamp}"
        
        # In the extreme edge case of exact same second, append a counter
        counter = 1
        while run_dir.exists():
            run_dir = self.base_dir / f"{run_name}_{config.model.model_name}_{config_hash}_{timestamp}_{counter}"
            counter += 1
            
        run_dir.mkdir(parents=True, exist_ok=False)
        return run_dir

    def write_manifest(self, run_dir: Path, config: FairFuzzKVConfig, metrics: Dict[str, Any]) -> None:
        """Write a structured JSONL run manifest."""
        manifest_path = run_dir / "manifest.jsonl"
        
        record = {
            "timestamp": datetime.now().isoformat(),
            "config_hash": compute_config_hash(config),
            "config": config.model_dump(mode='json'),
            "metrics": metrics
        }
        
        with open(manifest_path, "a") as f:
            f.write(json.dumps(record) + "\n")
