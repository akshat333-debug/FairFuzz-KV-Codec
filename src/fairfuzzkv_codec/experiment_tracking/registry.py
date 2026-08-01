"""Experiment tracking: an append-only JSONL registry of study runs.

The last empty module from the Prompt 1 package list. Deliberately minimal and
dependency-free (no MLflow/W&B): this project's studies already write their own
artifacts, so what was actually missing is a single queryable index tying each
run to its code version, config, seeds, metrics, and artifact paths.

Append-only by design - a run record is never rewritten, so a later run cannot
quietly restate an earlier result. `git_commit` is captured automatically
(with a `-dirty` marker) so a recorded metric can always be traced to the code
that produced it.
"""

import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

REGISTRY_SCHEMA_VERSION = 1
DEFAULT_REGISTRY_PATH = Path("experiment_registry.jsonl")


def git_commit() -> str:
    """Short git SHA of the working tree, with `-dirty` when uncommitted
    changes exist. Returns an explicit marker (never a fabricated hash) when
    git metadata is unavailable."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()
        return f"{sha}{'-dirty' if dirty else ''}"
    except Exception:  # noqa: BLE001
        return "unavailable"


@dataclass
class RunRecord:
    schema_version: int
    run_id: str
    study: str  # e.g. "gate1", "gate4", "baseline_matrix"
    timestamp_utc: str
    git_commit: str
    config: Dict[str, Any] = field(default_factory=dict)
    seeds: List[int] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ExperimentRegistry:
    """Append-only run index backed by one JSONL file."""

    def __init__(self, path: Path = DEFAULT_REGISTRY_PATH):
        self.path = Path(path)

    def log_run(
        self,
        study: str,
        config: Optional[Dict[str, Any]] = None,
        seeds: Optional[List[int]] = None,
        metrics: Optional[Dict[str, float]] = None,
        artifacts: Optional[List[str]] = None,
        notes: str = "",
        run_id: Optional[str] = None,
    ) -> RunRecord:
        """Append one run. `run_id` defaults to `{study}_{utc-timestamp}`, which
        is unique per second per study; pass an explicit id for determinism in
        tests or when a caller has its own naming scheme."""
        now = datetime.now(timezone.utc)
        record = RunRecord(
            schema_version=REGISTRY_SCHEMA_VERSION,
            run_id=run_id or f"{study}_{now.strftime('%Y%m%dT%H%M%S')}",
            study=study,
            timestamp_utc=now.isoformat(),
            git_commit=git_commit(),
            config=dict(config or {}),
            seeds=list(seeds or []),
            metrics=dict(metrics or {}),
            artifacts=list(artifacts or []),
            notes=notes,
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record.to_dict()) + "\n")
        return record

    def all_runs(self) -> List[RunRecord]:
        if not self.path.exists():
            return []
        runs: List[RunRecord] = []
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    runs.append(RunRecord(**json.loads(line)))
        return runs

    def runs_for(self, study: str) -> List[RunRecord]:
        return [r for r in self.all_runs() if r.study == study]

    def latest(self, study: str) -> Optional[RunRecord]:
        runs = self.runs_for(study)
        return runs[-1] if runs else None

    def metric_history(self, study: str, metric: str) -> List[tuple]:
        """[(run_id, value)] for one metric across a study's runs - the point of
        an index: seeing whether a number moved between runs. Runs missing the
        metric are skipped rather than reported as 0.0."""
        return [(r.run_id, r.metrics[metric]) for r in self.runs_for(study) if metric in r.metrics]
