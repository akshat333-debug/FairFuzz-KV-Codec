"""Frozen-artifact loading and provenance for the research dashboard.

Two rules encoded here, both from Prompt 19's acceptance gates:

1. **No fabricated or placeholder metrics in production mode.** A missing
   artifact yields `ArtifactStatus.MISSING` with the exact command that would
   produce it - never zeros, never sample data. The dashboard renders that
   status instead of a chart.
2. **Never hide unfavorable results.** `matched_bit_warning()` flags a
   comparison whose budgets are not matched, and the gate summary surfaces
   FAIL/WEAK_PASS outcomes with the same prominence as PASS.
"""

import json
import re
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[3]


class ArtifactStatus(str, Enum):
    OK = "ok"
    MISSING = "missing"
    UNREADABLE = "unreadable"


@dataclass
class Artifact:
    name: str
    path: Path
    status: ArtifactStatus
    data: Optional[Any] = None
    produced_by: str = ""
    error: str = ""

    @property
    def available(self) -> bool:
        return self.status == ArtifactStatus.OK

    def missing_message(self) -> str:
        if self.status == ArtifactStatus.MISSING:
            return (
                f"Artifact `{self.path}` has not been generated yet. "
                f"Run: `{self.produced_by}`\n\n"
                f"No placeholder numbers are shown - this panel stays empty until "
                f"a real run produces real data."
            )
        if self.status == ArtifactStatus.UNREADABLE:
            return f"Artifact `{self.path}` exists but could not be parsed: {self.error}"
        return ""


# name -> (relative path, command that produces it)
ARTIFACT_REGISTRY: Dict[str, tuple] = {
    "gate1": ("gate1_study/GATE1_REPORT.json", "uv run python scripts/run_gate1_study.py"),
    "gate2": ("gate2_fairness_study/GATE2_REPORT.json", "uv run python scripts/run_gate2_study.py"),
    "gate3": ("gate3_study/gate3_report.json", "uv run python scripts/run_gate3_study.py"),
    "gate3_interactions": ("gate3_study/gate3_interactions.json", "uv run python scripts/run_gate3_interactions.py"),
    "gate4": ("gate4_fairness_study/gate4_report.json", "uv run python scripts/run_gate4_study.py"),
    "quantization_benchmark": ("quantization_benchmark/benchmark_results.json", "uv run python scripts/run_quantization_benchmark.py"),
    "lbg_benchmark": ("lbg_benchmark/lbg_vs_scalar.json", "uv run python scripts/run_lbg_benchmark.py"),
    "allocation": ("allocation_study/allocation_result.json", "uv run python scripts/run_allocation_demo.py"),
    "minimax": ("gate2_study/gate2_result.json", "uv run python scripts/run_minimax_demo.py"),
    "baseline_matrix": ("baseline_matrix_study/result_tables.json", "uv run python scripts/run_baseline_matrix.py"),
    "baseline_cards": ("baseline_matrix_study/baseline_cards.json", "uv run python scripts/run_baseline_matrix.py"),
    "systems_profile": ("systems_profile/systems_profile.json", "uv run python scripts/run_systems_profile.py"),
    "repair_scoring": ("repair_scoring_study/scorer_comparison.json", "uv run python scripts/run_repair_scoring_demo.py"),
    "cross_tokenizer": ("cross_tokenizer_study/cross_tokenizer_stability.json", "uv run python scripts/run_cross_tokenizer_stability.py"),
    "indic_course_card": ("indic_longcomp_study/course/dataset_card.json", "uv run python scripts/run_indiclongcomp_study.py"),
    "indic_journal_card": ("indic_longcomp_study/journal/dataset_card.json", "uv run python scripts/run_indiclongcomp_study.py"),
}


def load_artifact(name: str, root: Optional[Path] = None) -> Artifact:
    root = root or REPO_ROOT
    if name not in ARTIFACT_REGISTRY:
        raise KeyError(f"unknown artifact '{name}'; known: {sorted(ARTIFACT_REGISTRY)}")
    rel, command = ARTIFACT_REGISTRY[name]
    path = root / rel
    if not path.exists():
        return Artifact(name=name, path=path, status=ArtifactStatus.MISSING, produced_by=command)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        return Artifact(name=name, path=path, status=ArtifactStatus.UNREADABLE, produced_by=command, error=str(e))
    return Artifact(name=name, path=path, status=ArtifactStatus.OK, data=data, produced_by=command)


def load_jsonl(rel_path: str, root: Optional[Path] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    root = root or REPO_ROOT
    path = root / rel_path
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
    return rows


# ---- provenance (item 132) -------------------------------------------------

@dataclass
class Provenance:
    git_commit: str
    artifact_path: str
    model: str = "unknown"
    tokenizer: str = "unknown"
    seed: Optional[int] = None
    config_hash: str = "unknown"
    hardware: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "git_commit": self.git_commit, "artifact_path": self.artifact_path,
            "model": self.model, "tokenizer": self.tokenizer, "seed": self.seed,
            "config_hash": self.config_hash, "hardware": self.hardware,
        }


def git_commit(root: Optional[Path] = None) -> str:
    try:
        root = root or REPO_ROOT
        sha = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()
        return f"{sha}{'-dirty' if dirty else ''}"
    except Exception:  # noqa: BLE001
        return "unavailable"


def build_provenance(artifact: Artifact, root: Optional[Path] = None) -> Provenance:
    """Extract whatever provenance the artifact itself carries. Absent fields
    stay "unknown" rather than being invented."""
    data = artifact.data if isinstance(artifact.data, dict) else {}
    hardware = data.get("hardware_manifest", {}) if isinstance(data.get("hardware_manifest"), dict) else {}
    seed = data.get("seed")
    if seed is None and isinstance(data.get("seeds"), list) and data["seeds"]:
        seed = data["seeds"][0]
    return Provenance(
        git_commit=git_commit(root),
        artifact_path=str(artifact.path),
        model=str(data.get("model_name") or data.get("model") or hardware.get("device") or "unknown"),
        tokenizer=str(data.get("tokenizer_family") or data.get("tokenizer") or "unknown"),
        seed=seed if isinstance(seed, int) else None,
        config_hash=str(data.get("config_hash", "unknown")),
        hardware=hardware,
    )


# ---- matched-bit guard (item 131) ------------------------------------------

def matched_bit_warning(
    bits_by_system: Dict[str, float], tolerance: float = 0.5
) -> Optional[str]:
    """Return a warning string when the compared systems are NOT at matched
    bits, else None. The dashboard must refuse to present such a comparison as
    a like-for-like result."""
    values = [v for v in bits_by_system.values() if v is not None]
    if len(values) < 2:
        return None
    spread = max(values) - min(values)
    if spread > tolerance:
        detail = ", ".join(f"{k}={v:.2f}" for k, v in sorted(bits_by_system.items()))
        return (
            f"UNMATCHED BUDGETS: bits/element spread is {spread:.2f}, above the "
            f"{tolerance} tolerance ({detail}). This is NOT a like-for-like "
            f"comparison and must not be read as one."
        )
    return None


# ---- claims ledger panel (item 133) ----------------------------------------

@dataclass
class Claim:
    claim_id: str
    description: str
    validation: str
    status: str

    @property
    def is_negative(self) -> bool:
        s = self.status.upper()
        return "FAIL" in s or "WEAK_PASS" in s


def load_claims(root: Optional[Path] = None) -> List[Claim]:
    """Parse CLAIMS_LEDGER.md's table. The dashboard's claims panel is driven
    from the ledger file itself, so it cannot drift from the source of truth."""
    root = root or REPO_ROOT
    path = root / "CLAIMS_LEDGER.md"
    if not path.exists():
        return []
    claims: List[Claim] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| C-"):
            continue
        cells = [c.strip() for c in line.split("|")[1:-1]]
        if len(cells) < 4:
            continue
        claims.append(Claim(claim_id=cells[0], description=cells[1], validation=cells[2], status=cells[3]))
    return claims


def load_limitations(root: Optional[Path] = None) -> List[str]:
    """Section headings from PENDING.md - the honest limitations list."""
    root = root or REPO_ROOT
    path = root / "PENDING.md"
    if not path.exists():
        return []
    return [
        re.sub(r"^##\s*", "", line).strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith("## ")
    ]


def gate_summary(root: Optional[Path] = None) -> List[Dict[str, str]]:
    """Every gate's decision, including the negative ones. Gate decisions are
    read from the frozen reports, never hardcoded here."""
    out: List[Dict[str, str]] = []
    for gate, key in (("Gate 1", "gate1"), ("Gate 2", "gate2"), ("Gate 3", "gate3"), ("Gate 4", "gate4")):
        art = load_artifact(key, root)
        decision = "not run"
        if art.available and isinstance(art.data, dict):
            data = art.data
            report = data.get("report", data)
            decision = str(
                report.get("decision")
                or data.get("decision")
                or (data.get("gate3_decision") if isinstance(data.get("gate3_decision"), str) else None)
                or "unknown"
            )
        out.append({"gate": gate, "decision": decision, "artifact": str(art.path), "status": art.status.value})
    return out
