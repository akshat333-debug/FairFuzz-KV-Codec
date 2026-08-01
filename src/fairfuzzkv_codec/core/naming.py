"""Project claims switch, driven automatically by the frozen Gate 4 decision
(`evaluation.gate4.Gate4Decision`). Per Prompt 14's non-negotiable
instruction ("do not postpone the naming decision"), this is applied by
`scripts/run_gate4_study.py` immediately after the decision is frozen, not
as a manual follow-up - the claim framing FOLLOWS the decision file, and the
decision file never follows a desired framing.

**Project name is a fixed, owner-chosen constant** (`FairFuzzKV-Codec`) and
is deliberately NOT switched by the gate outcome. Renaming the project was
reconsidered and rejected by the project owner: the distribution name is
identity/branding, not a scientific claim, and swapping it also broke the
build (the import package is `fairfuzzkv_codec`, so a `fragkv-codec`
distribution name made uv_build look for a `src/fragkv_codec/` that does not
exist). What must follow the evidence is the CLAIM FRAMING, and it still
does - a Gate 4 FAIL is reported below in full, unsoftened. Keeping the name
is explicitly NOT a claim that fuzzy scoring was validated; see
`GATE4_REPORT.md`, RISK_REGISTER R-10, CLAIMS_LEDGER C-23.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from fairfuzzkv_codec.evaluation.gate4 import Gate4Decision


@dataclass(frozen=True)
class ProjectIdentity:
    decision: Gate4Decision
    display_name: str
    package_name: str  # pip-normalized (lowercase, hyphenated)
    claim_framing: str


_IDENTITY_BY_DECISION: Dict[Gate4Decision, ProjectIdentity] = {
    Gate4Decision.PASS: ProjectIdentity(
        decision=Gate4Decision.PASS,
        display_name="FairFuzzKV-Codec",
        package_name="fairfuzzkv-codec",
        claim_framing=(
            "Fuzzy repair-priority scoring (Module 3) is validated: it beats no-repair and is "
            "not dominated by the simpler competitors, consistently, at matched bits (Gate 4 PASS)."
        ),
    ),
    Gate4Decision.WEAK_PASS: ProjectIdentity(
        decision=Gate4Decision.WEAK_PASS,
        display_name="FairFuzzKV-Codec",
        package_name="fairfuzzkv-codec",
        claim_framing=(
            "Fuzzy repair-priority scoring (Module 3) shows a real but modest or inconsistent "
            "benefit over no-repair (Gate 4 WEAK_PASS) - keep the name, do not claim a validated win."
        ),
    ),
    Gate4Decision.FAIL: ProjectIdentity(
        decision=Gate4Decision.FAIL,
        display_name="FairFuzzKV-Codec",
        package_name="fairfuzzkv-codec",
        claim_framing=(
            "Fuzzy repair-priority scoring did not beat no-repair and/or the simpler competitors "
            "(Gate 4 FAIL) - negative evidence, not fabricated into a claim. The 'Fuzzy' in the "
            "project name is historical identity only and must NOT be read as a validated claim: "
            "the fuzzy scorer remains in the codebase as an optional, NON-DEFAULT scorer. The codec "
            "(capture, Unicode grouping, fragility estimation, quantization, pruning, allocation, "
            "metadata coding, decoder) is unaffected and preserved; its surviving, evidence-grounded "
            "contribution is tokenizer-fragmentation-aware KV compression."
        ),
    ),
}


def resolve_project_identity(decision: Gate4Decision) -> ProjectIdentity:
    return _IDENTITY_BY_DECISION[decision]


_PYPROJECT_NAME_PATTERN = re.compile(r'^name = "[^"]*"$', re.MULTILINE)


def apply_project_identity(identity: ProjectIdentity, repo_root: Path) -> Dict[str, bool]:
    """Rewrites the `[project] name` field in pyproject.toml to match the
    resolved identity, and writes PROJECT_IDENTITY.json as the inspectable,
    versioned record of which decision produced the current name. Returns
    which files were actually changed (so a re-run with an unchanged
    decision is a no-op, not a spurious diff)."""
    changed = {"pyproject.toml": False, "PROJECT_IDENTITY.json": False}

    pyproject_path = repo_root / "pyproject.toml"
    text = pyproject_path.read_text(encoding="utf-8")
    new_text, n = _PYPROJECT_NAME_PATTERN.subn(f'name = "{identity.package_name}"', text, count=1)
    if n == 1 and new_text != text:
        pyproject_path.write_text(new_text, encoding="utf-8")
        changed["pyproject.toml"] = True

    import json

    identity_path = repo_root / "PROJECT_IDENTITY.json"
    payload = {
        "gate4_decision": identity.decision.value,
        "display_name": identity.display_name,
        "package_name": identity.package_name,
        "claim_framing": identity.claim_framing,
    }
    new_json = json.dumps(payload, indent=2) + "\n"
    if not identity_path.exists() or identity_path.read_text(encoding="utf-8") != new_json:
        identity_path.write_text(new_json, encoding="utf-8")
        changed["PROJECT_IDENTITY.json"] = True

    return changed
