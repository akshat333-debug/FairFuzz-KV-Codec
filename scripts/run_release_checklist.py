"""Automated release checklist (Prompt 20 item 135).

Runs the checks a grader or an independent reproducer would run, and reports
PASS/FAIL per item with the evidence. Nothing here asserts success on its own
authority - every item either executes a real command or inspects a real file.

    uv run python scripts/run_release_checklist.py            # fast checks
    uv run python scripts/run_release_checklist.py --full     # + real-model experiments

The fast path deliberately skips the heavy real-model studies (they need model
downloads and minutes of compute); those are marked SKIPPED, never PASSED.
"""

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))


@dataclass
class CheckResult:
    name: str
    status: str  # PASS | FAIL | SKIPPED
    detail: str = ""
    seconds: float = 0.0
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _run(cmd: List[str], timeout: int = 3600) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=timeout)


def check_environment_install() -> CheckResult:
    """`uv sync` must resolve and build the package from the lock file."""
    t0 = time.perf_counter()
    proc = _run(["uv", "sync", "--frozen"], timeout=900)
    ok = proc.returncode == 0
    return CheckResult(
        "environment_install (uv sync --frozen)",
        "PASS" if ok else "FAIL",
        detail="lock file resolves and the package builds" if ok else proc.stderr[-500:],
        seconds=time.perf_counter() - t0,
    )


def check_lint_and_types() -> CheckResult:
    t0 = time.perf_counter()
    ruff = _run(["uv", "run", "--active", "ruff", "check", "."], timeout=600)
    mypy = _run(["uv", "run", "--active", "mypy", "."], timeout=1200)
    ok = ruff.returncode == 0 and mypy.returncode == 0
    return CheckResult(
        "lint_and_types (ruff + mypy)",
        "PASS" if ok else "FAIL",
        detail=("both clean" if ok else f"ruff={ruff.returncode} mypy={mypy.returncode}"),
        seconds=time.perf_counter() - t0,
        evidence={"mypy_tail": mypy.stdout.strip().splitlines()[-1:] if mypy.stdout else []},
    )


def check_test_suite() -> CheckResult:
    t0 = time.perf_counter()
    proc = _run(["uv", "run", "--active", "pytest", "-q"], timeout=3600)
    last = [ln for ln in proc.stdout.strip().splitlines() if "passed" in ln or "failed" in ln]
    return CheckResult(
        "test_suite (pytest)",
        "PASS" if proc.returncode == 0 else "FAIL",
        detail=last[-1] if last else proc.stdout[-300:],
        seconds=time.perf_counter() - t0,
    )


def check_binary_compatibility() -> CheckResult:
    """Golden bitstreams must still decode byte-identically - the format
    compatibility guarantee. Failure here means a released bitstream would
    no longer be readable."""
    t0 = time.perf_counter()
    proc = _run(
        ["uv", "run", "--active", "pytest", "-q",
         "tests/metadata_coding/test_golden_vectors.py", "tests/decoder", "-q"],
        timeout=900,
    )
    last = [ln for ln in proc.stdout.strip().splitlines() if "passed" in ln or "failed" in ln]
    return CheckResult(
        "binary_compatibility (golden vectors + container round-trip)",
        "PASS" if proc.returncode == 0 else "FAIL",
        detail=last[-1] if last else proc.stdout[-300:],
        seconds=time.perf_counter() - t0,
    )


def check_gate_decision_records() -> CheckResult:
    """Acceptance gate: all four gates must have immutable decision records."""
    from fairfuzzkv_codec.dashboard.artifacts import gate_summary

    rows = gate_summary(REPO)
    missing = [r["gate"] for r in rows if r["status"] != "ok"]
    decisions = {r["gate"]: r["decision"] for r in rows}
    return CheckResult(
        "gate_decision_records (all four gates)",
        "PASS" if not missing else "FAIL",
        detail=("; ".join(f"{k}={v}" for k, v in decisions.items())
                if not missing else f"missing records: {missing}"),
        evidence={"decisions": decisions},
    )


def check_gate_reproducibility_from_raw() -> CheckResult:
    """Gates 1 and 4 must recompute from committed raw predictions WITHOUT
    model access - the strongest reproducibility property this project has."""
    t0 = time.perf_counter()
    details, ok = [], True
    try:
        # Loaded by file path rather than `from scripts...` so `scripts/` never
        # becomes an importable package - that made mypy see the same file under
        # two module names and broke type checking.
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "_gate1_study_mod", REPO / "scripts" / "run_gate1_study.py"
        )
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        g1 = mod.compute_gate1_from_predictions(REPO / "gate1_study" / "predictions.jsonl")
        details.append(f"Gate1={g1.decision.value}")
        ok &= g1.decision.value == "WEAK_PASS"
    except Exception as e:  # noqa: BLE001
        details.append(f"Gate1 ERROR {e}")
        ok = False
    try:
        from fairfuzzkv_codec.evaluation.gate4 import compute_gate4_from_predictions

        g4 = compute_gate4_from_predictions(str(REPO / "gate4_fairness_study" / "predictions.jsonl"), n_boot=200)
        details.append(f"Gate4={g4.decision.value}")
        ok &= g4.decision.value == "FAIL"
    except Exception as e:  # noqa: BLE001
        details.append(f"Gate4 ERROR {e}")
        ok = False
    return CheckResult(
        "gate_reproducibility_from_raw_predictions",
        "PASS" if ok else "FAIL",
        detail="; ".join(details) + " (recomputed from raw, no model access)",
        seconds=time.perf_counter() - t0,
    )


def check_dashboard_launch() -> CheckResult:
    """Every dashboard page must render against the frozen artifacts."""
    t0 = time.perf_counter()
    proc = _run(["uv", "run", "--active", "pytest", "-q", "tests/dashboard"], timeout=1800)
    last = [ln for ln in proc.stdout.strip().splitlines() if "passed" in ln or "failed" in ln]
    return CheckResult(
        "dashboard_launch (all pages render via AppTest)",
        "PASS" if proc.returncode == 0 else "FAIL",
        detail=last[-1] if last else proc.stdout[-300:],
        seconds=time.perf_counter() - t0,
    )


def check_report_generation() -> CheckResult:
    """The offline demo export must regenerate from the frozen artifacts."""
    t0 = time.perf_counter()
    proc = _run(["uv", "run", "--active", "python", "scripts/export_demo_assets.py"], timeout=600)
    ok = proc.returncode == 0 and (REPO / "demo_assets" / "demo.html").exists()
    return CheckResult(
        "report_generation (offline demo export)",
        "PASS" if ok else "FAIL",
        detail=proc.stdout.strip().splitlines()[-1] if ok and proc.stdout else proc.stderr[-300:],
        seconds=time.perf_counter() - t0,
    )


def check_required_documents() -> CheckResult:
    required = [
        "README.md", "ARCHITECTURE.md", "FORMAT.md", "CLAIMS_LEDGER.md",
        "RISK_REGISTER.md", "PENDING.md", "SPEC_TRACEABILITY.md", "PERFORMANCE.md",
        "ALLOCATION_MATH.md", "DEMO_SCRIPT.md", "FINAL_REPORT.md", "VIVA_PACK.md",
        "REPRODUCIBILITY.md", "CLAIMS_AUDIT.md", "JOURNAL_EXPANSION_PLAN.md",
        "MODEL_CARD.md", "CITATION.cff",
    ]
    missing = [d for d in required if not (REPO / d).exists()]
    return CheckResult(
        "required_release_documents",
        "PASS" if not missing else "FAIL",
        detail=f"{len(required) - len(missing)}/{len(required)} present"
               + (f"; MISSING: {missing}" if missing else ""),
        evidence={"missing": missing},
    )


def check_dataset_regeneration(full: bool) -> CheckResult:
    if not full:
        return CheckResult(
            "dataset_regeneration (IndicLongComp)", "SKIPPED",
            detail="needs a real model download; run with --full",
        )
    t0 = time.perf_counter()
    proc = _run(["uv", "run", "--active", "python", "scripts/run_indiclongcomp_study.py"], timeout=3600)
    return CheckResult(
        "dataset_regeneration (IndicLongComp)",
        "PASS" if proc.returncode == 0 else "FAIL",
        detail=proc.stdout.strip().splitlines()[-1] if proc.stdout else proc.stderr[-300:],
        seconds=time.perf_counter() - t0,
    )


def check_core_experiments(full: bool) -> CheckResult:
    if not full:
        return CheckResult(
            "core_experiments (smoke demo + systems profile)", "SKIPPED",
            detail="needs a real model download; run with --full",
        )
    t0 = time.perf_counter()
    demo = _run(["uv", "run", "--active", "python", "scripts/demo.py"], timeout=1800)
    prof = _run(["uv", "run", "--active", "python", "scripts/run_systems_profile.py"], timeout=3600)
    ok = demo.returncode == 0 and prof.returncode == 0
    return CheckResult(
        "core_experiments (smoke demo + systems profile)",
        "PASS" if ok else "FAIL",
        detail=f"demo rc={demo.returncode}, profile rc={prof.returncode}",
        seconds=time.perf_counter() - t0,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="also run real-model experiments")
    parser.add_argument("--skip-install", action="store_true", help="skip uv sync (already installed)")
    args = parser.parse_args()

    checks: List[CheckResult] = []
    if not args.skip_install:
        checks.append(check_environment_install())
    checks.append(check_lint_and_types())
    checks.append(check_test_suite())
    checks.append(check_binary_compatibility())
    checks.append(check_gate_decision_records())
    checks.append(check_gate_reproducibility_from_raw())
    checks.append(check_dashboard_launch())
    checks.append(check_report_generation())
    checks.append(check_required_documents())
    checks.append(check_dataset_regeneration(args.full))
    checks.append(check_core_experiments(args.full))

    print("\n" + "=" * 78)
    print("RELEASE CHECKLIST")
    print("=" * 78)
    for c in checks:
        mark = {"PASS": "PASS ", "FAIL": "FAIL ", "SKIPPED": "SKIP "}[c.status]
        print(f"[{mark}] {c.name}")
        if c.detail:
            print(f"         {c.detail}")
        if c.seconds:
            print(f"         ({c.seconds:.1f}s)")

    failed = [c for c in checks if c.status == "FAIL"]
    skipped = [c for c in checks if c.status == "SKIPPED"]
    print("=" * 78)
    print(f"{len(checks) - len(failed) - len(skipped)} passed, {len(failed)} failed, {len(skipped)} skipped")
    if skipped:
        print("SKIPPED items are NOT passes - rerun with --full to execute them.")

    out = REPO / "release" / "release_checklist.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({
        "checks": [c.to_dict() for c in checks],
        "passed": len(checks) - len(failed) - len(skipped),
        "failed": len(failed),
        "skipped": len(skipped),
        "full_mode": args.full,
    }, indent=2), encoding="utf-8")
    print(f"saved -> {out}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
