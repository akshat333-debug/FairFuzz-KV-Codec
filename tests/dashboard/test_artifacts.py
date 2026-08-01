"""Dashboard artifact-loading and honesty-guard tests.

The acceptance gates these enforce: no fabricated/placeholder metrics, gate
outcomes and limitations visible, unmatched budgets flagged.
"""

import json

import pytest

from fairfuzzkv_codec.dashboard.artifacts import (
    ARTIFACT_REGISTRY,
    ArtifactStatus,
    build_provenance,
    gate_summary,
    load_artifact,
    load_claims,
    load_limitations,
    matched_bit_warning,
)


def test_missing_artifact_reports_status_and_command_not_placeholder_data(tmp_path):
    """The core 'no placeholder metrics' guard: a missing artifact must yield
    MISSING + the command to produce it, and carry NO data."""
    art = load_artifact("gate1", root=tmp_path)
    assert art.status is ArtifactStatus.MISSING
    assert art.data is None
    assert not art.available
    msg = art.missing_message()
    assert "has not been generated yet" in msg
    assert "scripts/run_gate1_study.py" in msg
    assert "No placeholder numbers" in msg


def test_unreadable_artifact_is_reported_not_silently_empty(tmp_path):
    path = tmp_path / "gate1_study" / "GATE1_REPORT.json"
    path.parent.mkdir(parents=True)
    path.write_text("{ this is not json")
    art = load_artifact("gate1", root=tmp_path)
    assert art.status is ArtifactStatus.UNREADABLE
    assert art.data is None
    assert "could not be parsed" in art.missing_message()


def test_valid_artifact_loads(tmp_path):
    path = tmp_path / "gate1_study" / "GATE1_REPORT.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"decision": "WEAK_PASS"}))
    art = load_artifact("gate1", root=tmp_path)
    assert art.available and art.data["decision"] == "WEAK_PASS"


def test_unknown_artifact_name_raises():
    with pytest.raises(KeyError):
        load_artifact("no_such_artifact")


def test_every_registered_artifact_declares_a_producing_command():
    for name, (rel, command) in ARTIFACT_REGISTRY.items():
        assert rel and command.startswith("uv run"), name


# ---- gate visibility -------------------------------------------------------

def test_gate_summary_surfaces_real_decisions_including_negative_ones():
    """Gate outcomes must be visible - especially the FAILs."""
    summary = gate_summary()
    assert len(summary) == 4
    decisions = {row["gate"]: row["decision"] for row in summary}
    # These are the project's real, committed outcomes. If a gate is ever
    # re-run to a different result this test should be updated deliberately,
    # not silently - which is the point.
    assert decisions["Gate 1"] == "WEAK_PASS"
    assert decisions["Gate 2"] == "FAIL"
    assert decisions["Gate 4"] == "FAIL"


def test_gate_summary_handles_missing_artifacts_without_inventing_a_decision(tmp_path):
    for row in gate_summary(root=tmp_path):
        assert row["decision"] == "not run"
        assert row["status"] == "missing"


# ---- claims / limitations panel -------------------------------------------

def test_claims_panel_is_parsed_from_the_ledger_and_flags_negatives():
    claims = load_claims()
    assert len(claims) > 20
    negative = [c for c in claims if c.is_negative]
    assert negative, "the ledger contains FAIL/WEAK_PASS claims that must be flagged"
    ids = {c.claim_id for c in claims}
    assert any(cid.startswith("C-21") for cid in ids)  # Gate 2 FAIL claim present


def test_limitations_are_loaded_from_pending():
    limitations = load_limitations()
    assert len(limitations) > 5
    assert any("Gate 1" in item for item in limitations)


# ---- matched-bit guard -----------------------------------------------------

def test_unmatched_budgets_produce_a_warning():
    warning = matched_bit_warning({"a": 4.0, "b": 8.0})
    assert warning is not None
    assert "UNMATCHED BUDGETS" in warning
    assert "NOT a like-for-like" in warning


def test_matched_budgets_produce_no_warning():
    assert matched_bit_warning({"a": 4.00, "b": 4.05}) is None


def test_single_system_needs_no_matching():
    assert matched_bit_warning({"only": 4.0}) is None


# ---- provenance ------------------------------------------------------------

def test_provenance_reports_unknown_rather_than_inventing_fields(tmp_path):
    path = tmp_path / "gate1_study" / "GATE1_REPORT.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"decision": "PASS"}))
    prov = build_provenance(load_artifact("gate1", root=tmp_path), root=tmp_path)
    assert prov.model == "unknown"
    assert prov.config_hash == "unknown"
    assert prov.seed is None
    assert prov.artifact_path.endswith("GATE1_REPORT.json")
