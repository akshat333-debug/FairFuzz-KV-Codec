import json

from fairfuzzkv_codec.core.naming import apply_project_identity, resolve_project_identity
from fairfuzzkv_codec.evaluation.gate4 import Gate4Decision


def test_resolve_identity_covers_every_decision():
    for decision in Gate4Decision:
        identity = resolve_project_identity(decision)
        assert identity.decision == decision
        assert identity.display_name
        assert identity.package_name


def test_project_name_is_a_fixed_constant_across_every_decision():
    """The project NAME is owner-chosen identity, not evidence: it must stay
    FairFuzzKV-Codec whatever Gate 4 decided. (It must also stay consistent
    with the import package `fairfuzzkv_codec`, or the build breaks.)"""
    for decision in Gate4Decision:
        identity = resolve_project_identity(decision)
        assert identity.display_name == "FairFuzzKV-Codec"
        assert identity.package_name == "fairfuzzkv-codec"


def test_claim_framing_still_follows_the_decision():
    """What DOES follow the evidence is the claim wording - a FAIL must be
    reported as negative evidence and must not be framed as a validated win."""
    fail = resolve_project_identity(Gate4Decision.FAIL).claim_framing
    assert "FAIL" in fail and "negative evidence" in fail
    assert "not fabricated" in fail

    passed = resolve_project_identity(Gate4Decision.PASS).claim_framing
    assert "validated" in passed
    assert passed != fail  # the framings genuinely differ per decision


def test_apply_project_identity_keeps_name_and_writes_identity_file(tmp_path):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\nname = "fairfuzzkv-codec"\nversion = "0.1.0"\n', encoding="utf-8")

    identity = resolve_project_identity(Gate4Decision.FAIL)
    changed = apply_project_identity(identity, tmp_path)

    # name already correct -> no rewrite; identity record is still written.
    assert changed["pyproject.toml"] is False
    assert changed["PROJECT_IDENTITY.json"] is True
    assert 'name = "fairfuzzkv-codec"' in pyproject.read_text(encoding="utf-8")
    assert 'version = "0.1.0"' in pyproject.read_text(encoding="utf-8")  # rest of file untouched

    recorded = json.loads((tmp_path / "PROJECT_IDENTITY.json").read_text(encoding="utf-8"))
    assert recorded["gate4_decision"] == "FAIL"
    assert recorded["package_name"] == "fairfuzzkv-codec"


def test_package_name_matches_import_package_so_build_cannot_break(tmp_path):
    """Regression guard for a real bug: a distribution name that doesn't match
    the `src/fairfuzzkv_codec/` import package made uv_build look for
    `src/fragkv_codec/` and fail the whole build (tests could not even run)."""
    for decision in Gate4Decision:
        pkg = resolve_project_identity(decision).package_name
        assert pkg.replace("-", "_") == "fairfuzzkv_codec"


def test_apply_project_identity_is_idempotent_no_op_on_rerun(tmp_path):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\nname = "fairfuzzkv-codec"\n', encoding="utf-8")
    identity = resolve_project_identity(Gate4Decision.PASS)

    first = apply_project_identity(identity, tmp_path)
    second = apply_project_identity(identity, tmp_path)

    assert first["pyproject.toml"] is False  # already named correctly - no change
    assert second["pyproject.toml"] is False
    assert second["PROJECT_IDENTITY.json"] is False  # unchanged on second call
