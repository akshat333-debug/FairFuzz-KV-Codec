import json

from fairfuzzkv_codec.core.naming import apply_project_identity, resolve_project_identity
from fairfuzzkv_codec.evaluation.gate4 import Gate4Decision


def test_resolve_identity_covers_every_decision():
    for decision in Gate4Decision:
        identity = resolve_project_identity(decision)
        assert identity.decision == decision
        assert identity.display_name
        assert identity.package_name


def test_pass_and_weak_pass_keep_fairfuzzkv_name():
    assert resolve_project_identity(Gate4Decision.PASS).display_name == "FairFuzzKV-Codec"
    assert resolve_project_identity(Gate4Decision.WEAK_PASS).display_name == "FairFuzzKV-Codec"


def test_fail_renames_to_fragkv_codec():
    identity = resolve_project_identity(Gate4Decision.FAIL)
    assert identity.display_name == "FragKV-Codec"
    assert identity.package_name == "fragkv-codec"


def test_apply_project_identity_rewrites_pyproject_and_writes_identity_file(tmp_path):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\nname = "fairfuzzkv-codec"\nversion = "0.1.0"\n', encoding="utf-8")

    identity = resolve_project_identity(Gate4Decision.FAIL)
    changed = apply_project_identity(identity, tmp_path)

    assert changed["pyproject.toml"] is True
    assert changed["PROJECT_IDENTITY.json"] is True
    assert 'name = "fragkv-codec"' in pyproject.read_text(encoding="utf-8")
    assert 'version = "0.1.0"' in pyproject.read_text(encoding="utf-8")  # rest of file untouched

    recorded = json.loads((tmp_path / "PROJECT_IDENTITY.json").read_text(encoding="utf-8"))
    assert recorded["gate4_decision"] == "FAIL"
    assert recorded["package_name"] == "fragkv-codec"


def test_apply_project_identity_is_idempotent_no_op_on_rerun(tmp_path):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\nname = "fairfuzzkv-codec"\n', encoding="utf-8")
    identity = resolve_project_identity(Gate4Decision.PASS)

    first = apply_project_identity(identity, tmp_path)
    second = apply_project_identity(identity, tmp_path)

    assert first["pyproject.toml"] is False  # already named correctly - no change
    assert second["pyproject.toml"] is False
    assert second["PROJECT_IDENTITY.json"] is False  # unchanged on second call
