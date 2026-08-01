from fairfuzzkv_codec.experiment_tracking import ExperimentRegistry, git_commit


def test_log_and_read_back(tmp_path):
    reg = ExperimentRegistry(tmp_path / "runs.jsonl")
    record = reg.log_run(
        study="gate1", config={"model": "qwen"}, seeds=[42],
        metrics={"effect_size": 0.075}, artifacts=["gate1_study/predictions.jsonl"],
        run_id="gate1_fixed",
    )
    assert record.run_id == "gate1_fixed"
    runs = reg.all_runs()
    assert len(runs) == 1
    assert runs[0].metrics["effect_size"] == 0.075
    assert runs[0].seeds == [42]


def test_registry_is_append_only(tmp_path):
    """A later run must never overwrite an earlier one - otherwise a rerun
    could quietly restate a previous result."""
    reg = ExperimentRegistry(tmp_path / "runs.jsonl")
    reg.log_run(study="gate1", metrics={"effect_size": 0.075}, run_id="r1")
    reg.log_run(study="gate1", metrics={"effect_size": 0.02}, run_id="r2")

    runs = reg.runs_for("gate1")
    assert [r.run_id for r in runs] == ["r1", "r2"]
    assert [r.metrics["effect_size"] for r in runs] == [0.075, 0.02]


def test_runs_are_separated_by_study(tmp_path):
    reg = ExperimentRegistry(tmp_path / "runs.jsonl")
    reg.log_run(study="gate1", run_id="a")
    reg.log_run(study="gate4", run_id="b")
    assert [r.run_id for r in reg.runs_for("gate1")] == ["a"]
    assert [r.run_id for r in reg.runs_for("gate4")] == ["b"]
    assert reg.latest("gate4").run_id == "b"


def test_latest_returns_none_for_unknown_study(tmp_path):
    reg = ExperimentRegistry(tmp_path / "runs.jsonl")
    assert reg.latest("never_run") is None
    assert reg.all_runs() == []  # missing file is empty, not an error


def test_metric_history_skips_runs_missing_the_metric(tmp_path):
    """A run without the metric must be omitted, never reported as 0.0."""
    reg = ExperimentRegistry(tmp_path / "runs.jsonl")
    reg.log_run(study="s", metrics={"acc": 0.5}, run_id="r1")
    reg.log_run(study="s", metrics={}, run_id="r2")
    reg.log_run(study="s", metrics={"acc": 0.7}, run_id="r3")
    assert reg.metric_history("s", "acc") == [("r1", 0.5), ("r3", 0.7)]


def test_every_run_records_a_git_commit(tmp_path):
    reg = ExperimentRegistry(tmp_path / "runs.jsonl")
    record = reg.log_run(study="s", run_id="r")
    assert record.git_commit  # either a sha (possibly -dirty) or "unavailable"


def test_git_commit_never_fabricates():
    commit = git_commit()
    assert commit == "unavailable" or len(commit.split("-")[0]) >= 6
