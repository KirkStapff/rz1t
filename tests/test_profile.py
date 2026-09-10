from rz1t.profile import _is_oom, run_profile


def test_tiny_profile_runs_both_arms(tmp_path):
    model = dict(vocab=16, sequence=8, n_embed=8, aft_heads=2, aft_ksize=2, linear_fan_in=4)
    record = run_profile(out=tmp_path, batches=[1], warmup=1, timed=2, seed=0, model=model)
    assert record["arms"]["U12"]["stopped_reason"] is None
    assert record["arms"]["R6x2"]["batches"]["1"]["status"] == "ok"
    assert record["arms"]["U12"]["parameter_counts"]["body"] > record["arms"]["R6x2"]["parameter_counts"]["body"]
    assert (tmp_path / "profile.json").is_file()


def test_single_arm_merges_existing_profile(tmp_path):
    model = dict(vocab=16, sequence=8, n_embed=8, aft_heads=2, aft_ksize=2, linear_fan_in=4)
    run_profile(out=tmp_path, batches=[1], warmup=1, timed=1, seed=0, model=model, arms=["U12"])
    record = run_profile(out=tmp_path, batches=[1], warmup=1, timed=1, seed=0, model=model, arms=["R6x2"])
    assert "U12" in record["arms"] and "R6x2" in record["arms"]


def test_oom_detector():
    assert _is_oom(RuntimeError("RESOURCE_EXHAUSTED: Out of memory"))
    assert not _is_oom(ValueError("nonfinite training loss"))
