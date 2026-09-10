import copy
from pathlib import Path

import pytest
import yaml

from rz1t.estimate import ARMS, REMAINING, RUNS, SEEDS, analyze_estimate, expected_run_id, write_manifest
from rz1t.train import validate_config, _seeds, canonical_hash

ROOT = Path(__file__).resolve().parents[1]


def test_estimate_configs_match_duration_horizon_and_use_calibration():
    configs = []
    for seed in SEEDS:
        for slug in ("u6", "r6x2", "u12"):
            cfg = yaml.safe_load((ROOT / f"configs/calibration/estimate-v1-{slug}-s{seed}.yaml").read_text())
            model, work, schedule, budget = validate_config(cfg)
            assert cfg["lane"] == "calibration"
            assert cfg["data"]["evaluation_split"] == "calibration"
            assert schedule["steps"] == 15513
            assert work["tokens_per_step"] * schedule["steps"] == 254164992
            assert model.injection == "none"
            configs.append(cfg)
    assert len({c["run_id"] for c in configs}) == 9
    for seed in SEEDS:
        seeds = [_seeds(yaml.safe_load((ROOT / f"configs/calibration/estimate-v1-{slug}-s{seed}.yaml").read_text()))
                 for slug in ("u6", "r6x2", "u12")]
        assert seeds[0] == seeds[1] == seeds[2]


def test_manifest_hashes_match_on_disk_configs(tmp_path):
    manifest = write_manifest(tmp_path / "manifest.json")
    assert manifest["confirmation_authorized"] is False
    assert len(manifest["runs"]) == 9
    for run in manifest["runs"]:
        cfg = yaml.safe_load((ROOT / run["config_path"]).read_text())
        assert canonical_hash(cfg) == run["config_sha256"]
        assert cfg["run_id"] == run["run_id"]


def _records(gap_u12=0.007, gap_u6=-0.025):
    records = []
    for seed in SEEDS:
        base = 5.7 + 0.001 * seed
        values = {"U12": base, "R6x2": base + gap_u12, "U6": base - gap_u6}
        for arm in ARMS:
            records.append({
                "run_id": expected_run_id(seed, arm), "arm": arm,
                "pair_id": f"estimate-v1-s{seed}", "status": "completed",
                "checkpoint_selection": "final_budget", "nll": values[arm],
                "evaluation_split": "calibration",
            })
    return records


def test_estimate_intervals_and_incomplete_study(tmp_path):
    manifest = write_manifest(tmp_path / "manifest.json")
    report = analyze_estimate(_records(), manifest)
    assert report["complete"]
    assert report["n_complete_triples"] == 3
    assert report["evidence_status"] == "independent_estimation_not_confirmation"
    assert report["contrasts"]["r_minus_u12"]["mean"] == pytest.approx(0.007)
    missing = analyze_estimate(_records()[1:], manifest)
    assert missing["complete"] is False
    assert missing["missing_run_ids"]
    failed = _records()
    failed[0]["status"] = "model_failure"
    failed[0]["nll"] = None
    report = analyze_estimate(failed, manifest)
    assert report["complete"] is False
    with pytest.raises(ValueError, match="duplicate"):
        analyze_estimate(_records() + [_records()[0]], manifest)
    changed = copy.deepcopy(_records())
    changed[0]["checkpoint_selection"] = "best"
    with pytest.raises(ValueError, match="final-checkpoint"):
        analyze_estimate(changed, manifest)


def test_launcher_matrix_is_nine_unique_runs():
    assert len(RUNS) == 9
    assert len({(arm, seed) for arm, seed in RUNS}) == 9
    assert [seed for arm, seed in RUNS] == [4, 4, 4, 5, 5, 5, 6, 6, 6]
    assert REMAINING == [('U6', 5), ('R6x2', 5), ('U12', 5), ('U6', 6), ('R6x2', 6), ('U12', 6)]
