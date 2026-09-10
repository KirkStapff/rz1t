import copy
from pathlib import Path

import pytest
import yaml

from rz1t.claim_a import ARMS, FIRST, RECOVERY, REMAINING, RUNS, SEEDS, analyze_claim_a, expected_run_id, write_manifest
from rz1t.train import validate_config, _seeds, canonical_hash

ROOT = Path(__file__).resolve().parents[1]


def test_claim_a_configs_use_confirmation_split_and_new_seeds():
    configs = []
    for seed in SEEDS:
        for slug in ("u6", "r6x2", "u12"):
            cfg = yaml.safe_load((ROOT / f"configs/claim-a/claim-a-v1-{slug}-s{seed}.yaml").read_text())
            model, work, schedule, budget = validate_config(cfg)
            assert cfg["lane"] == "claim-a"
            assert cfg["data"]["evaluation_split"] == "confirmation"
            assert schedule["steps"] == 15513
            assert work["tokens_per_step"] * schedule["steps"] == 254164992
            assert model.injection == "none"
            configs.append(cfg)
    assert len({c["run_id"] for c in configs}) == 18
    assert set(SEEDS) == {7, 8, 9, 10, 11, 12}
    for seed in SEEDS:
        seeds = [_seeds(yaml.safe_load((ROOT / f"configs/claim-a/claim-a-v1-{slug}-s{seed}.yaml").read_text()))
                 for slug in ("u6", "r6x2", "u12")]
        assert seeds[0] == seeds[1] == seeds[2]


def test_claim_a_manifest_does_not_authorize_h1(tmp_path):
    manifest = write_manifest(tmp_path / "manifest.json")
    assert manifest["confirmation_authorized"] is False
    assert manifest["h1_authorized"] is False
    assert manifest["evaluation_split"] == "confirmation"
    assert len(manifest["runs"]) == 18
    for run in manifest["runs"]:
        cfg = yaml.safe_load((ROOT / run["config_path"]).read_text())
        assert canonical_hash(cfg) == run["config_sha256"]
        assert cfg["run_id"] == run["run_id"]


def _records(gap_u12=0.012, gap_u6=-0.021):
    records = []
    for seed in SEEDS:
        base = 5.9 + 0.001 * seed
        values = {"U12": base, "R6x2": base + gap_u12, "U6": base - gap_u6}
        for arm in ARMS:
            records.append({
                "run_id": expected_run_id(seed, arm), "arm": arm,
                "pair_id": f"claim-a-v1-s{seed}", "status": "completed",
                "checkpoint_selection": "final_budget", "nll": values[arm],
                "evaluation_split": "confirmation",
            })
    return records


def test_claim_a_intervals_and_incomplete_study(tmp_path):
    manifest = write_manifest(tmp_path / "manifest.json")
    report = analyze_claim_a(_records(), manifest)
    assert report["complete"]
    assert report["n_complete_triples"] == 6
    assert report["evidence_status"] == "claim_a_confirmation_holdout_not_h1"
    assert report["contrasts"]["r_minus_u12"]["mean"] == pytest.approx(0.012)
    missing = analyze_claim_a(_records()[1:], manifest)
    assert missing["complete"] is False
    failed = _records()
    failed[0]["status"] = "model_failure"
    failed[0]["nll"] = None
    assert analyze_claim_a(failed, manifest)["complete"] is False
    with pytest.raises(ValueError, match="duplicate"):
        analyze_claim_a(_records() + [_records()[0]], manifest)
    changed = copy.deepcopy(_records())
    changed[0]["checkpoint_selection"] = "best"
    with pytest.raises(ValueError, match="final-checkpoint"):
        analyze_claim_a(changed, manifest)
    bad = copy.deepcopy(manifest)
    bad["h1_authorized"] = True
    with pytest.raises(ValueError, match="H1"):
        analyze_claim_a(_records(), bad)


def test_launcher_matrix_is_eighteen_unique_runs():
    assert len(RUNS) == 18
    assert len(FIRST) == len(REMAINING) == 9
    assert len({(arm, seed) for arm, seed in RUNS}) == 18
    assert [seed for arm, seed in FIRST] == [7, 7, 7, 8, 8, 8, 9, 9, 9]
    assert [seed for arm, seed in REMAINING] == [10, 10, 10, 11, 11, 11, 12, 12, 12]
    assert {arm for arm, seed in RUNS} == {"U6", "R6x2", "U12"}
    assert RECOVERY == [("U12", 12)]
