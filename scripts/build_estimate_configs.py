"""Write frozen estimate-v1 calibration configs. Do not edit generated YAMLs by hand."""
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "configs/calibration"
ARMS = {
    "U6": {"m": 6, "k": 1, "slug": "u6"},
    "R6x2": {"m": 6, "k": 2, "slug": "r6x2"},
    "U12": {"m": 12, "k": 1, "slug": "u12"},
}


def config(arm, seed):
    spec = ARMS[arm]
    return {
        "lane": "calibration",
        "seed": seed,
        "arm": arm,
        "pair_id": f"estimate-v1-s{seed}",
        "run_id": f"estimate-v1-s{seed}-{arm}",
        "model": {
            "vocab": 50257, "sequence": 256, "n_embed": 384, "aft_heads": 8,
            "aft_ksize": 4, "linear_fan_in": 4, "p": 0, "m": spec["m"],
            "k": spec["k"], "q": 0, "injection": "none",
        },
        "data": {
            "kind": "prepared",
            "manifest": "data/owt-subset-256/manifest.json",
            "evaluation_split": "calibration",
        },
        "training": {
            "steps": 15513, "batch_size": 64, "eval_every": 1000,
            "checkpoint_every": 1000, "diagnostic_probe_batches": 8,
            "diagnostic_probe_seed": 1729,
        },
        "optimizer": {
            "lr": 0.0003, "warmup": 100, "weight_decay": 0.1, "b1": 0.9,
            "b2": 0.95, "grad_clip": 1.0, "final_lr_ratio": 0.1,
        },
        "precision": "float32",
        "accounting_version": "semantic-proxy-v1",
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for seed in (4, 5, 6):
        for arm, spec in ARMS.items():
            path = OUT / f"estimate-v1-{spec['slug']}-s{seed}.yaml"
            path.write_text(yaml.safe_dump(config(arm, seed), sort_keys=False))


if __name__ == "__main__":
    main()
