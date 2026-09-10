"""Generate the predeclared development duration configs; no cloud calls."""
from pathlib import Path
import yaml
from rz1t.train import validate_config

ROOT = Path(__file__).resolve().parents[1]

def main():
    for arm, slug in [('U6', 'u6'), ('R6x2', 'r6x2'), ('U12', 'u12')]:
        cfg = yaml.safe_load((ROOT/f'configs/dev/owt-{slug}-1e16.yaml').read_text())
        cfg.update(seed=3, arm=arm, pair_id='duration-v1-s3', run_id=f'duration-v1-s3-{arm}')
        cfg['training'] = dict(steps=15513, batch_size=64, eval_every=1000,
                               checkpoint_every=1000, diagnostic_probe_batches=8,
                               diagnostic_probe_seed=1729)
        validate_config(cfg)
        out = ROOT/f'configs/explore/duration-{slug}.yaml'
        out.write_text(yaml.safe_dump(cfg, sort_keys=False))

if __name__ == '__main__':
    main()
