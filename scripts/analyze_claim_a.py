"""Regenerate claim-a-v1 summary and figure from verified metrics bundles."""
import json
from pathlib import Path
import tarfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rz1t.checkpoint import file_hash
from rz1t.claim_a import ARMS, SEEDS, analyze_claim_a, write_manifest

ROOT = Path(__file__).resolve().parents[1]
WORKS = (ROOT / "artifacts/claim-a",)
OUT = ROOT / "reference/claim-a-review"


def load_final(arm, seed):
    name = f"{arm}-s{seed}-metrics.tgz"
    archive = next((work / name for work in WORKS if (work / name).exists()), None)
    if archive is None:
        raise FileNotFoundError(name)
    expected = json.loads(Path(str(archive) + ".sha256.json").read_text())
    assert file_hash(archive) == expected["sha256"] and archive.stat().st_size == expected["bytes"]
    with tarfile.open(archive) as t:
        final = json.load(t.extractfile(f"{arm}-s{seed}/final.json"))
    return final


def main():
    OUT.mkdir(exist_ok=True)
    manifest = write_manifest(OUT / "manifest.json")
    records = []
    finals = {}
    for seed in SEEDS:
        for arm in ARMS:
            final = load_final(arm, seed)
            records.append(final)
            finals[(arm, seed)] = final
    report = analyze_claim_a(records, manifest)
    (OUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    xs = {"U6": 102589, "R6x2": 102589, "U12": 204409}
    offsets = {"U6": -1200, "R6x2": 1200, "U12": 0}
    colors = {"U6": "#4c78a8", "R6x2": "#f58518", "U12": "#54a24b"}
    for arm in ARMS:
        nlls = [finals[(arm, seed)]["nll"] for seed in SEEDS]
        ax.scatter([xs[arm] + offsets[arm]] * len(nlls), nlls, color=colors[arm], label=arm, zorder=3)
        ax.hlines(sum(nlls) / len(nlls), xs[arm] + offsets[arm] - 800, xs[arm] + offsets[arm] + 800,
                  color=colors[arm])
    ax.set(xlabel="Unique trainable body parameters", ylabel="Confirmation-split NLL (nats/token)",
           title="Claim-A v1: unused confirmation holdout, matched tokens, n=6")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "quality-storage.png", dpi=160)
    plt.close(fig)
    print(json.dumps({k: report[k] for k in ("complete", "means", "contrasts", "n_complete_triples")}, indent=2))


if __name__ == "__main__":
    main()
