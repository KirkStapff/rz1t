"""Generate paper Figures 1–5 from frozen C8 / duration artifacts.

Does not train. Regenerates from claim-a-review/summary.json (already produced
from checksum-verified metrics), claim-a metrics bundles, and duration-launch
metrics bundles.
"""
from __future__ import annotations

import json
import tarfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from rz1t.checkpoint import file_hash

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reference" / "paper-figures"
SUMMARY = ROOT / "reference" / "claim-a-review" / "summary.json"
DURATION = ROOT / "artifacts" / "duration"
CLAIM_A_DIRS = (ROOT / "artifacts" / "claim-a",)
EVAL_STEPS = (1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000,
              11000, 12000, 13000, 14000, 15000, 15513)

BODY = {"U6": 102_589, "R6x2": 102_589, "U12": 204_409}
LABEL = {"U6": "U6", "R6x2": r"R6$\times$2", "U12": "U12"}
COLOR = {"U6": "#1f4e79", "R6x2": "#c45c26", "U12": "#2e7d4f"}
MARKER = {"U6": "o", "R6x2": "s", "U12": "D"}

# Distinct unique-block colors (12 unique blocks for U12).
BLOCK = [
    "#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2", "#eeca3b",
    "#b279a2", "#ff9da6", "#9d755d", "#bab0ac", "#2e7d4f", "#1f4e79",
]


def _style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _save(fig, stem: str):
    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / f"{stem}.png"
    pdf = OUT / f"{stem}.pdf"
    fig.savefig(png)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def _hex_mix(color: str, other: str = "#ffffff", t: float = 0.22) -> str:
    def rgb(s):
        s = s.lstrip("#")
        return tuple(int(s[i:i + 2], 16) for i in (0, 2, 4))

    a, b = rgb(color), rgb(other)
    m = tuple(round(a[i] * (1 - t) + b[i] * t) for i in range(3))
    return f"#{m[0]:02x}{m[1]:02x}{m[2]:02x}"


def figure1_architecture():
    """Two-column methods diagram: unique body storage vs applied depth."""
    fig, ax = plt.subplots(figsize=(8.2, 3.65))
    ax.set_xlim(0.0, 15.35)
    ax.set_ylim(0.0, 4.62)
    ax.axis("off")

    w, h, gap = 0.38, 0.46, 0.055
    y_u6, y_r6, y_u12 = 3.32, 2.00, 0.68
    x_store, x_apply = 1.58, 7.55
    x_arm, x_unique_n, x_applied_n = 0.10, 6.95, 13.55

    def span(n):
        return n * w + max(n - 1, 0) * gap

    def draw_tiles(x, y, n, *, start=0):
        for i in range(n):
            xi = x + i * (w + gap)
            ax.add_patch(Rectangle(
                (xi, y), w, h,
                facecolor=_hex_mix(BLOCK[start + i], "#ffffff", 0.10),
                edgecolor="#2f2f2f", linewidth=0.35, zorder=3,
            ))
            ax.text(
                xi + w / 2, y + h / 2, str(start + i + 1),
                ha="center", va="center", fontsize=6.2,
                color="white", fontweight="bold", zorder=4,
            )

    def header(x, text, ha="center"):
        ax.text(x, 4.22, text, ha=ha, va="bottom", fontsize=8.4, color="#333")

    def brace(x, y_lo, y_hi, label, color, *, inward=0.08, text_dx=-0.20):
        ax.plot([x, x], [y_lo, y_hi], color=color, lw=0.85, zorder=1,
                solid_capstyle="butt", clip_on=False)
        ax.plot([x, x + inward], [y_lo, y_lo], color=color, lw=0.85, zorder=1,
                solid_capstyle="butt", clip_on=False)
        ax.plot([x, x + inward], [y_hi, y_hi], color=color, lw=0.85, zorder=1,
                solid_capstyle="butt", clip_on=False)
        ax.text(x + text_dx, (y_lo + y_hi) / 2, label, ha="center", va="center",
                fontsize=7, color=color, rotation=90, zorder=2)

    header(x_store + span(12) / 2, "Unique body")
    header(x_apply + span(12) / 2, "Forward pass")
    header(x_unique_n, "unique", ha="center")
    header(x_applied_n, "applied", ha="center")
    ax.plot([x_store, x_store + span(12)], [4.14, 4.14], color="#c8c8c8", lw=0.5)
    ax.plot([x_apply, x_apply + span(12)], [4.14, 4.14], color="#c8c8c8", lw=0.5)

    rows = (
        (y_u6, "U6", COLOR["U6"], 6, 6),
        (y_r6, r"R6$\times$2", COLOR["R6x2"], 6, 12),
        (y_u12, "U12", COLOR["U12"], 12, 12),
    )
    for y, name, color, n_unique, n_applied in rows:
        ax.text(x_arm, y + h / 2, name, ha="left", va="center",
                fontsize=9.5, fontweight="bold", color=color)
        draw_tiles(x_store, y, n_unique)
        if n_unique == 6 and n_applied == 12:
            draw_tiles(x_apply, y, 6, start=0)
            draw_tiles(x_apply + span(6) + gap, y, 6, start=0)
        else:
            draw_tiles(x_apply, y, n_applied)
        ax.text(x_unique_n, y + h / 2, str(n_unique), ha="center", va="center",
                fontsize=8, color="#444")
        ax.text(x_applied_n, y + h / 2, str(n_applied), ha="center", va="center",
                fontsize=8, color="#444")

    brace(
        x_store + span(6) + 0.18, y_r6, y_u6 + h,
        "storage-matched", COLOR["R6x2"], inward=-0.08, text_dx=0.22,
    )
    brace(
        x_apply + span(12) + 0.18, y_u12, y_r6 + h,
        "applied-depth-matched", COLOR["U12"], inward=-0.08, text_dx=0.22,
    )
    return _save(fig, "fig1-architecture")


def _claim_a_metric_archives():
    archives = []
    for directory in CLAIM_A_DIRS:
        archives.extend(sorted(directory.glob("*-metrics.tgz")))
    if len(archives) != 18:
        raise SystemExit(f"expected 18 claim-A metrics archives, found {len(archives)}")
    return archives


def load_claim_a_holdout_curves():
    """Confirmation-split NLL every 1,000 steps plus final (16 points × 18 runs)."""
    series = {arm: {} for arm in ("U6", "R6x2", "U12")}
    for archive in _claim_a_metric_archives():
        expected = json.loads(Path(str(archive) + ".sha256.json").read_text())
        if file_hash(archive) != expected["sha256"]:
            raise SystemExit(f"checksum mismatch: {archive}")
        if archive.stat().st_size != expected["bytes"]:
            raise SystemExit(f"byte-count mismatch: {archive}")
        stem = archive.name.replace("-metrics.tgz", "")
        arm, seed_s = stem.split("-s")
        seed = int(seed_s)
        with tarfile.open(archive) as tar:
            member = next(n for n in tar.getnames() if n.endswith("metrics.json"))
            rows = json.load(tar.extractfile(member))
        points = []
        for row in rows:
            if "eval_nll" not in row:
                continue
            diagnostics = row.get("diagnostics") or {}
            holdout = diagnostics.get("holdout") or {}
            nll = holdout.get("nll", row["eval_nll"])
            if diagnostics.get("evaluation_split") not in (None, "confirmation"):
                raise SystemExit(f"{stem}: unexpected evaluation_split {diagnostics.get('evaluation_split')}")
            points.append({
                "step": int(row["step"]),
                "tokens": int(row["tokens"]),
                "nll": float(nll),
            })
        steps = tuple(p["step"] for p in points)
        if steps != EVAL_STEPS:
            raise SystemExit(f"{stem}: unexpected eval steps {steps}")
        series[arm][seed] = points
    for arm in series:
        if sorted(series[arm]) != [7, 8, 9, 10, 11, 12]:
            raise SystemExit(f"{arm}: missing seeds {sorted(series[arm])}")
    return series


def figure5_holdout_curves(series: dict, *, log_x: bool = False):
    """Mean confirmation-split NLL versus training tokens; faint per-seed traces."""
    import csv

    fig, axes = plt.subplots(2, 1, figsize=(7.4, 6.2), sharex=True,
                             gridspec_kw={"height_ratios": [1.55, 1]})
    tokens_m = None
    means = {}
    for arm in ("U6", "R6x2", "U12"):
        seed_ys = []
        for seed in range(7, 13):
            points = series[arm][seed]
            x = [p["tokens"] / 1e6 for p in points]
            y = [p["nll"] for p in points]
            if tokens_m is None:
                tokens_m = x
            axes[0].plot(x, y, color=COLOR[arm], lw=0.8, alpha=0.28, zorder=2)
            seed_ys.append(y)
        mean_y = [sum(col) / 6 for col in zip(*seed_ys)]
        means[arm] = mean_y
        axes[0].plot(tokens_m, mean_y, color=COLOR[arm], lw=2.2, marker=MARKER[arm],
                     ms=4.5, zorder=3, label=LABEL[arm])

    axes[0].set_ylabel("Confirmation Loss")
    axes[0].legend(frameon=False, loc="upper right")
    axes[0].grid(alpha=0.3, which="both")
    scale = "log x" if log_x else "linear x"
    axes[0].set_title(
        f"Confirmation holdout  ·  n=6  ·  {scale}  ·  mean (bold) and seeds (faint)"
    )

    gap_u6 = [a - b for a, b in zip(means["R6x2"], means["U6"])]
    gap_u12 = [a - b for a, b in zip(means["R6x2"], means["U12"])]
    for seed in range(7, 13):
        y_r = [p["nll"] for p in series["R6x2"][seed]]
        y6 = [p["nll"] for p in series["U6"][seed]]
        y12 = [p["nll"] for p in series["U12"][seed]]
        axes[1].plot(tokens_m, [a - b for a, b in zip(y_r, y6)],
                     color=COLOR["U6"], lw=0.7, alpha=0.28)
        axes[1].plot(tokens_m, [a - b for a, b in zip(y_r, y12)],
                     color=COLOR["U12"], lw=0.7, alpha=0.28)
    axes[1].plot(tokens_m, gap_u6, color=COLOR["U6"], lw=2.0, marker=MARKER["U6"],
                 ms=4, label=r"R6$\times$2 − U6")
    axes[1].plot(tokens_m, gap_u12, color=COLOR["U12"], lw=2.0, marker=MARKER["U12"],
                 ms=4, label=r"R6$\times$2 − U12")
    axes[1].axhline(0, color="#222", lw=0.8)
    if log_x:
        for ax in axes:
            ax.set_xscale("log")
        axes[1].set_xlabel("Tokens (millions, log scale)")
        axes[1].text(18, -0.021, "← favors R6×2", fontsize=8, color="#555")
    else:
        axes[1].set_xlabel("Tokens (millions)")
        axes[1].text(8, -0.021, "← favors R6×2", fontsize=8, color="#555")
    axes[1].set_ylabel("Paired Confirmation Loss")
    axes[1].legend(frameon=False, loc="center right")
    axes[1].grid(alpha=0.3, which="both")

    fig.suptitle(
        "Periodic confirmation-split evaluation (every 1,000 steps + final).\n"
        "Primary paper numbers remain the final-checkpoint intervals in Figure 2."
        + ("  y is linear NLL; only x is log." if log_x else ""),
        fontsize=8.5, y=1.02, color="#333",
    )
    fig.tight_layout()

    stem = "fig5-holdout-curves-logx" if log_x else "fig5-holdout-curves"
    if not log_x:
        csv_path = OUT / "fig5-holdout-curves.csv"
        OUT.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["arm", "seed", "step", "tokens", "confirmation_nll"])
            for arm in ("U6", "R6x2", "U12"):
                for seed in range(7, 13):
                    for point in series[arm][seed]:
                        writer.writerow([arm, seed, point["step"], point["tokens"],
                                         f"{point['nll']:.12f}"])
    return _save(fig, stem)


def figure2_main_result(report: dict):
    means = report["means"]
    seeds = report["seeds"]
    c12 = report["contrasts"]["r_minus_u12"]
    c6 = report["contrasts"]["r_minus_u6"]

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.35),
                             gridspec_kw={"width_ratios": [1.15, 1]})

    ax = axes[0]
    for arm in ("U6", "R6x2", "U12"):
        xs = [BODY[arm]] * 6
        ys = [row[arm] for row in seeds]
        ax.scatter(xs, ys, marker=MARKER[arm], s=36, alpha=0.45,
                   color=COLOR[arm], zorder=3, linewidths=0, label="_nolegend_")
        ax.scatter([BODY[arm]], [means[arm]], marker=MARKER[arm], s=90,
                   color=COLOR[arm], edgecolor="white", linewidths=0.8, zorder=4,
                   label=LABEL[arm])
    ax.set_xlabel("Unique trainable body parameters")
    ax.set_ylabel("Confirmation-split NLL (nats/token)")
    ax.set_title("A. Quality vs unique body storage")
    ax.set_xticks([102589, 204409], ["102,589", "204,409"])
    ax.set_xlim(70_000, 240_000)
    ymin = min(row[arm] for row in seeds for arm in ("U6", "R6x2", "U12"))
    ymax = max(row[arm] for row in seeds for arm in ("U6", "R6x2", "U12"))
    pad = 0.004
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False, loc="upper right")
    ax.text(102589, ymax + 0.0012, "same x", ha="center", fontsize=7, color="#666")

    ax = axes[1]
    rows = [
        (r"R6$\times$2 − U6", c6, COLOR["U6"]),
        (r"R6$\times$2 − U12", c12, COLOR["U12"]),
    ]
    for i, (name, contrast, color) in enumerate(rows):
        y = 1 - i
        diffs = contrast["differences"]
        ax.scatter(diffs, [y + 0.12] * len(diffs), s=26, alpha=0.55, color=color, zorder=3)
        ax.plot([contrast["lower"], contrast["upper"]], [y, y], color=color, lw=3.0, zorder=2,
                solid_capstyle="round")
        ax.scatter([contrast["mean"]], [y], s=42, color=color, zorder=4, edgecolor="white", linewidths=0.6)
        ax.text(-0.034, y, name, fontsize=8, color=color, ha="right", va="center")
    ax.axvline(0, color="#222", lw=0.9, zorder=1)
    ax.set_yticks([])
    ax.set_xlabel("Paired NLL difference (nats/token)")
    ax.set_title("B. Seed-level paired gaps, n=6")
    ax.set_xlim(-0.048, 0.022)
    ax.set_ylim(-0.55, 1.55)
    ax.text(-0.027, -0.45, "← favors R6×2", fontsize=7.5, color="#555")
    ax.text(0.020, -0.45, "favors control →", fontsize=7.5, color="#555", ha="right")
    ax.grid(axis="x", alpha=0.3)

    fig.suptitle(
        "OWT 40k-document subset · unused confirmation split · matched 254.16M tokens\n"
        "n=6 training seeds · final checkpoint · body parameters only · 97.5% paired t (Bonferroni, 2 contrasts)",
        fontsize=8.5, y=1.02, color="#333",
    )
    fig.tight_layout()
    return _save(fig, "fig2-main-result")


def figure3_residency():
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.6),
                             gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    ax.set_xlim(0, 10.2)
    ax.set_ylim(0.2, 7.4)
    ax.axis("off")
    ax.set_title("A")

    def bank(ax, x, y, colors):
        for i, c in enumerate(colors):
            ax.add_patch(Rectangle((x + i * 0.42, y), 0.38, 0.85,
                                   facecolor=c, edgecolor="#222", lw=0.6))

    ax.text(0.3, 6.75, r"U6 / R6$\times$2", fontsize=10, fontweight="bold")
    bank(ax, 0.3, 5.55, BLOCK[:6])
    ax.text(2.82, 5.12, "resident", fontsize=8, color="#555", ha="center")

    ax.text(0.3, 3.55, "U12", fontsize=10, fontweight="bold")
    bank(ax, 0.3, 2.35, BLOCK[:6])
    ax.annotate("", xy=(3.55, 1.55), xytext=(2.82, 2.35),
                arrowprops=dict(arrowstyle="->", color="#444", lw=1.2))
    bank(ax, 3.85, 0.75, BLOCK[6:])

    ax = axes[1]
    r = [i * 0.05 for i in range(0, 81)]
    ax.plot(r, [1 + x for x in r], color="#333", lw=2)
    ax.axhline(1.0, color="#888", lw=0.8, ls="--")
    ax.set_xlabel(r"$L/C$")
    ax.set_ylabel(r"$T_{\mathrm{U12}}/T_{\mathrm{R6}\times 2}$")
    ax.set_title("B")
    ax.set_xlim(0, 4)
    ax.set_ylim(0.9, 5.2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return _save(fig, "fig3-residency")


def figure4_duration():
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.6), sharex=True,
                             gridspec_kw={"height_ratios": [1.6, 1]})
    series = {}
    for arm in ("U6", "R6x2", "U12"):
        archive = DURATION / f"{arm}-s3-metrics.tgz"
        expected = json.loads(Path(str(archive) + ".sha256.json").read_text())
        assert file_hash(archive) == expected["sha256"]
        assert archive.stat().st_size == expected["bytes"]
        with tarfile.open(archive) as t:
            metrics = json.load(t.extractfile(f"{arm}-s3/metrics.json"))
        rows = [r for r in metrics if "diagnostics" in r]
        assert len(rows) == 16
        x = [r["tokens"] / 1e6 for r in rows]
        y = [r["diagnostics"]["development"]["nll"] for r in rows]
        series[arm] = (x, y)
        axes[0].plot(x, y, color=COLOR[arm], marker=MARKER[arm], ms=3.5,
                     lw=1.6, label=LABEL[arm])
    xref, yref = series["R6x2"]
    for arm in ("U6", "U12"):
        x, y = series[arm]
        axes[1].plot(x, [a - b for a, b in zip(y, yref)],
                     color=COLOR[arm], marker=MARKER[arm], ms=3.5, lw=1.5,
                     label=f"{LABEL[arm]} − R6×2")
    axes[1].axhline(0, color="#222", lw=0.8)
    axes[0].set_ylabel("Development NLL")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.3)
    axes[0].set_title("DEVELOPMENT  ·  one seed  ·  not the confirmation result")
    axes[1].set_xlabel("Tokens (millions)")
    axes[1].set_ylabel("NLL gap vs R6×2")
    axes[1].legend(frameon=False, loc="upper right")
    axes[1].grid(alpha=0.3)
    fig.suptitle("Duration diagnostic (C6): matched tokens, longer cosine, seed 3", fontsize=10)
    fig.tight_layout()
    return _save(fig, "fig4-duration-development")


def write_captions(paths: dict):
    text = """# Paper figures

Generated by `uv run python scripts/make_paper_figures.py`.
PNG (300 dpi) and PDF (vector) for each figure.

| File | Use |
|---|---|
| `fig1-architecture.png/.pdf` | Figure 1 — unique vs applied depth (diagram) |
| `fig2-main-result.png/.pdf` | Figure 2 — C8 confirmation holdout, n=6 |
| `fig3-residency.png/.pdf` | Figure 3 — hypothetical six-block schedule |
| `fig4-duration-development.png/.pdf` | Optional / appendix — one development seed |
| `fig5-holdout-curves.png/.pdf` | Confirmation-split NLL vs tokens, n=6 (linear x) |
| `fig5-holdout-curves-logx.png/.pdf` | Same curves, log x, linear y |
| `fig5-holdout-curves.csv` | Plot data: 18 runs × 16 evaluations |

## Suggested captions

**Figure 1.** Controlled contrasts. U6 stores and applies six unique residual blocks.
R6×2 stores the same six blocks and applies them twice. U12 stores twelve unique
blocks and applies them once. Embedding and classifier are identical in structure
across arms. Diagram only.

**Figure 2.** Final-checkpoint negative log-likelihood on the unused confirmation
split of an archived 40k-document OpenWebText subset. All arms trained for
254,164,992 token presentations (15,513 steps × batch 64 × sequence 256).
Panel A: unique trainable body parameters (embedding and classifier excluded).
U6 and R6×2 share the x-coordinate. Faint markers are six training seeds; solid
markers are means. Panel B: paired seed differences with Bonferroni 97.5%
Student-t intervals (two contrasts). Negative values favor R6×2. Not a
non-inferiority test and not pooled with calibration or development runs.

**Figure 3.** Conditional whole-bank replacement schedule if a fabric holds six
block programs. R6×2 retains the bank; U12 loads two banks each decode round
under this schedule (not an optimal-cache bound). Panel B plots
speedup = 1 + (exposed U12 reload time)/(common non-load compute) assuming equal
non-load compute. Not a TSU measurement. Observed language-model quality is
unmatched; U6 is also weight-resident and performs less work.

**Figure 4 (appendix).** Development-split NLL versus tokens
for one development seed under a longer cosine schedule. Not independent
replication of Figure 2; the scored corpus is the development split, not the
confirmation holdout.

**Figure 5.** Confirmation-split negative log-likelihood versus tokens
for the six claim-A seeds. Bold lines are means; faint lines are
individual seeds. Lower panel: mean paired gaps versus U6 and U12 (negative
favors R6×2). Evaluations every 1,000 steps plus the final checkpoint
(810,496 scored tokens). Primary numerical claims remain the final-checkpoint
intervals in Figure 2; these trajectories are descriptive and were not used
for checkpoint selection. Linear x is the main display.
`fig5-holdout-curves-logx` is the same data with a logarithmic token axis
and linear NLL (not Extropic-style log-log loss).
"""
    (OUT / "README.md").write_text(text)
    return OUT / "README.md"


def main():
    _style()
    report = json.loads(SUMMARY.read_text())
    if not report.get("complete") or report.get("n_complete_triples") != 6:
        raise SystemExit("claim-a summary is incomplete")
    curves = load_claim_a_holdout_curves()
    paths = {
        "fig1": figure1_architecture(),
        "fig2": figure2_main_result(report),
        "fig3": figure3_residency(),
        "fig4": figure4_duration(),
        "fig5": figure5_holdout_curves(curves),
        "fig5_logx": figure5_holdout_curves(curves, log_x=True),
    }
    captions = write_captions(paths)
    listing = {k: [str(p.relative_to(ROOT)) for p in v] for k, v in paths.items()}
    listing["captions"] = str(captions.relative_to(ROOT))
    print(json.dumps(listing, indent=2))


if __name__ == "__main__":
    main()
