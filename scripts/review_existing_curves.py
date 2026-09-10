"""Offline review of preserved OWT curves; never treats train loss as held-out loss."""
from pathlib import Path
import json
import statistics
import tarfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reference/curve-review'


def read_complete_prefix(text):
    """Recover only complete row objects from a truncated JSON array; label at callsite."""
    try:
        return json.loads(text), False
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        rows, pos = [], text.index('[') + 1
        while pos < len(text):
            while pos < len(text) and text[pos] in ' \n\r\t,':
                pos += 1
            try:
                row, pos = decoder.raw_decode(text, pos)
            except json.JSONDecodeError:
                break
            rows.append(row)
        return rows, True


def load_curves():
    curves = {}
    for seed, folder in [(0, 'owt-dev-launch-3/owt-dev-1e16'),
                         (1, 'owt-c34-launch/owt-c34/U12-s1')]:
        paths = sorted((ROOT / 'results' / folder / 'checkpoints').glob('*/metadata.json'))
        if paths:
            p = paths[-1]
            curves[f'U12-s{seed}'] = (json.loads(p.read_text())['metrics'], str(p.relative_to(ROOT)))
    for name in ['R6x2-s1', 'R2x6']:
        p = ROOT / 'results/owt-c34-launch/owt-c34' / name / 'metrics.json'
        rows, truncated = read_complete_prefix(p.read_text())
        curves[name] = (rows, str(p.relative_to(ROOT)) + (' [TRUNCATED: complete prefix only]' if truncated else ''))
    for seed in range(3):
        p = ROOT / f'results/u6-launch/U6-s{seed}-metrics.tgz'
        with tarfile.open(p) as archive:
            members = [m for m in archive if m.name.endswith('/metrics.json')]
            assert len(members) == 1
            curves[f'U6-s{seed}'] = (json.load(archive.extractfile(members[0])),
                                    f'{p.relative_to(ROOT)}::{members[0].name}')
    return curves


def summarize(rows):
    assert all(r['step'] == i + 1 for i, r in enumerate(rows))
    result = {'steps_available': len(rows), 'tokens_available': rows[-1]['tokens'],
              'intermediate_eval_rows': sum('eval_nll' in r for r in rows)}
    for lo, hi in [(501, 1000), (2001, 2500), (3501, 4000), (4001, 4500), (4501, 5000)]:
        values = [r['train_nll'] for r in rows if lo <= r['step'] <= hi]
        result[f'train_mean_{lo}_{hi}'] = statistics.mean(values) if len(values) == hi-lo+1 else None
    result['last_500_train_mean'] = statistics.mean([r['train_nll'] for r in rows[-500:]])
    # Descriptive slope only, not an extrapolation or independent-sample significance test.
    tail = [r for r in rows if 4001 <= r['step'] <= 5000]
    x = [r['tokens'] / 1e6 for r in tail]
    y = [r['train_nll'] for r in tail]
    result['train_slope_nats_per_million_tokens_steps_4001_5000'] = None
    if len(tail) == 1000:
        xm, ym = statistics.mean(x), statistics.mean(y)
        result['train_slope_nats_per_million_tokens_steps_4001_5000'] = (
            sum((a-xm)*(b-ym) for a,b in zip(x,y)) / sum((a-xm)**2 for a in x))
    return result


def main():
    curves = load_curves()
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {name: {'source': source, **summarize(rows)} for name, (rows, source) in curves.items()}
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    colors = {'U12': 'tab:blue', 'U6': 'tab:orange', 'R6x2': 'tab:green', 'R2x6': 'tab:red'}
    for name, (rows, _) in curves.items():
        chunks = [rows[i:i+250] for i in range(0, len(rows)-249, 250)]
        xs = [statistics.mean(r['tokens'] for r in c)/1e6 for c in chunks]
        ys = [statistics.mean(r['train_nll'] for r in c) for c in chunks]
        for ax in axes:
            ax.plot(xs, ys, label=name, color=colors[name.split('-')[0]],
                    linestyle='--' if name.endswith('s1') else ':' if name.endswith('s2') else '-')
    axes[1].set_xlim(50, 86)
    axes[1].set_ylim(5.95, 6.5)
    for ax in axes:
        ax.set_xlabel('Training-token presentations (millions)')
        ax.set_ylabel('Online minibatch NLL; 250-step block means')
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('Training only: U12 ends at 5000; R6x2-s1 is a truncated prefix; no held-out trajectories')
    fig.savefig(OUT / 'training-curves.png', dpi=180)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
