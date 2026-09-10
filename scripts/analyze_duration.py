"""Regenerate duration-v1 curves and summary from verified metrics bundles."""
import json
import math
from pathlib import Path
import tarfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rz1t.checkpoint import file_hash

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT/'artifacts/duration'
OUT = ROOT/'reference/duration-review'

def main():
    OUT.mkdir(exist_ok=True)
    summary = {'status':'one_seed_development_not_confirmation', 'arms':{}}
    fig, axes = plt.subplots(1, 3, figsize=(16,4))
    for arm in ('U6','R6x2','U12'):
        archive = WORK/f'{arm}-s3-metrics.tgz'
        expected = json.loads(Path(str(archive)+'.sha256.json').read_text())
        assert file_hash(archive) == expected['sha256'] and archive.stat().st_size == expected['bytes']
        with tarfile.open(archive) as t:
            def read(name):
                return json.load(t.extractfile(f'{arm}-s3/{name}'))
            final, metrics = read('final.json'), read('metrics.json')
        rows = [r for r in metrics if 'diagnostics' in r]
        assert len(rows) == 16 and rows[-1]['step'] == 15513
        summary['arms'][arm] = {'final': final, 'evaluations':rows}
        x = [r['tokens']/1e6 for r in rows]
        dev = [r['diagnostics']['development']['nll'] for r in rows]
        train = [r['diagnostics']['train_probe']['nll'] for r in rows]
        axes[0].plot(x, dev, label=arm)
        axes[1].plot(x, [d-t for d,t in zip(dev,train)], label=arm)
    reference = summary['arms']['R6x2']['evaluations']
    for arm in ('U6','U12'):
        rows = summary['arms'][arm]['evaluations']
        axes[2].plot([r['tokens']/1e6 for r in rows], [r['eval_nll']-q['eval_nll'] for r,q in zip(rows,reference)], label=f'{arm} minus R6x2')
    for ax, title in zip(axes, ['Development NLL', 'Development − fixed train probe', 'Paired trajectory differences']):
        ax.set(title=title, xlabel='Training target presentations (millions)', ylabel='Nats/token')
        ax.grid(alpha=.3); ax.legend()
    fig.suptitle('Duration-v1: one development seed, matched tokens, longer cosine schedule')
    fig.tight_layout(); fig.savefig(OUT/'curves.png', dpi=160); plt.close(fig)
    f = {a:summary['arms'][a]['final']['nll'] for a in summary['arms']}
    summary['contrasts'] = {'r_minus_u12_nats':f['R6x2']-f['U12'],
        'r_vs_u12_ppl_increase_pct':100*math.expm1(f['R6x2']-f['U12']),
        'r_vs_u6_ppl_reduction_pct':100*(-math.expm1(f['R6x2']-f['U6'])),
        'fraction_u6_u12_nll_gain_recovered':(f['U6']-f['R6x2'])/(f['U6']-f['U12'])}
    (OUT/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    for a, v in summary['arms'].items():
        print(a, 'final',v['final']['nll'],'train probe',v['final']['diagnostics']['train_probe']['nll'], 'training sec',v['final']['training_seconds'])
        print('steps dev train:',[(r['step'],round(r['eval_nll'],6),round(r['diagnostics']['train_probe']['nll'],6)) for r in v['evaluations']])
        print('position bins',v['final']['diagnostics']['development']['position_bins'])
    print(summary['contrasts'])

if __name__ == '__main__':
    main()
