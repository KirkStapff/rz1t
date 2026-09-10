import copy
import json

import pytest
from scipy.stats import t

from rz1t.analyze import analyze_records, paired_interval, main
from rz1t.make_sweep import generate_sweep


def study(gap=0.0):
    manifest = generate_sweep(pairs=4)
    records = [dict(run_id=run['run_id'], config_sha256=run['config_sha256'],
                    status='completed', checkpoint_selection='final_budget',
                    nll=3.0+(gap if run['arm'] == 'R6x2' else 0.0)) for run in manifest['runs']]
    return records, manifest


def test_known_student_interval():
    ci = paired_interval([-0.001, 0.0, 0.001])
    assert ci['mean'] == 0
    assert ci['sd'] == .001
    assert ci['confidence'] == pytest.approx(1-.05/3)
    assert ci['upper'] == pytest.approx(t.ppf(1-.05/6, 2)*.001/(3**.5))
    assert ci['lower'] == -ci['upper']
    with pytest.raises(ValueError):
        paired_interval([1])
    with pytest.raises(ValueError):
        paired_interval([float('nan'), 0])


@pytest.mark.parametrize('gap,decision', [(0., 'supported'), (.02, 'contradicted'), (.01, 'inconclusive')])
def test_complete_classifications(gap, decision):
    records, manifest = study(gap)
    # Floating subtraction of 3.01 - 3 is slightly below .01: set exact test
    # boundary by using zero baseline, permitted finite nonnegative NLL.
    if gap == .01:
        for row, run in zip(records, manifest['runs']):
            row['nll'] = gap if run['arm'] == 'R6x2' else 0
    report = analyze_records(records, manifest)
    assert report['decision'] == decision
    assert report['complete']
    assert report['evidence_status'] == 'exploratory_not_preregistered'


def test_missing_failures_do_not_disappear():
    records, manifest = study()
    report = analyze_records(records[1:], manifest)
    assert report['decision'] == 'inconclusive'
    assert report['missing_run_ids'] == [records[0]['run_id']]
    records[0]['status'] = 'model_failure'
    report = analyze_records(records, manifest)
    assert report['decision'] == 'inconclusive'
    assert report['failures'][0]['status'] == 'model_failure'


def test_rejects_selection_identity_and_incomplete_design():
    records, manifest = study()
    with pytest.raises(ValueError, match='duplicate'):
        analyze_records(records+[records[0]], manifest)
    changed = copy.deepcopy(records)
    changed[0]['checkpoint_selection'] = 'best'
    with pytest.raises(ValueError, match='final_budget'):
        analyze_records(changed, manifest)
    changed[0]['checkpoint_selection'] = 'final_budget'
    changed[0]['config_sha256'] = 'wrong'
    with pytest.raises(ValueError, match='identity'):
        analyze_records(changed, manifest)
    with pytest.raises(ValueError, match='margin'):
        analyze_records(records, manifest, delta=.1)
    manifest['runs'].pop()
    with pytest.raises(ValueError, match='controls'):
        analyze_records(records, manifest)


def test_metrics_to_figures_cli(tmp_path):
    records, manifest = study()
    (tmp_path/'metrics.json').write_text(json.dumps(records))
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    main(['--metrics', str(tmp_path/'metrics.json'), '--manifest', str(tmp_path/'manifest.json'),
          '--out', str(tmp_path/'figures')])
    assert (tmp_path/'figures'/'primary_gap.png').stat().st_size > 0
    report = json.loads((tmp_path/'figures'/'analysis.json').read_text())
    assert report['decision'] == 'supported'
