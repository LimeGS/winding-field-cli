"""Blind review cannot hide missing evidence or overweight the larger dataset."""
import importlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from winding_field.util import atomic_json, file_sha


@pytest.fixture
def review(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'tools'))
    assert importlib.util.find_spec('flattening_review'), 'review not implemented'
    return importlib.import_module('flattening_review')


def fixture_run(tmp_path, stage='pilot', missing=False):
    from bench_flattening import seal
    run = tmp_path / 'run'
    run.mkdir()
    atomic_json(run / 'RUN.json', dict(schema='flattening-benchmark-v1', fields=['175a', 'paris4']))
    rows = []
    for field in ('175a', 'paris4'):
        for method in ('lscm', 'arap'):
            relative = f'cells/{field}/{method}'
            root = run / relative / 'ct'
            root.mkdir(parents=True)
            Image.new('RGBA', (8, 8), (90, 90, 90, 255)).save(root / 'plane.png')
            np.savez(root / 'raw.npz', ct=np.full((1, 8, 8), 90), support=np.ones((1, 8, 8), bool))
            regions = [dict(id='roi-00', pitch_um=2.4 if field == 'paris4' else 8.64,
                            offsets_um=[0.], missing=missing and method == 'arap',
                            tiles=[] if missing and method == 'arap' else [
                                dict(raw='raw.npz', planes=['plane.png'], canvas=dict(width=8, height=8,
                                      pitch=2.4 if field == 'paris4' else 8.64, u0=0., v_top=8.))])]
            seal(root, 'CT.json', dict(source=field), dict(regions=regions), ['plane.png', 'raw.npz'])
            rows.append(dict(case_id=field + '-coarse-Wm060', field_id=field, resolution_id='coarse',
                             method=method, scope='sheet', track='A', stage=stage, status='COMPLETE', output=relative))
    atomic_json(run / 'QUEUE.json', dict(rows=rows))
    return run


def test_blind_page_has_no_method_identifiers(review, tmp_path):
    run = fixture_run(tmp_path)
    manifest = review.build_review(run, run / 'review')
    served = '\n'.join(p.read_text() for p in (run / 'review').rglob('*') if p.suffix in ('.html', '.json', '.js'))
    assert 'lscm' not in served.lower() and 'arap' not in served.lower()
    assert len(manifest['panels']) == 4
    assert (run / 'PRIVATE-review-key.json').exists()
    assert not (run / 'review' / 'PRIVATE-review-key.json').exists()


def test_same_roi_and_scale_survive_different_uv_layouts(review, tmp_path):
    run = fixture_run(tmp_path)
    m = review.build_review(run, run / 'review')
    panels = [p for p in m['panels'] if p['field_id'] == '175a']
    assert {r['id'] for p in panels for r in p['regions']} == {'roi-00'}
    assert all(p['regions'][0]['pitch_um'] == 8.64 for p in panels)
    assert all(p['regions'][0]['tiles'][0]['canvas']['pitch'] == 8.64 for p in panels)


def test_missing_roi_is_visible_and_not_replaced(review, tmp_path):
    run = fixture_run(tmp_path, missing=True)
    m = review.build_review(run, run / 'review')
    assert sum(r['missing'] for p in m['panels'] for r in p['regions']) == 2


def test_votes_bind_run_roi_and_artifact_hashes(review, tmp_path):
    run = fixture_run(tmp_path)
    m = review.build_review(run, run / 'review')
    token = m['panels'][0]['token']
    votes = dict(review_sha256='0' * 64, entries=[])
    atomic_json(run / 'votes.json', votes)
    with pytest.raises(ValueError, match='identity|hash'):
        review.summarize(run, run / 'votes.json', run / 'report')
    votes['review_sha256'] = file_sha(run / 'review/manifest.json')
    votes['entries'] = [dict(case_id=m['panels'][0]['case_id'], roi_id='fake', preferred=[token], common_depth_um=0.)]
    atomic_json(run / 'votes.json', votes)
    with pytest.raises(ValueError, match='ROI'):
        review.summarize(run, run / 'votes.json', run / 'report')


def test_holdout_is_not_exposed_before_freeze(review, tmp_path):
    run = fixture_run(tmp_path, stage='holdout')
    with pytest.raises(ValueError, match='freeze|FREEZE'):
        review.build_review(run, run / 'review')
    assert not (run / 'review' / 'index.html').exists()


def test_incomplete_cells_cannot_be_called_winners(review, tmp_path):
    run = fixture_run(tmp_path)
    q = json.loads((run / 'QUEUE.json').read_text())
    q['rows'][0]['status'] = 'FAILED'
    atomic_json(run / 'QUEUE.json', q)
    m = review.build_review(run, run / 'review')
    assert sum(p['status'] == 'FAILED' and not p['regions'] for p in m['panels']) == 1


def test_fields_have_equal_weight_despite_different_case_counts(review):
    rows = [dict(field_id='175a', case_id=str(i), method='a', scope='sheet', score=1.) for i in range(3)]
    rows += [dict(field_id='paris4', case_id='fourth', method='a', scope='sheet', score=0.)]
    result = review.aggregate_preferences(rows)
    assert result['balanced']['a/sheet'] == .5


def test_missing_one_field_blocks_global_conclusion(review):
    result = review.aggregate_preferences([dict(field_id='175a', case_id='x', method='a', scope='sheet', score=1.)])
    assert result['balanced'] == {}
    assert result['missing_fields'] == ['paris4']


def test_pending_holdout_is_hidden_from_pilot_gallery(review, tmp_path):
    run = fixture_run(tmp_path)
    queue = json.loads((run / 'QUEUE.json').read_text())
    pending = dict(queue['rows'][0], stage='holdout', status='PENDING', case_id='withheld')
    queue.update(active_stage='pilot', rows=[*queue['rows'], pending])
    atomic_json(run / 'QUEUE.json', queue)
    m = review.build_review(run, run / 'review')
    assert len(m['panels']) == 4 and all(p['case_id'] != 'withheld' for p in m['panels'])


def test_served_directory_cannot_contain_private_review_key(review, tmp_path):
    run = fixture_run(tmp_path)
    with pytest.raises(ValueError, match='private|separate'):
        review.build_review(run, run)
    assert not (run / 'index.html').exists()
    with pytest.raises(ValueError, match='private|separate'):
        review.build_review(run, tmp_path)
    alias = tmp_path / 'run-alias'
    alias.symlink_to(run, target_is_directory=True)
    queue = json.loads((run / 'QUEUE.json').read_text())
    for row in queue['rows']:
        row['status'] = 'MISSING_INPUT'
    atomic_json(run / 'QUEUE.json', queue)
    with pytest.raises(ValueError, match='private|symlink'):
        review.build_review(alias, run)


def test_missing_roi_does_not_improve_candidate_preference(review, tmp_path):
    run = fixture_run(tmp_path)
    from bench_flattening import read_json, seal
    for cell in (run / 'cells').glob('*/*'):
        ct = cell / 'ct'
        record = read_json(ct / 'CT.json')
        first = record['report']['regions'][0]
        second = dict(first, id='roi-01', missing=cell.name == 'lscm',
                      tiles=[] if cell.name == 'lscm' else first['tiles'])
        seal(ct, 'CT.json', record['identity'], dict(regions=[first, second]), ['plane.png', 'raw.npz'])
    manifest = review.build_review(run, run / 'review')
    key = read_json(run / 'PRIVATE-review-key.json')
    entries = []
    for field in ('175a', 'paris4'):
        panels = [p for p in manifest['panels'] if p['field_id'] == field]
        tokens = {key['keys'][p['token']]['method']: p['token'] for p in panels}
        for roi, winner in [('roi-00', 'lscm'), ('roi-01', 'arap')]:
            entries.append(dict(case_id=panels[0]['case_id'], roi_id=roi, preferred=[tokens[winner]], common_depth_um=0.))
    atomic_json(run / 'votes.json', dict(review_sha256=file_sha(run / 'review/manifest.json'), entries=entries))
    result = review.summarize(run, run / 'votes.json', run / 'report')
    assert result['preferences']['balanced'] == {'arap/sheet': .5, 'lscm/sheet': .5}
    assert result['missing_candidate_regions'] == 2


def reference_controls(run, output):
    from bench_flattening import seal
    rows, files = [], []
    for field in ('175a', 'paris4'):
        case_id = field + '-coarse-Wm060'
        root = output / case_id / 'references'
        root.mkdir(parents=True)
        Image.new('RGBA', (8, 8), (77, 77, 77, 255)).save(root / 'xy.png')
        np.savez(root / 'raw.npz', ct=np.full((4, 8, 8), 77), xyz_um=np.zeros((4, 8, 8, 3)),
                 support=np.ones((4, 8, 8), bool), basis_xyz=np.zeros((4, 2, 3)))
        region = dict(id='roi-00', plane_names=['tangent', 'xy', 'xz', 'yz'],
                      pitch_um=8.64 if field == '175a' else 2.4, offset_um=0., width=8, height=8,
                      raw='raw.npz', planes=['xy.png'] * 4, source_face_row=0,
                      center_xyz_um=[100., 200., 300.], tangent_missing_reason=None)
        seal(root, 'REFERENCES.json', dict(case=case_id), dict(case_id=case_id, regions=[region],
             diagnostic_only=True, physical_acceptance=False), ['raw.npz', 'xy.png'])
        name = case_id + '/references/REFERENCES.json'
        rows.append(dict(case_id=case_id, field_id=field, status='COMPLETE', references=name))
        files.extend([name, case_id + '/references/raw.npz', case_id + '/references/xy.png'])
    seal(output, 'CONTROL-RUN.json', dict(source_run_sha256=file_sha(run / 'RUN.json')),
         dict(rows=rows, fields=['175a', 'paris4'], diagnostic_only=True, physical_acceptance=False),
         files)
    return output


def test_native_controls_are_visible_but_not_vote_candidates(review, tmp_path):
    run = fixture_run(tmp_path)
    controls = reference_controls(run, tmp_path / 'controls')
    manifest = review.build_review(run, run / 'review', controls=controls)
    assert len(manifest['panels']) == 4 and len(manifest['references']) == 2
    region = manifest['references'][0]['regions'][0]
    assert region['plane_names'] == ['tangent', 'xy', 'xz', 'yz'] and region['offset_um'] == 0.
    assert file_sha(run / 'review' / region['raw']) == file_sha(controls / '175a-coarse-Wm060/references/raw.npz')
    assert len(region['planes']) == 4 and all((run / 'review' / p).is_file() for p in region['planes'])
    served = '\n'.join(p.read_text() for p in (run / 'review').rglob('*') if p.suffix in ('.html', '.json'))
    assert 'lscm' not in served.lower() and 'arap' not in served.lower()
    token = manifest['panels'][0]['token']
    entry = dict(case_id=manifest['panels'][0]['case_id'], roi_id='roi-00', preferred=[token], common_depth_um=0.)
    atomic_json(run / 'votes.json', dict(review_sha256=file_sha(run / 'review/manifest.json'), entries=[entry]))
    result = review.summarize(run, run / 'votes.json', run / 'report')
    assert result['reviewed_regions'] == 1 and result['missing_candidate_regions'] == 0
    assert all(set(methods) == {'lscm/sheet', 'arap/sheet'} for methods in result['preferences']['fields'].values())
    with (controls / 'CONTROL-RUN.json').open('a') as stream:
        stream.write('\n')
    with pytest.raises(ValueError, match='identity|changed'):
        review.summarize(run, run / 'votes.json', run / 'changed-report')


def test_controls_from_another_run_are_rejected_before_publication(review, tmp_path):
    from bench_flattening import seal
    run = fixture_run(tmp_path)
    controls = reference_controls(run, tmp_path / 'controls')
    record = json.loads((controls / 'CONTROL-RUN.json').read_text())
    seal(controls, 'CONTROL-RUN.json', dict(source_run_sha256='0' * 64), record['report'], record['files'])
    with pytest.raises(ValueError, match='source|run'):
        review.build_review(run, run / 'review', controls=controls)
    assert not (run / 'review').exists()


def run_with_baseline_reference(tmp_path):
    for name in ('current', 'prior'):
        (tmp_path / name).mkdir()
    run, prior = [fixture_run(tmp_path / name) for name in ('current', 'prior')]
    inventory = dict(entries=[dict(status='READY', case=dict(id=field + '-coarse-Wm060',
                     source_sha256=field + '-original-source', ct_profile=dict(field_id=field, origin_xyz_um=[0., 0., 0.])))
                     for field in ('175a', 'paris4')])
    for root in (run, prior):
        atomic_json(root / 'INVENTORY.json', inventory)
    queue = json.loads((prior / 'QUEUE.json').read_text())
    baselines = []
    for row in queue['rows']:
        if row['method'] == 'lscm':
            row['method'] = 'legacy-lscm'
            baselines.append({k: row[k] for k in ('case_id', 'status', 'output')})
    atomic_json(prior / 'QUEUE.json', queue)
    baseline = dict(run=str(prior), run_sha256=file_sha(prior / 'RUN.json'),
                    inventory_sha256=file_sha(prior / 'INVENTORY.json'), queue_sha256=file_sha(prior / 'QUEUE.json'),
                    rows=baselines)
    saved = json.loads((run / 'RUN.json').read_text())
    atomic_json(run / 'RUN.json', dict(saved, config=dict(baseline_reference=baseline)))
    return run, prior


def test_pinned_prior_baselines_remain_blind_visible_and_votable(review, tmp_path):
    run, prior = run_with_baseline_reference(tmp_path)
    controls = reference_controls(prior, tmp_path / 'controls')
    manifest = review.build_review(run, run / 'review', controls=controls)
    assert len(manifest['panels']) == 6 and len(manifest['references']) == 2
    key = json.loads((run / 'PRIVATE-review-key.json').read_text())
    baseline = next(p for p in manifest['panels'] if key['keys'][p['token']]['method'] == 'legacy-lscm')
    entry = dict(case_id=baseline['case_id'], roi_id='roi-00', preferred=[baseline['token']], common_depth_um=0.)
    atomic_json(run / 'votes.json', dict(review_sha256=file_sha(run / 'review/manifest.json'), entries=[entry]))
    result = review.summarize(run, run / 'votes.json', run / 'report')
    assert result['preferences']['fields'][baseline['field_id']]['legacy-lscm/sheet'] == 1.
    served = '\n'.join(p.read_text() for p in (run / 'review').rglob('*') if p.suffix in ('.html', '.json'))
    assert 'legacy-lscm' not in served and str(prior) not in served
    with (prior / 'QUEUE.json').open('a') as stream:
        stream.write('\n')
    with pytest.raises(ValueError, match='baseline|identity'):
        review.summarize(run, run / 'votes.json', run / 'changed-report')


def test_different_physical_source_cannot_import_prior_baseline(review, tmp_path):
    run, prior = run_with_baseline_reference(tmp_path)
    inventory = json.loads((run / 'INVENTORY.json').read_text())
    inventory['entries'][0]['case']['ct_profile']['origin_xyz_um'] = [1., 0., 0.]
    atomic_json(run / 'INVENTORY.json', inventory)
    with pytest.raises(ValueError, match='source|profile'):
        review.build_review(run, run / 'review')
    assert not (run / 'review').exists()


@pytest.mark.parametrize('active_holdout', [False, True])
def test_baseline_run_private_key_cannot_be_served_as_new_output(review, tmp_path, active_holdout):
    run, prior = run_with_baseline_reference(tmp_path)
    if active_holdout:
        queue = json.loads((run / 'QUEUE.json').read_text())
        for row in queue['rows']:
            row['stage'] = 'holdout'
        atomic_json(run / 'QUEUE.json', dict(queue, active_stage='holdout'))
        freeze_run(run)
    with pytest.raises(ValueError, match='private|separate'):
        review.build_review(run, prior)
    assert not (prior / 'index.html').exists()


def test_resource_limited_prior_baseline_stays_in_comparison_matrix(review, tmp_path):
    run, prior = run_with_baseline_reference(tmp_path)
    queue = json.loads((prior / 'QUEUE.json').read_text())
    old = next(r for r in queue['rows'] if r['method'] == 'legacy-lscm' and r['field_id'] == 'paris4')
    old['status'] = 'RESOURCE_LIMIT'
    atomic_json(prior / 'QUEUE.json', queue)
    saved = json.loads((run / 'RUN.json').read_text())
    baseline = saved['config']['baseline_reference']
    baseline['queue_sha256'] = file_sha(prior / 'QUEUE.json')
    next(r for r in baseline['rows'] if r['case_id'] == old['case_id'])['status'] = 'RESOURCE_LIMIT'
    atomic_json(run / 'RUN.json', saved)
    manifest = review.build_review(run, run / 'review')
    assert len(manifest['panels']) == 6
    missing = [p for p in manifest['panels'] if p['status'] == 'RESOURCE_LIMIT']
    assert len(missing) == 1 and missing[0]['field_id'] == 'paris4' and not missing[0]['regions']


def run_with_trial_reference(tmp_path, status='COMPLETE'):
    from bench_flattening import seal
    run, prior = run_with_baseline_reference(tmp_path)
    queue = json.loads((prior / 'QUEUE.json').read_text())
    references = []
    for row in queue['rows']:
        if row['method'] != 'legacy-lscm':
            continue
        row.update(method='slim', status=status)
        seal(prior / row['output'], 'CELL.json', dict(source=row['case_id']),
             dict(status=status), ['ct/CT.json'])
        reference = dict(row)
        reference['cell_sha256'] = file_sha(prior / row['output'] / 'CELL.json')
        if status == 'COMPLETE':
            reference['ct_sha256'] = file_sha(prior / row['output'] / 'ct/CT.json')
        references.append(reference)
    atomic_json(prior / 'QUEUE.json', queue)
    saved = json.loads((run / 'RUN.json').read_text())
    source = saved['config'].pop('baseline_reference')
    source.update(rows=references, queue_sha256=file_sha(prior / 'QUEUE.json'))
    saved['config']['trial_references'] = [source]
    atomic_json(run / 'RUN.json', saved)
    return run, prior


def test_explicit_trial_import_is_blind_votable_and_invalidated_by_source_change(review, tmp_path):
    run, prior = run_with_trial_reference(tmp_path)
    manifest = review.build_review(run, run / 'review', controls=reference_controls(prior, tmp_path / 'controls'))
    assert len(manifest['panels']) == 6 and len(manifest['references']) == 2
    key = json.loads((run / 'PRIVATE-review-key.json').read_text())
    imported = next(p for p in manifest['panels'] if key['keys'][p['token']]['method'] == 'slim')
    region = imported['regions'][0]
    assert file_sha(run / 'review' / region['tiles'][0]['raw']) == file_sha(
        prior / f"cells/{imported['field_id']}/lscm/ct/raw.npz")
    entry = dict(case_id=imported['case_id'], roi_id='roi-00', preferred=[imported['token']], common_depth_um=0.)
    atomic_json(run / 'votes.json', dict(review_sha256=file_sha(run / 'review/manifest.json'), entries=[entry]))
    result = review.summarize(run, run / 'votes.json', run / 'report')
    assert result['preferences']['fields'][imported['field_id']]['slim/sheet'] == 1.
    served = '\n'.join(p.read_text() for p in (run / 'review').rglob('*') if p.suffix in ('.html', '.json'))
    assert str(prior) not in served and 'slim' not in served.lower()
    with (prior / 'cells/175a/lscm/CELL.json').open('a') as stream:
        stream.write('\n')
    with pytest.raises(ValueError, match='trial|identity'):
        review.summarize(run, run / 'votes.json', run / 'changed-report')


@pytest.mark.parametrize('problem', ['source', 'row', 'unpinned', 'ct', 'cell', 'duplicate',
                                   'duplicate-current', 'running', 'holdout'])
def test_invalid_trial_import_is_rejected_before_served_files_are_created(review, tmp_path, problem):
    run, prior = run_with_trial_reference(tmp_path)
    saved = json.loads((run / 'RUN.json').read_text())
    source = saved['config']['trial_references'][0]
    row = source['rows'][0]
    queue = json.loads((prior / 'QUEUE.json').read_text())
    old = next(r for r in queue['rows'] if r['output'] == row['output'])
    if problem == 'source':
        inventory = json.loads((run / 'INVENTORY.json').read_text())
        inventory['entries'][0]['case']['ct_profile']['origin_xyz_um'] = [1., 0., 0.]
        atomic_json(run / 'INVENTORY.json', inventory)
    elif problem == 'row':
        row['scope'] = 'patches'
    elif problem == 'unpinned':
        del row['ct_sha256']
    elif problem in ('ct', 'cell'):
        path = prior / row['output'] / ('ct/CT.json' if problem == 'ct' else 'CELL.json')
        with path.open('a') as stream:
            stream.write('\n')
    elif problem == 'duplicate':
        source['rows'].append(dict(row))
    else:
        if problem == 'duplicate-current':
            old['method'] = row['method'] = 'lscm'
        elif problem == 'running':
            old['status'] = row['status'] = 'RUNNING'
        else:
            old['stage'] = row['stage'] = 'holdout'
        atomic_json(prior / 'QUEUE.json', queue)
        source['queue_sha256'] = file_sha(prior / 'QUEUE.json')
    atomic_json(run / 'RUN.json', saved)
    with pytest.raises(ValueError, match='trial|source|profile|duplicate|FREEZE'):
        review.build_review(run, run / 'review')
    assert not (run / 'review').exists()


@pytest.mark.parametrize('hidden', [False, True])
def test_imported_trial_private_run_stays_unservable_even_when_stage_hidden(review, tmp_path, hidden):
    run, prior = run_with_trial_reference(tmp_path)
    if hidden:
        queue = json.loads((run / 'QUEUE.json').read_text())
        for row in queue['rows']:
            row['stage'] = 'holdout'
        atomic_json(run / 'QUEUE.json', dict(queue, active_stage='holdout'))
        freeze_run(run)
    with pytest.raises(ValueError, match='private|separate'):
        review.build_review(run, prior)
    assert not (prior / 'index.html').exists()


def test_inherited_trial_resource_gaps_remain_in_the_denominator(review, tmp_path):
    run, prior = run_with_trial_reference(tmp_path, 'RESOURCE_LIMIT')
    manifest = review.build_review(run, run / 'review')
    assert len(manifest['panels']) == 6
    assert sum(p['status'] == 'RESOURCE_LIMIT' and not p['regions'] for p in manifest['panels']) == 2
    panel = next(p for p in manifest['panels'] if p['field_id'] == '175a' and p['status'] == 'COMPLETE')
    atomic_json(run / 'votes.json', dict(review_sha256=file_sha(run / 'review/manifest.json'), entries=[
        dict(case_id=panel['case_id'], roi_id='roi-00', preferred=[panel['token']], common_depth_um=0.)]))
    result = review.summarize(run, run / 'votes.json', run / 'report')
    assert result['missing_candidate_regions'] == 1
    assert result['preferences']['fields']['175a']['slim/sheet'] == 0.


def test_trial_cohort_keeps_pinned_legacy_baselines_as_well_as_new_and_inherited_trials(review, tmp_path):
    run, prior = run_with_trial_reference(tmp_path)
    (tmp_path / 'baseline-pair').mkdir()
    _, baseline = run_with_baseline_reference(tmp_path / 'baseline-pair')
    baseline_rows = [r for r in json.loads((baseline / 'QUEUE.json').read_text())['rows']
                     if r['method'] == 'legacy-lscm']
    saved = json.loads((run / 'RUN.json').read_text())
    saved['config']['baseline_reference'] = dict(
        run=str(baseline), run_sha256=file_sha(baseline / 'RUN.json'),
        inventory_sha256=file_sha(baseline / 'INVENTORY.json'), queue_sha256=file_sha(baseline / 'QUEUE.json'),
        rows=[{k: r[k] for k in ('case_id', 'output', 'status')} for r in baseline_rows])
    atomic_json(run / 'RUN.json', saved)
    manifest = review.build_review(run, run / 'review')
    assert len(manifest['panels']) == 8
    key = json.loads((run / 'PRIVATE-review-key.json').read_text())
    assert {r['method'] for r in key['keys'].values()} == {'legacy-lscm', 'lscm', 'arap', 'slim'}
    assert sum(r['_source_run'] == str(prior) for r in key['keys'].values()) == 2


def test_inherited_holdout_requires_cohort_freeze_not_just_source_freeze(review, tmp_path):
    run, prior = run_with_trial_reference(tmp_path)
    queue = json.loads((prior / 'QUEUE.json').read_text())
    saved = json.loads((run / 'RUN.json').read_text())
    source = saved['config']['trial_references'][0]
    for reference in source['rows']:
        reference['stage'] = 'holdout'
        next(r for r in queue['rows'] if r['output'] == reference['output'])['stage'] = 'holdout'
    atomic_json(prior / 'FREEZE.json', {})
    atomic_json(prior / 'QUEUE.json', queue)
    source['queue_sha256'] = file_sha(prior / 'QUEUE.json')
    atomic_json(run / 'RUN.json', saved)
    with pytest.raises(ValueError, match='FREEZE'):
        review.build_review(run, run / 'review')
    assert not (run / 'review').exists()


def freeze_run(run):
    decision = run / 'human-decision.json'
    atomic_json(decision, dict(status='fixture human decision'))
    atomic_json(run / 'FREEZE.json', dict(run_sha256=file_sha(run / 'RUN.json'),
                decision=str(decision), decision_sha256=file_sha(decision),
                candidates=[['lscm', 'sheet'], ['arap', 'sheet'], ['slim', 'sheet'], ['legacy-lscm', 'sheet']]))


@pytest.mark.parametrize('invalid', ['empty', 'run', 'decision'])
def test_unbound_holdout_freeze_cannot_expose_panels(review, tmp_path, invalid):
    run = fixture_run(tmp_path, stage='holdout')
    freeze_run(run)
    freeze = json.loads((run / 'FREEZE.json').read_text())
    if invalid == 'empty':
        freeze = {}
    elif invalid == 'run':
        freeze['run_sha256'] = '0' * 64
    else:
        (run / 'human-decision.json').write_text('changed decision')
    atomic_json(run / 'FREEZE.json', freeze)
    with pytest.raises(ValueError, match='FREEZE|freeze|decision'):
        review.build_review(run, run / 'review')
    assert not (run / 'review').exists()


def test_bound_holdout_review_rechecks_human_decision_on_vote_import(review, tmp_path):
    run = fixture_run(tmp_path, stage='holdout')
    freeze_run(run)
    manifest = review.build_review(run, run / 'review')
    assert len(manifest['panels']) == 4
    atomic_json(run / 'votes.json', dict(review_sha256=file_sha(run / 'review/manifest.json'), entries=[]))
    (run / 'human-decision.json').write_text('changed decision')
    with pytest.raises(ValueError, match='FREEZE|freeze|decision'):
        review.summarize(run, run / 'votes.json', run / 'report')
    assert not (run / 'report').exists()


@pytest.mark.parametrize('location', ['gallery', 'child', 'ancestor'])
def test_unblinded_report_cannot_overlap_served_gallery(review, tmp_path, location):
    run = fixture_run(tmp_path)
    gallery = run / 'review'
    review.build_review(run, gallery)
    atomic_json(run / 'votes.json', dict(review_sha256=file_sha(gallery / 'manifest.json'), entries=[]))
    output = {'gallery': gallery, 'child': gallery / 'report', 'ancestor': run}[location]
    before = {str(p.relative_to(gallery)): file_sha(p) for p in gallery.rglob('*') if p.is_file()}
    with pytest.raises(ValueError, match='private|served|overlap'):
        review.summarize(run, run / 'votes.json', output)
    assert before == {str(p.relative_to(gallery)): file_sha(p) for p in gallery.rglob('*') if p.is_file()}
    assert not (output / 'RESULTS.csv').exists()
