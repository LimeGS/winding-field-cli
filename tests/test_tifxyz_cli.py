"""T7 wiring tests: only the external Linux job boundary is substituted.

The receipt fixture is a protocol double, not a geometric acceptance oracle.
Source binding, the legacy producer, configuration and parent state are real.
"""
from dataclasses import asdict
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

from winding_field.cli import main
from winding_field.tifxyz_contract import Settings, load_settings, validate_request
from winding_field.tifxyz_field import open_physical_field, selected_frame
from winding_field.util import atomic_json, digest, file_sha
from tifxyz_fixture import field_run, mutate_json


FLAGS = dict(physical_acceptance=False, native_face_identity=False,
             native_topology_equivalence='UNMEASURED')
READER = dict(cell_rule='four-corners', diagonals=['nw-se', 'ne-sw'], inpainting=False)


def call(*args):
    try:
        return main(list(map(str, args)))
    except SystemExit as error:
        return error.code


def tree_state(root):
    return {str(p.relative_to(root)): (file_sha(p), p.stat().st_mtime_ns)
            for p in root.rglob('*') if p.is_file()}


@pytest.fixture
def request_file(tmp_path):
    path = tmp_path/'export.json'
    atomic_json(path, dict(levels=[1.], reserve_disk_gib=0, max_pixels=4096))
    return path


@pytest.fixture
def job(monkeypatch):
    from winding_field import tifxyz_runtime
    outcomes, calls = [], []

    def supervise(request, stop=None):
        calls.append(request)
        out = Path(request['out'])
        verifying = request['operation']=='verify'
        saved = json.loads((out/'EXPORT.json').read_bytes()) if verifying or request.get('resume') else None
        source = saved['source_opener']['path'] if verifying else request['field_run']
        config = saved['effective_settings'] if verifying else request['settings']
        with open_physical_field(source) as field:
            settings = Settings(**config) if 'memory_bytes' in config else load_settings(config, field.binding['geometry'], selected_frame(field.binding))
            identity = dict(schema='winding-field-tifxyz-export-v1', source=field.identity,
                            source_opener=dict(kind='field-run', path=str(source)),
                            effective_settings=asdict(settings), reader=READER, **FLAGS)
            identity = json.loads(json.dumps(identity))
            if saved is not None and saved!=identity:
                raise ValueError('export source/settings identity differs')
            field.validate(hashes=True)
        if verifying:
            receipt = json.loads((out/'TIFXYZ.json').read_bytes())
            ready = json.loads((out/(receipt['status']+'.json')).read_bytes())
            assert ready['tifxyz_sha256']==file_sha(out/'TIFXYZ.json')
            for name, record in receipt['levels'][0]['manifest']['files'].items():
                if file_sha(out/'level-0000'/name)!=record['sha256']:
                    raise ValueError('segment artifact changed')
            return dict(status='TIFXYZ_VERIFIED' if receipt['coverage_complete'] else 'TIFXYZ_PARTIAL_VERIFIED',
                        coverage_complete=receipt['coverage_complete'], levels=1, **FLAGS)
        if not request['resume']:
            out.mkdir()
            atomic_json(out/'EXPORT.json', identity)
            (out/'producer-run.snapshot.json').write_bytes((Path(source)/'RUN.json').read_bytes())
            (out/'.lock').touch()
        result = outcomes.pop(0) if outcomes else dict(status='READY', coverage_complete=True, **FLAGS)
        if stop and stop():
            result = dict(status='PAUSED')
        atomic_json(out/'STATUS.json', result)
        if result['status'] in ('READY', 'PARTIAL'):
            quality = dict(status=result['status'], coverage_complete=result['coverage_complete'])
            complete = quality['coverage_complete']
            audit = dict(status='PASS' if complete else 'BLOCKED', geometry_complete=True,
                         blocked_phase=None if complete else 'coverage', source_scan_complete=complete,
                         remaining_source_coverage='NONE' if complete else 'UNMEASURED',source_identity=identity['source'],
                         reasons=[] if complete else [dict(code='SOURCE_UNCOVERED', region=dict(
                             cell_zyx=[0, 0, 0], global_cell_zyx=list(settings.roi_lo_zyx),
                             subcell_index_zyx=[0, 0, 0], depth=0))],**FLAGS)
            segment = out/'level-0000'; segment.mkdir(exist_ok=True)
            for name in ('x.tif', 'y.tif', 'z.tif'):
                (segment/name).write_bytes(b'external-job-protocol-fixture')
            atomic_json(segment/'meta.json', dict(format='tifxyz', type='seg', scale=[1., 1.]))
            atomic_json(segment/'VERIFIED.json', dict(reader=READER, audit=audit, **quality, **FLAGS))
            work = out/'work'; work.mkdir(exist_ok=True)
            (work/'retained.bin').write_bytes(b'retained work')
            manifest = dict(files={p.name: dict(bytes=p.stat().st_size, sha256=file_sha(p)) for p in segment.iterdir()})
            atomic_json(out/'TIFXYZ.json', dict(identity_sha256=digest(identity), reader=READER,
                        levels=[dict(level=settings.levels[0], segment='level-0000', manifest=manifest, **quality)],
                        **quality, **FLAGS))
            atomic_json(out/(result['status']+'.json'), dict(tifxyz_sha256=file_sha(out/'TIFXYZ.json'),
                        identity_sha256=digest(identity), **quality, **FLAGS))
        return result

    monkeypatch.setattr(tifxyz_runtime, 'supervise', supervise)
    return outcomes, calls


@pytest.fixture
def integrated(tmp_path, request_file, monkeypatch):
    from winding_field.demo import synthetic
    def forbidden(*args, **kwargs):
        pytest.fail('direct branch entered native mesh/flatten/gallery')
    monkeypatch.setattr('winding_field.meshes.mesh', forbidden)
    monkeypatch.setattr('winding_field.flatten.flatten', forbidden)
    monkeypatch.setattr('winding_field.demo.gallery', forbidden)
    source = synthetic(tmp_path/'ct.zarr')
    root = tmp_path/'run'
    args = ['run', '--source', source, '--output', root, '--prediction', 'classical',
            '--sweeps', '1', '--seed-stride', '4', '--threads', '1', '--reserve-gib', '0',
            '--cache', tmp_path/'cache', '--surface-output', 'tifxyz', '--tifxyz-config', request_file]
    return root, args


def test_direct_default_terminal_has_no_native_artifacts(integrated, job):
    root, args = integrated
    assert call(*args)==0
    state = json.loads((root/'RUN.json').read_bytes())
    assert state['status']=='COMPLETE'
    assert set(state['stages'])=={'field', 'fibres', 'prepare', 'tifxyz'}
    assert state['stages']['tifxyz']['receipt']=='tifxyz/TIFXYZ.json'
    assert state['stages']['tifxyz']['ready_sha256']==file_sha(root/'tifxyz/READY.json')
    assert state['experimental'] is True and state['comparison_status']=='UNMEASURED'
    assert not (root/'meshes').exists() and not (root/'flat').exists()
    assert not list(root.rglob('*.obj')) and not (root/'index.html').exists()
    assert call('verify-tifxyz', root/'tifxyz')==0


def test_until_field_freezes_content_and_resume_uses_saved_branch(integrated, job, request_file):
    root, args = integrated
    assert call(*args, '--until', 'field', '--levels', '1')==0
    state = json.loads((root/'RUN.json').read_bytes())
    saved = state['config']
    assert state['status']=='PAUSED' and not job[1]
    assert saved['tifxyz_request']==validate_request(dict(levels=[1.], reserve_disk_gib=0, max_pixels=4096))
    assert saved['tifxyz_request_sha256']==digest(saved['tifxyz_request'])
    assert saved['tifxyz_request']['pitch_um'] is None
    request_file.write_text('{"levels": [999]}')
    assert call('resume', root)==0
    state = json.loads((root/'RUN.json').read_bytes())
    assert state['status']=='COMPLETE' and state['config']==saved
    assert job[1][-1]['settings']['levels']==[1.]


@pytest.mark.parametrize('until', ['mesh', 'flatten'])
def test_incompatible_stage_rejected_before_source_io(tmp_path, request_file, monkeypatch, until):
    monkeypatch.setattr('winding_field.cli.resolve_source', lambda *a, **k: pytest.fail('source resolution before validation'))
    assert call('run', '--source', 'missing', '--output', tmp_path/'run', '--surface-output', 'tifxyz',
                '--tifxyz-config', request_file, '--until', until)==1
    assert not (tmp_path/'run').exists()


@pytest.mark.parametrize('text, levels', [
    ('{"levels":[1],"typo":3}', []), ('{"levels":[1],"levels":[2]}', []),
    ('{"levels":[NaN]}', []), ('{"levels":[1,2]}', ['--levels', '2', '1']),
    ('{"levels":[1]}', ['--levels', '2']), ('{}', []),
])
def test_bad_request_rejected_before_source_io(tmp_path, request_file, monkeypatch, text, levels):
    request_file.write_text(text)
    monkeypatch.setattr('winding_field.cli.resolve_source', lambda *a, **k: pytest.fail('source resolution before validation'))
    assert call('plan', '--source', 'missing', '--output', tmp_path/'run', '--surface-output', 'tifxyz',
                '--tifxyz-config', request_file, *levels)==1
    assert not (tmp_path/'run').exists()


def test_plan_estimates_only_branch_products(tmp_path, request_file, capsys):
    from winding_field.demo import synthetic
    source = synthetic(tmp_path/'ct.zarr')
    assert call('plan', '--source', source, '--output', tmp_path/'run', '--surface-output', 'tifxyz',
                '--tifxyz-config', request_file)==0
    result = json.loads(capsys.readouterr().out)
    assert result['stages']==['prepare', 'fibres', 'field', 'tifxyz']
    assert set(result['estimated_products_by_stage'])==set(result['stages'])
    assert result['estimated_products_bytes']==sum(result['estimated_products_by_stage'].values())
    assert result['reserve_ram_bytes']==2<<30
    assert result['tifxyz_reserve_disk_bytes']==0
    assert result['tifxyz_scratch_bytes']==16<<30
    assert not (tmp_path/'run').exists()


@pytest.mark.parametrize('result, code', [
    (dict(status='PAUSED'), 0),
    (dict(status='BLOCKED', reasons=[dict(code='RAM_HEADROOM')]), 75),
    (dict(status='BLOCKED', reasons=[dict(code='RAY_WORK_LIMIT')]), 75),
    (dict(status='BLOCKED', reasons=[dict(code='COVERAGE_DEPTH_LIMIT')]), 75),
    (dict(status='BLOCKED', reasons=[dict(code='PROJECTION_AMBIGUOUS')]), 2),
    (dict(status='FAILED', error='integrity failure'), 1),
])
def test_nonready_child_never_completes_stage(integrated, job, result, code):
    root, args = integrated
    job[0].append(result)
    assert call(*args)==code
    state = json.loads((root/'RUN.json').read_bytes())
    assert state['status']==result['status']
    assert 'tifxyz' not in state['stages']
    assert not (root/'tifxyz/READY.json').exists()
    with sqlite3.connect(root/'products.sqlite') as db:
        assert db.execute("SELECT count(*) FROM exports WHERE stage='tifxyz'").fetchone()[0]==0


def test_paused_export_reuses_effective_settings_and_parent_stop(integrated, job, request_file):
    root, args = integrated
    job[0].append(dict(status='PAUSED'))
    assert call(*args)==0
    effective = json.loads((root/'tifxyz/EXPORT.json').read_bytes())['effective_settings']
    (root/'STOP').touch()
    request_file.unlink()
    assert call('resume', root)==0
    assert not (root/'STOP').exists()
    assert job[1][-1]['resume'] is True and job[1][-1]['settings']==effective
    assert call('verify-tifxyz', root/'tifxyz')==0


@pytest.mark.parametrize('artifact', ['level-0000/x.tif', 'level-0000/meta.json', 'work/retained.bin', 'READY.json', 'EXPORT.json', 'producer-run.snapshot.json'])
def test_completed_resume_detects_every_export_artifact(integrated, job, artifact):
    root, args = integrated
    assert call(*args)==0
    (root/'tifxyz'/artifact).unlink()
    assert call('resume', root)==1
    assert json.loads((root/'RUN.json').read_bytes())['status']=='FAILED'


def test_standalone_old_source_is_readonly_without_ct_or_workflow(tmp_path, request_file, job, monkeypatch):
    source = field_run(tmp_path/'old-producer')
    out = tmp_path/'out'
    before = tree_state(source)
    def forbidden(*args, **kwargs):
        pytest.fail('standalone tried workflow/CT')
    monkeypatch.setattr('winding_field.workflow.run', forbidden)
    monkeypatch.setattr('winding_field.cli.Volume', forbidden)
    monkeypatch.setattr('winding_field.workflow.open_volume', forbidden)
    for p in source.rglob('*'):
        p.chmod(0o555 if p.is_dir() else 0o444)
    source.chmod(0o555)
    try:
        assert call('tifxyz', source, '--output', out, '--config', request_file)==0
        request_file.unlink()
        assert call('tifxyz', source, '--output', out, '--resume')==0
        export_before = tree_state(out)
        assert call('verify-tifxyz', out)==0
        assert tree_state(out)==export_before and tree_state(source)==before
        assert not (source/'.run.lock').exists()
    finally:
        source.chmod(0o755)
        for p in source.rglob('*'):
            p.chmod(0o755 if p.is_dir() else 0o644)


@pytest.mark.parametrize('target', ['producer', 'source', 'field', 'checkpoint', 'payload', 'metadata'])
def test_standalone_rejects_changed_bindings(tmp_path, request_file, job, target):
    source = field_run(tmp_path/'source'); out = tmp_path/'out'
    assert call('tifxyz', source, '--output', out, '--config', request_file)==0
    if target=='producer':
        mutate_json(source/'RUN.json', lambda d: d.update(implementation_sha256='c'*64))
    elif target=='source':
        mutate_json(source/'RUN.json', lambda d: d['source'].update(metadata_sha256='c'*64))
    elif target=='field':
        mutate_json(source/'FIELD.json', lambda d: d['geometry'].update(origin_xyz=[3, 0, 0]))
    elif target=='checkpoint':
        mutate_json(source/'field/CHECKPOINT.json', lambda d: d.update(changed=True))
    elif target=='payload':
        p = next((source/'field').rglob('*.npz')); p.write_bytes(p.read_bytes()+b'changed')
    else:
        (out/'level-0000/meta.json').write_text('{}')
    before = tree_state(source)
    assert call('verify-tifxyz', out)==1
    assert tree_state(source)==before


@pytest.mark.parametrize('spelling', ['source-parent', 'output-parent', 'symlink'])
def test_standalone_normalizes_paths_before_protecting_source(tmp_path, request_file, job, spelling):
    source = field_run(tmp_path/'source'); out = source/'nested'
    given_source, given_out = source, out
    if spelling=='source-parent': given_source = source/'field'/'..'
    elif spelling=='output-parent':
        (tmp_path/'other').mkdir()
        given_out = tmp_path/'other'/'..'/'source'/'nested'
    else:
        given_source = tmp_path/'alias'; given_source.symlink_to(source, target_is_directory=True)
    before = tree_state(source)
    assert call('tifxyz', given_source, '--output', given_out, '--config', request_file)==1
    assert not out.exists() and tree_state(source)==before


def test_standalone_resume_config_must_agree_and_missing_root_is_not_repaired(tmp_path, request_file, job):
    source = field_run(tmp_path/'source'); out = tmp_path/'out'
    job[0].append(dict(status='PAUSED'))
    assert call('tifxyz', source, '--output', out, '--config', request_file)==0
    before = tree_state(out)
    request_file.write_text('{"levels":[99]}')
    assert call('tifxyz', source, '--output', out, '--config', request_file, '--resume')==1
    assert tree_state(out)==before
    assert call('tifxyz', source, '--output', tmp_path/'missing', '--resume')==1
    assert not (tmp_path/'missing').exists()


def test_legacy_inspect_has_no_early_tifxyz_dependency_imports(tmp_path):
    from winding_field.demo import synthetic
    source = synthetic(tmp_path/'ct.zarr')
    script = '''import sys
class Guard:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in ('igl', 'tifffile', 'torch') or fullname.startswith('winding_field.tifxyz'):
            raise AssertionError('unexpected dependency: '+fullname)
sys.meta_path.insert(0, Guard())
from winding_field.cli import main
raise SystemExit(main(['inspect', sys.argv[1]]))
'''
    result = subprocess.run([sys.executable, '-B', '-c', script, source], cwd=tmp_path,
                            env=dict(os.environ, OMP_NUM_THREADS='1'), capture_output=True, text=True)
    assert result.returncode==0, result.stderr


@pytest.mark.parametrize('damage', ['missing-ready', 'wrong-binding', 'wrong-flags'])
def test_ready_reply_requires_bound_publication_before_registering(integrated, job, monkeypatch, damage):
    from winding_field import tifxyz_runtime
    original = tifxyz_runtime.supervise
    def damaged(request, stop=None):
        result = original(request, stop)
        out = Path(request['out'])
        if damage=='missing-ready':
            (out/'READY.json').unlink()
        elif damage=='wrong-binding':
            mutate_json(out/'READY.json', lambda d: d.update(tifxyz_sha256='0'*64))
        else:
            result['physical_acceptance'] = True
        return result
    monkeypatch.setattr(tifxyz_runtime, 'supervise', damaged)
    root, args = integrated
    assert call(*args)==1
    state = json.loads((root/'RUN.json').read_bytes())
    assert state['status']=='FAILED' and 'tifxyz' not in state['stages']
    with sqlite3.connect(root/'products.sqlite') as db:
        assert db.execute("SELECT count(*) FROM exports WHERE stage='tifxyz'").fetchone()[0]==0


def test_standalone_sigterm_uses_stop_callback_and_restores_handler(tmp_path, request_file, job, monkeypatch):
    import signal
    from winding_field import tifxyz_runtime
    original = tifxyz_runtime.supervise
    previous = signal.getsignal(signal.SIGTERM)
    def interrupt(request, stop=None):
        assert callable(stop), 'CLI must provide a cooperative signal callback'
        signal.raise_signal(signal.SIGTERM)
        return original(request, stop)
    monkeypatch.setattr(tifxyz_runtime, 'supervise', interrupt)
    source = field_run(tmp_path/'source'); out = tmp_path/'out'
    assert call('tifxyz', source, '--output', out, '--config', request_file)==0
    assert json.loads((out/'STATUS.json').read_bytes())['status']=='PAUSED'
    assert signal.getsignal(signal.SIGTERM)==previous
    assert not (out/'READY.json').exists()


def test_resume_verification_blocked_does_not_report_complete(integrated, job, monkeypatch):
    from winding_field import tifxyz_runtime
    root, args = integrated
    assert call(*args)==0
    before = tree_state(root/'tifxyz')
    original = tifxyz_runtime.supervise
    monkeypatch.setattr(tifxyz_runtime, 'supervise', lambda request, stop=None:
                        dict(status='BLOCKED', reasons=[dict(code='RAM_HEADROOM')]))
    assert call('resume', root)==75
    assert json.loads((root/'RUN.json').read_bytes())['status']=='BLOCKED'
    assert tree_state(root/'tifxyz')==before
    monkeypatch.setattr(tifxyz_runtime, 'supervise', original)
    assert call('resume', root)==0
    state = json.loads((root/'RUN.json').read_bytes())
    assert state['status']=='COMPLETE'
    assert state['tifxyz_result']['status']=='TIFXYZ_VERIFIED'
    assert tree_state(root/'tifxyz')==before


def test_integrated_resume_does_not_repair_foreign_export_root(integrated, job):
    root, args = integrated
    assert call(*args, '--until', 'field')==0
    (root/'tifxyz').mkdir()
    (root/'tifxyz/foreign').write_text('keep')
    before = tree_state(root/'tifxyz')
    assert call('resume', root)==1
    assert tree_state(root/'tifxyz')==before and not job[1]


@pytest.mark.skipif(sys.platform=='linux', reason='unsupported platform contract')
def test_mac_cli_reports_unsupported_runtime_without_creating_output(tmp_path, request_file, capsys):
    source = field_run(tmp_path/'source'); out = tmp_path/'out'
    assert call('tifxyz', source, '--output', out, '--config', request_file)==75
    result = json.loads(capsys.readouterr().out)
    assert result['status']=='BLOCKED' and result['reasons'][0]['code']=='RUNTIME_UNSUPPORTED'
    assert not out.exists()


@pytest.fixture
def partial_request(request_file):
    mutate_json(request_file, lambda d: d.update(coverage_policy='allow_partial'))
    return request_file


def partial_result():
    return dict(status='PARTIAL', coverage_complete=False, **FLAGS)


def test_partial_stage_registered_verify_readonly_and_resume_stays_partial(integrated, job, partial_request):
    root, args = integrated
    job[0].append(partial_result())
    assert call(*args)==2
    state = json.loads((root/'RUN.json').read_bytes())
    assert state['status']=='PARTIAL' and state['coverage_complete'] is False
    record = state['stages']['tifxyz']
    assert record['status']=='PARTIAL' and record['coverage_complete'] is False
    assert record['partial_sha256']==file_sha(root/'tifxyz/PARTIAL.json')
    assert 'ready_sha256' not in record and not (root/'tifxyz/READY.json').exists()
    with sqlite3.connect(root/'products.sqlite') as db:
        assert db.execute("SELECT sha FROM exports WHERE path='tifxyz/PARTIAL.json'").fetchone()[0]==record['partial_sha256']
    before = tree_state(root/'tifxyz')
    assert call('verify-tifxyz', root/'tifxyz')==2
    assert tree_state(root/'tifxyz')==before
    assert call('resume', root)==2
    state = json.loads((root/'RUN.json').read_bytes())
    assert state['status']=='PARTIAL' and state['coverage_complete'] is False
    assert state['tifxyz_result']['status']=='TIFXYZ_PARTIAL_VERIFIED'
    assert tree_state(root/'tifxyz')==before
    assert not (root/'meshes').exists() and not (root/'index.html').exists()


def test_standalone_partial_export_and_verify_exit_two(tmp_path, partial_request, job, capsys):
    source = field_run(tmp_path/'source'); out = tmp_path/'out'
    job[0].append(partial_result())
    assert call('tifxyz', source, '--output', out, '--config', partial_request)==2
    assert json.loads(capsys.readouterr().out)['coverage_complete'] is False
    before = tree_state(out)
    assert call('verify-tifxyz', out)==2
    result = json.loads(capsys.readouterr().out)
    assert result['status']=='TIFXYZ_PARTIAL_VERIFIED' and result['coverage_complete'] is False
    assert tree_state(out)==before


def test_allow_partial_can_still_complete(integrated, job, partial_request):
    root, args = integrated
    assert call(*args)==0
    state = json.loads((root/'RUN.json').read_bytes())
    assert state['status']=='COMPLETE' and state['coverage_complete'] is True
    assert state['stages']['tifxyz']['ready_sha256']==file_sha(root/'tifxyz/READY.json')
    assert not (root/'tifxyz/PARTIAL.json').exists()


@pytest.mark.parametrize('artifact', ['PARTIAL.json', 'level-0000/x.tif', 'level-0000/VERIFIED.json', 'work/retained.bin'])
def test_partial_resume_detects_changed_artifacts(integrated, job, partial_request, artifact):
    root, args = integrated
    job[0].append(partial_result())
    assert call(*args)==2
    path = root/'tifxyz'/artifact
    path.write_bytes(path.read_bytes()+b'changed')
    assert call('resume', root)==1
    assert json.loads((root/'RUN.json').read_bytes())['status']=='FAILED'


@pytest.mark.parametrize('damage', [
    'both-markers', 'missing-marker', 'marker-status', 'marker-coverage', 'root-coverage',
    'root-status', 'result-coverage', 'level-coverage', 'verified-coverage', 'strict-child',
    'wrong-binding', 'identity-binding', 'geometry', 'scan-complete', 'missing-region', 'hard-failure',
    'remaining-none','malformed-region','audit-pass','audit-physical',
])
def test_partial_publication_rejected_before_registration(integrated, job, partial_request, monkeypatch, damage):
    from winding_field import tifxyz_runtime
    original = tifxyz_runtime.supervise
    def damaged(request, stop=None):
        result = original(request, stop)
        out = Path(request['out'])
        marker = out/'PARTIAL.json'
        if damage=='both-markers':
            (out/'READY.json').write_bytes(marker.read_bytes())
        elif damage=='missing-marker':
            marker.unlink()
        elif damage=='marker-status':
            mutate_json(marker, lambda d: d.update(status='READY'))
        elif damage=='marker-coverage':
            mutate_json(marker, lambda d: d.update(coverage_complete=True))
        elif damage in ('root-coverage', 'root-status', 'level-coverage'):
            def change(d):
                if damage=='root-status': d['status']='READY'
                elif damage=='level-coverage': d['levels'][0]['coverage_complete']=True
                else: d['coverage_complete']=True
            mutate_json(out/'TIFXYZ.json', change)
            mutate_json(marker, lambda d: d.update(tifxyz_sha256=file_sha(out/'TIFXYZ.json')))
        elif damage=='result-coverage':
            result['coverage_complete']=True
        elif damage=='strict-child':
            mutate_json(out/'EXPORT.json', lambda d: d['effective_settings'].update(coverage_policy='strict'))
            identity = json.loads((out/'EXPORT.json').read_bytes())
            mutate_json(out/'TIFXYZ.json', lambda d: d.update(identity_sha256=digest(identity)))
            mutate_json(marker, lambda d: d.update(identity_sha256=digest(identity), tifxyz_sha256=file_sha(out/'TIFXYZ.json')))
        elif damage=='wrong-binding':
            mutate_json(marker, lambda d: d.update(tifxyz_sha256='0'*64))
        elif damage=='identity-binding':
            mutate_json(marker, lambda d: d.update(identity_sha256='0'*64))
        else:
            verified = out/'level-0000/VERIFIED.json'
            def change(d):
                if damage=='verified-coverage': d['coverage_complete']=True
                elif damage=='geometry': d['audit']['geometry_complete']=False
                elif damage=='scan-complete': d['audit']['source_scan_complete']=True
                elif damage=='missing-region': d['audit']['reasons'][0].pop('region')
                elif damage=='remaining-none':d['audit']['remaining_source_coverage']='NONE'
                elif damage=='malformed-region':d['audit']['reasons'][0]['region']={'depth':1000000}
                elif damage=='audit-pass':d['audit']['status']='PASS'
                elif damage=='audit-physical':d['audit']['physical_acceptance']=True
                else: d['audit']['reasons'][0]['code']='RAM_HEADROOM'
            mutate_json(verified, change)
            mutate_json(out/'TIFXYZ.json', lambda d: d['levels'][0]['manifest']['files']['VERIFIED.json'].update(
                bytes=verified.stat().st_size, sha256=file_sha(verified)))
            mutate_json(marker, lambda d: d.update(tifxyz_sha256=file_sha(out/'TIFXYZ.json')))
        return result
    monkeypatch.setattr(tifxyz_runtime, 'supervise', damaged)
    job[0].append(partial_result())
    root, args = integrated
    assert call(*args)==1
    state = json.loads((root/'RUN.json').read_bytes())
    assert state['status']=='FAILED' and 'tifxyz' not in state['stages']
    with sqlite3.connect(root/'products.sqlite') as db:
        assert db.execute("SELECT count(*) FROM exports WHERE stage='tifxyz'").fetchone()[0]==0


def test_strict_parent_refuses_partial_receipt(integrated, job):
    root, args = integrated
    job[0].append(partial_result())
    assert call(*args)==1
    state = json.loads((root/'RUN.json').read_bytes())
    assert state['status']=='FAILED' and 'tifxyz' not in state['stages']


@pytest.mark.parametrize('status', ['PARTIAL', 'TIFXYZ_PARTIAL_VERIFIED'])
def test_partial_exit_codes(status):
    from winding_field.cli import tifxyz_exit_code
    assert tifxyz_exit_code(dict(status=status, coverage_complete=False))==2


@pytest.fixture
def publication(tmp_path):
    """Frozen worker protocol, independent of concurrent Settings implementation."""
    out = tmp_path/'publication'; out.mkdir()
    request_sha = 'a'*64
    identity = dict(effective_settings=dict(levels=[1., 2.], coverage_policy='allow_partial',
                    request_sha256=request_sha), reader=READER, **FLAGS)
    atomic_json(out/'EXPORT.json', identity)
    levels = []
    for index, status in enumerate(('READY', 'PARTIAL')):
        segment = out/f'level-{index:04d}'; segment.mkdir()
        complete = status=='READY'
        quality = dict(status=status, coverage_complete=complete)
        audit = dict(status='PASS' if complete else 'BLOCKED', geometry_complete=True,
                      blocked_phase=None if complete else 'coverage', source_scan_complete=complete,
                      remaining_source_coverage='NONE' if complete else 'UNMEASURED',
                      source_identity={'roi':{'lo_zyx':[3,3,3],'hi_zyx':[8,9,10]}},
                      reasons=[] if complete else [dict(code='COVERAGE_DEPTH_LIMIT', region=dict(
                          cell_zyx=[1, 2, 3], global_cell_zyx=[4, 5, 6],
                          subcell_index_zyx=[0, 1, 0], depth=1))],**FLAGS)
        atomic_json(segment/'VERIFIED.json', dict(audit=audit, reader=READER, **quality, **FLAGS))
        manifest = dict(files={'VERIFIED.json':dict(sha256=file_sha(segment/'VERIFIED.json'))})
        levels.append(dict(level=index+1., segment=segment.name, manifest=manifest, **quality))
    atomic_json(out/'TIFXYZ.json', dict(identity_sha256=digest(identity), reader=READER,
                levels=levels, **partial_result()))
    atomic_json(out/'PARTIAL.json', dict(identity_sha256=digest(identity),
                tifxyz_sha256=file_sha(out/'TIFXYZ.json'), **partial_result()))
    return out, request_sha


def test_parent_accepts_mixed_publication_without_upgrading_unknown_coverage(publication):
    from winding_field.workflow import _tifxyz_ready
    out, request_sha = publication
    before = tree_state(out)
    marker_sha = _tifxyz_ready(out, partial_result(), request_sha, 'allow_partial')
    assert marker_sha==file_sha(out/'PARTIAL.json')
    assert tree_state(out)==before
    audit = json.loads((out/'level-0001/VERIFIED.json').read_bytes())['audit']
    assert audit['source_scan_complete'] is False
    assert audit['reasons'][0]['region']['depth']==1
    # The first uncertified subcell is evidence, not an exhaustive map of missing coverage.
    assert 'uncovered_region_count' not in audit


@pytest.mark.parametrize('damage', ['both-markers', 'dangling-marker', 'missing-marker', 'strict-parent', 'root-complete',
                                  'result-complete', 'level-complete', 'verified-complete', 'bad-binding'])
def test_parent_protocol_rejects_forged_partial(publication, damage):
    from winding_field.workflow import _tifxyz_ready
    out, request_sha = publication
    result = partial_result()
    if damage=='both-markers': (out/'READY.json').write_bytes((out/'PARTIAL.json').read_bytes())
    elif damage=='dangling-marker': (out/'READY.json').symlink_to(out/'missing')
    elif damage=='missing-marker': (out/'PARTIAL.json').unlink()
    elif damage=='root-complete':
        mutate_json(out/'TIFXYZ.json', lambda d: d.update(coverage_complete=True))
        mutate_json(out/'PARTIAL.json', lambda d: d.update(tifxyz_sha256=file_sha(out/'TIFXYZ.json')))
    elif damage=='result-complete': result['coverage_complete']=True
    elif damage=='level-complete':
        mutate_json(out/'TIFXYZ.json', lambda d: d['levels'][1].update(status='READY', coverage_complete=True))
        mutate_json(out/'PARTIAL.json', lambda d: d.update(tifxyz_sha256=file_sha(out/'TIFXYZ.json')))
    elif damage=='verified-complete':
        mutate_json(out/'level-0001/VERIFIED.json', lambda d: d.update(status='READY', coverage_complete=True))
    elif damage=='bad-binding':
        mutate_json(out/'PARTIAL.json', lambda d: d.update(tifxyz_sha256='0'*64))
    with pytest.raises((ValueError, OSError)):
        _tifxyz_ready(out, result, request_sha, 'strict' if damage=='strict-parent' else 'allow_partial')
