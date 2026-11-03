"""Controller contracts; numerical fixtures are used only without the optional math backend."""
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import types
import uuid
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from functools import partial

import numpy as np
import pytest

from ct_fixture import write_ct
from test_native_geometry import plane_bundle


@pytest.fixture
def tool():
    path = Path(__file__).resolve().parents[1]/'tools/bench_global_native.py'
    assert path.is_file(), 'experimental global command is not implemented'
    spec = importlib.util.spec_from_file_location('global_benchmark', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def inputs(tmp_path):
    mesh = plane_bundle(tmp_path, copies=2)
    z, y, x = np.indices((12, 12, 12))
    ct = write_ct(tmp_path/'ct.zarr', (x + 2*y + 3*z).astype(np.uint8))
    frame = tmp_path/'frame.json'
    frame.write_text(json.dumps(dict(id='test', origin_xyz_um=[0., 0., 0.],
        spacing_xyz_um=[1., 1., 1.], shape_zyx=[12, 12, 12], voxel_um=1.)))
    return mesh, ct, frame


def arguments(tool, inputs, out, *extra):
    mesh, ct, frame = inputs
    return tool.parser().parse_args(['--mesh', str(mesh), '--component', '1', '--ct', str(ct),
        '--frame', str(frame), '--output', str(out), '--pitch-um', '.5', '--seconds', '60',
        '--reserve-gib', '0', '--reserve-ram-gib', '0', '--export-tifxyz', *extra])


@pytest.fixture
def numerical_backend(monkeypatch):
    if importlib.util.find_spec('winding_field.native_global') is not None and importlib.util.find_spec('igl') is not None:
        return 'real-native-global'
    module = types.ModuleType('winding_field.native_global')
    module.cut_component = lambda v, f, check=None: (
        dict(vertices=v.copy(), faces=f.astype(np.int32), parent=np.arange(len(v), dtype=np.int64),
             boundary=np.array([], np.int64)), {'backend': 'analytic-plane-test-fixture'})
    module.mean_value_map = lambda cut, check=None: (cut['vertices'][:, :2].copy(), {'backend': 'analytic-plane-test-fixture'})
    module.arap_map = lambda cut, uv, check=None: (uv.copy(), {'backend': 'analytic-plane-test-fixture'})
    module.qualify_map = lambda cut, uv, check=None: (
        dict(valid_faces=np.ones(len(cut['faces']), bool), reasons=np.zeros(len(cut['faces']), np.uint8)),
        {'backend': 'analytic-plane-test-fixture'})
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return 'analytic-plane-test-fixture'


@pytest.fixture
def inline_workers(tool, numerical_backend, monkeypatch):
    # Real phase code, Journals, input hashes, CT and renderer; only process/resource admission is replaced.
    def launch(phase, out, request, budget, stop):
        if phase == 'verify':
            attempt = out/'attempts'/uuid.uuid4().hex; attempt.mkdir(parents=True)
            return tool.verification_receipt(out, request, attempt, budget.remaining, stop)
        tool.execute_phase(out, phase, request, budget.remaining, stop)
        return dict(status='COMMITTED')
    monkeypatch.setattr(tool, 'launch_phase', launch)
    return launch


def test_fresh_genuine_mesh_ct_roundtrip_and_explicit_whole_mesh_denominator(tool, inputs, tmp_path, inline_workers):
    from PIL import Image
    out = tmp_path/'output'
    request = tool.make_request(arguments(tool, inputs, out))
    result = tool.run(request, out)
    assert result['status'] == 'PARTIAL'
    assert result['source_faces'] == 48 and result['mapped_faces'] == 24
    assert result['source_area_um2'] == 24. and result['mapped_area_um2'] == 12.
    assert result['coverage_fraction'] == .5 and not result['target_met']
    assert result['physical_acceptance'] is result['field_coverage_certified'] is False
    assert result['usable_render'] and result['timing_scope'] == 'fresh-mesh-to-global-render'
    assert all((out/phase/'PHASE.json').is_file() for phase in tool.PHASES)
    source = tool.verify_phase(out, 'source', request)['report']
    assert source['selected_component'] == 1
    assert source['excluded_components'] == [{'component': 25, 'faces': 24, 'area_um2': 12.}]
    with np.load(out/'source/excluded-000000.npz', allow_pickle=False) as excluded:
        np.testing.assert_array_equal(excluded['face_ids'], np.arange(25, 49))
        assert np.all(excluded['reasons'] == 1)
    image = np.asarray(Image.open(out/'render'/result['components'][0]['path']/'sheet.png'))
    assert np.count_nonzero(image[..., 1]) > 20
    for path in (out/'render').glob('component-*/tiles/[0-9]*.npz'):
        with np.load(path, allow_pickle=False) as tile:
            valid = tile['mask'] == 2
            xyz = tile['xyz_um'][valid]
            np.testing.assert_allclose(tile['ct'][valid], xyz[:, 0] + 2*xyz[:, 1] + 3*xyz[:, 2], atol=2e-6)
    assert result['components'][0]['tifxyz']['status'] == 'WRITTEN'


def test_stop_resume_keeps_committed_phase_bytes_and_excludes_paused_wall_time(tool, inputs, tmp_path, inline_workers, monkeypatch):
    out = tmp_path/'output'; request = tool.make_request(arguments(tool, inputs, out))
    def stop_after_cut(phase, *args):
        value = inline_workers(phase, *args)
        if phase == 'cut':
            (out/'STOP').touch()
        return value
    monkeypatch.setattr(tool, 'launch_phase', stop_after_cut)
    result = tool.run(request, out)
    assert result['status'] == 'PAUSED' and result['reason'] == 'STOP'
    assert not (out/'mvc/PHASE.json').exists()
    before = {str(p): (p.read_bytes(), p.stat().st_mtime_ns) for phase in ('source', 'cut')
              for p in (out/phase).rglob('*') if p.is_file()}
    elapsed = json.loads((out/'TIME.json').read_bytes())['elapsed_s']
    (out/'STOP').unlink()
    monkeypatch.setattr(tool, 'launch_phase', inline_workers)
    result = tool.run(request, out, resume=True)
    assert result['usable_render'] and result['active_elapsed_s'] >= elapsed
    for name, expected in before.items():
        p = Path(name)
        assert (p.read_bytes(), p.stat().st_mtime_ns) == expected
    terminal = (out/'RESULT.json').read_bytes()
    tool.run(request, out, resume=True)
    assert (out/'RESULT.json').read_bytes() == terminal


def test_clock_charges_active_time_across_sessions_not_pause_gaps(tool, tmp_path, monkeypatch):
    now = [100.]
    monkeypatch.setattr(tool.time, 'monotonic', lambda: now[0])
    budget = tool.Budget(tmp_path, 5.)
    now[0] += 2.
    budget.close()
    now[0] += 10000.
    budget = tool.Budget(tmp_path, 5.)
    assert budget.elapsed == 2. and budget.remaining == 3.
    now[0] += 3.1
    assert budget.remaining == 0.
    budget.close()
    assert tool.Budget(tmp_path, 5.).remaining == 0.


def test_argument_source_read_cost_is_charged_before_first_phase(tool, inputs, tmp_path, inline_workers, monkeypatch):
    out = tmp_path/'output'; request = tool.make_request(arguments(tool, inputs, out))
    monkeypatch.setattr(tool, 'launch_phase', lambda *a: {'status': 'PAUSED', 'reason': 'STOP'})
    result = tool.run(request, out, preflight_s=14.)
    assert result['active_elapsed_s'] >= 14.


def test_final_verification_and_summary_cost_is_not_hidden_from_result_time(tool, inputs, tmp_path, inline_workers, monkeypatch):
    out = tmp_path/'output'; request = tool.make_request(arguments(tool, inputs, out)); now = [100.]
    monkeypatch.setattr(tool.time, 'monotonic', lambda: now[0])
    original = tool.attempt_summary
    def expensive_summary(path):
        value = original(path); now[0] += 14.; return value
    monkeypatch.setattr(tool, 'attempt_summary', expensive_summary)
    result = tool.run(request, out)
    assert result['active_elapsed_s'] >= 14.


def test_nonzero_mesh_level_is_bound_into_renderer_identity(tool, inputs, tmp_path):
    from test_native_mesh import Field
    from winding_field.surface_mesh import extract_mesh
    source = Field(); mesh = tmp_path/'level-two'
    extract_mesh(source, mesh, [2.], tile=8)
    frame = json.loads(inputs[2].read_bytes()); frame['id'] = 'analytic'; inputs[2].write_text(json.dumps(frame))
    request = tool.make_request(arguments(tool, (mesh, inputs[1], inputs[2]), tmp_path/'out'))
    assert request['level'] == 2.
    (tmp_path/'out').mkdir(); tool.check_sources(request, tmp_path/'out')
    assert tool.phase_identity(tmp_path/'out', 'source', request)['level'] == 2.


def test_changed_request_or_phase_payload_rejected_without_rewriting_products(tool, inputs, tmp_path, inline_workers):
    out = tmp_path/'output'; request = tool.make_request(arguments(tool, inputs, out))
    tool.run(request, out)
    before = (out/'RESULT.json').read_bytes()
    with pytest.raises(ValueError, match='request|identity'):
        tool.run(dict(request, pitch_um=1.), out, resume=True)
    path = out/'mvc/data.npz'; path.write_bytes(path.read_bytes() + b'corrupt')
    with pytest.raises(ValueError, match='changed|hash|payload'):
        tool.run(request, out, resume=True)
    assert (out/'RESULT.json').read_bytes() == before


def test_changed_mesh_or_ct_rejected_on_completed_resume(tool, inputs, tmp_path, inline_workers):
    out = tmp_path/'output'; request = tool.make_request(arguments(tool, inputs, out))
    tool.run(request, out)
    mesh, ct, _ = inputs
    payload = ct/'0/0.0.0'; payload.write_bytes(bytes([250])*payload.stat().st_size)
    with pytest.raises(ValueError, match='CT|chunk|changed'):
        tool.run(request, out, resume=True)
    manifest = mesh/'MESH.json'; manifest.write_bytes(manifest.read_bytes() + b' ')
    with pytest.raises(ValueError, match='mesh|source|changed|request'):
        tool.run(request, out, resume=True)


@pytest.mark.parametrize('bad', ['mesh-child', 'mesh-parent', 'ct-child', 'frame-parent', 'symlink'])
def test_output_never_contains_or_sits_in_a_source_tree(tool, inputs, tmp_path, bad):
    mesh, ct, frame = inputs
    paths = {'mesh-child': mesh/'new', 'mesh-parent': tmp_path, 'ct-child': ct/'new', 'frame-parent': frame.parent}
    if bad == 'symlink':
        link = tmp_path/'link'; link.symlink_to(mesh, target_is_directory=True); out = link/'new'
    else:
        out = paths[bad]
    with pytest.raises(ValueError, match='source|overlap|symlink|contain'):
        tool.make_request(arguments(tool, inputs, out))


def test_component_and_frame_are_explicit_and_physical_units_are_checked(tool, inputs, tmp_path):
    args = arguments(tool, inputs, tmp_path/'out')
    with pytest.raises(SystemExit):
        tool.parser().parse_args(['--mesh', str(inputs[0]), '--output', str(tmp_path/'out')])
    frame = json.loads(inputs[2].read_bytes()); frame['voxel_um'] = float('nan')
    inputs[2].write_text(json.dumps(frame))
    with pytest.raises(ValueError, match='finite|frame|JSON'):
        tool.make_request(args)


def test_npz_header_envelope_is_checked_before_numpy_allocation(tool, tmp_path, monkeypatch):
    import io
    import zipfile
    path = tmp_path/'bomb.npz'; header = io.BytesIO()
    np.lib.format.write_array_header_1_0(header, dict(descr='<f8', fortran_order=False, shape=(10**12, 3)))
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('vertices.npy', header.getvalue())
    def forbidden(*args, **kwargs):
        raise AssertionError('np.load reached before envelope rejection')
    monkeypatch.setattr(np, 'load', forbidden)
    with pytest.raises(ValueError, match='budget|envelope|header|byte'):
        tool.load_unit(path)


def test_global_uv_bits_cut_ids_and_original_ancestry_survive_internal_chunks(tool, tmp_path, monkeypatch):
    from winding_field.native_store import Journal
    source = dict(vertices=np.array([[2., 3., 4.], [3., 3., 4.], [3., 4., 4.], [2., 4., 4.]]),
                  faces=np.array([[0, 1, 2], [0, 2, 3]], np.int32), face_ids=np.array([81, 99], np.int64),
                  vertex_ids=np.array([11, 23, 37, 45], np.int64), areas=np.array([.5, .5]))
    parent = np.array([0, 1, 2, 3, 0, 2], np.int64)
    cut = dict(vertices=source['vertices'][parent], faces=np.array([[0, 1, 2], [4, 5, 3]], np.int32), parent=parent)
    uv = np.array([[0., 0.], [1., 0.], [1., 1.], [4., 1.], [4., 0.], [5., 1.]])
    monkeypatch.setattr(tool, 'CHUNK_FACES', 1)
    with Journal(tmp_path/'render', {'kind': 'test'}) as store:
        tool.register_global(store, source, cut, uv, np.ones(2, bool), 7, 4, 2.)
        rows = store.db.execute('SELECT name,uv_path FROM charts ORDER BY name').fetchall()
        assert len(rows) == 2
        for name, uv_path in rows:
            chart = store.load_arrays('charts/'+name+'.npz'); saved = store.load_arrays(uv_path)['uv']
            np.testing.assert_array_equal(saved.view(np.uint64), uv[chart['vertex_ids']].view(np.uint64))
            np.testing.assert_array_equal(chart['source_vertex_ids'], source['vertex_ids'][parent[chart['vertex_ids']]])
            np.testing.assert_array_equal(chart['vertices'], cut['vertices'][chart['vertex_ids']])
        assert store.db.execute('SELECT count(*) FROM edges e JOIN edges p ON e.a=p.a AND e.b=p.b AND e.chart!=p.chart').fetchone()[0] == 0
        assert store.get('source_faces') == 4 and store.get('source_area') == 2.
    shared = dict(vertices=source['vertices'], faces=source['faces'], parent=np.arange(4, dtype=np.int64))
    with Journal(tmp_path/'shared-edge', {'kind': 'test'}) as store:
        tool.register_global(store, source, shared, source['vertices'][:, :2], np.ones(2, bool), 7, 4, 2.)
        assert store.db.execute('SELECT count(*) FROM edges e JOIN edges p ON e.a=p.a AND e.b=p.b AND e.chart!=p.chart').fetchone()[0] == 2


def test_data_ceiling_uses_live_headroom_and_parent_margin_not_address_space(tool):
    assert tool.data_ceiling(4 << 30, 2 << 30, 128 << 20) == (2 << 30) - (64 << 20)
    assert tool.data_ceiling(20 << 30, 2 << 30, 128 << 20) == (3 << 30) - (192 << 20)
    with pytest.raises(ValueError, match='headroom|DATA|memory'):
        tool.data_ceiling(2 << 30, 2 << 30, 128 << 20)


def test_phase_report_cannot_disagree_with_its_hash_bound_payload(tool, inputs, tmp_path, inline_workers, monkeypatch):
    out = tmp_path/'output'; request = tool.make_request(arguments(tool, inputs, out))
    def one_phase(phase, *args):
        value = inline_workers(phase, *args); (out/'STOP').touch(); return value
    monkeypatch.setattr(tool, 'launch_phase', one_phase)
    assert tool.run(request, out)['status'] == 'PAUSED'
    marker = out/'source/PHASE.json'; record = json.loads(marker.read_bytes())
    record['report']['source_area_um2'] = .0001
    marker.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='report|changed'):
        tool.verify_phase(out, 'source', request)


def test_resume_rejects_missing_clock_and_owned_symlink_before_reading_it(tool, inputs, tmp_path, inline_workers):
    out = tmp_path/'output'; request = tool.make_request(arguments(tool, inputs, out))
    tool.run(request, out)
    clock = (out/'TIME.json').read_bytes(); (out/'TIME.json').unlink()
    with pytest.raises(ValueError, match='time|clock'):
        tool.run(request, out, resume=True)
    (out/'TIME.json').write_bytes(clock)
    (out/'source/PHASE.json').unlink()
    (out/'source/PHASE.json').symlink_to(inputs[2])
    with pytest.raises(ValueError, match='symlink'):
        tool.run(request, out, resume=True)


def test_resume_source_validation_time_is_in_the_active_budget(tool, inputs, tmp_path, inline_workers, monkeypatch):
    out = tmp_path/'output'; request = tool.make_request(arguments(tool, inputs, out))
    def source_then_stop(phase, *args):
        result = inline_workers(phase, *args); (out/'STOP').touch(); return result
    monkeypatch.setattr(tool, 'launch_phase', source_then_stop)
    tool.run(request, out)
    (out/'STOP').unlink(); monkeypatch.setattr(tool, 'launch_phase', inline_workers)
    before = json.loads((out/'TIME.json').read_bytes())['elapsed_s']
    original = tool.check_sources; clock = [100.]; delayed = []
    monkeypatch.setattr(tool.time, 'monotonic', lambda: clock[0])
    def slow_validation(*args):
        result = original(*args)
        if not delayed:
            clock[0] += 14.; delayed.append(True)
        return result
    monkeypatch.setattr(tool, 'check_sources', slow_validation)
    tool.run(request, out, resume=True)
    assert json.loads((out/'TIME.json').read_bytes())['elapsed_s'] >= before + 14.


def test_exhausted_budget_does_not_launch_or_reset_on_resume(tool, inputs, tmp_path, inline_workers, monkeypatch):
    out = tmp_path/'output'; request = tool.make_request(arguments(tool, inputs, out))
    monkeypatch.setattr(tool, 'launch_phase', lambda *a: {'status': 'PAUSED', 'reason': 'STOP'})
    tool.run(request, out)
    (out/'TIME.json').write_text(json.dumps(dict(elapsed_s=60.)))
    def forbidden(*args):
        raise AssertionError('exhausted run performed source reads or launched work')
    monkeypatch.setattr(tool, 'check_sources', forbidden)
    monkeypatch.setattr(tool, 'launch_phase', forbidden)
    result = tool.run(request, out, resume=True)
    assert result['status'] == 'PAUSED' and result['reason'] == 'TIME_LIMIT'
    assert result['active_elapsed_s'] >= 60.


def test_lexical_dotdot_cannot_evade_source_containment(tool, inputs, tmp_path):
    (tmp_path/'decoy').mkdir()
    out = tmp_path/'decoy/../ct.zarr/new'
    with pytest.raises(ValueError, match='contain|overlap'):
        tool.make_request(arguments(tool, inputs, out))


def test_selected_qualification_exclusions_preserve_original_face_ids(tool, inputs, tmp_path, inline_workers, monkeypatch):
    api = importlib.import_module('winding_field.native_global'); original = api.qualify_map
    def exclude(cut, uv, check=None):
        arrays, report = original(cut, uv, check=check)
        arrays['valid_faces'][0] = False; arrays['reasons'][0] |= 8
        return arrays, report
    monkeypatch.setattr(api, 'qualify_map', exclude)
    out = tmp_path/'output'; request = tool.make_request(arguments(tool, inputs, out))
    result = tool.run(request, out)
    assert result['source_faces'] == 48 and result['mapped_faces'] == 23 and result['status'] == 'PARTIAL'
    with np.load(out/'qualification/excluded.npz', allow_pickle=False) as arrays:
        assert arrays['face_ids'][0] == 1 and arrays['reasons'][0] & 8


def test_reopened_render_keeps_global_uv_units_after_partial_registration(tool, inputs, tmp_path, inline_workers, monkeypatch):
    out = tmp_path/'output'; request = tool.make_request(arguments(tool, inputs, out))
    monkeypatch.setattr(tool, 'CHUNK_FACES', 8)
    original = tool.register_global
    def interrupted(store, *args):
        store.stop = lambda: store.get('global_face_cursor', 0) >= 8
        original(store, *args)
    monkeypatch.setattr(tool, 'register_global', interrupted)
    result = tool.run(request, out)
    assert result['status'] == 'PAUSED'
    before = {str(p): p.read_bytes() for name in ('charts', 'uv') for p in (out/'render'/name).glob('*.npz')}
    assert before and not (out/'render/PHASE.json').exists()
    monkeypatch.setattr(tool, 'register_global', original)
    result = tool.run(request, out, resume=True)
    assert result['usable_render'] and result['mapped_faces'] == 24
    assert all(Path(name).read_bytes() == payload for name, payload in before.items())


@pytest.mark.parametrize('phase,payload', [('source', 'REPORT.json'), ('render', 'LEVEL.json')])
def test_kill_between_payload_commit_and_phase_seal_never_rewrites_payload(
        tool, inputs, tmp_path, inline_workers, monkeypatch, phase, payload):
    from winding_field.native_store import Journal
    class Interrupted(RuntimeError):
        pass
    original = Journal.register
    def crash_after_commit(store, name):
        original(store, name)
        if store.root.name == phase and name == payload:
            raise Interrupted('committed payload, interrupted before phase seal')
    monkeypatch.setattr(Journal, 'register', crash_after_commit)
    out = tmp_path/'output'; request = tool.make_request(arguments(tool, inputs, out))
    with pytest.raises(Interrupted):
        tool.run(request, out)
    path = out/phase/payload; before = (path.read_bytes(), path.stat().st_mtime_ns)
    assert not (out/phase/'PHASE.json').exists()
    monkeypatch.setattr(Journal, 'register', original)
    result = tool.run(request, out, resume=True)
    assert result['usable_render']
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before


def test_non_um_mesh_and_absent_component_reject_without_implicit_selection(tool, inputs, tmp_path):
    from winding_field.surface_mesh import import_mesh
    from winding_field.util import file_sha
    mesh, ct, frame = inputs
    out = tmp_path/'absent'; request = tool.make_request(arguments(tool, inputs, out, '--component', '2'))
    out.mkdir()
    with pytest.raises(ValueError, match='component'):
        tool.execute_phase(out, 'source', request, 30)
    wrong = tmp_path/'wrong-units'
    metadata = json.loads((mesh/'MESH.json').read_bytes())['metadata']; metadata['voxel_um'] = 8.64
    source = tmp_path/'source.npz'
    import_mesh(dict(path=str(source), sha256=file_sha(source)), wrong, metadata)
    out = tmp_path/'bad-units'; request = tool.make_request(arguments(tool, (wrong, ct, frame), out)); out.mkdir()
    with pytest.raises(ValueError, match='voxel_um=1'):
        tool.execute_phase(out, 'source', request, 30)


def test_bad_cut_ancestry_cannot_be_committed(tool, inputs, tmp_path, inline_workers, monkeypatch):
    api = importlib.import_module('winding_field.native_global'); original = api.cut_component
    def bad_cut(v, f, check=None):
        arrays, report = original(v, f, check=check)
        arrays['vertices'][0, 0] += .01
        return arrays, report
    monkeypatch.setattr(api, 'cut_component', bad_cut)
    out = tmp_path/'output'; request = tool.make_request(arguments(tool, inputs, out))
    with pytest.raises(ValueError, match='ancestry|XYZ'):
        tool.run(request, out)
    assert (out/'source/PHASE.json').is_file() and not (out/'cut/PHASE.json').exists()


def test_math_cannot_silently_modify_a_committed_input_in_place(tool, inputs, tmp_path, inline_workers, monkeypatch):
    api = importlib.import_module('winding_field.native_global'); original = api.cut_component
    def mutated_input(v, f, check=None):
        v[0, 0] += .01
        return original(v, f, check=check)
    monkeypatch.setattr(api, 'cut_component', mutated_input)
    out = tmp_path/'output'; request = tool.make_request(arguments(tool, inputs, out))
    with pytest.raises(ValueError, match='read-only|XYZ|immutable'):
        tool.run(request, out)
    assert not (out/'cut/PHASE.json').exists()


def test_real_supervisor_kill_preserves_phase_and_charges_clock(tool, inputs, tmp_path):
    from winding_field.native_pipeline import _supervise
    out = tmp_path/'output'; out.mkdir()
    request = tool.make_request(arguments(tool, inputs, out))
    tool.execute_phase(out, 'source', request, 30)
    before = (out/'source/data.npz').read_bytes()
    attempt = out/'killed'; attempt.mkdir()
    command = [sys.executable, '-B', '-c',
               'import signal,time; from pathlib import Path; '
               'signal.signal(signal.SIGTERM,signal.SIG_IGN); '
               'Path(__import__("sys").argv[1]).touch(); time.sleep(30)', str(attempt/'started')]
    budget = tool.Budget(out, 30.)
    result = _supervise(command, attempt, .3)
    budget.close()
    assert (attempt/'started').exists() and result['reason'] == 'TIME_LIMIT'
    assert (out/'source/data.npz').read_bytes() == before
    assert tool.verify_phase(out, 'source', request)
    assert json.loads((out/'TIME.json').read_bytes())['elapsed_s'] >= 2.
    with pytest.raises(ProcessLookupError):
        os.kill(result['supervision']['pid'], 0)


@pytest.mark.skipif(sys.platform != 'linux' or importlib.util.find_spec('igl') is None,
                    reason='real numerical subprocess control requires existing libigl/Linux')
def test_real_linux_command_runs_fresh_math_without_saved_uvs(tool, inputs, tmp_path):
    args = arguments(tool, inputs, tmp_path/'cli')
    command = [sys.executable, '-B', tool.__file__, '--mesh', str(args.mesh), '--component', '1',
               '--ct', args.ct, '--frame', str(args.frame), '--output', str(args.output), '--pitch-um', '.5',
               '--seconds', '60', '--reserve-gib', '0', '--reserve-ram-gib', '0', '--export-tifxyz']
    process = subprocess.run(command, capture_output=True, text=True, timeout=75)
    assert process.returncode in (0, 2), process.stdout + process.stderr
    result = json.loads((args.output/'RESULT.json').read_bytes())
    assert result['source_faces'] == 48 and result['usable_render']
    assert all((args.output/p/'PHASE.json').is_file() for p in tool.PHASES)


@pytest.mark.parametrize('member', ['mesh.sqlite', 'RUN.json'])
def test_source_mutation_after_reader_validate_cannot_be_checkpointed(tool, inputs, tmp_path, monkeypatch, member):
    from winding_field.surface_mesh import MeshReader
    mesh = inputs[0]; out = tmp_path/'output'; out.mkdir()
    request = tool.make_request(arguments(tool, inputs, out))
    original = MeshReader.validate; calls = []; changed = []
    def mutate_after_validation(reader):
        original(reader)
        calls.append(True)
        if len(calls) == 2:
            if member == 'mesh.sqlite':
                with sqlite3.connect(mesh/member) as db:
                    db.execute('UPDATE vertices SET x=x+1 WHERE id=1')
            else:
                run = json.loads((mesh/member).read_bytes()); run['identity']['changed_after_validate'] = True
                (mesh/member).write_text(json.dumps(run))
            changed.append(True)
    monkeypatch.setattr(MeshReader, 'validate', mutate_after_validation)
    with pytest.raises(ValueError, match='source|changed|digest|binding'):
        tool.execute_phase(out, 'source', request, 30.)
    assert changed and not (out/'source/PHASE.json').exists()
    assert not (out/'source/REPORT.json').exists()


def test_frame_replacement_after_parse_does_not_bind_replacement_bytes(tool, inputs, tmp_path, monkeypatch):
    frame = inputs[2]; before = frame.read_bytes(); real_loads = json.loads; changed = []
    def replace_after_parse(raw, *args, **kwargs):
        result = real_loads(raw, *args, **kwargs)
        if raw == before and not changed:
            replacement = dict(result, origin_xyz_um=[100., 0., 0.])
            path = frame.with_name('replacement.json'); path.write_text(json.dumps(replacement)); path.replace(frame)
            changed.append(True)
        return result
    monkeypatch.setattr(tool.json, 'loads', replace_after_parse)
    with pytest.raises(ValueError, match='changed|capture'):
        tool.make_request(arguments(tool, inputs, tmp_path/'output'))
    assert changed


def test_manifest_envelope_precedes_any_unbounded_file_hash(tool, inputs, tmp_path, monkeypatch):
    manifest = inputs[0]/'MESH.json'; manifest.write_bytes(b' '*(tool.JSON_BYTES + 1))
    original = tool.file_sha
    def forbidden_unbounded_hash(path):
        if Path(path) == manifest:
            raise AssertionError('unbounded manifest hash before bounded read')
        return original(path)
    monkeypatch.setattr(tool, 'file_sha', forbidden_unbounded_hash)
    with pytest.raises(ValueError, match='envelope|budget|size'):
        tool.make_request(arguments(tool, inputs, tmp_path/'output'))


@pytest.mark.parametrize('kind', ['fifo', 'symlink'])
def test_frame_nonregular_input_rejects_without_blocking(tool, inputs, tmp_path, kind):
    frame = inputs[2]; original = frame.read_bytes(); frame.unlink()
    if kind == 'fifo':
        os.mkfifo(frame)
    else:
        target = tmp_path/'real-frame.json'; target.write_bytes(original); frame.symlink_to(target)
    code = ('import runpy,sys; m=runpy.run_path(sys.argv[1]); '
            'a=m["parser"]().parse_args(sys.argv[2:]); '
            '\ntry: m["make_request"](a)\nexcept ValueError: print("REJECTED"); sys.exit(0)\nsys.exit(1)')
    args = ['--mesh', str(inputs[0]), '--component', '1', '--ct', str(inputs[1]), '--frame', str(frame),
            '--output', str(tmp_path/'output'), '--pitch-um', '.5']
    result = subprocess.run([sys.executable, '-B', '-c', code, tool.__file__, *args],
                            capture_output=True, text=True, timeout=3)
    assert result.returncode == 0 and 'REJECTED' in result.stdout, result.stdout + result.stderr


@pytest.mark.parametrize('field,value', [
    ('status', 'COMPLETE'), ('source_faces', 24), ('source_area_um2', 12.), ('coverage_fraction', 1.),
    ('physical_acceptance', True), ('field_coverage_certified', True), ('active_elapsed_s', 0.),
    ('phase_elapsed_s', {'source': 0.}), ('execution', {'phase_worker_seconds': {'source': 0.}}),
])
def test_completed_resume_rejects_result_claims_and_timing_tampering(tool, inputs, tmp_path, inline_workers, field, value):
    out = tmp_path/'output'; request = tool.make_request(arguments(tool, inputs, out))
    original = tool.run(request, out)
    assert original['source_faces'] == 48 and original['mapped_faces'] == 24 and original['coverage_fraction'] == .5
    path = out/'RESULT.json'; changed = dict(json.loads(path.read_bytes()), **{field: value})
    path.write_text(json.dumps(changed)); tampered = path.read_bytes()
    with pytest.raises(ValueError, match='publication|result|timing|execution'):
        tool.run(request, out, resume=True)
    assert path.read_bytes() == tampered


@pytest.mark.parametrize('changes', [
    dict(status='COMPLETE', physical_acceptance=True, source_faces=24, coverage_fraction=1.),
    dict(physical_acceptance=0), dict(source_faces=48.),
])
def test_rehashing_result_seal_cannot_promote_authoritative_partial_to_complete(tool, inputs, tmp_path, inline_workers, changes):
    out = tmp_path/'output'; request = tool.make_request(arguments(tool, inputs, out)); tool.run(request, out)
    path = out/'RESULT.json'; changed = json.loads(path.read_bytes())
    changed.update(changes)
    path.write_text(json.dumps(changed))
    seal = out/'PUBLICATION.json'
    if seal.exists():
        binding = json.loads(seal.read_bytes()); binding['result_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        seal.write_text(json.dumps(binding))
    with pytest.raises(ValueError, match='publication|result|render'):
        tool.run(request, out, resume=True)


def test_completed_verification_has_separate_timing_and_keeps_publication_bytes(tool, inputs, tmp_path, inline_workers):
    out = tmp_path/'output'; request = tool.make_request(arguments(tool, inputs, out)); initial = tool.run(request, out)
    before = ((out/'RESULT.json').read_bytes(), (out/'RESULT.json').stat().st_mtime_ns)
    resumed = tool.run(request, out, resume=True)
    assert resumed['active_elapsed_s'] == initial['active_elapsed_s']
    assert resumed['verification_elapsed_s'] > 0
    assert resumed['cumulative_active_elapsed_s'] > initial['active_elapsed_s']
    assert ((out/'RESULT.json').read_bytes(), (out/'RESULT.json').stat().st_mtime_ns) == before


@pytest.mark.parametrize('corrupt', [False, True])
def test_result_before_publication_seal_recovers_only_authoritative_claims(tool, inputs, tmp_path, inline_workers, monkeypatch, corrupt):
    out = tmp_path/'output'; request = tool.make_request(arguments(tool, inputs, out)); original = tool.atomic_json
    def crash_before_seal(path, value):
        if Path(path) == out/'PUBLICATION.json':
            raise RuntimeError('publication seal interruption')
        return original(path, value)
    monkeypatch.setattr(tool, 'atomic_json', crash_before_seal)
    with pytest.raises(RuntimeError, match='seal interruption'):
        tool.run(request, out)
    assert (out/'RESULT.json').is_file() and not (out/'PUBLICATION.json').exists()
    if corrupt:
        value = json.loads((out/'RESULT.json').read_bytes()); value['physical_acceptance'] = True
        (out/'RESULT.json').write_text(json.dumps(value))
    before = ((out/'RESULT.json').read_bytes(), (out/'RESULT.json').stat().st_mtime_ns)
    monkeypatch.setattr(tool, 'atomic_json', original)
    if corrupt:
        with pytest.raises(ValueError, match='result|publication'):
            tool.run(request, out, resume=True)
        assert not (out/'PUBLICATION.json').exists()
    else:
        result = tool.run(request, out, resume=True)
        assert result['status'] == 'PARTIAL' and (out/'PUBLICATION.json').exists()
    assert ((out/'RESULT.json').read_bytes(), (out/'RESULT.json').stat().st_mtime_ns) == before


@pytest.mark.parametrize('stop,exhausted', [(True, False), (False, True), (True, True)])
def test_completed_stop_or_expired_budget_reads_no_source_payload_or_ct(tool, inputs, tmp_path, inline_workers, monkeypatch, stop, exhausted):
    out = tmp_path/'output'; request = tool.make_request(arguments(tool, inputs, out)); tool.run(request, out)
    if stop:
        (out/'STOP').touch()
    if exhausted:
        (out/'TIME.json').write_text(json.dumps(dict(elapsed_s=60.)))
    def forbidden(*args, **kwargs):
        raise AssertionError('STOP/expired completed resume performed heavyweight verification')
    for name in ('check_sources', 'verify_phase', 'open_volume', 'launch_phase'):
        monkeypatch.setattr(tool, name, forbidden)
    result = tool.run(request, out, resume=True)
    assert result['status'] == 'PAUSED' and result['reason'] in ('STOP', 'TIME_LIMIT')


def test_completed_controller_delegates_heavy_verification_to_worker(tool, inputs, tmp_path, inline_workers, monkeypatch):
    out = tmp_path/'output'; request = tool.make_request(arguments(tool, inputs, out)); tool.run(request, out)
    def forbidden(*args, **kwargs):
        raise AssertionError('controller performed heavyweight source/artifact/CT verification')
    for name in ('check_sources', 'verify_phase', 'open_volume', 'attempt_summary'):
        monkeypatch.setattr(tool, name, forbidden)
    calls = []
    def worker_action(phase, *args):
        calls.append(phase); return dict(status='PAUSED', reason='TEST_WORKER_BOUNDARY')
    monkeypatch.setattr(tool, 'launch_phase', worker_action)
    result = tool.run(request, out, resume=True)
    assert calls == ['verify'] and result['reason'] == 'TEST_WORKER_BOUNDARY'


def test_cached_phase_still_checks_current_source_before_returning(tool, inputs, tmp_path):
    out = tmp_path/'output'; out.mkdir(); request = tool.make_request(arguments(tool, inputs, out))
    tool.execute_phase(out, 'source', request, 30.)
    before = (out/'source/PHASE.json').read_bytes()
    with sqlite3.connect(inputs[0]/'mesh.sqlite') as db:
        db.execute('UPDATE vertices SET x=x+1 WHERE id=1')
    with pytest.raises(ValueError, match='source|changed|digest'):
        tool.execute_phase(out, 'source', request, 30.)
    assert (out/'source/PHASE.json').read_bytes() == before


def test_blocked_ct_verification_is_killed_without_mutating_publication(tool, inputs, tmp_path, inline_workers):
    from winding_field.native_pipeline import _supervise
    block = threading.Event(); entered = threading.Event(); release = threading.Event()
    class Handler(SimpleHTTPRequestHandler):
        def do_GET(self):
            if block.is_set() and self.path.endswith('/.zattrs'):
                entered.set(); release.wait(20)
            try:
                super().do_GET()
            except (BrokenPipeError, ConnectionResetError):
                pass
        def log_message(self, *unused):
            pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), partial(Handler, directory=str(inputs[1])))
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    try:
        out = tmp_path/'output'; args = arguments(tool, inputs, out)
        args.ct = f'http://127.0.0.1:{server.server_port}'
        request = tool.make_request(args); tool.run(request, out)
        before = {str(p): (p.read_bytes(), p.stat().st_mtime_ns) for phase in tool.PHASES
                  for p in (out/phase).rglob('*') if p.is_file()}
        before.update({str(out/name): ((out/name).read_bytes(), (out/name).stat().st_mtime_ns)
                       for name in tool.PUBLICATION_FILES})
        block.set(); attempt = out/'blocked-verification'; attempt.mkdir()
        code = ('import runpy,sys,json,signal; from pathlib import Path; '
                'signal.signal(signal.SIGTERM,lambda *a:None); m=runpy.run_path(sys.argv[1]); out=Path(sys.argv[2]); '
                'm["execute_verification"](out,json.loads((out/"REQUEST.json").read_bytes()),30.)')
        budget = tool.Budget(out, 60.)
        try:
            result = _supervise([sys.executable, '-B', '-c', code, tool.__file__, str(out)],
                                attempt, 10., entered.is_set)
        finally:
            budget.close()
        assert entered.is_set(), (attempt/'worker.log').read_text()
        assert result['status'] == 'PAUSED' and result['reason'] == 'STOP'
        with pytest.raises(ProcessLookupError):
            os.kill(result['supervision']['pid'], 0)
        assert all((Path(name).read_bytes(), Path(name).stat().st_mtime_ns) == value for name, value in before.items())
    finally:
        release.set(); server.shutdown(); server.server_close(); thread.join(timeout=2)
