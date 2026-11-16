"""Hand-derived metric controls, independent of solver energy and pixel count."""
import importlib
from pathlib import Path

import numpy as np
import pytest

from test_flattening_benchmark import source_mesh


@pytest.fixture
def metrics(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'tools'))
    assert importlib.util.find_spec('flattening_metrics'), 'metrics not implemented'
    return importlib.import_module('flattening_metrics')


def inputs(tmp_path, disconnected=False):
    import bench_flattening as b
    v = np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]])
    f = np.array([[0, 1, 2], [0, 2, 3]])
    if disconnected:
        v = np.r_[v[:3], v[:3] + [3., 0., 0.]]
        f = np.array([[0, 1, 2], [3, 4, 5]])
    src, _, _ = source_mesh(tmp_path / 'src', unit=1., geometry=(v, f))
    case = b.load_case(src, tmp_path / 'case')
    mapped = dict(world=v, faces=f, uv=v[:, :2].copy(), parent=np.arange(len(v)),
                  source_face_rows=np.arange(2), domain_ids=np.zeros(2, np.int64))
    return case, mapped


def test_identity_and_rigid_rotation_have_zero_distortion(metrics, tmp_path):
    case, mapped = inputs(tmp_path)
    mapped['uv'] = mapped['uv'] @ np.array([[0., -1.], [1., 0.]]) + [13., 29.]
    r = metrics.measure(case, mapped, tmp_path / 'result', lambda: None)
    assert r['valid_area_fraction'] == 1.
    assert r['scale_error_p95'] == pytest.approx(0., abs=1e-14)
    assert r['area_ratio_p95'] == pytest.approx(1.)


def test_known_stretch_has_expected_singular_values(metrics, tmp_path):
    case, mapped = inputs(tmp_path)
    mapped['uv'] *= [2., .5]
    r = metrics.measure(case, mapped, tmp_path / 'result', lambda: None)
    assert r['sigma_max_p95'] == pytest.approx(2.)
    assert r['sigma_min_p50'] == pytest.approx(.5)
    assert r['anisotropy_p95'] == pytest.approx(4.)
    assert r['area_ratio_p95'] == pytest.approx(1.)


def test_area_weighted_p95_is_not_face_count_p95(metrics):
    assert metrics.weighted_quantile(np.array([.1, 9.]), np.array([99., 1.]), .95) == .1
    assert metrics.weighted_quantile(np.array([9., .1]), np.array([1., 99.]), .95) == .1


def test_duplicate_seams_do_not_inflate_coverage(metrics, tmp_path):
    case, mapped = inputs(tmp_path)
    old = mapped['faces']
    mapped['world'] = mapped['world'][old].reshape(-1, 3)
    mapped['uv'] = mapped['uv'][old].reshape(-1, 2)
    mapped['parent'] = old.ravel()
    mapped['faces'] = np.arange(6).reshape(-1, 3)
    mapped['domain_ids'] = np.arange(2)
    r = metrics.measure(case, mapped, tmp_path / 'result', lambda: None)
    assert r['valid_area_fraction'] == 1.
    assert r['source_area_mm2'] == pytest.approx(1e-6)
    mapped['source_face_rows'] = np.array([0, 0])
    with pytest.raises(ValueError, match='ownership|correspondence'):
        metrics.measure(case, mapped, tmp_path / 'bad', lambda: None)


def test_overlap_or_incomplete_audit_cannot_win(metrics, tmp_path, monkeypatch):
    case, mapped = inputs(tmp_path, disconnected=True)
    mapped['uv'][3:] = mapped['uv'][:3]
    r = metrics.measure(case, mapped, tmp_path / 'result', lambda: None)
    assert r['valid_area_fraction'] == 0.
    assert r['overlap_area_fraction'] == 1.
    def incomplete(*args, report, **kwargs):
        report['complete'] = False
        return iter(())
    monkeypatch.setattr(metrics, 'iter_overlap_pairs', incomplete)
    with pytest.raises(ValueError, match='incomplete'):
        metrics.measure(case, mapped, tmp_path / 'bad', lambda: None)


def test_cuts_reduce_contiguity_not_source_area(metrics, tmp_path):
    case, mapped = inputs(tmp_path)
    mapped['world'] = mapped['world'][mapped['faces']].reshape(-1, 3)
    mapped['uv'] = mapped['uv'][mapped['faces']].reshape(-1, 2)
    mapped['uv'][3:] += [3., 0.]
    mapped['parent'] = mapped['faces'].ravel()
    mapped['faces'] = np.arange(6).reshape(-1, 3)
    mapped['domain_ids'] = np.arange(2)
    r = metrics.measure(case, mapped, tmp_path / 'result', lambda: None)
    assert r['valid_area_fraction'] == 1.
    assert r['largest_piece_area_fraction'] == .5
    assert r['artificial_seam_length_um'] == pytest.approx(np.sqrt(2.))


def test_omitting_faces_records_new_interior_cut_boundary(metrics, tmp_path):
    case, mapped = inputs(tmp_path)
    mapped['faces'] = mapped['faces'][:1]
    mapped['source_face_rows'] = np.array([0])
    mapped['domain_ids'] = np.array([0])
    report = metrics.measure(case, mapped, tmp_path / 'result', lambda: None)
    assert report['omitted_area_fraction'] == .5
    assert report['artificial_seam_length_um'] == pytest.approx(np.sqrt(2.))
