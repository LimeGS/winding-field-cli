"""Geometric controls over the real trilinear PhysicalField and disk SQLite."""
from contextlib import contextmanager
import sqlite3
from types import SimpleNamespace

import numpy as np
import pytest

from winding_field.tifxyz_contract import Blocked, Paused, load_settings
from winding_field.tifxyz_field import PhysicalField


READER = dict(cell_rule='four-corners', diagonals=['nw-se', 'ne-sw'], inpainting=False)


class PagedCursor:
    def __init__(self, cursor): self.cursor = cursor
    def fetchall(self): raise AssertionError('unbounded SQL fetch')
    def fetchmany(self, size):
        assert 0 < size <= 4096
        return self.cursor.fetchmany(size)
    def fetchone(self): return self.cursor.fetchone()
    def close(self): self.cursor.close()


class PagedDB:
    def __init__(self, path): self.raw = sqlite3.connect(path)
    def execute(self, sql, args=()):
        assert not any(word in sql.upper() for word in ('ORDER BY', 'GROUP BY', 'DISTINCT', 'TEMP TABLE'))
        return PagedCursor(self.raw.execute(sql, args))
    def executemany(self, sql, rows):
        rows = list(rows)
        assert len(rows) <= 128
        return PagedCursor(self.raw.executemany(sql, rows))


class Store:
    """Protocol fixture: immutable committed tiles, owned-pixel region reads."""
    def __init__(self, root, grid, xyz=None, mask=None, errors=None):
        root.mkdir()
        self.root, self.grid = root, grid
        self.identity = dict(grid=grid)
        self.budget = SimpleNamespace(scratch_bytes=1 << 30, database_bytes=64 << 20,
                                      reserve_disk_bytes=0, scope_root=root)
        self.db = PagedDB(root / 'audit.sqlite')
        self.tiles = {}
        self.fail_guard = False
        if xyz is not None:
            mask = np.ones(xyz.shape[:2], bool) if mask is None else mask
            errors = np.zeros(mask.shape) if errors is None else errors
            for ty in range((grid['shape_yx'][0]+255)//256):
                for tx in range((grid['shape_yx'][1]+255)//256):
                    lo = np.array([ty, tx])*256
                    hi = np.minimum(lo+256, grid['shape_yx'])
                    low, high = np.maximum(lo-1, 0), np.minimum(hi+1, grid['shape_yx'])
                    sl = tuple(slice(a, b) for a, b in zip(low, high))
                    points = xyz[sl].astype(np.float32).copy()
                    points[~mask[sl]] = -1
                    error = errors[sl].copy(); error[~mask[sl]] = 0
                    self.commit(dict(tile_id=[ty, tx], owned_box_yx=[lo.tolist(), hi.tolist()],
                        box_yx=[low.tolist(), high.tolist()], xyz=points, mask=mask[sl].copy(), root_error_um=error))

    def commit(self, tile):
        key = tuple(tile['tile_id'])
        assert key not in self.tiles
        self.tiles[key] = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in tile.items()}
        return {}

    def read(self, tile_id):
        return {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in self.tiles[tuple(tile_id)].items()}

    def iter_ids(self): yield from sorted(self.tiles)

    def read_region(self, lo, hi):
        lo, hi = np.array(lo), np.array(hi)
        assert ((hi-lo) <= 260).all() and (hi > lo).all()
        shape = tuple(hi-lo)
        result = dict(xyz=np.empty((*shape, 3), np.float32), mask=np.empty(shape, bool),
                      root_error_um=np.empty(shape, float))
        for ty in range(lo[0]//256, (hi[0]-1)//256+1):
            for tx in range(lo[1]//256, (hi[1]-1)//256+1):
                tile = self.tiles[(int(ty), int(tx))]
                a = np.maximum(lo, tile['owned_box_yx'][0]); b = np.minimum(hi, tile['owned_box_yx'][1])
                src = tuple(slice(x-o, y-o) for x, y, o in zip(a, b, tile['box_yx'][0]))
                dst = tuple(slice(x-o, y-o) for x, y, o in zip(a, b, lo))
                for key in result: result[key][dst] = tile[key][src]
        return result

    def guard(self, pending_bytes=0, sql=False):
        if self.fail_guard: raise Blocked('DATABASE_LIMIT')

    @contextmanager
    def transaction(self):
        self.guard(sql=True)
        with self.db.raw: yield self.db


def field(values, support=None, spacing=(1., 1., 1.), origin=(0., 0., 0.)):
    values = np.asarray(values, float)
    support = np.ones(values.shape, bool) if support is None else np.asarray(support, bool)
    def read(lo, hi):
        sl = tuple(slice(a, b) for a, b in zip(lo, hi))
        return values[sl].copy(), support[sl].copy()
    source = SimpleNamespace(shape=values.shape, origin=np.array(origin), spacing=np.array(spacing),
        voxel_um=1., identity={'fixture': 'audit'}, validate=lambda **kw: None, read=read, close=lambda: None)
    return PhysicalField(source)


def settings(f, tau=1., frame=None):
    frame = frame or dict(id='fixture', origin_xyz_um=[0, 0, 0], voxel_um=1.,
                         shape_zyx=[4096]*3, source_identity_sha256='a'*64)
    return load_settings(dict(levels=[0.], pitch_um=1., tolerance_um=tau),
        dict(shape_zyx=list(f.shape), spacing_xyz=f.spacing_um.tolist(), voxel_um=1.), frame)


def grid(shape):
    return dict(shape_yx=list(shape), uv_min_um=[0., 0.], pitch_um=1., level=0., tile_pixels=256)


def plane(n=3, height=.5):
    y, x = np.indices((n, n), dtype=float)
    return np.stack([x, y, np.full_like(x, height)], axis=-1)


def delivered(tmp_path, xyz, mask=None, errors=None):
    from winding_field.tifxyz_audit import normalize_delivered
    g = grid(xyz.shape[:2])
    src = Store(tmp_path/'projected', g, xyz, mask, errors)
    dst = Store(tmp_path/'delivered', g)
    counts = normalize_delivered(src, dst, g, READER)
    return dst, counts


def audit(f, store, tau=1., guide=None, **kwargs):
    from winding_field.tifxyz_audit import audit_level
    return audit_level(f, store, guide or {}, settings(f, tau), **kwargs)


def assert_blocked(result, code):
    assert result['status'] == 'BLOCKED', result
    assert code in [r['code'] for r in result['reasons']], result


@pytest.mark.parametrize('mask', [np.array([[1, 1], [1, 0]], bool), np.ones((1, 4), bool)])
def test_orphan_cap_or_strip_is_empty(tmp_path, mask):
    xyz = np.zeros((*mask.shape, 3), np.float32)
    with pytest.raises(Blocked, match='EMPTY_DELIVERED_SURFACE'):
        delivered(tmp_path, xyz, mask)


def test_orphans_removed_even_with_surface_and_halos_agree(tmp_path):
    y, x = np.indices((3, 259), dtype=float)
    xyz = np.stack([x, y, np.full_like(x, .5)], -1)
    mask = np.zeros((3, 259), bool); mask[:2, 255:257] = True; mask[2, 258] = True
    store, counts = delivered(tmp_path, xyz, mask)
    assert counts['nodes'] == 4 and counts['cells'] == 1 and counts['orphans_removed'] == 1
    assert not store.read((0, 1))['mask'][2, 3]
    a, b = store.read((0, 0)), store.read((0, 1))
    np.testing.assert_array_equal(a['mask'][:, 255:257], b['mask'][:, :2])
    np.testing.assert_array_equal(a['xyz'][:, 255:257], b['xyz'][:, :2])


def test_reader_rule_must_be_explicit(tmp_path):
    from winding_field.tifxyz_audit import normalize_delivered
    g = grid((2, 2)); src = Store(tmp_path/'p', g, plane(2)); dst = Store(tmp_path/'d', g)
    with pytest.raises(Blocked, match='READER_CONTRACT'):
        normalize_delivered(src, dst, g, dict(READER, inpainting=True))


def test_planar_pitch_h_tau_h_passes_both_diagonals_and_refines_coverage(tmp_path):
    f = field(np.indices((2, 3, 3))[0]-.5)
    store, _ = delivered(tmp_path, plane())
    result = audit(f, store)
    assert result['status'] == 'PASS', result
    assert result['counters']['triangles'] == 16
    assert result['counters']['subcells'] > 0
    assert result['counters']['supported_source_cells'] == 4
    assert result['maxima']['output_bound_um'] < .817
    assert result['native_topology_equivalence'] == 'UNMEASURED'
    assert result['physical_acceptance'] is False
    assert f.stats['read_calls'] <= 3


def test_curved_trilinear_positive_control(tmp_path):
    z, y, x = np.indices((3, 4, 4), dtype=float)
    f = field(z-.5-.03125*x*y)
    xyz = plane(4); xyz[..., 2] += .03125*xyz[..., 0]*xyz[..., 1]
    store, _ = delivered(tmp_path, xyz)
    result = audit(f, store, 1.1)
    assert result['status'] == 'PASS', result


@pytest.mark.parametrize('tau,expected', [(1., 'BLOCKED'), (4., 'PASS')])
def test_omitted_island_distance_sensitivity_and_full_roi(tmp_path, tau, expected):
    z, y, x = np.indices((5, 3, 3), dtype=float)
    values = z-.5
    values[3, 1, 1] = -1.  # A closed fine component, absent from the delivered sheet.
    f = field(values); store, _ = delivered(tmp_path, plane())
    result = audit(f, store, tau)
    assert result['status'] == expected, result
    assert result['native_topology_equivalence'] == 'UNMEASURED'
    if expected == 'BLOCKED': assert_blocked(result, 'SOURCE_UNCOVERED')
    else: assert result['counters']['supported_source_cells'] == 16


def test_orphan_cannot_witness_omitted_island_on_nonempty_surface(tmp_path):
    z, y, x = np.indices((6, 2, 2), dtype=float)
    values = z-.5; values[4, 0, 0] = -1.
    xyz = np.full((2, 5, 3), -1., np.float32); xyz[:, :2] = plane(2)
    xyz[0, 4] = [0, 0, 4]
    mask = np.zeros((2, 5), bool); mask[:, :2] = True; mask[0, 4] = True
    store, _ = delivered(tmp_path, xyz, mask)
    assert_blocked(audit(field(values), store, 1.5), 'SOURCE_UNCOVERED')


def test_supported_vertices_do_not_hide_interior_support_hole(tmp_path):
    z, y, x = np.indices((2, 5, 5), dtype=float)
    support = np.ones(z.shape, bool); support[0, 2, 2] = False
    xyz = plane(2); xyz[..., :2] *= 4
    store, _ = delivered(tmp_path, xyz)
    assert_blocked(audit(field(z-.5, support), store, 5.), 'OUTPUT_UNSUPPORTED_STENCIL')


def test_adjacent_fold_overlaps_more_than_shared_edge(tmp_path):
    xyz = plane(3); xyz[:, 2, 0] = .5
    store, _ = delivered(tmp_path, xyz)
    result = audit(field(np.indices((2, 3, 3))[0]-.5), store, 2.)
    assert result['status'] == 'BLOCKED'
    assert any(r['code'] in ('OUTPUT_ORIENTATION', 'OUTPUT_INTERSECTION') for r in result['reasons'])


@pytest.mark.parametrize('offset', [0., 1.])
def test_nonlocal_coplanar_overlap_or_unexplained_contact(tmp_path, offset):
    xyz = np.full((2, 5, 3), -1., np.float32)
    xyz[:, :2] = plane(2); xyz[:, 3:] = plane(2); xyz[:, 3:, 0] += offset
    mask = np.zeros((2, 5), bool); mask[:, :2] = mask[:, 3:] = True
    store, _ = delivered(tmp_path, xyz, mask)
    assert_blocked(audit(field(np.indices((2, 2, 3))[0]-.5), store, 2.), 'OUTPUT_INTERSECTION')


def test_exact_contact_predicate_shared_edge_and_vertex_only():
    from winding_field.tifxyz_audit import triangle_contact
    a = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float)
    edge = np.array([[1, 0, 0], [0, 0, 0], [0, -1, 0]], float)
    fold = np.array([[1, 0, 0], [0, 0, 0], [.25, .25, 0]], float)
    cross = np.array([[.25, .25, -1], [.25, .25, 1], [1, 1, 0]], float)
    assert triangle_contact(a, edge, a[:2]) is False
    assert triangle_contact(a, fold, a[:2]) is True
    assert triangle_contact(a, cross, []) is True
    assert triangle_contact(a, a+np.array([1., 0., 0.]), [a[1]]) is False
    assert triangle_contact(a, a+np.array([0., 0., np.nextafter(0., 1.)]), []) is False


def seam_guide():
    # Two boundary copies of the same source edge, with opposite orientation.
    world = np.array([[0,0,.5], [1,0,.5], [1,1,.5], [0,1,.5],
                      [1,0,.5], [2,0,.5], [2,1,.5], [1,1,.5]])
    uv = np.array([[0,0], [1,0], [1,1], [0,1], [3,0], [4,0], [4,1], [3,1]], float)
    return dict(world=world, uv=uv, faces=np.array([[0,1,2], [0,2,3], [4,5,6], [4,6,7]]),
                source_vertices=np.array([0,1,2,3,1,4,5,2]), normals=np.tile([0.,0.,1.], (8,1)))


def test_verified_guide_ancestry_allows_true_seam(tmp_path):
    from winding_field.tifxyz_project import ray_roots
    z, y, x = np.indices((2, 17, 17), dtype=float)
    f = field((x*.25-2)**2+(y*.25-2)**2-1, spacing=(.25, .25, 1.))
    points, errors = [], []
    for angle in np.arange(16)*np.pi/8:
        root = ray_roots(f, [2,2,0], [np.cos(angle), np.sin(angle), 0], (.5, 1.5), 1e-8, level=0.)
        assert root['status'] == 'UNIQUE'
        p = np.array(root['roots'][0]['xyz_um']); encoded = p.astype(np.float32)
        points.append(encoded); errors.append(root['roots'][0]['error_um']+np.linalg.norm(encoded-p))
    points.append(points[0]); errors.append(errors[0])
    xyz = np.array([points, points]); xyz[1, :, 2] = 1
    store, _ = delivered(tmp_path, xyz, errors=np.array([errors, errors]))
    faces = np.array([(c, c+1, c+18) for c in range(16)]+[(c, c+18, c+17) for c in range(16)])
    yy, xx = np.indices((2, 17))
    guide = dict(world=xyz.reshape(-1,3).astype(float), uv=np.column_stack([xx.ravel(), yy.ravel()]),
                 faces=faces, source_vertices=np.array([*range(16), 0, *range(16,32), 16]))
    result = audit(f, store, 1., guide)
    assert result['status'] == 'PASS', result
    assert result['counters']['seam_contacts'] > 0
    assert_blocked(audit(f, store, 1.), 'OUTPUT_INTERSECTION')


def test_fake_seam_ancestry_is_not_proximity_welding(tmp_path):
    xyz = np.full((2, 5, 3), -1., np.float32)
    xyz[:, :2] = plane(2); xyz[:, 3:] = plane(2); xyz[:, 3:, 0] += 1
    mask = np.zeros((2, 5), bool); mask[:, :2] = mask[:, 3:] = True
    store, _ = delivered(tmp_path, xyz, mask)
    guide = seam_guide(); guide['world'][4, 0] += .001
    assert_blocked(audit(field(np.indices((2, 2, 3))[0]-.5), store, 1., guide), 'INVALID_SEAM_ANCESTRY')


@pytest.mark.parametrize('limit,code', [('MAX_DEPTH', 'COVERAGE_DEPTH_LIMIT'), ('MAX_SUBCELLS', 'COVERAGE_WORK_LIMIT')])
def test_refinement_exhaustion_is_blocked(tmp_path, monkeypatch, limit, code):
    import winding_field.tifxyz_audit as module
    store, _ = delivered(tmp_path, plane())
    monkeypatch.setattr(module, limit, 0)
    assert_blocked(audit(field(np.indices((2, 3, 3))[0]-.5), store), code)


def test_identically_level_cell_blocks_even_when_witness_is_close(tmp_path):
    store, _ = delivered(tmp_path, plane())
    values = np.indices((3, 3, 3))[0].astype(float)-.5; values[1:] = 0
    assert_blocked(audit(field(values), store, 3.), 'IDENTICALLY_LEVEL_CELL')


def test_certified_delivered_float32_error_enters_bound(tmp_path):
    # At 2**24, encoding the half-integer root moves it by 0.5 um.
    xyz = plane(2); xyz[..., 2] = 2**24+.5
    f = field(np.indices((2, 2, 2))[0]-.5, origin=(0, 0, 2**24))
    store, _ = delivered(tmp_path, xyz, errors=np.full((2, 2), .5))
    frame = dict(settings(f).output_frame, shape_zyx=[2**25, 10, 10])
    from winding_field.tifxyz_audit import audit_level
    result = audit_level(f, store, {}, settings(f, 1., frame))
    assert_blocked(result, 'OUTPUT_DISTANCE_BOUND')


def test_thin_right_triangles_use_boundary_cover_without_relaxing_tau(tmp_path):
    from winding_field.tifxyz_audit import _output_geometry, _SourceCache
    xyz = plane(2); xyz[..., 0] *= 2; xyz[..., 1] *= .25
    store, _ = delivered(tmp_path, xyz)
    f = field(np.indices((2, 2, 3))[0]-.5)
    counters = dict(support_stencils=0)
    maxima = dict(output_bound_um=0., root_error_um=0.)
    _output_geometry(f, store, _SourceCache(f), settings(f).output_frame,
                     1.02, counters, maxima, None)
    assert 1.007 < maxima['output_bound_um'] < 1.008
    assert counters['support_stencils'] > 0


@pytest.mark.parametrize('height, fits', [
    (.25, True), (np.nextafter(np.float32(1), np.float32(0)), True),
    (1., True), (np.nextafter(np.float32(1), np.float32(2)), False), (1.5, False)])
def test_vertex_cover_classifies_right_obtuse_and_acute_exactly(height, fits):
    from fractions import Fraction
    from winding_field.tifxyz_audit import _vertex_cover_bound
    points = [tuple(Fraction(float(x)) for x in p)
              for p in [(-1,0,0), (1,0,0), (0,height,0)]]
    # At height 1.5 the true radius is 13/12: L/2=1 would falsely pass tau1.04.
    bound = _vertex_cover_bound(points, 0.)
    assert (bound <= 1.04) is fits
    assert _vertex_cover_bound(points, .05) > 1.04


def test_frame_is_applied_once_to_delivered_bytes(tmp_path):
    from winding_field.tifxyz_audit import audit_level
    f = field(np.indices((2, 3, 3))[0]-.5, spacing=(2., 2., 2.), origin=(10., 20., 30.))
    store, _ = delivered(tmp_path, plane())
    frame = dict(settings(f).output_frame, origin_xyz_um=[10., 20., 30.], voxel_um=2.)
    result = audit_level(f, store, {}, settings(f, 2., frame))
    assert result['status'] == 'PASS', result


def test_local_pairs_do_not_consume_nonlocal_budget(tmp_path, monkeypatch):
    import winding_field.tifxyz_audit as module
    store, _ = delivered(tmp_path, plane(5))
    monkeypatch.setattr(module, 'MAX_NONLOCAL_PAIRS', 0)
    result = audit(field(np.indices((2, 5, 5))[0]-.5), store)
    assert result['status'] == 'PASS', result
    assert result['counters']['local_pairs'] > 0
    assert result['counters']['nonlocal_pairs'] == 0


def test_nonlocal_candidate_budget_exhaustion_is_blocked(tmp_path, monkeypatch):
    import winding_field.tifxyz_audit as module
    xyz = np.full((2, 5, 3), -1., np.float32); xyz[:, :2] = xyz[:, 3:] = plane(2)
    mask = np.zeros((2, 5), bool); mask[:, :2] = mask[:, 3:] = True
    store, _ = delivered(tmp_path, xyz, mask)
    monkeypatch.setattr(module, 'MAX_NONLOCAL_PAIRS', 0)
    assert_blocked(audit(field(np.indices((2, 2, 2))[0]-.5), store), 'INTERSECTION_WORK_LIMIT')


def test_index_uses_disk_pages_and_store_budget(tmp_path):
    from winding_field.tifxyz_audit import build_spatial_index
    store, _ = delivered(tmp_path, plane())
    receipt = build_spatial_index(store, store.budget)
    assert receipt['nodes'] == 9 and receipt['triangles'] == 16
    assert store.db.execute('SELECT count(*) FROM audit_nodes').fetchone()[0] == 9
    store.fail_guard = True
    with pytest.raises(Blocked, match='DATABASE_LIMIT'): build_spatial_index(store, store.budget)


def test_stop_is_propagated_not_reported_as_pass(tmp_path):
    store, _ = delivered(tmp_path, plane())
    with pytest.raises(Paused): audit(field(np.indices((2, 3, 3))[0]-.5), store, stop=lambda: True)


def test_all_degenerate_cells_are_empty(tmp_path):
    with pytest.raises(Blocked, match='EMPTY_DELIVERED_SURFACE'):
        delivered(tmp_path, np.zeros((2, 2, 3), np.float32))


def test_second_diagonal_cannot_hide_collapsed_triangle(tmp_path):
    xyz = plane(2); xyz[..., 0] += 1
    xyz[1, 0] = [0., 0., .5]  # NW, NE, SW collinear; NW-SE triangles are positive.
    store, _ = delivered(tmp_path, xyz)
    assert_blocked(audit(field(np.indices((2, 2, 3))[0]-.5), store, 2.), 'DEGENERATE_TRIANGLE')


def test_truncated_output_does_not_truncate_source_roi(tmp_path):
    store, _ = delivered(tmp_path, plane(2))
    result = audit(field(np.indices((2, 4, 4))[0]-.5), store)
    assert result['status'] == 'BLOCKED', result
    assert result['reasons'][0]['code'] in ('SOURCE_UNCOVERED', 'COVERAGE_DEPTH_LIMIT')


def test_uncertain_nonfinite_intersection_never_becomes_no_contact():
    from winding_field.tifxyz_audit import triangle_contact
    with pytest.raises(Blocked, match='INTERSECTION_UNCERTAIN'):
        triangle_contact(np.array([[np.nan,0,0], [1,0,0], [0,1,0]]), plane(2)[[0,0,1],[0,1,0]])


def test_indices_and_normalization_do_not_allocate_the_output_image(tmp_path, monkeypatch):
    import scipy.spatial
    from winding_field.tifxyz_audit import normalize_delivered, build_spatial_index
    yy, xx = np.indices((3, 513), dtype=float)
    xyz = np.stack([xx, yy, np.full_like(xx, .5)], -1)
    g = grid((3, 513)); src = Store(tmp_path/'p', g, xyz); dst = Store(tmp_path/'d', g)
    for name in ('empty', 'zeros', 'ones', 'full'):
        original = getattr(np, name)
        def bounded(shape, *args, _original=original, **kwargs):
            dimensions = (shape,) if np.isscalar(shape) else tuple(shape)
            assert dimensions not in ((3,513), (3,513,3), (1539,), (4617,))
            return _original(shape, *args, **kwargs)
        monkeypatch.setattr(np, name, bounded)
    def no_tree(*args, **kwargs): raise AssertionError('output tree is not disk bounded')
    monkeypatch.setattr(scipy.spatial, 'cKDTree', no_tree)
    monkeypatch.setattr(scipy.spatial, 'KDTree', no_tree)
    normalize_delivered(src, dst, g, READER)
    receipt = build_spatial_index(dst, dst.budget)
    assert receipt['nodes'] == 1539 and receipt['triangles'] == 4096
    query = 'SELECT id FROM audit_nodes WHERE x0<=1 AND x1>=0 AND y0<=1 AND y1>=0 AND z0<=1 AND z1>=0'
    rows = dst.db.execute('EXPLAIN QUERY PLAN '+query).fetchmany(4096)
    assert all('TEMP B-TREE' not in str(row).upper() for row in rows)


def test_subcell_value_bounds_do_not_drop_near_zero_level():
    from fractions import Fraction
    from winding_field.tifxyz_audit import _value_interval
    values = np.array([1., -1., 1., -1., 1., -1., 1., -1.]).reshape(2,2,2)
    point = (.5, .5, float(np.nextafter(.5, 1.)))
    low, high = _value_interval(values, point, -2.**-52)
    exact = Fraction(1)-2*Fraction(point[2])+Fraction(2.**-52)
    assert Fraction(low) <= exact <= Fraction(high)
    assert low <= 0 <= high


def test_roi_identity_mismatch_is_not_a_whole_source_certificate(tmp_path):
    from dataclasses import replace
    from winding_field.tifxyz_audit import audit_level
    f = field(np.indices((2, 3, 3))[0]-.5); store, _ = delivered(tmp_path, plane())
    wrong_roi = replace(settings(f), roi_hi_zyx=(2, 4, 4))
    with pytest.raises(ValueError, match='ROI'):
        audit_level(f, store, {}, wrong_roi)


def test_underflowing_metric_is_blocked_without_unbounded_nextafter_loop(monkeypatch):
    from fractions import Fraction
    import winding_field.tifxyz_audit as module
    original = module.math.nextafter; calls = 0
    def bounded_nextafter(*args):
        nonlocal calls
        calls += 1
        assert calls < 10, 'underflow must not walk billions of subnormal floats'
        return original(*args)
    monkeypatch.setattr(module.math, 'nextafter', bounded_nextafter)
    with pytest.raises(Blocked, match='NUMERICAL_UNCERTAINTY'):
        module._sqrt_up(Fraction(1, 10**400))


def test_real_store_live_integration_and_repeat_audit(tmp_path):
    from winding_field.tifxyz_audit import normalize_delivered, audit_level
    from winding_field.tifxyz_store import TileStore
    f = field(np.indices((2, 3, 3))[0]-.5); g = grid((3, 3)); config = settings(f)
    ident = dict(grid=g, output_frame=config.output_frame)
    budget = SimpleNamespace(scope_root=tmp_path, scratch_bytes=128 << 20,
                             database_bytes=4 << 20, reserve_disk_bytes=0)
    fixture = Store(tmp_path/'fixture', g, plane())
    with TileStore(tmp_path/'p', ident, budget, 'projected') as src, TileStore(tmp_path/'d', ident, budget, 'delivered') as dst:
        src.commit(fixture.read((0, 0)))
        normalize_delivered(src, dst, g, READER)
        result = audit_level(f, dst, {}, config)
        assert result['status'] == 'PASS', result
        result = audit_level(f, dst, {}, config)
        assert result['status'] == 'PASS', result
        assert result['counters']['nodes'] == 9
    with TileStore(tmp_path/'d', ident, budget, 'delivered', resume=True) as dst:
        assert audit_level(f, dst, {}, config)['status'] == 'PASS'


def test_nearby_double_layer_is_spatially_covered_but_topology_unmeasured(tmp_path):
    values = np.broadcast_to(np.array([-.5,.5,.5,-.5])[:,None,None], (4,2,2))
    store, _ = delivered(tmp_path, plane(2))
    result = audit(field(values), store, 3.)
    assert result['status'] == 'PASS', result
    assert result['native_topology_equivalence'] == 'UNMEASURED'
    assert result['counters']['supported_source_cells'] == 3


def test_bridge_between_two_level_sheets_is_blocked(tmp_path):
    values = np.broadcast_to(np.array([-.5,.5,.5,-.5])[:,None,None], (4,2,2))
    xyz = plane(2); xyz[1,:,2] = 2.5
    store, _ = delivered(tmp_path, xyz)
    assert_blocked(audit(field(values), store, 3.), 'OUTPUT_ORIENTATION')


def test_clear_contacts_use_exact_integer_signs_without_rational_clipping(monkeypatch):
    import winding_field.tifxyz_audit as module
    def no_rational(*args): raise AssertionError('clear contact needs no Fraction construction/clipping')
    monkeypatch.setattr(module, '_exact', no_rational)
    a = np.array([[0,0,0], [1,0,0], [0,1,0]], dtype=np.float32)
    opposite = np.array([[1,0,0], [0,0,0], [0,-1,0]], dtype=np.float32)
    bent = np.array([[1,0,0], [0,0,0], [0,0,1]], dtype=np.float32)
    fold = np.array([[1,0,0], [0,0,0], [.25,.25,0]], dtype=np.float32)
    # Different powers of two and exact translations must not change any signs.
    for scale, offset in ((1., 0.), (2.**-100, 0.), (2.**60, 2.**64)):
        first = (a*scale+offset).astype(np.float32)
        for second in (opposite, bent):
            assert module.triangle_contact(first, (second*scale+offset).astype(np.float32), first[:2]) is False
        assert module.triangle_contact(first, (fold*scale+offset).astype(np.float32), first[:2]) is True
    assert module.triangle_contact(a, a+[1.,0.,0.], [a[1]]) is False
    assert module.triangle_contact(a, a+[0.,0.,np.nextafter(0.,1.)]) is False
    assert module.triangle_contact(a, a+[0.,0.,1.]) is False


def test_plane_keeps_every_local_pair_without_rational_clipping(tmp_path, monkeypatch):
    import winding_field.tifxyz_audit as module
    store, _ = delivered(tmp_path, plane())
    def no_rational(*args): raise AssertionError(('plane narrowphase fell back to Fraction', args))
    monkeypatch.setattr(module, '_rational_contact', no_rational)
    result = audit(field(np.indices((2,3,3))[0]-.5), store)
    assert result['status'] == 'PASS', result
    assert result['counters']['local_pairs'] == 104
    assert result['counters']['intersection_tests'] == 104
    assert result['counters']['nonlocal_pairs'] == 0


def test_integer_filter_matches_rational_clipping_for_adversarial_pairs():
    import winding_field.tifxyz_audit as module
    rng = np.random.default_rng(715)
    checked = 0
    for kind in ('coplanar', 'spatial', 'edge', 'vertex'):
        for _ in range(80):
            a, b = rng.integers(-8, 9, (2,3,3)).astype(np.float32)/8
            if kind == 'coplanar': a[:,2] = b[:,2] = 0
            if kind == 'edge': b[:2] = a[:2]
            if kind == 'vertex': b[0] = a[0]
            if any(not np.cross(t[1]-t[0], t[2]-t[0]).any() for t in (a,b)): continue
            allowed = a[:2] if kind == 'edge' else a[:1] if kind == 'vertex' else ()
            expected = module._rational_contact(a, b, allowed)
            assert module.triangle_contact(a, b, allowed) == expected, (a, b, allowed)
            checked += 1
    assert checked > 300


def test_complete_audit_uses_bounded_batches_and_matches_exact_fallback(tmp_path,monkeypatch):
    from winding_field.tifxyz_predicates import Backend
    calls=[]; original=Backend.classify
    def classify(self,mode,rows):
        calls.append((mode,len(rows)))
        return original(self,mode,rows)
    monkeypatch.setattr(Backend,'classify',classify)
    store,_=delivered(tmp_path,plane(9));f=field(np.indices((2,9,9))[0]-.5)
    fast=audit(f,store)
    assert fast['status']=='PASS'
    assert {mode for mode,n in calls}=={0,1,2}
    assert max(n for mode,n in calls)<=4096 and max(n for mode,n in calls)>128
    monkeypatch.setattr(Backend,'classify',lambda self,mode,rows:np.full(len(rows),-1,dtype=np.int8))
    assert audit(f,store)==fast


def test_readonly_audit_guards_are_not_directory_walks_per_query(tmp_path,monkeypatch):
    store,_=delivered(tmp_path,plane(9));calls=0;original=store.guard
    def guard(*args,**kwargs):
        nonlocal calls
        calls+=1;return original(*args,**kwargs)
    monkeypatch.setattr(store,'guard',guard)
    result=audit(field(np.indices((2,9,9))[0]-.5),store)
    assert result['status']=='PASS'
    assert calls<64, calls


def test_audit_progress_names_subphases_and_does_not_mutate_acceptance(tmp_path):
    from winding_field.tifxyz_audit import audit_level
    store,_=delivered(tmp_path,plane());f=field(np.indices((2,3,3))[0]-.5);events=[]
    def progress(event):
        events.append(event['phase']);event['counters']['nodes']=-1
    result=audit_level(f,store,{},settings(f),progress=progress)
    assert result['status']=='PASS' and result['counters']['nodes']==9
    assert {'index','geometry','intersections_local','intersections_nonlocal','coverage'}<=set(events)


def test_index_groups_bounded_insert_batches_into_fewer_transactions(tmp_path,monkeypatch):
    from winding_field.tifxyz_audit import build_spatial_index
    store,_=delivered(tmp_path,plane(17));calls=0;original=store.transaction
    def transaction():
        nonlocal calls
        calls+=1;return original()
    monkeypatch.setattr(store,'transaction',transaction)
    result=build_spatial_index(store,store.budget)
    assert result['nodes']==289 and result['triangles']==1024
    assert calls<=4,calls
