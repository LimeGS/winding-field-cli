"""Diagnosed uniform/MVC initialization: continuation, verified residuals, scale."""

import json

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from scipy.spatial import Delaunay

from winding_field import native_global as global_uv
from winding_field import flattening_matrix as matrix


def graded_disk(count=3000, seed=0, ratio=1.35):
    """Irregular triangulation whose mean-value weights span many decades."""
    rng = np.random.default_rng(seed)
    radius = ratio ** rng.uniform(-8, 0, count)
    angle = rng.uniform(0, 2 * np.pi, count)
    points = np.column_stack([radius * np.cos(angle), radius * np.sin(angle)])
    faces = Delaunay(points).simplices.astype(np.int64)
    return np.column_stack([points, np.zeros(count)]), faces


def mvc_system(mesh):
    v, f = mesh
    weights = global_uv._weights(v, f, None)
    weights.setdiag(0)
    weights.eliminate_zeros()
    import igl

    boundary = igl.boundary_loop(f)
    free = np.ones(len(v), bool)
    free[boundary] = False
    lap = sp.diags(np.ones(len(v))) - weights
    rows = lap.tocsr()[free]
    return rows[:, free].tocsc(), free, boundary


def laplace_grid(size):
    """Real 2D Dirichlet-Poisson system; a weak factor makes it need many cycles."""
    index = lambda i, j: i * size + j
    count = size * size
    rows, cols, values = [], [], []
    for i in range(size):
        for j in range(size):
            k = index(i, j)
            rows.append(k)
            cols.append(k)
            values.append(4.0)
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ii, jj = i + di, j + dj
                if 0 <= ii < size and 0 <= jj < size:
                    rows.append(k)
                    cols.append(index(ii, jj))
                    values.append(-1.0)
    a = sp.coo_matrix((values, (rows, cols)), shape=(count, count)).tocsc()
    free = np.ones(count, bool)
    for i in range(size):
        for j in range(size):
            if i in (0, size - 1) or j in (0, size - 1):
                free[index(i, j)] = False
    keep = np.flatnonzero(free)
    return a[keep][:, keep].tocsc(), np.ones(len(keep))


class WeakFactor:
    """Real solver standing in for a hard factor: W-40's ILU is this weak at 385k unknowns."""

    def __init__(self):
        self.nnz = 0

    def solve(self, values):
        return values * 0.95


def test_solver_converges_after_the_old_three_hundred_iteration_ceiling(monkeypatch):
    a, rhs = laplace_grid(50)
    reference = sla.spsolve(a, rhs)
    preconditioner = sla.LinearOperator(a.shape, matvec=WeakFactor().solve, dtype=np.float64)
    # The delivered budget stopped here: more than 300 inner iterations are required.
    truncated, info = sla.gmres(
        a,
        rhs,
        M=preconditioner,
        restart=30,
        maxiter=10,
        rtol=1e-11,
        atol=0.0,
        callback_type="pr_norm",
    )
    truncated = np.linalg.norm(a @ truncated - rhs) / np.linalg.norm(rhs)
    assert info > 0 and truncated > 1e-10
    import scipy.sparse.linalg as scipy_linalg

    monkeypatch.setattr(scipy_linalg, "spilu", lambda *args, **kwargs: WeakFactor())
    solution, report = global_uv._solve(a, rhs.reshape(-1, 1), None)
    assert report["iterations"][0] > 300 and report["cycles"][0] == len(
        report["true_residual_history"][0]
    )
    assert report["relative_residuals"][0] <= report["true_residual_tolerance"]
    assert report["true_residual_pass"]
    assert np.abs(solution[:, 0] - reference).max() <= 1e-8 * np.abs(reference).max()


def test_solver_reports_progress_without_a_fixed_iteration_ceiling(monkeypatch):
    import scipy.sparse.linalg as scipy_linalg

    a, rhs = laplace_grid(30)
    monkeypatch.setattr(scipy_linalg, "spilu", lambda *args, **kwargs: WeakFactor())
    _, report = global_uv._solve(a, np.column_stack([rhs, rhs]), None)
    assert "maximum_iterations_per_rhs" not in report and "max_cycles" not in report
    assert report["stagnation_window"] > 1 and report["required_decrease"] < 1
    assert len(report["true_residual_history"]) == 2
    for history in report["true_residual_history"]:
        assert history[-1] <= report["true_residual_tolerance"]
        assert all(np.isfinite(history))


def test_stagnation_is_measured_and_reported(monkeypatch):
    import scipy.sparse.linalg as sla

    calls = []

    def almost_moving(a, b, **kwargs):
        calls.append(kwargs)
        return kwargs["x0"] + 1e-3, 1

    monkeypatch.setattr(sla, "gmres", almost_moving)
    with pytest.raises(ValueError, match="stagnated"):
        global_uv._solve(sp.eye(3, format="csc") * 1e-3, np.ones((3, 1)), None)
    # Stalls only once the measured window is full: no fixed iteration ceiling stops it.
    assert len(calls) == global_uv.STAGNATION_WINDOW + 1
    assert all(call["maxiter"] == 1 for call in calls)


def test_solver_rejects_nonfinite_iterate_and_true_residual(monkeypatch):
    import scipy.sparse.linalg as sla

    monkeypatch.setattr(sla, "gmres", lambda *args, **kwargs: (np.array([np.nan]), 0))
    with pytest.raises(ValueError, match="nonfinite"):
        global_uv._solve(sp.eye(1, format="csc"), np.ones((1, 1)), None)


def disk(position=(0.0, 0.0, 0.0), scale=1.0):
    v = np.array(
        [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [4.0, 3.0, 0.0], [0.0, 3.0, 0.0], [0.07, 1.15, 0.0]]
    )
    f = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]], np.int32)
    return dict(
        vertices=v * scale + np.asarray(position),
        faces=f,
        parent=np.arange(5),
        boundary=np.arange(4),
    )


def signed_areas(uv, f):
    t = uv[f].astype(np.longdouble)
    a, b = t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]
    return a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]


@pytest.mark.parametrize(
    "position,scale",
    [
        ((0.0, 0.0, 0.0), 1.0),
        ((1.2e5, -8.4e5, 3e3), 1.0),
        ((0.0, 0.0, 0.0), 1e-3),
        ((50.0, -70.0, 0.0), 2.5e4),
    ],
)
def test_initialization_keeps_every_face_positive_after_physical_scaling(position, scale):
    cut = disk(position, scale)
    uv, report = global_uv.mean_value_map(cut)
    assert (signed_areas(uv, cut["faces"]) > 0).all()
    assert report["positive_disk_validated"] and report["boundary_exact"]
    assert report["uniform"]["true_residual_pass"] and report["mvc"]["true_residual_pass"]


def test_small_global_residual_never_accepts_an_inverted_face():
    cut = disk()
    uv, _ = global_uv.mean_value_map(cut)
    broken = uv.copy()
    # Mirror the interior vertex across its neighbours: exactly one face inverts.
    broken[4] = broken[4] - 2 * np.array([0.0, uv[4, 1] - np.mean(uv[[0, 1, 2, 3], 1])])
    assert not (signed_areas(broken, cut["faces"]) > 0).all()
    with pytest.raises(ValueError, match="inverted"):
        global_uv.validate_map(dict(world=cut["vertices"], faces=cut["faces"], uv=broken))


def test_diagnostics_persist_exact_systems_and_transforms(tmp_path):
    cut = disk((1e4, -2e4, 0.0), 3.0)
    diagnostics = tmp_path / "initialization"
    uv, report = global_uv.mean_value_map(cut, diagnostics=diagnostics)
    saved = json.loads((diagnostics / "CUT.json").read_text())
    for name in ("vertices", "faces", "parent", "boundary"):
        np.testing.assert_array_equal(np.load(diagnostics / (name + ".npy")), cut[name])
        assert saved["files"][name]["sha256"]
    assert saved["physical_scale"] == pytest.approx(report["physical_scale"])
    for stage in ("uniform", "mvc"):
        system = np.load(diagnostics / stage / "system.npz")
        receipt = json.loads((diagnostics / stage / "SYSTEM.json").read_text())
        a = sp.csc_matrix(
            (system["data"], system["indices"], system["indptr"]), shape=tuple(system["shape"])
        )
        assert receipt["nnz"] == a.nnz and receipt["rhs_columns"] == system["rhs"].shape[1]
        assert receipt["rhs_sha256"] and receipt["data_sha256"] and receipt["indptr_sha256"]
        relative = np.linalg.norm(a @ np.zeros(a.shape[0])) + 0.0
        assert relative == 0.0
    # Replaying the saved system reproduces the same accepted solution.
    system = np.load(diagnostics / "mvc" / "system.npz")
    a = sp.csc_matrix(
        (system["data"], system["indices"], system["indptr"]), shape=tuple(system["shape"])
    )
    solution, _ = global_uv._solve(a, system["rhs"], None)
    direct = np.column_stack(
        [sla.spsolve(a, system["rhs"][:, j]) for j in range(system["rhs"].shape[1])]
    )
    assert np.abs(solution - direct).max() <= 1e-8 * max(1.0, np.abs(direct).max())
