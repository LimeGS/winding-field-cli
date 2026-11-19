"""Benchmarks must not change source geometry, identities or live inference."""

import importlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest

from winding_field.util import atomic_json, file_sha


@pytest.fixture
def bench(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    assert importlib.util.find_spec("bench_flattening"), "benchmark is not implemented"
    return importlib.import_module("bench_flattening")


def source_mesh(tmp_path, field="175a", unit=8.64, geometry=None):
    from winding_field.surface_mesh import import_mesh

    tmp_path.mkdir(parents=True, exist_ok=True)
    v = np.array(
        [
            [0.0, 0.0, 7000.0],
            [2.0, 0.0, 7000.0],
            [0.0, 2.0, 7000.0],
            [8.0, 0.0, 7000.0],
            [9.0, 0.0, 7000.0],
            [8.0, 1.0, 7000.0],
        ]
    )
    f = np.array([[0, 1, 2], [3, 4, 5]], np.int64)
    if geometry is not None:
        v, f = geometry
    path = tmp_path / "input.npz"
    np.savez(path, vertices_xyz=v, faces=f)
    mesh = tmp_path / "mesh"
    metadata = dict(
        frame=field,
        voxel_um=unit,
        origin_xyz=[0, 0, 7000],
        spacing_xyz=[32, 32, 32],
        shape_zyx=[8, 8, 8],
        sheet_step=1.0,
    )
    import_mesh(dict(path=str(path), sha256=file_sha(path)), mesh, metadata)
    source = dict(
        field_id=field,
        resolution_id="coarse",
        W=0,
        raw_phi=0.0,
        frame=field,
        mesh=str(mesh),
        mesh_manifest_sha256=file_sha(mesh / "MESH.json"),
        ct_profile=dict(
            field_id=field,
            frame=field,
            source="https://example.test/" + field,
            level=0,
            spacing_xyz_um=[unit] * 3,
            origin_xyz_um=[0.0] * 3,
            shape_zyx=[8, 8, 8000],
            pitch_um=unit,
            overview_pitch_um=unit * 2,
            offsets_um=[-unit, 0.0, unit],
        ),
    )
    return source, v, f


def test_mesh_units_are_not_field_spacing(bench, tmp_path):
    src, v, f = source_mesh(tmp_path / "src")
    original = file_sha(Path(src["mesh"]) / "mesh.sqlite")
    case = bench.load_case(src, tmp_path / "case")
    a = bench.case_arrays(case)
    np.testing.assert_array_equal(a["world_um"], v * 8.64)
    np.testing.assert_array_equal(a["faces"], f)
    assert case["area_mm2"] == pytest.approx(2.5 * 8.64**2 / 1e6)
    assert len(a["source_face_ids"]) == 2
    assert file_sha(Path(src["mesh"]) / "mesh.sqlite") == original
    saved = file_sha(tmp_path / "case" / "SOURCE.json")
    assert bench.load_case(src, tmp_path / "case")["source_sha256"] == case["source_sha256"]
    assert file_sha(tmp_path / "case" / "SOURCE.json") == saved


def test_paris4_raw_phi_and_global_origin(bench, tmp_path):
    src, v, _ = source_mesh(tmp_path / "src", "paris4", 9.6)
    case = bench.load_case(src, tmp_path / "case")
    np.testing.assert_array_equal(bench.case_arrays(case)["world_um"], v * 9.6)
    assert bench.raw_level(-60, 1.192497988541921) == -71.54987931251526


def test_resolves_moved_database_by_hash(bench, tmp_path):
    src, _, _ = source_mesh(tmp_path / "src")
    job = dict(
        source_control=dict(
            database="/gone/mesh.sqlite",
            database_sha256=file_sha(Path(src["mesh"]) / "mesh.sqlite"),
        )
    )
    assert bench.resolve_mesh(job, [Path(src["mesh"]) / "MESH.json"]) == Path(src["mesh"])
    job["source_control"]["database_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="unresolved"):
        bench.resolve_mesh(job, [Path(src["mesh"]) / "MESH.json"])


def test_rejects_changed_mesh_and_ambiguous_binding(bench, tmp_path):
    src, _, _ = source_mesh(tmp_path / "src")
    case = bench.load_case(src, tmp_path / "case")
    np.save(tmp_path / "case" / "world_um.npy", np.zeros((1, 3)))
    with pytest.raises(ValueError, match="hash|changed"):
        bench.load_case(src, tmp_path / "case")
    changed = dict(src, ct_profile=dict(src["ct_profile"], field_id="paris4"))
    with pytest.raises(ValueError, match="field|binding"):
        bench.load_case(changed, tmp_path / "bad")
    with pytest.raises(ValueError, match="separate|overlap"):
        bench.load_case(src, Path(src["mesh"]) / "output")
    assert case["field_id"] == "175a"


def test_two_field_profiles_do_not_alias_cases_or_ct(bench, tmp_path):
    a, _, _ = source_mesh(tmp_path / "a")
    b, _, _ = source_mesh(tmp_path / "b", "paris4", 9.6)
    cases = [bench.load_case(s, tmp_path / s["field_id"]) for s in (a, b)]
    assert cases[0]["id"] != cases[1]["id"]
    assert cases[0]["ct_profile_sha256"] != cases[1]["ct_profile_sha256"]


def test_missing_w26_is_not_invented(bench, tmp_path):
    src, _, _ = source_mesh(tmp_path / "src")
    absent = dict(src, W=26, mesh=str(tmp_path / "missing"))
    with pytest.raises(FileNotFoundError):
        bench.load_case(absent, tmp_path / "case")
    assert not (tmp_path / "case").exists()


def test_unobserved_paris4_sentinel_is_not_used_as_support(bench):
    cfg = json.loads(
        (Path(__file__).resolve().parents[1] / "examples/paris4-remote.json").read_text()
    )
    assert bench.field_descriptor(cfg)["field"]["support_policy"] == "stored"
    cfg["field"]["support_policy"] = "observed"
    with pytest.raises(ValueError, match="stored"):
        bench.field_descriptor(cfg)


def test_sheet_and_patch_domains_are_solver_independent(bench, tmp_path):
    from test_flattening_matrix import geometry

    src, v, f = source_mesh(tmp_path / "src", unit=1.0, geometry=geometry())
    case = bench.load_case(src, tmp_path / "case")
    for scope in ("sheet", "patches"):
        domains = bench.prepare_domains(
            case, tmp_path / scope, scope, lambda: None, max_patch_faces=32
        )
        before = file_sha(tmp_path / scope / "DOMAINS.json")
        again = bench.prepare_domains(
            case, tmp_path / scope, scope, lambda: None, max_patch_faces=32
        )
        assert domains["identity"] == again["identity"]
        assert file_sha(tmp_path / scope / "DOMAINS.json") == before
        rows = []
        for d in domains["domains"]:
            np.testing.assert_array_equal(d["vertices"], v[d["parent"]])
            np.testing.assert_array_equal(d["parent"][d["faces"]], f[d["source_face_rows"]])
            rows.extend(d["source_face_rows"])
        np.testing.assert_array_equal(np.sort(rows), np.arange(121))
        assert len(domains["domains"]) == 2 if scope == "sheet" else len(domains["domains"]) > 2


def test_hole_is_not_filled_and_island_survives(bench, tmp_path):
    from test_flattening_matrix import geometry

    src, _, f = source_mesh(tmp_path / "src", unit=1.0, geometry=geometry())
    case = bench.load_case(src, tmp_path / "case")
    d = bench.prepare_domains(case, tmp_path / "domains", "sheet", lambda: None)
    solved = [
        dict(status="COMPLETE", uv=cut["vertices"][:, [0, 2]].copy(), receipt={})
        for cut in d["domains"]
    ]
    mapped = bench.assemble_map(case, d, solved, tmp_path / "mapped")
    np.testing.assert_array_equal(np.sort(mapped["source_face_rows"]), np.arange(len(f)))
    assert not mapped["omitted_face_rows"].size
    assert len(np.unique(mapped["domain_ids"])) == 2


def test_patch_calibration_is_per_original_component(bench, tmp_path):
    v = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    f = np.array([[0, 1, 2], [0, 2, 3]])
    src, _, _ = source_mesh(tmp_path / "src", unit=1.0, geometry=(v, f))
    case = bench.load_case(src, tmp_path / "case")
    d = bench.prepare_domains(
        case, tmp_path / "domains", "patches", lambda: None, max_patch_faces=1
    )
    solved = [
        dict(status="COMPLETE", uv=c["vertices"][:, :2] * (2 + 2 * i), receipt={})
        for i, c in enumerate(d["domains"])
    ]
    mapped = bench.assemble_map(case, d, solved, tmp_path / "mapped")
    tri = mapped["uv"][mapped["faces"]]
    lengths = np.linalg.norm(tri[:, 1] - tri[:, 0], axis=1)
    world = mapped["world"][mapped["faces"]]
    stretch = lengths / np.linalg.norm(world[:, 1] - world[:, 0], axis=1)
    assert max(stretch) / min(stretch) == pytest.approx(2.0)
    assert len(mapped["report"]["calibration"]) == 1


def test_patch_budget_does_not_repartition_sheet_scope(bench, tmp_path):
    from test_flattening_matrix import geometry

    src, _, f = source_mesh(tmp_path / "src", unit=1.0, geometry=geometry())
    case = bench.load_case(src, tmp_path / "case")
    domains = bench.prepare_domains(
        case, tmp_path / "domains", "sheet", lambda: None, max_patch_faces=1
    )
    rows = []
    for d in domains["domains"]:
        np.testing.assert_array_equal(d["parent"][d["faces"]], f[d["source_face_rows"]])
        rows.extend(d["source_face_rows"])
    np.testing.assert_array_equal(np.sort(rows), np.arange(len(f)))
    assert len(domains["domains"]) == 2


def folded_grid():
    n = 8
    x, y = np.meshgrid(np.arange(n, dtype=float), np.arange(n, dtype=float))
    vertices = np.c_[x.ravel(), y.ravel(), 4 * np.random.default_rng(3).normal(size=n * n)]
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            k = i * n + j
            faces.extend([[k, k + 1, k + n + 1], [k, k + n + 1, k + n]])
    return vertices, np.asarray(faces)


def test_spatial_patch_cut_refines_pinched_vertices_without_losing_faces(bench, tmp_path):
    import igl

    v, f = folded_grid()
    assert igl.is_vertex_manifold(f).all()
    source, _, _ = source_mesh(tmp_path / "source", unit=1.0, geometry=(v, f))
    case = bench.load_case(source, tmp_path / "case")
    domains = bench.prepare_domains(
        case, tmp_path / "domains", "patches", lambda: None, max_patch_faces=16
    )
    owned = []
    for cut in domains["domains"]:
        assert igl.is_vertex_manifold(cut["faces"]).all()
        assert len(cut["faces"]) <= 16
        np.testing.assert_array_equal(cut["vertices"], v[cut["parent"]])
        np.testing.assert_array_equal(cut["parent"][cut["faces"]], f[cut["source_face_rows"]])
        owned.extend(cut["source_face_rows"])
    np.testing.assert_array_equal(np.sort(owned), np.arange(98))
    assert len(owned) == len(set(owned))
    assert {cut["component_id"] for cut in domains["domains"]} == {0}
    assert bench.read_json(tmp_path / "domains/DOMAINS.json")["report"]["topology_refinements"] > 0


def test_pinched_sheet_is_rejected_instead_of_silently_using_patches(bench, tmp_path):
    v, f = folded_grid()
    rows = [43, 42, 45, 57, 70, 59, 73, 71, 58, 86, 72, 89, 87, 84]
    source, _, _ = source_mesh(tmp_path / "source", unit=1.0, geometry=(v, f[rows]))
    case = bench.load_case(source, tmp_path / "case")
    with pytest.raises(ValueError, match="nonmanifold vertex"):
        bench.prepare_domains(case, tmp_path / "domains", "sheet", lambda: None)
    assert not (tmp_path / "domains/DOMAINS.json").exists()


def disk_cut():
    return dict(
        vertices=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]),
        faces=np.array([[0, 1, 2], [0, 2, 3]], np.int64),
        boundary=np.arange(4),
        parent=np.arange(4),
    )


def test_solver_name_does_not_hide_fallback(bench, tmp_path):
    with pytest.raises(ValueError, match="unknown"):
        bench.solve_domain("invented", disk_cut(), tmp_path / "solver", {}, lambda: None)
    for method in ("lscm", "mvc", "arap"):
        result = bench.solve_domain(method, disk_cut(), tmp_path / method, {}, lambda: None)
        assert result["status"] == "COMPLETE"
        assert result["receipt"]["method"] == method
        assert result["uv"].shape == (4, 2)
        assert np.isfinite(result["uv"]).all()


def test_arap_seed_is_validated_before_slim(bench, tmp_path, monkeypatch):
    from winding_field import native_global

    monkeypatch.setattr(native_global, "arap_map", lambda cut, uv, guard: (np.zeros((4, 2)), {}))
    with pytest.raises(ValueError, match="seed"):
        bench.solve_domain(
            "arap-slim",
            disk_cut(),
            tmp_path / "solver",
            dict(app_run="unused", native_recipe="historical"),
            lambda: None,
        )
    assert not (tmp_path / "solver" / "native").exists()


def test_external_reorder_requires_exact_correspondence(bench):
    cut = disk_cut()
    parent = np.array([2, 3, 0, 1])
    inverse = np.argsort(parent)
    data = dict(
        world=cut["vertices"][parent],
        faces=inverse[cut["faces"]],
        uv=cut["vertices"][parent, :2],
        parent=parent,
        source_face_rows=np.arange(2),
    )
    uv = bench.external_uv(cut, data)
    np.testing.assert_array_equal(uv, cut["vertices"][:, :2])
    data["world"][0, 0] += 0.01
    with pytest.raises(ValueError, match="geometry"):
        bench.external_uv(cut, data)


def test_geometry_edit_and_hole_fill_are_incompatible(bench):
    cut = disk_cut()
    data = dict(
        world=cut["vertices"],
        faces=np.r_[cut["faces"], [[0, 1, 3]]],
        uv=cut["vertices"][:, :2],
        parent=np.arange(4),
        source_face_rows=np.arange(3),
    )
    with pytest.raises(ValueError, match="geometry|correspondence"):
        bench.external_uv(cut, data)


def test_missing_backend_is_not_a_pass(bench, tmp_path):
    result = bench.solve_domain(
        "bff",
        disk_cut(),
        tmp_path / "solver",
        dict(status="UNAVAILABLE", reason="build not qualified"),
        lambda: None,
    )
    assert result["status"] == "UNAVAILABLE"
    assert "uv" not in result
    assert result["receipt"]["reason"] == "build not qualified"


def cell_request(bench, tmp_path):
    from ct_fixture import write_ct
    from test_flattening_ct import profile, config

    src, _, _ = source_mesh(
        tmp_path / "src",
        unit=1.0,
        geometry=(
            np.array([[4.0, 4.0, 3.0], [8.0, 4.0, 3.0], [8.0, 8.0, 3.0], [4.0, 8.0, 3.0]]),
            np.array([[0, 1, 2], [0, 2, 3]]),
        ),
    )
    src["ct_profile"] = profile(write_ct(tmp_path / "ct", np.full((32, 32, 32), 90, "u1")))
    case = bench.load_case(src, tmp_path / "case")
    return dict(
        case=case,
        method="lscm",
        scope="sheet",
        track="A",
        backend={},
        shared=str(tmp_path / "shared"),
        ct_config=config(tmp_path, src["ct_profile"]),
        reserve_bytes=0,
        seconds=60.0,
    )


def test_completed_resume_keeps_receipts(bench, tmp_path):
    req = cell_request(bench, tmp_path)
    out = tmp_path / "cell"
    result = bench.run_cell(req, out)
    assert result["status"] == "COMPLETE"
    before = {str(p): file_sha(p) for p in out.rglob("*") if p.is_file()}
    again = bench.run_cell(req, out)
    assert again == result
    assert before == {str(p): file_sha(p) for p in out.rglob("*") if p.is_file()}


def test_changed_ct_or_backend_requires_new_identity(bench, tmp_path):
    req = cell_request(bench, tmp_path)
    bench.run_cell(req, tmp_path / "cell")
    changed = dict(req, backend=dict(revision="changed"))
    with pytest.raises(ValueError, match="identity"):
        bench.run_cell(changed, tmp_path / "cell")
    ct = Path(req["ct_config"]["profile"]["source"]) / "0/0.0.0"
    ct.write_bytes(bytes(32**3))
    with pytest.raises(ValueError, match="changed"):
        bench.run_cell(req, tmp_path / "cell")


def test_disk_reserve_pauses_before_write(bench, tmp_path, monkeypatch):
    import shutil

    monkeypatch.setattr(bench.shutil, "disk_usage", lambda p: shutil._ntuple_diskusage(100, 99, 1))
    with pytest.raises(InterruptedError, match="reserve"):
        bench.guard_output(tmp_path, reserve_bytes=20)
    assert not list(tmp_path.iterdir())


def test_failed_cell_does_not_complete_parent(bench):
    assert (
        bench.queue_status([dict(status="COMPLETE"), dict(status="FAILED")]) == "FINISHED_WITH_GAPS"
    )
    assert bench.queue_status([dict(status="COMPLETE"), dict(status="PENDING")]) == "RUNNING"


def test_owned_cleanup_never_signals_unrelated_process(bench):
    import subprocess

    other = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(60)"])

    def stop():
        raise InterruptedError("test stop")

    try:
        with pytest.raises(InterruptedError):
            bench.supervise([sys.executable, "-c", "import time;time.sleep(60)"], stop)
        assert other.poll() is None
    finally:
        other.terminate()
        other.wait()


def test_shared_phase_is_not_clobbered_by_two_cells(bench, tmp_path):
    with bench.locked(tmp_path):
        with pytest.raises(BlockingIOError):
            with bench.locked(tmp_path):
                pytest.fail("second writer acquired shared phase")


def test_inventory_and_pilot_queue_are_field_bound(bench, tmp_path):
    from ct_fixture import write_ct
    from test_flattening_ct import profile

    sources = []
    for field in ("175a", "paris4"):
        src, _, _ = source_mesh(
            tmp_path / field,
            field,
            1.0,
            geometry=(
                np.array([[4.0, 4.0, 3.0], [8.0, 4.0, 3.0], [8.0, 8.0, 3.0], [4.0, 8.0, 3.0]]),
                np.array([[0, 1, 2], [0, 2, 3]]),
            ),
        )
        ct = write_ct(tmp_path / (field + "-ct"), np.full((32, 32, 32), 90, "u1"))
        src.update(stage="pilot", ct_profile=profile(ct, field=field))
        sources.append(src)
    cfg = dict(
        sources=sources,
        methods=["lscm"],
        scopes=["sheet"],
        backends={},
        resources=dict(
            reserve_bytes=0, seconds=60.0, cache_bytes=1 << 24, decoded_cache_bytes=1 << 20
        ),
    )
    run = tmp_path / "run"
    result = bench.inventory(cfg, run)
    assert result["fields"] == ["175a", "paris4"]
    assert bench.run_queue(run, "pilot")["status"] == "COMPLETE"
    rows = bench.read_json(run / "QUEUE.json")["rows"]
    assert len(rows) == 2 and all(r["status"] == "COMPLETE" for r in rows)
    before = {p: file_sha(p) for p in run.rglob("CELL.json")}
    bench.run_queue(run, "pilot")
    assert before == {p: file_sha(p) for p in run.rglob("CELL.json")}


def test_holdout_queue_requires_bound_review_freeze(bench, tmp_path):
    with pytest.raises(ValueError, match="FREEZE|freeze"):
        bench.run_queue(tmp_path, "holdout")


def test_inference_stall_pauses_benchmark_not_inference(bench, tmp_path):
    import os

    status = tmp_path / "STATUS.json"
    atomic_json(status, dict(phase="INFERENCE", W=16))
    os.utime(status, (0, 0))
    with pytest.raises(InterruptedError, match="protected|inference"):
        bench.protect_inference(dict(root=str(tmp_path), max_stale_seconds=900))
    assert bench.read_json(status) == dict(phase="INFERENCE", W=16)


def test_stop_keeps_latest_uv_and_ct_tiles(bench, tmp_path, monkeypatch):
    import flattening_ct as ct

    req = cell_request(bench, tmp_path)
    out = tmp_path / "cell"
    real = ct.seal
    stopped = []

    def stop_after_tile(root, name, *args):
        record = real(root, name, *args)
        if name == "TILE.json" and not stopped:
            stopped.append(Path(root) / name)
            (out / "STOP").write_text("test pause")
        return record

    monkeypatch.setattr(ct, "seal", stop_after_tile)
    with pytest.raises(InterruptedError, match="STOP"):
        bench.run_cell(req, out)
    before = file_sha(stopped[0])
    assert (out / "map/MAP.json").exists() and not (out / "CELL.json").exists()
    (out / "STOP").unlink()
    assert bench.run_cell(req, out)["status"] == "COMPLETE"
    assert file_sha(stopped[0]) == before


def test_inventory_resume_rejects_changed_snapshot(bench, tmp_path):
    from test_flattening_ct import profile

    sources = []
    for field in ("175a", "paris4"):
        src, _, _ = source_mesh(tmp_path / field, field, 1.0)
        src["stage"] = "pilot"
        sources.append(src)
    cfg = dict(
        sources=sources,
        methods=["lscm"],
        scopes=["sheet"],
        backends={},
        resources=dict(
            reserve_bytes=0, seconds=60.0, cache_bytes=1 << 24, decoded_cache_bytes=1 << 20
        ),
    )
    root = tmp_path / "run"
    bench.inventory(cfg, root)
    np.save(root / "cases/0000/world_um.npy", np.zeros((1, 3)))
    with pytest.raises(ValueError, match="hash|changed"):
        bench.inventory(cfg, root)


def test_partial_inventory_can_continue_without_replacing_sources(bench, tmp_path):
    sources = []
    for field in ("175a", "paris4"):
        src, _, _ = source_mesh(tmp_path / field, field, 1.0)
        src["stage"] = "pilot"
        sources.append(src)
    cfg = dict(
        sources=sources,
        methods=["lscm"],
        scopes=["sheet"],
        backends={},
        resources=dict(
            reserve_bytes=0, seconds=60.0, cache_bytes=1 << 24, decoded_cache_bytes=1 << 20
        ),
    )
    root = tmp_path / "run"
    bench.inventory(cfg, root)
    before = file_sha(root / "cases/0000/SOURCE.json")
    (root / "INVENTORY.json").unlink()
    assert len(bench.inventory(cfg, root)["entries"]) == 2
    assert file_sha(root / "cases/0000/SOURCE.json") == before


def test_worker_sigterm_leaves_owned_stop_marker(bench, tmp_path):
    import os
    import signal

    previous = signal.getsignal(signal.SIGTERM)
    with pytest.raises(InterruptedError, match="signal"):
        with bench.stop_signals(tmp_path):
            os.kill(os.getpid(), signal.SIGTERM)
    assert (tmp_path / "STOP").exists()
    assert signal.getsignal(signal.SIGTERM) == previous


def test_cell_scratch_budget_is_checked_before_solver(bench, tmp_path):
    req = cell_request(bench, tmp_path)
    out = tmp_path / "cell"
    out.mkdir()
    (out / "retained.bin").write_bytes(b"x" * 128)
    req["scratch_bytes"] = 64
    with pytest.raises(MemoryError, match="scratch"):
        bench.run_cell(req, out)
    assert (out / "retained.bin").read_bytes() == b"x" * 128
    assert not (out / "map").exists()


def test_slim_seed_obj_preserves_exact_float64_xyz(bench, tmp_path, monkeypatch):
    from winding_field import flatboi_checkpoint as native
    import shutil

    cut = disk_cut()
    cut["vertices"] = cut["vertices"] * np.pi + [
        0.12345678901234567,
        0.23456789012345678,
        0.34567890123456789,
    ]

    def run(args, *, guard):
        native.positive_uv(args.input, cut["vertices"], cut["faces"], guard)
        args.output.mkdir()
        shutil.copyfile(args.input, args.output / "checkpoint.obj")
        return dict(checkpoint=dict(obj="checkpoint.obj"))

    monkeypatch.setattr(native, "run", run)
    result = bench.solve_domain(
        "slim",
        cut,
        tmp_path / "solver",
        dict(app_run="unused", native_recipe="historical"),
        lambda: None,
    )
    assert result["status"] == "COMPLETE"


def test_175a_legacy_requires_saved_job_map(bench, tmp_path):
    req = dict(cell_request(bench, tmp_path), method="legacy-lscm")
    with pytest.raises(ValueError, match="saved|legacy"):
        bench.run_cell(req, tmp_path / "cell")


@pytest.mark.parametrize("name", ["native_global", "native_mesh", "native_cache_index"])
def test_identity_binds_consumed_project_implementations(bench, tmp_path, monkeypatch, name):
    module = importlib.import_module("winding_field." + name)
    before = bench.tool_identity()
    replacement = tmp_path / (name + ".py")
    replacement.write_text("changed implementation")
    monkeypatch.setattr(module, "__file__", str(replacement))
    assert bench.tool_identity() != before


def test_missing_sources_keep_all_review_combinations(bench, tmp_path):
    sources = []
    for field in ("175a", "paris4"):
        src, _, _ = source_mesh(tmp_path / field, field, 1.0)
        sources.append(dict(src, stage="pilot", mesh=str(tmp_path / ("absent-" + field))))
    config = dict(
        sources=sources,
        methods=["lscm", "arap"],
        scopes=["sheet", "patches"],
        backends={},
        resources=dict(
            reserve_bytes=0, seconds=60.0, cache_bytes=1 << 24, decoded_cache_bytes=1 << 20
        ),
    )
    bench.inventory(config, tmp_path / "run")
    queue = bench.read_json(tmp_path / "run/QUEUE.json")
    assert len(queue["rows"]) == 8
    assert {r["status"] for r in queue["rows"]} == {"MISSING_INPUT"}
    assert bench.run_queue(tmp_path / "run", "pilot")["status"] == "FINISHED_WITH_GAPS"


@pytest.mark.parametrize(
    "changed,value",
    [
        ("cpu.max", "max 100000"),
        ("memory.max", str(21 << 30)),
        ("memory.swap.max", str(64 << 30)),
        ("nice", "0"),
        ("available", "1"),
    ],
)
def test_heavy_queue_requires_actual_safe_envelope(bench, tmp_path, changed, value):
    cgroups, proc = tmp_path / "cgroup", tmp_path / "proc"
    (proc / "self").mkdir(parents=True)
    (proc / "self/cgroup").write_text("0::/field-flattening-test.slice/queue.service\n")
    fields = ["S"] + ["0"] * 18
    fields[16], fields[17] = "10", "1"  # nice (field19), then thread count (field20).
    (proc / "self/stat").write_text("1 (worker) " + " ".join(fields))
    (proc / "meminfo").write_text(f"MemAvailable: {30 << 20} kB\n")
    group = cgroups / "field-flattening-test.slice"
    for p in (group, group / "queue.service"):
        p.mkdir(parents=True, exist_ok=True)
        for name, text in [
            ("cpu.max", "200000 100000"),
            ("memory.max", str(20 << 30)),
            ("memory.swap.max", "0"),
        ]:
            (p / name).write_text(text)
    assert bench.require_benchmark_envelope(cgroups, proc)["slice"] == group.name
    if changed == "nice":
        fields[16] = value
        (proc / "self/stat").write_text("1 (worker) " + " ".join(fields))
    elif changed == "available":
        (proc / "meminfo").write_text(f"MemAvailable: {value} kB\n")
    else:
        for p in (group, group / "queue.service"):
            (p / changed).write_text(value)
    with pytest.raises(ValueError, match="envelope|headroom|Nice"):
        bench.require_benchmark_envelope(cgroups, proc)


@pytest.mark.skipif(
    sys.platform != "linux" or os.environ.get("FLATTENING_SYSTEMD_TESTS") != "1",
    reason="set FLATTENING_SYSTEMD_TESTS=1 on a privileged Linux qualification host",
)
def test_failed_worker_cgroup_kills_detached_solver(bench, tmp_path, monkeypatch):
    import os
    import subprocess
    import uuid

    (tmp_path / "expected_worker_env.py").write_text("VALUE = 7\n")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")
    pidfile = tmp_path / "detached.pid"
    worker = (
        "import subprocess,sys,pathlib,os,expected_worker_env;"
        "assert expected_worker_env.VALUE==7 and os.environ['OPENBLAS_NUM_THREADS']=='1';"
        "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(120)'],start_new_session=True);"
        f"pathlib.Path({str(pidfile)!r}).write_text(str(p.pid));sys.exit(7)"
    )
    try:
        with pytest.raises(subprocess.CalledProcessError):
            bench.supervise(
                [sys.executable, "-c", worker],
                lambda: None,
                unit="field-flattening-cell-test-" + uuid.uuid4().hex,
            )
        pid = int(pidfile.read_text())
        stat = Path(f"/proc/{pid}/stat")
        assert not stat.exists() or stat.read_text().rsplit(")", 1)[1].split()[0] == "Z"
    finally:
        if pidfile.exists():
            try:
                os.kill(int(pidfile.read_text()), 9)
            except ProcessLookupError:
                pass


def test_worker_stop_exports_paused_exit_for_supervisor(bench, tmp_path):
    req = cell_request(bench, tmp_path)
    out = tmp_path / "cell"
    out.mkdir()
    (out / "STOP").write_text("pause")
    atomic_json(out / "WORKER-REQUEST.json", req)
    with pytest.raises(SystemExit) as stopped:
        bench.main(["worker", "--output", str(out)])
    assert stopped.value.code == 75
    assert bench.read_json(out / "WORKER-EXIT.json")["status"] == "PAUSED"


def test_unavailable_backend_is_recorded_without_preparing_domains(bench, tmp_path, monkeypatch):
    req = dict(
        cell_request(bench, tmp_path),
        method="bff",
        backend=dict(status="UNAVAILABLE", reason="build not qualified"),
    )

    def unsupported_domain(*args, **kwargs):
        raise ValueError("unsupported topology")

    monkeypatch.setattr(bench, "prepare_domains", unsupported_domain)
    result = bench.run_cell(req, tmp_path / "cell")
    assert result["status"] == "UNAVAILABLE"
    assert (tmp_path / "cell/CELL.json").exists()
    assert not (tmp_path / "cell/map").exists()


def test_each_worker_admission_pauses_when_host_headroom_falls(bench, tmp_path, monkeypatch):
    sources = []
    for field in ("175a", "paris4"):
        source, _, _ = source_mesh(tmp_path / field, field, 1.0)
        sources.append(dict(source, stage="pilot"))
    config = dict(
        sources=sources,
        methods=["lscm"],
        scopes=["sheet"],
        backends={},
        resources=dict(
            reserve_bytes=0, seconds=60.0, cache_bytes=1 << 24, decoded_cache_bytes=1 << 20
        ),
    )
    run = tmp_path / "run"
    bench.inventory(config, run)
    probes = iter([dict(slice="field-flattening-test.slice"), ValueError("host RAM headroom fell")])

    def envelope():
        result = next(probes)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(bench, "require_benchmark_envelope", envelope)
    queue = bench.run_queue(run, "pilot", workers=True)
    assert [row["status"] for row in queue["rows"]] == ["PAUSED", "PENDING"]
    assert not list(run.glob("cells/**/WORKER-REQUEST.json"))


def test_supervisor_deadline_kills_worker_that_never_checks_its_guard(bench, tmp_path):
    import os
    import time

    pidfile = tmp_path / "worker.pid"
    command = [
        sys.executable,
        "-c",
        f"import os,pathlib,time;pathlib.Path({str(pidfile)!r}).write_text(str(os.getpid()));time.sleep(60)",
    ]
    started = time.monotonic()
    with pytest.raises(TimeoutError, match="wall time"):
        bench.supervise(command, lambda: None, timeout=0.4)
    assert time.monotonic() - started < 5.0
    pid = int(pidfile.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


@pytest.mark.skipif(
    sys.platform != "linux" or os.environ.get("FLATTENING_SYSTEMD_TESTS") != "1",
    reason="set FLATTENING_SYSTEMD_TESTS=1 on a privileged Linux qualification host",
)
def test_systemd_deadline_reaps_detached_blocking_solver(bench, tmp_path):
    import os
    import time
    import uuid

    pidfile = tmp_path / "solver.pid"
    worker = (
        "import subprocess,sys,pathlib,time;"
        "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],start_new_session=True);"
        f"pathlib.Path({str(pidfile)!r}).write_text(str(p.pid));time.sleep(60)"
    )
    started = time.monotonic()
    try:
        with pytest.raises(TimeoutError, match="wall time"):
            bench.supervise(
                [sys.executable, "-c", worker],
                lambda: None,
                timeout=1.0,
                unit="field-flattening-cell-deadline-" + uuid.uuid4().hex,
            )
        assert time.monotonic() - started < 10.0
        pid = int(pidfile.read_text())
        stat = Path(f"/proc/{pid}/stat")
        assert not stat.exists() or stat.read_text().rsplit(")", 1)[1].split()[0] == "Z"
    finally:
        if pidfile.exists():
            try:
                os.kill(int(pidfile.read_text()), 9)
            except ProcessLookupError:
                pass


@pytest.mark.parametrize("already_spent", [False, True])
def test_queue_wall_deadline_preserves_spent_time_and_resource_status(
    bench, tmp_path, monkeypatch, already_spent
):
    import time

    sources = []
    for field in ("175a", "paris4"):
        source, _, _ = source_mesh(tmp_path / field, field, 1.0)
        sources.append(dict(source, stage="pilot"))
    config = dict(
        sources=sources,
        methods=["lscm"],
        scopes=["sheet"],
        backends={},
        resources=dict(
            reserve_bytes=0, seconds=0.3, cache_bytes=1 << 24, decoded_cache_bytes=1 << 20
        ),
    )
    run = tmp_path / "run"
    bench.inventory(config, run)
    queue = bench.read_json(run / "QUEUE.json")
    for i, row in enumerate(queue["rows"]):
        path = run / row["output"]
        path.mkdir(parents=True)
        atomic_json(path / "TIME.json", dict(active_seconds=0.4 if already_spent or i else 0.2))
    monkeypatch.setattr(
        bench, "require_benchmark_envelope", lambda: dict(slice="field-flattening-test.slice")
    )

    def blocking_service(command, guard, *, timeout, **kwargs):
        assert not already_spent
        assert timeout == pytest.approx(0.1)
        time.sleep(timeout + 0.01)
        raise TimeoutError("parent cell wall time budget exhausted")

    monkeypatch.setattr(bench, "supervise", blocking_service)
    result = bench.run_queue(run, "pilot", workers=True)
    assert all(row["status"] == "RESOURCE_LIMIT" for row in result["rows"])
    assert all(
        bench.read_json(run / row["output"] / "TIME.json")["active_seconds"] >= 0.3
        for row in result["rows"]
    )
    if already_spent:
        assert not list(run.glob("cells/**/WORKER-REQUEST.json"))


@pytest.mark.parametrize("kind", ["parent_pause", "worker_pause", "crash"])
def test_forced_worker_exit_keeps_elapsed_budget(bench, tmp_path, monkeypatch, kind):
    import subprocess

    sources = [
        dict(source_mesh(tmp_path / field, field, 1.0)[0], stage="pilot")
        for field in ("175a", "paris4")
    ]
    config = dict(
        sources=sources,
        methods=["lscm"],
        scopes=["sheet"],
        backends={},
        resources=dict(
            reserve_bytes=0, seconds=60.0, cache_bytes=1 << 24, decoded_cache_bytes=1 << 20
        ),
    )
    run = tmp_path / "run"
    bench.inventory(config, run)
    queue = bench.read_json(run / "QUEUE.json")
    queue["rows"] = queue["rows"][:1]
    atomic_json(run / "QUEUE.json", queue)
    output = run / queue["rows"][0]["output"]
    output.mkdir(parents=True)
    atomic_json(output / "TIME.json", dict(active_seconds=0.25))
    clock = [3.0]
    monkeypatch.setattr(bench.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        bench, "require_benchmark_envelope", lambda: dict(slice="field-flattening-test.slice")
    )

    def terminate(command, guard, **kwargs):
        clock[0] = 5.0
        if kind == "parent_pause":
            raise InterruptedError("protected inference stale")
        if kind == "worker_pause":
            atomic_json(
                output / "WORKER-EXIT.json",
                dict(
                    status="PAUSED",
                    reason="STOP requested",
                    request_sha256=bench.digest(bench.read_json(output / "WORKER-REQUEST.json")),
                ),
            )
        raise subprocess.CalledProcessError(75 if kind == "worker_pause" else 1, command)

    monkeypatch.setattr(bench, "supervise", terminate)
    result = bench.run_queue(run, "pilot", workers=True)
    assert result["rows"][0]["status"] == ("FAILED" if kind == "crash" else "PAUSED")
    assert bench.read_json(output / "TIME.json")["active_seconds"] == pytest.approx(2.25)


@pytest.mark.parametrize("unknown", ["source_vertex_ids", "source_face_ids"])
def test_legacy_ids_below_first_source_id_cannot_alias_valid_geometry(bench, tmp_path, unknown):
    request = dict(cell_request(bench, tmp_path), method="legacy-lscm", legacy_unit_um=1.0)
    arrays = bench.case_arrays(request["case"])
    data = dict(
        world=arrays["world_um"],
        faces=arrays["faces"],
        uv=arrays["world_um"][:, :2],
        source_vertex_ids=arrays["source_vertex_ids"].copy(),
        source_face_ids=arrays["source_face_ids"].copy(),
    )
    assert data[unknown].min() == 1
    data[unknown][0] = 0
    path = tmp_path / "legacy.npz"
    np.savez(path, **data)
    request.update(legacy_map=str(path), legacy_map_sha256=file_sha(path))
    with pytest.raises(ValueError, match="legacy source IDs unknown"):
        bench.run_cell(request, tmp_path / "cell")
    assert not (tmp_path / "cell/map/MAP.json").exists()
