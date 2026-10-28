"""Frozen CT visualization regressions migrated from the numerical benchmark."""

import json
from pathlib import Path

import numpy as np
import pytest

from winding_field.cli import main
from winding_field.util import file_sha as sha256


def arguments(case, name="run"):
    return ["ct-visuals", "--case", str(case["case"]), "--field", str(case["field"]),
            "--metadata", str(case["metadata"]), "--output", str(case["root"] / name)]


@pytest.fixture
def visual_case(tmp_path):
    frame = "PHercParis4/20260411134726:L2"
    field = np.broadcast_to(np.arange(80, dtype=float) / 30, (5, 6, 80))
    field_path = tmp_path / "field.npy"
    np.save(field_path, field)
    meta = dict(frame=frame, voxel_um=9.6, origin_xyz=[0, 0, 0], spacing_xyz=[1, 1, 1],
                shape_zyx=list(field.shape), step=2.0, field_sha256=sha256(field_path))
    metadata = tmp_path / "field.npy.json"
    metadata.write_text(json.dumps(meta))
    archive = tmp_path / "visual-archive"
    archive.mkdir()
    ct = (np.arange(field.size).reshape(field.shape) % 256).astype(np.uint8)
    receipt = dict(frame=frame, step=2.0, voxel_um=9.6, ct_level=2, native_l0_scale=4,
                   origin_xyz=[0, 0, 0], size_xyz=list(field.shape[::-1]), ct_url="synthetic",
                   input_ct={}, field_sha256={"coarse": "b" * 64}, ct_window=[0, 255],
                   contour_levels=[-1, 0, 1, 2, 3], planes=[], synthetic=True)
    for name, axis in [("XY", 2), ("XZ", 1), ("YZ", 0)]:
        index = field.shape[2 - axis] // 2
        ct_plane = np.take(ct, index, axis=2 - axis)
        reference = np.take(field, index, axis=2 - axis)
        np.savez_compressed(archive / (name + "-samples.npz"), ct=ct_plane, coarse=reference,
                            coarse_valid=np.ones(reference.shape, bool))
        receipt["planes"].append(dict(name=name, fixed_axis_xyz=axis, fixed_position_l2=index,
                                      ct_pixels=ct_plane.size, coarse_covered=ct_plane.size))
    (archive / "RECEIPT.json").write_text(json.dumps(receipt))
    manifest = archive / "MANIFEST.json"
    manifest.write_text(json.dumps({p.name: sha256(p) for p in archive.iterdir()}))
    descriptor = dict(schema="winding-benchmark-case-v1", frame=frame, step=2.0, voxel_um=9.6,
                      visualization=dict(schema="ct-winding-overlays-v1",
                          archive_manifest=dict(path=str(manifest), sha256=sha256(manifest)),
                          reference_key="coarse", reference_label="Frozen reference"))
    case = tmp_path / "case.json"
    case.write_text(json.dumps(descriptor))
    return dict(root=tmp_path, case=case, field=field_path, metadata=metadata, archive=archive)


def test_cli_emits_matched_ct_views_from_the_actual_candidate(visual_case):
    from PIL import Image

    assert main(arguments(visual_case)) == 0
    output = visual_case["root"] / "run"
    for name in ["ct-winding.png", "ct-winding-XY.png", "ct-winding-XZ.png", "ct-winding-YZ.png"]:
        with Image.open(output / name) as image:
            assert image.width > 500 and image.height > 300
            image.verify()
    report = json.loads((output / "VISUALS.json").read_text())
    assert report["contour_levels"] == [-1, 0, 1, 2, 3] and report["ct_window"] == [0, 255]
    field = np.load(visual_case["field"])
    with np.load(output / "ct-winding-samples.npz") as samples:
        for name, axis in [("XY", 2), ("XZ", 1), ("YZ", 0)]:
            expected = np.take(field, field.shape[2 - axis] // 2, axis=2 - axis) / 2
            np.testing.assert_allclose(samples[name + "_candidate"], expected)
            with np.load(visual_case["archive"] / (name + "-samples.npz")) as frozen:
                np.testing.assert_array_equal(samples[name + "_ct"], frozen["ct"])
                np.testing.assert_array_equal(samples[name + "_reference"], frozen["coarse"])
    manifest = json.loads((output / "MANIFEST.json").read_text())
    assert manifest["ct-winding.png"] == sha256(output / "ct-winding.png")
    assert manifest["VISUALS.json"] == sha256(output / "VISUALS.json")
    before = (output / "VISUALS.json").read_bytes()
    assert main(arguments(visual_case)) == 2
    assert (output / "VISUALS.json").read_bytes() == before


def test_visuals_preserve_candidate_gauge_and_frozen_contour_range(visual_case):
    field = np.load(visual_case["field"]) + 20
    np.save(visual_case["field"], field)
    meta = json.loads(visual_case["metadata"].read_text())
    meta["field_sha256"] = sha256(visual_case["field"])
    visual_case["metadata"].write_text(json.dumps(meta))
    assert main(arguments(visual_case)) == 0
    output = visual_case["root"] / "run"
    with np.load(output / "ct-winding-samples.npz") as samples:
        np.testing.assert_allclose(samples["XY_candidate"], field[2] / 2)
    report = json.loads((output / "VISUALS.json").read_text())
    assert report["contour_levels"] == [-1, 0, 1, 2, 3]
    assert all(p["candidate_outside_contour_range"] == p["ct_pixels"] for p in report["planes"])


@pytest.mark.parametrize("problem", ["hash", "frame", "step", "shape", "levels"])
def test_visual_preflight_rejects_bad_ct_or_geometry(visual_case, problem):
    archive = visual_case["archive"]
    receipt_path = archive / "RECEIPT.json"
    receipt = json.loads(receipt_path.read_text())
    if problem == "hash":
        path = archive / "XY-samples.npz"
        path.write_bytes(path.read_bytes() + b"changed")
    elif problem == "shape":
        np.savez_compressed(archive / "XY-samples.npz", ct=np.zeros((2, 2), np.uint8), coarse=np.zeros((2, 2)))
    else:
        if problem == "frame":
            receipt["frame"] = "another-scan"
        elif problem == "step":
            receipt["step"] = 99
        else:
            receipt["contour_levels"] = [0, float("nan"), 2]
        receipt_path.write_text(json.dumps(receipt))
    if problem != "hash":
        manifest = archive / "MANIFEST.json"
        manifest.write_text(json.dumps({p.name: sha256(p) for p in archive.iterdir() if p != manifest}))
        descriptor = json.loads(visual_case["case"].read_text())
        descriptor["visualization"]["archive_manifest"]["sha256"] = sha256(manifest)
        visual_case["case"].write_text(json.dumps(descriptor))
    assert main(arguments(visual_case)) == 2
    assert not (visual_case["root"] / "run/SUMMARY.json").exists()


def test_visual_preflight_only_generates_no_figures(visual_case):
    assert main(arguments(visual_case) + ["--check-only"]) == 0
    output = visual_case["root"] / "run"
    receipt = json.loads((output / "PREFLIGHT.json").read_text())
    assert "visual_manifest" in receipt["inputs"]
    assert not list(output.glob("*.png"))


def test_visuals_keep_unsupported_candidate_samples_missing(visual_case):
    support = np.ones((5, 6, 80), bool)
    support[2, 2, 1] = False
    path = visual_case["root"] / "support.npy"
    np.save(path, support)
    meta = json.loads(visual_case["metadata"].read_text())
    meta["support_sha256"] = sha256(path)
    visual_case["metadata"].write_text(json.dumps(meta))
    assert main(arguments(visual_case) + ["--support", str(path)]) == 0
    with np.load(visual_case["root"] / "run/ct-winding-samples.npz") as samples:
        assert np.isnan(samples["XY_candidate"][2, 1])
        assert not samples["XY_candidate_valid"][2, 1]
        assert np.isfinite(samples["XY_reference"][2, 1])


def test_empty_visual_coverage_is_explicit_without_a_reference_column(visual_case):
    descriptor = json.loads(visual_case["case"].read_text())
    descriptor["visualization"].pop("reference_key")
    visual_case["case"].write_text(json.dumps(descriptor))
    meta = json.loads(visual_case["metadata"].read_text())
    meta["origin_xyz"] = [1000, 1000, 1000]
    visual_case["metadata"].write_text(json.dumps(meta))
    assert main(arguments(visual_case)) == 0
    output = visual_case["root"] / "run"
    report = json.loads((output / "VISUALS.json").read_text())
    assert report["reference_field_sha256"] is None
    assert all(p["candidate_covered"] == 0 and p["candidate_range"] is None for p in report["planes"])
    with np.load(output / "ct-winding-samples.npz") as samples:
        assert np.isnan(samples["XY_candidate"]).all() and "XY_reference" not in samples


def test_visual_archive_replacement_during_read_cannot_change_frozen_settings(visual_case, monkeypatch):
    manifest = visual_case["archive"] / "MANIFEST.json"
    receipt = visual_case["archive"] / "RECEIPT.json"
    original_open = Path.open
    swapped = False

    def open_with_replacement(path, *args, **kwargs):
        nonlocal swapped
        mode = args[0] if args else kwargs.get("mode", "r")
        if path == manifest and "r" in mode and not swapped:
            swapped = True
            with original_open(receipt) as stream:
                data = json.load(stream)
            data["contour_levels"] = [10, 11, 12]
            with original_open(receipt, "w") as stream:
                json.dump(data, stream)
            with original_open(manifest) as stream:
                hashes = json.load(stream)
            hashes["RECEIPT.json"] = sha256(receipt)
            with original_open(manifest, "w") as stream:
                json.dump(hashes, stream)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_with_replacement)
    assert main(arguments(visual_case)) == 2
    assert swapped and not (visual_case["root"] / "run/SUMMARY.json").exists()


@pytest.mark.parametrize("identity", [None, "", "not-a-field-hash"])
def test_visual_reference_requires_a_source_field_identity(visual_case, identity):
    archive = visual_case["archive"]
    receipt = archive / "RECEIPT.json"
    data = json.loads(receipt.read_text())
    data["field_sha256"]["coarse"] = identity
    receipt.write_text(json.dumps(data))
    manifest = archive / "MANIFEST.json"
    hashes = json.loads(manifest.read_text())
    hashes[receipt.name] = sha256(receipt)
    manifest.write_text(json.dumps(hashes))
    case = json.loads(visual_case["case"].read_text())
    case["visualization"]["archive_manifest"]["sha256"] = sha256(manifest)
    visual_case["case"].write_text(json.dumps(case))
    assert main(arguments(visual_case) + ["--check-only"]) == 2


@pytest.mark.parametrize("problem", ["hash", "frame", "step", "spacing", "shape", "nonfinite", "missing_support", "support_hash", "solver", "pilot"])
def test_visual_command_rejects_invalid_candidate(visual_case, problem):
    meta = json.loads(visual_case["metadata"].read_text())
    extra = []
    if problem == "hash":
        meta["field_sha256"] = "0" * 64
    elif problem == "frame":
        meta["frame"] = "wrong"
    elif problem == "step":
        meta["step"] = 0
    elif problem == "spacing":
        meta["spacing_xyz"] = [0, 1, 1]
    elif problem == "shape":
        meta["shape_zyx"] = [1, 2, 3]
    elif problem == "solver":
        meta["solver"] = {"converged": False}
    elif problem == "pilot":
        meta["pilot_gate_status"] = "FAIL"
    elif problem == "nonfinite":
        field = np.load(visual_case["field"])
        field[0, 0, 0] = np.nan
        np.save(visual_case["field"], field)
        meta["field_sha256"] = sha256(visual_case["field"])
    else:
        meta["support_sha256"] = "0" * 64
        if problem == "support_hash":
            path = visual_case["root"] / "support.npy"
            np.save(path, np.ones((5, 6, 80), bool))
            extra = ["--support", str(path)]
    visual_case["metadata"].write_text(json.dumps(meta))
    assert main(arguments(visual_case) + extra) == 2
    assert not (visual_case["root"] / "run/SUMMARY.json").exists()


@pytest.mark.parametrize("frame", [None, "", "   "])
def test_visual_command_rejects_matching_but_undeclared_frames(visual_case, frame):
    for path in (visual_case["case"], visual_case["metadata"], visual_case["archive"] / "RECEIPT.json"):
        value = json.loads(path.read_text())
        value["frame"] = frame
        path.write_text(json.dumps(value))
    manifest = visual_case["archive"] / "MANIFEST.json"
    hashes = json.loads(manifest.read_text())
    hashes["RECEIPT.json"] = sha256(visual_case["archive"] / "RECEIPT.json")
    manifest.write_text(json.dumps(hashes))
    case = json.loads(visual_case["case"].read_text())
    case["visualization"]["archive_manifest"]["sha256"] = sha256(manifest)
    visual_case["case"].write_text(json.dumps(case))
    assert main(arguments(visual_case) + ["--check-only"]) == 2


def test_visual_command_preserves_calibration_roundoff_tolerance(visual_case):
    meta = json.loads(visual_case["metadata"].read_text())
    meta["step"] = 2.0000000000000004
    visual_case["metadata"].write_text(json.dumps(meta))
    assert main(arguments(visual_case) + ["--check-only"]) == 0


@pytest.mark.parametrize("problem", [None, "frame", "units"])
def test_legacy_visual_case_uses_explicit_hashed_configuration(visual_case, problem):
    from test_surface_dataset import configuration

    case = json.loads(visual_case["case"].read_text())
    unit = case.pop("voxel_um")
    visual_case["case"].write_text(json.dumps(case))
    config = configuration("different" if problem == "frame" else case["frame"])
    config["voxel_um"] = unit if problem != "units" else 12
    config["afv"]["native_voxel_um"] = (np.array(config["afv"]["scale_xyz"]) * config["voxel_um"]).tolist()
    path = visual_case["root"] / "dataset.json"
    path.write_text(json.dumps(config))
    before = {p: p.read_bytes() for p in (path, visual_case["case"], visual_case["metadata"])}
    assert main(arguments(visual_case) + ["--config", str(path), "--check-only"]) == (0 if problem is None else 2)
    assert all(p.read_bytes() == raw for p, raw in before.items())
    if problem is None:
        receipt = json.loads((visual_case["root"] / "run/PREFLIGHT.json").read_text())
        assert receipt["inputs"]["dataset_config"]["sha256"] == sha256(path)


def test_figures_are_visibly_labeled_synthetic(visual_case, monkeypatch):
    from matplotlib.figure import Figure

    titles = []
    savefig = Figure.savefig

    def save_and_inspect(figure, *args, **kwargs):
        titles.append(([text.get_text() for text in figure.texts], figure.axes[0].get_title()))
        return savefig(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", save_and_inspect)
    assert main(arguments(visual_case)) == 0
    assert len(titles) == 4
    assert all(any("synthetic CT-like data" in title for title in figure_titles) for figure_titles, _ in titles)
    assert all("Synthetic CT-like data" in panel_title for _, panel_title in titles)
