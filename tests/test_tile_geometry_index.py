import numpy as np
import pytest

from winding_field.flattening import patches
from winding_field.flattening import original_tifxyz
from winding_field.flattening.original import raster_candidates


def test_index_retains_halo_and_original_order_even_with_giant_faces():
    # A small index budget forces multiple source rows into a leaf.
    rng = np.random.default_rng(45)
    uv = rng.uniform(-113, 81, (900, 2))
    uv[:4] = [[-55.5, 14], [-55, 14], [-55.5, 13], [57, -23]]
    f = rng.integers(0, len(uv), (600, 3))
    f[:2] = [[0, 1, 2], [3, 1, 0]]
    index = patches.PixelBounds(f, uv, -113, 81, 0.5, [389, 389], lambda: None, budget=512)
    for row, col in [(0, 0), (132, 112), (256, 256), (380, 380)]:
        c = dict(
            width=9,
            height=9,
            pitch=0.5,
            u0=-113.0,
            v_top=81.0,
            global_u0=-113.0,
            global_v_top=81.0,
            column_offset=col,
            row_offset=row,
        )
        ids = list(index.candidates(c, lambda: None))
        assert ids == sorted(set(ids))
        assert set(raster_candidates(f, uv, c)) <= set(ids)


def test_tif_boundary_incidence_is_global_and_includes_nonmanifold_edges():
    uv = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 1.0], [1.0, -1.0], [1.0, 2.0]])
    faces = np.array([[0, 1, 2], [1, 0, 3], [0, 1, 4], [0, 2, 1]])
    edges = original_tifxyz.domain_edges(faces, uv, np.ones(4, bool), lambda: None)
    assert edges.tolist() == [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [1, 4]]
    edges = original_tifxyz.domain_edges(faces, uv, [True, True, False, True], lambda: None)
    assert edges.tolist() == [[0, 2], [0, 3], [1, 2], [1, 3]]


@pytest.mark.parametrize("signed_indices", [False, True])
def test_boundary_edges_match_tuple_incidence_with_holes_and_seams(signed_indices):
    from collections import Counter

    rng = np.random.default_rng(54)
    uv = rng.normal(size=(31, 2))
    uv[7, 0] = np.nan
    f = rng.integers(-5 if signed_indices else 0, 31, size=(180, 3), dtype=np.int32)
    enabled = rng.random(len(f)) < 0.75
    expected = Counter()
    for face in f[enabled]:
        a, b, c = uv[face]
        if (b[0] - a[0]) * (c[1] - a[1]) > (b[1] - a[1]) * (c[0] - a[0]):
            expected.update(tuple(sorted(pair)) for pair in zip(face, np.roll(face, -1)))
    got = original_tifxyz.domain_edges(f, uv, enabled, lambda: None)
    assert list(map(tuple, got)) == sorted(edge for edge, count in expected.items() if count != 2)


def test_empty_index_skips_tiles():
    index = patches.PixelBounds(
        np.empty((0, 2), int), np.empty((0, 2)), 0, 0, 1, [70_000, 2], lambda: None
    )
    assert not list(
        index.candidates(
            {"column_offset": 65536, "row_offset": 0, "width": 5, "height": 2}, lambda: None
        )
    )


def test_indexed_tiles_match_exhaustive_render_and_skip_distant_clusters(tmp_path, monkeypatch):
    uv = np.array(
        [[-20.0, -3.0], [-12.0, -3.0], [-20.0, 5.0], [6.0, -3.0], [14.0, -3.0], [6.0, 5.0]]
    )
    faces = np.array([[0, 1, 2], [3, 4, 5], [0, 2, 1]], np.int32)
    # The first cluster is blocked by an inverted copy; the other still emits valid TIFXYZ.
    mapped = {
        "uv": uv,
        "world": np.column_stack((uv + 40, np.ones(6))),
        "faces": faces,
        "display": np.ones(3, bool),
        "source_face_rows": np.array([23, 47, 9]),
    }
    index = patches.PixelBounds(faces, uv, -20.0, 5.0, 1.0, [35, 9], lambda: None)
    c = {"column_offset": 24, "row_offset": 0, "width": 6, "height": 9}
    assert list(index.candidates(c, lambda: None)) == [1]

    def render(root, workers):
        return patches.render_map(
            mapped,
            root,
            lambda xyz: (xyz[:, 0], xyz[:, 1] != 40),
            {"voxel_um": 1.0},
            pitch_um=1.0,
            tile_size=7,
            export_tifxyz=True,
            raster_workers=workers,
        )

    actual = render(tmp_path / "indexed", 2)
    monkeypatch.setattr(
        patches.PixelBounds, "batches", lambda self, canvas, guard: [np.arange(self.count)]
    )
    assert actual == render(tmp_path / "exhaustive", 1)
