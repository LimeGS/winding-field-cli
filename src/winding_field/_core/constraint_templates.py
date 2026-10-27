"""Bounded static physical restrictions; current exterior values are never cached."""

import hashlib
import math
import weakref
from collections import OrderedDict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import coo_matrix, csr_matrix

from .field_constraints import QuadraticConstraints
from .physical_grid import point_stencils


def _integers(value: ArrayLike) -> NDArray[np.int64]:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iu" or (
        not isinstance(value, np.ndarray)
        and any(isinstance(v, (bool, np.bool_)) for v in np.asarray(value, dtype=object).flat)
    ):
        raise ValueError("integer coordinates/groups required")
    if raw.dtype.kind == "u" and raw.size and raw.max() > np.iinfo(np.int64).max:
        raise ValueError("indices exceed int64")
    return raw.astype(np.int64, copy=False)


def _real(value: ArrayLike) -> NDArray[np.float64]:
    if np.iscomplexobj(value):
        raise ValueError("real stencil coefficients required")
    result = np.asarray(value, np.float64)
    if not np.isfinite(result).all():
        raise ValueError("finite stencil coefficients required")
    return result


def _owned(array: NDArray[Any]) -> NDArray[Any]:
    # An immutable bytes owner breaks writable aliases as well as setting flags.
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


class _Template:
    def __init__(self, matrix, weights, targets, groups, outside_nodes, outside_coeff, outside_slots, width):
        self.columns, inverse = np.unique(matrix.indices, return_inverse=True)
        compact = csr_matrix((matrix.data, inverse, matrix.indptr), shape=(matrix.shape[0], len(self.columns)))
        self.compact = QuadraticConstraints(compact, weights, targets, groups)
        c = self.compact
        self.matrix = csr_matrix((c.matrix.data, self.columns[c.matrix.indices], c.matrix.indptr), shape=matrix.shape)
        self.nodes, self.coeff, self.slots, self.width = outside_nodes, outside_coeff, outside_slots, width
        self.live = None
        arrays = (
            self.columns,
            outside_nodes,
            outside_coeff,
            outside_slots,
            c.targets,
            self.matrix.data,
            self.matrix.indices,
            self.matrix.indptr,
            *c._structural_arrays(),
        )
        self.nbytes = 4096
        for array in arrays:
            array.setflags(write=False)
            while isinstance(array.base, np.ndarray):
                array = array.base
            self.nbytes += array.nbytes  # Conservatively charge shared backing buffers twice.

    def restrict(self, read_nodes):
        terms = np.zeros((self.matrix.shape[0], self.width), np.float64)
        if self.nodes.size:
            values = _real(read_nodes(self.nodes))
            if values.shape != self.nodes.shape:
                raise ValueError("exterior node reader returned a different shape")
            terms.ravel()[self.slots] = values * self.coeff
        targets = self.compact.targets - terms.sum(1)
        live = None if self.live is None else self.live()
        result = (self.compact if live is None else live).with_targets(targets)
        if live is None:
            result.matrix = self.matrix
            result.diagonal = np.zeros(self.matrix.shape[1], np.float64)
            result.diagonal[self.columns] = self.compact.diagonal
            result.diagonal.setflags(write=False)
        self.live = weakref.ref(result)
        return result


class StencilConstraints:
    """Immutable fixed-width rows with a single-caller byte-bounded template LRU.

    Nodes are flat global ZYX indices. Coefficients retain their original slot
    order (8 for samples, 16 for pairs). Nonnegative group labels eliminate one
    mean per group; -1 means a plain row. Caller weights already include any
    length/trace normalization. Supply complete groups, including exterior rows.

    Cache bytes include retained backing buffers and 4096 bytes/entry, never a
    strongly retained dense tile diagonal. Input snapshots, returned constraints
    and build/RHS temporaries are additional memory. No full-grid array is built.
    Construct a new object when geometry, rows, weights or support changes.
    """

    def __init__(
        self,
        nodes: ArrayLike,
        coefficients: ArrayLike,
        shape_zyx: Sequence[int],
        *,
        weights: ArrayLike = 1.0,
        targets: ArrayLike = 0.0,
        groups: ArrayLike | None = None,
        cache_bytes: int = 0,
    ):
        shape = _integers(shape_zyx)
        if shape.shape != (3,) or (shape < 1).any() or math.prod(map(int, shape)) > np.iinfo(np.intp).max:
            raise ValueError("invalid stencil grid")
        nodes, coefficients = _integers(nodes), _real(coefficients)
        if (
            nodes.ndim != 2
            or nodes.shape[1] == 0
            or coefficients.shape != nodes.shape
            or (nodes < 0).any()
            or (nodes >= math.prod(map(int, shape))).any()
        ):
            raise ValueError("invalid stencil node inventory")
        n = len(nodes)
        weights = np.broadcast_to(_real(weights), (n,))
        targets = np.broadcast_to(_real(targets), (n,))
        groups = np.full(n, -1, np.int64) if groups is None else _integers(groups)
        if groups.shape != (n,) or (groups < -1).any() or (weights < 0).any():
            raise ValueError("invalid stencil weights/groups")
        if type(cache_bytes) is not int or cache_bytes < 0:
            raise ValueError("nonnegative integer cache budget required")
        self.shape = tuple(map(int, shape))
        self._nodes, self._coeff, self._weights, self._targets, self._groups = map(
            _owned, (nodes, coefficients, weights, targets, groups)
        )
        identity = hashlib.sha256()
        for array in (shape, self._nodes, self._coeff, self._weights, self._targets, self._groups):
            identity.update(np.asarray(array.shape, np.int64).tobytes())
            identity.update(array.tobytes())
        self.digest = identity.hexdigest()
        self.max_cache_bytes = cache_bytes
        self.stats = dict(bytes=0, hits=0, misses=0, evictions=0)
        self._entries: OrderedDict[tuple[tuple[int, ...], tuple[int, ...]], tuple[_Template | None, int]] = (
            OrderedDict()
        )

    def save(self, path: str | Path) -> None:
        """Write immutable row snapshots to a new NPZ at exactly ``path``.

        Existing files/directories/symlinks are never overwritten. On failure,
        remove only this call's newly created partial archive and re-raise.
        The parent directory must already exist; no template cache is saved.
        """
        path = Path(path)
        stream = path.open("xb")
        try:
            with stream:
                np.savez_compressed(
                    stream,
                    nodes=self._nodes,
                    coefficients=self._coeff,
                    shape_zyx=np.array(self.shape, np.int64),
                    weights=self._weights,
                    targets=self._targets,
                    groups=self._groups,
                )
        except BaseException:
            path.unlink(missing_ok=True)
            raise

    @classmethod
    def load(cls, path: str | Path | BinaryIO, *, cache_bytes: int = 0) -> "StencilConstraints":
        """Load exactly the six numeric NPZ row members, without pickle.

        Arrays are validated/snapshotted by the constructor. All rows, including
        complete exterior groups, remain resident; ``cache_bytes`` bounds only
        subsequently built restriction templates and does not affect identity.
        An already-open binary stream can bind verification and loading to one file.
        """
        archive = np.load(path, allow_pickle=False)
        if not isinstance(archive, np.lib.npyio.NpzFile):
            raise ValueError("expected a stencil NPZ archive")
        with archive:
            keys = {"nodes", "coefficients", "shape_zyx", "weights", "targets", "groups"}
            if len(archive.files) != len(keys) or set(archive.files) != keys:
                raise ValueError("stencil NPZ must contain exactly the six row members")
            arrays = {key: archive[key] for key in keys}
            if any(array.dtype.kind not in "iuf" for array in arrays.values()):
                raise ValueError("stencil NPZ members must have real numeric dtypes")
            return cls(
                arrays["nodes"],
                arrays["coefficients"],
                arrays["shape_zyx"].tolist(),
                weights=arrays["weights"],
                targets=arrays["targets"],
                groups=arrays["groups"],
                cache_bytes=cache_bytes,
            )

    @classmethod
    def pairs(
        cls,
        a_xyz: ArrayLike,
        b_xyz: ArrayLike,
        origin_xyz: ArrayLike,
        spacing_xyz: ArrayLike,
        shape_zyx: Sequence[int],
        *,
        weight: ArrayLike = 1.0,
        targets: ArrayLike = 0.0,
        cache_bytes: int = 0,
    ) -> "StencilConstraints":
        if np.asarray(a_xyz).shape != np.asarray(b_xyz).shape:
            raise ValueError("pair endpoints differ")
        a, wa, va = point_stencils(a_xyz, origin_xyz, spacing_xyz, shape_zyx)
        b, wb, vb = point_stencils(b_xyz, origin_xyz, spacing_xyz, shape_zyx)
        if not (va.all() and vb.all()):
            raise ValueError("pair outside physical grid support")
        return cls(
            np.concatenate([a, b], axis=1),
            np.concatenate([wa, -wb], axis=1),
            shape_zyx,
            weights=weight,
            targets=targets,
            cache_bytes=cache_bytes,
        )

    @classmethod
    def consensus(
        cls,
        points_xyz: ArrayLike,
        groups: ArrayLike,
        length_weights: ArrayLike,
        origin_xyz: ArrayLike,
        spacing_xyz: ArrayLike,
        shape_zyx: Sequence[int],
        *,
        weight: ArrayLike = 1.0,
        cache_bytes: int = 0,
    ) -> "StencilConstraints":
        nodes, coefficients, valid = point_stencils(points_xyz, origin_xyz, spacing_xyz, shape_zyx)
        if not valid.all():
            raise ValueError("sample outside physical grid support")
        return cls(
            nodes,
            coefficients,
            shape_zyx,
            weights=_real(length_weights) * _real(weight),
            groups=groups,
            cache_bytes=cache_bytes,
        )

    def _build(self, lo, hi):
        coords = np.array(np.unravel_index(self._nodes.ravel(), self.shape)).T.reshape(*self._nodes.shape, 3)
        inside = ((coords >= lo) & (coords < hi)).all(-1) & (self._coeff != 0)
        touched = inside.any(1)
        if not touched.any():
            return None
        groups = self._groups
        selected_groups = groups[touched & (groups >= 0)]
        rows = np.flatnonzero(touched | ((groups >= 0) & np.isin(groups, selected_groups)))
        inside, coords = inside[rows], coords[rows]
        coefficients, nodes = self._coeff[rows], self._nodes[rows]
        row_ids = np.broadcast_to(np.arange(len(rows))[:, None], inside.shape)
        columns = np.ravel_multi_index((coords[inside] - lo).T, hi - lo)
        matrix = coo_matrix(
            (coefficients[inside], (row_ids[inside], columns)), shape=(len(rows), math.prod(map(int, hi - lo)))
        ).tocsr()
        matrix.eliminate_zeros()
        outside = ~inside & (coefficients != 0)
        return _Template(
            matrix,
            self._weights[rows],
            self._targets[rows],
            groups[rows],
            nodes[outside],
            coefficients[outside],
            np.flatnonzero(outside),
            self._nodes.shape[1],
        )

    def restrict(
        self, lo: Sequence[int], hi: Sequence[int], read_nodes: Callable[[NDArray[np.int64]], ArrayLike]
    ) -> QuadraticConstraints | None:
        low, high = _integers(lo), _integers(hi)
        if (
            low.shape != (3,)
            or high.shape != (3,)
            or (low < 0).any()
            or (high > self.shape).any()
            or (high <= low).any()
        ):
            raise ValueError("restriction bounds outside grid")
        key = (tuple(map(int, low)), tuple(map(int, high)))
        if key in self._entries:
            self.stats["hits"] += 1
            template, _ = self._entries[key]
            self._entries.move_to_end(key)
        else:
            self.stats["misses"] += 1
            template = self._build(low, high)
            size = 4096 if template is None else template.nbytes
            if size <= self.max_cache_bytes:
                while self.stats["bytes"] + size > self.max_cache_bytes:
                    _, (_, retired) = self._entries.popitem(last=False)
                    self.stats["bytes"] -= retired
                    self.stats["evictions"] += 1
                self._entries[key] = template, size
                self.stats["bytes"] += size
        return None if template is None else template.restrict(read_nodes)
