"""A completed field on the same grid as the fixed boundary of a new solve.

A tile solved on its own leaves its internal faces free. Where the data determine W
poorly, such a free face drifts whole windings away from what the neighbouring tiles find
and the overlap between them does not close. ``run --field-boundary RUN`` solves the same
tile with the outer layers of its internal faces held at RUN's values, where RUN (usually
the merge of the tiles' previous solutions) has observed nodes. The solve needs no gauge
node then, and starts from RUN instead of the coarse prior.
"""

import json
from pathlib import Path

import numpy as np

from .util import file_sha

RULE = "fixed-observed-layers-v1"
FACES = ("z-", "z+", "y-", "y+", "x-", "x+")


class FieldBoundary:
    def __init__(self, path, geometry, layers=2, source=None):
        from .field import open_field

        if type(layers) is not int or layers < 1:
            raise ValueError("field_boundary_layers must be a positive integer")
        self.run = Path(path).expanduser().resolve()
        meta_path = self.run / "FIELD.json"
        if not meta_path.is_file():
            raise ValueError(f"field boundary has no FIELD.json: {self.run}")
        meta = json.loads(meta_path.read_bytes())
        theirs = meta["geometry"]
        frame_match = None
        if theirs.get("frame") != geometry.get("frame"):
            frame_match = self._same_ct(meta, geometry, source)
        if float(theirs.get("sheet_step", 1.0)) != float(geometry.get("sheet_step", 1.0)):
            raise ValueError("field boundary sheet_step differs")
        spacing = np.asarray(geometry["spacing_xyz"], float)
        if not np.allclose(theirs["spacing_xyz"], spacing, rtol=1e-12, atol=0):
            raise ValueError("field boundary spacing differs from this grid")
        offset = (np.asarray(geometry["origin_xyz"], float) - theirs["origin_xyz"]) / spacing
        # ZYX node of RUN that holds this grid's first node.
        self.offset = np.rint(offset[::-1]).astype(np.int64)
        if not np.allclose(offset[::-1], self.offset, atol=1e-6):
            raise ValueError("field boundary nodes are not aligned with this grid")
        self.shape = np.asarray(geometry["shape_zyx"], np.int64)
        self.extent = np.asarray(theirs["shape_zyx"], np.int64)
        first, last = self.offset, self.offset + self.shape - 1
        if (last < 0).any() or (first > self.extent - 1).any():
            raise ValueError("field boundary does not overlap this grid")
        # A face is internal when RUN continues beyond it.
        self.internal = [
            (bool(0 < first[a] <= self.extent[a] - 1), bool(0 <= last[a] < self.extent[a] - 1))
            for a in range(3)
        ]
        for axis, sides in enumerate(self.internal):
            if self.shape[axis] <= layers * sum(sides):
                raise ValueError("field_boundary_layers leaves no free node across this grid")
        self.layers = layers
        self.observed = open_field(self.run)
        self._stored = None
        self._slabs = []
        counts = {}
        for name, lo, hi in self._layer_boxes():
            values, seen = self._read(self.observed, lo, hi)
            values[~seen] = np.nan
            self._slabs.append((lo, hi, values))
            counts[name] = int(seen.sum())
        if not sum(counts.values()):
            self.close()
            raise ValueError(
                "field boundary fixes no node: it has no observed value on the internal "
                "faces of this grid"
            )
        self.identity = dict(
            rule=RULE,
            field_json_sha256=file_sha(meta_path),
            checkpoint_sha256=meta["checkpoint_sha256"],
            layers=layers,
            offset_zyx=self.offset.tolist(),
            fixed_nodes=counts,
        )
        if frame_match:
            self.identity["frame_match"] = frame_match
        self.receipt = dict(self.identity, run=str(self.run))

    def _same_ct(self, meta, geometry, source):
        """Accept a frame that differs only in the CT reader, as ``--field-prior`` does.

        A merged field has no CT identity of its own: it lists the frames of the tiles it
        accepted under that rule (FIELD.json ``frame_match``).
        """
        from .field_prior import ct_differences, frame_match

        theirs = meta["geometry"]
        accepted = (meta.get("frame_match") or {}).get("frames") or {}
        if geometry.get("frame") in accepted:
            return frame_match(via="field-boundary-frame-match")
        message = (
            f"field boundary frame {theirs.get('frame')!r} differs from {geometry.get('frame')!r}"
        )
        prepared = self.run / "PREPARED.json"
        mine = json.loads(prepared.read_bytes()).get("source") if prepared.is_file() else None
        if not isinstance(source, dict) or not isinstance(mine, dict):
            raise ValueError(message + " and a CT source identity is unavailable to compare")
        keys = ct_differences(source, mine)
        if keys:
            raise ValueError(message + "; CT identity differs in " + ", ".join(keys))
        return frame_match()

    def _layer_boxes(self):
        """Disjoint boxes (name, lo, hi) that together hold every layer node of the internal
        faces: each axis takes its slabs from what the previous axes left."""
        lo, hi = np.zeros(3, np.int64), self.shape.copy()
        for axis, sides in enumerate(self.internal):
            for side, internal in enumerate(sides):
                if not internal:
                    continue
                a, b = lo.copy(), hi.copy()
                if side:
                    a[axis] = hi[axis] - self.layers
                else:
                    b[axis] = lo[axis] + self.layers
                yield FACES[2 * axis + side], a, b
            lo[axis] += self.layers * sides[0]
            hi[axis] -= self.layers * sides[1]

    def _read(self, source, lo, hi):
        """RUN on this grid's nodes ``lo..hi``: values and support, False outside RUN."""
        lo, hi = np.asarray(lo, np.int64), np.asarray(hi, np.int64)
        values = np.zeros(tuple(hi - lo))
        seen = np.zeros(tuple(hi - lo), bool)
        a = np.maximum(lo + self.offset, 0)
        b = np.minimum(hi + self.offset, self.extent)
        if (b > a).all():
            crop, support = source.read(a.tolist(), b.tolist())
            target = tuple(
                slice(x - low, y - low) for x, y, low in zip(a - self.offset, b - self.offset, lo)
            )
            values[target] = crop
            seen[target] = support & np.isfinite(crop)
        return values, seen

    def close(self):
        self.observed.close()
        if self._stored is not None:
            self._stored.close()

    def fill(self, values, lo, hi):
        """Write the fixed layer values that fall inside ``lo..hi`` into ``values``."""
        lo, hi = np.asarray(lo), np.asarray(hi)
        for slab_lo, slab_hi, slab in self._slabs:
            a, b = np.maximum(lo, slab_lo), np.minimum(hi, slab_hi)
            if (b <= a).any():
                continue
            part = slab[tuple(slice(x - o, y - o) for x, y, o in zip(a, b, slab_lo))]
            target = values[tuple(slice(x - o, y - o) for x, y, o in zip(a, b, lo))]
            np.copyto(target, part, where=np.isfinite(part))
        return values

    def provider(self, base=None):
        """Fixed values for solve_blocks: ``base`` (NaN where free) with the layers on top."""

        def fixed(lo, hi):
            lo, hi = np.asarray(lo), np.asarray(hi)
            values = np.full(tuple(hi - lo), np.nan) if base is None else base(lo, hi)
            return self.fill(values, lo, hi)

        return fixed

    def initial(self, lo, hi, fallback):
        """RUN's stored values on nodes ``lo..hi``; ``fallback()`` where RUN stores none."""
        if self._stored is None:
            from .field import open_field

            self._stored = open_field(self.run, support_policy="stored")
        values, stored = self._read(self._stored, lo, hi)
        if stored.all():
            return values
        return np.where(stored, values, fallback())
