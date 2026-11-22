"""Exact integer topology for nonempty triangle meshes, without structured row sorts."""

import numpy as np


def duplicate_faces(faces):
    rows = np.sort(faces, axis=1)
    a, b, c = rows.T
    rows = rows[np.lexsort((c, b, a))]
    return int(np.count_nonzero((rows[1:] == rows[:-1]).all(axis=1)))


def edge_groups(faces):
    """Sorted undirected edges, incidence counts, winding conflicts and directed-edge order."""
    a = faces.T.ravel()
    b = np.roll(a, -len(faces))
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    order = np.lexsort((hi, lo))
    lo, hi = lo[order], hi[order]
    starts = np.flatnonzero(np.r_[True, (lo[1:] != lo[:-1]) | (hi[1:] != hi[:-1])])
    counts = np.diff(starts, append=len(order))
    paired = starts[counts == 2]
    forward = (a < b)[order]
    inconsistent = int(np.count_nonzero(forward[paired] == forward[paired + 1]))
    return np.column_stack((lo[starts], hi[starts])), counts, inconsistent, order
