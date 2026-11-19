"""The single libigl LSCM call shared by every LSCM path.

Callers choose their own pins (they differ by design: farthest pair, boundary-based
pair); the second pin is placed at ``(length, 0)``. ``length`` defaults to the distance
between the pins in ``points`` and is passed explicitly where a caller measures it in a
different frame, so results stay bit-identical to the previous inline calls.
"""

import numpy as np


def lscm(points, faces, first, second, *, length=None, index_dtype=np.int64, solver=None):
    """Return ``(uv, pins, pin_uv)`` for a two-pin libigl LSCM solve.

    ``solver`` is the ``igl`` module to call (callers with an injectable solver pass it).
    """
    if solver is None:
        import igl as solver

    if length is None:
        length = np.linalg.norm(points[second] - points[first])
    pins = np.array([first, second], index_dtype)
    fixed = np.array([[0.0, 0.0], [length, 0.0]])
    uv, _ = solver.lscm(points, faces.astype(index_dtype), pins, fixed)
    return uv, pins, fixed
