"""Write the native seed OBJ with exactly the legacy np.savetxt row formats."""

import numpy as np


def write_obj(stream, vertices, faces, uv=None):
    for rows, fmt, kind in (
        (vertices, "v %.17g %.17g %.17g\n", "v"),
        (uv, "vt %.17g %.17g\n", "vt"),
        (faces, "f %d %d %d\n" if uv is None else "f %d/%d %d/%d %d/%d\n", "f"),
    ):
        if rows is None:
            continue
        rows = np.asarray(rows)
        for first in range(0, len(rows), 4096):
            block = rows[first : first + 4096]
            if kind == "f":
                block = block + 1
                if uv is not None:
                    block = np.repeat(block, 2, axis=1)
            stream.write((fmt * len(block)) % tuple(block.ravel().tolist()))
