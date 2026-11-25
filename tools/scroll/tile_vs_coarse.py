"""How far a fine tile's field is from the coarse prior, split into large and small scale.

usage: tile-vs-coarse.py <plan dir> <coarse run> <tile id> [<tile id> ...]
"""

import json
import sys

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

from winding_field.field import open_field

plan, coarse = sys.argv[1], sys.argv[2]
cg = json.load(open(coarse + "/FIELD.json"))["geometry"]
with open_field(coarse) as s:
    CW, Cok = s.read([0, 0, 0], s.shape)
co = np.array(cg["origin_xyz"][::-1])
cs = cg["spacing_xyz"][0]
for tid in sys.argv[3:]:
    run = f"{plan}/tiles/{tid}"
    meta = json.load(open(run + "/FIELD.json"))
    g = meta["geometry"]
    with open_field(run) as s:
        T, ok = s.read([0, 0, 0], s.shape)
    o = np.array(g["origin_xyz"][::-1])
    sp = g["spacing_xyz"][0]
    # sample every 4th node to keep it light
    T, ok = T[::4, ::4, ::4], ok[::4, ::4, ::4]
    idx = np.indices(T.shape).reshape(3, -1).T * 4
    pos = (o + idx * sp - co) / cs
    C = map_coordinates(CW, pos.T, order=1, mode="nearest").reshape(T.shape)
    Cm = (
        map_coordinates(Cok.astype(np.float32), pos.T, order=0, mode="nearest").reshape(T.shape)
        > 0.5
    )
    m = ok & Cm
    d = np.where(m, T - C, 0.0)
    d0 = np.median(d[m])
    d = np.where(m, d - d0, 0.0)
    w = m.astype(np.float64)
    sig = 500.0 / (sp * 4)  # 0.5 mm in sampled nodes
    low = gaussian_filter(d, sig) / np.maximum(gaussian_filter(w, sig), 1e-6)
    high = d - low
    a, lo, hi = np.abs(d[m]), np.abs(low[m]), np.abs(high[m])
    q = lambda x: " ".join(f"{v:5.2f}" for v in np.percentile(x, [50, 90, 99]))
    sw = json.load(open(run + "/field/CHECKPOINT.json"))["progress"].get("completed_sweeps", [])
    print(
        f"{tid}: nodos {m.sum():,}  |tile-coarse| p50/90/99 {q(a)} | escala >0.5mm {q(lo)} | escala <0.5mm {q(hi)} | "
        f"cambio por pasada {[round(c['max_core_update'], 2) for c in sw]}",
        flush=True,
    )
