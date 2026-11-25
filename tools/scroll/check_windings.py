"""Winding count of a coarse field vs papyrus layers in level-0 CT, on a cross-section
perpendicular to the calibrated scroll axis (local caches only, offline).

usage: check-windings2.py <coarse run> <ct url> <ct cache L0> <voxel um>
"""

import json
import sys

import numpy as np
from scipy.ndimage import gaussian_filter1d, map_coordinates
from scipy.signal import find_peaks

from winding_field.field import open_field
from winding_field.workflow import open_volume

run, url, cache, vox = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
g = json.load(open(run + "/FIELD.json"))["geometry"]
cal = json.load(open(run + "/CALIBRATION.json"))
with open_field(run) as s:
    W, ok = s.read([0, 0, 0], s.shape)
ct = open_volume(
    dict(source=url, voxel_um=vox, level=0, cache=cache, cache_bytes=450 << 30, offline=True)
)
o = np.array(g["origin_xyz"])
sp = g["spacing_xyz"][0]
c = np.array(cal["centre_xyz"]) / vox  # L0 voxels, xyz
cx, cy, cz = c.round().astype(int)
print(
    f"axis {np.round(cal['axis_xyz'], 3)}; centre voxel x{cx} y{cy} z{cz}; Lasagna mean spacing {1 / cal['mean_density_per_um']:.0f} um",
    flush=True,
)

# Cross-section at y = cy (the axis is mostly +Y): CT averaged over 5 voxels in y.
hz = (g["shape_zyx"][0] * sp + o[2]) / vox
hx = (g["shape_zyx"][2] * sp + o[0]) / vox
z0, z1 = int(max(0, o[2] / vox)), int(min(ct.shape[0], hz))
x0, x1 = int(max(0, o[0] / vox)), int(min(ct.shape[2], hx))
slab = ct.read([z0, cy - 2, x0], [z1, cy + 3, x1]).astype(np.float32).mean(axis=1)  # z, x
print(f"CT slab {slab.shape} read", flush=True)
iy = (cy * vox - o[1]) / sp
Wy = W[:, int(round(iy)), :].astype(np.float64)
oky = ok[:, int(round(iy)), :]

tot_peaks = tot_dw = 0.0
rows = []
for k in range(24):
    th = 2 * np.pi * k / 24
    d = np.array([np.cos(th), np.sin(th)])  # (x, z)
    t = np.arange(0, 6000)  # voxels
    px, pz = cx + t * d[0], cz + t * d[1]
    inside = (px >= x0) & (px < x1 - 1) & (pz >= z0) & (pz < z1 - 1)
    px, pz, t = px[inside], pz[inside], t[inside]
    prof = map_coordinates(slab, [pz - z0, px - x0], order=1)
    fz, fx = (pz * vox - o[2]) / sp, (px * vox - o[0]) / sp
    w = map_coordinates(Wy, [fz, fx], order=1)
    m = map_coordinates(oky.astype(np.float32), [fz, fx], order=0) > 0.5
    # observed runs at least 1.5 mm long
    edges = np.flatnonzero(np.diff(np.r_[0, m.astype(int), 0]))
    for a, b in zip(edges[::2], edges[1::2]):
        if (b - a) * vox < 1500:
            continue
        seg = gaussian_filter1d(prof[a:b], 1.0)
        detr = seg - gaussian_filter1d(seg, 10.0)
        peaks, _ = find_peaks(detr, distance=5, prominence=max(1.5, 0.3 * detr.std()))
        dw = abs(w[b - 1] - w[a])
        tot_peaks += len(peaks)
        tot_dw += dw
        rows.append((k * 15, a * vox / 1000, b * vox / 1000, len(peaks), dw))

for ang, a, b, n, dw in rows[:30]:
    print(
        f"rayo {ang:3d}°  {a:5.1f}-{b:5.1f} mm: capas CT {n:4d} | W sube {dw:6.1f} | W/capas {dw / max(n, 1):.2f}"
    )
L = sum(b - a for _, a, b, _, _ in rows)
print(
    f"TOTAL {len(rows)} tramos, {L:.0f} mm con soporte: capas CT {tot_peaks:.0f} (cada {L * 1000 / max(tot_peaks, 1):.0f} um), W sube {tot_dw:.0f} (cada {L * 1000 / max(tot_dw, 1e-9):.0f} um), razón W/capas {tot_dw / max(tot_peaks, 1):.2f}"
)
