"""Area-weighted distortion against immutable original faces, never UV pixels."""
from pathlib import Path
import numpy as np

from winding_field.flatten import cross2, iter_overlap_pairs
from winding_field.surface_atlas import face_components
from bench_flattening import case_arrays, locked, seal


def weighted_quantile(values, weights, q):
    values, weights = np.asarray(values), np.asarray(weights)
    if values.shape != weights.shape or not 0 <= q <= 1 or (weights < 0).any():
        raise ValueError('invalid weighted quantile')
    use = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not use.any():
        return None
    values, weights = values[use], weights[use]
    order = np.argsort(values, kind='stable')
    index = min(len(order) - 1, int(np.searchsorted(np.cumsum(weights[order]), q * weights.sum(), side='left')))
    return float(values[order[index]])


def artificial_seams(v, original, mapped):
    edges = lambda f: np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    original_edges, original_count = np.unique(edges(original), axis=0, return_counts=True)
    cut_edges, cut_count = np.unique(edges(mapped['faces']), axis=0, return_counts=True)
    parent_edges = np.sort(mapped['parent'][cut_edges[cut_count == 1]], axis=1)
    all_edges, inverse = np.unique(np.r_[original_edges, parent_edges], axis=0, return_inverse=True)
    counts = np.zeros(len(all_edges), int)
    counts[inverse[:len(original_edges)]] = original_count
    boundaries = np.bincount(inverse[len(original_edges):], minlength=len(all_edges))
    seams = all_edges[(counts == 2) & (boundaries > 0)]
    return float(np.linalg.norm(v[seams[:, 1]] - v[seams[:, 0]], axis=1).sum())


def measure(case, mapped, output, guard):
    source = case_arrays(case)
    v, f = source['world_um'], source['faces']
    rows, parent = np.asarray(mapped['source_face_rows']), np.asarray(mapped['parent'])
    faces, uv = np.asarray(mapped['faces']), np.asarray(mapped['uv'])
    if (rows.dtype.kind not in 'iu' or len(rows) != len(faces) or len(np.unique(rows)) != len(rows)
            or (len(rows) and (rows.min() < 0 or rows.max() >= len(f)))
            or not np.array_equal(mapped['world'], v[parent])
            or not np.array_equal(parent[faces], f[rows])):
        raise ValueError('source face ownership/correspondence changed')
    area = np.empty(len(f))
    for first in range(0, len(f), 4096):
        guard()
        tri = v[f[first:first + 4096]]
        area[first:first + len(tri)] = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) / 2
    if not np.isfinite(area).all() or (area <= 0).any():
        raise ValueError('source has degenerate/nonfinite faces')
    sigma = np.full((len(f), 2), np.nan)
    reasons = np.full(len(f), 1, np.uint8)  # omitted=1, invalid=2, overlap=4
    signed = np.full(len(faces), np.nan)
    for first in range(0, len(faces), 4096):
        guard()
        sl = slice(first, first + 4096)
        own = rows[sl]
        tri, t = mapped['world'][faces[sl]], uv[faces[sl]]
        e1, e2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
        length = np.linalg.norm(e1, axis=1)
        basis = np.zeros((len(tri), 2, 2))
        basis[:, 0, 0] = length
        basis[:, 0, 1] = np.sum(e1 * e2, axis=1) / length
        basis[:, 1, 1] = np.linalg.norm(np.cross(e1, e2), axis=1) / length
        matrix = np.stack([t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]], axis=-1)
        determinant = cross2(matrix[:, :, 0], matrix[:, :, 1])
        signed[sl] = determinant
        good = np.isfinite(matrix).all((1, 2)) & (determinant > 0)
        reasons[own] = np.where(good, 0, 2)
        sigma[own[good]] = np.linalg.svd(matrix[good] @ np.linalg.inv(basis[good]), compute_uv=False)
    finite = np.isfinite(uv[faces]).all((1, 2)) & (signed != 0)
    audit, involved = {}, np.zeros(len(f), bool)
    indices = np.flatnonzero(finite)
    positive = faces[indices].copy()
    positive[signed[indices] < 0] = positive[signed[indices] < 0][:, [0, 2, 1]]
    for a, b in iter_overlap_pairs(uv, positive, guard, max_candidates=262144, max_tests=30_000_000, report=audit):
        involved[rows[indices[[a, b]]]] = True
    if not audit.get('complete'):
        raise ValueError('overlap audit incomplete')
    reasons[involved] |= 4
    valid = reasons == 0
    total = float(area.sum())
    fraction = lambda mask: float(area[mask].sum() / total)
    error = np.max(abs(sigma - 1), axis=1)
    report = dict(source_area_mm2=total / 1e6, mapped_area_fraction=fraction((reasons & 1) == 0),
                  valid_area_fraction=fraction(valid), omitted_area_fraction=fraction((reasons & 1) != 0),
                  invalid_area_fraction=fraction((reasons & 2) != 0), overlap_area_fraction=fraction(involved),
                  overlap=audit, overlap_area_policy='whole affected source faces; conservative lower bound',
                  physical_acceptance=False, reason_bits=dict(omitted=1, invalid=2, overlap=4))
    weights = np.where(valid, area, 0.)
    for name, values in [('sigma_max', sigma[:, 0]), ('sigma_min', sigma[:, 1]), ('scale_error', error),
                         ('anisotropy', sigma[:, 0] / sigma[:, 1]), ('area_ratio', np.prod(sigma, axis=1))]:
        for q in (.5, .95):
            report[f'{name}_p{int(q * 100)}'] = weighted_quantile(values, weights, q)
    report['usable_area_curve'] = {str(t): fraction(valid & (error <= t))
                                   for t in (.02, .05, .1, .2, .5, 1.)}
    pieces = face_components(faces, np.flatnonzero(valid[rows])) if valid[rows].any() else []
    report['pieces'] = len(pieces)
    report['largest_piece_area_fraction'] = max((float(area[rows[p]].sum() / total) for p in pieces), default=0.)
    report['artificial_seam_length_um'] = artificial_seams(v, f, mapped) if len(faces) else 0.
    report['artificial_seam_policy'] = 'unique original interior edges exposed as mapped boundaries, including omitted faces'
    with locked(output):
        np.savez_compressed(Path(output) / 'face-metrics.npz', area_um2=area, sigma=sigma, reasons=reasons, valid=valid)
        seal(output, 'METRICS.json', dict(case=case['source_sha256']), report, ['face-metrics.npz'])
    return report
