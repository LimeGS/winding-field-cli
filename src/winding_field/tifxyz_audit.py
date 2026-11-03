"""Bounded, fail-closed audit of delivered float32 TIFXYZ geometry.

The caller certifies root_error_um by reprojection before auditing. Reader rules
are explicit; both diagonals are checked, including mixed choices across cells.
Spatial indices contain only grid IDs and outward SQLite R-tree boxes in output
voxel coordinates. No native mesh or output-sized in-memory index is constructed.
"""
from collections import OrderedDict
from fractions import Fraction as Q
import itertools
import json
import math
import sqlite3
import time

import numpy as np

from .tifxyz_contract import Blocked, Paused, validate_frame
from .tifxyz_project import _value_interval
from .tifxyz_predicates import Backend, BATCH_ROWS

MAX_DEPTH = 8
MAX_SUBCELLS = 256_000_000
MAX_NONLOCAL_PAIRS = 20_000_000
PAGE = 4096
BITS = tuple(itertools.product((0, 1), repeat=3))
# Corners NW, NE, SW, SE; positive UV orientation for either diagonal.
TRIANGLES = ((0, 1, 3), (0, 3, 2), (0, 1, 2), (1, 3, 2))
CORNERS = ((0, 0), (0, 1), (1, 0), (1, 1))


def _q(x): return x if isinstance(x, Q) else Q(float(x))
def _sub(a, b): return tuple(x-y for x, y in zip(a, b))
def _dot(a, b): return sum(x*y for x, y in zip(a, b))
def _cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])
def _lerp(a, b, t): return tuple(x+t*(y-x) for x, y in zip(a, b))
def _exact(points): return [tuple(_q(x) for x in p) for p in points]


def _checkpoint(store, stop=None):
    if isinstance(stop,_Control):
        stop.check();return
    if stop and stop(): raise Paused()
    store.guard()


class _Control:
    def __init__(self,store,stop,progress,counters):
        self.store,self.stop,self.progress,self.counters=store,stop,progress,counters
        self.last_poll=self.last_guard=-float('inf');self.phase='index'

    def check(self,force=False):
        now=time.monotonic()
        if force or now-self.last_poll>=.1:
            self.last_poll=now
            if self.stop and self.stop():raise Paused()
        if force or now-self.last_guard>=1.:
            self.last_guard=now;self.store.guard()
            if self.progress:self.progress(dict(phase=self.phase,counters=dict(self.counters)))

    def set_phase(self,phase):
        self.phase=phase;self.check(force=True)


def _reader(contract):
    if (not isinstance(contract, dict) or contract.get('cell_rule') != 'four-corners'
            or contract.get('inpainting') is not False
            or contract.get('diagonals') != ['nw-se', 'ne-sw']):
        raise Blocked('READER_CONTRACT', 'require four-corners, both diagonals, no inpainting')


def _windows(grid):
    height, width = grid['shape_yx']
    if grid.get('tile_pixels') != 256 or min(height, width) < 1:
        raise ValueError('invalid audit grid')
    for r in range(0, height, 256):
        for c in range(0, width, 256):
            yield (r//256, c//256), np.array([r, c]), np.minimum([r+256, c+256], [height, width])


def _validate_region(data):
    p, m, e = data['xyz'], data['mask'], data['root_error_um']
    if (m.ndim != 2 or max(m.shape) > 260 or p.shape != (*m.shape, 3) or e.shape != m.shape
            or p.dtype != np.float32 or m.dtype != bool or e.dtype != np.float64
            or not np.isfinite(p).all() or not np.isfinite(e).all() or (e < 0).any()
            or (p[~m] != -1).any() or (e[~m] != 0).any()):
        raise ValueError('invalid delivered tile arrays')


def _participation(mask):
    cells = mask[:-1, :-1] & mask[1:, :-1] & mask[:-1, 1:] & mask[1:, 1:]
    used = np.zeros_like(mask)
    for r, c in CORNERS: used[r:r+cells.shape[0], c:c+cells.shape[1]] |= cells
    return cells, used


def normalize_delivered(projected_store, delivered_store, grid, reader_contract):
    """Write a distinct immutable generation; ownership resolves every halo pixel."""
    _reader(reader_contract)
    if projected_store is delivered_store or projected_store.root == delivered_store.root:
        raise ValueError('normalization requires a distinct destination')
    if projected_store.grid != grid or delivered_store.grid != grid:
        raise ValueError('normalization grid differs')
    counts = dict(nodes=0, cells=0, triangles=0, orphans_removed=0)
    nondegenerate = False
    for tile_id, lo, hi in _windows(grid):
        delivered_store.guard()
        outer_lo, outer_hi = np.maximum(lo-2, 0), np.minimum(hi+2, grid['shape_yx'])
        data = projected_store.read_region(outer_lo.tolist(), outer_hi.tolist())
        _validate_region(data)
        cells, used = _participation(data['mask'])
        owned = tuple(slice(a-o, b-o) for a, b, o in zip(lo, hi, outer_lo))
        counts['nodes'] += int(used[owned].sum())
        counts['orphans_removed'] += int((data['mask'][owned] & ~used[owned]).sum())
        owned_cells = tuple(slice(a-o, min(b, n-1)-o) for a, b, n, o in zip(lo, hi, grid['shape_yx'], outer_lo))
        counts['cells'] += int(cells[owned_cells].sum())
        if not nondegenerate:
            for rr, cc in zip(*np.nonzero(cells[owned_cells])):
                r, c = int(rr+lo[0]-outer_lo[0]), int(cc+lo[1]-outer_lo[1])
                q = _exact([data['xyz'][r+dy, c+dx] for dy, dx in CORNERS])
                if any(any(_cross(_sub(q[b], q[a]), _sub(q[d], q[a]))) for a, b, d in TRIANGLES):
                    nondegenerate = True
                    break
        low, high = np.maximum(lo-1, 0), np.minimum(hi+1, grid['shape_yx'])
        sl = tuple(slice(a-o, b-o) for a, b, o in zip(low, high, outer_lo))
        mask = used[sl].copy(); xyz = data['xyz'][sl].copy(); errors = data['root_error_um'][sl].copy()
        xyz[~mask] = -1; errors[~mask] = 0
        delivered_store.commit(dict(tile_id=list(tile_id), owned_box_yx=[lo.tolist(), hi.tolist()],
            box_yx=[low.tolist(), high.tolist()], xyz=xyz, mask=mask, root_error_um=errors))
    if not counts['cells'] or not nondegenerate: raise Blocked('EMPTY_DELIVERED_SURFACE')
    counts['triangles'] = 4*counts['cells']
    with delivered_store.transaction() as db:
        db.execute('CREATE TABLE IF NOT EXISTS audit_reader (id INTEGER PRIMARY KEY, contract TEXT NOT NULL)')
        db.execute('INSERT OR REPLACE INTO audit_reader VALUES (1, ?)', (json.dumps(reader_contract, sort_keys=True),))
    return counts


def _normalized_windows(store, stop=None):
    for tile_id, lo, hi in _windows(store.grid):
        _checkpoint(store, stop)
        low, high = np.maximum(lo-1, 0), np.minimum(hi+1, store.grid['shape_yx'])
        data = store.read_region(low.tolist(), high.tolist()); _validate_region(data)
        cells, used = _participation(data['mask'])
        owned = tuple(slice(a-o, b-o) for a, b, o in zip(lo, hi, low))
        if not np.array_equal(used[owned], data['mask'][owned]):
            raise Blocked('UNNORMALIZED_DELIVERY', str(tile_id))
        yield lo, hi, low, data, cells


def _cells(store, stop=None):
    width = store.grid['shape_yx'][1]
    for lo, hi, low, data, cells in _normalized_windows(store, stop):
        end = np.minimum(hi, np.array(store.grid['shape_yx'])-1)
        sl = tuple(slice(a-o, b-o) for a, b, o in zip(lo, end, low))
        for rr, cc in zip(*np.nonzero(cells[sl])):
            r, c = int(rr+lo[0]), int(cc+lo[1]); y, x = r-low[0], c-low[1]
            points = np.array([data['xyz'][y+dy, x+dx] for dy, dx in CORNERS])
            errors = np.array([data['root_error_um'][y+dy, x+dx] for dy, dx in CORNERS])
            yield r*(width-1)+c, points, errors


def _pages(cursor):
    try:
        while True:
            rows = cursor.fetchmany(PAGE)
            if not rows: break
            yield rows
    finally:
        cursor.close()


def _write_rows(store, table, rows):
    rows=iter(rows)
    while batch:=list(itertools.islice(rows,2048)):
        with store.transaction() as db:
            for offset in range(0,len(batch),128):
                db.executemany(f'INSERT INTO {table} VALUES (?,?,?,?,?,?,?)',batch[offset:offset+128])


def _box_row(identifier, points):
    low, high = points.min(axis=0), points.max(axis=0)
    return (int(identifier), *(float(v) for pair in zip(low, high) for v in pair))


def build_spatial_index(delivered_store, budget):
    """Rebuild derived R-trees under the store's aggregate DB/journal quotas."""
    return _build_index(delivered_store, budget, None)


def _build_index(store, budget, stop):
    if budget is not store.budget and budget != store.budget: raise ValueError('store budget differs')
    _checkpoint(store, stop)
    try:
        row = store.db.execute('SELECT contract FROM audit_reader WHERE id=1').fetchone()
    except sqlite3.OperationalError as error:
        raise Blocked('UNNORMALIZED_DELIVERY', 'reader contract missing') from error
    if row is None: raise Blocked('UNNORMALIZED_DELIVERY')
    _reader(json.loads(row[0]))
    pages = store.db.execute('PRAGMA cache_size').fetchone()[0]
    page_size = store.db.execute('PRAGMA page_size').fetchone()[0]
    cache_bytes = -pages*1024 if pages < 0 else pages*page_size
    # T4 admits at most four 16MiB connections; never enlarge its shared cache.
    if not 0 < cache_bytes <= 16 << 20: raise Blocked('SQLITE_CACHE_BUDGET')
    try:
        with store.transaction() as db:
            for table in ('audit_nodes', 'audit_triangles'):
                db.execute(f'DROP TABLE IF EXISTS {table}')
                db.execute(f'CREATE VIRTUAL TABLE {table} USING rtree(id,x0,x1,y0,y1,z0,z1)')
    except sqlite3.OperationalError as error:
        if 'no such module' in str(error): raise Blocked('RTREE_UNAVAILABLE') from error
        raise
    counts = dict(nodes=0, triangles=0, cells=0)
    width = store.grid['shape_yx'][1]
    def nodes():
        for lo, hi, low, data, _ in _normalized_windows(store, stop):
            sl = tuple(slice(a-o, b-o) for a, b, o in zip(lo, hi, low))
            for rr, cc in zip(*np.nonzero(data['mask'][sl])):
                r, c = int(rr+lo[0]), int(cc+lo[1]); counts['nodes'] += 1
                yield _box_row(r*width+c, data['xyz'][r-low[0], c-low[1]][None, :])
    def triangles():
        for identifier, points, _ in _cells(store, stop):
            counts['cells'] += 1
            for k, indices in enumerate(TRIANGLES):
                counts['triangles'] += 1
                yield _box_row(4*identifier+k, points[list(indices)])
    _write_rows(store, 'audit_nodes', nodes())
    _write_rows(store, 'audit_triangles', triangles())
    if not counts['cells']: raise Blocked('EMPTY_DELIVERED_SURFACE')
    store.guard(sql=True)
    return dict(counts, coordinates='delivered-voxel', sqlite_cache_bytes=cache_bytes,
                page_rows=PAGE, write_rows=128, identity=store.identity)


class _OutputCache:
    def __init__(self, store): self.store, self.cache = store, OrderedDict()

    def region(self, r, c):
        key = (r//256, c//256)
        if key not in self.cache:
            if len(self.cache) == 4: self.cache.popitem(last=False)
            lo = [key[0]*256, key[1]*256]
            hi = [min(a+257, n) for a, n in zip(lo, self.store.grid['shape_yx'])]
            self.cache[key] = self.store.read_region(lo, hi)
            _validate_region(self.cache[key])
        self.cache.move_to_end(key)
        return self.cache[key], r-key[0]*256, c-key[1]*256

    def node(self, identifier):
        r, c = divmod(identifier, self.store.grid['shape_yx'][1])
        data, y, x = self.region(r, c)
        if not data['mask'][y, x]: raise ValueError('spatial index references invalid node')
        return data['xyz'][y, x]

    def cell(self, r, c):
        h, w = self.store.grid['shape_yx']
        if not (0 <= r < h-1 and 0 <= c < w-1): return None
        data, y, x = self.region(r, c)
        if not data['mask'][y:y+2, x:x+2].all(): return None
        return np.array([data['xyz'][y+dy, x+dx] for dy, dx in CORNERS])

    def triangle(self, identifier):
        cell, k = divmod(identifier, 4)
        r, c = divmod(cell, self.store.grid['shape_yx'][1]-1)
        points = self.cell(r, c)
        if points is None: raise ValueError('spatial index references invalid cell')
        return points[list(TRIANGLES[k])]


class _SourceCache:
    def __init__(self, field): self.field, self.cache = field, OrderedDict()

    def tile(self, key):
        key = tuple(key)
        if key not in self.cache:
            if len(self.cache) == 8: self.cache.popitem(last=False)
            low = np.array(key)*64; high = np.minimum(low+65, self.field.shape)
            values, support = self.field.read(low.tolist(), high.tolist())
            self.cache[key] = (values, support & np.isfinite(values))
        self.cache.move_to_end(key)
        return self.cache[key]

    def cell(self, cell):
        key = tuple(int(x)//64 for x in cell); values, support = self.tile(key)
        sl = tuple(slice(int(x)-k*64, int(x)-k*64+2) for x, k in zip(cell, key))
        return values[sl], bool(support[sl].all())


def _on_allowed(p, allowed):
    if not allowed: return False
    if len(allowed) == 1: return p == allowed[0]
    a, b = allowed[:2]; delta = _sub(b, a)
    if not any(delta): return p == a
    if any(_cross(delta, _sub(p, a))): return False
    return 0 <= _dot(_sub(p, a), delta) <= _dot(delta, delta)


def triangle_contact(first, second, allowed=()):
    """Exact integer signs first, rational clipping only for unresolved contact.

    Binary floats share a power-of-two denominator. Scaling *all* coordinates by
    it preserves incidence and signs, including subnormals, without an epsilon.
    """
    if not np.isfinite(first).all() or not np.isfinite(second).all() or not np.isfinite(allowed).all():
        raise Blocked('INTERSECTION_UNCERTAIN')
    ratios = [tuple(float(x).as_integer_ratio() for x in p) for p in (*first, *second, *allowed)]
    denominator = max(d for point in ratios for _, d in point)
    points = [tuple(n*(denominator//d) for n, d in p) for p in ratios]
    a, b, shared = points[:3], points[3:6], points[6:]
    na = _cross(_sub(a[1], a[0]), _sub(a[2], a[0]))
    nb = _cross(_sub(b[1], b[0]), _sub(b[2], b[0]))
    if not any(na) or not any(nb): raise Blocked('DEGENERATE_TRIANGLE')
    if len(shared) == 2 and shared[0] != shared[1] and all(p in a and p in b for p in shared):
        # Distinct supporting planes meet only on the common edge's line.
        if any(_cross(na, nb)): return False
        edge = _sub(shared[1], shared[0])
        sa = _cross(edge, _sub(next(p for p in a if p not in shared), shared[0]))
        sb = _cross(edge, _sub(next(p for p in b if p not in shared), shared[0]))
        # Same-side coplanar triangles overlap in an open wedge along the edge.
        return _dot(sa, sb) > 0
    da = [_dot(nb, _sub(p, b[0])) for p in a]
    db = [_dot(na, _sub(p, a[0])) for p in b]
    for triangle, distances in ((a, da), (b, db)):
        if not any(distances): continue
        if min(distances) >= 0 or max(distances) <= 0:
            zeros = [p for p, d in zip(triangle, distances) if d == 0]
            if not zeros or (len(shared) <= 2 and all(_on_allowed(p, shared) for p in zeros)): return False
    if not any(db):
        # Coplanar SAT; boundary intersections are collinear integer intervals.
        for triangle, other, normal in ((a, b, na), (b, a, nb)):
            for u, v in zip(triangle, triangle[1:]+triangle[:1]):
                inward = _cross(normal, _sub(v, u))
                signs = [_dot(inward, _sub(p, u)) for p in other]
                if max(signs) <= 0:
                    zeros = [p for p, sign in zip(other, signs) if sign == 0]
                    if not zeros: return False
                    axis = next(i for i in range(3) if u[i] != v[i])
                    lo = max(min(u[axis], v[axis]), min(p[axis] for p in zeros))
                    hi = min(max(u[axis], v[axis]), max(p[axis] for p in zeros))
                    if lo > hi: return False
                    return any(not _on_allowed(p, shared) for p in (u, v, *zeros) if p[axis] in (lo, hi))
        return True  # No separating/supporting edge: positive-area overlap.
    return _rational_contact(first, second, allowed)


def _rational_contact(first, second, allowed=()):
    """Exact fallback for intersection extrema requiring rational division."""
    a, b, shared = _exact(first), _exact(second), _exact(allowed)
    na = _cross(_sub(a[1], a[0]), _sub(a[2], a[0]))
    nb = _cross(_sub(b[1], b[0]), _sub(b[2], b[0]))
    if not any(na) or not any(nb): raise Blocked('DEGENERATE_TRIANGLE')
    da = [_dot(nb, _sub(p, b[0])) for p in a]
    db = [_dot(na, _sub(p, a[0])) for p in b]
    if any(all(x > 0 for x in d) or all(x < 0 for x in d) for d in (da, db)): return False
    if not any(db):
        # Convex clipping retains all contact extrema, including zero-area contacts.
        axis = max(range(3), key=lambda i: abs(na[i]))
        axes = [i for i in range(3) if i != axis]
        def side(p, q, r):
            i, j = axes
            return (q[i]-p[i])*(r[j]-p[j])-(q[j]-p[j])*(r[i]-p[i])
        sign = 1 if side(*a) > 0 else -1
        polygon = b
        for p, q in zip(a, a[1:]+a[:1]):
            clipped = []
            if not polygon: return False
            for u, v in zip(polygon, polygon[1:]+polygon[:1]):
                su, sv = sign*side(p, q, u), sign*side(p, q, v)
                if su >= 0: clipped.append(u)
                if (su < 0 < sv) or (sv < 0 < su): clipped.append(_lerp(u, v, su/(su-sv)))
            polygon = clipped
        return any(not _on_allowed(p, shared) for p in polygon)
    direction = _cross(na, nb)
    if not any(direction): return False
    axis = max(range(3), key=lambda i: abs(direction[i]))
    def section(triangle, distances):
        points = [p for p, d in zip(triangle, distances) if d == 0]
        for i, j in ((0, 1), (1, 2), (2, 0)):
            if distances[i]*distances[j] < 0:
                points.append(_lerp(triangle[i], triangle[j], distances[i]/(distances[i]-distances[j])))
        return sorted(points, key=lambda p: p[axis])
    sa, sb = section(a, da), section(b, db)
    if not sa or not sb: return False
    lo, hi = max(sa[0][axis], sb[0][axis]), min(sa[-1][axis], sb[-1][axis])
    if lo > hi: return False
    if sa[0][axis] == sa[-1][axis]: points = [sa[0]]
    else: points = [_lerp(sa[0], sa[-1], (t-sa[0][axis])/(sa[-1][axis]-sa[0][axis])) for t in (lo, hi)]
    return any(not _on_allowed(p, shared) for p in points)


class _Seams:
    """Bounded guide ancestry only. Grid-node ancestry is computed on demand."""
    def __init__(self, guide, grid):
        self.grid, self.edges, self.keys = grid, [], OrderedDict()
        if 'source_vertices' not in guide: return
        v, uv, f, parent = (np.asarray(guide[k]) for k in ('world', 'uv', 'faces', 'source_vertices'))
        if (len(f) > 250000 or v.shape != (len(parent), 3) or uv.shape != (len(parent), 2)
                or parent.dtype.kind not in 'iu' or (parent < 0).any() or not np.isfinite(v).all()
                or not np.isfinite(uv).all() or f.ndim != 2 or f.shape[1] != 3
                or f.dtype.kind not in 'iu' or (f < 0).any() or (f >= len(v)).any()):
            raise Blocked('INVALID_SEAM_ANCESTRY')
        representatives = {}
        for i, p in enumerate(parent):
            if int(p) in representatives and not np.array_equal(v[i], v[representatives[int(p)]]):
                raise Blocked('INVALID_SEAM_ANCESTRY')
            representatives[int(p)] = i
        edges = {}
        for face in f:
            for a, b in zip(face, np.roll(face, -1)):
                key = tuple(sorted((int(a), int(b))))
                if key in edges: edges[key] = None
                else: edges[key] = (int(a), int(b))
        groups = {}
        for edge in edges.values():
            if edge is None: continue
            a, b = edge; pa, pb = int(parent[a]), int(parent[b])
            if pa == pb: raise Blocked('INVALID_SEAM_ANCESTRY')
            groups.setdefault(tuple(sorted((pa, pb))), []).append((a, b))
        for key, copies in groups.items():
            if len(copies) == 1: continue
            if len(copies) != 2: raise Blocked('INVALID_SEAM_ANCESTRY')
            a, b = copies[0]; c, d = copies[1]
            if parent[a] != parent[d] or parent[b] != parent[c]: raise Blocked('INVALID_SEAM_ANCESTRY')
            for a, b in copies:
                if int(parent[a]) != key[0]: a, b = b, a
                self.edges.append((key, tuple(map(_q, uv[a])), tuple(map(_q, uv[b]))))

    def key(self, node):
        if node in self.keys:
            self.keys.move_to_end(node); return self.keys[node]
        r, c = divmod(node, self.grid['shape_yx'][1])
        p = tuple(_q(a)+i*_q(self.grid['pitch_um']) for a, i in zip(self.grid['uv_min_um'], (c, r)))
        found = None
        for key, a, b in self.edges:
            delta = _sub(b, a)
            if not any(delta): raise Blocked('INVALID_SEAM_ANCESTRY')
            if (p[0]-a[0])*delta[1] != (p[1]-a[1])*delta[0]: continue
            axis = 0 if delta[0] else 1; t = (p[axis]-a[axis])/delta[axis]
            if 0 <= t <= 1:
                found = ('vertex', key[0]) if t == 0 else ('vertex', key[1]) if t == 1 else ('edge', key, t)
                break
        if len(self.keys) == 4096: self.keys.popitem(last=False)
        self.keys[node] = found
        return found


class _Pairs:
    def __init__(self,backend,seams,counters,store,stop):
        self.backend,self.seams,self.counters,self.store,self.stop=backend,seams,counters,store,stop
        self.pending=[]

    def add(self,aid,a,bid,b):
        self.pending.append((aid,a,bid,b))
        if len(self.pending)>=BATCH_ROWS:self.flush()

    def flush(self):
        if not self.pending:return
        _checkpoint(self.store,self.stop)
        ids=np.array([[x[0],x[2]] for x in self.pending],dtype=np.int64)
        a=np.stack([x[1] for x in self.pending]);b=np.stack([x[3] for x in self.pending]);self.pending.clear()
        keep=~((a.max(1)<b.min(1)).any(1)|(b.max(1)<a.min(1)).any(1))
        ids,a,b=ids[keep],a[keep],b[keep]
        if not len(a):return
        width=self.seams.grid['shape_yx'][1]
        corners=np.asarray(CORNERS)[np.asarray(TRIANGLES)[ids%4]]
        cells=ids//4;rr,cc=cells//(width-1),cells%(width-1)
        nodes=(rr[:,:,None]+corners[:,:,:,0])*width+cc[:,:,None]+corners[:,:,:,1]
        equal=(a[:,:,None,:]==b[:,None,:,:]).all(3)
        same=nodes[:,0,:,None]==nodes[:,1,None,:]
        seam=np.zeros(len(a),bool)
        if self.seams.edges:
            for n,i,j in zip(*np.nonzero(equal&~same)):
                key=self.seams.key(int(nodes[n,0,i]))
                if key is not None and key==self.seams.key(int(nodes[n,1,j])):
                    same[n,i,j]=True;seam[n]=True
        allowed=(same&equal).any(2)
        rows=np.zeros((len(a),24));rows[:,:9]=a.reshape(-1,9);rows[:,9:18]=b.reshape(-1,9);rows[:,18:21]=allowed
        decisions=self.backend.classify(2,rows) if self.backend else np.full(len(a),-1,dtype=np.int8)
        for n,decision in enumerate(decisions):
            if n%128==0:_checkpoint(self.store,self.stop)
            self.counters['intersection_tests']+=1
            contact=bool(decision) if decision>=0 else triangle_contact(a[n],b[n],a[n][allowed[n]])
            if contact:raise Blocked('OUTPUT_INTERSECTION',f'{ids[n,0]},{ids[n,1]}')
            if seam[n] and allowed[n].any():self.counters['seam_contacts']+=1


def _query(db, table, low, high, after=None):
    args = tuple(float(v) for pair in zip(high, low) for v in pair)
    sql = f'SELECT id FROM {table} WHERE x0<=? AND x1>=? AND y0<=? AND y1>=? AND z0<=? AND z1>=?'
    if after is not None: sql += ' AND id>?'; args += (int(after),)
    return db.execute(sql, args)


def _intersections(store, cache, seams, counters, stop, backend=None):
    width = store.grid['shape_yx'][1]
    pairs_batch=_Pairs(backend,seams,counters,store,stop)
    if isinstance(stop,_Control):stop.set_phase('intersections_local')
    # A separate constant-neighbour sweep never debits the nonlocal cap.
    for cell, q, _ in _cells(store, stop):
        r, c = divmod(cell, width-1)
        for dr, dc in ((0, 0), (0, 1), (1, -1), (1, 0), (1, 1)):
            other = cache.cell(r+dr, c+dc)
            if other is None: continue
            other_id = (r+dr)*(width-1)+c+dc
            pairs = ((0, 1), (2, 3)) if (dr, dc) == (0, 0) else itertools.product(range(4), repeat=2)
            for i, j in pairs:
                counters['local_pairs'] += 1
                pairs_batch.add(4*cell+i,q[list(TRIANGLES[i])],4*other_id+j,other[list(TRIANGLES[j])])
        if counters['local_pairs'] % 4096 < 80: _checkpoint(store, stop)
    pairs_batch.flush()
    if isinstance(stop,_Control):stop.set_phase('intersections_nonlocal')
    for cell, q, _ in _cells(store, stop):
        r, c = divmod(cell, width-1)
        for k, indices in enumerate(TRIANGLES):
            identifier = 4*cell+k; a = q[list(indices)]
            for rows in _pages(_query(store.db, 'audit_triangles', a.min(0), a.max(0), identifier)):
                _checkpoint(store, stop)
                for (other,) in rows:
                    counters['nonlocal_candidates'] += 1
                    rr, cc = divmod(other//4, width-1)
                    if abs(rr-r) <= 1 and abs(cc-c) <= 1: continue
                    if counters['nonlocal_pairs'] >= MAX_NONLOCAL_PAIRS: raise Blocked('INTERSECTION_WORK_LIMIT')
                    counters['nonlocal_pairs'] += 1
                    pairs_batch.add(identifier,a,other,cache.triangle(other))
    pairs_batch.flush()


def _sqrt_up(value):
    converted = float(value)
    if converted == 0 and value != 0: raise Blocked('NUMERICAL_UNCERTAINTY', 'squared distance underflow')
    rounded = math.sqrt(converted)
    if not math.isfinite(rounded): raise Blocked('NUMERICAL_UNCERTAINTY')
    while _q(rounded)**2 < value: rounded = math.nextafter(rounded, math.inf)
    return rounded


def _world(points, frame):
    unit = _q(frame['voxel_um']); origin = tuple(map(_q, frame['origin_xyz_um']))
    return [tuple(_q(x)*unit+o for x, o in zip(p, origin)) for p in points]


def _vertex_cover_bound(points, error):
    edges2 = [_dot(_sub(a, b), _sub(a, b)) for a, b in itertools.combinations(points, 2)]
    longest = max(edges2); edge = _sqrt_up(longest)
    # A right/obtuse triangle's farthest nearest-vertex point lies on its boundary.
    divisor = 4 if 2*longest >= sum(edges2) else 3
    cover = _sqrt_up(longest/divisor)
    margin = 32*np.finfo(float).eps*max(1., edge, *(abs(float(x)) for p in points for x in p))
    return math.nextafter(math.nextafter(cover+error, math.inf)+margin, math.inf)


def _output_geometry(field, store, cache, frame, tau, counters, maxima, stop, backend=None):
    origin = tuple(map(_q, field.origin_um)); spacing = tuple(map(_q, field.spacing_um))
    rows=np.zeros((BATCH_ROWS,24));pending=[]
    def flush():
        if not pending:return
        decisions=backend.classify(1,rows[:len(pending)]) if backend else np.full(len(pending),-1,dtype=np.int8)
        for n,(normal,identifier) in enumerate(pending):
            if n%128==0:_checkpoint(store,stop)
            if decisions[n]==1:continue
            if decisions[n]==0:raise Blocked('OUTPUT_ORIENTATION',str(identifier))
            values=rows[n,13:21].reshape(2,2,2);lower=Q(0)
            for axis in range(3):
                low,high=np.take(values,0,axis=axis).ravel(),np.take(values,1,axis=axis).ravel()
                lower+=min((_q(b)-_q(a))/spacing[2-axis]*normal[2-axis] for a,b in zip(low,high))
            if lower<=0:raise Blocked('OUTPUT_ORIENTATION',str(identifier))
        pending.clear()
    for identifier, encoded, errors in _cells(store, stop):
        if (encoded < 0).any() or (encoded > np.array(frame['shape_zyx'][::-1])-1).any():
            raise Blocked('OUTPUT_FRAME_BOUNDS')
        world = _world(encoded, frame)
        grid = [tuple((x-o)/h for x, o, h in zip(p, origin, spacing))[::-1] for p in world]
        if any(not 0 <= x <= n-1 for p in grid for x, n in zip(p, field.shape)):
            raise Blocked('OUTPUT_OUTSIDE_ROI')
        for k, indices in enumerate(TRIANGLES):
            points = [world[i] for i in indices]
            normal = _cross(_sub(points[1], points[0]), _sub(points[2], points[0]))
            if not any(normal): raise Blocked('DEGENERATE_TRIANGLE', str(4*identifier+k))
            error = float(max(errors[list(indices)]))
            bound = _vertex_cover_bound(points, error)
            maxima['output_bound_um'] = max(maxima['output_bound_um'], bound)
            maxima['root_error_um'] = max(maxima['root_error_um'], error)
            if bound > tau: raise Blocked('OUTPUT_DISTANCE_BOUND', str(4*identifier+k))
            g = [grid[i] for i in indices]
            # shortcut: AABB may over-block; use exact cell clipping if real cases require it.
            # Include edges and both sides of exact internal faces.
            ranges = [range(max(0, math.ceil(min(p[a] for p in g))-1),
                            min(field.shape[a]-2, math.floor(max(p[a] for p in g)))+1) for a in range(3)]
            for cell in itertools.product(*ranges):
                values, supported = cache.cell(cell); counters['support_stencils'] += 1
                if not supported: raise Blocked('OUTPUT_UNSUPPORTED_STENCIL', str(cell))
                n=len(pending);rows[n,:9]=encoded[list(indices)].ravel();rows[n,9]=frame['voxel_um']
                rows[n,10:13]=field.spacing_um;rows[n,13:21]=values.ravel()
                pending.append((normal,4*identifier+k))
                if len(pending)==BATCH_ROWS:flush()
                if counters['support_stencils'] % 4096 == 0: _checkpoint(store, stop)
    flush()


def _float_bounds(q):
    x = float(q)
    return (math.nextafter(x, -math.inf) if _q(x) > q else x,
            math.nextafter(x, math.inf) if _q(x) < q else x)


def _witness(store, output, center, half2, frame, tau, counters, stop, backend=None):
    half = _sqrt_up(half2)
    # A larger box also proves a genuinely uncovered region when it is empty.
    radius = _q(tau)+_q(half)
    origin, unit = tuple(map(_q, frame['origin_xyz_um'])), _q(frame['voxel_um'])
    low = [_float_bounds((c-radius-o)/unit)[0] for c, o in zip(center, origin)]
    high = [_float_bounds((c+radius-o)/unit)[1] for c, o in zip(center, origin)]
    any_candidate = False
    for rows in _pages(_query(store.db, 'audit_nodes', low, high)):
        _checkpoint(store, stop)
        points=np.stack([output.node(identifier) for (identifier,) in rows])
        decisions=np.full(len(rows),-1,dtype=np.int8)
        if backend:
            records=np.zeros((len(rows),24));records[:,:3]=points;records[:,3:6]=frame['origin_xyz_um'];records[:,6]=frame['voxel_um']
            bounds=[_float_bounds(x) for x in center]
            records[:,7:10]=[b[0] for b in bounds];records[:,10:13]=[b[1] for b in bounds]
            records[:,13:15]=_float_bounds(half2);records[:,15]=tau
            decisions=backend.classify(0,records)
        for n,(identifier,) in enumerate(rows):
            any_candidate = True; counters['coverage_candidates'] += 1
            if decisions[n]==1:return True,True
            if decisions[n]==0:continue
            point = _world([points[n]], frame)[0]
            distance2 = _dot(_sub(center, point), _sub(center, point))
            # Exact equivalent of sqrt(d2)+sqrt(h2)<=tau; no rounded false PASS.
            remainder = _q(tau)**2-distance2-half2
            if remainder >= 0 and remainder**2 >= 4*distance2*half2: return True, True
    return False, any_candidate


class CoverageFailure(Blocked):
    def __init__(self,code,cell,index,depth,field):
        super().__init__(code,str(cell))
        self.region=dict(cell_zyx=list(cell),global_cell_zyx=[a+b for a,b in zip(field.lo,cell)],
                         subcell_index_zyx=list(index),depth=depth)


def _coverage(field, store, source, output, frame, tau, counters, stop, backend=None):
    level = float(store.grid['level'])
    origin, spacing = tuple(map(_q, field.origin_um)), tuple(map(_q, field.spacing_um))
    half2_by_depth=[sum((h/(2**(depth+1)))**2 for h in spacing) for depth in range(MAX_DEPTH+1)]
    steps=[tuple(h/(2**(depth+1)) for h in spacing) for depth in range(MAX_DEPTH+1)]
    for key in itertools.product(*(range((n-2)//64+1) for n in field.shape)):
        _checkpoint(store, stop)
        values, valid = source.tile(key)
        shape = tuple(n-1 for n in values.shape)
        supported = np.ones(shape, bool); minimum = np.full(shape, math.inf); maximum = np.full(shape, -math.inf)
        for bits in BITS:
            sl = tuple(slice(b, b+n) for b, n in zip(bits, shape))
            supported &= valid[sl]; np.minimum(minimum, values[sl], out=minimum); np.maximum(maximum, values[sl], out=maximum)
        counters['source_cells'] += math.prod(shape)
        counters['supported_source_cells'] += int(supported.sum())
        active = supported & (minimum <= level) & (maximum >= level)
        counters['active_source_cells'] += int(active.sum())
        for local in zip(*np.nonzero(active)):
            cell = tuple(int(k*64+i) for k, i in zip(key, local))
            corners = values[tuple(slice(i, i+2) for i in local)]
            if (corners == level).all(): raise Blocked('IDENTICALLY_LEVEL_CELL', str(cell))
            cell_origin=tuple(o+c*h for o,c,h in zip(origin,cell[::-1],spacing))
            stack = [((0,0,0),0)]
            while stack:
                index,depth=stack.pop();size=2.**-depth
                lo=tuple(i*size for i in index);hi=tuple((i+1)*size for i in index)
                if depth:
                    if counters['subcells'] >= MAX_SUBCELLS: raise Blocked('COVERAGE_WORK_LIMIT')
                    counters['subcells'] += 1
                    intervals = [_value_interval(corners, tuple(hi[a] if bit else lo[a] for a, bit in enumerate(bits)), level) for bits in BITS]
                    if min(v[0] for v in intervals) > 0 or max(v[1] for v in intervals) < 0: continue
                center=tuple(o+(2*i+1)*h for o,i,h in zip(cell_origin,index[::-1],steps[depth]))
                found, candidate = _witness(store, output, center, half2_by_depth[depth], frame, tau, counters, stop, backend)
                if found:
                    counters['covered_boxes'] += 1
                    continue
                if not candidate: raise CoverageFailure('SOURCE_UNCOVERED',cell,index,depth,field)
                if depth >= MAX_DEPTH: raise CoverageFailure('COVERAGE_DEPTH_LIMIT',cell,index,depth,field)
                for bits in reversed(BITS):
                    stack.append((tuple(2*i+b for i,b in zip(index,bits)),depth+1))
                if counters['subcells'] % 4096 == 0: _checkpoint(store, stop)


def audit_level(field, store, guide, settings, stop=None, *, progress=None, expected_accelerator=None):
    """Return explicit PASS/BLOCKED. Integrity errors and Paused propagate.

    Support/orientation use a conservative stencil AABB enclosure, so difficult
    geometry can BLOCK even when a tighter triangle/cell clipping would certify it.
    This is a geometric tolerance certificate, never source topology equivalence.
    """
    counters = dict(nodes=0, cells=0, triangles=0, support_stencils=0, local_pairs=0,
        nonlocal_candidates=0, nonlocal_pairs=0, intersection_tests=0, seam_contacts=0,
        source_cells=0, supported_source_cells=0, active_source_cells=0, subcells=0,
        coverage_candidates=0, covered_boxes=0)
    maxima = dict(output_bound_um=0., root_error_um=0.)
    result = dict(status='BLOCKED', reasons=[], counters=counters, maxima=maxima,
        physical_acceptance=False, native_face_identity=False, native_topology_equivalence='UNMEASURED',
        source_identity=field.identity, grid=store.grid, source_scan_complete=False,
        geometry_complete=False,blocked_phase=None,remaining_source_coverage='UNMEASURED',
        root_error_authority='producer-certified; independent reprojection required',
        support_enclosure='conservative triangle AABB, all incident stencils',
        limits=dict(depth=MAX_DEPTH, subcells=MAX_SUBCELLS, nonlocal_pairs=MAX_NONLOCAL_PAIRS,
                    fetch_rows=PAGE, source_cache_tiles=8, output_cache_tiles=4))
    try:
        stop=_Control(store,stop,progress,counters)
        _checkpoint(store, stop)
        frame = validate_frame(settings.output_frame)
        tau = float(settings.tolerance_um)
        if not math.isfinite(tau) or tau <= 0 or not math.isfinite(store.grid['level']): raise ValueError('invalid audit level/tolerance')
        if (tuple(settings.roi_lo_zyx) != tuple(field.lo)
                or tuple(b-a for a, b in zip(settings.roi_lo_zyx, settings.roi_hi_zyx)) != tuple(field.shape)):
            raise ValueError('audit ROI differs from physical source view')
        if math.prod(store.grid['shape_yx']) > settings.max_pixels: raise Blocked('PIXEL_LIMIT')
        field.validate()
        with Backend(settings.audit_backend,expected=expected_accelerator) as backend:
            index = _build_index(store, store.budget, stop)
            counters.update({key: index[key] for key in ('nodes', 'cells', 'triangles')})
            source, output, seams = _SourceCache(field), _OutputCache(store), _Seams(guide, store.grid)
            stop.set_phase('geometry')
            _output_geometry(field, store, source, frame, tau, counters, maxima, stop, backend)
            _intersections(store, output, seams, counters, stop, backend)
            result['geometry_complete']=True
            stop.set_phase('coverage')
            try:
                _coverage(field, store, source, output, frame, tau, counters, stop, backend)
            except CoverageFailure as error:
                result.update(blocked_phase='coverage',reasons=[dict(code=error.code,detail=error.detail,region=error.region)])
            else:
                result.update(source_scan_complete=True,remaining_source_coverage='NONE')
        field.validate(); _checkpoint(store, stop)
        if result['source_scan_complete']:result['status']='PASS'
    except Blocked as error:
        result.update(blocked_phase=getattr(stop,'phase',None),reasons=[dict(code=error.code,detail=error.detail)])
    except (OverflowError, FloatingPointError) as error:
        result.update(blocked_phase=getattr(stop,'phase',None),reasons=[dict(code='NUMERICAL_UNCERTAINTY',detail=str(error))])
    return result
