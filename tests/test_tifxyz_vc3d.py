"""Independent, bounded T8 reader acceptance; no exporter helpers as oracle.

Run local tests with pytest. Real control is opt-in via VC3D_APPRUN and
TIFXYZ_VC3D_CONTROL=1. This file also provides a CLI for an already independently
verified READY or PARTIAL level; its OBJ/log/proof are written only to a new --work
directory. Reader acceptance proves neither source coverage nor root reprojection.
Adapted from vesubius-challenge/tools/mesh_to_tifxyz/test_vc3d.py.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import struct
import subprocess
import sys
import time

import numpy as np
import pytest

tifffile = pytest.importorskip('tifffile')
MAX_PIXELS = 4_194_304
XYZ_ATOL = 5.1e-7  # Pinned VC3D writes fixed six decimal places; compare to stored float32.
GIB = 1 << 30


def _require(ok, reason):
    if not ok:
        raise ValueError(reason)


def _regular(path, cap):
    info = path.lstat()
    _require(stat.S_ISREG(info.st_mode) and info.st_size <= cap, f'file type/size: {path}')
    return info


def _json(path):
    _regular(path, 1 << 20)
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    return json.loads(path.read_text(), object_pairs_hook=pairs)


def _record(path):
    info = _regular(path, 2*GIB)
    sha = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            sha.update(block)
    after = path.stat()
    _require((info.st_size, info.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
             'source changed while hashing')
    return dict(bytes=info.st_size, mtime_ns=info.st_mtime_ns, sha256=sha.hexdigest())


def _snapshot(segment):
    paths = [segment / name for name in ('x.tif', 'y.tif', 'z.tif', 'meta.json')]
    paths += [p for p in (segment/'VERIFIED.json', segment.parent/'READY.json',
                          segment.parent/'PARTIAL.json', segment.parent/'EXPORT.json',
                          segment.parent/'TIFXYZ.json') if p.exists()]
    return {str(p): _record(p) for p in paths}


def _publication(segment, *, required=False):
    """Check publication binding/labels independently of the geometric OBJ check."""
    root = segment.parent
    markers = [root/name for name in ('READY.json', 'PARTIAL.json')
               if (root/name).exists() or (root/name).is_symlink()]
    if not required and not markers and not (root/'TIFXYZ.json').exists() and not (root/'EXPORT.json').exists():
        return {}
    _require(len(markers)==1, 'publication requires exactly one READY/PARTIAL marker')
    marker = _json(markers[0]); receipt = _json(root/'TIFXYZ.json'); identity = _json(root/'EXPORT.json')
    status = markers[0].stem; complete = status=='READY'
    identity_sha = hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False,
                                            separators=(',', ':')).encode()).hexdigest()
    flags = dict(physical_acceptance=False, native_face_identity=False,
                 native_topology_equivalence='UNMEASURED')
    for value in (marker, receipt, identity):
        _require(all(type(value.get(k)) is type(v) and value.get(k)==v for k,v in flags.items()),
                 'publication acceptance flags differ')
    for value in (marker, receipt):
        _require(value.get('status')==status and value.get('coverage_complete') is complete,
                 'publication status/coverage mismatch')
        _require(value.get('identity_sha256')==identity_sha, 'publication identity binding mismatch')
    _require(marker.get('tifxyz_sha256')==_record(root/'TIFXYZ.json')['sha256'],
             'publication TIFXYZ binding mismatch')
    settings = identity.get('effective_settings', {})
    policy = settings.get('coverage_policy', 'strict')
    _require(policy in ('strict', 'allow_partial') and (complete or policy=='allow_partial'),
             'partial publication requires allow_partial policy')
    levels = receipt.get('levels')
    _require(isinstance(levels, list) and bool(levels) and len(levels)==len(settings.get('levels', [])),
             'publication level count mismatch')
    selected = None
    for index, item in enumerate(levels):
        _require(isinstance(item, dict) and item.get('status') in ('READY', 'PARTIAL'),
                 'invalid level quality')
        _require(item.get('coverage_complete') is (item['status']=='READY'), 'level coverage mismatch')
        _require(item.get('segment')==f'level-{index:04d}' and item.get('level')==settings['levels'][index],
                 'publication level selection mismatch')
        if item['segment']==segment.name:
            selected = item
    _require(all(item['coverage_complete'] for item in levels)==complete, 'root/level coverage mismatch')
    _require(selected is not None, 'segment not in publication')
    verified = _json(segment/'VERIFIED.json')
    _require(all(verified.get(k)==selected[k] and type(verified.get(k)) is type(selected[k])
                 for k in ('status', 'coverage_complete')), 'VERIFIED level quality mismatch')
    _require(all(type(verified.get(k)) is type(v) and verified.get(k)==v for k,v in flags.items()),
             'VERIFIED acceptance flags differ')
    record = selected.get('manifest', {}).get('files', {}).get('VERIFIED.json', {})
    _require(record.get('sha256')==_record(segment/'VERIFIED.json')['sha256'], 'VERIFIED binding mismatch')
    return dict(publication_status=status, coverage_complete=complete,
                level_status=selected['status'], level_coverage_complete=selected['coverage_complete'])


def _raw_header(path):
    """Bound IFD/tag arrays before tifffile parses metadata or allocates pixels."""
    _regular(path, 256 << 20)
    with path.open('rb') as stream:
        def unpack(fmt):
            size = struct.calcsize(fmt)
            raw = stream.read(size)
            _require(len(raw) == size, 'truncated TIFF header')
            return struct.unpack(fmt, raw)
        order = stream.read(2)
        _require(order in (b'II', b'MM'), 'invalid TIFF header')
        endian = '<' if order == b'II' else '>'
        version, = unpack(endian+'H')
        _require(version in (42, 43), 'invalid TIFF version')
        big = version == 43
        if big:
            _require(unpack(endian+'HH') == (8, 0), 'invalid BigTIFF header')
        offset, = unpack(endian+('Q' if big else 'I'))
        stream.seek(offset)
        count, = unpack(endian+('Q' if big else 'H'))
        _require(1 <= count <= 64, 'TIFF tag count cap')
        dimensions = {}
        for _ in range(count):
            tag, kind, length = unpack(endian+('HHQ' if big else 'HHI'))
            raw = stream.read(8 if big else 4)
            _require(len(raw) == (8 if big else 4), 'truncated TIFF tag')
            _require(length <= (32768 if tag in (324, 325) else 4096), 'TIFF tag length cap')
            if tag in (256, 257):
                _require(tag not in dimensions and length == 1 and kind in (3, 4, 16),
                         'invalid TIFF dimensions')
                fmt = {3:'H', 4:'I', 16:'Q'}[kind]
                _require(struct.calcsize(fmt) <= len(raw), 'invalid TIFF dimension type')
                dimensions[tag] = struct.unpack(endian+fmt, raw[:struct.calcsize(fmt)])[0]
        _require(unpack(endian+('Q' if big else 'I')) == (0,), 'multiple TIFF pages')
    h, w = dimensions.get(257, 0), dimensions.get(256, 0)
    _require(h >= 2 and w >= 2 and h*w <= MAX_PIXELS, 'TIFF pixel cap / dimensions')
    return h, w


def _headers(segment):
    shapes = [_raw_header(segment/f'{name}.tif') for name in 'xyz']
    _require(shapes[0] == shapes[1] == shapes[2], 'TIFF shape mismatch')
    shape = shapes[0]
    for name in 'xyz':
        with tifffile.TiffFile(segment/f'{name}.tif') as tif:
            page = tif.pages[0]
            _require(page.shape == shape and page.dtype == np.dtype('float32') and
                     page.is_tiled and (page.tilelength, page.tilewidth) == (256, 256),
                     'TIFF float32/tile layout mismatch')
    meta = _json(segment/'meta.json')
    _require(meta.get('shape_yx') == list(shape), 'metadata shape mismatch')
    scale = meta.get('scale')
    _require(isinstance(scale, list) and len(scale) == 2 and
             all(type(s) in (int, float) and math.isfinite(s) and s > 0 for s in scale),
             'invalid scale')
    pitch, voxel = meta.get('pitch_um'), meta.get('output_frame', {}).get('voxel_um')
    _require(all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in (pitch, voxel)),
             'invalid pitch/frame')
    _require(all(math.isclose(s, voxel/pitch, rel_tol=1e-12) for s in scale),
             'scale differs from voxel/pitch')
    return shape, scale


def check_obj(segment, obj):
    started = time.monotonic()
    segment, obj = Path(segment), Path(obj)
    (h, w), scale = _headers(segment)
    before = _snapshot(segment)
    publication = _publication(segment)
    n = h*w
    _regular(obj, 320*n+(1 << 20))
    # shortcut: independent small-pilot full decode only, refuse >4,194,304 pixels.
    grid = np.empty((h, w, 3), np.float32)
    for axis, name in enumerate('xyz'):
        with tifffile.TiffFile(segment/f'{name}.tif') as tif:
            grid[..., axis] = tif.pages[0].asarray(maxworkers=1)
    _require(np.isfinite(grid).all(), 'nonfinite TIFF XYZ')
    valid = grid[..., 0] != -1
    _require(np.all(grid[~valid] == -1) and np.all(grid[valid] >= 0), 'inconsistent XYZ mask')
    cells = valid[:-1, :-1] & valid[1:, :-1] & valid[:-1, 1:] & valid[1:, 1:]
    normalized = np.zeros((h, w), bool)
    for dy, dx in ((0,0), (0,1), (1,0), (1,1)):
        normalized[dy:dy+h-1, dx:dx+w-1] |= cells
    _require(cells.any() and np.array_equal(normalized, valid), 'empty/non-normalized orphan mask')
    del normalized
    count = int(valid.sum())
    positions = np.empty((count, 3), np.float64)
    uv_nodes = np.empty(count, np.int32)
    vertex_nodes = np.full(count, -1, np.int32)
    uv_seen = np.zeros(n, bool)
    uv_used = np.zeros(count, bool)
    face_bits = np.zeros((h-1, w-1), np.uint8)
    vertices = uvs = normals = faces = 0
    winding = 0
    uv_error = 0.
    with obj.open('rb') as stream:
        while line := stream.readline(4097):
            _require(len(line) <= 4096, 'OBJ line length cap')
            words = line.partition(b'#')[0].split()
            if not words:
                continue
            kind = words[0]
            if kind in (b'v', b'vt', b'vn'):
                _require(len(words) == (3 if kind == b'vt' else 4), 'malformed OBJ coordinates')
                values = tuple(map(float, words[1:]))
                _require(all(map(math.isfinite, values)), 'nonfinite OBJ coordinates')
                if kind == b'v':
                    _require(vertices < count, 'extra OBJ vertex')
                    positions[vertices] = values
                    vertices += 1
                elif kind == b'vn':
                    normals += 1
                    _require(normals <= count, 'extra OBJ normal')
                else:
                    _require(uvs < count, 'extra OBJ UV')
                    xy = []
                    for value, s, bound in zip(values, scale, (w, h)):
                        scaled = value*s
                        _require(math.isfinite(scaled) and -.25 <= scaled < bound-.75,
                                 'UV outside grid')
                        index = round(scaled)
                        expected = index/s
                        # UV arithmetic in reader is float32, then fixed-six OBJ text.
                        tolerance = abs(float(np.spacing(np.float32(expected)))) + XYZ_ATOL
                        error = abs(value-expected)
                        _require(tolerance*s < .25 and error <= tolerance, 'UV scale/grid mismatch')
                        uv_error = max(uv_error, error)
                        xy.append(index)
                    col, row = xy
                    node = row*w+col
                    _require(valid[row, col] and not uv_seen[node], 'invalid/duplicate UV node')
                    uv_seen[node] = True
                    uv_nodes[uvs] = node
                    uvs += 1
            elif kind == b'f':
                _require(len(words) == 4, 'OBJ face must be a triangle')
                nodes = []
                for word in words[1:]:
                    ids = word.split(b'/')
                    _require(len(ids) in (2, 3) and all(ids), 'OBJ face needs vertex/UV IDs')
                    vi, ti = int(ids[0])-1, int(ids[1])-1
                    _require(0 <= vi < vertices and 0 <= ti < uvs, 'OBJ index out of range')
                    if len(ids) == 3:
                        _require(0 < int(ids[2]) <= normals, 'OBJ normal index out of range')
                    node = int(uv_nodes[ti])
                    _require(vertex_nodes[vi] in (-1, node), 'inconsistent vertex/UV mapping')
                    vertex_nodes[vi] = node
                    uv_used[ti] = True
                    nodes.append(divmod(node, w))
                rows, cols = zip(*nodes)
                row, col = min(rows), min(cols)
                _require(max(rows)-row == max(cols)-col == 1 and len(set(nodes)) == 3,
                         'triangle does not belong to one grid cell')
                _require(cells[row, col], 'face over invalid cell/mask hole')
                corners = [(r-row)*2+c-col for r, c in nodes]
                bit = 1 << (6-sum(corners))  # Identify a triangle by its missing corner.
                old = int(face_bits[row, col])
                _require(not old & bit, 'duplicate triangle')
                new = old | bit
                _require(new in (1, 2, 4, 8, 6, 9), 'third/overlapping/incompatible triangles')
                face_bits[row, col] = new
                (y0,x0), (y1,x1), (y2,x2) = nodes
                sign = (x1-x0)*(y2-y0)-(y1-y0)*(x2-x0)
                _require(winding in (0, sign), 'inconsistent triangle winding')
                winding = sign
                faces += 1
            else:
                _require(kind in (b'o', b'g', b's', b'mtllib', b'usemtl'), 'unsupported OBJ record')
    _require(vertices == uvs == count and np.all(vertex_nodes >= 0) and uv_used.all(),
             'missing/unreferenced valid nodes')
    _require(np.all((face_bits[cells] == 6) | (face_bits[cells] == 9)),
             'missing triangle in complete cell')
    recovered = np.bincount(vertex_nodes, minlength=n)
    _require(np.array_equal(recovered, valid.ravel()), 'duplicate/missing vertex mapping')
    maximum_error = 0.
    for start in range(0, count, 65536):
        expected = grid.reshape(-1, 3)[vertex_nodes[start:start+65536]]
        error = np.abs(positions[start:start+65536]-expected)
        _require(np.all(error <= XYZ_ATOL + np.abs(expected)*2e-15), 'XYZ differs from TIFF float32')
        maximum_error = max(maximum_error, float(error.max()))
    _require(before == _snapshot(segment), 'source artifacts changed')
    return dict(status='READER_VERIFIED', source_coverage_verified=False,
                independent_root_reprojection=False, **publication,
                shape_yx=[h,w], pixels=n, pixel_cap=MAX_PIXELS,
                vertices=vertices, uv_vertices=uvs, valid_nodes=count, faces=faces,
                complete_cells=int(cells.sum()), diagonal_counts={
                    'TL_BR':int(np.count_nonzero(face_bits == 6)),
                    'TR_BL':int(np.count_nonzero(face_bits == 9))},
                maximum_xyz_error_input_units=maximum_error, xyz_text_atol=XYZ_ATOL,
                maximum_uv_error=uv_error, scale=scale, all_valid_nodes_recovered=True,
                exact_cell_triangles=True, source_artifacts_unchanged=True,
                source_artifacts=before, seconds=time.monotonic()-started,
                numeric_array_bytes=sum(a.nbytes for a in (grid, valid, cells, positions,
                    uv_nodes, vertex_nodes, uv_seen, uv_used, face_bits, recovered)))


def _run_reader(command, work, pixels):
    import resource
    import shutil
    import signal
    _require(sys.platform == 'linux', 'real-reader resource acceptance requires Linux')
    output_cap = 320*pixels+(1 << 20)
    reserve_disk = int(12.25*GIB)
    def available_ram():
        mem = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
        available = int(mem['MemAvailable'].split()[0])*1024
        # cgroup-v2 ancestors can be tighter than host MemAvailable.
        entries = Path('/proc/self/cgroup').read_text().splitlines()
        entry = next((s[3:] for s in entries if s.startswith('0::')), None)
        _require(entry is not None, 'cannot observe cgroup-v2 memory headroom')
        root = Path('/sys/fs/cgroup')
        group = root / entry.lstrip('/')
        _require(group.is_dir(), 'cannot resolve cgroup memory headroom')
        for path in (group, *group.parents):
            if path == root.parent:
                break
            limit = path/'memory.max'
            if limit.exists():
                value = limit.read_text().strip()
                if value != 'max':
                    available = min(available, int(value)-int((path/'memory.current').read_text()))
        return available
    _require(available_ram() >= 5*GIB, 'RAM admission needs 3 GiB plus 2 GiB reserve')
    _require(shutil.disk_usage(work).free >= reserve_disk+2*output_cap, 'disk reserve admission')
    def child_limits():
        os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
        os.nice(max(0, 19-os.getpriority(os.PRIO_PROCESS, 0)))
        resource.setrlimit(resource.RLIMIT_AS, (2*GIB, 2*GIB))
        resource.setrlimit(resource.RLIMIT_FSIZE, (output_cap, output_cap))
        resource.setrlimit(resource.RLIMIT_CPU, (300, 300))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    env = {**os.environ, 'PYTHONDONTWRITEBYTECODE':'1', 'CUDA_VISIBLE_DEVICES':'',
           **{key:'1' for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                                  'PASTIX_NUM_THREADS', 'NUMEXPR_NUM_THREADS')}}
    started = time.monotonic()
    peak = 0
    minimum_ram = available_ram()
    minimum_disk = shutil.disk_usage(work).free
    with (work/'reader.log').open('xb') as log:
        process = subprocess.Popen(command, cwd=work, env=env, stdout=log, stderr=log,
                                   start_new_session=True, preexec_fn=child_limits)
        try:
            while True:
                rss = 0
                for path in Path('/proc').iterdir():
                    if not path.name.isdecimal():
                        continue
                    try:
                        fields = (path/'stat').read_text().rpartition(')')[2].split()
                        if int(fields[2]) == process.pid or int(path.name) == os.getpid():
                            rss += int(fields[21])*os.sysconf('SC_PAGE_SIZE')
                    except (FileNotFoundError, ProcessLookupError):
                        continue
                peak = max(peak, rss)
                minimum_ram = min(minimum_ram, available_ram())
                minimum_disk = min(minimum_disk, shutil.disk_usage(work).free)
                _require(rss <= 3*GIB and minimum_ram >= 2*GIB and minimum_disk >= reserve_disk,
                         'reader exceeded RAM/disk reserve')
                _require(time.monotonic()-started <= 300, 'reader wall timeout')
                if process.poll() is not None:
                    break
                time.sleep(.1)
            _require(process.returncode == 0, f'reader exit {process.returncode}; see reader.log')
        finally:
            # Kill/reap only this new process group, including wrapper descendants on failure.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
    return dict(reader_seconds=time.monotonic()-started, observed_aggregate_rss_bytes=peak,
                child_peak_rss_bytes=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss*1024,
                minimum_available_ram_bytes=minimum_ram, minimum_free_disk_bytes=minimum_disk,
                child_address_space_cap_bytes=2*GIB, aggregate_rss_cap_bytes=3*GIB,
                reserve_ram_bytes=2*GIB, reserve_disk_bytes=reserve_disk,
                reader_timeout_seconds=300, obj_byte_cap=output_cap)


def check_reader(segment, work, apprun):
    segment, work, apprun = Path(segment).resolve(), Path(work).resolve(), Path(apprun).resolve()
    _require(not work.is_relative_to(segment) and not segment.is_relative_to(work),
             'reader work must be separate from source')
    shape, _ = _headers(segment)
    before = _snapshot(segment)
    reader = _record(apprun)
    work.mkdir()  # Exclusive ownership; never reuse another job's evidence.
    command = [str(apprun), 'vc_tifxyz2obj', str(segment), str(work/'roundtrip.obj'),
               '--keep=100', '--no-inpaint']
    try:
        resources = _run_reader(command, work, math.prod(shape))
    finally:
        _require(before == _snapshot(segment), 'source artifacts changed during reader')
    proof = check_obj(segment, work/'roundtrip.obj')
    _require(before == _snapshot(segment), 'source artifacts changed during acceptance')
    proof.update(command=command, reader=reader, resources=resources,
                 obj=_record(work/'roundtrip.obj'))
    with (work/'READER_VERIFIED.json').open('x') as stream:
        json.dump(proof, stream, indent=2, sort_keys=True)
        stream.write('\n')
    return proof


def _fixture(root, shape=(2, 3), scale=(0.5, 0.5), hole=False,
             diagonal=0, separate_ids=True):
    """Absolute volume coordinates, not ROI-local XYZ; TIFFs have 256-pixel tiles."""
    segment = root / 'level-0000'
    segment.mkdir()
    rows, cols = np.indices(shape)
    xyz = np.stack((1100.1234 + cols * .75, 2300.8765 + rows * .875,
                    4200.2345 + cols * .03125 + rows * .0625), axis=-1).astype('float32')
    valid = np.ones(shape, bool)
    if hole:
        valid[shape[0] // 2, shape[1] // 2] = False
        xyz[~valid] = -1
    for axis, name in enumerate('xyz'):
        tifffile.imwrite(segment / f'{name}.tif', xyz[..., axis], tile=(256, 256),
                         compression='deflate', metadata=None)
    meta = dict(format='tifxyz', type='seg', uuid='reader-control', shape_yx=list(shape),
                dtype='float32', scale=list(scale), pitch_um=17.28,
                uv_min_um=[123.5, -41.25], level=-4,
                output_frame=dict(origin_xyz_um=[86.4, 172.8, 259.2], voxel_um=8.64,
                                  shape_zyx=[10000]*3))
    (segment / 'meta.json').write_text(json.dumps(meta))
    obj = root / 'expected.obj'
    ids = np.full(shape, -1, np.int32)
    ids[valid] = np.arange(valid.sum())
    count = int(valid.sum())
    with obj.open('w') as stream:
        for row, col in zip(*np.nonzero(valid)):
            stream.write('v ' + ' '.join(f'{v:.6f}' for v in xyz[row, col]) + '\n')
        indices = np.flatnonzero(valid)
        for index in indices[::-1] if separate_ids else indices:
            row, col = divmod(int(index), shape[1])
            stream.write(f'vt {np.float32(col/scale[0]):.6f} {np.float32(row/scale[1]):.6f}\n')
        for row in range(shape[0]-1):
            for col in range(shape[1]-1):
                a, b, c, d = (int(ids[row, col]), int(ids[row, col+1]),
                              int(ids[row+1, col]), int(ids[row+1, col+1]))
                if min(a, b, c, d) < 0:
                    continue
                triangles = ((a,b,c), (b,d,c)) if diagonal == 0 else ((a,b,d), (a,d,c))
                for face in triangles:
                    stream.write('f ' + ' '.join(f'{v+1}/{count-v if separate_ids else v+1}'
                                                for v in face) + '\n')
    return segment, obj


@pytest.mark.parametrize('diagonal', [0, 1])
def test_absolute_xyz_nonunit_scale_separate_uv_ids(tmp_path, diagonal):
    segment, obj = _fixture(tmp_path, diagonal=diagonal)
    proof = check_obj(segment, obj)
    assert proof['vertices'] == proof['valid_nodes'] == 6
    assert proof['faces'] == 4
    assert proof['complete_cells'] == 2
    assert proof['all_valid_nodes_recovered']
    assert proof['source_artifacts_unchanged']
    assert proof['maximum_xyz_error_input_units'] <= 5.1e-7


def test_partial_256_tails_and_mask_hole(tmp_path):
    segment, obj = _fixture(tmp_path, (259, 258), hole=True, diagonal=1)
    proof = check_obj(segment, obj)
    assert proof['valid_nodes'] == 259*258-1
    assert proof['complete_cells'] == 258*257-4
    assert proof['faces'] == 2*(258*257-4)


def _publish_control(segment, status='PARTIAL', policy='allow_partial'):
    """A bound publication protocol fixture, not a source-coverage oracle."""
    root = segment.parent
    flags = dict(physical_acceptance=False, native_face_identity=False,
                 native_topology_equivalence='UNMEASURED')
    quality = dict(status=status, coverage_complete=status=='READY')
    identity = dict(effective_settings=dict(coverage_policy=policy, levels=[-4]), **flags)
    identity_sha = hashlib.sha256(json.dumps(identity, sort_keys=True,
                                 separators=(',', ':')).encode()).hexdigest()
    (root/'EXPORT.json').write_text(json.dumps(identity))
    (segment/'VERIFIED.json').write_text(json.dumps(dict(**quality, **flags)))
    receipt = dict(identity_sha256=identity_sha, levels=[dict(segment=segment.name, level=-4,
                   manifest=dict(files={'VERIFIED.json':_record(segment/'VERIFIED.json')}), **quality)],
                   **quality, **flags)
    (root/'TIFXYZ.json').write_text(json.dumps(receipt))
    (root/(status+'.json')).write_text(json.dumps(dict(identity_sha256=identity_sha,
        tifxyz_sha256=_record(root/'TIFXYZ.json')['sha256'], **quality, **flags)))


@pytest.mark.parametrize('status, complete', [('PARTIAL', False), ('READY', True)])
def test_reader_proof_preserves_publication_coverage(tmp_path, status, complete):
    segment, obj = _fixture(tmp_path)
    _publish_control(segment, status)
    proof = check_obj(segment, obj)
    assert proof['publication_status']==status
    assert proof['coverage_complete'] is complete
    assert proof['source_coverage_verified'] is False
    assert proof['independent_root_reprojection'] is False
    assert str(tmp_path/(status+'.json')) in proof['source_artifacts']


def test_reader_only_control_does_not_claim_coverage(tmp_path):
    segment, obj = _fixture(tmp_path)
    proof = check_obj(segment, obj)
    assert 'coverage_complete' not in proof
    assert proof['source_coverage_verified'] is False
    assert proof['independent_root_reprojection'] is False


@pytest.mark.parametrize('damage', ['both-markers', 'dangling-marker', 'missing-marker', 'root-coverage', 'root-status',
                                  'marker-coverage', 'strict-policy', 'identity-sha', 'tifxyz-sha',
                                  'level-coverage', 'verified-coverage'])
def test_reader_refuses_corrupt_publication_quality(tmp_path, damage):
    segment, obj = _fixture(tmp_path)
    _publish_control(segment, policy='strict' if damage=='strict-policy' else 'allow_partial')
    marker = tmp_path/'PARTIAL.json'
    if damage=='both-markers':
        (tmp_path/'READY.json').write_bytes(marker.read_bytes())
    elif damage=='dangling-marker':
        (tmp_path/'READY.json').symlink_to(tmp_path/'missing')
    elif damage=='missing-marker':
        marker.unlink()
    elif damage in ('root-coverage', 'root-status', 'level-coverage', 'verified-coverage'):
        receipt = _json(tmp_path/'TIFXYZ.json')
        if damage=='root-status': receipt['status']='READY'
        elif damage=='level-coverage': receipt['levels'][0]['coverage_complete']=True
        elif damage=='verified-coverage':
            verified = _json(segment/'VERIFIED.json'); verified['coverage_complete']=True
            (segment/'VERIFIED.json').write_text(json.dumps(verified))
            receipt['levels'][0]['manifest']['files']['VERIFIED.json']=_record(segment/'VERIFIED.json')
        else: receipt['coverage_complete']=True
        (tmp_path/'TIFXYZ.json').write_text(json.dumps(receipt))
        data = _json(marker); data['tifxyz_sha256']=_record(tmp_path/'TIFXYZ.json')['sha256']
        marker.write_text(json.dumps(data))
    elif damage in ('marker-coverage', 'identity-sha', 'tifxyz-sha'):
        data = _json(marker)
        key = {'marker-coverage':'coverage_complete', 'identity-sha':'identity_sha256',
               'tifxyz-sha':'tifxyz_sha256'}[damage]
        data[key] = True if damage=='marker-coverage' else '0'*64
        marker.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        check_obj(segment, obj)


def test_partial_marker_mutation_during_reader_is_rejected(tmp_path, monkeypatch):
    import shutil
    segment, obj = _fixture(tmp_path)
    _publish_control(segment)
    def reader(command, work, pixels):
        shutil.copyfile(obj, work/'roundtrip.obj')
        marker = tmp_path/'PARTIAL.json'
        marker.write_text(marker.read_text()+' ')
        return {}
    monkeypatch.setattr(__import__(__name__), '_run_reader', reader)
    with pytest.raises(ValueError, match='source artifacts changed'):
        check_reader(segment, tmp_path/'reader', obj)


def test_reader_interleaves_vertices_uv_normals_and_faces(tmp_path):
    segment, obj = _fixture(tmp_path, separate_ids=False)
    lines = obj.read_text().splitlines()
    declarations = [[lines[i], lines[i+6], 'vn 0 0 1'] for i in range(6)]
    faces = ['f '+' '.join(word+'/'+word.split('/')[0] for word in line.split()[1:])
             for line in lines[12:]]
    # Native reader emits each cell's new vertices immediately before its faces.
    obj.write_text('\n'.join([*sum(declarations[:4], []), faces[0],
                              *sum(declarations[4:], []), *faces[1:]])+'\n')
    assert check_obj(segment, obj)['faces'] == 4


@pytest.mark.parametrize('damage', [
    'missing', 'duplicate_same_count', 'third_same_count', 'cross_cell',
    'quad', 'bad_index', 'missing_uv', 'nan', 'truncated', 'extra_vertex',
    'wrong_mapping', 'wrong_xyz', 'wrong_uv', 'wrong_scale', 'corrupt_tiff',
])
def test_rejects_corrupt_reader_output(tmp_path, damage):
    segment, obj = _fixture(tmp_path)
    lines = obj.read_text().splitlines()
    faces = [i for i, line in enumerate(lines) if line.startswith('f ')]
    if damage == 'missing':
        lines.pop()
    elif damage == 'duplicate_same_count':
        lines[faces[1]] = lines[faces[0]]
    elif damage == 'third_same_count':
        lines[faces[2]] = 'f 1/6 2/5 5/2'
    elif damage == 'cross_cell':
        lines[faces[0]] = 'f 1/6 3/4 5/2'
    elif damage == 'quad':
        lines[faces[0]] += ' 5/2'
    elif damage == 'bad_index':
        lines[faces[0]] = 'f 0/6 2/5 4/3'
    elif damage == 'missing_uv':
        lines[faces[0]] = 'f 1 2 4'
    elif damage == 'nan':
        lines[0] = 'v nan 2300 4200'
    elif damage == 'truncated':
        lines[0] = 'v 1100 2300'
    elif damage == 'extra_vertex':
        lines.insert(0, lines[0])
    elif damage == 'wrong_mapping':
        lines[faces[0]] = 'f 1/5 2/6 4/3'
    elif damage == 'wrong_xyz':
        lines[0] = 'v 1100 2300 4200'
    elif damage == 'wrong_uv':
        lines[6] = 'vt 4.25 2'
    elif damage == 'wrong_scale':
        meta = json.loads((segment/'meta.json').read_text())
        meta['scale'] = [1., 1.]
        (segment/'meta.json').write_text(json.dumps(meta))
    elif damage == 'corrupt_tiff':
        (segment/'z.tif').write_bytes(b'broken')
    obj.write_text('\n'.join(lines)+'\n')
    with pytest.raises(ValueError):
        check_obj(segment, obj)


def test_header_pixel_cap_precedes_any_decode(tmp_path, monkeypatch):
    segment, obj = _fixture(tmp_path)
    # A real sparse TIFF header exceeds the hard limit without a large fixture array.
    tifffile.imwrite(segment/'z.tif', shape=(2049, 2048), dtype='float32',
                     tile=(256, 256), metadata=None)
    def forbidden(*args, **kwargs):
        pytest.fail('decoded an image before validating ALL three headers')
    monkeypatch.setattr(tifffile.TiffPage, 'asarray', forbidden)
    with pytest.raises(ValueError, match='pixel cap'):
        check_obj(segment, obj)


@pytest.mark.parametrize('damage', ['mtime', 'bytes'])
def test_source_change_during_reader_is_rejected(tmp_path, monkeypatch, damage):
    import shutil
    segment, obj = _fixture(tmp_path)
    def reader(command, work, pixels):
        shutil.copyfile(obj, work/'roundtrip.obj')
        path = segment/'meta.json'
        info = path.stat()
        if damage == 'mtime':
            os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns+1_000_000))
        else:
            path.write_text(path.read_text().replace('reader-control', 'reader-corrupt'))
            os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns))
        return {}
    monkeypatch.setattr(__import__(__name__), '_run_reader', reader)
    with pytest.raises(ValueError, match='source artifacts changed'):
        check_reader(segment, tmp_path/'reader', obj)
    assert not (tmp_path/'reader/READER_VERIFIED.json').exists()


def test_reader_wrapper_writes_proof_outside_readonly_source(tmp_path, monkeypatch):
    import shutil
    segment, obj = _fixture(tmp_path)
    before = {p.name:(p.read_bytes(), p.stat().st_mtime_ns) for p in segment.iterdir()}
    def reader(command, work, pixels):
        assert command[1:] == ['vc_tifxyz2obj', str(segment), str(work/'roundtrip.obj'),
                               '--keep=100', '--no-inpaint']
        shutil.copyfile(obj, work/'roundtrip.obj')
        return {}
    monkeypatch.setattr(__import__(__name__), '_run_reader', reader)
    for path in segment.iterdir():
        path.chmod(0o444)
    segment.chmod(0o555)
    try:
        proof = check_reader(segment, tmp_path/'reader', obj)
        assert proof['source_artifacts_unchanged']
        assert json.loads((tmp_path/'reader/READER_VERIFIED.json').read_text())['faces'] == 4
        assert before == {p.name:(p.read_bytes(), p.stat().st_mtime_ns) for p in segment.iterdir()}
    finally:
        segment.chmod(0o755)


@pytest.mark.parametrize('location', ['source', 'existing'])
def test_reader_refuses_unsafe_work_directory(tmp_path, location):
    segment, obj = _fixture(tmp_path)
    work = segment/'new' if location == 'source' else tmp_path
    with pytest.raises((ValueError, FileExistsError)):
        check_reader(segment, work, obj)


@pytest.mark.parametrize('damage', ['hole_face', 'orphan', 'xyz_precision', 'flipped',
                                  'extra_face', 'extra_uv', 'mixed_mask'])
def test_additional_mask_and_topology_negatives(tmp_path, damage):
    segment, obj = _fixture(tmp_path, (5, 5), hole=True)
    lines = obj.read_text().splitlines()
    if damage == 'hole_face':
        # IDs 7,8,12 lie on the three valid corners beside the invalid centre.
        lines.append('f 7/18 8/17 12/13')
    elif damage in ('orphan', 'mixed_mask'):
        for name in ('xyz' if damage == 'orphan' else 'y'):
            path = segment/f'{name}.tif'
            data = tifffile.imread(path)
            if damage == 'orphan':
                data[0, 1] = data[1, 0] = -1
            else:
                data[0, 0] = -1
            tifffile.imwrite(path, data, tile=(256,256), compression='deflate', metadata=None)
    elif damage == 'xyz_precision':
        words = lines[0].split()
        words[1] = str(float(words[1])+2e-6)
        lines[0] = ' '.join(words)
    elif damage == 'flipped':
        index = next(i for i, line in enumerate(lines) if line.startswith('f '))
        words = lines[index].split()
        lines[index] = ' '.join([words[0], *words[:0:-1]])
    elif damage == 'extra_face':
        lines.append(lines[-1])
    else:
        lines.insert(24, 'vt 0 0')
    obj.write_text('\n'.join(lines)+'\n')
    with pytest.raises(ValueError):
        check_obj(segment, obj)


@pytest.mark.skipif(not (os.environ.get('VC3D_APPRUN') and
                         os.environ.get('TIFXYZ_VC3D_CONTROL') == '1'),
                    reason='optional real VC3D control requires explicit environment opt-in')
def test_real_vc3d_control(tmp_path):
    segment, _ = _fixture(tmp_path, (259, 258), hole=True)
    proof = check_reader(segment, tmp_path/'reader', Path(os.environ['VC3D_APPRUN']))
    assert proof['valid_nodes'] == 259*258-1
    assert proof['faces'] == 2*(258*257-4)


def _main():
    import argparse
    import resource
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('segment', type=Path)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--apprun', type=Path, required=True)
    args = parser.parse_args()
    segment = args.segment.resolve()
    _require(not args.work.resolve().is_relative_to(segment.parent),
             'CLI work must be outside the accepted export namespace')
    _publication(segment, required=True)
    _require(sys.platform == 'linux', 'real-reader resource acceptance requires Linux')
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    os.nice(max(0, 19-os.getpriority(os.PRIO_PROCESS, 0)))
    # Parent decoding happens after reader exits; cap both processes independently.
    resource.setrlimit(resource.RLIMIT_AS, (768 << 20, 3*GIB))
    resource.setrlimit(resource.RLIMIT_CPU, (600, 600))
    print(json.dumps(check_reader(segment, args.work, args.apprun), indent=2))


if __name__ == '__main__':
    _main()
