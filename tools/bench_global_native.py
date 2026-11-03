"""Experimental source→cut→MVC→ARAP→qualification→native render command.

Frame JSON requires id, origin_xyz_um, spacing_xyz_um, shape_zyx and voxel_um.
The ID must match mesh.metadata.frame; mesh coordinates must already be micrometres
(metadata.voxel_um=1). Component is an explicit mesh.sqlite component root ID.
Each phase is a separate supervised worker. Remove OUTPUT/STOP before --resume;
clean pauses do not consume active time. VC3D acceptance is a separate operation.
STATUS.json reports progress/pauses/failures. A supervised verification worker
checks every publication; PUBLICATION.json is its final seal. Completed resumes
consume the remaining active budget and leave the original publication unchanged.
"""
import argparse
from contextlib import closing
import fcntl
import hashlib
import importlib
from importlib.metadata import version, PackageNotFoundError
from importlib.util import find_spec
import json
import math
import os
from pathlib import Path
import re
import resource
import shutil
import signal
import sqlite3
import stat
import sys
import time
import uuid
from zipfile import ZipFile

from winding_field.util import atomic_json, digest, file_sha

PHASES = ('source', 'cut', 'mvc', 'arap', 'qualification', 'render')
ACTIONS = (*PHASES, 'verify')
CHUNK_FACES = 8192
MAX_FACES, MAX_VERTICES = 1_250_000, 750_000
UNIT_BYTES, JSON_BYTES = 64 << 20, 16 << 20
RSS_BYTES, SCRATCH_BYTES, MARGIN = 3 << 30, 16 << 30, 64 << 20


def parser(worker=False):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--mesh', type=Path, required=not worker)
    p.add_argument('--component', type=int, required=not worker)
    p.add_argument('--ct', required=not worker)
    p.add_argument('--ct-local', type=Path)
    p.add_argument('--frame', type=Path, required=not worker)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--pitch-um', type=float, required=not worker)
    p.add_argument('--coverage-target', type=float, default=.9)
    p.add_argument('--seconds', type=float, default=1800.)
    p.add_argument('--reserve-gib', type=float, default=12.25)
    p.add_argument('--reserve-ram-gib', type=float, default=2.)
    p.add_argument('--export-tifxyz', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--worker', choices=ACTIONS, help=argparse.SUPPRESS)
    p.add_argument('--attempt', help=argparse.SUPPRESS)
    p.add_argument('--token', help=argparse.SUPPRESS)
    return p


def file_signature(path):
    value = Path(path).lstat()
    if not stat.S_ISREG(value.st_mode):
        raise ValueError('regular file required (no symlink/FIFO): ' + str(path))
    return stat_signature(value)


def stat_signature(value):
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def capture_file(path, limit=None):
    """Hash the same open regular file that was read; nonblocking open rejects FIFOs."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    except OSError as error:
        raise ValueError('regular readable file required: ' + str(path)) from error
    with os.fdopen(fd, 'rb') as stream:
        info = os.fstat(stream.fileno()); before = stat_signature(info)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError('regular file required: ' + str(path))
        if limit is not None and info.st_size > limit:
            raise ValueError('JSON envelope exceeded')
        raw = None if limit is None else stream.read(limit + 1)
        if raw is not None and len(raw) > limit:
            raise ValueError('JSON envelope exceeded')
        sha = hashlib.file_digest(stream, 'sha256').hexdigest() if raw is None else hashlib.sha256(raw).hexdigest()
        if stat_signature(os.fstat(stream.fileno())) != before or file_signature(path) != before:
            raise ValueError('file changed during capture: ' + str(path))
    return raw, sha, before


def read_json_record(path, limit=JSON_BYTES):
    raw, sha, signature = capture_file(path, limit)
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate JSON key')
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('nonfinite JSON: ' + value)
    value = json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)
    if file_signature(path) != signature:
        raise ValueError('file changed after parsing: ' + str(path))
    return value, sha, signature


def read_json(path, limit=JSON_BYTES):
    return read_json_record(path, limit)[0]


def safe(root, name):
    if not isinstance(name, str) or '\\' in name or any(p in ('', '.', '..') for p in name.split('/')):
        raise ValueError('unsafe relative payload path')
    path = root/name
    if not path.is_relative_to(root) or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('payload escapes output or contains symlink')
    return path


def validate_paths(mesh, out, ct, local, frame):
    if any(p.is_symlink() for p in (out, *out.parents)):
        raise ValueError('symlink in output root')
    out = out.resolve()
    sources = [mesh, frame]
    if not ct.startswith(('http://', 'https://', 's3://')):
        sources.append(Path(ct))
    if local:
        sources.append(local)
    for source in sources:
        resolved = source.resolve()
        if out.is_relative_to(resolved) or resolved.is_relative_to(out):
            raise ValueError('output/source containment overlap')


def runtime_binding():
    from winding_field.workflow import implementation_sha
    versions = {}
    for name in ('numpy', 'scipy', 'libigl', 'tifffile'):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return dict(implementation_sha256=implementation_sha(), tool_sha256=file_sha(__file__), versions=versions)


def make_request(args):
    if args.mesh is None or args.component is None or not args.ct or args.frame is None or args.pitch_um is None:
        raise ValueError('explicit --mesh, --component, --ct, --frame and --pitch-um required')
    if args.component < 1 or not 0 < args.coverage_target <= 1:
        raise ValueError('invalid component/coverage target')
    for name, value, positive in (('pitch', args.pitch_um, True), ('seconds', args.seconds, True),
                                  ('disk reserve', args.reserve_gib, False), ('RAM reserve', args.reserve_ram_gib, False)):
        if not math.isfinite(value) or value < 0 or (positive and value == 0):
            raise ValueError('invalid finite ' + name)
    out = args.output.absolute()
    validate_paths(args.mesh, out, args.ct, args.ct_local, args.frame)
    frame, frame_sha, frame_signature = read_json_record(args.frame, 64 << 10)
    if set(frame) != {'id', 'origin_xyz_um', 'spacing_xyz_um', 'shape_zyx', 'voxel_um'}:
        raise ValueError('explicit physical frame fields required')
    if not isinstance(frame['id'], str) or not frame['id']:
        raise ValueError('invalid frame ID')
    for name in ('origin_xyz_um', 'spacing_xyz_um', 'shape_zyx'):
        values = frame[name]
        if (not isinstance(values, list) or len(values) != 3
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in values)
                or (name != 'origin_xyz_um' and min(values) <= 0)
                or (name == 'shape_zyx' and any(type(v) is not int for v in values))):
            raise ValueError('invalid physical frame ' + name)
    unit = frame['voxel_um']
    if type(unit) not in (int, float) or not math.isfinite(unit) or unit <= 0:
        raise ValueError('invalid finite frame voxel_um')
    manifest, mesh_sha, mesh_signature = read_json_record(args.mesh/'MESH.json')
    levels = manifest.get('levels')
    if (not isinstance(levels, list) or len(levels) != 1 or not isinstance(levels[0], dict)
            or type(levels[0].get('level')) not in (int, float) or not math.isfinite(levels[0]['level'])):
        raise ValueError('single finite mesh level required')
    if file_signature(args.mesh/'MESH.json') != mesh_signature or file_signature(args.frame) != frame_signature:
        raise ValueError('source metadata changed after capture')
    ct = args.ct if args.ct.startswith(('http://', 'https://', 's3://')) else str(Path(args.ct).resolve())
    return dict(schema='experimental-global-native-v3', output=str(out.resolve()), mesh=str(args.mesh.resolve()),
        mesh_manifest_sha256=mesh_sha, level=float(levels[0]['level']), component=args.component, ct=ct,
        ct_local=str(args.ct_local.resolve()) if args.ct_local else None, frame=frame,
        frame_path=str(args.frame.resolve()), frame_sha256=frame_sha, pitch_um=args.pitch_um,
        coverage_target=args.coverage_target, seconds=args.seconds, export_tifxyz=args.export_tifxyz,
        reserve_bytes=int(args.reserve_gib*(1 << 30)), reserve_ram_bytes=int(args.reserve_ram_gib*(1 << 30)),
        cache_bytes=2 << 30, aggregate_rss_bytes=RSS_BYTES, scratch_bytes=SCRATCH_BYTES)


def load_unit(path):
    """Reject oversized/forged NPY headers before np.load can allocate anything."""
    import numpy as np
    with ZipFile(path) as archive:
        infos = archive.infolist()
        if not 0 < len(infos) <= 32 or len({i.filename for i in infos}) != len(infos):
            raise ValueError('numeric unit member envelope')
        charge = 0
        for info in infos:
            if not re.fullmatch(r'[a-z_][a-z_0-9]*\.npy', info.filename):
                raise ValueError('invalid numeric unit member')
            with archive.open(info) as member:
                v = np.lib.format.read_magic(member)
                if v not in ((1, 0), (2, 0)):
                    raise ValueError('unsupported NPY header')
                read_header = np.lib.format.read_array_header_1_0 if v == (1, 0) else np.lib.format.read_array_header_2_0
                shape, _, dtype = read_header(member)
                size = math.prod(shape)*dtype.itemsize
                if not 1 <= len(shape) <= 3 or dtype.kind not in 'buif' or dtype.hasobject:
                    raise ValueError('invalid numeric header shape/dtype')
                charge += size
                if charge > UNIT_BYTES or info.file_size != member.tell() + size:
                    raise ValueError('numeric header/data byte budget differs')
        if sum(i.file_size for i in infos) > UNIT_BYTES + (64 << 10):
            raise ValueError('numeric unit envelope exceeded')
    with np.load(path, allow_pickle=False) as arrays:
        return {name: arrays[name] for name in arrays.files}


def phase_identity(out, phase, request):
    return dict(producer='experimental-global-native-v3', phase=phase, level=request['level'], request_sha256=digest(request),
                runtime_sha256=read_json_record(safe(out, 'RUNTIME.json'))[1],
                inputs={p: read_json_record(safe(out/p, 'PHASE.json'))[1] for p in PHASES[:PHASES.index(phase)]})


def verify_phase(out, phase, request, check=lambda: None):
    check()
    root = safe(out, phase); marker = safe(root, 'PHASE.json')
    if not marker.exists():
        return None
    record = read_json(marker)
    if record['identity'] != phase_identity(out, phase, request) or read_json(safe(root, 'IDENTITY.json')) != record['identity']:
        raise ValueError('phase identity changed')
    if file_sha(safe(root, 'work.sqlite')) != record['database_sha256']:
        raise ValueError('committed phase database changed')
    with closing(sqlite3.connect((root/'work.sqlite').resolve().as_uri()+'?mode=ro&immutable=1', uri=True)) as db:
        files = {name: dict(sha256=sha, bytes=size) for name, sha, size in db.execute('SELECT name,sha,bytes FROM files')}
    if files != record['files']:
        raise ValueError('phase file ledger changed')
    for name, expected in files.items():
        check()
        path = safe(root, name)
        if path.stat().st_size != expected['bytes'] or file_sha(path) != expected['sha256']:
            raise ValueError('committed phase payload changed: ' + name)
    if record['report'] != read_json(safe(root, 'REPORT.json')):
        raise ValueError('phase report changed independently of committed payload')
    check()
    return record


def unit(out, phase, request):
    if verify_phase(out, phase, request) is None:
        raise ValueError('uncommitted phase: ' + phase)
    arrays = load_unit(out/phase/'data.npz')
    for value in arrays.values():
        value.setflags(write=False)
    return arrays


def publish_report(store, name, report):
    if store.db.execute('SELECT 1 FROM files WHERE name=?', (name,)).fetchone():
        if read_json(store.validate_file(name)) != report:
            raise ValueError('immutable phase report changed: ' + name)
        return
    atomic_json(store.path(name), report)
    store.register(name)


def open_volume(out, request):
    from winding_field.io import Volume
    volume = Volume(request['ct'], local=request['ct_local'], voxel_um=request['frame']['voxel_um'],
                    cache=out/'ct-cache', cache_bytes=request['cache_bytes'], reserve_bytes=request['reserve_bytes'])
    frame = request['frame']
    if (list(volume.shape) != frame['shape_zyx'] or list(volume.spacing_xyz) != frame['spacing_xyz_um']
            or list(volume.origin_xyz) != frame['origin_xyz_um']):
        volume.close()
        raise ValueError('original CT physical frame differs')
    return volume


def check_sources(request, out):
    mesh = Path(request['mesh'])
    signatures = {name: file_signature(mesh/name) for name in ('MESH.json', 'RUN.json', 'mesh.sqlite')}
    manifest, manifest_sha, _ = read_json_record(mesh/'MESH.json')
    if manifest_sha != request['mesh_manifest_sha256']:
        raise ValueError('source mesh manifest changed')
    frame, frame_sha, _ = read_json_record(request['frame_path'], 64 << 10)
    if frame_sha != request['frame_sha256'] or frame != request['frame']:
        raise ValueError('source frame changed')
    runtime = runtime_binding(); path = safe(out, 'RUNTIME.json')
    if path.exists():
        if read_json(path) != runtime:
            raise ValueError('runtime identity changed')
    elif any((out/p).exists() for p in PHASES):
        raise ValueError('missing runtime binding for existing phases')
    else:
        atomic_json(path, runtime)
    record = verify_phase(out, 'source', request)
    if record:
        for name, expected in record['report']['source_files'].items():
            if capture_file(mesh/name)[1] != expected:
                raise ValueError('source mesh changed: ' + name)
        if record['report']['source_files']['mesh.sqlite'] != manifest['database_sha256']:
            raise ValueError('source database digest contradicts validated manifest')
    if any(file_signature(mesh/name) != expected for name, expected in signatures.items()):
        raise ValueError('source changed during verification')
    return record


def source_phase(store, request, out):
    import numpy as np
    from winding_field.surface_mesh import MeshReader
    started = time.monotonic(); mesh = Path(request['mesh'])
    signatures = {name: file_signature(mesh/name) for name in ('MESH.json', 'RUN.json', 'mesh.sqlite')}
    frame_signature = file_signature(request['frame_path'])
    with MeshReader(request['mesh']) as reader:
        mesh_open_s = time.monotonic() - started
        levels = reader.manifest['levels']; metadata = reader.metadata
        if (len(levels) != 1 or levels[0]['level'] != request['level']
                or reader.manifest_sha != request['mesh_manifest_sha256'] or type(metadata.get('voxel_um')) not in (int, float)
                or metadata['voxel_um'] != 1. or metadata.get('frame') != request['frame']['id']):
            raise ValueError('single-level genuine mesh with matching frame and voxel_um=1 required')
        count = reader.db.execute('SELECT count(*) FROM components').fetchone()[0]
        if count > 65536:
            raise ValueError('component metadata envelope exceeded')
        components = [dict(component=int(c), faces=int(n), area_um2=float(a)) for c, n, a in
                      reader.db.execute('SELECT root,faces,area FROM components ORDER BY root')]
        selected = next((c for c in components if c['component'] == request['component']), None)
        if selected is None or not 0 < selected['faces'] <= MAX_FACES:
            raise ValueError('explicit component absent or exceeds bounded face envelope')
        nf = selected['faces']
        global_faces = np.empty((nf, 3), np.int64); face_ids = np.empty(nf, np.int64); areas = np.empty(nf)
        cursor = reader.db.execute('SELECT id,a,b,c,area FROM faces WHERE parent=? ORDER BY id', (request['component'],))
        first = 0
        while rows := cursor.fetchmany(CHUNK_FACES):
            store.check()
            if first + len(rows) > nf:
                raise ValueError('source component face count differs')
            face_ids[first:first+len(rows)] = [r[0] for r in rows]
            global_faces[first:first+len(rows)] = [r[1:4] for r in rows]
            areas[first:first+len(rows)] = [r[4] for r in rows]
            first += len(rows)
        if first != nf or not np.isfinite(areas).all() or (areas <= 0).any():
            raise ValueError('invalid source face census')
        vertex_ids, remap = np.unique(global_faces, return_inverse=True)
        if len(vertex_ids) > MAX_VERTICES:
            raise ValueError('source vertex envelope exceeded')
        vertices = np.empty((len(vertex_ids), 3), np.float64)
        for first in range(0, len(vertex_ids), 512):
            store.check(); ids = vertex_ids[first:first+512]
            rows = reader.db.execute('SELECT id,x,y,z FROM vertices WHERE id IN (' + ','.join('?' for _ in ids) + ') ORDER BY id',
                                     list(map(int, ids))).fetchall()
            if not np.array_equal([r[0] for r in rows], ids):
                raise ValueError('missing original source vertex')
            vertices[first:first+len(ids)] = [r[1:] for r in rows]
        if not np.isfinite(vertices).all():
            raise ValueError('nonfinite original XYZ')
        store.save_arrays('data.npz', vertices=vertices, faces=remap.reshape(-1, 3).astype(np.int32),
                          vertex_ids=vertex_ids, face_ids=face_ids, areas=areas)
        excluded = reader.db.execute('SELECT id,parent FROM faces WHERE parent!=? ORDER BY id', (request['component'],))
        index = 0
        while rows := excluded.fetchmany(CHUNK_FACES):
            store.check()
            store.save_arrays(f'excluded-{index:06d}.npz', face_ids=np.array([r[0] for r in rows], np.int64),
                              components=np.array([r[1] for r in rows], np.int64), reasons=np.ones(len(rows), np.uint8))
            index += 1
        reader.validate()
        manifest, manifest_sha, _ = read_json_record(mesh/'MESH.json')
        binding, run_sha, _ = read_json_record(mesh/'RUN.json')
        _, database_sha, database_signature = capture_file(mesh/'mesh.sqlite')
        committed_binding = json.loads(reader.db.execute("SELECT value FROM meta WHERE key='binding'").fetchone()[0])
        if (manifest_sha != reader.manifest_sha or manifest != reader.manifest
                or database_sha != reader.manifest['database_sha256'] or database_signature != reader.signature
                or binding != committed_binding or binding['metadata'] != reader.metadata
                or binding['identity'] != reader.manifest['identity']):
            raise ValueError('source capture differs from validated manifest/database/RUN binding')
        reader.validate()
        if any(file_signature(mesh/name) != expected for name, expected in signatures.items()):
            raise ValueError('source changed across capture')
        report = dict(level=request['level'], selected_component=request['component'], selected_faces=nf, selected_vertices=len(vertices),
            source_faces=levels[0]['faces'], source_vertices=levels[0]['vertices'], source_area_um2=levels[0]['area_world2'],
            selected_area_um2=float(areas.sum()), excluded_components=[c for c in components if c != selected],
            exclusion_units=index, exclusion_reason_codes={'1': 'unselected_component'}, metadata=metadata,
            face_id_space='original mesh.sqlite faces.id', vertex_id_space='original mesh.sqlite vertices.id',
            source_files={'MESH.json': manifest_sha, 'RUN.json': run_sha, 'mesh.sqlite': database_sha},
            mesh_open_s=mesh_open_s)
    phase = time.monotonic()
    with closing(open_volume(out, request)) as volume:
        report.update(ct_identity=volume.identity, ct_open_s=time.monotonic()-phase, ct_cache_initial_bytes=volume.stats()['cache_bytes'])
    if (any(file_signature(mesh/name) != expected for name, expected in signatures.items())
            or file_signature(request['frame_path']) != frame_signature):
        raise ValueError('source changed before source checkpoint')
    return report


def register_global(store, source, cut, uv, valid, component, whole_faces, whole_area):
    import numpy as np
    from winding_field.native_geometry import _schema
    _schema(store)
    selected = np.flatnonzero(valid)
    with store.db:
        store.set('source_faces', whole_faces); store.set('source_area', whole_area)
    first = store.get('global_face_cursor', 0)
    for offset in range(first, len(selected), CHUNK_FACES):
        store.phase = 'register'; store.check()
        use = selected[offset:offset+CHUNK_FACES]
        ids, local = np.unique(cut['faces'][use], return_inverse=True); faces = local.reshape(-1, 3)
        vertices = cut['vertices'][ids]; values = uv[ids]
        name = f'global-{offset:012d}'; uv_path = 'uv/'+name+'.npz'
        store.save_arrays('charts/'+name+'.npz', vertices=vertices, faces=faces, uv=values,
            face_ids=source['face_ids'][use], vertex_ids=ids.astype(np.int64),
            source_vertex_ids=source['vertex_ids'][cut['parent'][ids]])
        store.save_arrays(uv_path, uv=values)
        edge = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
        keys = ids[edge]; signs = np.where(keys[:, 0] < keys[:, 1], 1, -1)
        order = np.argsort(keys, axis=1); edge = np.take_along_axis(edge, order, axis=1); keys = np.take_along_axis(keys, order, axis=1)
        _, starts, counts = np.unique(keys, axis=0, return_index=True, return_counts=True)
        rows = [(int(keys[i, 0]), int(keys[i, 1]), name, int(edge[i, 0]), int(edge[i, 1]), int(signs[i])) for i in starts[counts == 1]]
        low, high = values.min(0), values.max(0)
        with store.db:
            store.db.execute('INSERT OR IGNORE INTO charts(name,component,area,faces,status,uv_path,minx,maxx,miny,maxy,quality) '
                'VALUES(?,?,?,?,?,?,?,?,?,?,?)', (name, component, float(source['areas'][use].sum()), len(use), 'MAPPED', uv_path,
                float(low[0]), float(high[0]), float(low[1]), float(high[1]), json.dumps({'global_uv_exact': True, 'vertex_ids': 'cut-node'})))
            store.db.executemany('INSERT OR IGNORE INTO edges VALUES(?,?,?,?,?,?)', rows)
            store.set('global_face_cursor', offset+len(use))
    with store.db:
        store.set('mapped_area', float(source['areas'][valid].sum())); store.set('flatten_complete', True)


def execute_phase(out, phase, request, remaining_s, stop=None):
    import numpy as np
    from winding_field.native_store import Journal
    from winding_field.native_store import Paused
    started = time.monotonic()
    def check():
        if stop and stop():
            raise Paused('STOP')
        if time.monotonic()-started >= remaining_s:
            raise Paused('TIME_LIMIT')
    check()
    check_sources(request, out)
    old = verify_phase(out, phase, request, check)
    if old:
        return old
    check()
    identity = phase_identity(out, phase, request); root = out/phase
    prior = 0.
    if (root/'work.sqlite').exists():
        with closing(sqlite3.connect((root/'work.sqlite').resolve().as_uri()+'?mode=rw', uri=True)) as db:
            row = db.execute("SELECT value FROM state WHERE name='elapsed_s'").fetchone()
            prior = float(json.loads(row[0])) if row else 0.
    with Journal(root, identity, stop, max(.001, prior+remaining_s-(time.monotonic()-started)), request['reserve_bytes']) as store:
        store.phase = phase; store.check()
        report_committed = store.db.execute("SELECT 1 FROM files WHERE name='REPORT.json'").fetchone()
        if phase == 'render' and store.db.execute("SELECT 1 FROM files WHERE name='LEVEL.json'").fetchone():
            from winding_field.native_pipeline import verify_ct_snapshot
            with closing(open_volume(out, request)) as volume:
                if volume.identity != verify_phase(out, 'source', request)['report']['ct_identity']:
                    raise ValueError('original CT identity changed')
                if (root/'CT-SNAPSHOT.sqlite').exists():
                    verify_ct_snapshot(root, volume)
            report = read_json(store.validate_file('LEVEL.json'))
            publish_report(store, 'REPORT.json', report)
            report_committed = True
        if not report_committed:
            if phase == 'source':
                report = source_phase(store, request, out)
            else:
                source_record = verify_phase(out, 'source', request)['report']
                source = unit(out, 'source', request) if phase in ('cut', 'qualification', 'render') else None
                if phase == 'cut':
                    math_api = importlib.import_module('winding_field.native_global')
                    arrays, report = math_api.cut_component(source['vertices'], source['faces'], check=store.check)
                    parent = arrays['parent']; faces = arrays['faces']
                    if (parent.dtype != np.int64 or parent.ndim != 1 or len(parent) > MAX_VERTICES
                            or parent.min() < 0 or parent.max() >= len(source['vertices'])
                            or faces.shape != source['faces'].shape or faces.dtype != np.int32
                            or faces.min() < 0 or faces.max() >= len(parent)
                            or arrays['vertices'].dtype != np.float64
                            or not np.array_equal(arrays['vertices'].view(np.uint64), source['vertices'][parent].view(np.uint64))
                            or not np.array_equal(parent[faces], source['faces'])):
                        raise ValueError('cut changed original XYZ or face ancestry')
                    store.save_arrays('data.npz', **arrays)
                else:
                    cut = unit(out, 'cut', request)
                    if phase in ('mvc', 'arap', 'qualification'):
                        math_api = importlib.import_module('winding_field.native_global')
                        if phase == 'mvc':
                            uv, report = math_api.mean_value_map(cut, check=store.check)
                        elif phase == 'arap':
                            uv, report = math_api.arap_map(cut, unit(out, 'mvc', request)['uv'], check=store.check)
                        else:
                            arrays, report = math_api.qualify_map(cut, unit(out, 'arap', request)['uv'], check=store.check)
                            valid, reasons = arrays['valid_faces'], arrays['reasons']
                            if (valid.dtype != np.bool_ or reasons.dtype != np.uint8 or valid.shape != (len(cut['faces']),)
                                    or reasons.shape != valid.shape or not np.array_equal(valid, reasons == 0)):
                                raise ValueError('qualification face/reason correspondence differs')
                            store.save_arrays('data.npz', **arrays)
                            store.save_arrays('excluded.npz', face_ids=source['face_ids'][~valid], reasons=reasons[~valid])
                        if phase != 'qualification':
                            if uv.shape != (len(cut['vertices']), 2) or uv.dtype != np.float64 or not np.isfinite(uv).all():
                                raise ValueError('invalid global UV array')
                            store.save_arrays('data.npz', uv=uv)
                    else:
                        from winding_field.native_render import render
                        valid = unit(out, 'qualification', request)['valid_faces']
                        uv = unit(out, 'arap', request)['uv']
                        register_global(store, source, cut, uv, valid, request['component'],
                                        source_record['source_faces'], source_record['source_area_um2'])
                        with closing(open_volume(out, request)) as volume:
                            if volume.identity != source_record['ct_identity']:
                                raise ValueError('original CT identity changed')
                            cache_before = volume.stats()['cache_bytes']
                            products = render(store, volume, request['pitch_um'], request['export_tifxyz']) if valid.any() else {'components': []}
                            ct_io = volume.stats()
                        area = float(source['areas'][valid].sum()); fraction = area/source_record['source_area_um2']
                        usable = any(c['counts']['observed_ct'] > 0 and c['ct_value_range'] is not None
                                     and c['ct_value_range'][1] > c['ct_value_range'][0] for c in products['components'])
                        all_tif = bool(products['components']) and all(
                            c['tifxyz']['status'] == 'WRITTEN' and not c['tifxyz'].get('partial_geometry', False)
                            and not c['tifxyz'].get('orphan_samples_omitted', 0)
                            for c in products['components']) if request['export_tifxyz'] else True
                        complete = int(valid.sum()) == source_record['source_faces'] and usable and all_tif
                        report = dict(products, status='COMPLETE' if complete else 'PARTIAL', usable_render=usable,
                            level=request['level'], source_faces=source_record['source_faces'], source_area_um2=source_record['source_area_um2'],
                            mapped_faces=int(valid.sum()), mapped_area_um2=area, coverage_fraction=fraction,
                            coverage_target=request['coverage_target'], target_met=fraction+1e-12 >= request['coverage_target'],
                            excluded_components=source_record['excluded_components'], selected_component=request['component'],
                            ct_cache_initial_bytes=cache_before, ct_io=ct_io, requested_tifxyz_complete=all_tif,
                            exclusions=dict(unselected_components='source/excluded-*.npz',
                                unselected_reason_codes=source_record['exclusion_reason_codes'],
                                selected_faces='qualification/excluded.npz',
                                qualification_report=verify_phase(out, 'qualification', request)['report']),
                            physical_acceptance=False, field_coverage_certified=False,
                            timing_scope='fresh-mesh-to-global-render', field_to_whole_W_goal='NOT_MEASURED')
                        publish_report(store, 'LEVEL.json', report)
            publish_report(store, 'REPORT.json', report)
        else:
            report = read_json(store.validate_file('REPORT.json'))
            if phase == 'source':
                for name, expected in report['source_files'].items():
                    if file_sha(Path(request['mesh'])/name) != expected:
                        raise ValueError('original source changed before phase sealing')
                with closing(open_volume(out, request)) as volume:
                    if volume.identity != report['ct_identity']:
                        raise ValueError('original CT changed before source sealing')
        elapsed = store.elapsed
    with closing(sqlite3.connect((root/'work.sqlite').resolve().as_uri()+'?mode=ro&immutable=1', uri=True)) as db:
        files = {name: dict(sha256=sha, bytes=size) for name, sha, size in db.execute('SELECT name,sha,bytes FROM files')}
    record = dict(identity=identity, files=files, database_sha256=file_sha(root/'work.sqlite'),
                  phase_elapsed_s=elapsed, report=report)
    atomic_json(root/'PHASE.json', record)
    return record


class Budget:
    """Only this command's cumulative active clock; unclean deaths charge a wall upper bound."""
    def __init__(self, out, seconds, startup_s=0.):
        self.out, self.seconds = out, seconds
        self.prior = float(read_json(out/'TIME.json')['elapsed_s']) if (out/'TIME.json').exists() else 0.
        if not math.isfinite(self.prior) or self.prior < 0:
            raise ValueError('invalid cumulative active time')
        active = out/'ACTIVE.json'
        if active.exists():
            old = read_json(active)
            if (type(old.get('pid')) is not int or old['pid'] <= 0
                    or not math.isfinite(old.get('prior_s', -1)) or old['prior_s'] < 0
                    or not math.isfinite(old.get('started_epoch', 0)) or old['started_epoch'] <= 0):
                raise ValueError('invalid interrupted active clock')
            try:
                os.kill(old['pid'], 0)
            except ProcessLookupError:
                pass
            else:
                raise ValueError('previous controller may still be alive')
            if old.get('attempt'):
                current = safe(out, old['attempt'])/'CURRENT.json'
                if current.exists():
                    pid = read_json(current).get('worker_pid')
                    if not pid:
                        raise ValueError('previous worker ownership unknown; reconcile before resume')
                    try:
                        os.kill(pid, 0)
                    except ProcessLookupError:
                        pass
                    else:
                        raise ValueError('previous worker may still be alive')
            self.prior = max(self.prior, old['prior_s'] + max(0., time.time()-old['started_epoch']))
        if not math.isfinite(startup_s) or startup_s < 0:
            raise ValueError('invalid preflight active time')
        self.history = self.prior
        self.prior += startup_s
        self.started = time.monotonic()
        self.active = dict(pid=os.getpid(), prior_s=self.prior, started_epoch=time.time())
        self.checkpoint()

    @property
    def elapsed(self):
        return self.prior + time.monotonic()-self.started

    @property
    def remaining(self):
        return max(0., self.seconds-self.elapsed)

    def checkpoint(self):
        atomic_json(self.out/'TIME.json', dict(elapsed_s=self.elapsed))
        atomic_json(self.out/'ACTIVE.json', self.active)

    def close(self):
        atomic_json(self.out/'TIME.json', dict(elapsed_s=self.elapsed))
        (self.out/'ACTIVE.json').unlink(missing_ok=True)


def data_ceiling(headroom, reserve, parent_rss):
    ceiling = min(RSS_BYTES-parent_rss-MARGIN, headroom-reserve-MARGIN)
    if ceiling <= 0:
        raise ValueError('insufficient live memory headroom for DATA ceiling')
    return ceiling


def scratch_bytes(out):
    total = 0
    for directory, dirs, files in os.walk(out, followlinks=False):
        for name in dirs + files:
            path = Path(directory)/name
            if path.is_symlink():
                raise ValueError('symlink in owned scratch')
            try:
                stat = path.stat()
            except FileNotFoundError:
                continue
            total += max(stat.st_size, stat.st_blocks*512) + 4096
    return total


def launch_phase(phase, out, request, budget, stop):
    from winding_field.native_pipeline import _supervise
    from winding_field.tifxyz_runtime import memory_report, process_rss
    if stop() or budget.remaining == 0.:
        return dict(status='PAUSED', reason='STOP' if stop() else 'TIME_LIMIT')
    memory = memory_report()
    if not memory['known']:
        return dict(status='PAUSED', reason='RAM_OBSERVATION')
    try:
        ceiling = data_ceiling(memory['effective_headroom_bytes'], request['reserve_ram_bytes'], process_rss(os.getpid()))
    except ValueError as error:
        return dict(status='PAUSED', reason='RAM_HEADROOM', detail=str(error))
    attempt_name = 'attempts/'+uuid.uuid4().hex; attempt = safe(out, attempt_name); attempt.mkdir(parents=True)
    token = uuid.uuid4().hex
    packet = dict(phase=phase, token=token, data_ceiling_bytes=ceiling, remaining_s=budget.remaining)
    atomic_json(attempt/'REQUEST.json', packet)
    budget.active['attempt'] = attempt_name; budget.checkpoint()
    monitor = dict(minimum_available_ram_bytes=memory['effective_headroom_bytes'], peak_scratch_bytes=0, reason=None)
    last = 0.
    def observed_stop():
        nonlocal last
        if stop():
            return True
        now = time.monotonic()
        if now-last > .5:
            last = now; memory = memory_report(); used = scratch_bytes(out); free = shutil.disk_usage(out).free
            monitor['minimum_available_ram_bytes'] = min(monitor['minimum_available_ram_bytes'], memory['effective_headroom_bytes'])
            monitor['peak_scratch_bytes'] = max(monitor['peak_scratch_bytes'], used)
            if not memory['known'] or memory['effective_headroom_bytes'] < request['reserve_ram_bytes']:
                monitor['reason'] = 'RAM_RESERVE'
            elif free < request['reserve_bytes']:
                monitor['reason'] = 'DISK_RESERVE'
            elif used >= request['scratch_bytes']-(64 << 20):
                monitor['reason'] = 'SCRATCH_HEADROOM'
            monitor['free_disk_bytes'] = free
            budget.checkpoint(); atomic_json(attempt/'MONITOR.json', monitor)
            atomic_json(out/'STATUS.json', dict(status='RUNNING', phase=phase, attempt=attempt_name,
                        active_elapsed_s=budget.elapsed, resources=monitor))
        return bool(stop() or monitor['reason'])
    if observed_stop():
        return dict(status='PAUSED', reason=monitor['reason'] or 'STOP')
    if budget.remaining == 0.:
        return dict(status='PAUSED', reason='TIME_LIMIT')
    command = [sys.executable, '-B', str(Path(__file__).resolve()), '--worker', phase, '--output', str(out),
               '--attempt', attempt_name, '--token', token]
    result = _supervise(command, attempt, budget.remaining, observed_stop, token)
    if monitor['reason']:
        result.update(supervision_reason=result.get('reason'), reason=monitor['reason'])
    result.update(phase=phase, resources=monitor)
    atomic_json(attempt/'RESULT.json', result)
    budget.active.pop('attempt', None); budget.checkpoint()
    return result


def attempt_summary(out):
    attempts = [read_json(path) for path in sorted((out/'attempts').glob('*/RESULT.json'))]
    attempts = [a for a in attempts if a['phase'] in PHASES]
    return dict(phase_worker_seconds={phase: sum(a.get('supervision', {}).get('worker_seconds', 0.)
                for a in attempts if a['phase'] == phase) for phase in PHASES},
                peak_aggregate_rss_bytes=max((a.get('supervision', {}).get('peak_aggregate_rss_bytes', 0) for a in attempts), default=0),
                attempts=[dict(phase=a['phase'], status=a['status'], supervision=a.get('supervision'),
                                resources=a.get('resources'), reason=a.get('reason')) for a in attempts])


PUBLICATION_FILES = ('PUBLICATION-RECORD.json', 'RESULT.json', 'PUBLICATION.json')


def execute_verification(out, request, remaining_s, stop=None):
    """Worker-only: source/phase hashes, CT IO and authoritative publication checks."""
    from winding_field.native_store import Paused
    from winding_field.native_pipeline import verify_ct_snapshot
    started = time.monotonic()
    def check():
        if stop and stop():
            raise Paused('STOP')
        if time.monotonic()-started >= remaining_s:
            raise Paused('TIME_LIMIT')
    check()
    source = check_sources(request, out)
    if source is None:
        raise ValueError('publication requires a committed source')
    records = {'source': source}
    for phase in PHASES[1:]:
        records[phase] = verify_phase(out, phase, request, check)
        if records[phase] is None:
            raise ValueError('publication requires committed phase: ' + phase)
    check()
    with closing(open_volume(out, request)) as volume:
        if volume.identity != source['report']['ct_identity']:
            raise ValueError('CT identity changed')
        if (out/'render/CT-SNAPSHOT.sqlite').exists():
            verify_ct_snapshot(out/'render', volume)
    check()
    summary = dict(records['render']['report'],
                   phase_elapsed_s={p: records[p]['phase_elapsed_s'] for p in PHASES}, execution=attempt_summary(out))
    bindings = {p: read_json_record(safe(out/p, 'PHASE.json'))[1] for p in PHASES}
    values = {}; hashes = {}
    for name in PUBLICATION_FILES:
        path = safe(out, name)
        if path.exists():
            values[name], hashes[name], _ = read_json_record(path)
        else:
            values[name] = hashes[name] = None
    record = values['PUBLICATION-RECORD.json']
    if record is None:
        if values['RESULT.json'] is not None or values['PUBLICATION.json'] is not None:
            raise ValueError('result/publication has no initial verification record')
    else:
        timing = record['timing']
        if (set(timing) != {'active_elapsed_s', 'original_deadline_met'}
                or type(timing['active_elapsed_s']) not in (int, float) or not math.isfinite(timing['active_elapsed_s'])
                or timing['active_elapsed_s'] < 0 or type(timing['original_deadline_met']) is not bool
                or timing['original_deadline_met'] != (timing['active_elapsed_s'] <= request['seconds'])):
            raise ValueError('invalid initial publication timing record')
        original, sha, _ = read_json_record(safe(out, record['verification']['path']))
        if (sha != record['verification']['sha256'] or digest(original['summary']) != digest(summary)
                or original['phase_hashes'] != bindings or original['request_sha256'] != digest(request)):
            raise ValueError('initial publication verification/timing/execution records changed')
        expected = dict(summary, **timing)
        # The seal alone is insufficient: authoritative fields come from verified phases.
        if values['RESULT.json'] is not None and digest(values['RESULT.json']) != digest(expected):
            raise ValueError('published result differs from authoritative render/timing/execution records')
        seal = values['PUBLICATION.json']
        if seal is not None and (values['RESULT.json'] is None or seal != dict(
                request_sha256=digest(request), result_sha256=hashes['RESULT.json'],
                record_sha256=hashes['PUBLICATION-RECORD.json'])):
            raise ValueError('publication seal changed')
    check()
    return dict(summary=summary, phase_hashes=bindings, request_sha256=digest(request),
                publication_record=record, publication_hashes=hashes, verification_s=time.monotonic()-started)


def verification_receipt(out, request, attempt, remaining_s, stop=None):
    value = execute_verification(out, request, remaining_s, stop)
    if len(json.dumps(value, allow_nan=False).encode()) > JSON_BYTES:
        raise ValueError('verification summary envelope exceeded')
    path = attempt/'VERIFIED.json'; atomic_json(path, value)
    return dict(status='VERIFIED', verification_path=str(path.relative_to(out)),
                verification_sha256=read_json_record(path)[1])


def publish_verified(out, request, response, budget):
    """Controller only reads bounded metadata returned by its owned verifier."""
    verified, sha, _ = read_json_record(safe(out, response['verification_path']))
    if sha != response['verification_sha256'] or verified['request_sha256'] != digest(request):
        raise ValueError('verification worker summary changed')
    for name, expected in verified['publication_hashes'].items():
        path = safe(out, name)
        actual = read_json_record(path)[1] if path.exists() else None
        if actual != expected:
            raise ValueError('publication changed after supervised verification')
    record = verified['publication_record']; existing = record is not None
    if record is None:
        record = dict(timing=dict(active_elapsed_s=budget.elapsed, original_deadline_met=budget.elapsed <= request['seconds']),
                      verification=dict(path=response['verification_path'], sha256=sha))
        atomic_json(out/'PUBLICATION-RECORD.json', record)
    result = dict(verified['summary'], **record['timing'])
    if not (out/'RESULT.json').exists():
        atomic_json(out/'RESULT.json', result)
    published, result_sha, _ = read_json_record(out/'RESULT.json')
    if digest(published) != digest(result):
        raise ValueError('result changed during publication')
    if not (out/'PUBLICATION.json').exists():
        atomic_json(out/'PUBLICATION.json', dict(request_sha256=digest(request), result_sha256=result_sha,
                    record_sha256=read_json_record(out/'PUBLICATION-RECORD.json')[1]))
    if existing:
        return dict(result, timing_scope='existing-products-verification',
                    verification_elapsed_s=budget.elapsed-budget.history, cumulative_active_elapsed_s=budget.elapsed)
    return result


def run(request, out, resume=False, preflight_s=0.):
    called = time.monotonic()
    from winding_field.native_store import Paused
    out = Path(out).absolute()
    validate_paths(Path(request['mesh']), out, request['ct'], Path(request['ct_local']) if request['ct_local'] else None,
                   Path(request['frame_path']))
    out = out.resolve()
    if str(out) != request['output']:
        raise ValueError('request output root changed')
    if resume:
        if not safe(out, 'TIME.json').is_file():
            raise ValueError('missing cumulative active clock on resume')
        if read_json(safe(out, 'REQUEST.json')) != request:
            raise ValueError('experimental request identity changed')
    else:
        out.mkdir()
        atomic_json(out/'REQUEST.json', request)
    with safe(out, '.controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        budget = Budget(out, request['seconds'], preflight_s + time.monotonic()-called); interrupted = False
        previous = {}
        def handle(*unused):
            nonlocal interrupted
            interrupted = True
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous[sig] = signal.signal(sig, handle)
        stop = lambda: interrupted or (out/'STOP').exists()
        result = None
        try:
            actions = ('verify',) if any(safe(out, name).exists() for name in PUBLICATION_FILES) else ACTIONS
            for phase in actions:
                if stop():
                    raise Paused('STOP')
                if budget.remaining == 0.:
                    raise Paused('TIME_LIMIT')
                result = launch_phase(phase, out, request, budget, stop)
                if result['status'] not in ('COMMITTED', 'VERIFIED'):
                    return dict(result, active_elapsed_s=budget.elapsed)
            if stop() or budget.remaining == 0.:
                raise Paused('STOP' if stop() else 'TIME_LIMIT')
            result = publish_verified(out, request, result, budget)
            return result
        except Paused as error:
            result = dict(status='PAUSED', reason=str(error), active_elapsed_s=budget.elapsed)
            return result
        except Exception as error:
            result = dict(status='FAILED', reason=f'{type(error).__name__}: {error}', active_elapsed_s=budget.elapsed)
            raise
        finally:
            budget.close()
            for sig, callback in previous.items():
                signal.signal(sig, callback)
            atomic_json(out/'STATUS.json', dict(result or {}, active_elapsed_s=budget.elapsed))


def main():
    entered = time.monotonic()
    p = parser(worker='--worker' in sys.argv); args = p.parse_args()
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'PASTIX_NUM_THREADS',
                'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[key] = '1'
    os.environ.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', OMP_DYNAMIC='FALSE', MKL_DYNAMIC='FALSE')
    if args.worker:
        out = args.output.absolute(); attempt = safe(out, args.attempt)
        request = read_json(out/'REQUEST.json'); packet = read_json(attempt/'REQUEST.json')
        if packet['token'] != args.token or packet['phase'] != args.worker:
            raise ValueError('worker ownership changed')
        current = read_json(attempt/'CURRENT.json'); current.update(worker_pid=os.getpid(), phase=args.worker)
        atomic_json(attempt/'CURRENT.json', current)
        if sys.platform != 'linux':
            raise ValueError('live DATA/headroom supervision requires Linux')
        os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[:3])); os.setpriority(os.PRIO_PROCESS, 0, 19)
        from winding_field.tifxyz_runtime import memory_report
        memory = memory_report()
        if not memory['known']:
            raise ValueError('unknown worker RAM headroom')
        ceiling = min(packet['data_ceiling_bytes'], data_ceiling(memory['effective_headroom_bytes'], request['reserve_ram_bytes'], 0))
        hard = resource.getrlimit(resource.RLIMIT_DATA)[1]
        if hard != resource.RLIM_INFINITY:
            ceiling = min(ceiling, hard)
        resource.setrlimit(resource.RLIMIT_DATA, (ceiling, ceiling))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        atomic_json(attempt/'LIMITS.json', dict(resource='RLIMIT_DATA', data_ceiling_bytes=ceiling,
            effective_headroom_bytes=memory['effective_headroom_bytes'], reserve_ram_bytes=request['reserve_ram_bytes'],
            cpu_affinity=sorted(os.sched_getaffinity(0)), nice=os.getpriority(os.PRIO_PROCESS, 0), threads=1))
        interrupted = False
        def handle(*unused):
            nonlocal interrupted
            interrupted = True
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, handle)
        from winding_field.native_store import Paused
        try:
            stop = lambda: interrupted or (out/'STOP').exists()
            if args.worker == 'verify':
                result = verification_receipt(out, request, attempt, packet['remaining_s'], stop)
            else:
                if args.worker == 'source' and not (out/'source/PHASE.json').exists() and find_spec('igl') is None:
                    raise ValueError('the experimental numerical recipe requires existing optional libigl')
                execute_phase(out, args.worker, request, packet['remaining_s'], stop)
                result = dict(status='COMMITTED')
        except Paused as error:
            result = dict(status='PAUSED', reason=str(error))
        except Exception as error:
            import traceback
            traceback.print_exc()
            result = dict(status='FAILED', reason=f'{type(error).__name__}: {error}')
        atomic_json(attempt/'WORKER-RESULT.json', dict(token=args.token, worker_pid=os.getpid(), result=result))
        return
    if args.mesh is None or args.component is None or args.ct is None or args.frame is None or args.pitch_um is None:
        p.error('--mesh, --component, --ct, --frame and --pitch-um are required')
    if not args.resume and find_spec('igl') is None:
        p.error('the experimental numerical recipe requires existing optional libigl')
    if find_spec('winding_field.native_global') is None:
        raise ValueError('native_global numerical API is not integrated')
    request = make_request(args)
    result = run(request, args.output, args.resume, preflight_s=time.monotonic()-entered)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result['status'] == 'COMPLETE' else 2


if __name__ == '__main__':
    raise SystemExit(main())
