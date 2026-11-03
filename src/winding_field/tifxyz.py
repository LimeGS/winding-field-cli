"""Direct-field orchestration. Public entry points always use a Linux worker.

The private engines also serve controlled mathematical tests and the explicitly
bound historical benchmark. They do not establish a runtime enforcement claim.
Source identity/metadata stay globally bound; payload hashes certify ROI reads,
with the frozen reader's cache/signature checks, not all checkpoint payload bytes.
"""
from contextlib import contextmanager
from dataclasses import asdict, replace
import fcntl
import hashlib
import importlib.metadata
from inspect import signature
import io
import itertools
import json
import math
import os
from pathlib import Path
import platform
import sqlite3
import stat
import tempfile
from types import SimpleNamespace
import zlib

from .tifxyz_contract import ALGORITHM, Blocked, Paused, Settings, parse_json, positive, validate_frame, validate_request
from .util import atomic_json, digest, file_sha

READER = dict(cell_rule='four-corners', diagonals=['nw-se', 'ne-sw'], inpainting=False)
FLAGS = dict(physical_acceptance=False, native_face_identity=False, native_topology_equivalence='UNMEASURED')
GUIDE_ARRAYS = ('world', 'faces', 'uv', 'normals', 'source_vertices', 'boundary')
STRIDE_RETRY = frozenset(('GUIDE_MULTIPLE_COMPONENTS', 'GUIDE_NONMANIFOLD', 'GUIDE_NOT_DISK',
    'GUIDE_COLLAPSED_UV', 'GUIDE_NO_SURFACE', 'GUIDE_NO_SUPPORT', 'PROJECTION_AMBIGUOUS',
    'PROJECTION_ABSENT', 'SOURCE_UNCOVERED', 'COVERAGE_DEPTH_LIMIT', 'EMPTY_DELIVERED_SURFACE'))


def _plain(value):
    return json.loads(json.dumps(value, allow_nan=False))


def _path(path):
    path = Path(os.path.abspath(path))
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('symlink in export path')
    return path


def _record(path):
    path = _path(path)
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError('private regular export artifact required')
    result = dict(bytes=info.st_size, sha256=file_sha(path))
    after = path.stat()
    if (info.st_size, info.st_mtime_ns, info.st_ctime_ns) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise ValueError('export artifact changed during read')
    return result


def _read(path):
    from .tifxyz_field import read_metadata
    _path(path)
    value, _ = read_metadata(path)
    return value


def _sync(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try: os.fsync(fd)
    finally: os.close(fd)


def _withdraw_ready(root):
    for name in ('READY','PARTIAL'):
        if not (root/(name+'.json')).exists():continue
        index=0
        while (root/f'{name}.failed-{index:04d}.json').exists():index+=1
        os.rename(root/(name+'.json'),root/f'{name}.failed-{index:04d}.json');_sync(root)


def _write(path, value, *, immutable=False):
    if len(json.dumps(value,allow_nan=False).encode())>16<<20: raise Blocked('METADATA_LIMIT')
    if path.exists():
        _record(path)
        if immutable:
            if _read(path) != _plain(value): raise ValueError('immutable export receipt differs: '+str(path))
            return
    atomic_json(path, value)
    _sync(path.parent)


def _code_identity():
    root = Path(__file__).parent
    names = ['tifxyz.py', 'tifxyz_contract.py', 'tifxyz_runtime.py', 'tifxyz_field.py',
              'tifxyz_guide.py', 'tifxyz_project.py', 'tifxyz_store.py', 'tifxyz_audit.py',
              'tifxyz_predicates.py','tifxyz_build.py','native/predicates.h','native/predicates.cpp','native/predicates.cu',
             'util.py', 'surface_sources.py']
    versions = {}
    for name in ('numpy', 'scipy', 'scikit-image', 'libigl', 'tifffile', 'winding-field-experiments'):
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name] = 'unavailable'
    import field_store
    files={name: file_sha(root/name) for name in names}
    files['field_store.py']=file_sha(Path(field_store.__file__))
    return dict(files=files, numeric=versions,
                python=platform.python_version(), machine=platform.machine(), system=platform.system(),
                sqlite=sqlite3.sqlite_version, zlib=zlib.ZLIB_RUNTIME_VERSION)


def _decode_settings(value):
    if isinstance(value, Settings): value = asdict(value)
    result = Settings(**value)
    positive(result.pitch_um,'effective pitch');positive(result.tolerance_um,'effective tolerance')
    validate_frame(result.output_frame)
    validate_request(dict(levels=list(result.levels), roi=dict(lo_zyx=list(result.roi_lo_zyx), hi_zyx=list(result.roi_hi_zyx)),
        pitch_um=result.pitch_um, tolerance_um=result.tolerance_um, output_frame=result.output_frame, max_pixels=result.max_pixels,
        memory_gib=result.memory_bytes/(1<<30), scratch_gib=result.scratch_bytes/(1<<30), reserve_disk_gib=result.reserve_disk_bytes/(1<<30),
        audit_backend=result.audit_backend,coverage_policy=result.coverage_policy))
    for name in ('memory_bytes', 'scratch_bytes', 'reserve_disk_bytes', 'reserve_ram_bytes'):
        if type(getattr(result, name)) is not int or getattr(result, name)<0: raise ValueError('invalid effective budget')
    if result.algorithm != ALGORITHM or result.reserve_ram_bytes != 2<<30:
        raise ValueError('invalid algorithm/reserve identity')
    if not isinstance(result.request_sha256, str) or len(result.request_sha256)!=64:
        raise ValueError('invalid request identity')
    if result.memory_bytes>3<<30 or result.scratch_bytes>16<<30:
        raise Blocked('BUDGET_ENVELOPE', 'Task 6 admits at most 3GiB memory and 16GiB scratch')
    return result


def _budget(settings, root):
    return SimpleNamespace(scratch_bytes=settings.scratch_bytes, reserve_disk_bytes=settings.reserve_disk_bytes,
                           database_bytes=settings.scratch_bytes//8, scope_root=root)


def _space(root, settings, pending=0):
    fs = os.statvfs(root); quantum = max(fs.f_frsize, fs.f_bsize, 4096); device = root.stat().st_dev
    def charge(path):
        info = path.lstat()
        if info.st_dev!=device or not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)):
            raise ValueError('nonregular/symlink export artifact')
        total = max(info.st_size, info.st_blocks*512)+quantum
        if stat.S_ISDIR(info.st_mode):
            with os.scandir(path) as entries:
                for entry in entries: total += charge(Path(entry.path))
        elif info.st_nlink!=1: raise ValueError('shared export artifact')
        return total
    used = charge(root)
    if used+pending+2*quantum>settings.scratch_bytes: raise Blocked('SCRATCH_BUDGET')
    if fs.f_bavail*fs.f_frsize<settings.reserve_disk_bytes+pending+2*quantum: raise Blocked('DISK_HEADROOM')
    return used


def _admit(settings, root, field=None, pixels=0, *, owned=True):
    from . import tifxyz_runtime as runtime
    # Includes bounded guide/MC/UV work, source cache, SQLite and tile buffers.
    work = (768<<20)+getattr(field, 'metadata_bytes', 0)+(128<<20 if settings.audit_backend=='cuda' else 0)
    disk = (256<<20)+256*pixels
    parent = int(os.environ.get('WINDING_TIFXYZ_PARENT_RSS', '0'))
    receipt = runtime.admit(settings, dict(working_bytes=work, disk_bytes=disk, parent_rss=parent), root)
    if owned and root.exists() and field is not None: _space(root, settings, disk)
    return receipt


def _check(stop):
    if stop and stop(): raise Paused()


def _validate_roi_payloads(field, stop):
    # Reused published work skips coarse/audit reads; reestablish payload hashes in this reader lifetime.
    field.validate()
    for lo in itertools.product(*(range(0,n,64) for n in field.shape)):
        _check(stop)
        field.read(list(lo),[min(a+64,n) for a,n in zip(lo,field.shape)])
    field.validate()


def _source_descriptor(field):
    if hasattr(field, 'binding'):
        root = _path(Path(field.source.identity['field']['path']).parent.parent)
        return dict(kind='field-run', path=str(root))
    return dict(kind='explicit-physical-adapter', identity_sha256=digest(field.identity))


def _identity(field, settings):
    from .tifxyz_predicates import identity as accelerator_identity
    if (tuple(settings.roi_lo_zyx)!=tuple(field.lo) or
        tuple(b-a for a,b in zip(settings.roi_lo_zyx, settings.roi_hi_zyx))!=tuple(field.shape)):
        raise ValueError('source ROI differs from settings')
    scope=dict(metadata_scope='global-bound-source',payload_hash_scope='consumed-roi-payloads',
               coverage_roi=dict(lo_zyx=list(settings.roi_lo_zyx),hi_zyx=list(settings.roi_hi_zyx)),
               all_checkpoint_payloads_verified=False)
    return _plain(dict(schema='winding-field-tifxyz-export-v1', source=field.identity,source_verification=scope,
        source_opener=_source_descriptor(field), effective_settings=asdict(settings), implementation=_code_identity(),
        audit_accelerator=accelerator_identity(settings.audit_backend),reader=READER, **FLAGS))


def _check_accelerator(settings,identity):
    from .tifxyz_predicates import identity as accelerator_identity
    if accelerator_identity(settings.audit_backend)!=identity['audit_accelerator']:
        raise ValueError('accelerator identity changed')


def _snapshot_run(run):
    stage=run['stages']['field']
    return dict(producer_schema=run['schema'], producer_implementation=run['implementation_sha256'], source=run['source'],
                field_receipt={k:stage[k] for k in ('receipt','sha256')})


def _snapshot(field, root, *, resume):
    if not hasattr(field,'binding'): return None
    path=root/'producer-run.snapshot.json'
    if not resume:
        from .util import atomic_bytes
        source=Path(_source_descriptor(field)['path'])/'RUN.json'
        fd=os.open(source,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
        with os.fdopen(fd,'rb') as stream:
            info=os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size>16<<20: raise ValueError('producer RUN envelope exceeded')
            raw=stream.read((16<<20)+1)
        if len(raw)>16<<20: raise ValueError('producer RUN envelope exceeded')
        saved=_snapshot_run(parse_json(raw))
        if saved!={key:field.binding[key] for key in saved}: raise ValueError('producer snapshot source differs')
        atomic_bytes(path,raw);_sync(root)
    saved=_snapshot_run(_read(path))
    if saved!={key:field.binding[key] for key in saved}: raise ValueError('producer snapshot stable source differs')
    return _record(path)


@contextmanager
def _owned(root, identity, resume):
    if root.exists():
        if not resume: raise ValueError('output already exists')
        if _read(root/'EXPORT.json')!=identity: raise ValueError('export identity differs')
    elif resume: raise ValueError('missing export to resume')
    else:
        root.mkdir(); _sync(root.parent)
        _write(root/'EXPORT.json',identity,immutable=True)
    fd=os.open(root/'.lock',os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
    try:
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield
    finally: os.close(fd)


def _tiles(grid):
    for y in range((grid['shape_yx'][0]+255)//256):
        for x in range((grid['shape_yx'][1]+255)//256): yield y,x


def _inventory(root, recursive=False):
    """Constant-memory digest; work files are flat, never an output-sized list."""
    count=0; total=0; checksum=0
    def visit(folder):
        nonlocal count,total,checksum
        with os.scandir(folder) as entries:
            for entry in entries:
                path=_path(entry.path)
                if recursive and path.is_dir():visit(path);continue
                if path.name=='.lock':
                    _record(path);continue
                record=_record(path)
                checksum=(checksum+int(digest(dict(name=str(path.relative_to(root)),**record)),16))%(1<<256)
                count+=1;total+=record['bytes']
    visit(root)
    return dict(files=count,bytes=total,sha256=digest([count,total,f'{checksum:064x}']))


def _guide_signature(guide):
    return dict(stride=guide['stride'],level=guide['level'],arrays={name:dict(shape=list(guide[name].shape),
        dtype=guide[name].dtype.str,sha256=hashlib.sha256(guide[name].tobytes(order='C')).hexdigest()) for name in GUIDE_ARRAYS})


def _save_guide(guide, root):
    import numpy as np
    from .tifxyz_store import _write_bytes
    root=_path(root)
    root.mkdir(exist_ok=True)
    expected={*(name+'.npy' for name in GUIDE_ARRAYS),'GUIDE.json'}
    interrupted=[path for path in root.iterdir() if path.name not in expected]
    prefixes=tuple('.'+name+'-' for name in expected)
    if any(not path.name.startswith(prefixes) for path in interrupted):raise ValueError('foreign guide artifact')
    if interrupted:
        for path in interrupted:_record(path)
        index=0
        while (root.parent/f'guide-interrupted-{index:04d}').exists():index+=1
        retained=root.parent/f'guide-interrupted-{index:04d}';retained.mkdir()
        for path in interrupted:os.rename(path,retained/path.name)
        _sync(retained);_sync(root)
    _sync(root.parent)
    for name in GUIDE_ARRAYS:
        data=io.BytesIO(); np.save(data,guide[name],allow_pickle=False); raw=data.getvalue();path=root/(name+'.npy')
        if path.exists():
            if _record(path)!=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()): raise ValueError('guide array differs')
        else: _write_bytes(path,raw)
    record=_guide_signature(guide)
    _write(root/'GUIDE.json',record,immutable=True)
    return record


def _load_guide(root):
    import numpy as np
    record=_read(root/'GUIDE.json');guide=dict(stride=record['stride'],level=record['level'])
    if set(record['arrays'])!=set(GUIDE_ARRAYS): raise ValueError('guide artifact set differs')
    if set(p.name for p in root.iterdir())!={*(name+'.npy' for name in GUIDE_ARRAYS),'GUIDE.json'}:
        raise ValueError('guide artifact set differs')
    for name in GUIDE_ARRAYS:
        path=root/(name+'.npy');size=_record(path)['bytes']
        if size>32<<20: raise ValueError('guide array envelope exceeded')
        with path.open('rb') as stream:
            if np.lib.format.read_magic(stream)!=(1,0): raise ValueError('invalid guide header')
            shape,fortran,dtype=np.lib.format.read_array_header_1_0(stream)
            if dtype.hasobject or fortran or math.prod(shape)*dtype.itemsize+stream.tell()!=size:
                raise ValueError('invalid guide array size')
        guide[name]=np.load(path,allow_pickle=False)
    if _guide_signature(guide)!=record: raise ValueError('guide array identity differs')
    if len(guide['faces'])>250000 or len(guide['world'])>750000: raise ValueError('guide geometry envelope exceeded')
    return guide


def _build_coarse(field, stride, out, stop, settings, scope):
    from .tifxyz_field import build_coarse
    remaining=SimpleNamespace(scratch_bytes=max(0,settings.scratch_bytes-_space(scope,settings)),
                              reserve_disk_bytes=settings.reserve_disk_bytes)
    # shortcut: legacy positional producer, remove compatibility when I4 budget= is universal.
    options={'budget':remaining} if 'budget' in signature(build_coarse).parameters else {}
    return build_coarse(field,stride,out,stop,**options)


def _coarse(field, stride, root, stop, settings, scope):
    from .tifxyz_field import load_coarse
    root.mkdir(exist_ok=True)
    index=0
    while (root/f'coarse-{index:04d}').exists():
        path=root/f'coarse-{index:04d}'
        if (path/'COARSE.json').exists():
            record=_read(path/'COARSE.json')
            if record['root']!=str(path) or record['source']!=field.identity or record['stride']!=stride: raise ValueError('coarse identity differs')
            load_coarse(record)
            return record
        index+=1
    return _build_coarse(field,stride,root/f'coarse-{index:04d}',stop,settings,scope)


def _audit_status(audit,policy):
    if policy not in ('strict','allow_partial'):raise ValueError('invalid coverage policy')
    for key,value in FLAGS.items():
        if type(audit.get(key)) is not type(value) or audit.get(key)!=value:raise ValueError('invalid audit acceptance flags')
    if audit.get('status')=='PASS':
        if (audit.get('source_scan_complete') is not True or audit.get('geometry_complete') is not True
            or audit.get('blocked_phase') is not None or audit.get('reasons') or audit.get('remaining_source_coverage')!='NONE'):
            raise ValueError('invalid complete audit')
        return 'READY'
    if audit.get('status')!='BLOCKED' or not audit.get('reasons'): raise ValueError('invalid audit result')
    if (policy=='allow_partial' and audit.get('geometry_complete') is True and audit.get('source_scan_complete') is False
        and audit.get('blocked_phase')=='coverage' and audit.get('remaining_source_coverage')=='UNMEASURED'
        and all(r.get('code') in ('SOURCE_UNCOVERED','COVERAGE_DEPTH_LIMIT') for r in audit['reasons'])):
        roi=audit['source_identity']['roi'];lo=roi['lo_zyx'];shape=[b-a for a,b in zip(lo,roi['hi_zyx'])]
        for reason in audit['reasons']:
            region=reason.get('region')
            if not isinstance(region,dict) or set(region)!={'cell_zyx','global_cell_zyx','subcell_index_zyx','depth'}:
                raise ValueError('partial audit requires an explicit uncertified source region')
            depth=region['depth'];cell=region['cell_zyx'];index=region['subcell_index_zyx'];global_cell=region['global_cell_zyx']
            if (type(depth) is not int or not 0<=depth<=8 or any(not isinstance(a,list) or len(a)!=3 for a in (cell,index,global_cell))
                or any(type(v) is not int for a in (cell,index,global_cell) for v in a)
                or any(not 0<=v<n-1 for v,n in zip(cell,shape)) or any(not 0<=v<2**depth for v in index)
                or global_cell!=[a+b for a,b in zip(lo,cell)]):
                raise ValueError('invalid partial coverage region')
        return 'PARTIAL'
    first=audit['reasons'][0]
    raise Blocked(first['code'],first.get('detail',''))


def _publication_status(levels):
    if not levels:raise ValueError('empty publication')
    for item in levels:
        if item.get('status') not in ('READY','PARTIAL') or item.get('coverage_complete') is not (item['status']=='READY'):
            raise ValueError('invalid level publication quality')
    return 'READY' if all(item['coverage_complete'] for item in levels) else 'PARTIAL'


def _publication_marker(root):
    paths=[root/(name+'.json') for name in ('READY','PARTIAL') if (root/(name+'.json')).exists() or (root/(name+'.json')).is_symlink()]
    if len(paths)!=1:raise ValueError('exactly one READY/PARTIAL publication marker required')
    return paths[0],_read(paths[0])


def _segment_files(root, manifest, stop):
    expected={'x.tif','y.tif','z.tif','meta.json','.segment.json','VERIFIED.json'}
    if set(manifest['files'])!=expected or set(p.name for p in root.iterdir())!=expected:
        raise ValueError('segment artifact set differs')
    for name,record in manifest['files'].items():
        _check(stop)
        if _record(root/name)!=record: raise ValueError('segment artifact hash differs: '+name)


def _check_published_segments(root, levels, stop):
    if {p.name for p in root.glob('level-*')}!={item['segment'] for item in levels}:
        raise ValueError('published segment set differs')
    for item in levels:
        _segment_files(_relative(root,item['segment']),item['manifest'],stop)


def _segment(root, manifest, grid, settings, store, stop, *, expected_status=None):
    from .tifxyz_store import iter_tiff_tiles
    _segment_files(root,manifest,stop)
    meta=_read(root/'meta.json')
    required=dict(shape_yx=grid['shape_yx'],dtype='float32',scale=[settings.output_frame['voxel_um']/grid['pitch_um']]*2,
                  level=grid['level'],pitch_um=grid['pitch_um'],uv_min_um=grid['uv_min_um'],output_frame=settings.output_frame)
    if any(meta.get(k)!=v for k,v in required.items()): raise ValueError('segment metadata differs')
    if set(meta)!=set(required)|{'format','type','uuid'}: raise ValueError('segment metadata field set differs')
    # T4 owns the standard format/type/uuid fields as well as their deterministic UUID.
    if meta.get('format')!='tifxyz' or meta.get('type')!='seg' or not isinstance(meta.get('uuid'),str) or not meta['uuid']:
        raise ValueError('standard tifxyz/seg metadata required')
    if any(manifest.get(k)!=required[k] for k in ('shape_yx','dtype','scale')):
        raise ValueError('artifact manifest geometry differs')
    verification=_read(root/'VERIFIED.json')
    if verification['grid']!=grid or verification['reader']!=READER: raise ValueError('segment verification mapping differs')
    if any(type(verification.get(k)) is not type(v) or verification.get(k)!=v for k,v in FLAGS.items()): raise ValueError('segment acceptance flags differ')
    status=_audit_status(verification['audit'],settings.coverage_policy)
    if (verification.get('status')!=status or verification.get('coverage_complete') is not (status=='READY')
        or (expected_status is not None and status!=expected_status)):
        raise ValueError('segment publication quality differs')
    marker=_read(root/'.segment.json')
    if marker.get('grid')!=grid or marker.get('output_frame')!=settings.output_frame: raise ValueError('segment binding differs')
    if store is not None and marker.get('store_sha256')!=digest(dict(format='winding-field-tiles-v1',identity=store.identity,phase='delivered')):
        raise ValueError('segment store identity differs')
    for axis,name in enumerate(('x.tif','y.tif','z.tif')):
        for (y,x),array in iter_tiff_tiles(root/name,grid['shape_yx'],settings.max_pixels):
            _check(stop)
            if store is not None:
                expected=store.read_region((y,x),(y+array.shape[0],x+array.shape[1]))['xyz'][...,axis]
                if array.tobytes()!=expected.tobytes(): raise ValueError('delivered TIFF projection bytes differ')


def _pending(work, binding, grid, frame):
    pending=work/'segment.pending'
    if not pending.exists():return pending
    marker=dict(store_sha256=digest(dict(format='winding-field-tiles-v1',identity=binding,phase='delivered')),grid=grid,output_frame=frame)
    if (pending/'.segment.json').exists() and _read(pending/'.segment.json')!=marker:
        raise ValueError('pending segment identity differs')
    names={p.name for p in pending.iterdir()}
    extras=names-{'x.tif','y.tif','z.tif','meta.json','.segment.json','VERIFIED.json'}
    if extras:
        prefixes=('.x.tif-','.y.tif-','.z.tif-','.meta.json-','.segment.json-','.VERIFIED.json-')
        if any(not name.startswith(prefixes) for name in extras):raise ValueError('foreign pending artifact')
        index=0
        while (work/f'segment-interrupted-{index:04d}').exists():index+=1
        os.rename(pending,work/f'segment-interrupted-{index:04d}');_sync(work)
    return pending


def _store_identity(identity, guide, grid):
    return dict(export_sha256=digest(identity),guide_sha256=digest(_guide_signature(guide)),grid=grid,output_frame=identity['effective_settings']['output_frame'])


def _attempt(field, root, work, guide, grid, settings, identity, stop):
    from .tifxyz_store import TileStore, write_segment
    from .tifxyz_project import precheck_projection, project_tile
    from .tifxyz_audit import normalize_delivered, audit_level
    binding=_store_identity(identity,guide,grid);budget=_budget(settings,root)
    work.mkdir(exist_ok=True)
    _write(root/'STATUS.json',dict(status='PRECHECKING_ROOTS',attempt=str(work.relative_to(root))))
    try:precheck=precheck_projection(field,guide,grid,stop)
    except Blocked as error:
        _write(work/'PRECHECK.json',dict(binding=binding,result=dict(status='BLOCKED',code=error.code,detail=error.detail)),immutable=True)
        raise
    _write(work/'PRECHECK.json',dict(binding=binding,result=precheck),immutable=True)
    _write(root/'STATUS.json',dict(status='PROJECTING',attempt=str(work.relative_to(root))))
    with TileStore(work/'projected',binding,budget,'projected',resume=(work/'projected').exists()) as projected:
        done=iter(projected.iter_ids()); current=next(done,None)
        for tid in _tiles(grid):
            _check(stop)
            if tid==current: current=next(done,None); continue
            projected.commit(project_tile(field,guide,grid,tid,settings,stop))
            _write(root/'STATUS.json',dict(status='PROJECTING',attempt=str(work.relative_to(root)),last_committed_tile=list(tid)))
        if current is not None: raise ValueError('unexpected committed tile')
        _check(stop)
        with TileStore(work/'delivered',binding,budget,'delivered',resume=(work/'delivered').exists()) as delivered:
            counts=normalize_delivered(projected,delivered,grid,READER)
            _write(root/'STATUS.json',dict(status='AUDITING',attempt=str(work.relative_to(root))))
            def progress(event):
                _write(root/'STATUS.json',dict(status='AUDITING',attempt=str(work.relative_to(root)),audit=event))
            _check(stop);audit=audit_level(field,delivered,guide,settings,stop,progress=progress,
                expected_accelerator=identity['audit_accelerator'])
            status=_audit_status(audit,settings.coverage_policy)
            _write(root/'STATUS.json',dict(status='PUBLISHING',attempt=str(work.relative_to(root))))
            _check(stop)
            pending=_pending(work,binding,grid,settings.output_frame)
            manifest=write_segment(delivered,grid,settings.output_frame,pending)
            verified=dict(status=status,coverage_complete=status=='READY',grid=grid,reader=READER,
                guide_sha256=binding['guide_sha256'],audit=audit,counts=counts,**FLAGS)
            _write(pending/'VERIFIED.json',verified,immutable=True)
            manifest['files'].update({name:_record(pending/name) for name in ('.segment.json','VERIFIED.json')})
            _segment(pending,manifest,grid,settings,delivered,stop,expected_status=status)
            field.validate();_check(stop)
    return dict(status=status,coverage_complete=status=='READY',grid=grid,manifest=manifest,attempt=str(work.relative_to(root)),guide=str((work.parent/'guide').relative_to(root)),
                guide_sha256=binding['guide_sha256'],projected=_inventory(work/'projected'),delivered=_inventory(work/'delivered'))


def _export_source(field, out, settings, resume=False, stop=None):
    """Engine for an already opened, explicitly admitted PhysicalField."""
    from .tifxyz_guide import stride_schedule, build_guide
    from .tifxyz_project import make_grid
    root=_path(out); settings=_decode_settings(settings); identity=_identity(field,settings)
    if root.exists() and not resume: raise ValueError('output already exists')
    try: _admit(settings,root if root.exists() else root.parent,field,owned=root.exists())
    except Blocked as error: return dict(status='BLOCKED',reasons=[dict(code=error.code,detail=error.detail)],**FLAGS)
    field.validate()
    if resume and any((root/name).exists() or (root/name).is_symlink() for name in ('READY.json','PARTIAL.json')):
        result=_verify_source(field,root,stop)
        published={'TIFXYZ_VERIFIED':'READY','TIFXYZ_PARTIAL_VERIFIED':'PARTIAL'}
        return dict(result,status=published[result['status']]) if result['status'] in published else result
    with _owned(root,identity,resume):
        token=os.environ.get('WINDING_TIFXYZ_SUPERVISION_ID')
        if token:_write(root/'SUPERVISOR.json',dict(token=token))
        snapshot=_snapshot(field,root,resume=resume)
        try:
            levels=[];roi_validated=False
            for index,level in enumerate(settings.levels):
                _check(stop)
                leveldir=root/'work'/f'level-{index:04d}';leveldir.mkdir(parents=True,exist_ok=True)
                accepted=leveldir/'ACCEPTED.json'
                if accepted.exists():
                    item=_read(accepted)
                    _validate_level_record(root,item,index,level,settings)
                    if not roi_validated:
                        _validate_roi_payloads(field,stop);roi_validated=True
                    location=root/item['segment'] if (root/item['segment']).exists() else root/item['attempt']/'segment.pending'
                    _segment(location,item['manifest'],item['grid'],settings,None,stop,expected_status=item['status'])
                    if not (root/item['segment']).exists():
                        os.rename(root/item['attempt']/'segment.pending',root/item['segment']);_sync(root)
                    levels.append(item);continue
                last=None
                for stride in stride_schedule(field.shape):
                    guide_root=leveldir/f'stride-{stride:04d}';guide_root.mkdir(exist_ok=True)
                    if (guide_root/'REJECTED.json').exists():
                        last=_read(guide_root/'REJECTED.json');continue
                    try:
                        _admit(settings,root,field)
                        _write(root/'STATUS.json',dict(status='PREPARING_GUIDE',level=level,stride=stride))
                        coarse=_coarse(field,stride,guide_root,stop,settings,root)
                        if not (guide_root/'guide/GUIDE.json').exists():
                            guide=build_guide(coarse,level,settings,stop);_save_guide(guide,guide_root/'guide')
                        guide=_load_guide(guide_root/'guide')
                        if guide['stride']!=stride or guide['level']!=level: raise ValueError('guide level/stride differs')
                        pitch=settings.pitch_um;refinement=0
                        while True:
                            _check(stop);grid=make_grid(guide,pitch,settings.max_pixels,level=level)
                            work=guide_root/f'pitch-{refinement:04d}'
                            if (work/'REJECTED.json').exists():
                                rejection=_read(work/'REJECTED.json')
                                if rejection['code']!='OUTPUT_DISTANCE_BOUND': raise Blocked(rejection['code'],rejection.get('detail',''))
                                pitch/=2;refinement+=1;continue
                            _admit(settings,root,field,math.prod(grid['shape_yx']))
                            _write(root/'STATUS.json',dict(status='PROJECTING',level=level,stride=stride,pitch_um=pitch))
                            try: item=_attempt(field,root,work,guide,grid,settings,identity,stop)
                            except Blocked as error:
                                if error.code=='OUTPUT_DISTANCE_BOUND':
                                    _write(work/'REJECTED.json',dict(code=error.code,detail=error.detail),immutable=True)
                                    pitch/=2;refinement+=1;continue
                                raise
                            item.update(segment=f'level-{index:04d}',level=level,stride=stride,refinement=refinement,
                                        coarse=str(Path(coarse['root']).relative_to(root)),coarse_inventory=_inventory(Path(coarse['root'])))
                            roi_validated=True
                            _write(accepted,item,immutable=True)
                            _check(stop)
                            os.rename(work/'segment.pending',root/item['segment']);_sync(root)
                            levels.append(item);break
                        break
                    except Blocked as error:
                        if error.code not in STRIDE_RETRY: raise
                        last=dict(code=error.code,detail=error.detail)
                        _write(guide_root/'REJECTED.json',last,immutable=True)
                else: raise Blocked(last['code'] if last else 'NO_ADMISSIBLE_GUIDE',last.get('detail','') if last else '')
            if resume:_verify_levels(field,root,settings,identity,levels,stop)
            field.validate();_check(stop);_space(root,settings)
            if _code_identity()!=identity['implementation']:raise ValueError('export implementation changed')
            _check_accelerator(settings,identity)
            status=_publication_status(levels);complete=status=='READY'
            receipt=dict(status=status,coverage_complete=complete,identity_sha256=digest(identity),producer_snapshot=snapshot,levels=levels,reader=READER,
                         work_inventory=_inventory(root/'work',recursive=True),source_verification=identity['source_verification'],**FLAGS)
            _write(root/'TIFXYZ.json',receipt,immutable=True)
            _write(root/'STATUS.json',dict(status=status,coverage_complete=complete,**FLAGS))
            _check(stop)
            result=dict(status=status,coverage_complete=complete,independent_reprojection=bool(resume),levels=len(levels),source_io=dict(field.stats),
                        metadata_charge_bytes=getattr(field,'metadata_bytes',0),source_verification=identity['source_verification'],**FLAGS)
            _check_published_segments(root,levels,stop)
            _write(root/(status+'.json'),dict(status=status,coverage_complete=complete,tifxyz_sha256=file_sha(root/'TIFXYZ.json'),identity_sha256=digest(identity),**FLAGS),immutable=True)
            return result
        except (Paused,Blocked) as error:
            _withdraw_ready(root)
            result=dict(status='PAUSED' if isinstance(error,Paused) else 'BLOCKED',**FLAGS)
            if isinstance(error,Blocked): result['reasons']=[dict(code=error.code,detail=error.detail)]
            _write(root/'STATUS.json',result);return result
        except BaseException:
            _withdraw_ready(root)
            _write(root/'STATUS.json',dict(status='FAILED',**FLAGS));raise


def _relative(root, value):
    if not isinstance(value,str) or Path(value).is_absolute() or '..' in Path(value).parts:
        raise ValueError('invalid relative export path')
    return _path(root/value)


def _validate_level_record(root,item,index,level,settings):
    _publication_status([item])
    if item['segment']!=f'level-{index:04d}' or item['level']!=level or item['grid']['level']!=level:
        raise ValueError('level selection differs')
    for name in ('attempt','guide','coarse'): _relative(root,item[name])
    from .tifxyz_field import load_coarse
    coarse=_read(root/item['coarse']/'COARSE.json')
    if coarse['root']!=str(root/item['coarse']) or _inventory(root/item['coarse'])!=item['coarse_inventory']:
        raise ValueError('coarse artifacts differ')
    load_coarse(coarse)
    for phase in ('projected','delivered'):
        if _inventory(root/item['attempt']/phase)!=item[phase]: raise ValueError('retained '+phase+' artifacts differ')
    guide=_load_guide(root/item['guide'])
    if digest(_guide_signature(guide))!=item['guide_sha256']: raise ValueError('guide identity differs')
    if type(item['refinement']) is not int or not 0<=item['refinement']<=64: raise ValueError('invalid pitch refinement')
    from .tifxyz_project import make_grid
    grid=make_grid(guide,settings.pitch_um/2**item['refinement'],settings.max_pixels,level=level)
    if grid!=item['grid'] or guide['stride']!=item['stride']: raise ValueError('grid mapping differs')
    return guide


@contextmanager
def _reconstruct(field, original, fresh, guide, grid, settings, identity, budget, stop, *, complete):
    """Recertify retained payloads without opening their accepted SQLite files."""
    from .tifxyz_project import project_tile
    from .tifxyz_store import TileStore
    from .tifxyz_audit import normalize_delivered
    binding=_store_identity(identity,guide,grid)
    def store_binding(phase):
        if _read(original/phase/'STORE.json')!=dict(format='winding-field-tiles-v1',identity=binding,phase=phase):
            raise ValueError('retained store identity differs')
    store_binding('projected')
    with TileStore(fresh/'projected',binding,budget,'projected') as projected:
        for tid in _tiles(grid):
            _check(stop);name=f'tile-{tid[0]}-{tid[1]}.bin';path=original/'projected'/name
            if not path.exists():
                if complete:raise ValueError('missing retained projection')
                continue
            projected.commit(project_tile(field,guide,grid,tid,settings,stop))
            if _record(projected.root/name)!=_record(path):raise ValueError('projection/root-error certificate differs')
        if not (original/'delivered').exists():
            if complete:raise ValueError('missing delivered generation')
            yield None,None
            return
        store_binding('delivered')
        with TileStore(fresh/'delivered',binding,budget,'delivered') as delivered:
            try:counts=normalize_delivered(projected,delivered,grid,READER)
            except Blocked as error:
                if complete or error.code!='EMPTY_DELIVERED_SURFACE':raise
                counts=None
            for tid in _tiles(grid):
                _check(stop);name=f'tile-{tid[0]}-{tid[1]}.bin';path=original/'delivered'/name
                if not path.exists():
                    if complete:raise ValueError('missing retained delivery')
                    continue
                if _record(delivered.root/name)!=_record(path):raise ValueError('normalized delivery/root-error certificate differs')
            yield delivered,counts


def _verify_rejected(field,root,scratch,settings,resource_settings,identity,items,stop):
    from .tifxyz_guide import stride_schedule,build_guide
    from .tifxyz_project import make_grid
    for index,(level,accepted) in enumerate(zip(settings.levels,items)):
        for stride in stride_schedule(field.shape):
            original=root/'work'/f'level-{index:04d}'/f'stride-{stride:04d}'
            guide=None;fresh=None
            for refinement in range(65):
                attempt=original/f'pitch-{refinement:04d}'
                if not (attempt/'projected/STORE.json').exists() or attempt==root/accepted['attempt']:continue
                _check(stop)
                if guide is None:
                    guide=_load_guide(original/'guide')
                    fresh=scratch/f'retained-{index:04d}-{stride:04d}';fresh.mkdir()
                    regenerated=build_guide(_build_coarse(field,stride,fresh/'coarse',stop,resource_settings,scratch),level,settings,stop)
                    if _guide_signature(regenerated)!=_guide_signature(guide):raise ValueError('retained guide regeneration differs')
                grid=make_grid(guide,settings.pitch_um/2**refinement,settings.max_pixels,level=level)
                _admit(resource_settings,scratch,field,math.prod(grid['shape_yx']))
                destination=fresh/f'pitch-{refinement:04d}';destination.mkdir()
                with _reconstruct(field,attempt,destination,guide,grid,settings,identity,_budget(resource_settings,scratch),stop,complete=False):pass


def _verify_levels(field, root, settings, identity, levels, stop):
    """Shared independent acceptance gate for verification and resumed publication."""
    from .tifxyz_guide import stride_schedule, build_guide
    from .tifxyz_audit import audit_level
    used=_space(root,settings)
    prefix='.'+root.name+'-verify-'
    # Abrupt exits bypass TemporaryDirectory cleanup; retained siblings still consume scratch.
    for retained in root.parent.iterdir():
        if retained.name.startswith(prefix): used+=_space(retained,settings)
    resource_settings=replace(settings,scratch_bytes=max(0,settings.scratch_bytes-used))
    _admit(resource_settings,root.parent,field,owned=False)
    field.validate()
    with tempfile.TemporaryDirectory(prefix=prefix,dir=root.parent) as temporary:
        scratch=Path(temporary);budget=_budget(resource_settings,scratch)
        for index,(item,level) in enumerate(zip(levels,settings.levels)):
            _check(stop)
            guide=_validate_level_record(root,item,index,level,settings)
            if item['stride'] not in stride_schedule(field.shape): raise ValueError('undeclared guide stride')
            grid=item['grid'];_admit(resource_settings,scratch,field,math.prod(grid['shape_yx']))
            work=scratch/f'level-{index:04d}';work.mkdir()
            fresh=build_guide(_build_coarse(field,item['stride'],work/'coarse',stop,resource_settings,scratch),level,settings,stop)
            if _guide_signature(fresh)!=_guide_signature(guide): raise ValueError('guide regeneration differs')
            with _reconstruct(field,root/item['attempt'],work,fresh,grid,settings,identity,budget,stop,complete=True) as (delivered,counts):
                _segment(root/item['segment'],item['manifest'],grid,settings,delivered,stop,expected_status=item['status'])
                audit=audit_level(field,delivered,fresh,settings,stop,expected_accelerator=identity['audit_accelerator'])
                if _audit_status(audit,settings.coverage_policy)!=item['status']:raise ValueError('independent coverage quality differs')
                verified=_read(root/item['segment']/'VERIFIED.json')
                if verified['audit']!=audit or verified['counts']!=counts or verified['guide_sha256']!=digest(_guide_signature(fresh)):
                    raise ValueError('independent audit receipt differs')
        _verify_rejected(field,root,scratch,settings,resource_settings,identity,levels,stop)
        field.validate();_check(stop)


def _verify_source(field, out, stop=None):
    """Read-only verification with a caller-provided, exactly bound source adapter."""
    root=_path(out);_record(root/'.lock')
    fd=os.open(root/'.lock',os.O_RDONLY|os.O_NOFOLLOW)
    try:
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return _verify_locked_source(field,root,stop)
    finally:os.close(fd)


def _verify_locked_source(field, root, stop):
    identity=_read(root/'EXPORT.json');settings=_decode_settings(identity['effective_settings'])
    try:
        _admit(settings,root.parent,field,owned=False)
        if _identity(field,settings)!=identity: raise ValueError('export source/settings/code identity differs')
        marker,ready=_publication_marker(root);receipt=_read(root/'TIFXYZ.json')
        status=marker.stem;complete=status=='READY'
        if status=='PARTIAL' and settings.coverage_policy!='allow_partial':raise ValueError('partial export requires explicit policy')
        if receipt.get('source_verification')!=identity['source_verification']:raise ValueError('source verification scope differs')
        for value in (identity,ready,receipt):
            if any(type(value.get(k)) is not type(v) or value.get(k)!=v for k,v in FLAGS.items()): raise ValueError('acceptance flags differ')
        if (ready.get('status')!=status or receipt.get('status')!=status or _publication_status(receipt.get('levels',[]))!=status
            or ready.get('coverage_complete') is not complete or receipt.get('coverage_complete') is not complete
            or ready.get('tifxyz_sha256')!=file_sha(root/'TIFXYZ.json')
            or ready.get('identity_sha256')!=digest(identity) or receipt.get('identity_sha256')!=digest(identity) or receipt.get('reader')!=READER):
            raise ValueError('publication binding/quality differs')
        if receipt['producer_snapshot']!=_snapshot(field,root,resume=True): raise ValueError('producer snapshot differs')
        if len(receipt['levels'])!=len(settings.levels): raise ValueError('level count differs')
        if _inventory(root/'work',recursive=True)!=receipt['work_inventory']: raise ValueError('retained work artifacts differ')
        if {p.name for p in root.glob('level-*')}!={f'level-{i:04d}' for i in range(len(settings.levels))}:
            raise ValueError('published segment set differs')
        _verify_levels(field,root,settings,identity,receipt['levels'],stop)
        # Recheck publication after the expensive scan; never write into accepted output.
        if _publication_marker(root)!=(marker,ready) or _read(root/'TIFXYZ.json')!=receipt: raise ValueError('publication changed during verification')
        if _read(root/'EXPORT.json')!=identity or _inventory(root/'work',recursive=True)!=receipt['work_inventory']:
            raise ValueError('export artifacts changed during verification')
        if _code_identity()!=identity['implementation']:raise ValueError('verification implementation changed')
        _check_accelerator(settings,identity)
        _check_published_segments(root,receipt['levels'],stop)
        return dict(status='TIFXYZ_VERIFIED' if complete else 'TIFXYZ_PARTIAL_VERIFIED',coverage_complete=complete,
            independent_reprojection=True,levels=len(receipt['levels']),source_io=dict(field.stats),source_verification=identity['source_verification'],**FLAGS)
    except (Paused,Blocked) as error:
        result=dict(status='PAUSED' if isinstance(error,Paused) else 'BLOCKED',**FLAGS)
        if isinstance(error,Blocked):result['reasons']=[dict(code=error.code,detail=error.detail)]
        return result


def export_field(field_run, out, settings, resume=False, stop=None):
    """Export a CLI FIELD_RUN inside the resource-supervised Linux worker."""
    from .tifxyz_runtime import supervise
    return supervise(dict(operation='export',field_run=str(_path(field_run)),out=str(_path(out)),
        settings=asdict(settings) if isinstance(settings,Settings) else validate_request(settings),resume=resume),stop)


def verify_export(out, stop=None):
    """Verify READY in a fresh admitted Linux worker without modifying its files."""
    from .tifxyz_runtime import supervise
    return supervise(dict(operation='verify',out=str(_path(out))),stop)


def _worker(request, stop):
    from .tifxyz_field import open_physical_field, selected_frame
    from .tifxyz_contract import load_settings
    if request['operation']=='verify':
        identity=_read(Path(request['out'])/'EXPORT.json');descriptor=identity['source_opener']
        if descriptor['kind']!='field-run': raise ValueError('historical source requires its explicit bound adapter')
        settings=_decode_settings(identity['effective_settings'])
        roi=dict(lo_zyx=list(settings.roi_lo_zyx),hi_zyx=list(settings.roi_hi_zyx))
        with open_physical_field(descriptor['path'],roi=roi) as field: return _verify_source(field,request['out'],stop)
    config=request['settings'];effective='memory_bytes' in config
    roi=dict(lo_zyx=config['roi_lo_zyx'],hi_zyx=config['roi_hi_zyx']) if effective else config.get('roi')
    with open_physical_field(request['field_run'],roi=roi) as field:
        frame=selected_frame(field.binding) if not effective and config.get('output_frame','selected-source')=='selected-source' else config['output_frame']
        settings=_decode_settings(config) if effective else load_settings(config,field.binding['geometry'],frame)
        return _export_source(field,request['out'],settings,request['resume'],stop)
