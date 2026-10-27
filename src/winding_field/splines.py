"""Bounded, resumable adapters around Qual's pinned global spline tracer."""

from functools import lru_cache
import hashlib
import io
import itertools
import json
import os
from pathlib import Path
import shutil
import signal
import sqlite3
import subprocess
import sys

import numpy as np

from .artifacts import Tiles
from .util import atomic_json,digest,file_sha

SPLINE_REV='3c8b73628ca56cb28042b9b2e79439b2e1c693d2'


def bind(path,identity):
    if path.exists():
        if json.loads(path.read_bytes())!=identity:
            raise ValueError('spline input or implementation changed: '+path.name)
    else:
        atomic_json(path,identity)


def materialize_predictions(maps,root,stop=None,reserve_bytes=0):
    root=Path(root); root.mkdir(parents=True,exist_ok=True)
    destination=root/'probabilities.npy'; pending=root/'probabilities.partial.npy'
    marker=root/'PROBABILITIES.json'; binding=maps.identity()
    if marker.exists():
        saved=json.loads(marker.read_bytes())
        if saved['maps']!=binding or not destination.exists() or file_sha(destination)!=saved['sha256']:
            raise ValueError('spline probabilities changed or corrupt')
        return destination
    bind(root/'INPUT.json',dict(maps=binding,mask='zero unsupported observations',order='V,H'))
    db=sqlite3.connect(root/'input.sqlite')
    db.execute('CREATE TABLE IF NOT EXISTS tiles (key TEXT PRIMARY KEY,sha TEXT)'); db.commit()
    array=None
    try:
        path=destination if destination.exists() else pending
        shape=(2,*maps.shape)
        if not path.exists():
            if db.execute('SELECT count(*) FROM tiles').fetchone()[0]:
                raise ValueError('partial spline probabilities missing')
            if shutil.disk_usage(root).free<np.prod(shape,dtype=object)*4+4096+reserve_bytes:
                raise OSError('insufficient space for spline probabilities and reserve')
            array=np.lib.format.open_memmap(path,mode='w+',dtype='<f4',shape=shape)
        else:
            array=np.load(path,mmap_mode='r+' if path==pending else 'r',allow_pickle=False)
            if array.shape!=shape or array.dtype!=np.dtype('<f4'):
                raise ValueError('invalid partial spline probability geometry')
        for key in maps.keys():
            if stop and stop(): return None
            lo,hi=maps.box(key); region=(slice(None),)+tuple(slice(a,b) for a,b in zip(lo,hi))
            data=maps.load(key,('probabilities','support'))
            values=data['probabilities']
            if values.shape!=(*tuple(hi-lo),2) or not np.isfinite(values).all() or np.any((values<0)|(values>1)):
                raise ValueError('spline input must contain finite V/H probabilities in [0,1]')
            values=np.where(data['support'][...,None],values,0).transpose(3,0,1,2).astype('<f4')
            sha=hashlib.sha256(values.tobytes()).hexdigest(); tag='_'.join(map(str,key))
            old=db.execute('SELECT sha FROM tiles WHERE key=?',(tag,)).fetchone()
            if old:
                if old!=(sha,) or hashlib.sha256(array[region].tobytes()).hexdigest()!=sha:
                    raise ValueError('partial spline probabilities changed or corrupt')
                continue
            if path==destination:
                raise ValueError('published spline input has incomplete progress')
            if shutil.disk_usage(root).free<values.nbytes+reserve_bytes:
                raise OSError('insufficient space for spline probabilities and reserve')
            array[region]=values; array.flush()
            with path.open('rb') as handle: os.fsync(handle.fileno())
            db.execute('INSERT INTO tiles VALUES (?,?)',(tag,sha)); db.commit()
        del array; array=None
        if path!=destination: pending.replace(destination)
        atomic_json(marker,dict(maps=binding,sha256=file_sha(destination)))
        return destination
    finally:
        del array
        db.close()


def run_child(command,log,stop=None,reserve_bytes=0,threads=4):
    if stop and stop(): return False
    log=Path(log); log.parent.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,NUMBA_NUM_THREADS=str(threads),OMP_NUM_THREADS=str(threads))
    with log.open('ab') as stream:
        child=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        paused=False; low_space=False
        try:
            while child.poll() is None:
                low_space=shutil.disk_usage(log.parent).free<reserve_bytes
                if low_space or (stop and stop()):
                    paused=True; child.send_signal(signal.SIGINT)
                    break
                try: child.wait(timeout=.2)
                except subprocess.TimeoutExpired: pass
        finally:
            if child.poll() is None:
                try: child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill(); child.wait()
        if low_space: raise OSError('spline tracing reached the disk reserve; free space and resume')
        if paused: return False
        if child.returncode:
            raise RuntimeError(f'Qual spline generator failed ({child.returncode}); see {log}')
    return True


def upstream_hashes(root):
    return {str(p.relative_to(root)):file_sha(p)
            for p in itertools.chain((root/'work').rglob('*'),[root/'curves.afv'])
            if p.is_file() and p.name!='.lock'}


def supported_pieces(points,maps):
    """Split edges at unsupported corners, sampling at most 0.25 observation voxels."""
    @lru_cache(maxsize=8)
    def tile(key): return maps.load(key,('support',))['support']
    def supported(q):
        inside=((q>=0)&(q<=np.asarray(maps.shape)-1)).all(1)
        q=np.clip(q,0,np.asarray(maps.shape)-1); low=np.floor(q).astype(int); frac=q-low
        for corner in itertools.product((0,1),repeat=3):
            c=np.asarray(corner); ids=np.flatnonzero(inside & np.all(np.where(c,frac,1-frac)>0,axis=1))
            ij=np.minimum(low[ids]+c,np.asarray(maps.shape)-1)
            keys,inverse=np.unique(ij//maps.chunk,axis=0,return_inverse=True)
            for i,key in enumerate(keys):
                selected=np.flatnonzero(inverse==i)
                inside[ids[selected]] &= tile(tuple(key))[tuple((ij[selected]%maps.chunk).T)]
        return inside
    valid=np.ones(len(points)-1,bool)
    for i,(a,b) in enumerate(zip(points[:-1],points[1:])):
        steps=max(1,int(np.ceil(np.linalg.norm(b-a)/.25)))
        for start in range(0,steps+1,8192):
            t=np.arange(start,min(start+8192,steps+1))/steps
            if not supported((a[None]+t[:,None]*(b-a))[:,::-1]).all():
                valid[i]=False; break
    transitions=np.diff(np.r_[False,valid,False].astype('i1'))
    for a,b in zip(np.flatnonzero(transitions==1),np.flatnonzero(transitions==-1)):
        yield points[a:b+1]


def afv_points(db,fid,count):
    blocks=[]; cursor=0
    if not 2<=count<=131074:
        raise ValueError('invalid upstream spline point count')
    for first,blob in db.execute('SELECT first_segment,points FROM blocks WHERE fiber_id=? ORDER BY first_segment',(fid,)):
        if len(blob)%24 or not 2<=len(blob)//24<=257 or first!=cursor:
            raise ValueError('invalid or discontinuous AFV block')
        points=np.frombuffer(blob,dtype='<f8').reshape(-1,3)
        if not np.isfinite(points).all() or (blocks and not np.array_equal(blocks[-1][-1],points[0])):
            raise ValueError('invalid or discontinuous AFV block points')
        blocks.append(points if not blocks else points[1:]); cursor+=len(points)-1
    if cursor+1!=count or not blocks:
        raise ValueError('AFV block point inventory changed')
    return np.concatenate(blocks)


def import_curves(source,root,maps,meta,stop=None):
    root=Path(root)
    bind(root/'IMPORT.json',dict(afv=file_sha(source),maps=meta['tiles_sha256'],
                               origin=meta['origin_xyz'],spacing=meta['spacing_xyz']))
    src=sqlite3.connect(Path(source).resolve().as_uri()+'?mode=ro',uri=True)
    db=sqlite3.connect(root/'fibres.sqlite')
    try:
        header={k:json.loads(v) for k,v in src.execute('SELECT key,value FROM metadata')}
        if (header.get('complete') is not True or header['root']['coordinate_base_shape_zyx']!=list(maps.shape)
                or not np.isclose(header['root']['voxel_size_mm']*1000,meta['spacing_xyz'][0])):
            raise ValueError('upstream AFV coordinate frame or completion mismatch')
        if src.execute('PRAGMA integrity_check').fetchall()!=[('ok',)] or src.execute("SELECT rtreecheck('block_bounds')").fetchone()!=('ok',):
            raise ValueError('corrupt upstream AFV')
        db.executescript('CREATE TABLE IF NOT EXISTS done (fid INTEGER PRIMARY KEY,sha TEXT);'
            'CREATE TABLE IF NOT EXISTS curves (id INTEGER PRIMARY KEY,tile TEXT,points BLOB,family TEXT,source_id INTEGER);'
            'CREATE INDEX IF NOT EXISTS curves_source ON curves(source_id);'
            'CREATE VIRTUAL TABLE IF NOT EXISTS curve_bounds USING rtree(id,min_x,max_x,min_y,max_y,min_z,max_z);')
        if (db.execute('SELECT 1 FROM curves c LEFT JOIN done d ON d.fid=c.source_id '
                       'LEFT JOIN curve_bounds b ON b.id=c.id WHERE d.fid IS NULL OR b.id IS NULL LIMIT 1').fetchone()
                or db.execute('SELECT 1 FROM curve_bounds b LEFT JOIN curves c ON c.id=b.id '
                              'WHERE c.id IS NULL LIMIT 1').fetchone()):
            raise ValueError('imported spline inventory changed')
        total=0
        for fid,family,count in src.execute('SELECT id,family,point_count FROM fibers ORDER BY id'):
            if stop and stop(): return False
            total+=1
            points=afv_points(src,fid,count)
            if family not in ('V','H') or np.any(points<0) or np.any(points>=np.asarray(maps.shape)[::-1]):
                raise ValueError('invalid upstream spline family or coordinates')
            records=[]
            for p in supported_pieces(points,maps):
                world=p*meta['spacing_xyz']+meta['origin_xyz']
                buffer=io.BytesIO(); np.save(buffer,world,allow_pickle=False)
                box=tuple(float(v) for pair in zip(world.min(0),world.max(0)) for v in pair)
                records.append((buffer.getvalue(),box))
            sha=digest([family,[hashlib.sha256(blob).hexdigest() for blob,box in records]])
            old=db.execute('SELECT sha FROM done WHERE fid=?',(fid,)).fetchone()
            if old:
                rows=db.execute('SELECT id,family,points FROM curves WHERE source_id=? ORDER BY id',(fid,)).fetchall()
                if old!=(sha,) or len(rows)!=len(records): raise ValueError('imported spline inventory changed')
                for (cid,label,blob),(expected,box) in zip(rows,records):
                    bounds=db.execute('SELECT min_x,max_x,min_y,max_y,min_z,max_z FROM curve_bounds WHERE id=?',(cid,)).fetchone()
                    if label!=family or blob!=expected or bounds is None or not np.allclose(bounds,box,rtol=2e-7,atol=1e-6):
                        raise ValueError('imported spline changed or corrupt')
                continue
            with db:
                for blob,box in records:
                    cid=db.execute('INSERT INTO curves(tile,points,family,source_id) VALUES (?,?,?,?)',
                                   ('qual',blob,family,fid)).lastrowid
                    db.execute('INSERT INTO curve_bounds VALUES (?,?,?,?,?,?,?)',(cid,*box))
                db.execute('INSERT INTO done VALUES (?,?)',(fid,sha))
        if total!=header['fiber_count'] or db.execute('SELECT count(*) FROM done').fetchone()[0]!=total:
            raise ValueError('upstream spline inventory changed')
        return True
    finally:
        src.close(); db.close()


def trace_qual(run,config,stop=None):
    from .workflow import implementation_sha
    run=Path(run); meta=json.loads((run/'PREPARED.json').read_bytes()); root=run/'fibres'
    root.mkdir(exist_ok=True); maps=Tiles(run/'maps')
    try:
        if maps.identity()!=meta['tiles_sha256']: raise ValueError('prepared maps changed')
        identity=dict(maps=meta['tiles_sha256'],generator_revision=SPLINE_REV,
                      implementation=implementation_sha(),method='qual-global-splines-v1',backend='cpu')
        bind(root/'IDENTITY.json',identity)
        if (run/'FIBRES.json').exists():
            saved=json.loads((run/'FIBRES.json').read_bytes())
            if saved['identity']!=identity or file_sha(root/'fibres.sqlite')!=saved['sha256']:
                raise ValueError('spline receipt or curves changed')
            if file_sha(root/'probabilities.npy')!=saved['probabilities_sha256']:
                raise ValueError('spline probabilities changed')
            return saved
        reserve=config.get('reserve_bytes',0)
        path=materialize_predictions(maps,root,stop,reserve)
        if path is None: return dict(status='PAUSED')
        upstream=root/'upstream'; upstream.mkdir(exist_ok=True)
        output=upstream/'curves.afv'; receipt=upstream/'OUTPUT.json'; work=upstream/'work'
        seal=upstream/'WORK.json'; retained=upstream_hashes(upstream)
        if seal.exists():
            if json.loads(seal.read_bytes())!=retained:
                raise ValueError('upstream work changed or interrupted before sealing; use a new output')
        elif retained:
            raise ValueError('unverified upstream work; use a new output')
        else:
            atomic_json(seal,retained)
        if receipt.exists():
            if file_sha(output)!=json.loads(receipt.read_bytes())['sha256']:
                raise ValueError('upstream spline output changed')
        elif not output.exists():
            threads=min(config.get('threads',4),os.cpu_count() or 1)
            command=[sys.executable,'-m','avf_spline_generator','--predictions',str(path.resolve()),
                     '--voxel-size-um',str(meta['spacing_xyz'][0]),'--output',str(output.resolve()),
                     '--work-dir',str(work.resolve()),'--threads',str(threads),'--device','cpu',
                     '--coordinate-space',meta['frame']]
            if (work/'run.json').exists(): command.append('--resume')
            try:
                complete=run_child(command,upstream/'generator.log',stop,reserve,threads)
            finally:
                # shortcut: unsealed work after a hard kill is rejected; add commit-time hashes for crash recovery.
                atomic_json(seal,upstream_hashes(upstream))
            if not complete:
                return dict(status='PAUSED')
        if not import_curves(output,root,maps,meta,stop): return dict(status='PAUSED')
        atomic_json(receipt,dict(sha256=file_sha(output)))
        with sqlite3.connect(root/'fibres.sqlite') as db:
            count=db.execute('SELECT count(*) FROM curves').fetchone()[0]
        result=dict(status='FIBRES_READY',count=count,identity=identity,sha256=file_sha(root/'fibres.sqlite'),
                    probabilities_sha256=json.loads((root/'PROBABILITIES.json').read_bytes())['sha256'],
                    coordinates='XYZ micrometres',boundary_policy='global splines split at unsupported gaps and ROI boundary',
                    physical_accuracy='UNVALIDATED')
        atomic_json(run/'FIBRES.json',result)
        return result
    finally:
        maps.close()
