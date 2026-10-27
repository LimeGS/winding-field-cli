"""Read-only, calibrated AFV import into bounded world-coordinate curve blocks."""

from contextlib import closing
from functools import lru_cache
import hashlib
import heapq
import io
import itertools
import json
import math
from pathlib import Path
import shutil
import sqlite3

import numpy as np

from .artifacts import Tiles
from .io import http_url
from .splines import bind
from .util import atomic_json,digest,file_sha


def _open(path):
    path=Path(path).expanduser().resolve()
    if not path.is_file(): raise FileNotFoundError('AFV not found: '+str(path))
    for suffix in ('-wal','-journal'):
        journal=Path(str(path)+suffix)
        if journal.exists() and journal.stat().st_size:
            raise ValueError('AFV has an uncheckpointed journal; close/checkpoint its writer first')
    return sqlite3.connect(path.as_uri()+'?mode=ro&immutable=1',uri=True)


def _header(db):
    if db.execute('PRAGMA application_id').fetchone()!=(1447249475,) or db.execute('PRAGMA user_version').fetchone()!=(1,):
        raise ValueError('AFV schema v1 required')
    for table,columns in [('metadata',{'key','value'}),('fibers',{'id','point_count','annotation'}),
                          ('blocks',{'id','fiber_id','first_segment','points'})]:
        if not columns<={row[1] for row in db.execute(f'PRAGMA table_info({table})')}:
            raise ValueError('invalid AFV schema: '+table)
    if (db.execute('SELECT count(*) FROM metadata').fetchone()[0]>64
            or db.execute('SELECT 1 FROM metadata WHERE length(value)>1048576 LIMIT 1').fetchone()):
        raise ValueError('AFV metadata exceeds bounded header size')
    header={key:json.loads(value) for key,value in db.execute('SELECT key,value FROM metadata')}
    if header.get('complete') is not True: raise ValueError('incomplete AFV')
    root=header.get('root',{})
    if root.get('coordinate_order')!='XYZ':
        raise ValueError('AFV requires XYZ coordinate order')
    counts=db.execute('SELECT count(*),coalesce(sum(point_count),0) FROM fibers').fetchone()
    blocks=db.execute('SELECT count(*) FROM blocks').fetchone()[0]
    if (counts!=(header.get('fiber_count'),header.get('point_count')) or blocks!=header.get('block_count')
            or db.execute('SELECT 1 FROM blocks b LEFT JOIN fibers f ON f.id=b.fiber_id WHERE f.id IS NULL LIMIT 1').fetchone()):
        raise ValueError('AFV block/fibre inventory mismatch')
    if (db.execute('SELECT 1 FROM blocks WHERE length(points)<48 OR length(points)>6168 OR length(points)%24!=0 LIMIT 1').fetchone()
            or db.execute('SELECT 1 FROM fibers WHERE length(annotation)>1048576 LIMIT 1').fetchone()):
        raise ValueError('invalid AFV block or annotation size')
    if db.execute('PRAGMA integrity_check').fetchall()!=[('ok',)]: raise ValueError('corrupt AFV')
    return header


def source_identity(path,geometry_path,source):
    path=Path(path).expanduser().resolve(); sha=file_sha(path)
    try:
        with closing(_open(path)) as db: header=_header(db)
    except (sqlite3.Error,KeyError,TypeError,json.JSONDecodeError) as error:
        raise ValueError('invalid AFV metadata/schema') from error
    frame=header.get('frame',{}); embedded=None
    if not isinstance(frame,dict): raise ValueError('invalid AFV frame')
    if 'source' in frame:
        original=frame['source']
        if not isinstance(original,dict): raise ValueError('invalid AFV source frame')
        root=header['root']
        if (frame.get('coordinate_order')!='XYZ' or frame.get('units')!='selected-source voxels'
                or root.get('coordinate_units')!='voxel'
                or frame.get('level')!=original.get('level')
                or frame.get('origin_xyz_um')!=original.get('origin_xyz')
                or frame.get('spacing_xyz_um')!=original.get('spacing_xyz')
                or root.get('coordinate_base_shape_zyx')!=original.get('shape')
                or root.get('vc_open_data_source_coordinate_level')!=original.get('level')
                or ('source' in root and http_url(root['source'])!=http_url(original.get('source')))):
            raise ValueError('conflicting AFV embedded coordinate frame')
        if original.get('level')==source.get('level'):
            for name in ('shape','spacing_xyz','origin_xyz'):
                if original.get(name)!=source.get(name): raise ValueError('AFV selected-source frame mismatch')
        embedded=dict(source=original['source'],origin_xyz_um=frame['origin_xyz_um'],spacing_xyz_um=frame['spacing_xyz_um'])
    explicit=None; descriptor_sha=None
    if geometry_path:
        geometry_path=Path(geometry_path).expanduser().resolve()
        if geometry_path.stat().st_size>65536: raise ValueError('AFV geometry descriptor too large')
        raw=geometry_path.read_bytes(); descriptor_sha=hashlib.sha256(raw).hexdigest()
        explicit=json.loads(raw)
        if not isinstance(explicit,dict) or set(explicit)!={'source','origin_xyz_um','spacing_xyz_um'}:
            raise ValueError('AFV geometry requires source, origin_xyz_um and spacing_xyz_um')
    geometry=embedded or explicit
    if geometry is None: raise ValueError('ambiguous AFV geometry; supply --fibres-geometry')
    if embedded and explicit and embedded!=explicit: raise ValueError('explicit AFV geometry conflicts with embedded frame')
    if http_url(geometry['source'])!=source['source']: raise ValueError('AFV source mismatch')
    for name in ('origin_xyz_um','spacing_xyz_um'):
        value=np.asarray(geometry[name],float)
        if value.shape!=(3,) or not np.isfinite(value).all() or (name=='spacing_xyz_um' and (value<=0).any()):
            raise ValueError('invalid AFV geometry '+name)
    if file_sha(path)!=sha: raise ValueError('AFV source changed during validation')
    return dict(path=str(path),sha256=sha,geometry_path=str(geometry_path) if geometry_path else None,
        geometry_sha256=descriptor_sha,**geometry,input_count=header['fiber_count'],block_count=header['block_count'])


def _pieces(points,maps,stop=None):
    """Split at exact voxel-cell boundaries: a sparse edge cannot jump a narrow gap."""
    @lru_cache(maxsize=8)
    def tile(key): return maps.load(key,('support',))['support']
    def supported(xyz):
        q=xyz[::-1]; low=np.floor(q).astype(int); fraction=q-low
        for corner in itertools.product((0,1),repeat=3):
            c=np.asarray(corner)
            if not np.all(np.where(c,fraction,1-fraction)>0): continue
            ij=np.minimum(low+c,np.asarray(maps.shape)-1)
            if not tile(tuple(ij//maps.chunk))[tuple(ij%maps.chunk)]: return False
        return True
    high=np.asarray(maps.shape)[::-1]-1; piece=[]
    for a,b in zip(points[:-1],points[1:]):
        delta=b-a
        near=np.divide(-a,delta,out=np.full(3,-np.inf),where=delta!=0)
        far=np.divide(high-a,delta,out=np.full(3,np.inf),where=delta!=0)
        enter=max(0.,np.minimum(near,far).max())
        leave=min(1.,np.maximum(near,far).min())
        if leave<=enter or np.any((delta==0)&((a<0)|(a>high))) or not np.any(delta):
            if len(piece)>1: yield np.asarray(piece)
            piece=[]; continue
        start=np.clip(a+enter*delta,0,high); end=np.clip(a+leave*delta,0,high); d=end-start
        axes=[]
        for i in range(3):
            if not d[i]: continue
            values=range(math.floor(min(start[i],end[i]))+1,math.ceil(max(start[i],end[i])))
            if d[i]<0: values=reversed(values)
            # Bind each axis before the lazy merge advances the generators.
            axes.append(map(lambda v,o=start[i],s=d[i]:(v-o)/s,values))
        previous=0.
        for t in itertools.chain(heapq.merge(*axes),[1.]):
            if stop and stop(): raise InterruptedError('AFV import paused')
            if t<=previous: continue
            left=np.clip(start+previous*d,0,high); right=np.clip(start+t*d,0,high)
            if supported((left+right)/2):
                if piece and not np.allclose(piece[-1],left,rtol=0,atol=1e-9):
                    if len(piece)>1: yield np.asarray(piece)
                    piece=[]
                if not piece: piece=[left,right]
                else: piece.append(right)
                if len(piece)==257:
                    yield np.asarray(piece); piece=[right]
            else:
                if len(piece)>1: yield np.asarray(piece)
                piece=[]
            previous=t
    if len(piece)>1: yield np.asarray(piece)


def validate_retained(run,config):
    run=Path(run); root=run/'fibres'; binding=root/'IMPORT.json'
    if not binding.exists(): return
    meta=json.loads((run/'PREPARED.json').read_bytes()); identity=json.loads(binding.read_bytes())
    if identity['input']!=source_identity(config['fibres'],config.get('fibres_geometry'),meta['source']):
        raise ValueError('external AFV input identity changed')
    if identity['maps']!=meta['tiles_sha256']: raise ValueError('AFV import maps changed')
    receipt=run/'FIBRES.json'
    if receipt.exists():
        saved=json.loads(receipt.read_bytes())
        if saved['identity']!=identity or file_sha(root/'fibres.sqlite')!=saved['sha256']:
            raise ValueError('imported AFV payload or inventory changed')


def import_fibres(run,config,stop=None):
    run=Path(run); meta=json.loads((run/'PREPARED.json').read_bytes())
    source=source_identity(config['fibres'],config.get('fibres_geometry'),meta['source'])
    identity=dict(input=source,maps=meta['tiles_sha256'],origin=meta['origin_xyz'],spacing=meta['spacing_xyz'])
    root=run/'fibres'; root.mkdir(exist_ok=True); bind(root/'IMPORT.json',identity)
    validate_retained(run,config)
    if (run/'FIBRES.json').exists(): return json.loads((run/'FIBRES.json').read_bytes())
    maps=Tiles(run/'maps'); src=_open(config['fibres']); db=sqlite3.connect(root/'fibres.sqlite')
    try:
        if maps.identity()!=meta['tiles_sha256']: raise ValueError('AFV import maps changed')
        db.executescript('CREATE TABLE IF NOT EXISTS done(fid INTEGER,block INTEGER,sha TEXT,PRIMARY KEY(fid,block));'
            'CREATE TABLE IF NOT EXISTS curves(id INTEGER PRIMARY KEY,tile TEXT,points BLOB,family TEXT,source_id INTEGER,source_block INTEGER,annotation TEXT);'
            'CREATE INDEX IF NOT EXISTS curves_source ON curves(source_id,source_block);'
            'CREATE VIRTUAL TABLE IF NOT EXISTS curve_bounds USING rtree(id,min_x,max_x,min_y,max_y,min_z,max_z);')
        if (db.execute('SELECT 1 FROM curves c LEFT JOIN done d ON d.fid=c.source_id AND d.block=c.source_block '
                       'LEFT JOIN curve_bounds b ON b.id=c.id WHERE d.fid IS NULL OR b.id IS NULL LIMIT 1').fetchone()
                or db.execute('SELECT 1 FROM curve_bounds b LEFT JOIN curves c ON c.id=b.id WHERE c.id IS NULL LIMIT 1').fetchone()):
            raise ValueError('imported AFV inventory changed')
        family_column='family' if 'family' in {r[1] for r in src.execute('PRAGMA table_info(fibers)')} else "'unknown'"
        total_blocks=0
        for fid,family,count,annotation in src.execute(f'SELECT id,{family_column},point_count,annotation FROM fibers ORDER BY id'):
            if type(count) is not int or count<2 or not isinstance(annotation,str) or len(annotation)>1048576:
                raise ValueError('invalid AFV fibre metadata')
            json.loads(annotation); family=family or 'unknown'; cursor=0; previous=None
            for first,raw in src.execute('SELECT first_segment,points FROM blocks WHERE fiber_id=? ORDER BY first_segment',(fid,)):
                if stop and stop(): return dict(status='PAUSED')
                if first!=cursor or len(raw)%24 or not 2<=len(raw)//24<=257:
                    raise ValueError('invalid or discontinuous AFV block')
                points=np.frombuffer(raw,dtype='<f8').reshape(-1,3)
                if not np.isfinite(points).all() or (previous is not None and not np.array_equal(previous,points[0])):
                    raise ValueError('nonfinite or discontinuous AFV block points')
                previous=points[-1]; cursor+=len(points)-1; total_blocks+=1
                world=points*source['spacing_xyz_um']+source['origin_xyz_um']
                if not np.isfinite(world).all(): raise ValueError('nonfinite AFV world coordinates')
                local=(world-meta['origin_xyz'])/meta['spacing_xyz']
                old=db.execute('SELECT sha FROM done WHERE fid=? AND block=?',(fid,first)).fetchone()
                rows=db.execute('SELECT id,points,family,annotation FROM curves WHERE source_id=? AND source_block=? ORDER BY id',(fid,first)) if old else None
                checksum=hashlib.sha256(digest([family,annotation]).encode())
                db.execute('BEGIN')
                for piece in _pieces(local,maps,stop):
                    world=piece*meta['spacing_xyz']+meta['origin_xyz']
                    buffer=io.BytesIO(); np.save(buffer,world,allow_pickle=False)
                    box=tuple(float(v) for pair in zip(world.min(0),world.max(0)) for v in pair)
                    blob=buffer.getvalue(); checksum.update(hashlib.sha256(blob).digest())
                    if old:
                        row=rows.fetchone()
                        if row is None: raise ValueError('imported AFV inventory changed')
                        cid,saved,label,note=row
                        bound=db.execute('SELECT min_x,max_x,min_y,max_y,min_z,max_z FROM curve_bounds WHERE id=?',(cid,)).fetchone()
                        if blob!=saved or label!=family or note!=annotation or bound is None or not np.allclose(bound,box,rtol=2e-7,atol=1e-6):
                            raise ValueError('imported AFV payload changed')
                    else:
                        if shutil.disk_usage(root).free<len(blob)+config.get('reserve_bytes',0)+65536:
                            raise OSError('insufficient space for AFV import and reserve')
                        cid=db.execute('INSERT INTO curves(tile,points,family,source_id,source_block,annotation) VALUES (?,?,?,?,?,?)',
                            ('imported',blob,family,fid,first,annotation)).lastrowid
                        db.execute('INSERT INTO curve_bounds VALUES (?,?,?,?,?,?,?)',(cid,*box))
                if old:
                    if old!=(checksum.hexdigest(),) or rows.fetchone() is not None:
                        raise ValueError('imported AFV inventory changed')
                else: db.execute('INSERT INTO done VALUES (?,?,?)',(fid,first,checksum.hexdigest()))
                db.commit()
            if cursor+1!=count: raise ValueError('AFV block point inventory mismatch')
        if total_blocks!=source['block_count'] or db.execute('SELECT count(*) FROM done').fetchone()[0]!=total_blocks:
            raise ValueError('imported AFV inventory changed')
        if stop and stop(): return dict(status='PAUSED')
        if source!=source_identity(config['fibres'],config.get('fibres_geometry'),meta['source']):
            raise ValueError('external AFV source changed during import')
        count=db.execute('SELECT count(*) FROM curves').fetchone()[0]
        kept=db.execute('SELECT count(DISTINCT source_id) FROM curves').fetchone()[0]
    except InterruptedError:
        db.rollback(); return dict(status='PAUSED')
    finally:
        maps.close(); src.close(); db.close()
    result=dict(status='FIBRES_READY',method='external-afv',spatial_index=True,count=count,
        input_count=source['input_count'],imported_sources=kept,excluded_sources=source['input_count']-kept,
        exclusion_reason='outside ROI or unsupported surface',identity=identity,sha256=file_sha(root/'fibres.sqlite'),
        coordinates='XYZ micrometres',physical_accuracy='UNVALIDATED')
    atomic_json(run/'FIBRES.json',result)
    return result
