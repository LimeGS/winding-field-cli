"""Portable OME-Zarr/Lasagna and AFV interchange, with explicit coordinate frames."""

import io
import json
from pathlib import Path
import sqlite3

import numpy as np
from numcodecs import Blosc

from .artifacts import Tiles
from .util import atomic_bytes,atomic_json,digest,file_sha


def export_db(run,stage,identity):
    db=sqlite3.connect(run/'exports.sqlite')
    db.executescript('CREATE TABLE IF NOT EXISTS identities (stage TEXT PRIMARY KEY,sha TEXT);'
                     'CREATE TABLE IF NOT EXISTS files (path TEXT PRIMARY KEY,sha TEXT);')
    sha=digest(identity)
    old=db.execute('SELECT sha FROM identities WHERE stage=?',(stage,)).fetchone()
    if old and old[0]!=sha:
        db.close(); raise ValueError('export input identity changed: '+stage)
    db.execute('INSERT OR IGNORE INTO identities VALUES (?,?)',(stage,sha)); db.commit()
    return db


def exported(db,run,path):
    record=db.execute('SELECT sha FROM files WHERE path=?',(str(path.relative_to(run)),)).fetchone()
    if record:
        if not path.is_file() or file_sha(path)!=record[0]:
            raise ValueError('export product changed or missing: '+str(path))
        return True
    return False


def record_export(db,run,path):
    db.execute('INSERT OR REPLACE INTO files VALUES (?,?)',(str(path.relative_to(run)),file_sha(path)))
    db.commit()


def export_lasagna(run,stop=None):
    run=Path(run); meta=json.loads((run/'PREPARED.json').read_bytes())
    root=run/'lasagna'; root.mkdir(exist_ok=True)
    maps=Tiles(run/'maps'); codec=Blosc(cname='lz4',clevel=5,shuffle=Blosc.SHUFFLE)
    if maps.identity()!=meta['tiles_sha256']:
        maps.close(); raise ValueError('export input maps changed')
    db=export_db(run,'lasagna',meta)
    names=['presence','nx','ny','surface_presence','surface_nx','surface_ny']
    for name in names:
        folder=root/(name+'.ome.zarr')
        atomic_json(folder/'.zgroup',{'zarr_format':2})
        atomic_json(folder/'.zattrs',{'multiscales':[{'version':'0.4','axes':[
            {'name':a,'type':'space','unit':'micrometer'} for a in 'zyx'],
            'datasets':[{'path':'0','coordinateTransformations':[
                {'type':'scale','scale':meta['spacing_xyz'][::-1]},
                {'type':'translation','translation':meta['origin_xyz'][::-1]}]}]}]})
        atomic_json(folder/'0/.zarray',dict(zarr_format=2,shape=list(maps.shape),chunks=[maps.chunk]*3,
                    dtype='|u1',compressor=codec.get_config(),fill_value=0,filters=None,order='C',dimension_separator='.'))
    try:
        for key in maps.keys():
            if stop and stop():
                return None
            paths=[root/(name+'.ome.zarr')/'0'/'.'.join(map(str,key)) for name in names]
            done=[exported(db,run,path) for path in paths]
            if all(done):
                continue
            data=maps.load(key)
            tangent=data['tangents']; normal=data['normals']
            tf=np.where(tangent[...,2]<0,-1.,1.); nf=np.where(normal[...,2]<0,-1.,1.)
            values=[data['presence']*255,tangent[...,0]*tf*127+128,tangent[...,1]*tf*127+128,
                    data['support']*255,normal[...,0]*nf*127+128,normal[...,1]*nf*127+128]
            for path,value,ready in zip(paths,values,done):
                if stop and stop():
                    return None
                if ready:
                    continue
                array=np.zeros((maps.chunk,)*3,np.uint8)
                array[tuple(slice(0,n) for n in value.shape)]=np.rint(np.clip(value,0,255)).astype(np.uint8)
                atomic_bytes(path,codec.encode(array))
                record_export(db,run,path)
        if stop and stop():
            return None
    finally:
        maps.close(); db.close()
    for kind,prefix in (('fibres',''),('surfaces','surface_')):
        groups={name:dict(zarr=prefix+name+'.ome.zarr/0',scaledown=0,channels=[name]) for name in ('presence','nx','ny')}
        atomic_json(root/(kind+'.lasagna.json'),dict(version=2,source_to_base=1.,grad_mag_encode_scale=1000.,
                     grad_mag_factor=1.,groups=groups,base_shape_zyx=list(maps.shape),
                     crops=[[0,0,0,*maps.shape[::-1]]],coordinate_frame='ROI-local output grid',
                     world_origin_xyz_um=meta['origin_xyz'],world_spacing_xyz_um=meta['spacing_xyz'],
                     original_ct=meta['source'],prediction=meta['prediction']))
    return True


def export_afv(run,stop=None):
    run=Path(run); meta=json.loads((run/'PREPARED.json').read_bytes())
    source=meta['source']; root=run/'fibres'; destination=root/'fibres.afv'; pending=root/'fibres.partial.afv'
    fibres=json.loads((run/'FIBRES.json').read_bytes())
    if file_sha(root/'fibres.sqlite')!=fibres['sha256']:
        raise ValueError('export input fibres changed')
    identity=digest([meta,fibres['sha256']])
    ledger=export_db(run,'afv',identity)
    if exported(ledger,run,destination):
        ledger.close(); return file_sha(destination)
    if destination.exists():
        # Recover publication interrupted after rename but before the ledger commit.
        with sqlite3.connect(destination) as finished:
            row=finished.execute("SELECT value FROM metadata WHERE key='export_identity'").fetchone()
            complete=finished.execute("SELECT value FROM metadata WHERE key='complete'").fetchone()
        if row!=(json.dumps(identity),) or complete!=('true',):
            ledger.close(); raise ValueError('AFV export identity changed or incomplete')
        record_export(ledger,run,destination); ledger.close(); return file_sha(destination)
    src=sqlite3.connect((root/'fibres.sqlite').resolve().as_uri()+'?mode=ro',uri=True)
    db=sqlite3.connect(pending)
    db.executescript('PRAGMA application_id=1447249475; PRAGMA user_version=1; PRAGMA foreign_keys=ON;'
        'CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);'
        'CREATE TABLE IF NOT EXISTS fibers(id INTEGER PRIMARY KEY,name TEXT NOT NULL,family TEXT NOT NULL,point_count INTEGER NOT NULL,'
        'length REAL NOT NULL,min_x REAL,max_x REAL,min_y REAL,max_y REAL,min_z REAL,max_z REAL,annotation TEXT NOT NULL);'
        'CREATE TABLE IF NOT EXISTS blocks(id INTEGER PRIMARY KEY,fiber_id INTEGER REFERENCES fibers(id),first_segment INTEGER,points BLOB,UNIQUE(fiber_id,first_segment));'
        'CREATE INDEX IF NOT EXISTS fibers_length ON fibers(length DESC,id);'
        'CREATE VIRTUAL TABLE IF NOT EXISTS block_bounds USING rtree(id,min_x,max_x,min_y,max_y,min_z,max_z);')
    try:
        row=db.execute("SELECT value FROM metadata WHERE key='export_identity'").fetchone()
        if row and row!=(json.dumps(identity),):
            raise ValueError('partial AFV export identity changed')
        db.execute('INSERT OR IGNORE INTO metadata VALUES (?,?)',('export_identity',json.dumps(identity))); db.commit()
        cursor=db.execute("SELECT value FROM metadata WHERE key='export_last_id'").fetchone()
        last=int(json.loads(cursor[0])) if cursor else 0
        retained=src.execute('SELECT count(*) FROM curves WHERE id<=?',(last,)).fetchone()[0]
        if (db.execute('SELECT coalesce(max(id),0) FROM fibers').fetchone()[0]!=last
                or db.execute('SELECT count(*) FROM fibers').fetchone()[0]!=retained
                or db.execute('SELECT count(*) FROM blocks').fetchone()!=db.execute('SELECT count(*) FROM block_bounds').fetchone()
                or db.execute('PRAGMA foreign_key_check').fetchone()):
            raise ValueError('partial AFV export inventory changed')
        family_column='family' if 'family' in {row[1] for row in src.execute('PRAGMA table_info(curves)')} else "'unknown'"
        for fid,tile,blob,family in src.execute(f'SELECT id,tile,points,{family_column} FROM curves ORDER BY id'):
            if stop and stop():
                return None
            world=np.load(io.BytesIO(blob),allow_pickle=False)
            points=((world-source['origin_xyz'])/source['spacing_xyz']).astype('<f8')
            if (points.ndim!=2 or points.shape[1]!=3 or len(points)<2 or not np.isfinite(points).all()
                    or (points<0).any() or (points>=np.asarray(source['shape'])[::-1]).any()):
                raise ValueError('invalid AFV source coordinates')
            lo=points.min(0); hi=points.max(0); box=[float(v) for pair in zip(lo,hi) for v in pair]
            length=float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum())
            annotation=dict(type='vc3d_fiber',version=1,sequence=fid,tile=tile,prediction=meta['prediction'],
                            automatically_traced=True,fragment=True,control_points=[points[0].tolist(),points[-1].tolist()],
                            length_voxels=length,length_mm=float(np.linalg.norm(np.diff(world,axis=0),axis=1).sum()/1000))
            if family in ('V','H'):
                annotation.update(hv_classification=dict(manual_tag=family),classification_source='model_prediction_family_locked_tracing')
            record=(fid,f'fiber-{fid}',family,len(points),length,*box,json.dumps(annotation))
            blocks=[(first,points[first:first+257]) for first in range(0,len(points)-1,256)]
            if fid<=last:
                saved=db.execute('SELECT id,first_segment,points FROM blocks WHERE fiber_id=? ORDER BY first_segment',(fid,)).fetchall()
                if db.execute('SELECT * FROM fibers WHERE id=?',(fid,)).fetchone()!=record or len(saved)!=len(blocks):
                    raise ValueError('partial AFV export record changed: '+str(fid))
                for (bid,start,raw),(first,block) in zip(saved,blocks):
                    bounds=db.execute('SELECT min_x,max_x,min_y,max_y,min_z,max_z FROM block_bounds WHERE id=?',(bid,)).fetchone()
                    box=[float(v) for pair in zip(block.min(0),block.max(0)) for v in pair]
                    if start!=first or raw!=block.tobytes() or bounds is None or not np.allclose(bounds,box,rtol=2e-7,atol=1e-6):
                        raise ValueError('partial AFV export block changed: '+str(bid))
                continue
            db.execute('INSERT INTO fibers VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',record)
            for first,block in blocks:
                bid=db.execute('INSERT INTO blocks(fiber_id,first_segment,points) VALUES (?,?,?)',
                               (fid,first,block.tobytes())).lastrowid
                box=[float(v) for pair in zip(block.min(0),block.max(0)) for v in pair]
                db.execute('INSERT INTO block_bounds VALUES (?,?,?,?,?,?,?)',(bid,*box))
            db.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)',('export_last_id',json.dumps(fid)))
            db.commit()
        if stop and stop():
            return None
        count,points_total=db.execute('SELECT count(*),coalesce(sum(point_count),0) FROM fibers').fetchone()
        root_meta=dict(type='vc3d_fiber_collection',version=1,scale=1,source=source['source'],
                       coordinate_order='XYZ',coordinate_units='voxel',coordinate_base_shape_zyx=source['shape'],
                       vc_open_data_source_coordinate_level=source['level'])
        if np.allclose(source['spacing_xyz'],source['spacing_xyz'][0],rtol=1e-6,atol=0):
            root_meta['voxel_size_mm']=source['spacing_xyz'][0]/1000
        metadata=dict(uuid=digest([source,meta['tiles_sha256']]),fiber_count=count,point_count=points_total,
                      block_count=db.execute('SELECT count(*) FROM blocks').fetchone()[0],block_segments=256,
                      frame=dict(source=source,coordinate_order='XYZ',units='selected-source voxels',
                                 level=source['level'],spacing_xyz_um=source['spacing_xyz'],origin_xyz_um=source['origin_xyz']),
                      root=root_meta,complete=True)
        db.executemany('INSERT OR REPLACE INTO metadata VALUES (?,?)',[(k,json.dumps(v)) for k,v in metadata.items()])
        db.commit()
        if (db.execute('PRAGMA integrity_check').fetchall()!=[('ok',)]
                or db.execute('PRAGMA foreign_key_check').fetchone()
                or db.execute("SELECT rtreecheck('block_bounds')").fetchone()!=('ok',)):
            raise ValueError('corrupt AFV export')
    finally:
        db.close(); src.close(); ledger.close()
    pending.replace(destination)
    with sqlite3.connect(run/'exports.sqlite') as ledger:
        record_export(ledger,run,destination)
    return file_sha(destination)
