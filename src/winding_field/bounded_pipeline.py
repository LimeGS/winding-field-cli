"""CLI-owned native bounded preparation stages using the existing solver inputs."""

from contextlib import contextmanager
import gc
import hashlib
import io
import json
from pathlib import Path
import shutil
import sqlite3

import numpy as np

from .bounded_coarse import NativeMaps,endpoint_grid,produce
from .bounded_lasagna import LasagnaModel
from .bounded_predictions import LasagnaWindows
from .io import bounds
from .lasagna import runtime_identity,decode_observations
from .models import checkpoint,LASAGNA_MODEL_SHA,QUAL_MODEL_SHA,QualV2
from .splines import bind,supported_pieces
from .util import atomic_json,digest,file_sha


@contextmanager
def surface_maps(volume,root,config,lo,hi):
    model_path=checkpoint(config.get('lasagna_model_path'),config.get('offline',False),prediction='lasagna')
    runtime=runtime_identity(config.get('lasagna_python'))
    with LasagnaModel(runtime['executable'],model_path,config.get('device','cpu'),Path(root)/'lasagna-bounded.log',
                      threads=config.get('threads',1)) as model:
        windows=LasagnaWindows(volume,model,cache_bytes=64<<20)
        def read(start,end):
            a=start+lo;b=end+lo
            raw=windows.read_raw(a,b)
            grad,nx,ny,cos=model.finalize(raw)
            invalid=config.get('invalid_value')
            valid=np.ones(tuple(end-start),bool)
            if invalid is not None:valid=volume.sample([np.arange(x,y) for x,y in zip(a,b)])!=invalid
            return decode_observations(grad,nx,ny,cos,input_spacing_xyz=volume.spacing_xyz,
                                       encode_scale=1000.,grad_mag_factor=1.,validity=valid)
        yield NativeMaps(tuple(map(int,hi-lo)),read),runtime


def prepare(volume,run,config,stop=None):
    run=Path(run)
    lo,hi=bounds(config.get('roi_lo',[0,0,0]),config.get('roi_hi',list(volume.shape)),volume.shape)
    native=dict(shape_zyx=(hi-lo).tolist(),spacing_xyz=np.asarray(volume.spacing_xyz).tolist(),
                origin_xyz=(np.asarray(volume.origin_xyz)+lo[::-1]*volume.spacing_xyz).tolist(),frame=digest(volume.identity))
    grid=endpoint_grid(native,config['field_spacing_um'])
    with surface_maps(volume,run,config,lo,hi) as (maps,runtime):
        identity=dict(source=volume.identity,roi_lo=lo.tolist(),roi_hi=hi.tolist(),runtime=runtime,
                      lasagna_sha256=LASAGNA_MODEL_SHA,qual_sha256=QUAL_MODEL_SHA,
                      precision='float32-cpu-preampere/bfloat16-ampere',invalid_value=config.get('invalid_value'))
        status=produce(run/'coarse',maps,native,grid,identity,chunk=config.get('field_chunk',64),stop=stop,
                       reserve_bytes=config.get('reserve_bytes',0),quota_bytes=config.get('bounded_bytes',12<<30)//4)
    if status=='PAUSED':return dict(status='PAUSED')
    coarse=json.loads((run/'coarse/COARSE.json').read_bytes())
    meta=dict(grid,status='PREPARED',bounded_preparation=True,native_geometry=native,
              source=volume.identity,roi_lo_zyx=lo.tolist(),roi_hi_zyx=hi.tolist(),stride=1,
              prediction='qual-v2',surface_method='lasagna',fibre_method='qual-v2-bounded-global',
              tiles_sha256=coarse['tiles_sha256'],calibration=coarse['calibration'],producer_identity=identity)
    atomic_json(run/'PREPARED.json',meta)
    return meta


def _curve_db(path):
    db=sqlite3.connect(path)
    db.execute('PRAGMA synchronous=FULL');db.execute('PRAGMA cache_size=-2048')
    db.executescript('CREATE TABLE IF NOT EXISTS done(id INTEGER PRIMARY KEY,digest TEXT);'
        'CREATE TABLE IF NOT EXISTS progress(id INTEGER PRIMARY KEY CHECK(id=1),next_id INTEGER);'
        'CREATE TABLE IF NOT EXISTS curves(id INTEGER PRIMARY KEY,tile TEXT,points BLOB,family TEXT,source_id INTEGER,commit_id INTEGER);'
        'CREATE INDEX IF NOT EXISTS curves_commit ON curves(commit_id,id);'
        'CREATE VIRTUAL TABLE IF NOT EXISTS curve_bounds USING rtree(id,min_x,max_x,min_y,max_y,min_z,max_z);')
    try:
        if (db.execute('PRAGMA integrity_check').fetchall()!=[('ok',)]
                or db.execute("SELECT rtreecheck('curve_bounds')").fetchone()!=('ok',)
                or db.execute('SELECT 1 FROM curve_bounds b LEFT JOIN curves c ON c.id=b.id WHERE c.id IS NULL LIMIT 1').fetchone()
                or db.execute('SELECT 1 FROM curves c LEFT JOIN done d ON d.id=c.commit_id WHERE d.id IS NULL LIMIT 1').fetchone()):
            raise ValueError('fitted curve inventory changed')
        next_id=1
        for key,sha in db.execute('SELECT id,digest FROM done ORDER BY id'):
            if key!=next_id or sha!=_curve_digest(db,key):raise ValueError('fitted commit inventory changed')
            next_id+=1
        progress=db.execute('SELECT next_id FROM progress WHERE id=1').fetchone()
        if progress!=(next_id,) and (progress is not None or next_id!=1):
            raise ValueError('fitted progress inventory changed')
    except BaseException:
        db.close();raise
    return db


def _curve_digest(db,key):
    sha=hashlib.sha256(str(key).encode())
    for row in db.execute('''SELECT c.id,c.points,c.family,c.source_id,b.min_x,b.max_x,b.min_y,b.max_y,b.min_z,b.max_z
        FROM curves c LEFT JOIN curve_bounds b ON b.id=c.id WHERE c.commit_id=? ORDER BY c.id''',(key,)):
        cid,blob,family,source,*box=row
        if None in box:raise ValueError('fitted spatial inventory changed')
        sha.update(blob);sha.update(json.dumps([cid,family,source,box],allow_nan=False).encode())
    return sha.hexdigest()


def _store_curves(db,root,key,pieces,family,source_id,quota,reserve):
    with db:
        previous=db.execute('SELECT next_id FROM progress WHERE id=1').fetchone()
        if key!=(previous[0] if previous else 1):raise ValueError('fitted commit inventory gap')
        size=db.execute('PRAGMA page_count').fetchone()[0]*db.execute('PRAGMA page_size').fetchone()[0]
        if 2*(size+65536)>quota or shutil.disk_usage(root).free<reserve+131072:
            raise OSError('fitted fibre quota/reserve exhausted; curve not committed')
        for points in pieces:
            if points.ndim!=2 or points.shape[1:]!=(3,) or len(points)<2 or not np.isfinite(points).all():
                raise ValueError('invalid fitted fibre geometry')
            buffer=io.BytesIO();np.save(buffer,points,allow_pickle=False);blob=buffer.getvalue()
            size=db.execute('PRAGMA page_count').fetchone()[0]*db.execute('PRAGMA page_size').fetchone()[0]
            allowance=len(blob)+65536
            if 2*(size+allowance)>quota or shutil.disk_usage(root).free<reserve+2*allowance:
                raise OSError('fitted fibre quota/reserve exhausted; curve not committed')
            cid=db.execute('INSERT INTO curves(tile,points,family,source_id,commit_id) VALUES (?,?,?,?,?)',
                           ('global',blob,family,source_id,key)).lastrowid
            box=[float(v) for p in zip(points.min(0),points.max(0)) for v in p]
            db.execute('INSERT INTO curve_bounds VALUES (?,?,?,?,?,?,?)',(cid,*box))
        db.execute('INSERT INTO done VALUES (?,?)',(key,_curve_digest(db,key)))
        db.execute('INSERT OR REPLACE INTO progress VALUES (1,?)',(key+1,))


def _trace_and_fit(volume,root,meta,config,stop):
    from .bounded_predictions import QualWindows
    from .bounded_qual import QualFields,PagedField,trace_seeds,finish_curves,sample_field
    from .bounded_journal import CurveJournal
    from .bounded_seeds import seeds
    model=QualV2(config.get('model_path'),config.get('device','cpu'),config.get('offline',False))
    reader=QualWindows(volume,model,invalid_value=config.get('invalid_value'))
    fields=QualFields(tuple(map(int,volume.shape)),reader.read)
    lo=np.asarray(meta['roi_lo_zyx'])//2;hi=np.asarray(meta['roi_hi_zyx'])//2
    class Domain:
        shape=tuple(map(int,hi-lo))
        def centrality(self,a,b,family):return fields.centrality(a+lo,b+lo,family)
    identity=dict(producer=meta['producer_identity'],seed_order='region1024-priority-centrality-zyx-v1',
                  roi_lo=meta['roi_lo_zyx'],roi_hi=meta['roi_hi_zyx'])
    quota=config.get('bounded_bytes',12<<30);reserve=config.get('reserve_bytes',0)
    with CurveJournal(root/'raw.sqlite',identity,quota_bytes=quota//4) as journal:
        families={family:fields.family(family) for family in (1,2)}
        if not (root/'TRACED.json').exists():
            for family in (1,2):
                source=seeds(Domain(),family,root,quota_bytes=quota//8,stop=stop)
                def global_seeds():
                    try:
                        for sequence,q,priority in source:
                            yield sequence,tuple(np.asarray(q)+lo),priority
                    finally:source.close()
                status=trace_seeds(global_seeds(),family,families[family],journal,stop=stop)
                if status=='PAUSED':return False
            atomic_json(root/'TRACED.json',dict(identity=identity,next_seed=[journal.next_seed(f) for f in (1,2)]))
        else:
            traced=json.loads((root/'TRACED.json').read_bytes())
            if traced!=dict(identity=identity,next_seed=[journal.next_seed(f) for f in (1,2)]):
                raise ValueError('global tracing completion changed')
        pooled=[PagedField(fields.shape,np.uint8,lambda a,b,i=i:fields.pooled(a,b)[i]) for i in range(2)]
        native=[PagedField(tuple(map(int,volume.shape)),np.float32,
                           lambda a,b,i=i:reader.read(a,b)[i]*255) for i in range(2)]
        class Native:
            def sample(self,xyz):return np.column_stack([sample_field(p,xyz) for p in native])
        db=_curve_db(root/'candidates.sqlite')
        try:
            for fid,family,blob in journal.db.execute('SELECT id,family,points FROM curves ORDER BY id'):
                if stop and stop():return False
                if db.execute('SELECT 1 FROM done WHERE id=?',(fid,)).fetchone():continue
                points=np.frombuffer(blob,dtype='<f4').reshape(-1,3)
                fitted,offsets=finish_curves([points],pooled,families[family]['presence'],family,Native())
                pieces=(fitted[a:b].astype('f8')*2+.5 for a,b in zip(offsets[:-1],offsets[1:]))
                _store_curves(db,root,fid,pieces,'VH'[family-1],fid,quota//8,reserve)
        finally:db.close()
    return True


def fibres(volume,run,config,stop=None):
    run=Path(run);root=run/'fibres';root.mkdir(exist_ok=True)
    meta=json.loads((run/'PREPARED.json').read_bytes())
    if np.any(np.asarray(meta['roi_lo_zyx'])%4):
        raise ValueError('bounded Qual ROI origin must preserve the global half-grid checkerboard (multiple of4)')
    bind(root/'INPUT.json',dict(producer=meta['producer_identity'],algorithm='paged-global-qual-v1'))
    marker=run/'FIBRES.json'
    if marker.exists():
        receipt=json.loads(marker.read_bytes())
        if any(file_sha(root/name)!=receipt[key] for name,key in [
                ('fibres.sqlite','sha256'),('raw.sqlite','raw_sha256'),('candidates.sqlite','candidates_sha256')]):
            raise ValueError('bounded fibre artifacts changed')
        return receipt
    try:
        if not _trace_and_fit(volume,root,meta,config,stop):return dict(status='PAUSED')
    finally:
        # Release all Qual readers/model references before the Lasagna support pass.
        gc.collect()
        import torch
        if torch.device(config.get('device','cpu')).type=='cuda':torch.cuda.empty_cache()
    source=sqlite3.connect((root/'candidates.sqlite').resolve().as_uri()+'?mode=ro',uri=True)
    db=_curve_db(root/'fibres.sqlite')
    try:
        with surface_maps(volume,run,config,np.zeros(3,dtype=int),np.asarray(volume.shape)) as (maps,_):
            for cid,blob,family,raw_id in source.execute('SELECT id,points,family,source_id FROM curves ORDER BY id'):
                if stop and stop():return dict(status='PAUSED')
                if db.execute('SELECT 1 FROM done WHERE id=?',(cid,)).fetchone():continue
                points=np.load(io.BytesIO(blob),allow_pickle=False)
                pieces=(p*volume.spacing_xyz+volume.origin_xyz for p in supported_pieces(points,maps))
                _store_curves(db,root,cid,pieces,family,raw_id,config.get('bounded_bytes',12<<30)//8,
                              config.get('reserve_bytes',0))
        count=db.execute('SELECT count(*) FROM curves').fetchone()[0]
    finally:source.close();db.close()
    receipt=dict(status='FIBRES_READY',method='qual-v2-bounded-global',spatial_index=True,curves=count,
                 sha256=file_sha(root/'fibres.sqlite'),raw_sha256=file_sha(root/'raw.sqlite'),
                 candidates_sha256=file_sha(root/'candidates.sqlite'),physical_acceptance=False)
    atomic_json(marker,receipt)
    return receipt
