"""Verified PNG tile continuation in a fresh namespace; original outputs stay intact."""
import json
from pathlib import Path
import re
import shutil

from .util import atomic_json,file_sha


def copy_verified_file(source,destination,*,guard,reserve_bytes):
    source=Path(source);destination=Path(destination)
    if shutil.disk_usage(destination.parent).free<reserve_bytes+source.stat().st_size:
        raise RuntimeError('copy exceeds disk reserve')
    with source.open('rb') as src,destination.open('wb') as dst:
        while data:=src.read(1<<20):guard();dst.write(data)


def map_array_bytes(path):
    import math,zipfile
    from numpy.lib import format
    size=0
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            with archive.open(name) as stream:
                version=format.read_magic(stream)
                if version==(1,0):shape,_,dtype=format.read_array_header_1_0(stream)
                elif version==(2,0):shape,_,dtype=format.read_array_header_2_0(stream)
                else:raise ValueError('unsupported map array header')
                if dtype.hasobject:raise ValueError('object map arrays are not supported')
                size+=math.prod(shape)*dtype.itemsize
    return size


class TileReuse:
    def __init__(self,level,*,identity_sha256,map_sha256,guard=lambda:None,reserve_bytes=0):
        if Path(level).is_symlink():raise ValueError('symlinked source level')
        self.level=Path(level).resolve();self.identity_sha256=identity_sha256;self.map_sha256=map_sha256
        self.guard=guard;self.reserve_bytes=reserve_bytes
        self.validate()
        self.receipts={p.parent.name:file_sha(p) for p in sorted((self.level/'render').glob('*/TILE.json'))}
        if any(not re.fullmatch(r'tile-\d{9}-\d{9}',name) for name in self.receipts):raise ValueError('invalid tile directory')

    def validate(self):
        if (self.level/'render').is_symlink():raise ValueError('symlinked render directory')
        for path,expected in ((self.level.parent/'IDENTITY.json',self.identity_sha256),(self.level/'map.npz',self.map_sha256)):
            if path.is_symlink() or file_sha(path)!=expected:raise ValueError('tile reuse identity/map changed')

    def get(self,name,canvas,destination):
        if name not in self.receipts:return None
        source=self.level/'render'/name;receipt=source/'TILE.json';expected=self.receipts[name]
        if (self.level/'render').is_symlink() or source.is_symlink() or receipt.is_symlink() or file_sha(receipt)!=expected:raise ValueError('tile receipt changed')
        record=json.loads(receipt.read_text())
        if record['directory']!=name or record['canvas']!=canvas or record['rendering']['canvas']!=canvas:
            raise ValueError('tile canvas differs')
        files=record['files']
        if not {'image.png','mask.png','mapping.npz'}<=files.keys() or record.get('tifxyz'):
            raise ValueError('PNG continuation requires complete PNG/mapping tiles')
        for relative,sha in files.items():
            self.guard()
            path=Path(relative)
            if path.is_absolute() or '..' in path.parts or len(path.parts)!=1 or relative=='TILE.json':raise ValueError('unsafe tile path')
            target=source/path
            if target.is_symlink() or not target.is_file() or file_sha(target)!=sha:raise ValueError('tile payload changed: '+relative)
        destination=Path(destination);destination.mkdir()
        for relative,sha in files.items():
            copy_verified_file(source/relative,destination/relative,guard=self.guard,reserve_bytes=self.reserve_bytes)
            if file_sha(destination/relative)!=sha:raise ValueError('copied tile payload differs')
        provenance='source-tile-'+expected+'.json'
        copy_verified_file(receipt,destination/provenance,guard=self.guard,reserve_bytes=self.reserve_bytes)
        if file_sha(destination/provenance)!=expected:raise ValueError('copied tile receipt differs')
        record['files']=dict(files,**{provenance:expected})
        record['reused_from']=dict(level=str(self.level),identity_sha256=self.identity_sha256,
            map_sha256=self.map_sha256,receipt_sha256=expected,receipt_file=provenance)
        return record


def continue_export(source,mesh,output,*,identity_sha256,overrides,stop=lambda:False):
    import sqlite3
    import numpy as np
    from importlib.metadata import version
    from . import patch_pipeline as p
    from .patch_mapping import PatchMapping
    from .native_patches import ResourceGuard
    source=Path(source).resolve();mesh=Path(mesh).resolve();output=Path(output).resolve()
    for original in (source,mesh):
        if output==original or output in original.parents or original in output.parents:raise ValueError('continuation requires disjoint output')
    if set(overrides)-{'cache_dir','cache_mib','cache_gib','memory_gib','raster_workers','process_tree_gib','disk_reserve_gib'}:
        raise ValueError('continuation cannot change geometric settings')
    if file_sha(source/'IDENTITY.json')!=identity_sha256:raise ValueError('source identity differs')
    previous=json.loads((source/'IDENTITY.json').read_text())
    if previous['export_tifxyz']:raise ValueError('continuation supports PNG only')
    if any(version(name)!=value for name,value in previous['runtime'].items()):raise ValueError('source numerical runtime differs')
    cfg=p.settings(dict(previous['settings'],**overrides));level=source/'level-0000'
    resources=ResourceGuard(output.parent,min(cfg['process_tree_gib'],cfg['memory_gib']),cfg['disk_reserve_gib'])
    reserve_bytes=int(cfg['disk_reserve_gib']*2**30)
    def guard():
        if stop():raise InterruptedError('continuation stopped; completed tiles retained')
        resources()
    guard()
    if sorted(x.name for x in source.glob('level-*'))!=['level-0000']:raise ValueError('continuation requires one saved level')
    mapping_record=json.loads((level/'MAPPING.json').read_text())
    reuse=TileReuse(level,identity_sha256=identity_sha256,map_sha256=mapping_record['map_sha256'],guard=guard,reserve_bytes=reserve_bytes)
    array_bytes=map_array_bytes(level/'map.npz')
    with p.open_mesh(mesh) as (reader,semantics),PatchMapping(level) as saved:
        if reader.manifest_sha!=previous['mesh_sha256'] or semantics!=previous['source']:
            raise ValueError('source mesh identity differs')
        if mapping_record['source_mesh_manifest_sha256']!=reader.manifest_sha or mapping_record['source_database_sha256']!=reader.manifest['database_sha256']:
            raise ValueError('source mapping provenance differs')
        face_count=reader.db.execute('SELECT COUNT(*) FROM faces WHERE level=0').fetchone()[0]
        if 2*(array_bytes+face_count*8)+p.raster_memory(cfg)>=cfg['memory_gib']*2**30:
            raise ValueError('continuation map exceeds memory budget before loading')
        guard();mapped={key:saved.arrays[key] for key in ('world','faces','uv','display')}
        ids=np.fromiter((r[0] for r in reader.db.execute('SELECT id FROM faces WHERE level=0 ORDER BY id')),dtype=np.int64)
        source_ids=saved.arrays['source_face_ids'];rows=np.searchsorted(ids,source_ids)
        if (rows>=len(ids)).any() or not np.array_equal(ids[rows],source_ids):raise ValueError('unknown original source face ID')
        mapped['source_face_rows']=rows
        geometry_bytes=sum(a.nbytes for a in mapped.values())+ids.nbytes+source_ids.nbytes
        if 2*geometry_bytes+p.raster_memory(cfg)>=cfg['memory_gib']*2**30:raise ValueError('continuation geometry exceeds memory budget')
        del ids,source_ids
        report_path=level/'MAP-REPORT.json'
        mapped['report']=json.loads(report_path.read_text()) if report_path.exists() else json.loads((level/'LEVEL.json').read_text())['mapping']
        source_geometry=reader.metadata
    output.mkdir(parents=True,exist_ok=False);products=output/'patches';target=products/'level-0000';target.mkdir(parents=True)
    volume=None
    try:
        for name in ('map.npz','keys.sqlite','MAPPING.json'):
            copy_verified_file(level/name,target/name,guard=guard,reserve_bytes=reserve_bytes)
        for name,key in (('map.npz','map_sha256'),('keys.sqlite','keys_sha256')):
            if file_sha(target/name)!=mapping_record[key]:raise ValueError('copied mapping differs')
        atomic_json(target/'MAP-REPORT.json',mapped['report'])
        copy_verified_file(source/'IDENTITY.json',products/'SOURCE-IDENTITY.json',guard=guard,reserve_bytes=reserve_bytes)
        if file_sha(products/'SOURCE-IDENTITY.json')!=identity_sha256:raise ValueError('copied identity differs')
        ledger=products/'SOURCE-CT.sqlite'
        with sqlite3.connect((source.parent/'ct-chunks.sqlite').as_uri()+'?mode=ro',uri=True) as old:
            size=old.execute('PRAGMA page_count').fetchone()[0]*old.execute('PRAGMA page_size').fetchone()[0]
            if shutil.disk_usage(products).free<reserve_bytes+size:raise RuntimeError('CT ledger exceeds disk reserve')
            with sqlite3.connect(ledger) as snapshot:old.backup(snapshot,pages=128,progress=lambda *_:guard())
        volume=p.PatchCT(previous['ct']['source'],output,cfg,voxel_um=previous['ct']['spacing_xyz'][0],level=previous['ct']['level'])
        for key in ('source','metadata_sha256','origin_xyz','spacing_xyz','shape','level'):
            if volume.identity[key]!=previous['ct'][key]:raise ValueError('CT identity differs: '+key)
        if 2*geometry_bytes+p.raster_memory(cfg)+p.ct_memory(volume.chunks,volume.dtype,cfg)>=cfg['memory_gib']*2**30:
            raise ValueError('continuation CT and geometry exceed memory budget')
        with sqlite3.connect(ledger) as snapshot:
            cursor=snapshot.execute('SELECT key,sha FROM chunks')
            while batch:=cursor.fetchmany(4096):
                guard();volume.db.executemany('INSERT INTO chunks VALUES (?,?)',batch)
        volume.db.commit()
        identity=dict(previous,settings=cfg,ct=volume.identity,implementation_sha256=file_sha(Path(p.__file__)))
        identity['kernel_sha256']=dict(previous['kernel_sha256'])
        for name in identity['kernel_sha256']:identity['kernel_sha256'][name]=file_sha(Path(__file__).with_name(name+'.py'))
        identity['kernel_sha256']['patch_continue']=file_sha(__file__)
        identity['continuation']=dict(source=str(source),identity_sha256=identity_sha256,map_sha256=reuse.map_sha256,
            source_ct_ledger_sha256=file_sha(ledger),tile_receipts=reuse.receipts,
            policy='copied verified completed tiles; original receipt preserved; only missing tiles rendered')
        identity['memory_admission']=dict(geometry_reserve_bytes=2*geometry_bytes,raster_cache_runtime_bytes=p.raster_memory(cfg),
            ct_chunk_workspace_bytes=p.ct_memory(volume.chunks,volume.dtype,cfg))
        atomic_json(products/'IDENTITY.json',identity)
        rendered=p.render_map(mapped,target/'render',lambda xyz:volume.sample(xyz,cfg['output_frame']),cfg['output_frame'],
            pitch_um=cfg['pitch_um'],tile_size=cfg['tile_size'],raster_workers=cfg['raster_workers'],guard=guard,reuse=reuse)
        record=dict(slot=0,semantics=semantics,mapping=mapped['report'],rendering=rendered,source_geometry=source_geometry,
            output_frame=cfg['output_frame'],map_sha256=reuse.map_sha256)
        atomic_json(target/'LEVEL.json',record)
        volume.close();volume=None;reuse.validate();guard()
        files={}
        for path in sorted(products.rglob('*')):
            guard()
            if path.is_file():files[str(path.relative_to(products))]=file_sha(path)
        result=dict(status='PATCHES_READY_DIAGNOSTIC' if rendered['CT_observed_pixels'] else 'PATCHES_EMPTY_DIAGNOSTIC',
            identity=identity,levels=[record],physical_acceptance=False,
            files=files,
            raster_continuation=dict(reused=len(reuse.receipts),rendered=len(rendered['tiles'])-len(reuse.receipts),
                peak_process_tree_rss_bytes=resources.peak))
        guard();atomic_json(products/'PATCHES.json',result)
        result=dict(result,ct_ledger_sha256=file_sha(output/'ct-chunks.sqlite'))
        guard();atomic_json(output/'PATCHES.json',result)
        return result
    except BaseException as error:
        atomic_json(output/'FAILED.json',dict(error=str(error),source=str(source),identity_sha256=identity_sha256));raise
    finally:
        if volume is not None:volume.close()


def main():
    import argparse,signal,threading
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-export',type=Path,required=True)
    parser.add_argument('--mesh-bundle',type=Path,required=True)
    parser.add_argument('--source-identity-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True,help='resource-only overrides JSON')
    args=parser.parse_args();stopped=threading.Event()
    signal.signal(signal.SIGTERM,lambda *_:stopped.set())
    result=continue_export(args.source_export,args.mesh_bundle,args.output,identity_sha256=args.source_identity_sha256,
        overrides=json.loads(args.config.read_text()),stop=lambda:stopped.is_set() or (args.output/'STOP').exists())
    print(json.dumps(dict(status=result['status'],**result['raster_continuation'])),flush=True)


if __name__=='__main__':main()
