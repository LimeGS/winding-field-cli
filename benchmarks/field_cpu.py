"""Small, frozen-input field-stage benchmark; never operates on production runs."""
import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import resource
import statistics
import subprocess
import sys
import time

import numpy as np


def summarize(records):
    if not records or len({r['input_sha256'] for r in records})!=1:
        raise ValueError('benchmark inputs differ')
    if len({r['field_sha256'] for r in records})!=1:
        raise ValueError('benchmark field results differ')
    groups={}
    for record in records:groups.setdefault(record['variant'],[]).append(record)
    for name,group in groups.items():
        for key in ('source_sha256','benchmark_sha256'):
            if any(key not in record for record in group):
                raise ValueError(f'benchmark provenance missing: {key} ({name})')
            if len({record[key] for record in group})!=1:
                raise ValueError(f'benchmark provenance differs within {name}: {key}')
        if any('config' not in record for record in group):
            raise ValueError(f'benchmark provenance missing: config ({name})')
        if len({json.dumps(record['config'],sort_keys=True) for record in group})!=1:
            raise ValueError(f'benchmark provenance differs within {name}: config')
    return dict(raw=records,median_seconds=statistics.median(r['wall_seconds'] for r in records),
                variants={name:dict(median_seconds=statistics.median(r['wall_seconds'] for r in group),
                                   minimum_seconds=min(r['wall_seconds'] for r in group),
                                   maximum_seconds=max(r['wall_seconds'] for r in group))
                          for name,group in groups.items()})


def paired_comparison(records,baseline,candidate):
    summarize(records)
    def timings(variant):
        values={}
        for record in records:
            if record.get('variant')!=variant:continue
            repetition=record.get('repetition');seconds=record.get('process_wall_seconds')
            if not isinstance(repetition,int) or isinstance(repetition,bool):
                raise ValueError(f'paired comparison requires integer repetition: {variant}')
            if not isinstance(seconds,(int,float)) or not math.isfinite(seconds) or seconds<=0:
                raise ValueError(f'paired comparison requires positive total time: {variant}')
            if repetition in values:
                raise ValueError(f'paired comparison duplicate repetition: {variant} {repetition}')
            values[repetition]=seconds
        if not values:raise ValueError(f'paired comparison missing variant: {variant}')
        return values
    base=timings(baseline);other=timings(candidate)
    if base.keys()!=other.keys():raise ValueError('paired comparison repetitions differ')
    ratios=[base[index]/other[index] for index in sorted(base)]
    return dict(ratios=ratios,median_ratio=statistics.median(ratios),
                wins=sum(ratio>1 for ratio in ratios),count=len(ratios))


def sweep_timings(commits,blocks):
    if len(commits)<2*blocks or len(commits)%blocks:
        raise ValueError('incomplete benchmark commit closures')
    closures=commits[blocks-1::blocks]
    return dict(initialization_seconds=closures[0],first_sweep_seconds=closures[1]-closures[0],
                later_sweep_seconds=[b-a for a,b in zip(closures[1:],closures[2:])])


def fixture(root):
    from winding_field.artifacts import Tiles
    from winding_field.util import atomic_json,file_sha
    import sqlite3
    root.mkdir()
    shape=(6,6,32);spacing=[2.,3.,4.];origin=[10.,20.,30.]
    maps=Tiles(root/'maps',shape=shape,chunk=8,identity={'test':'field-cpu-benchmark-v1'})
    for key in maps.keys():
        lo,hi=maps.box(key);local=tuple(hi-lo)
        normal=np.zeros(local+(3,),np.float32);normal[...,0]=1
        density=np.broadcast_to(np.where(np.arange(lo[2],hi[2])<12,1/8,1/16),local).astype(np.float32).copy()
        maps.write(key,dict(normals=normal,density=density,support=np.ones(local,bool)))
    meta=dict(shape_zyx=list(shape),spacing_xyz=spacing,origin_xyz=origin,frame='synthetic',
              tiles_sha256=maps.identity(),prediction='qual-v2')
    maps.close();atomic_json(root/'PREPARED.json',meta)
    (root/'fibres').mkdir()
    fingerprint=hashlib.sha256(json.dumps(meta,sort_keys=True).encode())
    with sqlite3.connect(root/'fibres/fibres.sqlite') as db:
        db.executescript('CREATE TABLE curves(id INTEGER PRIMARY KEY,tile TEXT,points BLOB,family TEXT);'
                        'CREATE VIRTUAL TABLE curve_bounds USING rtree(id,min_x,max_x,min_y,max_y,min_z,max_z);')
        for i in range(64):
            points=np.column_stack((np.full(256,11.+i%31*2),
                                    np.linspace(20.1,34.9,256),np.full(256,40.)))
            fingerprint.update(points.tobytes());buffer=io.BytesIO();np.save(buffer,points,allow_pickle=False)
            db.execute('INSERT INTO curves VALUES (?,?,?,?)',(i,'qual',buffer.getvalue(),'V'))
            bounds=tuple(v for pair in zip(points.min(0),points.max(0)) for v in pair)
            db.execute('INSERT INTO curve_bounds VALUES (?,?,?,?,?,?,?)',(i,*bounds))
    atomic_json(root/'FIBRES.json',dict(sha256=file_sha(root/'fibres/fibres.sqlite'),spatial_index=True))
    return fingerprint.hexdigest()


def run_once(args):
    sys.path.insert(0,str(args.source_root/'src'))
    import torch
    from winding_field import field
    torch.set_num_threads(1)
    root=args.output;root.mkdir(parents=True)
    inputs=fixture(root/'run')
    config=dict(sweeps=4,field_chunk=8,threads=1,prepared_cache_bytes=0,
                constraint_cache_bytes=0 if args.variant=='base' else args.cache_bytes)
    if args.variant=='parallel':
        config.update(field_cpu_workers=args.workers,field_cpu_buffer_bytes=args.buffer_bytes)
    commits=[];commit=field.BlockStore.commit
    def timed_commit(store,*args,**kwargs):
        result=commit(store,*args,**kwargs)
        commits.append(time.perf_counter()-started)
        return result
    field.BlockStore.commit=timed_commit
    started=time.perf_counter()
    try:receipt=field.solve(root/'run',config)
    finally:field.BlockStore.commit=commit
    field_seconds=time.perf_counter()-started
    assert receipt['status']=='FIELD_READY'
    with field.open_field(root/'run') as source:
        values,valid=source.read([0]*3,[6,6,32])
    code=hashlib.sha256()
    for path in sorted((args.source_root/'src').rglob('*.py')):
        code.update(str(path.relative_to(args.source_root)).encode());code.update(path.read_bytes())
    result=dict(variant=args.variant,input_sha256=inputs,
                field_sha256=hashlib.sha256(values.tobytes()+valid.tobytes()).hexdigest(),
                field_seconds=field_seconds,wall_seconds=field_seconds,
                source_sha256=code.hexdigest(),
                benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),config=config,
                cache=receipt.get('constraint_cache_stats'),pool=receipt.get('field_cpu_stats'),
                parent_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),
                sweep_timings=sweep_timings(commits,4),
                source_root=str(args.source_root),scope='synthetic 4-sweep field stage; fixture creation excluded')
    (root/'result.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant',choices=['base','cache','vector','parallel'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source-root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--against-root',type=Path)
    parser.add_argument('--against-variant',choices=['base','cache','vector','parallel'],default='base')
    parser.add_argument('--repetitions',type=int,default=5)
    parser.add_argument('--cache-bytes',type=int,default=128<<20)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--buffer-bytes',type=int,default=128<<20)
    parser.add_argument('--once',action='store_true')
    args=parser.parse_args()
    if args.repetitions<1:parser.error('positive repetitions required')
    if args.once:return run_once(args)
    benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.mkdir(parents=True)
    variants=[(args.variant,args.source_root)]
    if args.against_root:variants.insert(0,(args.against_variant,args.against_root))
    records=[]
    for index in range(args.repetitions):
        for variant,source in variants if index%2==0 else reversed(variants):
            output=args.output/f'{index}-{variant}'
            command=[sys.executable,str(Path(__file__).resolve()),'--once','--variant',variant,
                     '--source-root',str(source.resolve()),'--output',str(output.resolve()),
                     '--cache-bytes',str(args.cache_bytes),'--workers',str(args.workers),
                     '--buffer-bytes',str(args.buffer_bytes)]
            started=time.perf_counter()
            subprocess.run(command,check=True,env=dict(os.environ,OMP_NUM_THREADS='1',
                           MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
            record=json.loads((output/'result.json').read_text())
            record.update(process_wall_seconds=time.perf_counter()-started,
                          benchmark_sha256=benchmark_sha256,repetition=index)
            records.append(record)
    result=summarize(records)
    (args.output/'summary.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(result['variants'],sort_keys=True))


if __name__=='__main__':main()
