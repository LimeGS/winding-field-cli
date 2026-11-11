#!/usr/bin/env python3
"""Compare frozen/current AFV importers in separate, new output directories."""
import argparse
import cProfile
from contextlib import closing
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import pstats
import resource
import shutil
import sqlite3
import statistics
import subprocess
import sys
import time

import numpy as np


def logical_sha(root, receipt):
    """Stream canonical logical rows; never compare SQLite page layouts."""
    h=hashlib.sha256()
    h.update(json.dumps({k:v for k,v in receipt.items() if k!='sha256'},sort_keys=True).encode())
    with closing(sqlite3.connect(root/'fibres/fibres.sqlite')) as db:
        for table,order in [('curves','id'),('curve_bounds','id'),('done','fid,block')]:
            h.update(table.encode())
            for row in db.execute(f'SELECT * FROM {table} ORDER BY {order}'):
                for value in row:
                    raw=value if isinstance(value,bytes) else json.dumps(value,allow_nan=False).encode()
                    h.update(b'b' if isinstance(value,bytes) else b'j')
                    h.update(len(raw).to_bytes(8,'big'));h.update(raw)
    return h.hexdigest()


def generated_input(root, blocks, retained_every):
    from avf_spline_generator.afv import export_afv
    from winding_field.artifacts import Tiles
    from winding_field.util import atomic_json
    root.mkdir()
    shape=(8,8,24)
    maps=Tiles(root/'maps',shape=shape,chunk=8,identity={'fixture':'afv-benchmark-v1'})
    for key in maps.keys():
        lo,hi=maps.box(key);support=np.ones(tuple(hi-lo),bool)
        support[...,np.arange(lo[2],hi[2])==12]=False
        maps.write(key,dict(support=support))
    meta=dict(shape_zyx=list(shape),spacing_xyz=[9.]*3,origin_xyz=[100.,200.,300.],
        frame='afv-benchmark',tiles_sha256=maps.identity(),prediction='qual-v2',
        source=dict(source='afv-benchmark',level=0,shape=[8,8,64],spacing_xyz=[9.]*3,origin_xyz=[100.,200.,300.]))
    maps.close();atomic_json(root/'PREPARED.json',meta)
    def fibres():
        for i in range(blocks):
            points=np.array([[1.,2.,3.],[20.,2.,3.]]) if retained_every and i%retained_every==0 else np.array([[30.,2.,3.],[31.,2.,3.]])
            yield i%2+1,(points-.5)/2
    export_afv(fibres(),root/'input.afv',voxel_um=9.,origin_xyz=[0]*3,volume_shape=(8,8,64))
    atomic_json(root/'geometry.json',dict(source='afv-benchmark',origin_xyz_um=meta['origin_xyz'],spacing_xyz_um=meta['spacing_xyz']))
    return dict(fibres=str(root/'input.afv'),fibres_geometry=str(root/'geometry.json'),reserve_bytes=0)


def worker(module_path, run_root, config_path, profile=False):
    from winding_field.util import atomic_json,file_sha
    spec=importlib.util.spec_from_file_location('winding_field._afv_benchmark',module_path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    config=json.loads(config_path.read_bytes())
    connect=sqlite3.connect
    metrics=dict(commits=0,commit_seconds=0.)
    class Timed(sqlite3.Connection):
        def commit(self):
            writing=self.in_transaction;start=time.perf_counter()
            result=super().commit()
            if writing:
                metrics['commits']+=1;metrics['commit_seconds']+=time.perf_counter()-start
            return result
    def opened(database,*args,**kwargs):
        if str(database).endswith('/fibres.sqlite'):
            kwargs['factory']=Timed
            db=connect(database,*args,**kwargs)
            metrics.update(synchronous=db.execute('PRAGMA synchronous').fetchone()[0],
                           journal_mode=db.execute('PRAGMA journal_mode').fetchone()[0])
            return db
        return connect(database,*args,**kwargs)
    profiler=cProfile.Profile() if profile else None
    sqlite3.connect=opened
    start=time.perf_counter()
    try:
        if profiler:profiler.enable()
        receipt=module.import_fibres(run_root,config)
    finally:
        if profiler:profiler.disable()
        elapsed=time.perf_counter()-start
        sqlite3.connect=connect
    if receipt['status']!='FIBRES_READY':raise ValueError('benchmark import did not complete')
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
    blocks=receipt['identity']['input']['block_count']
    metrics.update(pid=os.getpid(),seconds=elapsed,blocks=blocks,blocks_per_second=blocks/elapsed,
        non_commit_seconds=elapsed-metrics['commit_seconds'],peak_rss_bytes=peak,
        curves_retained=receipt['count'],fibres_retained=receipt['imported_sources'],
        input_sha256=receipt['identity']['input']['sha256'],importer_sha256=file_sha(module_path),
        logical_sha256=logical_sha(run_root,receipt))
    if profiler:
        profiler.dump_stats(run_root/'PROFILE.pstats')
        stats=pstats.Stats(profiler)
        metrics['top_cumulative']=[dict(function=f'{Path(key[0]).name}:{key[1]}:{key[2]}',calls=value[1],
            self_seconds=value[2],cumulative_seconds=value[3])
            for key,value in sorted(stats.stats.items(),key=lambda item:item[1][3],reverse=True)[:30]]
    atomic_json(run_root/'METRICS.json',metrics)


def run(output, baseline_source, *, blocks=5000, retained_every=20, repeats=3,
        profile=False, prepared=None, config_path=None):
    from winding_field import afv
    from winding_field.util import atomic_json,file_sha
    output=Path(output).expanduser().resolve();baseline_source=Path(baseline_source).resolve()
    if not baseline_source.is_file():raise FileNotFoundError('baseline source missing')
    if (type(blocks) is not int or blocks<1 or type(retained_every) is not int or retained_every<0
            or type(repeats) is not int or not 1<=repeats<=20):raise ValueError('invalid benchmark sizes')
    if (prepared is None)!=(config_path is None):raise ValueError('supply both --prepared and --config')
    if prepared:
        prepared=Path(prepared).resolve();config_path=Path(config_path).resolve()
        if output==prepared or prepared in output.parents or output in prepared.parents:
            raise ValueError('benchmark output must be separate from prepared input')
        config=json.loads(config_path.read_bytes())
        for key in ('fibres','fibres_geometry'):
            if config.get(key):config[key]=str((config_path.parent/config[key]).resolve())
        for key in ('fibres','fibres_geometry'):
            if config.get(key) and output in Path(config[key]).parents:
                raise ValueError('benchmark output overlaps an input')
    output.mkdir(parents=True,exist_ok=False)
    if prepared is None:
        prepared=output/'input';config=generated_input(prepared,blocks,retained_every)
    effective_config=output/'config.json';atomic_json(effective_config,config)
    frozen=output/'baseline-afv.py';shutil.copyfile(baseline_source,frozen)
    current=output/'candidate-afv.py';shutil.copyfile(afv.__file__,current)
    measurements=[]
    def invoke(variant,sequence,profile=False):
        root=output/f'{variant}-{sequence}';root.mkdir()
        shutil.copyfile(prepared/'PREPARED.json',root/'PREPARED.json')
        shutil.copytree(prepared/'maps',root/'maps')
        command=[sys.executable,str(Path(__file__).resolve()),'--worker',str(current if variant=='candidate' else frozen),
                 '--output',str(root),'--config',str(effective_config)]
        if profile:command.append('--profile')
        child=subprocess.run(command,capture_output=True,text=True)
        (root/'worker.log').write_text(child.stdout+child.stderr)
        if child.returncode:raise RuntimeError('benchmark worker failed; see '+str(root/'worker.log'))
        return dict(json.loads((root/'METRICS.json').read_bytes()),variant=variant,repeat=sequence)
    for repeat in range(repeats):
        for variant in (('baseline','candidate') if repeat%2==0 else ('candidate','baseline')):
            measurements.append(invoke(variant,repeat))
    medians={name:statistics.median(r['seconds'] for r in measurements if r['variant']==name)
             for name in ('baseline','candidate')}
    report=dict(schema='afv-import-benchmark-v1',input=dict(blocks=measurements[0]['blocks'],
            sha256=measurements[0]['input_sha256'],prepared_sha256=file_sha(prepared/'PREPARED.json'),
            config_sha256=file_sha(effective_config),generated=prepared==output/'input'),
        measurements=measurements,median_seconds=medians,speedup=medians['baseline']/medians['candidate'],
        logical_results_equal=len({r['logical_sha256'] for r in measurements})==1,
        runtime=dict(python=platform.python_version(),numpy=np.__version__,sqlite=sqlite3.sqlite_version,
                     platform=platform.platform(),filesystem_block_bytes=os.statvfs(output).f_bsize),
        cache_policy='Alternating order, shared warm OS cache; no cache drops or durability changes.')
    if profile:report['profile']=invoke('candidate','profile',profile=True)
    atomic_json(output/'RESULT.json',report)
    if not report['logical_results_equal']:raise ValueError('baseline/candidate logical results differ')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--baseline-source',type=Path)
    parser.add_argument('--blocks',type=int,default=5000)
    parser.add_argument('--retained-every',type=int,default=20,help='generated input only; 0 excludes all blocks')
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--profile',action='store_true',help='separate extra candidate profile, excluded from timing medians')
    parser.add_argument('--prepared',type=Path,help='immutable prepared input; copied, never modified')
    parser.add_argument('--config',type=Path,help='AFV import config; required with --prepared')
    parser.add_argument('--worker',type=Path,help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.worker:
        worker(args.worker,args.output,args.config,args.profile)
    else:
        if args.baseline_source is None:parser.error('--baseline-source required')
        report=run(args.output,args.baseline_source,blocks=args.blocks,retained_every=args.retained_every,
                   repeats=args.repeats,profile=args.profile,prepared=args.prepared,config_path=args.config)
        print(json.dumps(report,indent=2,allow_nan=False))


if __name__=='__main__':main()
