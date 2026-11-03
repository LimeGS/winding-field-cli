"""Bounded native end-to-end benchmark; historical ROI and whole-W scopes stay explicit."""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import signal
import sys
import time
import uuid

import numpy as np

from winding_field.native_pipeline import _supervise,render_mesh
from winding_field.util import atomic_json,digest,file_sha
from native_historical_source import Historical

CT='https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0175A/volumes/20250521115057-8.640um-1.2m-116keV-masked.zarr'


class Plane:
    def __init__(self,volume):
        self.shape=volume.shape; self.origin=np.array(volume.origin_xyz); self.spacing=np.array(volume.spacing_xyz)
        self.identity={'kind':'analytic-plane','geometry':dict(shape_zyx=list(self.shape),origin_xyz=self.origin.tolist(),
            spacing_xyz=self.spacing.tolist(),voxel_um=1.,sheet_step=1.,frame=digest(volume.identity))}

    def validate(self,hashes=False):
        pass

    def read(self,lo,hi):
        x=np.arange(lo[2],hi[2],dtype=float)-(self.shape[2]-1)/2-.25
        shape=tuple(np.array(hi)-lo)
        return np.broadcast_to(x,shape).copy(),np.ones(shape,bool)


def execute(request,out,token):
    from winding_field.surface_mesh import MeshReader
    from winding_field.native_mesh import extract_mesh
    from winding_field.io import Volume
    interrupted=False
    def handle(signum,frame):
        nonlocal interrupted
        interrupted=True
    for sig in (signal.SIGINT,signal.SIGTERM):
        signal.signal(sig,handle)
    stop=lambda:interrupted or (out/'STOP').exists()
    started=time.monotonic()
    with ExitStack() as stack:
        if request['case']=='synthetic':
            from winding_field.demo import synthetic
            ct=out/'synthetic.zarr'
            if not ct.exists():
                synthetic(ct)
            volume=Volume(ct,cache=out/'ct-cache'); stack.callback(volume.close)
            source=Plane(volume); level=0.; roi=None
            expected_faces=2*(source.shape[0]-1)*(source.shape[1]-1)
            bindings={'kind':'synthetic'}
        else:
            from bench_direct_tifxyz import NATIVE,CONFIG_SHA,open_historical,pinned_config
            cfg=pinned_config()
            if request['case']=='tile':
                path=NATIVE/'tiles'/f"tile-{request['tile']:05d}.json"
                tile=json.loads(path.read_bytes())
                if tile['config_sha256']!=CONFIG_SHA:
                    raise ValueError('historical tile binding differs')
                roi=dict(lo_zyx=tile['lo_zyx'],hi_zyx=[min(a+1,n) for a,n in zip(tile['hi_zyx'],cfg['geometry']['shape_zyx'])])
                expected_faces=sum(p['faces'] for p in tile['parts'] if p['level']==-4)
                bindings={'config_sha256':CONFIG_SHA,'tile_receipt_sha256':file_sha(path)}
            elif request['case']=='roi':
                roi=dict(lo_zyx=[448,576,1984],hi_zyx=[961,1089,2497]); expected_faces=792198
                bindings={'config_sha256':CONFIG_SHA}
            else:
                roi=dict(lo_zyx=[0,0,0],hi_zyx=cfg['geometry']['shape_zyx'])
                inventory=json.loads((NATIVE/'INVENTORY.json').read_bytes())
                if inventory['config_sha256']!=CONFIG_SHA:
                    raise ValueError('historical inventory binding differs')
                expected_faces=inventory['verified']['-4']['faces']
                bindings={'config_sha256':CONFIG_SHA,'inventory_sha256':file_sha(NATIVE/'INVENTORY.json')}
            source=Historical(stack.enter_context(open_historical(roi)),
                              proof_root=NATIVE if request['case']=='whole' else None); level=-4.
            volume=Volume(CT,voxel_um=8.64,cache=out/'ct-cache',cache_bytes=2<<30,
                          reserve_bytes=request['reserve_bytes']); stack.callback(volume.close)
        cache_before=volume.stats()['cache_bytes']
        atomic_json(out/'SOURCE.json',dict(identity=source.identity,ct=volume.identity,roi=roi,
            whole_level=request['case']=='whole',expected_faces=expected_faces,bindings=bindings,
            source_coverage_certified=False))
        phase=time.monotonic(); mesh=out/'meshes'
        reused_mesh=(mesh/'MESH.json').exists()
        reused_products=(out/'level-0000/LEVEL.json').exists()
        if not reused_mesh:
            result=extract_mesh(source,mesh,[level],tile=request['mesh_tile'],resume=(mesh/'RUN.json').exists(),
                                stop=stop,reserve_bytes=request['reserve_bytes'])
            if result['status']!='MESH_READY':
                return dict(status='PAUSED',reason='STOP',phase='mesh',mesh=result,elapsed_s=time.monotonic()-started)
        mesh_seconds=time.monotonic()-phase
        projection_request=out/'RENDER-REQUEST.json'
        if projection_request.exists():
            render_config=json.loads(projection_request.read_bytes())
        else:
            render_config=dict(coverage_target=request['coverage_target'],
                surface_time_limit_s=max(.001,request['seconds']-(time.monotonic()-started)),
                chart_faces=request['chart_faces'],reserve_bytes=request['reserve_bytes'],
                export_tifxyz=request['export_tifxyz'],render_pitch_um=float(min(source.spacing)))
            atomic_json(projection_request,render_config)
        with MeshReader(mesh) as reader:
            faces=sum(r['faces'] for r in reader.manifest['levels'])
            if faces!=expected_faces:
                raise ValueError(f'native face count differs: {faces} != {expected_faces}')
            result=render_mesh(reader,out/'level-0000',volume,render_config,stop=stop)
        return dict(result,case=request['case'],whole_level=request['case']=='whole',roi=roi,
            mesh_seconds=mesh_seconds,end_to_end_seconds=time.monotonic()-started,faces=faces,
            ct_cache_initial_bytes=cache_before,ct_io=volume.stats(),bindings=bindings,
            timing_scope='existing-products-verification' if reused_products else 'existing-mesh-to-render' if reused_mesh else 'computed-field-to-render')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--case',choices=['synthetic','tile','roi','whole'],default='synthetic')
    parser.add_argument('--tile',type=int,default=7084)
    parser.add_argument('--seconds',type=float,default=120.)
    parser.add_argument('--coverage-target',type=float,default=1.)
    parser.add_argument('--chart-faces',type=int,default=1024)
    parser.add_argument('--mesh-tile',type=int,default=128)
    parser.add_argument('--reserve-gib',type=float,default=0.)
    parser.add_argument('--export-tifxyz',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--worker',action='store_true'); parser.add_argument('--token')
    args=parser.parse_args(); out=args.output.absolute()
    if args.worker:
        request=json.loads((out/'BENCHMARK.json').read_bytes())
        try:
            result=execute(request,out,args.token)
        except Exception as error:
            import traceback
            traceback.print_exc()
            result=dict(status='FAILED',reason=f'{type(error).__name__}: {error}')
        atomic_json(out/'WORKER-RESULT.json',dict(token=args.token,worker_pid=os.getpid(),result=result))
        return
    from winding_field.workflow import implementation_sha
    request=dict(case=args.case,tile=args.tile,seconds=args.seconds,coverage_target=args.coverage_target,
        chart_faces=args.chart_faces,mesh_tile=args.mesh_tile,reserve_bytes=int(args.reserve_gib*(1<<30)),
        export_tifxyz=args.export_tifxyz,implementation_sha256=implementation_sha(),benchmark_sha256=file_sha(__file__),
        source_adapter_sha256=file_sha(Path(__file__).with_name('native_historical_source.py')))
    if args.resume:
        if json.loads((out/'BENCHMARK.json').read_bytes())!=request:
            raise ValueError('benchmark request changed')
    else:
        out.mkdir()
        atomic_json(out/'BENCHMARK.json',request)
    token=uuid.uuid4().hex
    command=[sys.executable,'-B',__file__,'--worker','--output',str(out),'--token',token]
    prior=0.
    if (out/'level-0000/TIME.json').exists():
        prior=json.loads((out/'level-0000/TIME.json').read_bytes())['elapsed_s']
    result=_supervise(command,out,args.seconds,lambda:(out/'STOP').exists(),token,prior)
    result.update(case=args.case,whole_level=args.case=='whole',request=request)
    if (out/'meshes/STATUS.json').is_file():
        result['mesh_progress']=json.loads((out/'meshes/STATUS.json').read_bytes())
    atomic_json(out/'RESULT.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
