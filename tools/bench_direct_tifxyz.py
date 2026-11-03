"""Pinned historical-source benchmark, separate from the production CLI.

prepare fixes the four ROIs before observing outcomes. probe-guides checks a
necessary geometric precondition only; it never claims GO or publishes TIFXYZ.
"""
import argparse
import hashlib
import heapq
import json
import itertools
import math
import os
from pathlib import Path
import sys
import time

NATIVE=Path('/mnt/bulk/vesuvius-175A-20261024/native-meshes-20261026/job-01')
CONFIG_SHA='4fdac93398fa748a51d4b43d0e27c89b041ef41ee880dff44df661b2d6594d81'


def select_rois(rows,shape,level):
    rows=sorted((r for r in rows if any(p['level']==level and p['faces']>0 for p in r['parts'])),key=lambda r:r['tile'])
    if not rows or min(shape)<513:raise ValueError('no eligible source tiles/ROI')
    result=[];seen=set()
    for percent in (10,35,65,90):
        row=rows[percent*(len(rows)-1)//100]
        centre=[(a+b)//2 for a,b in zip(row['lo_zyx'],row['hi_zyx'])]
        lo=tuple(min(max(c-256,0),n-513) for c,n in zip(centre,shape))
        if lo not in seen:
            result.append(dict(percentile=percent,tile=row['tile'],lo_zyx=list(lo),hi_zyx=[x+513 for x in lo]))
            seen.add(lo)
    return result


def record(path):
    with Path(path).open('rb') as stream:
        before=os.fstat(stream.fileno());sha=hashlib.file_digest(stream,'sha256').hexdigest();after=os.fstat(stream.fileno())
    if (before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(after.st_size,after.st_mtime_ns,after.st_ctime_ns):raise ValueError('benchmark input changed')
    return dict(bytes=before.st_size,sha256=sha)


def pinned_config():
    path=NATIVE/'CONFIG.json'
    if record(path)['sha256']!=CONFIG_SHA:raise ValueError('native config binding differs')
    cfg=json.loads(path.read_text())
    if cfg['alpha']!=1.5843640137880175 or cfg['scope']!='FULL_Q1_NATIVE':raise ValueError('native source scope differs')
    return cfg


def prepare(out):
    from winding_field.util import atomic_json
    cfg=pinned_config();extraction=json.loads((NATIVE/'EXTRACTION.json').read_text())
    chain=hashlib.sha256();rows=[]
    for i in range(extraction['tiles']):
        raw=(NATIVE/'tiles'/f'tile-{i:05d}.json').read_bytes();chain.update(raw);row=json.loads(raw)
        if row['config_sha256']!=CONFIG_SHA or row['tile']!=i:raise ValueError('tile binding differs')
        parts=[{'level':p['level'],'faces':p['faces']} for p in row['parts'] if p['level']==-4 and p['faces']>0]
        if parts:rows.append({**{k:row[k] for k in ('tile','lo_zyx','hi_zyx')},'parts':parts})
    if chain.hexdigest()!=extraction['tiles_sha256']:raise ValueError('tile receipt chain differs')
    result=dict(status='PILOT_FIXED_NOT_EXECUTED',config_sha256=CONFIG_SHA,level=-4,
                source_geometry=cfg['geometry'],extraction=record(NATIVE/'EXTRACTION.json'),
                tile_receipts_sha256=chain.hexdigest(),rois=select_rois(rows,cfg['geometry']['shape_zyx'],-4),
                pitch_um=17.28,tolerance_um=17.28,max_pixels=4194304,physical_acceptance=False)
    if out.exists():raise FileExistsError(out)
    atomic_json(out,result);return result


def open_historical(roi):
    from winding_field.tifxyz_field import PhysicalField
    cfg=pinned_config()
    for path,expected in cfg['pinned'].items():
        if record(path)!=expected:raise ValueError('frozen source/runtime differs: '+path)
    sys.path[:0]=cfg['import_paths']
    from surface_sources import FieldSource
    from global_detail import CoarseDetailSource
    base=FieldSource(cfg['native'],cfg['geometry'],cache_bytes=128<<20)
    try:delta=FieldSource(cfg['delta'],cfg['coarse_geometry'])
    except BaseException:base.close();raise
    source=CoarseDetailSource(base,delta,cfg['alpha'])
    try:return PhysicalField(source,roi={k:roi[k] for k in ('lo_zyx','hi_zyx')})
    except BaseException:source.close();raise


def guard(out):
    import resource
    import shutil
    from winding_field.tifxyz_runtime import memory_report
    from winding_field.tifxyz_contract import Blocked
    m=memory_report()
    if not m['known'] or m['effective_headroom_bytes']<5*(1<<30):raise Blocked('RAM_HEADROOM')
    if shutil.disk_usage(out).free<(12<<30)+(512<<20):raise Blocked('DISK_HEADROOM')
    resource.setrlimit(resource.RLIMIT_AS,((3<<30)-(64<<20),resource.getrlimit(resource.RLIMIT_AS)[1]))


def probe(pilot_path,out,candidate):
    import resource
    import numpy as np
    from winding_field.tifxyz_contract import Blocked
    from winding_field.tifxyz_field import build_coarse
    from winding_field.tifxyz_guide import build_guide,stride_schedule
    from winding_field.util import atomic_json,file_sha
    if out.exists():raise FileExistsError(out)
    guard(out.parent);pilot=json.loads(pilot_path.read_text())
    if pilot['config_sha256']!=CONFIG_SHA or pilot['level']!=-4:raise ValueError('pilot binding differs')
    out.mkdir();started=time.monotonic();results=[];roi=pilot['rois'][candidate]
    with open_historical(roi) as field:
        for stride in stride_schedule(field.shape):
            attempt=out/f'stride-{stride:03d}';begin=time.monotonic()
            atomic_json(out/'STATUS.json',dict(stage='BUILDING_GUIDE',candidate=candidate,stride=stride,pid=os.getpid()))
            try:
                coarse=build_coarse(field,stride,attempt)
                guide=build_guide(coarse,pilot['level'])
                artifacts={}
                for name in ('world','faces','uv','normals','source_vertices'):
                    path=attempt/(name+'.npy')
                    with path.open('xb') as stream:np.save(stream,guide[name],allow_pickle=False);stream.flush();os.fsync(stream.fileno())
                    artifacts[name]=record(path)
                result=dict(status='GUIDE_CANDIDATE_NOT_TIFXYZ',stride=stride,vertices=len(guide['world']),faces=len(guide['faces']),artifacts=artifacts)
            except (Blocked,ValueError) as error:
                result=dict(status='GUIDE_REJECTED',stride=stride,error=str(error))
            result['elapsed_s']=time.monotonic()-begin;results.append(result)
            atomic_json(out/'RESULT.json',dict(candidate=candidate,roi=roi,pilot_sha256=file_sha(pilot_path),results=results,
                elapsed_s=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                source_reads=field.stats,whole_level=False,physical_acceptance=False))
            print(json.dumps(result),flush=True)
    atomic_json(out/'STATUS.json',dict(stage='GUIDE_PROBE_FINISHED',candidate=candidate,pid=os.getpid()))


def conformance(kernel_path,out):
    import numpy as np
    import resource
    from winding_field.util import atomic_json
    if out.exists():raise FileExistsError(out)
    guard(out.parent);cfg=pinned_config();original=json.loads(kernel_path.read_text())
    grouped={}
    for row in original['rows']:grouped.setdefault(tuple(row['tile']),[]).append(row)
    if len(grouped)!=12 or len(original['rows'])!=127:raise ValueError('unexpected historical conformance set')
    started=time.monotonic();checked=[]
    with open_historical(dict(lo_zyx=[0,0,0],hi_zyx=cfg['geometry']['shape_zyx'])) as field:
        from surface_mesh import _tile_mesh
        from native_meshes import tile_mesh
        np.testing.assert_array_equal(field.origin_um,field.source.origin*8.64)
        np.testing.assert_array_equal(field.spacing_um,field.source.spacing*8.64)
        for tile,rows in grouped.items():
            low=np.array(tile)*128;high=np.minimum(low+129,field.shape)
            a,known_a=field.source.read(low,high);b,known_b=field.read(low.tolist(),high.tolist())
            if not np.array_equal(a,b) or not np.array_equal(known_a,known_b):raise ValueError('adapter changes native field/support')
            for row in rows:
                reference=_tile_mesh(a,known_a,low,float(row['level']),field.source.origin,field.source.spacing)
                actual=tile_mesh(b,known_b,low,float(row['level']),field.source.origin,field.source.spacing)
                if (reference[0]!=actual[0] or not np.array_equal(reference[1],actual[1])
                    or not np.array_equal(reference[2],actual[2]) or len(actual[2])!=row['faces']):raise ValueError('native conformance differs')
                checked.append(dict(tile=list(tile),level=row['level'],faces=row['faces'],exact=True))
            print(json.dumps(dict(tiles_checked=len({tuple(r['tile']) for r in checked}),cases_checked=len(checked))),flush=True)
        field.validate()
    result=dict(status='HISTORICAL_SOURCE_ADAPTER_VERIFIED',config_sha256=CONFIG_SHA,kernel_reference=record(kernel_path),
                rows=checked,tiles=12,cases=127,elapsed_s=time.monotonic()-started,
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,physical_acceptance=False)
    atomic_json(out,result)


def probe_roots(pilot_path,guides_root,conformance_path,out,candidate):
    import numpy as np
    import resource
    from types import SimpleNamespace
    from winding_field.tifxyz_contract import Blocked
    from winding_field.tifxyz_field import read_metadata
    from winding_field.tifxyz_project import make_grid,project_tile
    from winding_field.util import atomic_json,file_sha
    if out.exists():raise FileExistsError(out)
    guard(out.parent);pilot=json.loads(pilot_path.read_text());source_proof=json.loads(conformance_path.read_text())
    if source_proof['status']!='HISTORICAL_SOURCE_ADAPTER_VERIFIED' or source_proof['config_sha256']!=CONFIG_SHA:
        raise ValueError('verified historical adapter required')
    previous=json.loads((guides_root/'RESULT.json').read_text())
    if previous['pilot_sha256']!=file_sha(pilot_path) or previous['candidate']!=candidate:raise ValueError('guide belongs to another pilot')
    out.mkdir();started=time.monotonic();results=[]
    frame=dict(id='PHerc0175A/20250521115057',origin_xyz_um=[0,0,0],voxel_um=8.64,
               shape_zyx=[12748,9363,9363],source_identity_sha256=CONFIG_SHA)
    settings=SimpleNamespace(output_frame=frame)
    with open_historical(pilot['rois'][candidate]) as field:
        for row in previous['results']:
            stride=row['stride'];directory=guides_root/f'stride-{stride:03d}';begin=time.monotonic();done=0
            if row['status']!='GUIDE_CANDIDATE_NOT_TIFXYZ':continue
            guide={}
            for name,expected in row['artifacts'].items():
                path=directory/(name+'.npy')
                if record(path)!=expected:raise ValueError('guide bytes changed')
                guide[name]=np.load(path,allow_pickle=False)
            guide['stride']=stride
            result=dict(stride=stride,guide_source=str(directory))
            try:
                grid=make_grid(guide,pilot['pitch_um'],pilot['max_pixels'],level=pilot['level']);result['grid']=grid
                for ty in range((grid['shape_yx'][0]+255)//256):
                    for tx in range((grid['shape_yx'][1]+255)//256):
                        atomic_json(out/'STATUS.json',dict(stage='PROBING_NATIVE_ROOTS',candidate=candidate,stride=stride,tile_yx=[ty,tx],tiles_completed=done,pid=os.getpid()))
                        tile=project_tile(field,guide,grid,(ty,tx),settings)
                        done+=1
                        del tile
                result.update(status='ROOTS_CANDIDATE_NOT_AUDITED',tiles_completed=done)
            except Blocked as error:
                result.update(status='PROJECTION_REJECTED',code=error.code,detail=error.detail,tiles_completed=done)
            result['elapsed_s']=time.monotonic()-begin;results.append(result)
            atomic_json(out/'RESULT.json',dict(candidate=candidate,roi=pilot['rois'][candidate],results=results,
                pilot_sha256=file_sha(pilot_path),source_conformance=record(conformance_path),
                elapsed_s=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                source_reads=field.stats,whole_level=False,physical_acceptance=False))
            print(json.dumps(result,allow_nan=False),flush=True)
    atomic_json(out/'STATUS.json',dict(stage='ROOT_PROBE_FINISHED',candidate=candidate,pid=os.getpid()))


def boundary_queries(field,guide,grid,limit=64):
    import numpy as np
    from winding_field.tifxyz_project import _uv_samples
    height,width=grid['shape_yx'];upper=field.origin_um+(np.array(field.shape[::-1])-1)*field.spacing_um;heap=[]
    for r in range(0,height,256):
        for c in range(0,width,256):
            low=[max(0,r-2),max(0,c-2)];high=[min(height,r+258),min(width,c+258)]
            rows,cols,ids,weights,used=_uv_samples(guide,grid,low,high)
            inside=used&(rows>=r)&(rows<r+256)&(cols>=c)&(cols<c+256)
            selected=np.flatnonzero(inside);tri=guide['faces'][ids.ravel()[selected]];w=weights.reshape(-1,3)[selected,:,None]
            points=np.sum(w*guide['world'][tri],axis=1).astype(float)
            directions=np.sum(w*guide['normals'][tri],axis=1).astype(float)
            unit=directions/np.linalg.norm(directions,axis=1)[:,None]
            margin=np.minimum(points-field.origin_um,upper-points)
            with np.errstate(divide='ignore',invalid='ignore'):exit_distance=np.min(np.where(abs(unit)>1e-15,margin/abs(unit),np.inf),axis=1)
            for j,k in enumerate(selected):
                index=int(rows.ravel()[k])*width+int(cols.ravel()[k])
                record=dict(pixel_yx=[int(rows.ravel()[k]),int(cols.ravel()[k])],origin_um=points[j].tolist(),
                            direction=directions[j].tolist(),exit_distance_um=float(exit_distance[j]))
                entry=(-float(exit_distance[j]),-index,record)
                if len(heap)<limit:heapq.heappush(heap,entry)
                elif entry[:2]>heap[0][:2]:heapq.heapreplace(heap,entry)
    return [x[2] for x in sorted(heap,reverse=True)]


def rank_native_exit(field,queries,*,level,radius):
    """Heuristic for finding counterexamples, never a root solver or acceptance gate."""
    import numpy as np
    if not queries:return []
    points=np.array([q['origin_um'] for q in queries]);direction=np.array([q['direction'] for q in queries])
    direction/=np.linalg.norm(direction,axis=1)[:,None]
    grid=np.clip(((points-field.origin_um)/field.spacing_um)[:,::-1],0,np.array(field.shape)-1)
    cells=np.minimum(np.floor(grid).astype(np.int64),np.array(field.shape)-2);local=grid-cells
    values,valid=field.read_cells(cells);f=np.zeros(len(queries));grad=np.zeros((len(queries),3))
    for bits in itertools.product((0,1),repeat=3):
        weights=np.where(bits,local,1-local);v=values[(slice(None),*bits)]
        f+=v*np.prod(weights,axis=1)
        for axis in range(3):grad[:,axis]+=v*(1 if bits[axis] else -1)*np.prod(weights[:,[i for i in range(3) if i!=axis]],axis=1)
    slope=np.sum(grad[:,::-1]/field.spacing_um*direction,axis=1)
    upper=field.origin_um+(np.array(field.shape[::-1])-1)*field.spacing_um
    with np.errstate(divide='ignore',invalid='ignore'):
        a=(field.origin_um-points)/direction;b=(upper-points)/direction
        lower_t=np.maximum(-radius,np.max(np.where(abs(direction)>1e-15,np.minimum(a,b),-np.inf),axis=1))
        upper_t=np.minimum(radius,np.min(np.where(abs(direction)>1e-15,np.maximum(a,b),np.inf),axis=1))
        predicted=-(f-level)/slope
    score=np.maximum(lower_t-predicted,predicted-upper_t)
    score[~valid|~np.isfinite(score)]=-1e300
    return sorted([dict(q,linearized_exit_violation_um=float(v)) for q,v in zip(queries,score)],
                  key=lambda q:(-q['linearized_exit_violation_um'],q['pixel_yx']))


def boundary_witnesses(pilot_path,guides_root,conformance_path,out,candidate,native_rank=False):
    import numpy as np
    from winding_field.tifxyz_contract import Blocked
    from winding_field.tifxyz_project import make_grid,ray_batch
    from winding_field.util import atomic_json,file_sha
    from validate_tifxyz_witness import exact_ray_segments
    if out.exists():raise FileExistsError(out)
    guard(out.parent);pilot=json.loads(pilot_path.read_text());proof=json.loads(conformance_path.read_text())
    if proof['status']!='HISTORICAL_SOURCE_ADAPTER_VERIFIED' or proof['config_sha256']!=CONFIG_SHA:raise ValueError('source not verified')
    previous=json.loads((guides_root/'RESULT.json').read_text())
    if previous['pilot_sha256']!=file_sha(pilot_path) or previous['candidate']!=candidate:raise ValueError('pilot differs')
    out.mkdir();results=[];started=time.monotonic()
    with open_historical(pilot['rois'][candidate]) as field:
        for row in previous['results']:
            if row['status']!='GUIDE_CANDIDATE_NOT_TIFXYZ':continue
            stride=row['stride'];directory=guides_root/f'stride-{stride:03d}';guide={'stride':stride}
            for name,expected in row['artifacts'].items():
                path=directory/(name+'.npy')
                if record(path)!=expected:raise ValueError('guide changed')
                guide[name]=np.load(path,allow_pickle=False)
            grid=make_grid(guide,pilot['pitch_um'],pilot['max_pixels'],level=pilot['level'])
            radius=2*float(np.linalg.norm(stride*field.spacing_um));found=None;sampled=[]
            queries=boundary_queries(field,guide,grid,limit=4096 if native_rank else 64)
            if native_rank:queries=rank_native_exit(field,queries,level=pilot['level'],radius=radius)[:64]
            criterion='native residual/gradient ranks 4096 boundary candidates; exact roots decide, not Newton' if native_rank else '64 reader-visible nodes with least ray exit distance; no root outcomes used'
            atomic_json(out/f'queries-{stride:03d}.json',dict(grid=grid,queries=queries,criterion=criterion))
            for start in range(0,len(queries),8):
                group=queries[start:start+8]
                roots=ray_batch(field,[q['origin_um'] for q in group],[q['direction'] for q in group],
                    [(-radius,radius)]*len(group),float(field.spacing_um.min())/1000,level=pilot['level'])
                for query,result in zip(group,roots):
                    sampled.append(dict(pixel_yx=query['pixel_yx'],status=result['status'],reason=result['reason'],roots=result['roots']))
                    if result['status']=='ABSENT' or result['reason'] in ('MULTIPLE_ROOTS','TANGENT_OR_PLATEAU','NONTRANSVERSE_BOUNDARY'):
                        unit=np.asarray(query['direction']);unit/=np.linalg.norm(unit)
                        if exact_ray_segments(query['origin_um'],unit.tolist(),field.origin_um.tolist(),field.spacing_um.tolist(),field.shape,radius):
                            found=(query,result);break
                if found:break
            atomic_json(out/f'samples-{stride:03d}.json',dict(samples=sampled))
            summary=dict(stride=stride,status='NO_REJECTION_WITNESS_IN_SAMPLE',candidate=candidate,sampled=len(sampled))
            if found:
                query,result=found;direction=np.asarray(query['direction']);direction/=np.linalg.norm(direction)
                segments=exact_ray_segments(query['origin_um'],direction.tolist(),field.origin_um.tolist(),field.spacing_um.tolist(),field.shape,radius)
                if not segments:raise ValueError('empty independently clipped witness')
                cells=np.array([s['cell'] for s in segments]);lo=cells.min(0);hi=cells.max(0)+2
                # Read original frozen source directly, not the production batched cell gatherer.
                values,valid=field.source.read(lo+field.lo,hi+field.lo)
                for segment,cell in zip(segments,cells-lo):
                    sl=tuple(slice(int(x),int(x)+2) for x in cell)
                    if not valid[sl].all():raise ValueError('unsupported witness cell')
                    segment['values']=values[sl].tolist()
                witness=dict(candidate=candidate,stride=stride,pixel_yx=query['pixel_yx'],level=pilot['level'],
                    origin_um=query['origin_um'],direction_unit=direction.tolist(),radius_um=radius,
                    field_origin_um=field.origin_um.tolist(),spacing_um=field.spacing_um.tolist(),shape_zyx=list(field.shape),
                    field_binding=dict(config_sha256=CONFIG_SHA,pilot=record(pilot_path),conformance=record(conformance_path),guide=row['artifacts']),
                    segments=segments,production_result=result,grid=grid)
                path=out/f'ray-{stride:03d}.json';atomic_json(path,witness)
                summary.update(status='RAY_WITNESS_CAPTURED_NOT_YET_INDEPENDENTLY_VALIDATED',reason=result['reason'],witness=record(path),pixel_yx=query['pixel_yx'])
            results.append(summary);atomic_json(out/'RESULT.json',dict(results=results,elapsed_s=time.monotonic()-started,source_reads=field.stats))
            print(json.dumps(summary),flush=True)


def replay_witness(witness_path,pilot_path,guides_root,conformance_path,out):
    import numpy as np
    from winding_field.tifxyz_project import make_grid,_uv_samples
    from winding_field.util import atomic_json,file_sha
    from validate_tifxyz_witness import exact_ray_segments
    if out.exists():raise FileExistsError(out)
    guard(out.parent);data=json.loads(witness_path.read_text());binding=data['field_binding']
    if (binding['config_sha256']!=CONFIG_SHA or binding['pilot']!=record(pilot_path)
        or binding['conformance']!=record(conformance_path)):raise ValueError('witness source binding differs')
    pilot=json.loads(pilot_path.read_text());conformance=json.loads(conformance_path.read_text())
    if conformance['status']!='HISTORICAL_SOURCE_ADAPTER_VERIFIED' or data['level']!=pilot['level']:raise ValueError('unverified source or level')
    previous=json.loads((guides_root/'RESULT.json').read_text())
    if previous['candidate']!=data['candidate'] or previous['pilot_sha256']!=file_sha(pilot_path):raise ValueError('guide pilot differs')
    row=next(r for r in previous['results'] if r['stride']==data['stride'])
    if row['status']!='GUIDE_CANDIDATE_NOT_TIFXYZ' or row['artifacts']!=binding['guide']:raise ValueError('guide binding differs')
    guide={'stride':data['stride']};directory=guides_root/f"stride-{data['stride']:03d}"
    for name,expected in row['artifacts'].items():
        path=directory/(name+'.npy')
        if record(path)!=expected:raise ValueError('guide artifact changed')
        guide[name]=np.load(path,allow_pickle=False)
    grid=make_grid(guide,pilot['pitch_um'],pilot['max_pixels'],level=pilot['level'])
    if grid!=data['grid']:raise ValueError('grid binding differs')
    y,x=data['pixel_yx'];height,width=grid['shape_yx'];r=(y//256)*256;c=(x//256)*256
    low=[max(0,r-2),max(0,c-2)];high=[min(height,r+258),min(width,c+258)]
    rows,cols,ids,weights,used=_uv_samples(guide,grid,low,high);i,j=y-low[0],x-low[1]
    if not (0<=i<used.shape[0] and 0<=j<used.shape[1] and used[i,j]):raise ValueError('witness pixel is not reader-visible')
    tri=guide['faces'][ids[i,j]];w=weights[i,j,:,None]
    origin=np.sum(w*guide['world'][tri],axis=0).astype(float)
    direction=np.sum(w*guide['normals'][tri],axis=0).astype(float);direction/=np.linalg.norm(direction)
    if not np.array_equal(origin,data['origin_um']) or not np.array_equal(direction,data['direction_unit']):raise ValueError('ray differs from guide pixel')
    with open_historical(pilot['rois'][data['candidate']]) as field:
        if (field.origin_um.tolist()!=data['field_origin_um'] or field.spacing_um.tolist()!=data['spacing_um']
            or list(field.shape)!=data['shape_zyx'] or 2*float(np.linalg.norm(data['stride']*field.spacing_um))!=data['radius_um']):
            raise ValueError('physical ROI/radius differs')
        segments=exact_ray_segments(data['origin_um'],data['direction_unit'],field.origin_um.tolist(),field.spacing_um.tolist(),field.shape,data['radius_um'])
        if not segments or len(segments)!=len(data['segments']):raise ValueError('incomplete ray segments')
        for actual,stored in zip(segments,data['segments']):
            if any(stored.get(k)!=v for k,v in actual.items()):raise ValueError('ray partition differs')
        cells=np.array([s['cell'] for s in segments]);lo=cells.min(0);hi=cells.max(0)+2
        values,valid=field.source.read(lo+field.lo,hi+field.lo)
        for cell,stored in zip(cells-lo,data['segments']):
            sl=tuple(slice(int(x),int(x)+2) for x in cell)
            if not valid[sl].all() or not np.array_equal(values[sl],stored['values']):raise ValueError('native corner values/support differ')
        field.validate()
    proof=dict(status='SOURCE_AND_GUIDE_REPLAY_VERIFIED',witness_sha256=file_sha(witness_path),field_binding=binding,
        candidate=data['candidate'],stride=data['stride'],pixel_yx=data['pixel_yx'],segments=len(segments),epoch=time.time())
    atomic_json(out,proof);print(json.dumps(proof,indent=2))


def main():
    os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PASTIX_NUM_THREADS='1')
    if hasattr(os,'sched_getaffinity'):
        allowed=os.sched_getaffinity(0);os.sched_setaffinity(0,{1 if 1 in allowed else min(allowed)})
    if os.getpriority(os.PRIO_PROCESS,0)<19:os.nice(19-os.getpriority(os.PRIO_PROCESS,0))
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='action',required=True)
    a=s.add_parser('prepare');a.add_argument('--output',type=Path,required=True)
    a=s.add_parser('probe-guides');a.add_argument('--pilot',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a.add_argument('--candidate',type=int,required=True)
    a=s.add_parser('conformance');a.add_argument('--kernel',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
    a=s.add_parser('probe-roots');a.add_argument('--pilot',type=Path,required=True);a.add_argument('--guides',type=Path,required=True);a.add_argument('--conformance',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a.add_argument('--candidate',type=int,required=True)
    a=s.add_parser('boundary-witnesses');a.add_argument('--pilot',type=Path,required=True);a.add_argument('--guides',type=Path,required=True);a.add_argument('--conformance',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a.add_argument('--candidate',type=int,required=True);a.add_argument('--native-rank',action='store_true')
    a=s.add_parser('replay-witness');a.add_argument('--witness',type=Path,required=True);a.add_argument('--pilot',type=Path,required=True);a.add_argument('--guides',type=Path,required=True);a.add_argument('--conformance',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.action=='prepare':print(json.dumps(prepare(args.output),indent=2))
    elif args.action=='probe-guides':probe(args.pilot,args.output,args.candidate)
    elif args.action=='conformance':conformance(args.kernel,args.output)
    elif args.action=='probe-roots':probe_roots(args.pilot,args.guides,args.conformance,args.output,args.candidate)
    elif args.action=='boundary-witnesses':boundary_witnesses(args.pilot,args.guides,args.conformance,args.output,args.candidate,args.native_rank)
    else:replay_witness(args.witness,args.pilot,args.guides,args.conformance,args.output)


if __name__=='__main__':main()
