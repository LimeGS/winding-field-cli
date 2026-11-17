"""Consultative-only progress evidence; no solver or verification authority."""

import itertools
import inspect
import json
import math
import os
from pathlib import Path
import stat
import time
import warnings
import uuid

import numpy as np

from .field_convergence import _directory,_publish
from .surface_sources import FieldSource
from .util import digest

QUOTA_BYTES=4<<20
MAX_POINTS=128
MAX_CHUNKS=64


def require_observer():
    import block_solver
    if ('observer' not in inspect.signature(block_solver.solve_blocks).parameters
            or not hasattr(block_solver,'AdvisoryIntegrityError')):
        raise ValueError('field advice requires a development core with observer hooks; published core 1.5.2 is unsupported')


def observer_options(enabled,root,geometry,*,reserve_bytes=0):
    if type(enabled) is not bool:raise ValueError('field_advice must be boolean')
    if not enabled:return {}
    require_observer()
    return dict(observer=Advice(root,geometry,reserve_bytes=reserve_bytes))


def record_verification(observer,sha,result,started,*,sweep,regions=None):
    if observer is None:return
    try:
        elapsed=time.perf_counter()-started
        observer.finish()
        observer.record_verification(sha,dict(result,regions=regions,
            regions_scope='at most the first 64 owned cores; omitted/missing entries unavailable'),elapsed,
                                     sweep=sweep,complete=result.get('status')!='PAUSED')
    except Exception as error:
        warnings.warn('field advice verification unavailable: '+str(error),RuntimeWarning,stacklevel=2)


def finish(observer):
    if observer is not None:
        try:observer.finish()
        except Exception as error:
            warnings.warn('field advice timing unavailable: '+str(error),RuntimeWarning,stacklevel=2)


def _finite(value):
    return type(value) in (int,float) and math.isfinite(value)


def _read(path):
    with _directory(path.parent) as directory:
        fd=os.open(path.name,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=directory)
        with os.fdopen(fd,'rb') as stream:
            info=os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size>QUOTA_BYTES:
                raise ValueError('invalid diagnostic file')
            envelope=json.load(stream)
    if set(envelope)!={'data','sha256'} or digest(envelope['data'])!=envelope['sha256']:
        raise ValueError('diagnostic checksum differs')
    return envelope['data']


class Advice:
    """A bounded sibling journal; every callback is void and has no control authority."""
    def __init__(self,root,geometry=None,*,reserve_bytes=0):
        self.root=Path(root).absolute()
        self.path=self.root.with_name(self.root.name+'-advice')
        self.geometry=geometry
        self.reserve_bytes=reserve_bytes
        self.identity=None
        self.write_s=0.
        self.timings=[]

    def _write(self,name,data,*,pending_bytes=0,reserve_bytes=0):
        envelope=dict(data=data,sha256=digest(data))
        size=len(json.dumps(envelope,indent=2,sort_keys=True,allow_nan=False).encode())+1
        with _directory(self.path,create=True):
            used=0
            for path in self.path.iterdir():
                info=path.lstat()
                if not stat.S_ISREG(info.st_mode):raise ValueError('invalid diagnostic directory entry')
                used+=info.st_size
            # Include the replacement's temporary bytes, not just its final size.
            if used+size>QUOTA_BYTES:raise OSError('field advice quota exhausted')
            free=os.statvfs(self.path)
            if free.f_bavail*free.f_frsize<size+pending_bytes+max(reserve_bytes,self.reserve_bytes):
                raise OSError('insufficient diagnostic headroom for scientific commit')
            started=time.perf_counter()
            try:_publish(self.path/name,envelope)
            finally:self.write_s+=time.perf_counter()-started

    def restore(self,snapshot):
        self.identity=None
        geometry=self.geometry or snapshot['manifest']['identity']['source_identity'].get('geometry')
        if geometry is None:raise ValueError('explicit sampling geometry required')
        self.geometry=geometry
        identity=dict(schema='field-advice-v1',scientific_sha256=snapshot['scientific_sha256'],geometry=geometry)
        if (self.path/'IDENTITY.json').exists():
            if _read(self.path/'IDENTITY.json')!=identity:raise ValueError('incompatible diagnostic history')
        else:self._write('IDENTITY.json',identity)
        plan=_read(self.path/'PLAN.json') if (self.path/'PLAN.json').exists() else None
        for record in self.records():
            if (record.get('schema')!='field-advice-closure-v1' or
                    record.get('scientific_sha256')!=identity['scientific_sha256'] or
                    record.get('complete') and (record.get('plan')!=plan or record.get('plan_sha256')!=digest(plan))):
                raise ValueError('incompatible diagnostic closure history')
        journal=self.path/'JOURNAL.json'
        saved=_read(journal) if journal.exists() else dict(current=None,pending=None)
        if set(saved)!={'current','pending'}:raise ValueError('invalid diagnostic journal')
        matched=next((s['metrics'] for s in saved.values() if s is not None and
                      s['checkpoint_sha256']==snapshot['checkpoint_sha256']),None)
        if matched is not None:
            from block_solver import _restore_advice
            progress=snapshot['manifest']['progress']
            _restore_advice(matched,progress['sweep'],progress['next_block'],snapshot['scientific_sha256'])
        timings=_read(self.path/'TIMING.json') if (self.path/'TIMING.json').exists() else []
        self.identity=identity;self.timings=timings
        return matched

    def prepare(self,transaction,metrics):
        if self.identity is None:raise ValueError('diagnostic identity not restored')
        journal=self.path/'JOURNAL.json'
        saved=_read(journal) if journal.exists() else dict(current=None,pending=None)
        current=next((s for s in saved.values() if s is not None and
                      s['checkpoint_sha256']==transaction['previous_checkpoint_sha256']),None)
        pending=dict(checkpoint_sha256=transaction['checkpoint_sha256'],metrics=metrics)
        self._write('JOURNAL.json',dict(current=current,pending=pending),
                    pending_bytes=transaction['scientific_bytes'],reserve_bytes=transaction['reserve_bytes'])

    def records(self):
        return sorted((_read(p) for p in self.path.glob('closure-*.json')),key=lambda r:r['sweep'])

    def complete(self,snapshot):
        if self.identity is None:raise ValueError('diagnostic identity not restored')
        metrics=snapshot['metrics'];sha=snapshot['checkpoint_sha256']
        name='closure-'+sha+'.json'
        if (self.path/name).exists():
            previous=_read(self.path/name)
            if previous['scientific_sha256']!=snapshot['scientific_sha256']:
                raise ValueError('incompatible diagnostic closure')
            return
        started=time.perf_counter();write_before=self.write_s
        plan_path=self.path/'PLAN.json'
        if not metrics['complete']:
            self._write(name,dict(schema='field-advice-closure-v1',sweep=metrics['sweep'],
                checkpoint_sha256=sha,scientific_sha256=snapshot['scientific_sha256'],
                complete=False,metrics=metrics,reason='missing committed diagnostic counters',
                plan=None,plan_sha256=None,gauge=None,samples=[],reference=None))
            return
        plan=_read(plan_path) if plan_path.exists() else make_plan(snapshot,self.geometry)
        samples,reference,io=sample(snapshot,self.geometry,plan)
        if not plan_path.exists():
            plan['required_ids']=[s['node'] for s in samples if s['observed'] and _finite(s['phi'])]
            self._write('PLAN.json',plan)
        gauge='zero-origin' if snapshot['manifest']['identity'].get('boundary')=='natural-zero-gauge' else None
        record=dict(schema='field-advice-closure-v1',sweep=metrics['sweep'],checkpoint_sha256=sha,
            scientific_sha256=snapshot['scientific_sha256'],plan=plan,plan_sha256=digest(plan),
            gauge=gauge,reference=reference,samples=samples,io=io,metrics=metrics,
            complete=metrics['complete'],observed_count=sum(s['observed'] for s in samples),
            planned_count=len(samples),control_before_record_s=time.perf_counter()-started,
            persistence_before_record_s=self.write_s-write_before,
            timing_scope='active intervals; final record fsync tail unavailable after restart')
        self._write(name,record)
        self.timings=(self.timings+[dict(checkpoint_sha256=sha,sweep=metrics['sweep'],
            sweep_s=metrics['active_wall_s']+time.perf_counter()-started,
            control_s=metrics['control_s']+time.perf_counter()-started,
            timing_complete=metrics['timing_complete'],includes_closure_record_fsync=True)])[-3:]

    def finish(self):
        if self.timings:
            path=self.path/'TIMING.json'
            if not path.exists() or _read(path)!=self.timings:self._write('TIMING.json',self.timings)

    def record_verification(self,checkpoint_sha256,result,elapsed_s,*,sweep,complete=True):
        if self.identity is None:raise ValueError('diagnostic identity not restored')
        row=dict(result,checkpoint_sha256=checkpoint_sha256,
                 scientific_sha256=self.identity['scientific_sha256'],sweep=sweep,
                 elapsed_s=elapsed_s,timing_complete=complete,measured_epoch=time.time())
        name='verification-'+checkpoint_sha256+'-'+uuid.uuid4().hex+'.json'
        self._write(name,row)


def make_plan(snapshot,geometry):
    shape=np.asarray(geometry['shape_zyx']);chunk=snapshot['manifest']['chunk']
    points={};chunks={(0,0,0)}
    def add(node,reason):
        ij=np.asarray(node,dtype=int);key=tuple(ij//chunk)
        number=int(np.ravel_multi_index(tuple(ij),shape))
        if number in points:
            if reason not in points[number]['reasons']:points[number]['reasons'].append(reason)
            return
        if len(points)>=MAX_POINTS or len(chunks|{key})>MAX_CHUNKS:return
        chunks.add(key)
        points[number]=dict(node=number,zyx=ij.tolist(),reasons=[reason],
                            stratum='-'.join(map(str,np.minimum(2,ij*3//shape))),stencil=[])
    for row in sorted(snapshot['metrics']['candidates'],key=lambda r:r['rank']):
        add(np.unravel_index(row['node'],shape),'observed')
    for axis in range(3):
        count=(shape[axis]-1)//chunk
        for boundary in np.unique(np.linspace(1,count,min(3,count),dtype=int)) if count else []:
            ij=shape//2;ij[axis]=boundary*chunk
            add(ij,'interface')
    for node in itertools.product(*(np.unique(np.linspace(0,n-1,3,dtype=int)) for n in shape)):
        add(node,'spatial')
    # Admit whole central stencils only; never exceed the actual payload budget.
    for point in points.values():
        ij=np.asarray(point['zyx'])
        nodes=[]
        for scale in (1,2):
            for axis in range(3):
                for side in (-1,1):
                    other=ij.copy();other[axis]+=side*scale
                    if np.all(other>=0) and np.all(other<shape):nodes.append(other.tolist())
        touched={tuple(np.asarray(n)//chunk) for n in nodes}
        if len(chunks|touched)<=MAX_CHUNKS:
            chunks|=touched;point['stencil']=nodes
    return dict(version='deterministic-strata-v1',points=list(points.values()),
                multiblock=bool(np.any(shape>chunk)),required_ids=[],
                selected_strata=sorted({p['stratum'] for p in points.values()}))


def sample(snapshot,geometry,plan):
    """Gather each touched payload once through the existing integrity-checked reader."""
    from block_solver import AdvisoryIntegrityError
    shape=np.asarray(geometry['shape_zyx']);chunk=snapshot['manifest']['chunk']
    nodes={(0,0,0)}
    for point in plan['points']:
        nodes.add(tuple(point['zyx']));nodes.update(map(tuple,point['stencil']))
    grouped={}
    for ij in nodes:grouped.setdefault(tuple(np.asarray(ij)//chunk),[]).append(ij)
    if len(grouped)>MAX_CHUNKS:raise ValueError('sampling plan exceeds chunk budget')
    values={};stats=dict(payload_reads=0,decoded_chunks=0,encoded_bytes=0,decoded_bytes=0,
                        metadata_scope='manifest and inventory checked; only touched payloads hashed/decoded')
    descriptor=dict(kind='chunks',path=snapshot['checkpoint_path'],
                    sha256=snapshot['checkpoint_sha256'],support_policy='observed')
    source=None
    try:
        source=FieldSource(descriptor,geometry,cache_bytes=0)
        source._validate(payloads=False)
        for key,selected in sorted(grouped.items()):
            label='-'.join(map(str,key));record=snapshot['manifest']['chunks'].get(label)
            if record is None:
                for ij in selected:values[ij]=dict(phi=None,observed=False,error=None)
                continue
            block_shape=tuple(np.minimum(chunk,shape-np.asarray(key)*chunk))
            delta,offset,packed=source._load(label,block_shape)
            mask=np.unpackbits(packed,count=delta.size).reshape(block_shape) if packed.size else np.zeros(block_shape,bool)
            stats['payload_reads']+=1;stats['decoded_chunks']+=1
            stats['encoded_bytes']+=record['bytes'];stats['decoded_bytes']+=delta.nbytes+packed.nbytes+8
            for ij in selected:
                local=tuple(np.asarray(ij)-np.asarray(key)*chunk)
                phi=float(delta[local])+float(offset)
                values[ij]=dict(phi=phi,observed=bool(mask[local]),error=record.get('quantization_max'))
        source._validate(payloads=False)
    except (OSError,ValueError) as error:
        raise AdvisoryIntegrityError('scientific sample read failed: '+str(error)) from error
    finally:
        if source is not None:source.close()
    result=[]
    spacing=np.asarray(geometry['spacing_xyz'])[::-1]*geometry['voxel_um']
    for point in plan['points']:
        ij=np.asarray(point['zyx']);centre=values[tuple(ij)]
        row=dict(centre,node=point['node'],gradient_h=None,gradient_2h=None,gradient_error=None,
                 radius_um=float(2*min(spacing)),gradient_exclusion='unsupported or incomplete observed stencil')
        errors=[]
        for scale,name in ((1,'gradient_h'),(2,'gradient_2h')):
            gradient=[];uncertainty=[]
            for axis in range(3):
                step=np.eye(3,dtype=int)[axis]*scale
                a=values.get(tuple(ij-step));b=values.get(tuple(ij+step))
                if not centre['observed'] or any(v is None or not v['observed'] or
                        not _finite(v['phi']) or not _finite(v['error']) for v in (a,b)):break
                gradient.append((b['phi']-a['phi'])/(2*scale*spacing[axis]))
                uncertainty.append((a['error']+b['error']+np.finfo(float).eps*4*(abs(a['phi'])+abs(b['phi'])))/
                                   (2*scale*spacing[axis]))
            if len(gradient)==3:
                row[name]=gradient[::-1];errors.append(float(np.linalg.norm(uncertainty)))
        if len(errors)==2:
            row['gradient_error']=max(errors);row['gradient_exclusion']=None
        result.append(row)
    return result,values[(0,0,0)],stats


def _rms(values):
    return math.hypot(*values)/math.sqrt(len(values)) if values else None


def compare(before,after):
    """Measured phi intervals and explicitly local, conditional displacement estimates."""
    old={s['node']:s for s in before['samples']};new={s['node']:s for s in after['samples']}
    reference_a=before['reference'];reference_b=after['reference']
    same_gauge=before['gauge'] is not None and before['gauge']==after['gauge']
    points=[]
    for node in old.keys()&new.keys():
        a,b=old[node],new[node]
        raw=b['phi']-a['phi'] if all(_finite(s['phi']) for s in (a,b)) else None
        relative=(raw-(reference_b['phi']-reference_a['phi']) if raw is not None and same_gauge
                  and all(_finite(r['phi']) for r in (reference_a,reference_b)) else None)
        error=None
        endpoints=(a,b,reference_a,reference_b)
        if relative is not None and all(_finite(v['error']) and v['error']>=0 for v in endpoints):
            error=sum(v['error'] for v in endpoints)+np.finfo(float).eps*8*sum(abs(v['phi']) for v in endpoints)
        row=dict(node=node,observed=bool(a['observed'] and b['observed']),raw_delta_phi=raw,
                 relative_delta_phi=relative,encoding_uncertainty=error,normal_displacement=None,
                 gradient_margin=None,displacement_exclusion='unknown gauge, support or gradient')
        gradients=[s.get(name) for s in (a,b) for name in ('gradient_h','gradient_2h')]
        if row['observed'] and relative is not None and error is not None and all(
                g is not None and np.isfinite(g).all() for g in gradients) and all(
                _finite(s.get('gradient_error')) for s in (a,b)):
            norms=[float(np.linalg.norm(g)) for g in gradients]
            variation=max(float(np.linalg.norm(np.asarray(x)-y)) for x in gradients for y in gradients)
            margin=min(norms)-variation-max(a['gradient_error'],b['gradient_error'])
            row['gradient_margin']=margin
            row['displacement_exclusion']='insufficient resolved gradient'
            if margin>0:
                row['displacement_exclusion']='outside local stencil radius'
                if (abs(relative)+error)/margin<=min(a['radius_um'],b['radius_um']):
                    row['normal_displacement']=abs(relative)/norms[-1]
                    row['displacement_exclusion']=None
        points.append(row)
    eligible=[p for p in points if p['observed'] and p['relative_delta_phi'] is not None
              and p['encoding_uncertainty'] is not None]
    return dict(points=points,planned_count=len(before['samples']),
                common_observed_count=len(eligible),common_ids=[p['node'] for p in eligible],
                sweep_span=after['sweep']-before['sweep'])


def recommend(records,verification,availability):
    """The fixed diagnostic precedence; never a stop or a physical-accuracy policy."""
    window=records[-3:]
    result=dict(recommendation='DATOS_INSUFICIENTES',reason='three adjacent complete closures required',
                actions=[],physical_usefulness='precisión útil no calibrada',
                compared_checkpoints=[r['checkpoint_sha256'] for r in window],intervals=[],
                global_measurements=verification,required_count=None,common_count=None)
    if len(window)!=3:return result
    if not all(r.get('complete') and r.get('plan') for r in window):
        result['reason']='incomplete diagnostic closure';return result
    first=window[0];required=set(first['plan']['required_ids'])
    result['required_count']=len(required)
    compatible=all(r.get('complete') and r['metrics'].get('complete') and r.get('gauge') is not None
        and all(r.get(k)==first.get(k) for k in ('scientific_sha256','plan_sha256','gauge'))
        and r['sweep']==first['sweep']+i for i,r in enumerate(window))
    if not compatible:
        result['reason']='incomplete, gapped or incompatible history/gauge';return result
    comparisons=[compare(a,b) for a,b in zip(window,window[1:])]
    common=set.intersection(*(set(c['common_ids']) for c in comparisons))
    result['common_count']=len(common&required)
    interfaces={p['node'] for p in first['plan']['points'] if 'interface' in p['reasons']}
    if (len(required)<2 or not required<=common or first['plan']['multiblock'] and
            not any(all(any(s['node']==node and _finite(s['phi']) for s in r['samples']) for r in window)
                    for node in interfaces)):
        result['reason']='fixed required observed set or planned interface coverage unavailable';return result
    for comparison in comparisons:
        selected=[p for p in comparison['points'] if p['node'] in required]
        changes=[abs(p['relative_delta_phi']) for p in selected]
        errors=[p['encoding_uncertainty'] for p in selected]
        result['intervals'].append(dict(measured_rms=_rms(changes),
            lower=_rms([max(d-e,0.) for d,e in zip(changes,errors)]),
            upper=_rms([d+e for d,e in zip(changes,errors)]),count=len(required)))
    current=window[-1]['checkpoint_sha256'];scientific=first['scientific_sha256']
    actual=[v for v in verification if v.get('scientific_sha256')==scientific and
            _finite(v.get('residual_l2')) and type(v.get('global_converged')) is bool]
    latest=next((v for v in reversed(actual) if v['checkpoint_sha256']==current),None)
    a,b=result['intervals']
    if latest and latest['global_converged']:
        result['reason']='current global numerical convergence measured; further geometric usefulness unknown'
    elif latest and len(actual)>=2 and actual[-1] is latest and actual[-2]['residual_l2']<=latest['residual_l2']:
        result.update(recommendation='POSIBLE_ESTANCAMIENTO',reason='comparable measured global residual did not decrease')
    elif a['lower']==b['lower']==0:
        if availability and latest is None:
            result.update(recommendation='CONVIENE_VERIFICAR',reason='both stored intervals unresolved at encoding resolution; current closure unverified')
        else:
            result.update(recommendation='POSIBLE_ESTANCAMIENTO',reason='unresolved stored intervals; current global nonconverged' if latest else
                          'unresolved stored intervals; equivalent global verifier unavailable')
    elif b['upper']<a['lower']:
        result.update(recommendation='SEGUIR_CALCULANDO',reason='resolved contraction on the same required observed IDs')
    elif b['lower']>=a['upper']:
        result.update(recommendation='POSIBLE_ESTANCAMIENTO',reason='resolved noncontraction on the same required observed IDs')
    else:result['reason']='encoding-resolution intervals overlap; trend ambiguous'
    return result


def report(source):
    source=Path(source).absolute()
    root=source if (source/'CHECKPOINT.json').exists() else source/'field'
    sink=Advice(root)
    result=dict(recommendation='DATOS_INSUFICIENTES',reason='diagnostics unavailable',actions=[],
                physical_usefulness='precisión útil no calibrada',measured={},estimated=dict(
                local_normal_displacement=None,remaining_error=None),
                unknown=['remaining error','whole-volume stability','physical sufficiency'],
                current_closure_recorded=False,equivalent_global_verifier_available=False,cost_ratio=None)
    try:
        identity=_read(sink.path/'IDENTITY.json')
        records=sink.records()
        verification=sorted((_read(p) for p in sink.path.glob('verification-*.json')),
                            key=lambda r:(r['sweep'],r['measured_epoch']))
        from .util import file_sha
        sha=file_sha(root/'CHECKPOINT.json')
        available=bool((root.parent/'CONVERGENCE.json').exists() or (root/'RUN.json').exists() or
            (root.parent/'PREPARED.json').exists() and
            json.loads((root.parent/'PREPARED.json').read_bytes()).get('bounded_preparation'))
        result.update(recommend(records,verification,available))
        result['equivalent_global_verifier_available']=available
        latest=records[-1] if records else None
        result['current_closure_recorded']=bool(latest and latest['checkpoint_sha256']==sha)
        if not result['current_closure_recorded']:
            result.update(recommendation='DATOS_INSUFICIENTES',reason='current scientific checkpoint has no closure record')
        recent=records[-3:]
        comparisons=[compare(a,b) for a,b in zip(recent,recent[1:]) if a.get('complete') and b.get('complete')
                     and all(a.get(k)==b.get(k) for k in ('scientific_sha256','plan_sha256'))]
        estimates=[]
        for comparison in comparisons:
            points=[]
            for row in comparison['points']:
                points.append(dict(node=row['node'],normal_displacement=row.pop('normal_displacement'),
                                   reason=row.pop('displacement_exclusion')))
            estimates.append(dict(points=points,valid_count=sum(p['normal_displacement'] is not None for p in points)))
        if estimates:
            last=estimates[-1];values=[p['normal_displacement'] for p in last['points'] if p['normal_displacement'] is not None]
            last.update(p50_um=float(np.percentile(values,50)) if values else None,
                        p95_um=float(np.percentile(values,95)) if values else None,
                        excluded_count=len(last['points'])-len(values),
                        scope='first-order local estimate; not remaining error or a bound')
            result['estimated']['local_normal_displacement']=last
        result['measured']=dict(scientific_sha256=identity['scientific_sha256'],
            latest_closure=latest,global_verification=verification,
            comparisons=comparisons)
        timings=_read(sink.path/'TIMING.json') if (sink.path/'TIMING.json').exists() else []
        complete=bool(timings and all(t.get('timing_complete') and t.get('sweep_s',0)>0 for t in timings))
        timed=next((v for v in reversed(verification) if v.get('timing_complete')),None)
        result['measured']['timing']=dict(recent_sweeps=timings,includes_closure_record_fsync=complete)
        if complete and timed:
            duration=float(np.median([t['sweep_s'] for t in timings]))
            result['cost_ratio']=timed['elapsed_s']/duration
            result['measured']['timing'].update(sweep_median_s=duration,
                sweep_range_s=[min(t['sweep_s'] for t in timings),max(t['sweep_s'] for t in timings)],
                verification_age_sweeps=latest['sweep']-timed['sweep'])
        result['timing_scope']='active sweeps including closure record fsync; stop callbacks excluded; timing receipt own fsync excluded'
    except (OSError,ValueError,KeyError,TypeError,IndexError) as error:
        result.update(recommendation='DATOS_INSUFICIENTES',reason='diagnostics unavailable: '+str(error))
    return result


def main(argv=None):
    import argparse
    from .cli_settings import parse_options
    parser=argparse.ArgumentParser(prog='winding-field field-advice',
        description='Read-only consultative evidence. No solves, checks, stops or added sweeps.')
    parser.add_argument('source',help='run directory or prepared field root')
    args=parse_options(parser,argv,paths=('source',))
    print(json.dumps(report(args.source),indent=2,sort_keys=True,allow_nan=False))
    return 0
