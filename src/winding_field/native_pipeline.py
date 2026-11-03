"""Per-level native surfaces with an owned, interruptible worker and durable resume."""
import argparse
from contextlib import closing
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import time
import uuid

from .native_store import Journal,Paused
from .util import atomic_json,digest,file_sha


def _read(path,default=None):
    return json.loads(path.read_bytes()) if path.is_file() else default


def _root(run):
    run=Path(run).absolute(); root=run/'surfaces'
    if any(p.is_symlink() for p in (root,*root.parents)):
        raise ValueError('symlink in native output path')
    root.mkdir(exist_ok=True)
    return run,root


def _settings(config):
    from .native_geometry import solver_identity
    result=dict(coverage_target=config.get('coverage_target',1.),
                surface_time_limit_s=config.get('surface_time_limit_s',1800.),
                chart_faces=config.get('chart_faces',1024),mesh_tile=config.get('mesh_tile',64),
                export_tifxyz=config.get('export_tifxyz',False),parameterizer=solver_identity())
    if (type(result['coverage_target']) not in (int,float) or not 0<result['coverage_target']<=1
            or type(result['surface_time_limit_s']) not in (int,float)
            or not math.isfinite(result['surface_time_limit_s']) or result['surface_time_limit_s']<=0
            or type(result['export_tifxyz']) is not bool):
        raise ValueError('invalid native coverage/time/export settings')
    if result['export_tifxyz']:
        try:
            import tifffile
        except ImportError as error:
            raise ValueError('native TIFXYZ requires the optional tifffile dependency') from error
    return result


def _elapsed(root):
    value=float(_read(root/'TIME.json',{}).get('elapsed_s',0.))
    if (root/'work.sqlite').is_file():
        with closing(sqlite3.connect(root/'work.sqlite')) as db:
            row=db.execute("SELECT value FROM state WHERE name='elapsed_s'").fetchone()
            if row:
                value=max(value,float(json.loads(row[0])))
    return value


def _committed_level(root):
    root=Path(root); path=root/'work.sqlite'
    if any(p.is_symlink() for p in (path,*path.parents)):
        raise ValueError('symlink in native level ledger')
    if not path.is_file():
        return False
    # rw permits SQLite to recover a hot rollback journal, but never creates a DB.
    with closing(sqlite3.connect(path.resolve().as_uri()+'?mode=rw',uri=True)) as db:
        return db.execute("SELECT 1 FROM files WHERE name='LEVEL.json'").fetchone() is not None


def verify_level(root,identity=None):
    root=Path(root)
    if any(p.is_symlink() for p in (root,*root.parents)):
        raise ValueError('symlink in native level')
    if identity is not None and _read(root/'IDENTITY.json')!=identity:
        raise ValueError('native level identity changed')
    with closing(sqlite3.connect(root/'work.sqlite')) as db:
        for name,sha,size in db.execute('SELECT name,sha,bytes FROM files'):
            if not isinstance(name,str) or any(v in ('','.','..') for v in name.split('/')) or '\\' in name:
                raise ValueError('unsafe native product path')
            path=root/name
            if any(p.is_symlink() for p in (path,*path.parents)) or not path.is_file() or path.stat().st_size!=size or file_sha(path)!=sha:
                raise ValueError('native product changed: '+name)
        expected=db.execute("SELECT sha FROM files WHERE name='LEVEL.json'").fetchone()
        if expected is None:
            raise ValueError('uncommitted native level publication')
    return _read(root/'LEVEL.json')


def verify_ct_snapshot(root,volume):
    previous=volume._ledger
    with closing(sqlite3.connect((Path(root)/'CT-SNAPSHOT.sqlite').resolve().as_uri()+'?mode=ro',uri=True)) as snapshot:
        volume._ledger=snapshot
        try:
            volume.validate_used()
        finally:
            volume._ledger=previous


def _geometry_and_render(reader,store,volume,settings,pitch,until):
    from .native_geometry import flatten_mesh,assemble,summary
    from .native_render import render
    target=settings['coverage_target']
    available=target
    if until in ('flatten','assemble'):
        if store.get('render_identity') is not None:
            return None
        flatten_mesh(reader,store,settings['chart_faces'],until=available)
        if until=='assemble':
            assemble(store,target)
        return None
    if store.get('render_identity') is not None:
        mapped=summary(store)
    else:
        uv_deadline=store.elapsed+max(0.,store.seconds-store.elapsed)*.6
        while True:
            previous=store.get('face_cursor',0)
            flatten_mesh(reader,store,settings['chart_faces'],until=available,max_batches=4)
            mapped=assemble(store,target)
            if (mapped['coverage_fraction']+1e-12>=target or store.get('flatten_complete',False)
                    or store.elapsed>=uv_deadline):
                break
            if store.get('face_cursor',0)==previous:
                if available>=1.:
                    break
                available=min(1.,available+.1)
    rendered=render(store,volume,pitch,settings['export_tifxyz'])
    reader.validate()
    # observed_ct counts in-bounds samples, including fill; image contrast is not physical acceptance.
    usable=any(c['counts']['observed_ct']>0 and c['ct_value_range'] is not None
               and c['ct_value_range'][1]>c['ct_value_range'][0] for c in rendered['components'])
    all_tiff=not settings['export_tifxyz'] or all(c['tifxyz']['status']=='WRITTEN' and not c['tifxyz'].get('partial_geometry',False)
                                               for c in rendered['components'])
    complete=mapped['coverage_fraction']>=1.-1e-12 and usable and all_tiff
    return dict(mapped,**rendered,status='COMPLETE' if complete else 'PARTIAL',
                coverage_target=target,target_met=mapped['coverage_fraction']+1e-12>=target,
                usable_render=usable,usable_render_reason=None if usable else 'ct-no-contrast',
                source_mesh_complete=True,requested_tifxyz_complete=all_tiff,
                timings_s=store.timings(),elapsed_s=store.elapsed,original_deadline_met=store.elapsed<=settings['surface_time_limit_s'])


def render_mesh(reader,output,volume,config,stop=None):
    """Reuse a real immutable indexed mesh; timing is explicitly mesh-to-render."""
    from .workflow import implementation_sha
    settings=_settings(config)
    levels=reader.manifest['levels']
    if len(levels)!=1:
        raise ValueError('precomputed native rendering requires a single-level bundle')
    pitch=float(config.get('render_pitch_um',min(reader.metadata['spacing_xyz'])))
    identity=dict(producer='native-mesh-v1',mesh=reader.manifest_sha,ct=volume.identity,
                  level=levels[0]['level'],settings=settings,pitch_um=pitch,implementation_sha256=implementation_sha())
    if _committed_level(output):
        result=verify_level(output,identity)
        verify_ct_snapshot(output,volume)
        return result
    with Journal(output,identity,stop,settings['surface_time_limit_s'],config.get('reserve_bytes',0)) as store:
        try:
            result=_geometry_and_render(reader,store,volume,settings,pitch,'render')
            result['timing_scope']='existing-mesh-to-render'
            atomic_json(store.path('LEVEL.json'),result); store.register('LEVEL.json')
            return result
        except Paused as error:
            return dict(status='PAUSED',reason=str(error),elapsed_s=store.elapsed)


def work(run,config,until='render',stop=None,token=None,started_epoch=None):
    from .surface_mesh import MeshReader
    from .native_mesh import extract_mesh
    from .field import open_field
    from .meshes import select_levels
    from .workflow import open_volume
    run,root=_root(run)
    settings=_settings(config)
    if until not in ('mesh','flatten','assemble','render'):
        raise ValueError('invalid native final stage')
    started_epoch=started_epoch or time.time(); token=token or uuid.uuid4().hex
    allowance=float(_read(root/'ALLOWANCE.json',{}).get('extra_seconds',0.))
    limit=settings['surface_time_limit_s']+allowance
    def stopped():
        return bool(stop and stop() or (run/'STOP').exists() or (root/'STOP').exists())
    records=[]; current=None; first=True
    try:
        with open_field(run) as source, closing(open_volume(config)) as volume:
            volume.bind_ledger(root/'source.sqlite'); volume.validate_used()
            identity=dict(schema='native-surfaces-v1',field_sha256=file_sha(run/'FIELD.json'),source=source.identity,
                          ct=volume.identity,settings=settings,requested_levels=config.get('levels'),
                          implementation_sha256=__import__('winding_field.workflow',fromlist=['implementation_sha']).implementation_sha())
            old=_read(root/'NATIVE.json')
            if old is not None:
                if {k:v for k,v in old.items() if k!='levels'}!=identity:
                    raise ValueError('native source/configuration identity changed')
                levels=old['levels']
                if _read(root/'LEVELS.json')!=levels:
                    raise ValueError('native level catalog changed')
                identity=old
            else:
                levels,_=select_levels(run,config,source,stopped)
                if levels is None:
                    raise Paused('STOP')
                atomic_json(root/'LEVELS.json',levels)
                identity=dict(identity,levels=levels)
                atomic_json(root/'NATIVE.json',identity)
            for index,level in enumerate(levels):
                path=root/f'level-{index:04d}'
                binding=dict(producer='native-mesh-v1',parent=digest(identity),level=float(level))
                if _committed_level(path):
                    records.append(verify_level(path,binding))
                    verify_ct_snapshot(path,volume)
                    continue
                current=path; prior=_elapsed(path) if path.exists() else 0.
                phase_start=started_epoch if first else time.time(); first=False
                if stopped():
                    raise Paused('STOP')
                if prior+time.time()-phase_start>=limit:
                    path.mkdir(exist_ok=True)
                    atomic_json(path/'TIME.json',dict(elapsed_s=prior+time.time()-phase_start))
                    raise Paused('TIME_LIMIT')
                atomic_json(root/'CURRENT.json',dict(status='RUNNING',token=token,worker_pid=os.getpid(),level=index,
                    phase='mesh',started_epoch=phase_start,prior_elapsed_s=prior,limit_s=limit))
                startup_elapsed=max(0.,time.time()-phase_start)
                with Journal(path,binding,stopped,limit,config.get('reserve_bytes',0)) as store:
                    store.prior=max(store.prior,prior)+startup_elapsed
                    with store.db:
                        store.set('elapsed_s',store.elapsed)
                    store.phase='mesh'; store.check()
                    mesh=path/'meshes'
                    if not (mesh/'MESH.json').exists():
                        def mesh_stop():
                            store.check()
                            return False
                        result=extract_mesh(source,mesh,[float(level)],tile=settings['mesh_tile'],
                                            resume=(mesh/'RUN.json').exists(),stop=mesh_stop,
                                            reserve_bytes=config.get('reserve_bytes',0))
                        if result['status']!='MESH_READY':
                            raise Paused('STOP')
                    for name in ('MESH.json','RUN.json','mesh.sqlite','level-0000.obj'):
                        store.register('meshes/'+name)
                    if until=='mesh':
                        continue
                    with MeshReader(mesh) as reader:
                        if reader.manifest['identity']['source']!=source.identity:
                            raise ValueError('native mesh source differs')
                        result=_geometry_and_render(reader,store,volume,settings,float(min(source.spacing)),until)
                    if result is None:
                        continue
                    result.update(level=float(level),path=path.name,timing_scope='computed-field-to-render')
                    atomic_json(store.path('LEVEL.json'),result); store.register('LEVEL.json')
                    records.append(result)
            source.validate()
        if until!='render':
            return dict(status='PAUSED',reason='STAGE_LIMIT',until=until,levels=records)
        result=dict(status='COMPLETE' if all(r['status']=='COMPLETE' for r in records) else 'PARTIAL',
                    levels=records,physical_acceptance=False,field_coverage_certified=False,
                    producer='native-mesh-v1',identity_sha256=digest(identity))
        target=run/'SURFACES.json'; old=_read(target)
        if old is not None and old!=result:
            raise ValueError('published native aggregate changed')
        if old is None:
            atomic_json(target,result)
        return result
    except Paused as error:
        return dict(status='PAUSED',reason=str(error),levels=records,
                    active_level=None if current is None else current.name,
                    elapsed_s=0. if current is None else _elapsed(current))


def _rss(process):
    try:
        output=subprocess.check_output(['ps','-axo','pid=,pgid=,rss='],text=True)
        return sum(int(rss)*1024 for line in output.splitlines() for pid,group,rss in [line.split()]
                   if int(group)==process.pid or int(pid)==os.getpid())
    except (OSError,ValueError,subprocess.SubprocessError):
        return None


def _supervise(command,root,seconds,stop=None,token=None,prior=0.,level=0):
    root=Path(root); token=token or uuid.uuid4().hex
    start=time.time(); reason=None; request_at=None; peak=0
    atomic_json(root/'CURRENT.json',dict(status='RUNNING',token=token,level=level,phase='starting',
        started_epoch=start,prior_elapsed_s=prior,limit_s=seconds))
    atomic_json(root/'STATUS.json',dict(status='RUNNING',phase='starting',active_level=level,elapsed_s=prior))
    with (root/'worker.log').open('ab') as log:
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        last_observation=0.; rss=0
        try:
            while process.poll() is None:
                now=time.time(); current=_read(root/'CURRENT.json',{})
                if current.get('token')!=token:
                    raise ValueError('native worker ownership changed')
                elapsed=current['prior_elapsed_s']+now-current['started_epoch']
                if now-last_observation>=.5:
                    last_observation=now; rss=_rss(process)
                    if rss is not None:
                        peak=max(peak,rss)
                    index=current.get('level',level)
                    if type(index) is not int or not 0<=index<4096:
                        raise ValueError('invalid native progress cursor')
                    path=root/f'level-{index:04d}'
                    detail=_read(path/'STATUS.json',{})
                    if detail.get('worker_pid')!=process.pid:
                        detail={}
                    if detail.get('phase')=='mesh':
                        detail=dict(detail,mesh_progress=_read(path/'meshes/STATUS.json'))
                    atomic_json(root/'STATUS.json',dict(status='RUNNING',active_level=index,
                        phase=detail.get('phase',current['phase']),elapsed_s=elapsed,
                        aggregate_rss_bytes=rss,progress=detail))
                if reason is None:
                    if stop and stop() or (root/'STOP').exists():
                        reason='STOP'
                    elif current['limit_s'] is not None and elapsed>=current['limit_s']:
                        reason='TIME_LIMIT'
                    elif rss is None:
                        reason='RSS_OBSERVATION'
                    elif rss>3<<30:
                        reason='RSS_LIMIT'
                    if reason:
                        request_at=now
                        try:
                            os.killpg(process.pid,signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                elif now-request_at>2:
                    try:
                        os.killpg(process.pid,signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                time.sleep(.1)
        finally:
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGKILL)
            process.wait()
    current=_read(root/'CURRENT.json',{})
    if reason:
        index=current.get('level',level)
        if type(index) is not int or not 0<=index<4096:
            raise ValueError('invalid native level cursor')
        path=root/f'level-{index:04d}'; path.mkdir(exist_ok=True)
        elapsed=max(_elapsed(path),current.get('prior_elapsed_s',prior)+time.time()-current.get('started_epoch',start))
        atomic_json(path/'TIME.json',dict(elapsed_s=elapsed))
        result=dict(status='PAUSED',reason=reason,active_level=path.name,elapsed_s=elapsed)
    else:
        result=_read(root/'WORKER-RESULT.json',{})
        if result.get('token')!=token or result.get('worker_pid')!=process.pid:
            result=dict(status='FAILED',reason='WORKER_EXIT',returncode=process.returncode)
        else:
            result=result['result']
    result=dict(result,supervision=dict(peak_aggregate_rss_bytes=peak,worker_seconds=time.time()-start,pid=process.pid))
    atomic_json(root/'STATUS.json',result)
    return result


def run_native(run,config,stop=None,until='render',extra_seconds=0):
    settings=_settings(config)
    if type(extra_seconds) not in (int,float) or not math.isfinite(extra_seconds) or extra_seconds<0:
        raise ValueError('extra surface seconds must be nonnegative finite')
    run,root=_root(run)
    with (root/'.native.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        allowance=_read(root/'ALLOWANCE.json',{'extra_seconds':0.,'extensions':[]})
        if extra_seconds:
            allowance['extra_seconds']+=extra_seconds
            allowance['extensions'].append(dict(seconds=extra_seconds,at_epoch=time.time()))
            atomic_json(root/'ALLOWANCE.json',allowance)
        root.joinpath('STOP').unlink(missing_ok=True)
        levels=_read(root/'LEVELS.json',config.get('levels') or [0])
        first=next((i for i in range(len(levels)) if not _committed_level(root/f'level-{i:04d}')),0)
        prior=_elapsed(root/f'level-{first:04d}') if (root/f'level-{first:04d}').exists() else 0.
        finished=(run/'SURFACES.json').is_file() and all(_committed_level(root/f'level-{i:04d}') for i in range(len(levels)))
        limit=None if finished and until=='render' else settings['surface_time_limit_s']+allowance['extra_seconds']
        if limit is not None and prior>=limit:
            result=dict(status='PAUSED',reason='TIME_LIMIT',active_level=f'level-{first:04d}',elapsed_s=prior)
            atomic_json(root/'STATUS.json',result)
            return result
        token=uuid.uuid4().hex
        packet=dict(config=config,until=until,token=token,started_epoch=time.time())
        atomic_json(root/'REQUEST.json',packet)
        command=[sys.executable,'-B','-m','winding_field.native_pipeline','--worker',str(run),'--token',token]
        return _supervise(command,root,limit,stop,token,prior,first)


def _main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',type=Path,required=True); parser.add_argument('--token',required=True)
    args=parser.parse_args(); root=args.worker/'surfaces'; packet=_read(root/'REQUEST.json')
    if packet['token']!=args.token:
        raise ValueError('native worker request ownership changed')
    interrupted=False
    def handle(signum,frame):
        nonlocal interrupted
        interrupted=True
    for sig in (signal.SIGINT,signal.SIGTERM):
        signal.signal(sig,handle)
    try:
        os.nice(19)
        import torch
        torch.set_num_threads(1)
        result=work(args.worker,packet['config'],packet['until'],lambda:interrupted,args.token,packet['started_epoch'])
    except Exception as error:
        import traceback
        traceback.print_exc()
        result=dict(status='FAILED',reason=f'{type(error).__name__}: {error}')
    atomic_json(root/'WORKER-RESULT.json',dict(token=args.token,worker_pid=os.getpid(),result=result))


if __name__=='__main__':
    _main()
