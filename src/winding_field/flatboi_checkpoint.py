"""Bounded native Flatboi batches with durable, source-bound UV warm starts."""
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time
import uuid

import numpy as np

from .native_patches import ResourceGuard
from .patch_pipeline import check_flatboi
from .sheet_atlas import _arrays, _json, _path, _verified
from .surface_export import read_flatboi_uv
from .util import atomic_json, file_sha


def sync_dir(path):
    fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def require_cgroup():
    """Native allocations require the same hard Linux envelope as the proven recipe."""
    try:
        group=Path('/sys/fs/cgroup')/Path('/proc/self/cgroup').read_text().strip().removeprefix('0::/').strip('/')
        ram=int((group/'memory.max').read_text());swap=int((group/'memory.swap.max').read_text())
        if not 0<ram<=21*2**30 or not 0<=swap<=64*2**30:raise ValueError('limits')
    except (OSError,ValueError) as error:
        raise ValueError('Flatboi requires cgroup v2 MemoryMax <=21G and MemorySwapMax <=64G') from error


def identity_for(source_map,input,app_run,batch_iterations=10,native_recipe='installed-2f89bfc9'):
    native=check_flatboi(app_run,native_recipe)
    prepared=Path(source_map).absolute().parent/'PREPARED.json'
    return dict(source_map_sha256=file_sha(source_map),source_obj_sha256=file_sha(input),
                source_prepared_sha256=file_sha(_path(prepared.parent,prepared.name)) if prepared.exists() else None,
                app_sha256=native['launcher_sha256'],binary_sha256=native['binary_sha256'],
                runner_sha256=file_sha(__file__),batch_iterations=batch_iterations,
                energy='symmetric_dirichlet',tolerance=1e-6,threads=2)


def load_checkpoint(root,identity):
    root=Path(root);latest=root/'LATEST.json'
    if not latest.exists():return None,None
    record=_json(_path(root,'LATEST.json'))
    if record['identity']!=identity:raise ValueError('checkpoint source/binary/config identity differs')
    path=_verified(root,record['obj'],record['obj_sha256'])
    if type(record['iterations']) is not int or not 0<record['iterations']<=200:raise ValueError('invalid checkpoint iteration count')
    if not math.isfinite(record['energy']) or record['energy']<=0:raise ValueError('invalid checkpoint energy')
    if type(record['converged']) is not bool:raise ValueError('invalid convergence flag')
    if _json(_path(root,str(path.parent.relative_to(root)/'CHECKPOINT.json')))!=record:
        raise ValueError('checkpoint pointer differs from durable batch receipt')
    if 'import_sha256' in identity:_verified(root,'IMPORTED.json',identity['import_sha256'])
    return record,path


def positive_uv(native,vertices,faces,guard=lambda:None):
    uv,binding=read_flatboi_uv(native,vertices,faces,vertex_atol=0)
    for first in range(0,len(faces),100000):
        guard();tri=uv[faces[first:first+100000]]
        a,b=tri[:,1]-tri[:,0],tri[:,2]-tri[:,0];signed=a[:,0]*b[:,1]-a[:,1]*b[:,0]
        if not np.isfinite(signed).all() or (signed<=0).any():raise ValueError('checkpoint contains inverted/degenerate UV faces')
    return uv,binding


def publish_checkpoint(root,batch,native,vertices,faces,identity,iterations,energy,converged,guard):
    guard();uv,binding=positive_uv(native,vertices,faces,guard)
    if not math.isfinite(energy) or energy<=0:raise ValueError('invalid checkpoint energy')
    destination=batch/'checkpoint.obj';temporary=batch/'checkpoint.obj.partial'
    with temporary.open('x') as stream:
        for values,fmt in ((vertices,'v %.17g %.17g %.17g'),(uv,'vt %.17g %.17g')):
            for first in range(0,len(values),100000):guard();np.savetxt(stream,values[first:first+100000],fmt=fmt)
        for first in range(0,len(faces),100000):
            guard();np.savetxt(stream,np.repeat(faces[first:first+100000]+1,2,axis=1),fmt='f %d/%d %d/%d %d/%d')
        stream.flush();os.fsync(stream.fileno())
    guard();os.replace(temporary,destination);sync_dir(batch)
    record=dict(identity=identity,iterations=iterations,energy=float(energy),converged=bool(converged),
                obj=str(destination.relative_to(root)),obj_sha256=file_sha(destination),native_obj_sha256=file_sha(native),
                binding=binding,physical_acceptance=False,resume_kind='UV warm-start; SLIM state reconstructed; no rescaling')
    guard();atomic_json(batch/'CHECKPOINT.json',record);sync_dir(batch)
    guard();atomic_json(root/'LATEST.json',record);sync_dir(root)
    return record


def import_checkpoint(source,output,source_map,input,app_run,runner_sha256,native_recipe='installed-2f89bfc9'):
    source=Path(source).absolute();out=Path(output).absolute()
    if any(q.is_symlink() for q in (out,*out.parents)):raise ValueError('output symlink refused')
    if not isinstance(runner_sha256,str) or not re.fullmatch('[a-f0-9]{64}',runner_sha256):raise ValueError('original runner SHA256 required')
    old=_json(_path(source,'IDENTITY.json'))
    batch=old.get('batch_iterations')
    if type(batch) is not int or not 1<=batch<=200:raise ValueError('invalid imported batch size')
    identity=identity_for(source_map,input,app_run,batch,native_recipe)
    expected=dict(identity,runner_sha256=runner_sha256)
    # Operational v1 did not bind PREPARED; the new migration identity always does.
    if 'source_prepared_sha256' not in old:expected.pop('source_prepared_sha256')
    record,path=load_checkpoint(source,expected)
    if old!=expected or record is None:raise ValueError('import identity differs or checkpoint missing')
    mapped=_arrays(source_map);positive_uv(path,mapped['world'],mapped['faces'])
    out.mkdir(parents=True,exist_ok=False);resources=ResourceGuard(out,6.,8.);resources()
    if shutil.disk_usage(out).free<8*2**30+2*path.stat().st_size:raise InterruptedError('import disk reserve')
    receipt=dict(source=str(source),source_latest_sha256=file_sha(source/'LATEST.json'),checkpoint=record,
                 source_identity_sha256=file_sha(source/'IDENTITY.json'),physical_acceptance=False)
    atomic_json(out/'IMPORTED.json',receipt);identity['import_sha256']=file_sha(out/'IMPORTED.json')
    batch=out/'imported';batch.mkdir();target=batch/'checkpoint.obj';shutil.copyfile(path,target)
    if file_sha(target)!=record['obj_sha256']:raise ValueError('import payload changed')
    with target.open('rb') as stream:os.fsync(stream.fileno())
    copied=dict(record,identity=identity,obj='imported/checkpoint.obj')
    atomic_json(out/'IDENTITY.json',identity);atomic_json(batch/'CHECKPOINT.json',copied);sync_dir(batch)
    resources();atomic_json(out/'LATEST.json',copied);sync_dir(out)
    atomic_json(out/'STATUS.json',dict(status='IMPORTED_CHECKPOINT',iterations=copied['iterations'],converged=copied['converged']))
    return copied


def run(args,*,guard=lambda:None):
    if not 1<=args.batch_iterations<=args.iterations<=200 or not math.isfinite(args.hours) or not 0<args.hours<=12:
        raise ValueError('require 1 <= batch <= iterations <=200 and 0 < hours <=12')
    require_cgroup();root=Path(args.output).absolute()
    if any(p.is_symlink() for p in (root,*root.parents)) or (root/'.lock').is_symlink():raise ValueError('native output symlink refused')
    root.mkdir(parents=True,exist_ok=True)
    with (root/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return _run_locked(args,root,guard)


def _run_locked(args,root,external_guard):
    start=time.monotonic();stopped=False;child=None;resources=ResourceGuard(root,22.,8.)
    def stop(*_):
        nonlocal stopped
        stopped=True
    previous={sig:signal.signal(sig,stop) for sig in (signal.SIGTERM,signal.SIGINT)}
    def guard():
        external_guard()
        resources()
        if stopped or (root/'STOP').exists():raise InterruptedError('stop requested; last published checkpoint retained')
        if time.monotonic()-start>args.hours*3600:raise TimeoutError('run timeout; last published checkpoint retained')
    try:
        guard();identity=identity_for(args.source_map,args.input,args.app_run,args.batch_iterations,args.native_recipe)
        saved_identity=root/'IDENTITY.json'
        if saved_identity.exists():
            saved=_json(_path(root,'IDENTITY.json'))
            if 'import_sha256' in saved:
                _verified(root,'IMPORTED.json',saved['import_sha256']);identity['import_sha256']=saved['import_sha256']
            if saved!=identity:raise ValueError('run identity changed')
        record,current=load_checkpoint(root,identity)
        atomic_json(saved_identity,identity);sync_dir(root)
        mapped=_arrays(args.source_map);vertices,faces=mapped['world'],mapped['faces']
        current=Path(args.input).resolve() if current is None else current
        positive_uv(current,vertices,faces,guard)
        done=0 if record is None else record['iterations']
        if done>args.iterations:raise ValueError('requested iteration budget precedes checkpoint')
        atomic_json(root/'STATUS.json',dict(status='RUNNING',iterations=done,target=args.iterations,started=time.time(),start_iterations=done))
        app=Path(args.app_run).expanduser().resolve()
        while done<args.iterations and not (record and record['converged']):
            guard()
            if shutil.disk_usage(root).free<8*2**30+6*current.stat().st_size:raise InterruptedError('insufficient space for batch plus8GiB reserve')
            count=min(args.batch_iterations,args.iterations-done);batch=root/f'batch-{done:06d}-{uuid.uuid4().hex[:8]}'
            batch.mkdir();sync_dir(root);mesh=batch/'inputmesh.obj'
            # Native tools must never have a writable hardlink to preserved source files.
            shutil.copyfile(current,mesh);input_sha=file_sha(mesh)
            command=[str(app),'flatboi',str(mesh),str(count),'symmetric_dirichlet','--tol=1e-6']
            env=dict(os.environ,PASTIX_NUM_THREADS='2',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
            atomic_json(batch/'COMMAND.json',dict(command=command,prior_iterations=done,input_sha256=input_sha,identity=identity));sync_dir(batch)
            print(f'BATCH_START completed={done} next={count} directory={batch}',flush=True)
            with (batch/'stdout.log').open('xb') as out,(batch/'stderr.log').open('xb') as err:
                child=subprocess.Popen(command,cwd=batch,env=env,stdout=out,stderr=err,start_new_session=True)
                while True:
                    guard()
                    with (batch/'stderr.log').open('rb') as errors:
                        errors.seek(max(0,os.fstat(errors.fileno()).st_size-65536))
                        if b'Negative diagonal term' in errors.read():raise RuntimeError('native negative diagonal')
                    try:code=child.wait(timeout=.5);break
                    except subprocess.TimeoutExpired:pass
            if code:raise RuntimeError(f'native exit {code}')
            guard();after=identity_for(args.source_map,args.input,app,args.batch_iterations,args.native_recipe)
            if file_sha(mesh)!=input_sha or any(after[k]!=identity[k] for k in after):raise ValueError('native input/binary changed')
            log=(batch/'stdout.log').read_text()
            if 'No per-corner UVs' in log:raise ValueError('native ignored warm-start UVs')
            energies=np.atleast_1d(np.loadtxt(batch/'inputmesh_energies_flatboi.txt'))
            if energies.ndim!=1 or len(energies)<2 or not np.isfinite(energies).all() or (energies<=0).any():raise ValueError('invalid native energy history')
            converged='Converged:' in log;steps=len(energies)-1
            if steps>count or (steps!=count and not converged):raise ValueError('unexpected completed iteration count')
            if (np.diff(energies)>1e-7*np.maximum(1.,energies[:-1])).any():raise ValueError('native energy increased')
            if record and abs(energies[0]-record['energy'])>1e-7*max(1.,record['energy']):raise ValueError('warm-start energy differs')
            native=_path(batch,'inputmesh_flatboi.obj')
            record=publish_checkpoint(root,batch,native,vertices,faces,identity,done+steps,energies[-1],converged,guard)
            done=record['iterations'];current=root/record['obj']
            print(f'CHECKPOINT iterations={done} energy={record["energy"]} sha256={record["obj_sha256"]}',flush=True)
        guard();status=dict(status='NATIVE_FINISHED_PENDING_QUALITY_VALIDATION',iterations=done,target=args.iterations,
                            converged=bool(record and record['converged']),seconds=time.monotonic()-start,
                            peak_tree_rss_bytes=resources.peak,checkpoint=record,physical_acceptance=False)
        atomic_json(root/'STATUS.json',status);sync_dir(root);return status
    except BaseException as error:
        if child is not None and child.poll() is None:
            try:os.killpg(child.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            child.wait()
        interrupted=dict(status='INTERRUPTED',error=repr(error),seconds=time.monotonic()-start,latest_checkpoint='LATEST.json if present')
        atomic_json(root/f'INTERRUPTED-{uuid.uuid4().hex[:8]}.json',interrupted)
        atomic_json(root/'STATUS.json',interrupted);sync_dir(root);raise
    finally:
        for sig,handler in previous.items():signal.signal(sig,handler)


def observe(root):
    """Read bounded log tails; calculated progress is never described as recoverable."""
    root=Path(root)
    def read(name):return _json(_path(root,name)) if (root/name).exists() else {}
    status=read('STATUS.json');checkpoint=read('LATEST.json')
    saved=checkpoint.get('iterations',0);calculated=saved;energy=checkpoint.get('energy')
    batches=list(root.glob('batch-*'))
    if batches and status.get('status')=='RUNNING':
        batch=max(batches,key=lambda p:p.stat().st_mtime)
        command=_json(_path(root,str(batch.relative_to(root)/'COMMAND.json')))
        logpath=batch/'stdout.log'
        if logpath.exists():
            with _path(root,str(logpath.relative_to(root))).open('rb') as stream:
                stream.seek(max(0,os.fstat(stream.fileno()).st_size-65536));log=stream.read().decode(errors='replace')
            progress=re.findall(r'PROGRESS (\d+)/(\d+)',log)
            if progress:calculated=max(saved,command['prior_iterations']+int(progress[-1][0]))
            energies=re.findall(r'\[it \d+\] E = ([\d.eE+\-]+)',log)
            if energies:energy=float(energies[-1])
    return dict(status=status.get('status','NOT_STARTED'),saved=saved,calculated=calculated,
                target=status.get('target',200),energy=energy,converged=checkpoint.get('converged',False))
