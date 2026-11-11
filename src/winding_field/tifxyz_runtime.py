"""Read-only admission; source parsing must run inside the admitted worker."""
from pathlib import Path
import re
import shutil

from .tifxyz_contract import Blocked


def metadata_charge(size,count):
    if type(size) is not int or type(count) is not int or not 0<=size<=16<<20 or not 0<=count<=65536:
        raise ValueError('metadata envelope exceeded')
    return 64*size+2048*count


def memory_report(proc=Path('/proc'),groups=Path('/sys/fs/cgroup')):
    # Adapted from the existing mesh_to_tifxyz observer; missing leaf never falls back.
    result=dict(known=False,effective_headroom_bytes=0,errors=[],bounds=[])
    try:
        match=re.search(r'^MemAvailable:\s+(\d+)\s+kB\s*$',(proc/'meminfo').read_text(),re.M)
        if match is None:raise ValueError('missing MemAvailable')
        host=int(match[1])*1024;entries=(proc/'self/cgroup').read_text().splitlines()
        unified=next((line.split('::',1)[1] for line in entries if line.startswith('0::')),None)
        legacy=next((line.split(':',2)[2] for line in entries if 'memory' in line.split(':',2)[1].split(',')),None)
        # Hybrid hierarchies can expose v2 without assigning it the memory controller.
        if legacy is not None:unified=None
        if unified is not None:
            relative=unified;root=groups;names=('memory.max','memory.current');keys=('file','shmem','file_dirty','file_writeback','file_mapped')
        else:
            if legacy is None:raise ValueError('missing memory cgroup')
            relative=legacy
            root=groups/'memory';names=('memory.limit_in_bytes','memory.usage_in_bytes');keys=('cache','shmem','dirty','writeback','mapped_file')
        relative=Path(relative)
        if not relative.is_absolute() or '..' in relative.parts:raise ValueError('ambiguous cgroup namespace')
        mount_root=Path('/')
        mountinfo=proc/'self/mountinfo'
        if mountinfo.exists():
            mounts=[]
            unescape=lambda text:re.sub(r'\\([0-7]{3})',lambda m:chr(int(m[1],8)),text)
            for line in mountinfo.read_text().splitlines():
                left,right=line.split(' - ',1);fields=left.split();kind,_,options=right.split()[:3]
                if (Path(unescape(fields[4]))==root and
                        (kind=='cgroup2' if unified is not None else kind=='cgroup' and 'memory' in options.split(','))):
                    mounts.append(Path(unescape(fields[3])))
            if len(mounts)!=1:raise ValueError('missing or ambiguous memory cgroup mount')
            mount_root=mounts[0]
            if not mount_root.is_absolute() or '..' in mount_root.parts:raise ValueError('invalid memory cgroup mount root')
        # Only a matching mount can translate a host cgroup path to a container leaf.
        leaf=root/relative.relative_to(mount_root);directory=leaf;available=[host]
        while True:
            if directory==leaf or any((directory/n).exists() for n in names):
                text=(directory/names[0]).read_text().strip();used=int((directory/names[1]).read_text())
                cap=None if text=='max' else int(text)
                if used<0 or (cap is not None and cap<0):raise ValueError('negative cgroup values')
                if unified is None and cap is not None and cap>=(1<<63)-4096:cap=None
                reclaim=0
                try:
                    pairs=[line.split() for line in (directory/'memory.stat').read_text().splitlines()]
                    stats=dict(pairs)
                    if len(stats)!=len(pairs) or any(not v.isdecimal() for v in stats.values()):raise ValueError('invalid memory.stat')
                    wanted=tuple('total_'+k for k in keys) if unified is None and any('total_'+k in stats for k in keys) else keys
                    file,*exclusions=[int(stats[k]) for k in wanted]
                    reclaim=max(0,min(file,used)-sum(exclusions))
                except (OSError,ValueError,KeyError):pass
                head=None if cap is None else max(0,cap-used+reclaim)
                result['bounds'].append(dict(path=str(directory),limit=cap,usage=used,reclaim=reclaim,headroom=head))
                if head is not None:available.append(head)
            if directory==root:break
            if root not in directory.parents:raise ValueError('cgroup outside mount')
            directory=directory.parent
        result.update(known=True,effective_headroom_bytes=min(available),host_available_bytes=host)
    except (OSError,ValueError,StopIteration,IndexError) as error:result['errors'].append(str(error))
    return result


def child_ceiling(total,parent_rss):
    if type(total) is not int or total<=0 or type(parent_rss) is not int or parent_rss<0:raise ValueError('invalid process memory accounting')
    remaining=total-parent_rss-(64<<20)
    if remaining<=0:raise Blocked('MEMORY_BUDGET','controller leaves no child allowance')
    return remaining


def admit(settings,estimate,out):
    memory=memory_report();work=estimate['working_bytes'];disk=estimate['disk_bytes']
    if type(work) is not int or type(disk) is not int or min(work,disk)<0:raise ValueError('invalid resource estimate')
    # working_bytes estimates the child; controller RSS and margin are separate.
    ceiling=child_ceiling(settings.memory_bytes,estimate.get('parent_rss',0))
    if not memory['known'] or memory['effective_headroom_bytes']<settings.reserve_ram_bytes+work:raise Blocked('RAM_HEADROOM')
    if work>ceiling:raise Blocked('MEMORY_BUDGET','worker estimate exceeds child allowance')
    ceiling=min(ceiling,memory['effective_headroom_bytes']-settings.reserve_ram_bytes)
    if disk>settings.scratch_bytes:raise Blocked('SCRATCH_BUDGET')
    free=shutil.disk_usage(out).free
    if free<settings.reserve_disk_bytes+disk:raise Blocked('DISK_HEADROOM')
    return dict(memory=memory,free_bytes=free,working_bytes=work,disk_bytes=disk,
                child_address_ceiling=ceiling)


# Supervision is additive: the reviewed observer/admission accounting above is shared unchanged.
def process_rss(pid):
    try:
        text=(Path('/proc')/str(pid)/'status').read_text()
        match=re.search(r'^VmRSS:\s+(\d+)\s+kB\s*$',text,re.M)
        if match is None:raise ValueError('missing process RSS')
        return int(match[1])*1024
    except (OSError,ValueError) as error:
        raise Blocked('RSS_OBSERVATION',str(error)) from error


def worker_environment(ceiling,parent_rss,*,audit_backend='numpy'):
    import os
    import uuid
    env=os.environ.copy()
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS',
                 'NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS','PASTIX_NUM_THREADS'):
        env[key]='1'
    env.update(CUDA_VISIBLE_DEVICES='',OMP_DYNAMIC='FALSE',MKL_DYNAMIC='FALSE',PYTHONHASHSEED='0',PYTHONDONTWRITEBYTECODE='1',
         WINDING_TIFXYZ_CHILD_CEILING=str(ceiling),WINDING_TIFXYZ_PARENT_RSS=str(parent_rss),
         WINDING_TIFXYZ_SUPERVISION_ID=uuid.uuid4().hex,WINDING_TIFXYZ_RUNTIME_BACKEND=audit_backend)
    if audit_backend not in ('numpy','cpu','cuda'):raise ValueError('invalid audit backend')
    if audit_backend=='cuda':
        from .tifxyz_predicates import cuda_device
        env['CUDA_VISIBLE_DEVICES']=cuda_device()
    return env


def _configure_worker():
    import os
    import resource
    import sys
    if sys.platform!='linux':raise Blocked('RUNTIME_UNSUPPORTED','Linux supervision required')
    ceiling=int(os.environ['WINDING_TIFXYZ_CHILD_CEILING'])
    if not 0<ceiling<=3<<30:raise ValueError('invalid worker address ceiling')
    # CUDA reserves nonresident address ranges; bound heap/data plus supervised RSS.
    limit=resource.RLIMIT_DATA if os.environ.get('WINDING_TIFXYZ_RUNTIME_BACKEND')=='cuda' else resource.RLIMIT_AS
    old=resource.getrlimit(limit)
    cap=ceiling if old[1]==resource.RLIM_INFINITY else min(ceiling,old[1])
    resource.setrlimit(limit,(cap,cap))
    cpus=os.sched_getaffinity(0)
    if not cpus:raise Blocked('CPU_AFFINITY')
    os.sched_setaffinity(0,{min(cpus)})
    os.setpriority(os.PRIO_PROCESS,0,19)


def _blocked(code,detail=''):
    return dict(status='BLOCKED',reasons=[dict(code=code,detail=detail)])


def _mark_interrupted(request,token,result):
    """Only a reaped export worker's token authorizes changing its own status."""
    import fcntl
    import os
    from .tifxyz_contract import parse_json
    from .util import atomic_json
    if request.get('operation')!='export':return
    root=Path(request['out'])
    if any(p.is_symlink() for p in (root,*root.parents)):return
    marker=root/'SUPERVISOR.json'
    if not marker.is_file() or marker.is_symlink() or marker.stat().st_size>4096:return
    if parse_json(marker.read_text())!=dict(token=token):return
    fd=os.open(root/'.lock',os.O_RDWR|os.O_NOFOLLOW)
    try:
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for name in ('READY','PARTIAL'):
            marker=root/(name+'.json')
            if marker.exists():os.rename(marker,root/(name+'.interrupted-'+token+'.json'))
        atomic_json(root/'STATUS.json',result)
        directory=os.open(root,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
        try:os.fsync(directory)
        finally:os.close(directory)
    finally:os.close(fd)


def _supervise_process(command,request,settings,anchor,stop):
    import json
    import os
    import signal
    import subprocess
    import time
    from .tifxyz_contract import parse_json
    if stop and stop():return dict(status='PAUSED')
    parent=process_rss(os.getpid())
    admission=admit(settings,dict(working_bytes=2<<30,disk_bytes=0,parent_rss=parent),anchor)
    backend=request.get('settings',{}).get('audit_backend','numpy')
    # Verification cannot load paths/devices from an untrusted receipt in its controller.
    if request.get('operation')=='verify' and os.environ.get('WINDING_TIFXYZ_CUDA_DEVICE'):backend='cuda'
    env=worker_environment(admission['child_address_ceiling'],parent,audit_backend=backend)
    payload=json.dumps(request,allow_nan=False).encode()
    if len(payload)>1<<20:raise ValueError('worker request envelope exceeded')
    started=time.monotonic();peak=parent;halt=None;deadline=None
    child=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                           env=env,start_new_session=True)
    def signal_child(sig):
        try:os.kill(child.pid,sig)
        except ProcessLookupError:pass
    try:
        while True:
            try:
                stdout,stderr=child.communicate(payload,timeout=.1)
                break
            except subprocess.TimeoutExpired:
                payload=None
            except KeyboardInterrupt:
                payload=None;halt=dict(status='PAUSED');deadline=time.monotonic()+2;signal_child(signal.SIGUSR1)
            if halt is None:
                try:
                    controller_rss=process_rss(os.getpid())
                    if controller_rss>settings.memory_bytes:raise Blocked('MEMORY_BUDGET','controller RSS exceeded budget')
                    try:worker_rss=process_rss(child.pid)
                    except Blocked as observation:
                        if observation.code!='RSS_OBSERVATION':raise
                        # Linux can remove mm/VmRSS before waitpid sees exit. Never invent zero RSS.
                        try:stdout,stderr=child.communicate(timeout=.5)
                        except subprocess.TimeoutExpired:raise observation
                        break
                    aggregate=controller_rss+worker_rss;peak=max(peak,aggregate)
                    if aggregate>settings.memory_bytes:raise Blocked('MEMORY_BUDGET','aggregate worker/controller RSS exceeded')
                    memory=memory_report()
                    if not memory['known'] or memory['effective_headroom_bytes']<settings.reserve_ram_bytes:
                        raise Blocked('RAM_HEADROOM','host/cgroup reserve consumed')
                    if (stop and stop()) or (request.get('out') and (Path(request['out'])/'STOP').exists()):
                        halt=dict(status='PAUSED');deadline=time.monotonic()+2;signal_child(signal.SIGUSR1)
                except Blocked as error:
                    halt=_blocked(error.code,error.detail);deadline=time.monotonic();signal_child(signal.SIGKILL)
                except KeyboardInterrupt:
                    halt=dict(status='PAUSED');deadline=time.monotonic()+2;signal_child(signal.SIGUSR1)
            if deadline is not None and time.monotonic()>=deadline:signal_child(signal.SIGKILL)
        if halt is not None:
            _mark_interrupted(request,env['WINDING_TIFXYZ_SUPERVISION_ID'],halt)
            result=halt
        elif child.returncode:
            if child.returncode==-signal.SIGXFSZ:result=_blocked('FILE_SIZE_LIMIT','worker exceeded file-size quota')
            elif child.returncode in (-signal.SIGKILL,-signal.SIGSEGV):result=_blocked('WORKER_RESOURCE_EXIT',str(child.returncode))
            else:result=dict(status='FAILED',error='worker exited '+str(child.returncode)+': '+stderr.decode(errors='replace')[-4096:])
            _mark_interrupted(request,env['WINDING_TIFXYZ_SUPERVISION_ID'],result)
        else:
            if len(stdout)>1<<20:raise ValueError('worker response envelope exceeded')
            result=parse_json(stdout)
            if not isinstance(result,dict) or result.get('status') not in ('READY','PARTIAL','TIFXYZ_VERIFIED','TIFXYZ_PARTIAL_VERIFIED','PAUSED','BLOCKED','FAILED'):
                raise ValueError('invalid worker result')
            if result['status'] in ('PAUSED','BLOCKED','FAILED'):
                _mark_interrupted(request,env['WINDING_TIFXYZ_SUPERVISION_ID'],result)
        result['supervision']=dict(parent_rss_bytes=parent,peak_observed_aggregate_rss_bytes=peak,
            enforced_limit='RLIMIT_DATA' if backend=='cuda' else 'RLIMIT_AS',seconds=time.monotonic()-started,pid=child.pid)
        result['supervision']['child_data_ceiling' if backend=='cuda' else 'child_address_ceiling']=admission['child_address_ceiling']
        return result
    except BaseException as error:
        if child.poll() is None:signal_child(signal.SIGKILL)
        child.communicate()
        result=dict(status='PAUSED') if isinstance(error,KeyboardInterrupt) else dict(status='FAILED',error=str(error))
        _mark_interrupted(request,env['WINDING_TIFXYZ_SUPERVISION_ID'],result)
        if isinstance(error,KeyboardInterrupt):return result
        raise
    finally:
        if child.poll() is None:signal_child(signal.SIGKILL)
        child.communicate()


def supervise(request,stop=None):
    import sys
    from types import SimpleNamespace
    if sys.platform!='linux':return _blocked('RUNTIME_UNSUPPORTED','Linux supervision required; portable engine tests do not enforce a runtime ceiling')
    config=request.get('settings',{})
    settings=SimpleNamespace(memory_bytes=config.get('memory_bytes',int(config.get('memory_gib',3)*(1<<30))),
        scratch_bytes=config.get('scratch_bytes',int(config.get('scratch_gib',16)*(1<<30))),
        reserve_disk_bytes=config.get('reserve_disk_bytes',int(config.get('reserve_disk_gib',12.25)*(1<<30))),reserve_ram_bytes=2<<30)
    # Verification parses its effective limits under supervision, then admits scratch before writing.
    if request['operation']=='verify':settings.reserve_disk_bytes=0
    if settings.memory_bytes>3<<30 or settings.scratch_bytes>16<<30:return _blocked('BUDGET_ENVELOPE')
    try:
        result=_supervise_process([sys.executable,'-m','winding_field.tifxyz_runtime','--worker'],request,settings,Path(request['out']).parent,stop)
    except Blocked as error:return _blocked(error.code,error.detail)
    if result['status']=='FAILED':raise ValueError(result.get('error','worker integrity failure'))
    return result


def _worker_main():
    import json
    import os
    import signal
    import sys
    from .tifxyz_contract import Paused,parse_json
    reply=os.dup(sys.stdout.fileno())
    # Native libraries can write fd 1 directly; reserve a separate protocol fd.
    os.dup2(sys.stderr.fileno(),sys.stdout.fileno())
    stopped=False
    def stop_signal(*unused):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGUSR1,stop_signal);signal.signal(signal.SIGINT,stop_signal)
    try:
        _configure_worker()
        raw=sys.stdin.buffer.read((1<<20)+1)
        if len(raw)>1<<20:raise ValueError('worker request envelope exceeded')
        request=parse_json(raw)
        from .tifxyz import _worker
        result=_worker(request,lambda:stopped or (Path(request['out'])/'STOP').exists())
    except (Blocked,MemoryError) as error:
        result=_blocked(error.code,error.detail) if isinstance(error,Blocked) else _blocked('MEMORY_BUDGET','worker address ceiling reached')
    except Paused:result=dict(status='PAUSED')
    except Exception as error:result=dict(status='FAILED',error=type(error).__name__+': '+str(error))
    with os.fdopen(reply,'w') as stream:
        stream.write(json.dumps(result,allow_nan=False)+'\n');stream.flush()


if __name__=='__main__':
    import sys
    if sys.argv[1:]!=['--worker']:raise SystemExit('private supervised worker invocation only')
    _worker_main()
