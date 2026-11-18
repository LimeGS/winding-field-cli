from pathlib import Path
from types import SimpleNamespace
import pytest


def test_resource_admission_blocks_unknown_and_counts_metadata(tmp_path,monkeypatch):
    from winding_field import tifxyz_runtime as code
    from winding_field.tifxyz_contract import Blocked
    settings=SimpleNamespace(memory_bytes=3<<30,reserve_ram_bytes=2<<30,scratch_bytes=16<<30,reserve_disk_bytes=0)
    monkeypatch.setattr(code,'memory_report',lambda:dict(known=False,effective_headroom_bytes=0))
    with pytest.raises(Blocked,match='RAM'):code.admit(settings,{'working_bytes':1,'disk_bytes':0},tmp_path)
    monkeypatch.setattr(code,'memory_report',lambda:dict(known=True,effective_headroom_bytes=8<<30))
    with pytest.raises(Blocked,match='MEMORY'):code.admit(settings,{'working_bytes':4<<30,'disk_bytes':0},tmp_path)
    assert code.metadata_charge(16<<20,65536)==(1<<30)+(128<<20)


def test_cgroup_leaf_required_and_ancestor_caps(tmp_path):
    from winding_field.tifxyz_runtime import memory_report
    proc=tmp_path/'proc';groups=tmp_path/'groups';proc.mkdir();groups.mkdir()
    (proc/'self').mkdir();(proc/'meminfo').write_text('MemAvailable: 10000 kB\n')
    (proc/'self/cgroup').write_text('0::/parent/leaf\n');parent=groups/'parent';parent.mkdir()
    (parent/'memory.max').write_text('5000');(parent/'memory.current').write_text('4000')
    assert not memory_report(proc,groups)['known']
    leaf=parent/'leaf';leaf.mkdir();(leaf/'memory.max').write_text('max');(leaf/'memory.current').write_text('100')
    report=memory_report(proc,groups)
    assert report['known'] and report['effective_headroom_bytes']==1000


def memory_namespace(tmp_path, *, v1, membership, mount_root='/tenant', hybrid=False):
    proc=tmp_path/'proc';groups=tmp_path/'groups'
    (proc/'self').mkdir(parents=True);groups.mkdir()
    (proc/'meminfo').write_text('MemAvailable: 10000 kB\n')
    (proc/'self/cgroup').write_text(('6:memory:'+membership+'\n' if v1 else '')+
        ('0::'+membership+'\n' if not v1 or hybrid else ''))
    root=groups/'memory' if v1 else groups;root.mkdir(exist_ok=True)
    kind='cgroup cgroup rw,memory' if v1 else 'cgroup2 cgroup rw'
    escaped=str(root).replace(' ','\\040')
    (proc/'self/mountinfo').write_text(f'12 1 0:32 {mount_root} {escaped} ro - {kind}\n')
    limit,usage=('memory.limit_in_bytes','memory.usage_in_bytes') if v1 else ('memory.max','memory.current')
    (root/limit).write_text('6000');(root/usage).write_text('5000')
    return proc,groups,root,limit,usage


def test_hybrid_memory_uses_v1_controller_and_verified_mount_root(tmp_path):
    from winding_field.tifxyz_runtime import memory_report
    proc,groups,root,_,_=memory_namespace(tmp_path,v1=True,membership='/tenant',hybrid=True)
    # A unified entry does not mean the memory controller belongs to v2.
    (groups/'tenant').mkdir();(groups/'tenant/memory.max').write_text('9999999')
    (groups/'tenant/memory.current').write_text('0')
    report=memory_report(proc,groups)
    assert report['known'] and report['effective_headroom_bytes']==1000
    assert report['bounds'][0]['path']==str(root)


@pytest.mark.parametrize('v1',[False,True])
def test_namespaced_memory_leaf_keeps_visible_ancestor_cap(tmp_path,v1):
    from winding_field.tifxyz_runtime import memory_report
    proc,groups,root,limit,usage=memory_namespace(tmp_path,v1=v1,membership='/tenant/child')
    child=root/'child';child.mkdir()
    (child/limit).write_text('9000');(child/usage).write_text('10')
    report=memory_report(proc,groups)
    assert report['known'] and report['effective_headroom_bytes']==1000
    assert [b['path'] for b in report['bounds']]==[str(child),str(root)]


@pytest.mark.parametrize('membership',['/tenant/missing','/other','/tenant/../other'])
@pytest.mark.parametrize('v1',[False,True])
def test_namespaced_memory_never_substitutes_mount_root_for_unknown_leaf(tmp_path,v1,membership):
    from winding_field.tifxyz_runtime import memory_report
    proc,groups,*_=memory_namespace(tmp_path,v1=v1,membership=membership)
    report=memory_report(proc,groups)
    assert not report['known'] and report['effective_headroom_bytes']==0
    assert report['errors']


def test_hybrid_missing_v1_controller_never_falls_back_to_v2(tmp_path):
    from winding_field.tifxyz_runtime import memory_report
    proc,groups,root,limit,_=memory_namespace(tmp_path,v1=True,membership='/tenant',hybrid=True)
    (root/limit).unlink()
    (groups/'tenant').mkdir();(groups/'tenant/memory.max').write_text('9999999')
    (groups/'tenant/memory.current').write_text('0')
    assert not memory_report(proc,groups)['known']


@pytest.mark.parametrize('damage',['missing','duplicate','wrong-controller','traversal'])
def test_memory_mount_mapping_must_be_unambiguous(tmp_path,damage):
    from winding_field.tifxyz_runtime import memory_report
    proc,groups,*_=memory_namespace(tmp_path,v1=True,membership='/tenant')
    path=proc/'self/mountinfo';text=path.read_text()
    if damage=='missing':text=''
    elif damage=='duplicate':text+=text
    elif damage=='wrong-controller':text=text.replace('rw,memory','rw,cpu')
    else:text=text.replace('/tenant','/tenant/..')
    path.write_text(text)
    assert not memory_report(proc,groups)['known']


@pytest.mark.parametrize('v1',[False,True])
def test_memory_mount_mapping_preserves_host_cap_and_reclaim_exclusions(tmp_path,v1):
    from winding_field.tifxyz_runtime import memory_report
    proc,groups,root,_,_=memory_namespace(tmp_path,v1=v1,membership='/tenant')
    keys=('total_cache','total_shmem','total_dirty','total_writeback','total_mapped_file') if v1 else ('file','shmem','file_dirty','file_writeback','file_mapped')
    (root/'memory.stat').write_text('\n'.join(f'{key} {value}' for key,value in zip(keys,[2000,100,200,300,400])))
    report=memory_report(proc,groups)
    assert report['known'] and report['effective_headroom_bytes']==2000
    assert report['bounds'][0]['reclaim']==1000
    (proc/'meminfo').write_text('MemAvailable: 1 kB\n')
    assert memory_report(proc,groups)['effective_headroom_bytes']==1024


def test_parent_and_margin_charged_before_child_limit():
    from winding_field.tifxyz_runtime import child_ceiling
    from winding_field.tifxyz_contract import Blocked
    assert child_ceiling(3<<30,256<<20)==(3<<30)-(320<<20)
    with pytest.raises(Blocked):child_ceiling(3<<30,3<<30)


def test_cuda_bootstrap_requires_operator_uuid_and_cpu_stays_disabled(monkeypatch):
    from winding_field.tifxyz_runtime import worker_environment
    from winding_field.tifxyz_contract import Blocked
    monkeypatch.delenv('WINDING_TIFXYZ_CUDA_DEVICE',raising=False)
    with pytest.raises(Blocked,match='ACCELERATOR_UNAVAILABLE'):
        worker_environment(512<<20,1024,audit_backend='cuda')
    device='GPU-838d3361-1ab1-68b0-7043-6fb0ef3afb7b'
    monkeypatch.setenv('WINDING_TIFXYZ_CUDA_DEVICE',device)
    env=worker_environment(512<<20,1024,audit_backend='cuda')
    assert env['CUDA_VISIBLE_DEVICES']==device and env['WINDING_TIFXYZ_RUNTIME_BACKEND']=='cuda'
    assert worker_environment(512<<20,1024)['CUDA_VISIBLE_DEVICES']==''
    monkeypatch.setenv('WINDING_TIFXYZ_CUDA_DEVICE','0')
    with pytest.raises(Blocked):worker_environment(512<<20,1024,audit_backend='cuda')


@pytest.mark.skipif(__import__('sys').platform!='linux',reason='Linux DATA enforcement')
def test_cuda_data_guard_allows_virtual_reservation_but_blocks_heap(tmp_path,monkeypatch):
    import subprocess,sys
    from winding_field.tifxyz_runtime import worker_environment
    monkeypatch.setenv('WINDING_TIFXYZ_CUDA_DEVICE','GPU-838d3361-1ab1-68b0-7043-6fb0ef3afb7b')
    env=worker_environment(512<<20,1024,audit_backend='cuda')
    script="from winding_field.tifxyz_runtime import _configure_worker; _configure_worker(); import resource,mmap,sys; assert resource.getrlimit(resource.RLIMIT_DATA)[0]==512<<20; assert 'numpy' not in sys.modules; reserved=mmap.mmap(-1,2<<30,flags=mmap.MAP_PRIVATE|mmap.MAP_ANONYMOUS,prot=0); bytearray(600<<20)"
    result=subprocess.run([sys.executable,'-c',script],env=env,capture_output=True,text=True)
    assert result.returncode!=0 and 'MemoryError' in result.stderr,result.stderr


def test_worker_estimate_must_fit_after_controller_charge(tmp_path,monkeypatch):
    from winding_field import tifxyz_runtime as code
    from winding_field.tifxyz_contract import Blocked
    settings=SimpleNamespace(memory_bytes=3<<30,reserve_ram_bytes=2<<30,scratch_bytes=16<<30,reserve_disk_bytes=0)
    monkeypatch.setattr(code,'memory_report',lambda:dict(known=True,effective_headroom_bytes=8<<30))
    with pytest.raises(Blocked,match='MEMORY'):
        code.admit(settings,{'working_bytes':2<<30,'parent_rss':1536<<20,'disk_bytes':0},tmp_path)


def test_negative_controller_rss_is_invalid():
    from winding_field.tifxyz_runtime import child_ceiling
    with pytest.raises(ValueError):child_ceiling(3<<30,-(1<<30))


def test_public_entry_points_reject_unsupported_runtime_before_source_parse(tmp_path, monkeypatch):
    import sys
    from winding_field import tifxyz_runtime as code
    from winding_field.tifxyz import export_field, verify_export
    monkeypatch.setattr(sys,'platform','darwin')
    for result in (export_field(tmp_path/'absent',tmp_path/'out',{'levels':[0.]}),verify_export(tmp_path/'absent')):
        assert result['status']=='BLOCKED'
        assert result['reasons'][0]['code']=='RUNTIME_UNSUPPORTED'
    assert not (tmp_path/'out').exists()


def test_supervisor_sets_environment_and_reaps_worker(tmp_path, monkeypatch):
    import os
    import sys
    from winding_field import tifxyz_runtime as code
    monkeypatch.setattr(code,'memory_report',lambda:dict(known=True,effective_headroom_bytes=8<<30))
    monkeypatch.setattr(code,'process_rss',lambda pid:1024)
    settings=SimpleNamespace(memory_bytes=3<<30,reserve_ram_bytes=2<<30,scratch_bytes=16<<30,reserve_disk_bytes=0)
    script="import os,json; print(json.dumps(dict(status='PAUSED',env={k:os.environ[k] for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','CUDA_VISIBLE_DEVICES')},pid=os.getpid())))"
    result=code._supervise_process([sys.executable,'-c',script],{},settings,tmp_path,None)
    assert result['status']=='PAUSED'
    assert result['env']==dict(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
    with pytest.raises(ChildProcessError):os.waitpid(result['pid'],os.WNOHANG)


@pytest.mark.parametrize('trigger',['stop','sigint','rss','headroom','xfsz'])
def test_supervisor_stops_and_reaps_without_success(tmp_path,monkeypatch,trigger):
    import os
    import sys
    import time
    from winding_field import tifxyz_runtime as code
    settings=SimpleNamespace(memory_bytes=3<<30,reserve_ram_bytes=2<<30,scratch_bytes=16<<30,reserve_disk_bytes=0)
    started=time.monotonic();pids=set()
    def rss(pid):
        pids.add(pid)
        return (3<<30) if trigger=='rss' and pid!=os.getpid() else 1024
    monkeypatch.setattr(code,'process_rss',rss)
    calls=0
    def memory():
        nonlocal calls
        calls+=1
        return dict(known=True,effective_headroom_bytes=(1<<30) if trigger=='headroom' and calls>1 else 8<<30)
    monkeypatch.setattr(code,'memory_report',memory)
    script="import signal,time,os; signal.signal(signal.SIGUSR1,signal.SIG_IGN); time.sleep(30)"
    if trigger=='xfsz':script="import os,signal; signal.signal(signal.SIGXFSZ,signal.SIG_DFL); os.kill(os.getpid(),signal.SIGXFSZ)"
    def stop():
        if trigger=='sigint' and time.monotonic()-started>.2:raise KeyboardInterrupt()
        return trigger=='stop' and time.monotonic()-started>.2
    result=code._supervise_process([sys.executable,'-c',script],{},settings,tmp_path,stop)
    assert result['status']==('PAUSED' if trigger in ('stop','sigint') else 'BLOCKED')
    assert time.monotonic()-started<6
    for pid in pids-{os.getpid()}:
        with pytest.raises(ChildProcessError):os.waitpid(pid,os.WNOHANG)


def test_worker_module_import_is_torch_and_numpy_free():
    import subprocess
    import sys
    result=subprocess.run([sys.executable,'-c',"import sys; import winding_field.tifxyz; import winding_field.tifxyz_runtime; assert 'numpy' not in sys.modules and 'torch' not in sys.modules"],capture_output=True,text=True)
    assert result.returncode==0,result.stderr


@pytest.mark.skipif(__import__('sys').platform!='linux',reason='Linux runtime enforcement')
def test_linux_bootstrap_limits_precede_heavy_imports(tmp_path):
    import os
    import subprocess
    import sys
    from winding_field.tifxyz_runtime import worker_environment
    env=worker_environment(512<<20,1024)
    script="from winding_field.tifxyz_runtime import _configure_worker; _configure_worker(); import resource,os,sys; assert resource.getrlimit(resource.RLIMIT_AS)[0]==512<<20; assert len(os.sched_getaffinity(0))==1; assert os.getpriority(os.PRIO_PROCESS,0)==19; assert 'numpy' not in sys.modules; bytearray(600<<20)"
    result=subprocess.run([sys.executable,'-c',script],env=env,capture_output=True,text=True)
    assert result.returncode!=0 and 'MemoryError' in result.stderr


@pytest.mark.parametrize('invalid_protocol',[False,True])
def test_blocked_response_cannot_leave_worker_owned_ready(tmp_path,monkeypatch,invalid_protocol):
    import sys
    from winding_field import tifxyz_runtime as code
    monkeypatch.setattr(code,'memory_report',lambda:dict(known=True,effective_headroom_bytes=8<<30))
    monkeypatch.setattr(code,'process_rss',lambda pid:1024)
    settings=SimpleNamespace(memory_bytes=3<<30,reserve_ram_bytes=2<<30,scratch_bytes=16<<30,reserve_disk_bytes=0)
    script="import os,sys,json,pathlib; r=pathlib.Path(json.load(sys.stdin)['out']); r.mkdir(); (r/'.lock').touch(); (r/'SUPERVISOR.json').write_text(json.dumps(dict(token=os.environ['WINDING_TIFXYZ_SUPERVISION_ID']))); (r/'READY.json').write_text('{}'); print(json.dumps(dict(status='BLOCKED',reasons=[dict(code='MEMORY_BUDGET')])))"
    if invalid_protocol:script=script.rsplit('; print',1)[0]+"; print('invalid protocol')"
    out=tmp_path/'out'
    if invalid_protocol:
        with pytest.raises(ValueError):
            code._supervise_process([sys.executable,'-c',script],dict(operation='export',out=str(out)),settings,tmp_path,None)
    else:
        result=code._supervise_process([sys.executable,'-c',script],dict(operation='export',out=str(out)),settings,tmp_path,None)
        assert result['status']=='BLOCKED'
    assert not (out/'READY.json').exists()
    assert list(out.glob('READY.interrupted-*.json'))


def test_worker_diagnostics_cannot_corrupt_result_protocol():
    import json
    import subprocess
    import sys
    script="import os; from winding_field import tifxyz_runtime as r,tifxyz as t; r._configure_worker=lambda:None; t._worker=lambda *args:(print('Python diagnostic',flush=True),os.write(1,b'native diagnostic\\n'),dict(status='PAUSED'))[-1]; r._worker_main()"
    result=subprocess.run([sys.executable,'-c',script],input='{}',capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert json.loads(result.stdout)['status']=='PAUSED'
    assert 'Python diagnostic' in result.stderr and 'native diagnostic' in result.stderr


@pytest.mark.parametrize('status',['PARTIAL','TIFXYZ_PARTIAL_VERIFIED'])
def test_supervisor_preserves_partial_protocol(tmp_path,monkeypatch,status):
    import sys
    from winding_field import tifxyz_runtime as code
    monkeypatch.setattr(code,'memory_report',lambda:dict(known=True,effective_headroom_bytes=8<<30))
    monkeypatch.setattr(code,'process_rss',lambda pid:1024)
    settings=SimpleNamespace(memory_bytes=3<<30,reserve_ram_bytes=2<<30,scratch_bytes=16<<30,reserve_disk_bytes=0)
    result=code._supervise_process([sys.executable,'-c',"import json; print(json.dumps(dict(status="+repr(status)+",coverage_complete=False)))"],{},settings,tmp_path,None)
    assert result['status']==status and result['coverage_complete'] is False


def test_interrupted_partial_marker_is_withdrawn(tmp_path,monkeypatch):
    import sys
    from winding_field import tifxyz_runtime as code
    monkeypatch.setattr(code,'memory_report',lambda:dict(known=True,effective_headroom_bytes=8<<30))
    monkeypatch.setattr(code,'process_rss',lambda pid:1024)
    settings=SimpleNamespace(memory_bytes=3<<30,reserve_ram_bytes=2<<30,scratch_bytes=16<<30,reserve_disk_bytes=0)
    script="import os,json,pathlib,sys; r=pathlib.Path(json.load(sys.stdin)['out']); r.mkdir(); (r/'.lock').touch(); (r/'SUPERVISOR.json').write_text(json.dumps(dict(token=os.environ['WINDING_TIFXYZ_SUPERVISION_ID']))); (r/'PARTIAL.json').write_text('{}'); print(json.dumps(dict(status='BLOCKED',reasons=[dict(code='MEMORY_BUDGET')])))"
    out=tmp_path/'out'
    result=code._supervise_process([sys.executable,'-c',script],dict(operation='export',out=str(out)),settings,tmp_path,None)
    assert result['status']=='BLOCKED' and not (out/'PARTIAL.json').exists()
    assert list(out.glob('PARTIAL.interrupted-*.json'))


def test_readonly_verifier_bootstrap_does_not_guess_disk_reserve(tmp_path,monkeypatch):
    import sys
    from winding_field import tifxyz_runtime as code
    monkeypatch.setattr(sys,'platform','linux')
    monkeypatch.setattr(code,'memory_report',lambda:dict(known=True,effective_headroom_bytes=8<<30))
    monkeypatch.setattr(code,'process_rss',lambda pid:1024)
    monkeypatch.setattr(code.shutil,'disk_usage',lambda p:SimpleNamespace(free=512<<20))
    original=code._supervise_process
    # The bootstrap child pauses before source opening; its real admission still runs.
    def launch(command,*args):return original([sys.executable,'-c',"print('{\"status\":\"PAUSED\"}')"],*args)
    monkeypatch.setattr(code,'_supervise_process',launch)
    result=code.supervise(dict(operation='verify',out=str(tmp_path/'out')))
    assert result['status']=='PAUSED',result


@pytest.mark.parametrize('exits',[True,False])
def test_missing_worker_rss_requires_confirmed_exit_before_result(tmp_path,monkeypatch,exits):
    import os
    import sys
    from winding_field import tifxyz_runtime as code
    from winding_field.tifxyz_contract import Blocked
    monkeypatch.setattr(code,'memory_report',lambda:dict(known=True,effective_headroom_bytes=8<<30))
    def rss(pid):
        if pid==os.getpid():return 1024
        raise Blocked('RSS_OBSERVATION','missing process RSS during exit')
    monkeypatch.setattr(code,'process_rss',rss)
    settings=SimpleNamespace(memory_bytes=3<<30,reserve_ram_bytes=2<<30,scratch_bytes=16<<30,reserve_disk_bytes=0)
    script="import time; time.sleep("+('.2' if exits else '30')+"); print('{\"status\":\"READY\"}')"
    result=code._supervise_process([sys.executable,'-c',script],{},settings,tmp_path,None)
    assert result['status']==('READY' if exits else 'BLOCKED'),result
    if not exits:assert result['reasons'][0]['code']=='RSS_OBSERVATION'
    with pytest.raises(ChildProcessError):os.waitpid(result['supervision']['pid'],os.WNOHANG)


def test_controller_over_budget_is_not_ignored_when_worker_finishes(tmp_path,monkeypatch):
    import os
    import sys
    import time
    from winding_field import tifxyz_runtime as code
    monkeypatch.setattr(code,'memory_report',lambda:dict(known=True,effective_headroom_bytes=8<<30))
    calls=0
    def rss(pid):
        nonlocal calls
        if pid==os.getpid():
            calls+=1
            if calls>1:
                time.sleep(.2)
                return (3<<30)+1
        return 1024
    monkeypatch.setattr(code,'process_rss',rss)
    settings=SimpleNamespace(memory_bytes=3<<30,reserve_ram_bytes=2<<30,scratch_bytes=16<<30,reserve_disk_bytes=0)
    result=code._supervise_process([sys.executable,'-c',"import time; time.sleep(.2); print('{\"status\":\"READY\"}')"],{},settings,tmp_path,None)
    assert result['status']=='BLOCKED' and result['reasons'][0]['code']=='MEMORY_BUDGET'
