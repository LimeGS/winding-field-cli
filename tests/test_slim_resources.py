"""Resource admission is about actual arrays and effective limits, not mesh ceilings."""
from pathlib import Path
import resource
from types import SimpleNamespace

import numpy as np
import pytest

from field_runtime import ResourceUnavailable


def observed_memory(monkeypatch,headroom=4<<30):
    from winding_field import native_patches as resources,tifxyz_runtime
    report=dict(known=True,host_available_bytes=128<<30,effective_headroom_bytes=headroom,
                bounds=[dict(path='/parent',limit=24<<30,usage=20<<30,headroom=4<<30),
                        dict(path='/parent/child',limit=64<<30,usage=1<<30,headroom=63<<30)],
                errors=[])
    monkeypatch.setattr(tifxyz_runtime,'memory_report',lambda:report)
    monkeypatch.setattr(resources,'sys',SimpleNamespace(platform='linux'),raising=False)
    monkeypatch.setattr(resource,'getrlimit',lambda kind:(resource.RLIM_INFINITY,resource.RLIM_INFINITY))
    return resources,report


def test_admission_honors_ancestor_headroom_and_reports_required_bytes(monkeypatch):
    resources,_=observed_memory(monkeypatch)
    with pytest.raises(ResourceUnavailable) as failed:
        resources.admit_memory(5<<30,phase='cut topology')
    assert failed.value.report['required_bytes']==5<<30
    assert failed.value.report['available_bytes']==4<<30
    assert failed.value.report['phase']=='cut topology'
    assert failed.value.report['memory']['bounds'][0]['path']=='/parent'
    assert resources.admit_memory(3<<30,phase='cut topology')['admitted']


def test_admission_honors_process_address_headroom(monkeypatch):
    resources,_=observed_memory(monkeypatch,headroom=128<<30)
    def limit(kind):
        return (2<<30,2<<30) if kind==resource.RLIMIT_AS else (resource.RLIM_INFINITY,resource.RLIM_INFINITY)
    monkeypatch.setattr(resource,'getrlimit',limit)
    real=Path.read_text
    def status(path,*args,**kwargs):
        if path==Path('/proc/self/status'):
            return 'VmSize:\t1572864 kB\nVmData:\t1048576 kB\n'
        return real(path,*args,**kwargs)
    monkeypatch.setattr(Path,'read_text',status)
    with pytest.raises(ResourceUnavailable) as failed:
        resources.admit_memory(1<<30,phase='factorization')
    assert failed.value.report['available_bytes']==512<<20
    assert failed.value.report['process_limits'][0]['kind']=='RLIMIT_AS'

def test_admission_respects_ancestor_memory_high(monkeypatch):
    resources,_=observed_memory(monkeypatch)
    real=Path.read_text
    def high(path,*args,**kwargs):
        if path==Path('/parent/memory.high'):return str(22<<30)
        return real(path,*args,**kwargs)
    monkeypatch.setattr(Path,'read_text',high)
    with pytest.raises(ResourceUnavailable) as failed:
        resources.admit_memory(3<<30,phase='factorization')
    assert failed.value.report['available_bytes']==2<<30


@pytest.mark.parametrize('known,available',[(False,0),(True,0)])
def test_unknown_or_exhausted_headroom_never_admits(monkeypatch,known,available):
    resources,report=observed_memory(monkeypatch,headroom=available);report['known']=known
    with pytest.raises(ResourceUnavailable):
        resources.admit_memory(1,phase='read')


def test_explicit_operator_budget_stays_effective(monkeypatch):
    resources,_=observed_memory(monkeypatch,headroom=128<<30)
    with pytest.raises(ResourceUnavailable) as failed:
        resources.admit_memory(2<<30,phase='read',operator_bytes=1<<30)
    assert failed.value.report['available_bytes']==1<<30


def test_read_level_uses_packed_arrays_and_keeps_exact_source_ids():
    from test_patch_mapping import source
    from winding_field.patch_pipeline import read_level
    reader,ids=source()
    try:
        v,f,vi,fi,vk,fk=read_level(reader,0,memory_gib=4096/2**30)
        assert vi.tolist()==ids and fi.tolist()==[2**53+9]
        assert f.tolist()==[[0,1,2]]
        assert v.tolist()==[[0.,0.,1.],[1.,0.,1.],[0.,1.,1.]]
        assert list(fk)==['face-key'] and len(list(vk))==3
    finally:
        reader.db.close()


def test_read_level_can_use_separate_disk_arrays(tmp_path):
    from test_patch_mapping import source
    from winding_field.patch_pipeline import read_level
    reader,ids=source()
    try:
        arrays=read_level(reader,0,memory_gib=4096/2**30,workspace=tmp_path/'arrays')
        assert all(isinstance(a,np.memmap) for a in arrays[:4])
        assert arrays[2].tolist()==ids
        assert arrays[1].tolist()==[[0,1,2]]
        assert list(arrays[5])==['face-key']
    finally:
        reader.db.close()


@pytest.mark.parametrize('nv,nf',[(750001,1),(3,1250001)])
def test_valid_input_has_no_functional_vertex_or_face_ceiling(tmp_path,monkeypatch,nv,nf):
    from winding_field import native_global,native_patches
    v=np.lib.format.open_memmap(tmp_path/'v.npy',mode='w+',dtype='f8',shape=(nv,3))
    v[:]=0.;v[1]=[1.,0.,0.];v[2]=[0.,1.,0.]
    f=np.lib.format.open_memmap(tmp_path/'f.npy',mode='w+',dtype='i8',shape=(nf,3))
    f[:]=[0,1,2];v.flush();f.flush()
    v=np.load(tmp_path/'v.npy',mmap_mode='r');f=np.load(tmp_path/'f.npy',mmap_mode='r')
    monkeypatch.setattr(native_patches,'admit_memory',lambda *a,**kw:dict(admitted=True),raising=False)
    areas=native_global._input(v,f,None)
    assert areas.shape==(nf,) and np.all(areas==.5)


def test_cut_phase_admits_measured_cumulative_peak(monkeypatch):
    from winding_field import native_global,native_patches
    calls=[]
    def admit(required,*,phase,**kwargs):
        calls.append((phase,required));return dict(admitted=True)
    monkeypatch.setattr(native_patches,'admit_memory',admit,raising=False)
    v=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
    f=np.array([[0,1,2]],np.int64)
    native_global.cut_component(v,f)
    assert ('cut phase total',1024*len(f)+3*(3*8+8)) in calls


def test_cut_flags_do_not_overflow_vertex_pair_keys():
    from winding_field.tifxyz_guide import _cut_flags
    n=1<<33
    faces=np.array([[0,n-1,n-2],[1<<31,n-1,n-3]],np.int64)
    flags=_cut_flags(faces,[np.array([0,n-1],np.int64)],n)
    assert flags.tolist()==[[True,False,False],[False,False,False]]


def test_matrix_accepts_operator_budgets_above_legacy_ceilings(tmp_path):
    from test_flattening_matrix import request
    from winding_field.flattening_matrix import settings
    cfg=settings(dict(request(tmp_path),memory_gib=64.,process_tree_gib=96.),'lscm','sheet')
    assert cfg['memory_gib']==64. and cfg['process_tree_gib']==96.


def test_matrix_without_operator_memory_caps_uses_automatic_admission(tmp_path):
    from test_flattening_matrix import request
    from winding_field.flattening_matrix import settings
    req=request(tmp_path);req.pop('memory_gib')
    cfg=settings(req,'lscm','sheet')
    assert cfg['memory_gib'] is None and cfg['process_tree_gib'] is None


def test_obj_binding_does_not_materialize_python_topology_records(tmp_path):
    import tracemalloc
    from winding_field.surface_export import read_flatboi_uv
    nf=4096;nv=3*nf
    v=np.column_stack((np.repeat(np.arange(nf)*2.,3)+np.tile([0.,1.,0.],nf),
                       np.tile([0.,0.,1.],nf),np.zeros(nv)))
    f=np.arange(nv,dtype=np.int64).reshape(-1,3)
    path=tmp_path/'mesh.obj'
    with path.open('w') as stream:
        np.savetxt(stream,v,fmt='v %.17g %.17g %.17g')
        np.savetxt(stream,v[:,:2],fmt='vt %.17g %.17g')
        np.savetxt(stream,np.repeat(f+1,2,axis=1),fmt='f %d/%d %d/%d %d/%d')
    tracemalloc.start()
    try:
        uv,receipt=read_flatboi_uv(path,v,f,vertex_atol=0)
        _,peak=tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    np.testing.assert_array_equal(uv,v[:,:2])
    assert receipt['status']=='PASS'
    packed=nv*(3+2+1+2)*8+nf*(3*8+3*8+3)
    topology=3*nf*(2*8*3+8*4)+nf*(9*8+6*8+3*8+8)
    assert peak<=packed+2*topology, f'OBJ binding allocated {peak} bytes for packed envelope {packed+2*topology}'
