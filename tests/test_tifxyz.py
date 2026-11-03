"""Task 6 integration: real sources, roots, tile ledger, audit and TIFF IO."""
from dataclasses import replace
import json

import numpy as np
import pytest

from winding_field.tifxyz_contract import Blocked, load_settings
from winding_field.tifxyz_field import open_physical_field, selected_frame
from winding_field.util import atomic_json, digest, file_sha
from tifxyz_fixture import mutate_json
from test_tifxyz_native import artifact


def physical_source(root,shape=(3,3,3),origin=(0,0,0),spacing=(1,1,1),chunk=None):
    from winding_field._core.field_store import BlockStore
    root.mkdir()
    source = dict(source='synthetic', level=0, shape=[20]*3,
                  origin_xyz=[0]*3, spacing_xyz=[1.]*3, metadata_sha256='a'*64)
    geometry = dict(frame=digest(source), voxel_um=1., sheet_step=1.,
                    origin_xyz=list(origin), spacing_xyz=list(spacing), shape_zyx=list(shape))
    chunk=max(shape) if chunk is None else chunk
    with BlockStore(root/'field', shape, chunk, {'geometry': geometry}) as store:
        for key in np.ndindex(*(int((n+chunk-1)//chunk) for n in shape)):
            lo=np.array(key)*chunk;hi=np.minimum(lo+chunk,shape);size=tuple(hi-lo)
            store.commit(key, np.indices(size)[0].astype(float)+lo[0]-.5, {}, np.ones(size, bool))
    atomic_json(root/'FIELD.json', dict(status='FIELD_READY', geometry=geometry,
                checkpoint_sha256=file_sha(root/'field/CHECKPOINT.json')))
    atomic_json(root/'RUN.json', dict(schema='winding-field-run-v1', implementation_sha256='b'*64,
                source=source, stages={'field': {'receipt':'FIELD.json', 'sha256':file_sha(root/'FIELD.json')}}, status='PAUSED'))
    return root


@pytest.fixture
def controlled(monkeypatch):
    # Portable engine tests observe resources explicitly; this is not a Mac runtime gate.
    from winding_field import tifxyz_runtime as runtime
    monkeypatch.setattr(runtime, 'memory_report', lambda: dict(known=True, effective_headroom_bytes=8<<30))


@pytest.fixture
def engine(monkeypatch, controlled):
    """Replace only libigl's parameterization/lookup with an exact planar chart."""
    from winding_field import tifxyz_guide, tifxyz_project
    def parameterize(v, f, stop=None):
        return dict(world=v, faces=f, uv=v[:,:2].copy(), normals=np.tile([0.,0.,1.], (len(v),1)),
                    source_vertices=np.arange(len(v)), boundary=np.arange(len(v)))
    def samples(guide, grid, low, high):
        rows, cols = np.mgrid[low[0]:high[0], low[1]:high[1]]
        queries = np.stack([cols, rows], -1)*grid['pitch_um']+grid['uv_min_um']
        ids = np.zeros(rows.shape, int); weights = np.zeros((*rows.shape,3), np.longdouble)
        used = np.zeros(rows.shape, bool)
        for i, face in enumerate(guide['faces']):
            a,b,c = guide['uv'][face]
            w = np.linalg.solve(np.column_stack([b-a,c-a]), (queries-a).reshape(-1,2).T).T
            w = np.column_stack([1-w.sum(1),w]).reshape((*rows.shape,3))
            good = (w>=0).all(-1); ids[good]=i; weights[good]=w[good]; used |= good
        return rows, cols, ids, weights, used
    monkeypatch.setattr(tifxyz_guide, 'parameterize', parameterize)
    monkeypatch.setattr(tifxyz_project, '_uv_samples', samples)


def settings(field, **kwargs):
    return load_settings(dict(dict(levels=[0.], reserve_disk_gib=0, scratch_gib=.5), **kwargs), field.binding['geometry'], selected_frame(field.binding))


@pytest.mark.parametrize('scratch_gib, fits', [(2., False), (16., True)])
def test_sqlite_index_can_grow_past_256m_only_with_scratch_allowance(tmp_path, scratch_gib, fits):
    from winding_field.tifxyz import _budget
    from winding_field.tifxyz_store import TileStore
    source = physical_source(tmp_path/'source')
    with open_physical_field(source) as f:
        budget = _budget(settings(f, scratch_gib=scratch_gib), tmp_path)
    grid = dict(shape_yx=[1247,1247], tile_pixels=256, pitch_um=8.64,
                uv_min_um=[0.,0.], level=-4.)
    with TileStore(tmp_path/'store', dict(grid=grid), budget, 'delivered') as store:
        with store.transaction() as db:
            db.execute('CREATE TABLE index_size_probe (payload BLOB)')
        def grow():
            with store.transaction() as db:
                db.execute('INSERT INTO index_size_probe VALUES (zeroblob(?))', (257<<20,))
        if fits:
            grow()
            assert store.db.execute('SELECT length(payload) FROM index_size_probe').fetchone() == (257<<20,)
            assert store.guard(sql=True)['reserved_bytes'] >= store.journal_cap
        else:
            with pytest.raises(Blocked, match='SQLITE_WRITE_LIMIT'):
                grow()


def tree_state(root):
    return {str(p.relative_to(root)): (file_sha(p), p.stat().st_mtime_ns)
            for p in root.rglob('*') if p.is_file()}


def export_plane(tmp_path, **kwargs):
    from winding_field.tifxyz import _export_source
    root = physical_source(tmp_path/'source')
    with open_physical_field(root) as field:
        result = _export_source(field, tmp_path/'out', settings(field, **kwargs), False, None)
    assert result['status']=='READY', result
    return root, tmp_path/'out'


def test_plane_export_and_fresh_readonly_verification(tmp_path, engine):
    from winding_field.tifxyz import _verify_source
    source, out = export_plane(tmp_path)
    before, source_before = tree_state(out), tree_state(source)
    with open_physical_field(source) as field:
        result = _verify_source(field, out)
    assert result['status']=='TIFXYZ_VERIFIED', result
    assert result['physical_acceptance'] is False
    assert result['native_topology_equivalence']=='UNMEASURED'
    assert tree_state(out)==before and tree_state(source)==source_before
    from winding_field.tifxyz_store import iter_tiff_tiles
    z = next(iter_tiff_tiles(out/'level-0000/z.tif', [3,3], 9))[1]
    np.testing.assert_array_equal(z, np.full((3,3), .5, np.float32))


def test_native_export_pins_operator_artifact_for_fresh_verification(tmp_path,engine,artifact,monkeypatch):
    import shutil
    from winding_field.tifxyz import _verify_source
    monkeypatch.setenv('WINDING_TIFXYZ_ACCELERATOR',str(artifact/'accelerator.json'))
    source,out=export_plane(tmp_path,audit_backend='cpu')
    identity=json.loads((out/'EXPORT.json').read_text())
    assert identity['audit_accelerator']['backend']=='cpu'
    assert identity['audit_accelerator']['library_sha256']==json.loads((artifact/'accelerator.json').read_text())['library_sha256']
    with open_physical_field(source) as field:assert _verify_source(field,out)['status']=='TIFXYZ_VERIFIED'
    shutil.copytree(artifact,tmp_path/'changed')
    path=tmp_path/'changed/accelerator.json';path.write_text(path.read_text()+' ')
    monkeypatch.setenv('WINDING_TIFXYZ_ACCELERATOR',str(path))
    with open_physical_field(source) as field,pytest.raises(ValueError,match='identity'):_verify_source(field,out)


@pytest.mark.parametrize('operation',['export','verify'])
@pytest.mark.parametrize('when',['before-audit','after-audit'])
def test_accelerator_cannot_change_after_export_identity_capture(tmp_path,engine,artifact,monkeypatch,operation,when):
    import shutil
    from winding_field import tifxyz as code,tifxyz_audit as audit_module
    monkeypatch.setenv('WINDING_TIFXYZ_ACCELERATOR',str(artifact/'accelerator.json'))
    shutil.copytree(artifact,tmp_path/'other')
    other=tmp_path/'other/accelerator.json';other.write_text(other.read_text()+' ')
    if operation=='verify':source,out=export_plane(tmp_path,audit_backend='cpu');before=tree_state(out)
    else:source=physical_source(tmp_path/'source');out=tmp_path/'out'
    original=audit_module.audit_level
    def changed(*args,**kwargs):
        if when=='before-audit':monkeypatch.setenv('WINDING_TIFXYZ_ACCELERATOR',str(other))
        result=original(*args,**kwargs)
        if when=='after-audit':monkeypatch.setenv('WINDING_TIFXYZ_ACCELERATOR',str(other))
        return result
    monkeypatch.setattr(audit_module,'audit_level',changed)
    with open_physical_field(source) as field,pytest.raises(ValueError,match='accelerator'):
        if operation=='export':code._export_source(field,out,settings(field,audit_backend='cpu'))
        else:code._verify_source(field,out)
    if operation=='export':assert not (out/'READY.json').exists()
    else:assert tree_state(out)==before


def test_pause_resume_preserves_committed_tiles_and_mutable_run_status(tmp_path, engine, monkeypatch):
    from winding_field import tifxyz as code, tifxyz_store
    source = physical_source(tmp_path/'source'); out=tmp_path/'out'; stopped=False
    original=tifxyz_store.TileStore.commit
    def commit(store, tile):
        nonlocal stopped
        result=original(store,tile)
        if store.phase=='projected': stopped=True
        return result
    monkeypatch.setattr(tifxyz_store.TileStore,'commit',commit)
    with open_physical_field(source) as field:
        config=settings(field)
        assert code._export_source(field,out,config,False,lambda:stopped)['status']=='PAUSED'
    payload=next(out.rglob('tile-0-0.bin')); before=(file_sha(payload),payload.stat().st_mtime_ns)
    assert not (out/'READY.json').exists()
    mutate_json(source/'RUN.json',lambda d:d.update(status='COMPLETE',seconds=99))
    monkeypatch.setattr(tifxyz_store.TileStore,'commit',original)
    with open_physical_field(source) as field:
        assert code._export_source(field,out,config,True,None)['status']=='READY'
    assert (file_sha(payload),payload.stat().st_mtime_ns)==before
    with open_physical_field(source) as field:
        assert code._verify_source(field,out)['status']=='TIFXYZ_VERIFIED'


@pytest.mark.parametrize('target', ['source','metadata','grid','hash','flags','guide','tile','extra'])
def test_corruption_never_verifies(tmp_path, engine, target):
    from winding_field.tifxyz import _verify_source
    source,out=export_plane(tmp_path)
    if target=='source':
        mutate_json(source/'RUN.json',lambda d:d.update(implementation_sha256='c'*64))
    elif target=='metadata':
        mutate_json(out/'level-0000/meta.json',lambda d:d.update(scale=[2,2]))
    elif target in ('grid','flags'):
        mutate_json(out/'TIFXYZ.json',lambda d:d.update(physical_acceptance=True) if target=='flags' else d['levels'][0]['grid'].update(pitch_um=2))
    elif target=='hash':
        mutate_json(out/'READY.json',lambda d:d.update(tifxyz_sha256='0'*64))
    elif target=='extra': (out/'level-0000/alien').write_text('foreign')
    else:
        path=next(out.rglob('uv.npy' if target=='guide' else 'tile-0-0.bin'))
        raw=bytearray(path.read_bytes()); raw[-1]^=1; path.write_bytes(raw)
    with open_physical_field(source) as field, pytest.raises(ValueError):
        _verify_source(field,out)


def test_geometry_failure_retains_attempts_without_ready(tmp_path, engine, monkeypatch):
    from winding_field import tifxyz as code, tifxyz_project
    source=physical_source(tmp_path/'source'); out=tmp_path/'out'
    def fail(*args,**kwargs): raise Blocked('PROJECTION_AMBIGUOUS')
    monkeypatch.setattr(tifxyz_project,'project_tile',fail)
    with open_physical_field(source) as field:
        result=code._export_source(field,out,settings(field),False,None)
    assert result['status']=='BLOCKED'
    assert len(list(out.rglob('REJECTED.json')))==3
    assert not (out/'READY.json').exists()


def test_low_resources_and_verifier_fresh_admission(tmp_path, engine, monkeypatch):
    from winding_field import tifxyz as code, tifxyz_runtime as runtime
    source,out=export_plane(tmp_path); before=tree_state(out)
    monkeypatch.setattr(runtime,'memory_report',lambda:dict(known=False,effective_headroom_bytes=0))
    with open_physical_field(source) as field:
        assert code._verify_source(field,out)['status']=='BLOCKED'
        result=code._export_source(field,tmp_path/'low',settings(field),False,None)
    assert result['status']=='BLOCKED' and not (tmp_path/'low/READY.json').exists()
    assert tree_state(out)==before


@pytest.mark.parametrize('kind',['foreign','existing','symlink'])
def test_rejects_unowned_output_without_changes(tmp_path, engine, kind):
    from winding_field.tifxyz import _export_source
    source=physical_source(tmp_path/'source');out=tmp_path/'out'
    if kind=='symlink': out.symlink_to(source,target_is_directory=True)
    else:
        out.mkdir()
        if kind=='foreign': (out/'foreign').write_text('keep')
    before=tree_state(source)
    with open_physical_field(source) as field, pytest.raises(ValueError):
        _export_source(field,out,settings(field),False,None)
    assert tree_state(source)==before


def test_real_guide_full_pipeline(tmp_path, controlled):
    pytest.importorskip('igl')
    from winding_field.tifxyz import _export_source, _verify_source
    source=physical_source(tmp_path/'source');out=tmp_path/'out'
    with open_physical_field(source) as field:
        config=settings(field,tolerance_um=3.)
        assert _export_source(field,out,config,False,None)['status']=='READY'
    with open_physical_field(source) as field:
        assert _verify_source(field,out)['status']=='TIFXYZ_VERIFIED'


def test_t4_standard_metadata_integration(tmp_path, engine):
    export_plane(tmp_path)


def rebind(out, edit):
    from winding_field.tifxyz import _inventory
    receipt=json.loads((out/'TIFXYZ.json').read_text());edit(receipt)
    receipt['work_inventory']=_inventory(out/'work',recursive=True)
    atomic_json(out/'TIFXYZ.json',receipt)
    mutate_json(out/'READY.json',lambda d:d.update(tifxyz_sha256=file_sha(out/'TIFXYZ.json')))


@pytest.mark.parametrize('forgery',['permutation','root-error','metadata','flags'])
def test_rehashed_forgery_cannot_replace_independent_certification(tmp_path,engine,forgery):
    from winding_field import tifxyz as code
    source,out=export_plane(tmp_path)
    receipt=json.loads((out/'TIFXYZ.json').read_text());item=receipt['levels'][0]
    if forgery in ('permutation','root-error'):
        folder=out/item['attempt']/'projected';path=folder/'tile-0-0.bin';raw=bytearray(path.read_bytes())
        if forgery=='permutation':
            xyz=np.frombuffer(raw,'<f4',27,48).reshape(3,3,3);xyz[:]=xyz[:,::-1].copy()
        else: np.frombuffer(raw,'<f8',9,48+13*9)[0]=.123
        path.write_bytes(raw)
        rebind(out,lambda d:d['levels'][0].update(projected=code._inventory(folder)))
    elif forgery=='metadata':
        path=out/'level-0000/meta.json';mutate_json(path,lambda d:d.update(scale=[2.,2.]))
        rebind(out,lambda d:d['levels'][0]['manifest']['files'].update({'meta.json':code._record(path)}))
    else: rebind(out,lambda d:d.update(native_face_identity=True))
    match='projection/root-error' if forgery in ('permutation','root-error') else None
    with open_physical_field(source) as field,pytest.raises(ValueError,match=match): code._verify_source(field,out)


def test_pause_after_level_publication_rechecks_artifacts_before_ready(tmp_path,engine,monkeypatch):
    from winding_field import tifxyz as code
    source=physical_source(tmp_path/'source');out=tmp_path/'out'
    with open_physical_field(source) as field:
        config=settings(field)
        assert code._export_source(field,out,config,False,lambda:(out/'level-0000').exists())['status']=='PAUSED'
    mutate_json(out/'level-0000/meta.json',lambda d:d.update(scale=[17,17]))
    with open_physical_field(source) as field,pytest.raises(ValueError):
        code._export_source(field,out,config,True,None)
    assert not (out/'READY.json').exists()


def test_resume_rejects_altered_coarse_arrays(tmp_path,engine):
    from winding_field import tifxyz as code
    source=physical_source(tmp_path/'source');out=tmp_path/'out'
    with open_physical_field(source) as field:
        config=settings(field)
        assert code._export_source(field,out,config,False,lambda:any(out.rglob('tile-0-0.bin')))['status']=='PAUSED'
    path=next(out.rglob('values.npy'));raw=bytearray(path.read_bytes());raw[-1]^=1;path.write_bytes(raw)
    with open_physical_field(source) as field,pytest.raises(ValueError):code._export_source(field,out,config,True,None)
    assert not (out/'READY.json').exists()


def test_publication_resumes_after_interrupted_tiff_temp(tmp_path,engine,monkeypatch):
    from winding_field import tifxyz as code, tifxyz_store
    from winding_field.tifxyz_contract import Paused
    source=physical_source(tmp_path/'source');out=tmp_path/'out';original=tifxyz_store.write_segment
    def interrupted(*args):
        result=original(*args)
        (args[3]/'.x.tif-interrupted').write_bytes(b'partial encoder bytes')
        raise Paused()
    monkeypatch.setattr(tifxyz_store,'write_segment',interrupted)
    with open_physical_field(source) as field:
        config=settings(field)
        assert code._export_source(field,out,config,False,None)['status']=='PAUSED'
    monkeypatch.setattr(tifxyz_store,'write_segment',original)
    with open_physical_field(source) as field:
        assert code._export_source(field,out,config,True,None)['status']=='READY'
    assert list(out.rglob('.x.tif-interrupted'))


def test_pitch_refinement_and_every_retained_attempt_are_recertified(tmp_path,engine):
    from winding_field import tifxyz as code
    source,out=export_plane(tmp_path,pitch_um=2.)
    receipt=json.loads((out/'TIFXYZ.json').read_text())
    assert receipt['levels'][0]['grid']['pitch_um']==1.
    with open_physical_field(source) as field:assert code._verify_source(field,out)['status']=='TIFXYZ_VERIFIED'
    rejected=next(out.rglob('pitch-0000/REJECTED.json')).parent
    path=rejected/'projected/tile-0-0.bin';raw=bytearray(path.read_bytes())
    np.frombuffer(raw,'<f8',4,48+13*4)[0]=.123;path.write_bytes(raw)
    rebind(out,lambda d:None)
    with open_physical_field(source) as field,pytest.raises(ValueError,match='projection/root-error'):
        code._verify_source(field,out)


def test_ready_is_last_and_failed_publication_remains_resumable(tmp_path,engine,monkeypatch):
    from winding_field import tifxyz as code
    source=physical_source(tmp_path/'source');out=tmp_path/'out';writes=[];original=code.atomic_json
    def write(path,value):
        writes.append(path.name)
        if path.name=='READY.json':raise OSError('injected fsync/write failure')
        return original(path,value)
    monkeypatch.setattr(code,'atomic_json',write)
    with open_physical_field(source) as field:
        config=settings(field)
        with pytest.raises(OSError):code._export_source(field,out,config,False,None)
    assert not (out/'READY.json').exists()
    monkeypatch.setattr(code,'atomic_json',lambda path,value:(writes.append(path.name),original(path,value))[1])
    with open_physical_field(source) as field:assert code._export_source(field,out,config,True,None)['status']=='READY'
    assert writes[-1]=='READY.json'


def test_forged_passing_audit_receipt_is_not_trusted(tmp_path,engine):
    from winding_field import tifxyz as code
    source,out=export_plane(tmp_path);path=out/'level-0000/VERIFIED.json'
    mutate_json(path,lambda d:d['audit']['counters'].update(nodes=999))
    rebind(out,lambda d:d['levels'][0]['manifest']['files'].update({'VERIFIED.json':code._record(path)}))
    with open_physical_field(source) as field,pytest.raises(ValueError,match='audit'):
        code._verify_source(field,out)


def test_source_code_change_during_export_prevents_ready(tmp_path,engine,monkeypatch):
    from winding_field import tifxyz as code
    source=physical_source(tmp_path/'source');out=tmp_path/'out';original=code._code_identity
    def identity():
        value=original()
        if (out/'level-0000').exists():value['files']['tifxyz.py']='f'*64
        return value
    monkeypatch.setattr(code,'_code_identity',identity)
    with open_physical_field(source) as field,pytest.raises(ValueError,match='implementation'):
        code._export_source(field,out,settings(field),False,None)
    assert not (out/'READY.json').exists()


def test_admission_counts_only_owned_scope_not_unrelated_parent_files(tmp_path,engine):
    from winding_field import tifxyz as code
    (tmp_path/'unrelated-link').symlink_to(tmp_path,target_is_directory=True)
    source,out=export_plane(tmp_path)
    with open_physical_field(source) as field:assert code._verify_source(field,out)['status']=='TIFXYZ_VERIFIED'


def test_verification_charges_accepted_output_against_fresh_scratch_budget(tmp_path,engine,monkeypatch):
    from winding_field import tifxyz as code
    source,out=export_plane(tmp_path);original=code._space
    def space(root,settings,pending=0):
        if root==out:return settings.scratch_bytes-(1<<20)
        return original(root,settings,pending)
    monkeypatch.setattr(code,'_space',space)
    before=tree_state(out)
    with open_physical_field(source) as field:
        result=code._verify_source(field,out)
    assert result['status']=='BLOCKED' and result['reasons'][0]['code']=='SCRATCH_BUDGET'
    assert tree_state(out)==before


def test_killed_verification_scratch_is_charged_on_next_attempt(tmp_path, engine):
    import os
    from pathlib import Path
    from winding_field import tifxyz as code
    source, out = export_plane(tmp_path)
    before = tree_state(out)
    child = os.fork()
    if child==0:
        original = code._build_coarse
        def crash(field, stride, destination, stop, config, scope):
            original(field, stride, destination, stop, config, scope)
            # Sparse length is charged as scratch without allocating hundreds of MiB.
            with (Path(scope)/'interrupted-work').open('wb') as stream:
                stream.truncate(384<<20)
            os._exit(73)
        code._build_coarse = crash
        try:
            with open_physical_field(source) as field: code._verify_source(field, out)
        finally: os._exit(74)
    _, status = os.waitpid(child, 0)
    assert os.waitstatus_to_exitcode(status)==73
    abandoned = list(tmp_path.glob('.out-verify-*'))
    assert len(abandoned)==1 and (abandoned[0]/'interrupted-work').stat().st_size==384<<20
    with open_physical_field(source) as field:
        result = code._verify_source(field, out)
    assert result['status']=='BLOCKED' and result['reasons'][0]['code']=='SCRATCH_BUDGET'
    assert abandoned[0].exists() and tree_state(out)==before


def test_verification_serializes_shared_scratch_budget(tmp_path, engine):
    import fcntl
    from winding_field.tifxyz import _verify_source
    source, out = export_plane(tmp_path)
    with (out/'.lock').open('rb') as lock, open_physical_field(source) as field:
        fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError): _verify_source(field, out)


def test_roi_absolute_frame_and_nonunit_scale_survive_verification(tmp_path,engine):
    from winding_field import tifxyz as code
    from winding_field.tifxyz_store import iter_tiff_tiles
    source=physical_source(tmp_path/'source',shape=(5,5,5),origin=(16,24,32),spacing=(2,2,2))
    roi=dict(lo_zyx=[0,1,1],hi_zyx=[3,4,4]);out=tmp_path/'out'
    frame=dict(id='explicit-volume',origin_xyz_um=[0,0,0],voxel_um=4.,shape_zyx=[100]*3,source_identity_sha256='d'*64)
    with open_physical_field(source,roi=roi) as field:
        config=settings(field,roi=roi,output_frame=frame)
        assert code._export_source(field,out,config,False,None)['status']=='READY'
    meta=json.loads((out/'level-0000/meta.json').read_text())
    assert meta['scale']==[2.,2.] and meta['uv_min_um']==[18.,26.]
    x=next(iter_tiff_tiles(out/'level-0000/x.tif',[3,3],9))[1]
    np.testing.assert_array_equal(x,np.tile([4.5,5.,5.5],(3,1)))
    with open_physical_field(source,roi=roi) as field:assert code._verify_source(field,out)['status']=='TIFXYZ_VERIFIED'


@pytest.mark.parametrize('value',[None,0,True,float('nan')])
def test_unresolved_or_invalid_effective_pitch_rejected_before_work(tmp_path,controlled,value):
    from winding_field import tifxyz as code
    source=physical_source(tmp_path/'source');out=tmp_path/'out'
    with open_physical_field(source) as field,pytest.raises(ValueError):
        code._export_source(field,out,replace(settings(field),pitch_um=value),False,None)
    assert not out.exists()


@pytest.mark.skipif(__import__('sys').platform!='linux',reason='Linux supervised end-to-end integration')
def test_public_export_and_separate_verifier_process(tmp_path):
    pytest.importorskip('igl')
    from winding_field.tifxyz import export_field,verify_export
    source=physical_source(tmp_path/'source');out=tmp_path/'out'
    result=export_field(source,out,dict(levels=[0.],tolerance_um=3.,reserve_disk_gib=0))
    assert result['status']=='READY',result
    before,source_before=tree_state(out),tree_state(source)
    verified=verify_export(out)
    assert verified['status']=='TIFXYZ_VERIFIED',verified
    assert result['supervision']['pid']!=verified['supervision']['pid']
    assert tree_state(out)==before and tree_state(source)==source_before


def test_failure_after_ready_rename_withdraws_acceptance(tmp_path,engine,monkeypatch):
    from winding_field import tifxyz as code
    source=physical_source(tmp_path/'source');out=tmp_path/'out';original=code._sync
    def sync(path):
        if path==out and (out/'READY.json').exists():raise OSError('injected directory fsync failure')
        return original(path)
    monkeypatch.setattr(code,'_sync',sync)
    with open_physical_field(source) as field,pytest.raises(OSError):
        code._export_source(field,out,settings(field),False,None)
    assert not (out/'READY.json').exists()


class RecordingSource:
    """Keep the real immutable reader, but forbid checkpoint-wide payload sweeps."""
    def __init__(self, source):
        self.source=source;self.validations=[];self.reads=[]

    def __getattr__(self, name):return getattr(self.source,name)

    def validate(self, *, hashes=False):
        self.validations.append(hashes)
        if hashes:raise AssertionError('forbidden hashes=True includes unrelated checkpoint chunks')
        return self.source.validate(hashes=hashes)

    def read(self, lo, hi):
        self.reads.append((tuple(lo),tuple(hi)))
        return self.source.read(lo,hi)


def roi_source(root):
    source=physical_source(root,shape=(3,3,9),chunk=3)
    chunks=json.loads((source/'field/CHECKPOINT.json').read_text())['chunks']
    paths={key:source/'field'/record['path'] for key,record in chunks.items()}
    return source,dict(lo_zyx=[0,0,3],hi_zyx=[3,3,6]),paths


def corrupt_payload(path):
    raw=bytearray(path.read_bytes());raw[-1]^=1;path.write_bytes(raw)


@pytest.mark.parametrize('resume',[False,True])
def test_roi_export_resume_and_verify_never_sweep_unrelated_payloads(tmp_path,engine,monkeypatch,resume):
    from winding_field import tifxyz as code
    from winding_field._core.surface_sources import _File
    source,roi,paths=roi_source(tmp_path/'source');out=tmp_path/'out'
    if resume:
        with open_physical_field(source,roi=roi) as field:
            result=code._export_source(field,out,settings(field,roi=roi),False,lambda:(out/'level-0000').exists())
            assert result['status']=='PAUSED'
    # These same-sized corrupt chunks predate the new reader and are outside its ROI.
    for key in ('0-0-0','0-0-2'):corrupt_payload(paths[key])
    payload_reads=[];original=_File.read
    def read(file):
        if file.path.suffix=='.npz':
            payload_reads.append(file.path)
            assert file.path==paths['0-0-1'], 'out-of-ROI payload bulk read'
        return original(file)
    monkeypatch.setattr(_File,'read',read)
    expected_scope=dict(metadata_scope='global-bound-source',payload_hash_scope='consumed-roi-payloads',
                        coverage_roi=roi,all_checkpoint_payloads_verified=False)
    for operation in ('export','verify'):
        payload_reads.clear()
        with open_physical_field(source,roi=roi) as field:
            field.source=reader=RecordingSource(field.source)
            result=(code._export_source(field,out,settings(field,roi=roi),resume,None) if operation=='export'
                    else code._verify_source(field,out))
            assert result['status']==('READY' if operation=='export' else 'TIFXYZ_VERIFIED'),result
            assert reader.validations and not any(reader.validations)
            assert reader.reads and all(all(a>=l and b<=h for a,b,l,h in zip(lo,hi,roi['lo_zyx'],roi['hi_zyx']))
                                       for lo,hi in reader.reads)
        assert payload_reads and set(payload_reads)=={paths['0-0-1']}
        assert result['source_verification']==expected_scope
    for name in ('EXPORT.json','TIFXYZ.json'):
        assert json.loads((out/name).read_text())['source_verification']==expected_scope


@pytest.mark.parametrize('phase',['export','verify','resume-projected','resume-published'])
def test_changed_roi_payload_is_rejected_with_a_fresh_reader(tmp_path,engine,phase):
    from winding_field import tifxyz as code
    source,roi,paths=roi_source(tmp_path/'source');out=tmp_path/'out'
    if phase!='export':
        stop=(lambda:any(out.rglob('tile-0-0.bin'))) if phase=='resume-projected' else (
             (lambda:(out/'level-0000').exists()) if phase=='resume-published' else None)
        with open_physical_field(source,roi=roi) as field:
            result=code._export_source(field,out,settings(field,roi=roi),False,stop)
            assert result['status']==('READY' if phase=='verify' else 'PAUSED')
    corrupt_payload(paths['0-0-1'])
    before=tree_state(out) if phase=='verify' else None
    with open_physical_field(source,roi=roi) as field:
        field.source=RecordingSource(field.source)
        with pytest.raises(ValueError,match='field source hash differs'):
            if phase=='verify':code._verify_source(field,out)
            else:code._export_source(field,out,settings(field,roi=roi),phase!='export',None)
    if phase=='verify':assert tree_state(out)==before
    else:assert not (out/'READY.json').exists()


def test_roi_payload_mutation_on_projection_cache_hit_remains_fail_closed(tmp_path,engine,monkeypatch):
    from winding_field import tifxyz as code,tifxyz_guide
    source,roi,paths=roi_source(tmp_path/'source');out=tmp_path/'out';cache_hits=[]
    with open_physical_field(source,roi=roi) as field:
        reader=field.source;load=reader._load;build=tifxyz_guide.build_guide
        field.source=RecordingSource(reader)
        def build_guide(*args,**kwargs):
            guide=build(*args,**kwargs)
            assert '0-0-1' in reader._cache  # Coarse construction hashed and warmed the real reader.
            corrupt_payload(paths['0-0-1'])
            return guide
        def load_chunk(label,shape):
            if label in reader._cache:cache_hits.append(label)
            return load(label,shape)
        monkeypatch.setattr(tifxyz_guide,'build_guide',build_guide)
        monkeypatch.setattr(reader,'_load',load_chunk)
        with pytest.raises(ValueError,match='field source changed'):
            code._export_source(field,out,settings(field,roi=roi),False,None)
    assert cache_hits==['0-0-1']
    assert not (out/'READY.json').exists()


def test_scope_cannot_be_promoted_to_all_checkpoint_payloads(tmp_path,engine):
    from winding_field import tifxyz as code
    source,out=export_plane(tmp_path)
    rebind(out,lambda d:d.update(source_verification=dict(all_checkpoint_payloads_verified=True)))
    with open_physical_field(source) as field,pytest.raises(ValueError,match='source verification scope'):
        code._verify_source(field,out)


def test_reused_roi_revalidation_is_bounded_and_stoppable(tmp_path):
    from winding_field import tifxyz as code
    from winding_field.tifxyz_contract import Paused
    source=physical_source(tmp_path/'source',shape=(3,3,135),chunk=3)
    roi=dict(lo_zyx=[0,0,3],hi_zyx=[3,3,133])
    with open_physical_field(source,roi=roi) as field:
        field.source=reader=RecordingSource(field.source)
        code._validate_roi_payloads(field,None)
        assert reader.reads==[((0,0,3),(3,3,67)),((0,0,67),(3,3,131)),((0,0,131),(3,3,133))]
        reader.reads.clear()
        with pytest.raises(Paused):code._validate_roi_payloads(field,lambda:bool(reader.reads))
        assert reader.reads==[((0,0,3),(3,3,67))]


def reseal_rotated_tile(folder):
    import hashlib
    import sqlite3
    path=folder/'tile-0-0.bin';raw=bytearray(path.read_bytes())
    xyz=np.frombuffer(raw,'<f4',27,48).reshape(3,3,3)
    xyz[:]=xyz[::-1,::-1].copy()  # Same plane, coverage and orientation; wrong UV correspondence.
    path.write_bytes(raw);sha=hashlib.sha256(raw).hexdigest()
    receipt=dict(tile_id=[0,0],path=path.name,bytes=len(raw),sha256=sha)
    with sqlite3.connect(folder/'store.sqlite') as db:
        db.execute('UPDATE tiles SET sha256=?,receipt_sha256=? WHERE ty=0 AND tx=0',(sha,digest(receipt)))


def replace_plane_tiff(path, value):
    import tifffile
    tile=np.full((256,256),-1,np.float32);tile[:3,:3]=value
    with tifffile.TiffWriter(path,byteorder='<') as writer:
        writer.write(iter([tile]),shape=(3,3),dtype='float32',tile=(256,256),compression='deflate',
                     photometric='minisblack',metadata=None,software=False,maxworkers=1)


def test_resume_recertifies_resealed_projected_uv_permutation(tmp_path,engine):
    from winding_field import tifxyz as code
    source=physical_source(tmp_path/'source');out=tmp_path/'out'
    with open_physical_field(source) as field:
        config=settings(field)
        assert code._export_source(field,out,config,False,lambda:any(out.rglob('tile-0-0.bin')))['status']=='PAUSED'
    projected=next(out.rglob('projected/STORE.json')).parent
    reseal_rotated_tile(projected)
    with open_physical_field(source) as field,pytest.raises(ValueError,match='projection/root-error certificate'):
        code._export_source(field,out,config,True,None)
    assert not (out/'READY.json').exists()


def test_resume_recertifies_resealed_published_tiff_readonly(tmp_path,engine):
    from winding_field import tifxyz as code
    source=physical_source(tmp_path/'source');out=tmp_path/'out'
    with open_physical_field(source) as field:
        config=settings(field)
        assert code._export_source(field,out,config,False,lambda:(out/'level-0000').exists())['status']=='PAUSED'
    path=out/'level-0000/z.tif';replace_plane_tiff(path,1.75)
    accepted=next(out.rglob('ACCEPTED.json'))
    mutate_json(accepted,lambda d:d['manifest']['files'].update({'z.tif':code._record(path)}))
    item=json.loads(accepted.read_text());delivered=out/item['attempt']/'delivered';before=tree_state(delivered)
    with open_physical_field(source) as field,pytest.raises(ValueError,match='delivered TIFF projection bytes'):
        code._export_source(field,out,config,True,None)
    assert not (out/'READY.json').exists()
    assert tree_state(delivered)==before


def test_verifier_rechecks_published_files_after_audit(tmp_path,engine,monkeypatch):
    from winding_field import tifxyz as code,tifxyz_audit
    source,out=export_plane(tmp_path);original=tifxyz_audit.audit_level;mutated=[]
    def audit(*args,**kwargs):
        result=original(*args,**kwargs)
        assert result['status']=='PASS'
        corrupt_payload(out/'level-0000/z.tif');mutated.append(True)
        return result
    monkeypatch.setattr(tifxyz_audit,'audit_level',audit)
    with open_physical_field(source) as field,pytest.raises(ValueError,match='segment artifact hash'):
        code._verify_source(field,out)
    assert mutated


def test_export_rechecks_earlier_level_after_later_audit(tmp_path,engine,monkeypatch):
    from winding_field import tifxyz as code,tifxyz_audit
    source=physical_source(tmp_path/'source');out=tmp_path/'out';original=tifxyz_audit.audit_level;mutated=[]
    def audit(*args,**kwargs):
        result=original(*args,**kwargs)
        if (out/'level-0000/z.tif').exists():
            assert result['status']=='PASS'
            corrupt_payload(out/'level-0000/z.tif');mutated.append(True)
        return result
    monkeypatch.setattr(tifxyz_audit,'audit_level',audit)
    with open_physical_field(source) as field,pytest.raises(ValueError,match='segment artifact hash'):
        code._export_source(field,out,settings(field,levels=[0.,1.]),False,None)
    assert mutated and not (out/'READY.json').exists()


def test_interrupted_guide_is_retained_outside_completed_namespace(tmp_path,engine,monkeypatch):
    from winding_field import tifxyz as code,tifxyz_store
    from winding_field.tifxyz_contract import Paused
    source=physical_source(tmp_path/'source');out=tmp_path/'out';original=tifxyz_store._write_bytes
    def interrupted(path,data):
        if path.name=='world.npy':
            (path.parent/'.world.npy-interrupted').write_bytes(b'partial NPY')
            raise Paused()
        return original(path,data)
    monkeypatch.setattr(tifxyz_store,'_write_bytes',interrupted)
    with open_physical_field(source) as field:
        config=settings(field)
        assert code._export_source(field,out,config,False,None)['status']=='PAUSED'
    monkeypatch.setattr(tifxyz_store,'_write_bytes',original)
    with open_physical_field(source) as field:assert code._export_source(field,out,config,True,None)['status']=='READY'
    retained=list(out.rglob('.world.npy-interrupted'))
    assert len(retained)==1 and retained[0].parent.name.startswith('guide-interrupted-')
    with open_physical_field(source) as field:assert code._verify_source(field,out)['status']=='TIFXYZ_VERIFIED'


def test_newly_saved_guide_is_loaded_before_acceptance(tmp_path,engine,monkeypatch):
    from winding_field import tifxyz as code
    source=physical_source(tmp_path/'source');out=tmp_path/'out';original=code._save_guide
    def corrupted(guide,root):
        result=original(guide,root)
        corrupt_payload(root/'world.npy')
        return result
    monkeypatch.setattr(code,'_save_guide',corrupted)
    with open_physical_field(source) as field,pytest.raises(ValueError,match='guide array identity'):
        code._export_source(field,out,settings(field),False,None)
    assert not (out/'READY.json').exists()


def test_coarse_budget_api_receives_remaining_owned_scratch_on_every_path(tmp_path,engine,monkeypatch):
    from winding_field import tifxyz as code,tifxyz_field
    source=physical_source(tmp_path/'source');out=tmp_path/'out';original=tifxyz_field.build_coarse;budgets=[]
    # Exercise the coordinator's optional keyword without changing the current producer.
    def budgeted(field,stride,destination,stop=None,*,budget=None):
        assert budget is not None, 'coarse writes need the remaining scratch allowance'
        budgets.append(budget)
        return original(field,stride,destination,stop)
    monkeypatch.setattr(tifxyz_field,'build_coarse',budgeted)
    with open_physical_field(source) as field:
        config=settings(field,pitch_um=2.)
        assert code._export_source(field,out,config,False,None)['status']=='READY'
    accepted_bytes=code._space(out,config)
    with open_physical_field(source) as field:assert code._verify_source(field,out)['status']=='TIFXYZ_VERIFIED'
    assert len(budgets)==3  # Export, fresh accepted reconstruction, then retained rejected work.
    assert budgets[0].scratch_bytes<config.scratch_bytes
    assert budgets[2].scratch_bytes<budgets[1].scratch_bytes<config.scratch_bytes-accepted_bytes
    assert all(b.reserve_disk_bytes==config.reserve_disk_bytes for b in budgets)


def test_boundary_precheck_receipt_does_not_replace_projection_or_audit(tmp_path,engine):
    source,out=export_plane(tmp_path)
    receipt=json.loads(next(out.rglob('PRECHECK.json')).read_text())
    assert receipt['result']['status']=='NO_REJECTION_WITNESS'
    assert receipt['result']['evaluated_nodes']==9
    assert receipt['binding']['export_sha256']==digest(json.loads((out/'EXPORT.json').read_text()))
    assert next(out.rglob('projected/tile-0-0.bin')).exists()
    verified=json.loads((out/'level-0000/VERIFIED.json').read_text())
    assert verified['audit']['status']=='PASS' and verified['audit']['source_scan_complete']
