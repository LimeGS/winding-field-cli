import json

import numpy as np
import pytest


def test_base_distribution_installs_pinned_slim_without_tiff_extra():
    from importlib.metadata import requires
    requirements=requires('winding-field-cli')
    assert 'libigl==2.6.3' in requirements
    assert not any(r.startswith('libigl') and ';' in r for r in requirements)
    assert any(r.startswith('tifffile') and 'extra == "tifxyz"' in r for r in requirements)
    assert not any(r.startswith('tifffile') and ';' not in r for r in requirements)


def test_slim_reduces_real_symmetric_dirichlet_energy_without_changing_source():
    pytest.importorskip('igl')
    from winding_field.native_geometry import _slim,_validate
    vertices=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[1.,1.,0.]])
    faces=np.array([[0,1,2],[1,3,2]])
    seed=vertices[:,:2]*[2.,.5]
    before=[a.copy() for a in (vertices,faces,seed)]
    result,quality=_slim(vertices,faces,seed)
    # Hand-derived inverse XY edge matrices for these two unit-square triangles.
    inverse=np.array([[[1.,0.],[0.,1.]],[[1.,1.],[-1.,0.]]])
    def energy(uv):
        tri=uv[faces]
        jac=np.stack([tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]],axis=-1)@inverse
        singular=np.linalg.svd(jac,compute_uv=False)
        return np.sum(singular**2+singular**-2)
    assert energy(result)<energy(seed)*.8
    _validate(vertices,faces,result)
    assert quality['method']=='SLIM'
    assert quality['iterations']==10
    for actual,expected in zip((vertices,faces,seed),before):
        np.testing.assert_array_equal(actual,expected)


def test_slim_native_pipeline_records_real_optimization_and_reuses_completed_charts(tmp_path):
    pytest.importorskip('igl')
    from test_native_runtime import field_case
    from winding_field.native_pipeline import work
    from winding_field.native_store import Journal
    run,config=field_case(tmp_path,levels=(.5,))
    config['flattening_method']='slim'
    assert work(run,config,until='flatten')['status']=='PAUSED'
    identity=json.loads((run/'surfaces/NATIVE.json').read_bytes())
    assert identity['settings']['parameterizer']['solver']=='libigl-slim'
    root=run/'surfaces/level-0000'
    with Journal(root,json.loads((root/'IDENTITY.json').read_bytes())) as store:
        qualities=[json.loads(q)['local'] for (q,) in store.db.execute('SELECT quality FROM charts')]
        assert qualities and all(q['method']=='SLIM' and q['iterations']==10 for q in qualities)
    before={p:p.read_bytes() for p in (root/'charts').glob('*.npz')}
    assert work(run,config)['status']=='COMPLETE'
    assert work(run,config)['status']=='COMPLETE'
    assert all(p.read_bytes()==raw for p,raw in before.items())
    with pytest.raises(ValueError,match='identity changed'):
        work(run,dict(config,flattening_method='default'))


@pytest.mark.parametrize('method',[None,'default'])
def test_historical_native_configs_keep_old_identity_and_parameterization(tmp_path,monkeypatch,method):
    import winding_field.native_geometry as geometry
    from test_native_runtime import field_case
    from winding_field.native_pipeline import work,_settings
    run,config=field_case(tmp_path,levels=(.5,))
    if method:config['flattening_method']=method
    before=config.copy()
    settings=_settings(config)
    assert settings==dict(coverage_target=1.,surface_time_limit_s=30.,chart_faces=32,mesh_tile=4,
                         export_tifxyz=False,parameterizer=geometry.solver_identity())
    monkeypatch.setattr(geometry,'_slim',lambda *a,**kw:pytest.fail('historical chart invoked SLIM'),raising=False)
    assert work(run,config,until='flatten')['status']=='PAUSED'
    assert work(run,config)['status']=='COMPLETE'
    assert config==before


@pytest.mark.parametrize('broken',['fold','overlap','stretch','nonfinite'])
def test_slim_never_optimizes_invalid_initialization(broken,monkeypatch):
    import winding_field.native_geometry as geometry
    vertices=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
    faces=np.array([[0,1,2]]);uv=vertices[:,:2].copy()
    if broken=='fold':uv[:,1]*=-1
    elif broken=='overlap':faces=np.vstack([faces,faces])
    elif broken=='stretch':uv*=10
    else:uv[0,0]=np.nan
    monkeypatch.setattr(geometry,'_solver',lambda:pytest.fail('invalid seed reached native solver'))
    with pytest.raises(ValueError):geometry._slim(vertices,faces,uv)


def test_slim_checks_stop_between_bounded_iterations():
    pytest.importorskip('igl')
    from winding_field.native_geometry import _slim
    from winding_field.native_store import Paused
    v=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
    calls=[]
    def check():
        calls.append(True)
        if len(calls)==3:raise Paused('STOP')
    with pytest.raises(Paused,match='STOP'):_slim(v,np.array([[0,1,2]]),v[:,:2]*[2.,.5],check)
    assert len(calls)==3


@pytest.mark.parametrize('shape',['curved-strip','closed-octahedron'])
def test_slim_keeps_source_ids_after_lscm_or_recursive_chart_splitting(shape):
    pytest.importorskip('igl')
    from winding_field.native_geometry import _pieces,_validate
    if shape=='curved-strip':
        from test_meshes import curved_strip
        vertices,faces=curved_strip()
    else:
        vertices=np.array([[1.,0.,0.],[-1.,0.,0.],[0.,1.,0.],[0.,-1.,0.],[0.,0.,1.],[0.,0.,-1.]])
        faces=np.array([[4,0,2],[4,1,3],[5,1,2],[5,0,3],[4,2,1],[4,3,0],[5,2,0],[5,3,1]])
    ids=np.arange(len(faces))+10;source_ids=np.arange(len(vertices))+100
    pieces=list(_pieces(vertices,faces,ids,source_ids,minimum=4,method='slim'))
    seen=[]
    for v,f,uv,face_ids,vertex_ids,quality in pieces:
        assert len(uv) and quality['method']=='SLIM'
        if shape=='curved-strip':assert quality['initialization']=='LSCM'
        _validate(v,f,uv);seen.extend(face_ids.tolist())
        np.testing.assert_array_equal(v[f],vertices[faces[face_ids-10]])
        np.testing.assert_array_equal(vertex_ids[f],source_ids[faces[face_ids-10]])
    assert sorted(seen)==ids.tolist()


def test_slim_rejects_collapsed_optimizer_output_instead_of_reusing_seed(monkeypatch):
    igl=pytest.importorskip('igl')
    from winding_field.native_geometry import _pieces
    v=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
    monkeypatch.setattr(igl,'slim_solve',lambda data,n:np.zeros((3,2)))
    [(vertices,faces,uv,ids,source_ids,quality)]=list(_pieces(v,np.array([[0,1,2]]),np.array([1]),np.arange(3),method='slim'))
    assert uv.shape==(0,2)
    assert quality['error']=='inverted or overlapping assembled chart'
