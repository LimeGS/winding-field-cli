import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def tool(name):
    path=Path(__file__).resolve().parents[1]/'tools'/(name+'.py')
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('has_root',[False,True])
def test_independent_exact_check_distinguishes_absence_from_a_face_root(tmp_path,has_root):
    pytest.importorskip('sympy');code=tool('validate_tifxyz_witness')
    segments=code.exact_ray_segments([0,.5,.5],[1,0,0],[0,0,0],[1,1,1],[2,2,3],2.)
    assert [s['cell'] for s in segments]==[[0,0,0],[0,0,1]]
    for row in segments:
        x=row['cell'][2];values=np.indices((2,2,2))[2]+x+(-1 if has_root else 1)
        row['values']=values.tolist()
    document=dict(level=0.,segments=segments,field_binding={'synthetic':True},pixel_yx=[0,0],candidate=0,stride=2,
        origin_um=[0,.5,.5],direction_unit=[1.,0.,0.],field_origin_um=[0.,0.,0.],spacing_um=[1.,1.,1.],shape_zyx=[2,2,3],radius_um=2.)
    path=tmp_path/'witness.json';path.write_text(json.dumps(document));result=code.validate(path)
    assert result['status']==('ABSENCE_NOT_CONFIRMED' if has_root else 'EXACT_ABSENCE_IN_SUPPLIED_DATA')
    assert result['source_bound'] is False
    assert result['roots_per_segment']==([1,1] if has_root else [0,0])


def test_adversarial_boundary_queries_are_visible_and_need_no_field_reads():
    pytest.importorskip('igl');code=tool('bench_direct_tifxyz')
    f=SimpleNamespace(origin_um=np.zeros(3),spacing_um=np.ones(3),shape=(3,3,3))
    guide=dict(world=np.array([[1.,0.,0.],[1.,2.,0.],[1.,2.,2.],[1.,0.,2.]]),faces=np.array([[0,1,2],[0,2,3]]),
        uv=np.array([[0.,0.],[2.,0.],[2.,2.],[0.,2.]]),normals=np.tile([1.,1.,0.],(4,1)))
    grid=dict(shape_yx=[3,3],uv_min_um=[0.,0.],pitch_um=1.)
    selected=code.boundary_queries(f,guide,grid,limit=4)
    assert [q['pixel_yx'] for q in selected]==[[0,0],[0,2],[1,0],[1,2]]
    assert all(q['exit_distance_um']==0 for q in selected)


@pytest.mark.parametrize('damage',['truncate','cell','interval'])
def test_independent_checker_requires_the_complete_claimed_ray(tmp_path,damage):
    pytest.importorskip('sympy');code=tool('validate_tifxyz_witness')
    segments=code.exact_ray_segments([0,.5,.5],[1,0,0],[0,0,0],[1,1,1],[2,2,3],2.)
    for row in segments:
        row['values']=(np.indices((2,2,2))[2]+row['cell'][2]-1.5).tolist()
    if damage=='truncate':segments=segments[:1]
    elif damage=='cell':segments[0]['cell']=[0,0,1]
    else:segments[0]['t1']=[1,2]
    document=dict(level=0.,segments=segments,field_binding={'synthetic':True},pixel_yx=[0,0],candidate=0,stride=2,
        origin_um=[0,.5,.5],direction_unit=[1.,0.,0.],field_origin_um=[0.,0.,0.],spacing_um=[1.,1.,1.],shape_zyx=[2,2,3],radius_um=2.)
    path=tmp_path/'witness.json';path.write_text(json.dumps(document))
    with pytest.raises(ValueError,match='ray|segment'):code.validate(path)


@pytest.mark.parametrize('kind',['multiple','cusp'])
def test_exact_checker_counts_distinct_roots_and_boundary_cusps(tmp_path,kind):
    pytest.importorskip('sympy');code=tool('validate_tifxyz_witness')
    origin=[0.,0.,0.] if kind=='multiple' else [0.,.5,.5]
    direction=(np.ones(3)/np.sqrt(3)).tolist() if kind=='multiple' else [1.,0.,0.]
    shape=[2,2,2] if kind=='multiple' else [2,2,3]
    segments=code.exact_ray_segments(origin,direction,[0.]*3,[1.]*3,shape,2.)
    z,y,x=np.indices(tuple(shape),dtype=float)
    values=(x-.25)*(y-.5)*(z-.75) if kind=='multiple' else abs(x-1)
    for row in segments:row['values']=values[tuple(slice(c,c+2) for c in row['cell'])].tolist()
    document=dict(level=0.,segments=segments,field_binding={'synthetic':True},pixel_yx=[0,0],candidate=0,stride=2,
        origin_um=origin,direction_unit=direction,field_origin_um=[0.]*3,spacing_um=[1.]*3,shape_zyx=shape,radius_um=2.)
    path=tmp_path/'witness.json';path.write_text(json.dumps(document));result=code.validate(path)
    assert result['distinct_roots']==(3 if kind=='multiple' else 1)
    assert result['status']==('EXACT_MULTIPLE_ROOTS_IN_SUPPLIED_DATA' if kind=='multiple' else 'EXACT_NONTRANSVERSE_IN_SUPPLIED_DATA')


def test_native_residual_ranks_a_ray_that_would_exit_the_roi():
    from test_tifxyz_project import field
    code=tool('bench_direct_tifxyz');f=field(np.indices((5,5,5))[2]-1.5)
    queries=[dict(pixel_yx=[1,1],origin_um=[2.,2.,2.],direction=[1.,1.,0.]),
             dict(pixel_yx=[0,0],origin_um=[2.,0.,2.],direction=[1.,1.,0.])]
    ranked=code.rank_native_exit(f,queries,level=0.,radius=10.)
    assert ranked[0]['pixel_yx']==[0,0]
    assert ranked[0]['linearized_exit_violation_um']>0
    assert ranked[1]['linearized_exit_violation_um']<0
