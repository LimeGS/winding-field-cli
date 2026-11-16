"""Large strides must stay calibrated, sparse and integrity-bound, not enable inference."""
import io
import itertools
import json
import sqlite3
from contextlib import closing

import numpy as np
import pytest

from ct_fixture import write_ct
from test_lasagna_input import forbid_models
from test_lasagna_stream import http_root
from test_splines import upstream_curves
from winding_field import workflow
from winding_field.artifacts import Tiles
from winding_field.config import validate_config
from winding_field.io import Volume
from winding_field.util import atomic_json, file_sha


def sparse_array(root, shape, axes, value, *, zero=None):
    """Only requested chunks exist: dense-envelope reads fail on missing chunks."""
    write_ct(root, np.zeros((1, 1, 1), np.uint8), chunk=4)
    metadata=json.loads((root/'0/.zarray').read_bytes())
    metadata['shape']=[int(n) for n in shape]; atomic_json(root/'0/.zarray',metadata)
    for key in itertools.product(*(np.unique(np.asarray(a)//4) for a in axes)):
        data=np.full((4,)*3,value,np.uint8)
        if zero is not None and tuple(np.asarray(zero)//4)==key:
            data[tuple(np.asarray(zero)%4)]=0
        (root/'0'/'.'.join(map(str,key))).write_bytes(data.tobytes())
    return root


def case(root, stride, physical):
    lo=np.array([3,5,7]); hi=lo+6*stride+2
    selected_shape=hi+3; base_shape=selected_shape*2
    selected_axes=[a+np.arange(7)*stride for a in lo]
    base_axes=[a*2 for a in selected_axes]
    ct=sparse_array(root/'ct',base_shape,[np.array([0])]*3,100)
    attrs=json.loads((ct/'.zattrs').read_bytes())
    attrs['multiscales'][0]['datasets'][0]['coordinateTransformations']=[
        dict(type='scale',scale=[10.8,5.4,2.7]),dict(type='translation',translation=[300.,200.,100.])]
    level=sparse_array(root/'selected',selected_shape,selected_axes,100,zero=lo)
    (level/'0').rename(ct/'1')
    attrs['multiscales'][0]['datasets'].append(dict(path='1',coordinateTransformations=[
        dict(type='scale',scale=[21.6,10.8,5.4]),dict(type='translation',translation=[300.,200.,100.])]))
    atomic_json(ct/'.zattrs',attrs)
    predictions=root/'predictions';predictions.mkdir();groups={}
    for name,value,scale in [('grad_mag',100,4),('nx',255,4),('ny',128,4),('cos',255,2),('presence',255,4)]:
        axes=[np.unique(np.r_[a//scale,(a+scale-1)//scale]) if physical else a//scale for a in base_axes]
        folder=sparse_array(predictions/name,(base_shape+scale-1)//scale,axes,value,
                            zero=[a[1]//scale for a in base_axes] if name=='presence' else None)
        # Pooled-reader arrays have pixel-space calibration matching scaledown.
        atomic_json(folder/'.zattrs',{'multiscales':[{'axes':[dict(name=a,unit='pixel') for a in 'zyx'],
            'datasets':[dict(path='0',coordinateTransformations=[dict(type='scale',scale=[scale]*3)])]}]})
        groups[name]=dict(zarr=name+'/0',scaledown=2 if scale==4 else 1,channels=[name])
    manifest=dict(version=2,base_shape_zyx=base_shape.tolist(),source_to_base=1.,
                  grad_mag_encode_scale=1000.,grad_mag_factor=1.,groups=groups)
    if not physical:
        # Base-grid reader names the optional validity channel mask.
        groups['mask']=groups.pop('presence');groups['mask']['channels']=['mask']
        split=int(base_axes[2][4]);last=int(base_axes[2][5]//4*4)
        manifest['crops']=[[0,0,0,split,int(base_shape[1]),int(base_shape[0])],
                           [last,0,0,int(base_shape[2])-last,int(base_shape[1]),int(base_shape[0])]]
    path=predictions/'input.lasagna.json';atomic_json(path,manifest)
    descriptor=root/'geometry.json'
    atomic_json(descriptor,dict(source=str(ct),origin_xyz_um=[100.,200.,300.],spacing_xyz_um=[2.7,5.4,10.8]))
    fibres=root/'fibres.afv'
    # Cross ROI in X at supported Y/Z nodes; AFV coordinates are base-grid XYZ.
    points=np.array([[0.,base_axes[1][2],base_axes[0][2]],
                     [base_shape[2]-1.,base_axes[1][2],base_axes[0][2]]])
    upstream_curves(fibres,tuple(map(int,base_shape)),[(1,points)])
    config=dict(source=str(ct),output=str(root/'run'),level=1,stride=stride,roi_lo=lo.tolist(),roi_hi=hi.tolist(),
        fibres=str(fibres),fibres_geometry=str(descriptor),lasagna_input=str(path),
        lasagna_geometry=str(descriptor) if physical else None,prediction='qual-v2',invalid_value=0,
        chunk=4,field_chunk=4,field_spacing_um=172.8,sweeps=1,maxiter=200,threads=1,
        reserve_bytes=0,cache=str(root/'cache'),previews=False)
    return config


@pytest.mark.parametrize('stride',[16,32])
@pytest.mark.parametrize('physical',[False,True])
def test_imported_stride_pipeline_172_8um(tmp_path,monkeypatch,stride,physical):
    config=case(tmp_path,stride,physical);root=tmp_path/'run'
    forbid_models(monkeypatch)
    # Source-copy tests lack a distribution RECORD; do not mock stages/readers/solver.
    monkeypatch.setattr(workflow,'implementation_sha',lambda:'stride-fixture')
    original=Volume.read;reads=[]
    def bounded_read(self,lo,hi):
        assert np.all(np.asarray(hi)-lo<=self.chunks), 'dense native envelope read'
        reads.append((self.source,tuple(lo),tuple(hi)))
        return original(self,lo,hi)
    monkeypatch.setattr(Volume,'read',bounded_read)
    planned=workflow.plan(config)
    assert planned['shape_zyx']==[7]*3
    np.testing.assert_allclose(planned['observations']['origin_xyz'],[137.8,254.,364.8])
    np.testing.assert_allclose(planned['observations']['spacing_xyz'],np.array([5.4,10.8,21.6])*stride)
    assert planned['field']['spacing_xyz']==[172.8]*3
    assert planned['input_lo_zyx']==[3,5,7]
    assert planned['input_hi_zyx']==config['roi_hi']
    assert planned['estimated_ct_bytes']==7**3*4**3
    assert planned['estimated_lasagna_staging_bytes']==planned['estimated_lasagna_workspace_bytes']==0
    assert not root.exists()
    # Interrupt after first durable main tile; resume must not rewrite it.
    from winding_field import predict
    first=root/'maps/0_0_0.npz'
    prepare=predict.prepare
    def pause(volume,run,config,stop=None):
        return prepare(volume,run,config,stop=lambda:first.exists() or (stop and stop()))
    monkeypatch.setattr(predict,'prepare',pause)
    assert workflow.run(config,until='mesh')['status']=='PAUSED'
    monkeypatch.setattr(predict,'prepare',prepare)
    saved=(file_sha(first),first.stat().st_mtime_ns)
    result=workflow.run(config,resume=True,until='mesh')
    assert set(result['stages'])=={'prepare','fibres','field','mesh'}
    assert (file_sha(first),first.stat().st_mtime_ns)==saved
    meta=json.loads((root/'PREPARED.json').read_bytes())
    assert meta['input_halo_voxels']==0 and meta['stride']==stride
    assert json.loads((root/'FIELD.json').read_bytes())['fibre_constraint_rows']>0
    assert list((root/'meshes').glob('level-*.obj'))
    with closing(Tiles(root/'maps')) as maps:
        data=maps.read_many(('support','density','normals','ct'),[0]*3,[7]*3)
    assert not data['support'][0,0,0] and not data['support'][1,1,1]
    assert not data['density'][~data['support']].any()
    np.testing.assert_allclose(data['density'][data['support']],.1/2.7,rtol=1e-6)
    np.testing.assert_allclose(data['normals'][data['support']],np.tile([1.,0.,0.],(data['support'].sum(),1)))
    assert data['support'][2,2,2]
    if not physical: assert not data['support'][:,:,4].any()
    with sqlite3.connect(root/'fibres/fibres.sqlite') as db:
        curves=[np.load(io.BytesIO(row[0]),allow_pickle=False) for row in db.execute('SELECT points FROM curves')]
    assert len(curves)==(1 if physical else 2)
    origin=np.array(meta['origin_xyz']);pitch=np.array(meta['spacing_xyz'])
    for points in curves:
        assert np.all(points>=origin-1e-8) and np.all(points<=origin+pitch*6+1e-8)
        np.testing.assert_allclose(points[:,1:],np.tile((origin+pitch*2)[1:],(len(points),1)))
    np.testing.assert_allclose(curves[0][0,0],origin[0])
    np.testing.assert_allclose(curves[-1][-1,0],origin[0]+pitch[0]*6)
    if not physical:
        np.testing.assert_allclose(curves[0][-1,0],origin[0]+pitch[0]*3)
        np.testing.assert_allclose(curves[1][0,0],origin[0]+pitch[0]*5)
    paths=[root/n for n in ('PREPARED.json','FIBRES.json','FIELD.json','MESHES.json','field/CHECKPOINT.json')]
    before={p:(file_sha(p),p.stat().st_mtime_ns) for p in paths}
    workflow.run(config,resume=True,until='mesh')
    assert before=={p:(file_sha(p),p.stat().st_mtime_ns) for p in paths}
    with pytest.raises(ValueError,match='config|changed'):
        workflow.run(dict(config,stride=32 if stride==16 else 16),resume=True,until='mesh')
    # Corrupt a consumed prediction chunk, not a metadata-only identity.
    with sqlite3.connect(root/('lasagna-source' if physical else 'lasagna-import')/'nx.sqlite') as db:
        key=db.execute('SELECT key FROM chunks ORDER BY key LIMIT 1').fetchone()[0]
    chunk=tmp_path/'predictions/nx/0'/key.rsplit('/',1)[-1]
    chunk.write_bytes(bytes([128])*64)
    with pytest.raises(ValueError,match='changed'):
        workflow.run(config,resume=True,until='mesh')
    assert before=={p:(file_sha(p),p.stat().st_mtime_ns) for p in paths}
    assert reads


@pytest.mark.parametrize('stride',[16,32])
@pytest.mark.parametrize('inputs',[{},dict(fibres='x'),dict(lasagna_input='x')])
@pytest.mark.parametrize('prediction',['classical','sd2','qual-v2'])
def test_large_stride_does_not_enable_any_inference(stride,inputs,prediction):
    with pytest.raises(ValueError,match='stride'):
        validate_config(dict(inputs,stride=stride,prediction=prediction))


@pytest.mark.parametrize('stride',[0,9,15,17,31,33,True,16.])
def test_imported_stride_still_rejects_other_values(stride):
    with pytest.raises(ValueError,match='stride'):
        validate_config(dict(fibres='x',lasagna_input='x',stride=stride))


@pytest.mark.parametrize('stride',[16,32])
def test_large_stride_keeps_bounded_route_restricted(stride):
    with pytest.raises(ValueError,match='bounded'):
        validate_config(dict(fibres='x',lasagna_input='x',stride=stride,prediction='qual-v2',
                             bounded_preparation=True,field_spacing_um=172.8))


@pytest.mark.parametrize('stride',[16,32])
def test_remote_download_prefetches_only_sampled_ct_chunks(tmp_path,monkeypatch,http_root,stride):
    config=case(tmp_path,stride,False)
    config.update(source=http_root+'/ct',ct_mode='download',cache_bytes=1<<20)
    descriptor=tmp_path/'geometry.json';frame=json.loads(descriptor.read_bytes())
    frame['source']=config['source'];atomic_json(descriptor,frame)
    forbid_models(monkeypatch)
    monkeypatch.setattr(workflow,'implementation_sha',lambda:'stride-fixture')
    assert workflow.plan(config)['ct_policy']=='download'
    result=workflow.run(config,until='prepare')
    assert set(result['stages'])=={'prepare'}
    with sqlite3.connect(tmp_path/'run/source.sqlite') as db:
        assert db.execute('SELECT count(*) FROM chunks').fetchone()[0]==343


def test_large_stride_keeps_reader_sample_limits(tmp_path):
    from winding_field.lasagna_stream import LasagnaInput
    config=case(tmp_path,32,False)
    with closing(workflow.open_volume(config)) as volume:
        with pytest.raises(ValueError,match='512 MiB'):
            volume.sample([np.zeros(1000,dtype=int)]*3)
        with LasagnaInput(volume,tmp_path/'reader',config) as reader:
            geometry=dict(shape_zyx=[129]*3)
            with pytest.raises(ValueError,match='sample budget'):
                reader.read(geometry,[0]*3,[129]*3)
