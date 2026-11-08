import json

import numpy as np
import pytest


@pytest.mark.parametrize('flags,factors',[
    ([],[2,4,4,4]),
    (['--lasagna-cos-downsample','8','--lasagna-grad-mag-downsample','1',
      '--lasagna-nx-downsample','2','--lasagna-ny-downsample','2'],[8,1,2,2]),
])
def test_plan_exposes_channel_grids_without_reducing_model_input(tmp_path,capsys,flags,factors):
    from ct_fixture import write_ct
    from winding_field.cli import main
    source=write_ct(tmp_path/'ct',np.ones((65,65,65),'u1'),spacing=(8.64,)*3)
    assert main(['plan','--source',str(source),'--output',str(tmp_path/'out'),*flags])==0
    plan=json.loads(capsys.readouterr().out)
    for name,factor in zip(('cos','grad_mag','nx','ny'),factors):
        grid=plan['lasagna_output_grids'][name]
        assert grid['downsample']==factor
        assert grid['shape_zyx']==[(65+factor-1)//factor]*3
        np.testing.assert_allclose(grid['spacing_xyz'],[8.64*factor]*3)
    assert plan['estimated_lasagna_staging_bytes']==96**3
    assert plan['estimated_lasagna_workspace_bytes']>65**3*32
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('config',[
    {'lasagna_cos_downsample':3}, {'lasagna_grad_mag_downsample':0},
    {'lasagna_nx_downsample':True}, {'lasagna_ny_downsample':8},
    {'lasagna_input':'external.json','lasagna_cos_downsample':2},
])
def test_invalid_generation_grids_rejected_before_io(config):
    from winding_field.config import validate_config
    with pytest.raises(ValueError,match='Lasagna|lasagna|nx.*ny'):
        validate_config(config)


@pytest.mark.parametrize('factors',[(2,4,4),(1,8,2),(8,1,4),(2,8,2),(2,2,8)])
def test_generated_mixed_grids_preserve_density_and_bind_settings(tmp_path,monkeypatch,factors):
    from test_lasagna import prepared_source
    from winding_field import lasagna
    volume,geometry,config=prepared_source(tmp_path,monkeypatch)
    cos,grad,normal=factors
    config.update(lasagna_cos_downsample=cos,lasagna_grad_mag_downsample=grad,
                  lasagna_nx_downsample=normal,lasagna_ny_downsample=normal)
    run=tmp_path/'run'
    try:
        result=lasagna.prepare(volume,run,config,geometry)
        assert result['encoding']['groups']['cos']['scaledown']==cos.bit_length()-1
        assert result['encoding']['groups']['nx']['scaledown']==normal.bit_length()-1
        data=lasagna.read_observations(run,geometry,[0]*3,[8]*3)
        assert data['support'].all()
        np.testing.assert_allclose(data['density'],.01,rtol=1e-6)
        before=(run/'LASAGNA.json').read_bytes()
        with pytest.raises(ValueError,match='changed|mismatch'):
            lasagna.prepare(volume,run,dict(config,lasagna_cos_downsample=1 if cos!=1 else 8),geometry)
        assert (run/'LASAGNA.json').read_bytes()==before
    finally: volume.close()


@pytest.mark.parametrize('factor',[1,2,4,8])
def test_bounded_raw_reduction_matches_global_reflected_pyramid(factor):
    from scipy.ndimage import convolve1d
    from winding_field.bounded_predictions import LasagnaProducts
    shape=(35,37,39)
    rng=np.random.default_rng(12)
    raw=rng.random((8,*shape),dtype=np.float32)
    class Windows:
        def __init__(self):self.shape=shape;self.model=None
        def read_raw(self,lo,hi):return raw[(slice(None),)+tuple(slice(a,b) for a,b in zip(lo,hi))]
    products=LasagnaProducts(Windows(),dict(cos=factor,grad_mag=factor,nx=factor,ny=factor))
    expected=raw
    step=factor
    while step>1:
        for axis in (1,2,3):expected=convolve1d(expected,np.array([1,4,6,4,1])/16,axis=axis,mode='mirror')
        expected=expected[:,::2,::2,::2];step//=2
    for lo,hi in [(np.zeros(3,int),np.array(expected.shape[1:])),
                  (np.array([1,1,1]),np.array(expected.shape[1:])-1)]:
        actual=products.raw(factor,lo,hi)
        np.testing.assert_allclose(actual,expected[(slice(None),)+tuple(slice(a,b) for a,b in zip(lo,hi))],atol=2e-7)


def test_bounded_defaults_calibrate_reduced_lattice(tmp_path,monkeypatch):
    from winding_field import bounded_pipeline as pipeline
    from test_bounded_pipeline import test_bounded_prepare_retains_coarse_only_and_releases_model
    test_bounded_prepare_retains_coarse_only_and_releases_model(tmp_path,monkeypatch)
    progress=json.loads((tmp_path/'coarse/calibration-progress.json').read_bytes())
    assert progress['count']==4**3
    identity=json.loads((tmp_path/'coarse/IDENTITY.json').read_bytes())
    assert identity['source']['lasagna_downsamples']==dict(cos=2,grad_mag=4,nx=4,ny=4)


def test_worker_keeps_odd_axis_samples():
    from winding_field.lasagna_worker import output_size
    assert [output_size(65,f) for f in (1,2,4,8)]==[65,33,17,9]


def test_singleton_reduced_grid_is_valid_for_calibration(tmp_path):
    from winding_field.bounded_coarse import NativeMaps,produce,endpoint_grid
    def read(lo,hi):
        shape=tuple(hi-lo)
        return dict(normals=np.broadcast_to(np.array([1.,0.,0.],'f4'),(*shape,3)).copy(),
                    density=np.ones(shape,'f4'),support=np.ones(shape,bool),surface_phase=np.ones(shape,'f4'))
    native=NativeMaps((8,8,8),read)
    calibration=NativeMaps((1,1,1),read,chunk=16)
    meta=dict(shape_zyx=[8]*3,spacing_xyz=[1.]*3,origin_xyz=[0.]*3,frame='fixture')
    reduced=dict(meta,shape_zyx=[1]*3,spacing_xyz=[8.]*3)
    assert produce(tmp_path,native,meta,endpoint_grid(meta,4.),{},calibration=(calibration,reduced))=='COMPLETE'
    result=json.loads((tmp_path/'CALIBRATION.json').read_bytes())
    assert result['supported_voxels']==1
    assert result['mean_density_per_um']==1.


def test_select_products_keeps_only_requested_verified_levels(tmp_path):
    from ct_fixture import write_ct
    from winding_field.lasagna import select_products,_products
    from winding_field.util import atomic_json
    output=tmp_path/'output';output.mkdir()
    groups={}
    for name,value in [('cos',255),('grad_mag',100),('nx',255),('ny',128)]:
        folder=output/f'surface_{name}.ome.zarr'
        write_ct(folder,np.full((33,33,33),value,'u1'))
        for level in (1,2,3):
            other=tmp_path/f'{name}-{level}'
            write_ct(other,np.full([(33+(1<<level)-1)//(1<<level)]*3,value,'u1'))
            (other/'0').rename(folder/str(level))
        groups[name]=dict(zarr=folder.name+'/0',scaledown=0,channels=[name])
    atomic_json(output/'surface.lasagna.json',dict(version=2,base_shape_zyx=[33]*3,
        source_to_base=1,grad_mag_encode_scale=1000.,grad_mag_factor=1.,groups=groups))
    factors=dict(cos=2,grad_mag=8,nx=4,ny=4)
    select_products(tmp_path,factors)
    manifest,paths=_products(tmp_path,[33]*3,factors)
    assert [manifest['groups'][n]['scaledown'] for n in factors]==[1,3,2,2]
    for name,path in paths.items():
        assert [p.name for p in path.parent.iterdir() if p.is_dir()]==[str(factors[name].bit_length()-1)]


@pytest.mark.parametrize('factors',[
    dict(cos=2,grad_mag=4,nx=4,ny=4),dict(cos=2,grad_mag=8,nx=4,ny=4),
    dict(cos=2,grad_mag=8,nx=2,ny=2),dict(cos=2,grad_mag=2,nx=8,ny=8),
])
def test_official_worker_odd_mixed_output_grids(tmp_path,factors):
    import os
    from pathlib import Path
    import subprocess
    from ct_fixture import write_ct
    from winding_field import lasagna,lasagna_worker
    from winding_field.io import Volume
    python=os.environ.get('WINDING_LASAGNA_PYTHON')
    if not python:pytest.skip('requires isolated pinned Lasagna runtime')
    source=write_ct(tmp_path/'input',np.full((35,37,39),128,'u1'))/'0'
    output=tmp_path/'output';output.mkdir()
    code='''
import runpy,sys,torch
import lasagna.preprocess_cos_omezarr as upstream
torch.save({},sys.argv[sys.argv.index('--unet-checkpoint')+1])
class Model(torch.nn.Module):
    def forward(self,tile):
        result=torch.ones((1,8,*tile.shape[2:]),device=tile.device)*.5
        result[:,1]=.1
        return result
def load(self,*,device):
    self.output_sigmoid=False
    return Model().to(device).eval()
upstream.LasagnaCosPredict3DAdapter.load_model=load
worker=sys.argv.pop(1)
runpy.run_path(worker,run_name='__main__')
'''
    process=subprocess.run([python,'-I','-c',code,str(Path(lasagna_worker.__file__).resolve()),'predict3d',
        '--input',str(source),'--output',str(output/'surface.lasagna.json'),
        '--unet-checkpoint',str(tmp_path/'unused.pt'),'--tile-size','64','--overlap','32','--border','16',
        '--cos-scaledown',str(factors['cos']),'--scaledown',str(min(factors['grad_mag'],factors['nx'])),
        '--levels',str(max(factors.values()).bit_length()),'--device','cpu',
        '--input-reader','python-zarr','--prefetch-workers','1','--prefetch-tiles-per-gpu','1',
        '--flush-workers','0','--accumulator-workers','0','--input-cache-gib','0',
        '--no-download','--ome-chunk','32','--ome-compressor','none'],
        capture_output=True,text=True,timeout=180)
    assert process.returncode==0,process.stdout+process.stderr
    lasagna.select_products(tmp_path,factors)
    _,paths=lasagna._products(tmp_path,[35,37,39],factors)
    for name,path in paths.items():
        product=Volume(path,voxel_um=1,cache=tmp_path/'cache')
        try:
            values=product.read([0]*3,list(product.shape))
            assert np.isfinite(values).all()
            if name=='grad_mag':assert (values>0).all()
        finally:product.close()
