import json
from contextlib import closing

import numpy as np
import pytest
from PIL import Image

from test_native_geometry import plane_bundle


def ready_plane(tmp_path):
    from winding_field.surface_mesh import MeshReader
    from winding_field.native_geometry import flatten_mesh,assemble
    from winding_field.native_store import Journal
    with MeshReader(plane_bundle(tmp_path)) as reader:
        store=Journal(tmp_path/'out',{'source':reader.manifest_sha})
        flatten_mesh(reader,store,8)
        assemble(store,1.)
    return store


def analytic_ct(tmp_path):
    from ct_fixture import write_ct
    from winding_field.io import Volume
    z,y,x=np.indices((12,12,12))
    return closing(Volume(write_ct(tmp_path/'ct.zarr',(x+2*y+3*z).astype(np.uint8)),cache=tmp_path/'cache'))


def test_shared_raster_samples_original_ct_and_writes_consistent_tifxyz(tmp_path):
    import tifffile
    from winding_field.native_render import render
    with ready_plane(tmp_path) as store, analytic_ct(tmp_path) as volume:
        result=render(store,volume,.5,export_tifxyz=True)
        assert len(result['components'])==1
        record=result['components'][0]
        assert record['tifxyz']['status']=='WRITTEN'
        folder=store.root/record['path']
        xyz=np.stack([tifffile.imread(folder/'tifxyz'/n) for n in ('x.tif','y.tif','z.tif')],axis=-1)
        valid=(xyz>=0).all(-1)
        assert valid.sum()>20
        cells=valid[:-1,:-1]&valid[1:,:-1]&valid[:-1,1:]&valid[1:,1:]
        used=np.zeros_like(valid)
        for dy,dx in ((0,0),(0,1),(1,0),(1,1)):
            used[dy:dy+len(cells),dx:dx+cells.shape[1]]|=cells
        np.testing.assert_array_equal(valid,used)
        np.testing.assert_allclose(xyz[valid,2],4.,atol=1e-6)
        image=np.array(Image.open(folder/'sheet.png'))
        observed=image[...,1]>0
        assert observed.sum()>20
        expected=np.rint(xyz[...,0]+2*xyz[...,1]+3*xyz[...,2]).astype(np.uint8)
        np.testing.assert_array_equal(image[...,0][observed&valid],expected[observed&valid])
        meta=json.loads((folder/'tifxyz/meta.json').read_text())
        assert meta['scale']==[2.,2.]
        assert record['field_coverage_certified'] is False
        from winding_field.native_render import _TiffTiles
        original=_TiffTiles(store,record['path'],record['grid'],meta['output_frame'])
        baseline=original.read_region((0,0),xyz.shape[:2])
        np.testing.assert_array_equal(xyz.view(np.uint32),baseline['xyz'].view(np.uint32))
        assert record['tifxyz']['partial_geometry'] is False
        assert record['tifxyz']['original_active_cells']==record['tifxyz']['retained_cells']
        assert record['tifxyz']['removed_grid_nodes']==record['tifxyz']['unsafe_cells']==0


def test_raster_conflicts_are_sticky_but_shared_edges_are_allowed():
    from winding_field.native_render import rasterize
    uv=np.array([[0.,0.],[4.,0.],[0.,4.]])
    vertices=np.column_stack([uv,np.zeros(3)])
    faces=np.array([[0,1,2]])
    chart=(vertices,faces,uv)
    points,normals,mask=rasterize(iter([chart,chart]),[0.,0.],1.,(5,5))
    assert mask[1,1]==1
    np.testing.assert_array_equal(points[1,1],[1,1,0])
    points,normals,mask=rasterize(iter([chart,(vertices+[0,0,2],faces,uv),chart]),[0.,0.],1.,(5,5))
    assert mask[1,1]==3
    assert np.isnan(points[1,1]).all()
    assert mask[4,4]==0


def test_cancelled_render_keeps_committed_xyz_and_resumes(tmp_path):
    from winding_field.native_render import render
    from winding_field.native_store import Journal,Paused
    with ready_plane(tmp_path) as store, analytic_ct(tmp_path) as volume:
        identity=store.identity
        store.stop=lambda:store.get('render_tiles',0)>0
        with pytest.raises(Paused):
            render(store,volume,.5)
        rows=store.db.execute("SELECT name,sha FROM files WHERE name LIKE '%/tiles/%'").fetchall()
        assert rows
        snapshots={name:(sha,(store.root/name).stat().st_mtime_ns) for name,sha in rows}
    from winding_field.io import Volume
    with Journal(tmp_path/'out',identity) as store, closing(Volume(tmp_path/'ct.zarr',cache=tmp_path/'cache')) as volume:
        result=render(store,volume,.5)
        assert result['components']
        for name,(sha,mtime) in snapshots.items():
            assert store.db.execute('SELECT sha FROM files WHERE name=?',(name,)).fetchone()[0]==sha
            assert (store.root/name).stat().st_mtime_ns==mtime


def test_pixel_pitch_and_ct_identity_cannot_change_on_resume(tmp_path):
    from winding_field.native_render import render
    with ready_plane(tmp_path) as store, analytic_ct(tmp_path) as volume:
        render(store,volume,.5)
        with pytest.raises(ValueError,match='identity|changed'):
            render(store,volume,1.)


def test_original_ct_changed_after_checkpoint_is_not_hidden_by_saved_tiles(tmp_path):
    from winding_field.native_render import render
    from winding_field.native_store import Journal
    with ready_plane(tmp_path) as store, analytic_ct(tmp_path) as volume:
        identity=store.identity
        render(store,volume,.5)
    path=tmp_path/'ct.zarr/0/0.0.0'
    path.write_bytes(bytes([250])*path.stat().st_size)
    from winding_field.io import Volume
    with Journal(tmp_path/'out',identity) as store, closing(Volume(tmp_path/'ct.zarr',cache=tmp_path/'cache')) as volume:
        with pytest.raises(ValueError,match='source chunk changed'):
            render(store,volume,.5)


def test_anisotropic_ct_uses_physical_coordinates_and_declines_tifxyz(tmp_path):
    from ct_fixture import write_ct
    from winding_field.io import Volume
    from winding_field.native_render import render
    z,y,x=np.indices((12,12,12))
    path=write_ct(tmp_path/'ct.zarr',(x+2*y+3*z).astype(np.uint8),spacing=(2.,3.,4.),origin=(1.,1.,0.))
    with ready_plane(tmp_path) as store, closing(Volume(path,cache=tmp_path/'cache')) as volume:
        result=render(store,volume,.5,export_tifxyz=True)
        assert result['components'][0]['tifxyz']['status']=='UNAVAILABLE'
        name=store.db.execute('SELECT name FROM rendered LIMIT 1').fetchone()[0]
        tile=store.load_arrays(name); valid=tile['mask']==2; xyz=tile['xyz_um'][valid]
        np.testing.assert_allclose(tile['ct'][valid],(xyz[:,0]-1)/2+2*(xyz[:,1]-1)/3+3*xyz[:,2]/4,atol=2e-6)


def test_tifxyz_declines_a_cell_bridging_discontinuous_geometry(tmp_path):
    from winding_field.native_render import _tifxyz
    from winding_field.native_store import Journal
    with analytic_ct(tmp_path) as volume, Journal(tmp_path/'out',{'level':0.}) as store:
        xyz=np.array([[[1.,1.,1.],[2.,1.,1.]],[[1.,2.,10.],[2.,2.,10.]]])
        store.save_arrays('component/tiles/000000000-000000000.npz',xyz_um=xyz,mask=np.ones((2,2),np.uint8))
        grid=dict(shape_yx=[2,2],uv_min_um=[0.,0.],pitch_um=1.,tile_pixels=256,level=0.)
        result=_tifxyz(store,'component',grid,volume)
        assert result['status']=='UNAVAILABLE'
        assert not (store.root/'component/tifxyz/x.tif').exists()
        assert result['original_active_cells']==result['unsafe_cells']==1
        assert result['removed_grid_nodes']==4 and result['retained_cells']==0
        assert result['orphan_samples_omitted']==result['collateral_cells_omitted']==0
        assert not (store.root/'component/.tifxyz.pending').exists()


def test_existing_mesh_export_binds_the_producer_implementation(tmp_path,monkeypatch):
    from winding_field.surface_mesh import MeshReader
    from winding_field.native_pipeline import render_mesh
    with MeshReader(plane_bundle(tmp_path)) as reader, analytic_ct(tmp_path) as volume:
        out=tmp_path/'render'
        assert render_mesh(reader,out,volume,{'chart_faces':8})['status']=='COMPLETE'
        before=(out/'LEVEL.json').read_bytes()
        monkeypatch.setattr('winding_field.workflow.implementation_sha',lambda:'f'*64)
        with pytest.raises(ValueError,match='identity'):
            render_mesh(reader,out,volume,{'chart_faces':8})
        assert (out/'LEVEL.json').read_bytes()==before


@pytest.mark.parametrize('offset,slope,usable',[(0.,0.,False),(128.,0.,False),(0.,1.,True),(-1.,1e-4,True)])
def test_mesh_render_usability_requires_measured_ct_contrast(tmp_path,offset,slope,usable):
    import tifffile
    from ct_fixture import write_ct
    from winding_field.io import Volume
    from winding_field.surface_mesh import MeshReader
    from winding_field.native_pipeline import render_mesh
    z,y,x=np.indices((12,12,12))
    source=write_ct(tmp_path/'ct.zarr',(offset+slope*(x+2*y+3*z)).astype(np.float32))
    out=tmp_path/'render'; config={'chart_faces':8,'render_pitch_um':.5,'export_tifxyz':True}
    with MeshReader(plane_bundle(tmp_path)) as reader, closing(Volume(source,cache=tmp_path/'cache')) as volume:
        result=render_mesh(reader,out,volume,config)
        record=result['components'][0]; folder=out/record['path']
        assert result['coverage_fraction']==1. and result['target_met']
        assert result['source_mesh_complete'] and result['requested_tifxyz_complete']
        assert result['mapped_faces']==result['source_faces']==24
        assert record['counts']['observed_ct']>20
        assert record['tifxyz']['status']=='WRITTEN'
        xyz=np.stack([tifffile.imread(folder/'tifxyz'/n) for n in ('x.tif','y.tif','z.tif')],axis=-1)
        valid=(xyz>=0).all(-1)
        assert valid.sum()>20
        np.testing.assert_allclose(xyz[valid,2],4.,atol=1e-6)
        assert (folder/'sheet.png').is_file() and (out/'CT-SNAPSHOT.sqlite').is_file()
        assert json.loads((out/'IDENTITY.json').read_bytes())['ct']==volume.identity
        assert result['physical_acceptance'] is False and result['field_coverage_certified'] is False
        low,high=record['ct_value_range']
        if usable:
            assert high>low
        else:
            assert low==high==offset
        assert result['usable_render'] is usable
        assert result['status']==('COMPLETE' if usable else 'PARTIAL')
        assert result['usable_render_reason']==(None if usable else 'ct-no-contrast')
        assert json.loads((out/'LEVEL.json').read_bytes())==result
        assert render_mesh(reader,out,volume,config)==result


def test_different_constant_components_do_not_establish_image_contrast(tmp_path):
    from ct_fixture import write_ct
    from winding_field.io import Volume
    from winding_field.surface_mesh import MeshReader
    from winding_field.native_pipeline import render_mesh
    z=np.indices((12,12,12))[0]
    source=write_ct(tmp_path/'ct.zarr',z.astype(np.float32))
    with MeshReader(plane_bundle(tmp_path,copies=2)) as reader, closing(Volume(source,cache=tmp_path/'cache')) as volume:
        result=render_mesh(reader,tmp_path/'render',volume,{'chart_faces':8})
        assert sorted(c['ct_value_range'] for c in result['components'])==[[4.,4.],[6.,6.]]
        assert result['coverage_fraction']==1.
        assert result['usable_render'] is False
        assert result['status']=='PARTIAL'
        assert result['usable_render_reason']=='ct-no-contrast'


def test_completed_mesh_export_revalidates_the_original_ct_payload(tmp_path):
    from winding_field.surface_mesh import MeshReader
    from winding_field.native_pipeline import render_mesh
    from winding_field.io import Volume
    mesh=plane_bundle(tmp_path); out=tmp_path/'render'
    with MeshReader(mesh) as reader, analytic_ct(tmp_path) as volume:
        render_mesh(reader,out,volume,{'chart_faces':8})
    chunk=tmp_path/'ct.zarr/0/0.0.0'
    chunk.write_bytes(bytes([240])*chunk.stat().st_size)
    with MeshReader(mesh) as reader, closing(Volume(tmp_path/'ct.zarr',cache=tmp_path/'cache')) as volume:
        with pytest.raises(ValueError,match='source chunk changed'):
            render_mesh(reader,out,volume,{'chart_faces':8})


def test_uv_budget_leaves_time_to_publish_a_partial_render(tmp_path,monkeypatch):
    import time
    import winding_field.native_geometry as geometry
    from winding_field.surface_mesh import MeshReader
    from winding_field.native_pipeline import render_mesh
    original=geometry._pieces
    def slow(*args,**kwargs):
        for result in original(*args,**kwargs):
            time.sleep(.05)
            yield result
    monkeypatch.setattr(geometry,'_pieces',slow)
    with MeshReader(plane_bundle(tmp_path,nx=17,ny=17)) as reader, analytic_ct(tmp_path) as volume:
        result=render_mesh(reader,tmp_path/'out',volume,{'chart_faces':2,'surface_time_limit_s':3.})
        assert result['status']=='PARTIAL',result
        assert result['source_faces']==512
        assert 0<result['coverage_fraction']<1
        assert result['target_met'] is False
        assert result['usable_render']
        assert list((tmp_path/'out').glob('component-*/sheet.png'))


def test_render_refuses_to_reuse_tiles_after_geometry_changes(tmp_path):
    from winding_field.surface_mesh import MeshReader
    from winding_field.native_geometry import flatten_mesh,assemble
    from winding_field.native_store import Journal,Paused
    from winding_field.native_render import render
    with MeshReader(plane_bundle(tmp_path)) as reader, Journal(tmp_path/'out',{'source':reader.manifest_sha}) as store, analytic_ct(tmp_path) as volume:
        flatten_mesh(reader,store,8,until=.3); assemble(store,.3)
        store.stop=lambda:store.get('render_tiles',0)>0
        with pytest.raises(Paused):
            render(store,volume,.5)
        store.stop=None
        flatten_mesh(reader,store,8); assemble(store,1.)
        with pytest.raises(ValueError,match='identity|geometry'):
            render(store,volume,.5)


def test_resume_finishes_the_frozen_partial_raster_without_moving_its_geometry(tmp_path,monkeypatch):
    import time
    import sqlite3
    import winding_field.native_geometry as geometry
    from winding_field.surface_mesh import MeshReader
    from winding_field.native_pipeline import render_mesh
    original=geometry._pieces
    def slow(*args,**kwargs):
        for result in original(*args,**kwargs):
            time.sleep(.05)
            yield result
    monkeypatch.setattr(geometry,'_pieces',slow)
    out=tmp_path/'out'; config={'chart_faces':2,'surface_time_limit_s':3.}
    with MeshReader(plane_bundle(tmp_path,nx=17,ny=17)) as reader, analytic_ct(tmp_path) as volume:
        result=render_mesh(reader,out,volume,config,stop=lambda:any(out.glob('component-*/tiles/*.npz')))
        assert result['status']=='PAUSED'
        with sqlite3.connect(out/'work.sqlite') as db:
            before=db.execute("SELECT name,uv_path FROM charts WHERE status='MAPPED' ORDER BY name").fetchall()
        assert render_mesh(reader,out,volume,config)['status']=='PARTIAL'
        with sqlite3.connect(out/'work.sqlite') as db:
            assert before==db.execute("SELECT name,uv_path FROM charts WHERE status='MAPPED' ORDER BY name").fetchall()


def test_optional_tiff_cancel_keeps_published_image_and_hides_incomplete_segment(tmp_path):
    from winding_field.native_render import render
    from winding_field.native_store import Paused
    with ready_plane(tmp_path) as store, analytic_ct(tmp_path) as volume:
        store.stop=lambda:any(store.root.glob('component-*/tifxyz/x.tif')) or any(store.root.glob('component-*/.tifxyz.pending/x.tif'))
        with pytest.raises(Paused):
            render(store,volume,.5,export_tifxyz=True)
        component=next(store.root.glob('component-*'))
        assert (component/'sheet.png').is_file()
        assert (component/'RENDER.json').is_file()
        assert not (component/'tifxyz').exists()
        image_before=(component/'sheet.png').read_bytes()
        x_before=(component/'.tifxyz.pending/x.tif').read_bytes()
        store.stop=None
        result=render(store,volume,.5,export_tifxyz=True)
        assert result['components'][0]['tifxyz']['status']=='WRITTEN'
        assert (component/'tifxyz/x.tif').read_bytes()==x_before
        assert (component/'sheet.png').read_bytes()==image_before


def test_changing_parameterizer_runtime_cannot_relabel_completed_geometry(tmp_path,monkeypatch):
    from winding_field.surface_mesh import MeshReader
    from winding_field.native_pipeline import render_mesh
    import winding_field.native_geometry as geometry
    with MeshReader(plane_bundle(tmp_path)) as reader, analytic_ct(tmp_path) as volume:
        out=tmp_path/'render'
        render_mesh(reader,out,volume,{'chart_faces':8})
        monkeypatch.setattr(geometry,'solver_identity',lambda:{'solver':'changed'},raising=False)
        with pytest.raises(ValueError,match='identity'):
            render_mesh(reader,out,volume,{'chart_faces':8})


def test_mesh_render_recovers_an_unregistered_level_receipt(tmp_path,monkeypatch):
    import winding_field.native_pipeline as pipeline
    from winding_field.surface_mesh import MeshReader
    from winding_field.native_store import Paused
    original=pipeline.atomic_json
    def interrupted(path,value):
        original(path,value)
        if path.name=='LEVEL.json':
            raise Paused('PUBLICATION_STOP')
    with MeshReader(plane_bundle(tmp_path)) as reader, analytic_ct(tmp_path) as volume:
        out=tmp_path/'render'
        monkeypatch.setattr(pipeline,'atomic_json',interrupted)
        assert pipeline.render_mesh(reader,out,volume,{'chart_faces':8})['status']=='PAUSED'
        assert (out/'LEVEL.json').is_file()
        monkeypatch.setattr(pipeline,'atomic_json',original)
        assert pipeline.render_mesh(reader,out,volume,{'chart_faces':8})['status']=='COMPLETE'


@pytest.mark.parametrize('low,high,outer',[(.4,.6,(0.,1.)),(.2,.3,(0.,1.)),(0.,1.,(-1.,2.))])
def test_tifxyz_cannot_fill_a_source_mesh_hole_inside_valid_corners(tmp_path,monkeypatch,low,high,outer):
    from winding_field.surface_mesh import import_mesh,MeshReader
    from winding_field.native_pipeline import render_mesh
    from winding_field.util import file_sha
    import winding_field.native_geometry as geometry
    def analytic_projection(v,f):
        return v[:,:2]-v[:,:2].min(0),{'method':'analytic-XY','edge_stretch_min':1.,'edge_stretch_max':1.}
    monkeypatch.setattr(geometry,'_project',analytic_projection)
    a,b=outer
    xy=np.array([[a,a],[b,a],[b,b],[a,b],[low,low],[high,low],[high,high],[low,high]])
    vertices=np.column_stack([xy+3,np.full(8,4.)]); faces=[]
    for i in range(4):
        j=(i+1)%4
        faces.extend([[i,j,i+4],[j,j+4,i+4]])
    source=tmp_path/'ring.npz'; np.savez(source,vertices_xyz=vertices,faces=np.array(faces))
    mesh=tmp_path/'mesh'
    import_mesh({'path':str(source),'sha256':file_sha(source)},mesh,
                {'shape_zyx':[12]*3,'origin_xyz':[0.]*3,'spacing_xyz':[1.]*3,'voxel_um':1.,'sheet_step':1.,'frame':'ring'})
    with MeshReader(mesh) as reader, analytic_ct(tmp_path) as volume:
        result=render_mesh(reader,tmp_path/'out',volume,{'chart_faces':16,'render_pitch_um':1.,'export_tifxyz':True})
        assert result['components'][0]['tifxyz']['status']=='UNAVAILABLE'
        assert list((tmp_path/'out').glob('component-*/sheet.png'))
        assert not list((tmp_path/'out').glob('component-*/tifxyz/meta.json'))


def test_frozen_ct_snapshot_is_not_the_mutable_working_ledger(tmp_path):
    from ct_fixture import write_ct
    from winding_field.io import Volume
    from winding_field.surface_mesh import MeshReader
    from winding_field.native_pipeline import render_mesh
    source=write_ct(tmp_path/'ct.zarr',np.full((40,40,40),128,np.uint8),chunk=16)
    with MeshReader(plane_bundle(tmp_path)) as reader, closing(Volume(source,cache=tmp_path/'cache')) as volume:
        out=tmp_path/'out'
        assert render_mesh(reader,out,volume,{'chart_faces':8})['status']=='PARTIAL'
        snapshot=out/'CT-SNAPSHOT.sqlite'; before=snapshot.read_bytes()
        volume.read([32,32,32],[33,33,33])
        assert snapshot.read_bytes()==before


def ring_bundle(tmp_path,monkeypatch,size=6,low=2.2,high=2.3):
    from winding_field.surface_mesh import import_mesh
    from winding_field.util import file_sha
    import winding_field.native_geometry as geometry
    def xy_projection(v,f):
        return v[:,:2]-v[:,:2].min(0),{'method':'analytic-XY','edge_stretch_min':1.,'edge_stretch_max':1.}
    monkeypatch.setattr(geometry,'_project',xy_projection)
    xy=np.array([[0.,0.],[size,0.],[size,size],[0.,size],
                 [low,low],[high,low],[high,high],[low,high]])
    vertices=np.column_stack([xy+2,np.full(8,4.)]); faces=[]
    for i in range(4):
        j=(i+1)%4
        faces.extend([[i,j,i+4],[j,j+4,i+4]])
    source=tmp_path/'ring.npz'; np.savez(source,vertices_xyz=vertices,faces=np.array(faces))
    mesh=tmp_path/'mesh'
    import_mesh({'path':str(source),'sha256':file_sha(source)},mesh,
                {'shape_zyx':[12,size+6,size+6],'origin_xyz':[0.]*3,'spacing_xyz':[1.]*3,
                 'voxel_um':1.,'sheet_step':1.,'frame':'ring'})
    return mesh


def read_tifxyz(folder):
    import tifffile
    xyz=np.stack([tifffile.imread(folder/n) for n in ('x.tif','y.tif','z.tif')],axis=-1)
    valid=(xyz>=0).all(-1)
    assert np.isfinite(xyz).all() and (xyz[~valid]==-1).all()
    cells=valid[:-1,:-1]&valid[1:,:-1]&valid[:-1,1:]&valid[1:,1:]
    return xyz,valid,cells


@pytest.mark.parametrize('low,high',[(2.2,2.3),(2.4,2.6)])
def test_partial_tifxyz_retains_supported_cells_without_changing_png_or_source(tmp_path,monkeypatch,low,high):
    from winding_field.surface_mesh import MeshReader
    from winding_field.native_pipeline import render_mesh
    from winding_field.native_store import Journal
    from winding_field.native_render import _TiffTiles
    from winding_field.util import file_sha
    mesh=ring_bundle(tmp_path,monkeypatch,low=low,high=high)
    source_before={p.name:file_sha(p) for p in mesh.iterdir() if p.is_file()}
    config={'chart_faces':16,'render_pitch_um':1.}
    with MeshReader(mesh) as reader, analytic_ct(tmp_path) as volume:
        baseline=render_mesh(reader,tmp_path/'png-only',volume,config)
        result=render_mesh(reader,tmp_path/'out',volume,dict(config,export_tifxyz=True))
    record=result['components'][0]; tif=record['tifxyz']; folder=tmp_path/'out'/record['path']
    assert tif['status']=='WRITTEN',tif
    assert tif['partial_geometry'] is True
    assert result['status']=='PARTIAL' and result['requested_tifxyz_complete'] is False
    assert result['coverage_fraction']==1. and result['mapped_faces']==result['source_faces']==8
    assert result['source_area']==baseline['source_area'] and result['target_met']
    assert result['physical_acceptance'] is False and result['field_coverage_certified'] is False
    assert tif['physical_acceptance'] is False and tif['field_coverage_certified'] is False
    for name in ('sheet.png','mask.png','overview.png'):
        assert (folder/name).read_bytes()==(tmp_path/'png-only'/baseline['components'][0]['path']/name).read_bytes()
    xyz,valid,cells=read_tifxyz(folder/'tifxyz')
    expected=np.ones((7,7),bool); expected[2:4,2:4]=False
    np.testing.assert_array_equal(valid,expected)
    # A sub-cell hole with a supported center must still invalidate the cell.
    assert not cells[1:4,1:4].any() and cells.sum()==27
    assert tif['original_active_cells']==36 and tif['unsafe_cells']==1
    assert tif['unsafe_cell_causes']=={'geometric_jump':0,'collapsed_or_folded':0,'unsupported_source':1}
    assert tif['removed_grid_nodes']==4 and tif['collateral_cells_omitted']==8
    assert tif['retained_cells']==tif['surface_cells']==27
    assert tif['original_grid_nodes']==49 and tif['nodes']==45 and tif['orphan_samples_omitted']==0
    identity=json.loads((tmp_path/'out/IDENTITY.json').read_bytes())
    frame=json.loads((folder/'tifxyz/meta.json').read_bytes())['output_frame']
    with Journal(tmp_path/'out',identity) as store:
        original=_TiffTiles(store,record['path'],record['grid'],frame)._raw_region((0,0),(7,7))
        np.testing.assert_array_equal(xyz[valid].view(np.uint32),original['xyz'][valid].view(np.uint32))
        assert original['mask'].all()
    assert source_before=={p.name:file_sha(p) for p in mesh.iterdir() if p.is_file()}


@pytest.mark.parametrize('size,cell',[(258,255),(257,256)])
def test_partial_tifxyz_masks_cross_tile_corners_and_tails(tmp_path,monkeypatch,size,cell):
    from ct_fixture import write_ct
    from winding_field.io import Volume
    from winding_field.surface_mesh import MeshReader
    from winding_field.native_geometry import flatten_mesh,assemble
    from winding_field.native_render import render
    from winding_field.native_store import Journal
    mesh=ring_bundle(tmp_path,monkeypatch,size=size,low=cell+.2,high=cell+.3)
    source=write_ct(tmp_path/'ct.zarr',np.zeros((8,size+6,size+6),np.uint8),chunk=64)
    with MeshReader(mesh) as reader, Journal(tmp_path/'out',{'mesh':reader.manifest_sha}) as store, closing(Volume(source,cache=tmp_path/'cache')) as volume:
        flatten_mesh(reader,store,16); assemble(store,1.)
        record=render(store,volume,1.,True)['components'][0]; tif=record['tifxyz']
        assert tif['status']=='WRITTEN',tif
        xyz,valid,cells=read_tifxyz(store.root/record['path']/'tifxyz')
        expected=np.ones((size+1,size+1),bool); expected[cell:cell+2,cell:cell+2]=False
        np.testing.assert_array_equal(valid,expected)
        y,x=np.indices(valid.shape)
        np.testing.assert_array_equal(xyz[valid],np.stack([x+2,y+2,np.full_like(x,4)],axis=-1)[valid].astype(np.float32))
        lost=9 if cell==255 else 4
        assert cells.sum()==size*size-lost==tif['retained_cells']
        assert tif['original_active_cells']==size*size and tif['unsafe_cells']==1
        assert tif['removed_grid_nodes']==4 and tif['collateral_cells_omitted']==lost-1
        assert tif['orphan_samples_omitted']==0
        masks=store.db.execute("SELECT name FROM files WHERE name LIKE '%/tiles/tifxyz-mask-%'").fetchall()
        assert len(masks)==4
        for name, in masks:
            mask=store.load_arrays(name)
            assert all(a.ndim==2 and max(a.shape)<=256 for a in mask.values())


@pytest.mark.parametrize('interrupt',['mask_commit','mask_unregistered','channel'])
def test_partial_tifxyz_resume_preserves_committed_masks_and_projection(tmp_path,monkeypatch,interrupt):
    from winding_field.surface_mesh import MeshReader
    from winding_field.native_geometry import flatten_mesh,assemble
    from winding_field.native_render import render
    from winding_field.native_store import Journal,Paused
    from winding_field.io import Volume
    mesh=ring_bundle(tmp_path,monkeypatch)
    with MeshReader(mesh) as reader, Journal(tmp_path/'out',{'mesh':reader.manifest_sha}) as store, analytic_ct(tmp_path) as volume:
        identity=store.identity
        flatten_mesh(reader,store,16); assemble(store,1.)
        register=store.register
        def interrupted(name):
            if interrupt=='mask_unregistered' and '/tiles/tifxyz-mask-' in name:
                raise Paused('UNREGISTERED_MASK')
            register(name)
        monkeypatch.setattr(store,'register',interrupted)
        if interrupt=='mask_commit':
            store.stop=lambda:bool(store.db.execute("SELECT 1 FROM files WHERE name LIKE '%/tiles/tifxyz-mask-%'").fetchone())
        elif interrupt=='channel':
            store.stop=lambda:any(store.root.glob('component-*/.tifxyz.pending/x.tif'))
        with pytest.raises(Paused):
            render(store,volume,1.,True)
        assert not list(store.root.glob('component-*/tifxyz'))
        snapshots={name:(store.path(name).read_bytes(),store.path(name).stat().st_mtime_ns)
                   for name, in store.db.execute('SELECT name FROM files')}
        assert any('/tiles/' in n for n in snapshots) and any(n.endswith('/sheet.png') for n in snapshots)
        if interrupt!='mask_unregistered':
            assert any('/tiles/tifxyz-mask-' in n for n in snapshots)
    with Journal(tmp_path/'out',identity) as store, closing(Volume(tmp_path/'ct.zarr',cache=tmp_path/'cache')) as volume:
        record=render(store,volume,1.,True)['components'][0]
        assert record['tifxyz']['status']=='WRITTEN' and record['tifxyz']['partial_geometry']
        assert read_tifxyz(store.root/record['path']/'tifxyz')[2].sum()==27
        for name,(payload,mtime) in snapshots.items():
            assert store.path(name).read_bytes()==payload and store.path(name).stat().st_mtime_ns==mtime


def frozen_plane(tmp_path,xyz,mask=None,seam=None):
    """A frozen raster plus real planar chart/boundary inputs to the local gates."""
    from winding_field.native_store import Journal
    from winding_field.native_geometry import _schema
    store=Journal(tmp_path/'out',{'source':'frozen-plane'}); _schema(store)
    height,width=xyz.shape[:2]
    panels=[(0.,width-1,[0,1,2,3])] if seam is None else [
        (0.,2.25,[0,1,2,3]),(2.25,width-1,[1,4,3,5] if seam=='paired' else [4,5,6,7])]
    for index,(low,high,ids) in enumerate(panels):
        uv=np.array([[low,0.],[high,0.],[low,height-1],[high,height-1]])
        vertices=np.column_stack([uv+2,np.ones(4)]); faces=np.array([[0,1,2],[1,3,2]])
        name=f'flat-{index}'; path=f'uv/{name}.npz'
        store.save_arrays(f'charts/{name}.npz',vertices=vertices,faces=faces)
        store.save_arrays(path,uv=uv)
        with store.db:
            store.db.execute('INSERT INTO charts VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
                (name,1,(high-low)*(height-1),2,'MAPPED',path,low,high,0.,height-1,None,'{}'))
            for a,b in ((0,1),(1,3),(3,2),(2,0)):
                ia,ib=(a,b) if ids[a]<ids[b] else (b,a)
                store.db.execute('INSERT INTO edges VALUES (?,?,?,?,?,?)',
                    (ids[ia],ids[ib],name,ia,ib,1 if ia==a else -1))
    if mask is None:
        mask=np.ones((height,width),np.uint8)
    prefix='component-000000000001'
    store.save_arrays(prefix+'/tiles/000000000-000000000.npz',xyz_um=xyz,mask=mask)
    grid=dict(shape_yx=[height,width],uv_min_um=[0.,0.],pitch_um=1.,tile_pixels=256,level=0.)
    return store,prefix,grid


@pytest.mark.parametrize('fault,cause,unsafe,removed,retained',[
    ('jump','geometric_jump',4,9,20),('fold','collapsed_or_folded',2,6,24),
    ('float32-collapse','collapsed_or_folded',2,6,24)])
def test_partial_tifxyz_classifies_original_float32_and_keeps_no_jump_or_fold(tmp_path,fault,cause,unsafe,removed,retained):
    from winding_field.native_render import _tifxyz
    y,x=np.indices((7,7)); original=np.stack([x+2,y+2,np.ones_like(x)],axis=-1).astype(float)
    if fault=='jump':
        original[3,3,2]=11.
    else:
        original[3,3,0]=3. if fault=='fold' else 4.+1e-8
    store,prefix,grid=frozen_plane(tmp_path,original)
    with store, analytic_ct(tmp_path) as volume:
        result=_tifxyz(store,prefix,grid,volume,1)
        assert result['status']=='WRITTEN',result
        xyz,valid,cells=read_tifxyz(store.root/prefix/'tifxyz')
        expected=np.ones((7,7),bool); expected[2:5,2:5 if fault=='jump' else 4]=False
        np.testing.assert_array_equal(valid,expected)
        np.testing.assert_array_equal(xyz[valid].view(np.uint32),original.astype(np.float32)[valid].view(np.uint32))
        assert result['partial_geometry'] and result['original_active_cells']==36
        assert result['unsafe_cells']==unsafe and result['removed_grid_nodes']==removed
        assert result['unsafe_cell_causes'][cause]==unsafe
        assert sum(result['unsafe_cell_causes'].values())==unsafe
        assert result['retained_cells']==cells.sum()==retained
        assert result['collateral_cells_omitted']==36-unsafe-retained
        assert result['orphan_samples_omitted']==0
        # Check actual published reader triangles, independently of the classifier.
        for row,col in zip(*np.nonzero(cells)):
            a,b=xyz[row,col:col+2].astype(float); c,d=xyz[row+1,col:col+2].astype(float)
            assert all(np.linalg.norm(p-q)<=8 for p,q in ((a,b),(a,c),(a,d),(b,c),(b,d),(c,d)))
            n1=np.cross(b-a,c-a); n2=np.cross(d-b,c-b)
            assert np.linalg.norm(n1)>1e-12 and np.linalg.norm(n2)>1e-12 and n1@n2>0


@pytest.mark.parametrize('seam,retained,removed',[('paired',36,0),('cut',18,14)])
def test_tifxyz_distinguishes_a_real_cut_from_an_exact_flat_seam(tmp_path,seam,retained,removed):
    from winding_field.native_render import _tifxyz
    y,x=np.indices((7,7)); original=np.stack([x+2,y+2,np.ones_like(x)],axis=-1).astype(float)
    store,prefix,grid=frozen_plane(tmp_path,original,seam=seam)
    with store, analytic_ct(tmp_path) as volume:
        result=_tifxyz(store,prefix,grid,volume,1)
        assert result['status']=='WRITTEN',result
        xyz,valid,cells=read_tifxyz(store.root/prefix/'tifxyz')
        expected=np.ones((7,7),bool)
        if seam=='cut':
            expected[:,2:4]=False
        np.testing.assert_array_equal(valid,expected)
        np.testing.assert_array_equal(xyz[valid].view(np.uint32),original.astype(np.float32)[valid].view(np.uint32))
        assert cells.sum()==result['retained_cells']==retained
        assert result['removed_grid_nodes']==removed
        assert result['unsafe_cell_causes']['unsupported_source']==(6 if seam=='cut' else 0)
        assert result['partial_geometry']==(seam=='cut')


@pytest.mark.parametrize('coordinate',[-1e-6,11.00001,np.nan,np.inf])
def test_tifxyz_frame_failure_is_global_even_with_a_valid_remainder(tmp_path,coordinate):
    from winding_field.native_render import _tifxyz
    y,x=np.indices((7,7)); xyz=np.stack([x+2,y+2,np.ones_like(x)],axis=-1).astype(float)
    xyz[3,3,2]=coordinate
    store,prefix,grid=frozen_plane(tmp_path,xyz)
    with store, analytic_ct(tmp_path) as volume:
        result=_tifxyz(store,prefix,grid,volume,1)
        assert result=={'status':'UNAVAILABLE','reason':'geometry outside output frame'}
        assert not (store.root/prefix/'tifxyz').exists() and not (store.root/prefix/'.tifxyz.pending').exists()


def test_partial_tifxyz_counts_existing_and_new_orphans_separately_from_corner_removals(tmp_path):
    from winding_field.native_render import _tifxyz
    y,x=np.indices((7,7)); xyz=np.stack([x+2,y+2,np.ones_like(x)],axis=-1).astype(float)
    xyz[0,0,2]=11.
    mask=np.zeros((7,7),np.uint8); mask[:2,:3]=1; mask[4:,4:]=1; mask[2,6]=1
    store,prefix,grid=frozen_plane(tmp_path,xyz,mask)
    with store, analytic_ct(tmp_path) as volume:
        result=_tifxyz(store,prefix,grid,volume,1)
        assert result['status']=='WRITTEN'
        xyz,valid,cells=read_tifxyz(store.root/prefix/'tifxyz')
        # One existing orphan plus two stranded neighbors; the four unsafe corners are separate.
        assert result['removed_grid_nodes']==4 and result['orphan_samples_omitted']==3
        assert result['original_grid_nodes']==16 and result['nodes']==valid.sum()==9
        assert result['original_active_cells']==6 and result['retained_cells']==cells.sum()==4
        assert result['unsafe_cells']==1 and result['collateral_cells_omitted']==1
        expected=np.zeros((7,7),bool); expected[4:,4:]=True
        np.testing.assert_array_equal(valid,expected)


def test_partial_tifxyz_mask_resume_rejects_corruption_and_rebinding(tmp_path):
    from winding_field.native_render import _tifxyz
    from winding_field.native_store import Journal,Paused
    y,x=np.indices((7,7)); xyz=np.stack([x+2,y+2,np.ones_like(x)],axis=-1).astype(float)
    xyz[3,3,2]=11.
    store,prefix,grid=frozen_plane(tmp_path,xyz)
    with store, analytic_ct(tmp_path) as volume:
        identity=store.identity
        store.stop=lambda:bool(store.db.execute("SELECT 1 FROM files WHERE name LIKE '%/tiles/tifxyz-mask-%'").fetchone())
        with pytest.raises(Paused):
            _tifxyz(store,prefix,grid,volume,1)
        store.stop=None
        with pytest.raises(ValueError,match='changed'):
            _tifxyz(store,prefix,dict(grid,pitch_um=.5),volume,1)
        name=store.db.execute("SELECT name FROM files WHERE name LIKE '%/tiles/tifxyz-mask-%'").fetchone()[0]
        payload=store.path(name).read_bytes(); store.path(name).write_bytes(payload[:-1]+bytes([payload[-1]^1]))
        with pytest.raises(ValueError,match='payload changed'):
            _tifxyz(store,prefix,grid,volume,1)
    with pytest.raises(ValueError,match='payload changed'):
        Journal(tmp_path/'out',identity)
