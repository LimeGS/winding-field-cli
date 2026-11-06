"""Atlas layout must preserve CT pixels, masks and physical coordinates."""
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from winding_field.patch_pipeline import render_map
from winding_field.util import atomic_json, file_sha


def sheet(root, w, anchor_v=22.):
    root.mkdir()
    uv=np.array([[-4.,20.],[2.,20.],[2.,24.],[-4.,24.]])
    world=np.column_stack([uv[:,0]+100,uv[:,1]+200,uv[:,1]*3+w+1000])
    faces=np.array([[0,1,2],[0,2,3]])
    mapped=dict(world=world,uv=uv,faces=faces,display=np.ones(2,bool),scientific=np.ones(2,bool),
                source_face_rows=np.array([7,9]),source_face_ids=np.array([107,109]),source_vertex_ids=np.arange(4)+50)
    np.savez_compressed(root/'map.npz',**mapped)
    frame=dict(id='test/L2',voxel_um=9.6,origin_xyz_um=[0,0,0])
    def sample(xyz):return np.full(len(xyz),80+w+61,dtype=float),np.ones(len(xyz),bool)
    render=render_map(mapped,root/'render',sample,frame,pitch_um=19.2,tile_size=3)
    atomic_json(root/'RENDER.json',render)
    # Real source calibration, not an unchecked label inferred from a directory name.
    source=dict(schema='winding-mesh-v1',status='MESH_READY',metadata=dict(sheet_step=1.25,frame='test/L2',voxel_um=9.6),
                levels=[dict(id=0,level=w*1.25)])
    atomic_json(root/'MESH.json',source)
    rec=dict(schema='winding-atlas-sheet-v1',kind='constant_w',w=w,level_index=0,frame=frame,
             ct=dict(source='synthetic-coordinate-coded-ct',shape=[100,100,100],spacing_xyz=[9.6]*3,origin_xyz=[0]*3),
             reference_z=1000.,anchor_uv=[0.,anchor_v],z_increases_up=True,
             files={name:file_sha(root/name) for name in ('map.npz','RENDER.json','MESH.json')})
    atomic_json(root/'SHEET.json',rec)
    return root/'SHEET.json'


def test_atlas_sorted_layout_pixels_and_xyz_roundtrip(tmp_path):
    from winding_field.sheet_atlas import build_atlas, query_pixel
    a=sheet(tmp_path/'a',-59,22.);b=sheet(tmp_path/'b',-61,24.)
    before={str(p):file_sha(p) for p in tmp_path.rglob('*') if p.is_file()}
    out=tmp_path/'atlas';atlas=build_atlas([a,b],out,reserve_gib=0)
    assert [s['w'] for s in atlas['sheets']]==[-61,-59]
    assert atlas['pitch_um']==19.2 and atlas['physical_continuity_verified'] is False
    assert (out/'index.html').is_file() and (out/'atlas.png').is_file()
    for entry,original in zip(atlas['sheets'],(b,a)):
        x,y=entry['offset_xy'];h,w=entry['shape_yx']
        assert w==4 and h==3
        result=query_pixel(out,x+1,y+1)
        assert result['w']==entry['w'] and result['mask']==2
        np.testing.assert_allclose(result['uv'],[-2.,22.],atol=0,rtol=0)
        np.testing.assert_allclose(result['xyz'],[98.,222.,1066.+entry['w']],atol=1e-12,rtol=0)
        assert result['source_face_id'] in (107,109)
        with Image.open(out/'atlas.png') as image:assert image.getpixel((x+1,y+1))==80+entry['w']+61
        assert file_sha(out/entry['map'])==file_sha(original.parent/'map.npz')
    gap=atlas['sheets'][0]['offset_xy'][0]+4
    assert query_pixel(out,gap,1)['mask']==0
    assert query_pixel(out,-1,0)['mask']==0
    assert before=={p:file_sha(p) for p in before}
    with pytest.raises(FileExistsError):build_atlas([a,b],out,reserve_gib=0)


@pytest.mark.parametrize('change', ['w','frame','pitch','ct','hash','path','reference'])
def test_atlas_rejects_incompatible_or_unbound_sources(tmp_path,change):
    from winding_field.sheet_atlas import build_atlas
    a=sheet(tmp_path/'a',-59);b=sheet(tmp_path/'b',-60)
    r=json.loads(b.read_text())
    if change=='w':r['w']=-58
    elif change=='frame':r['frame']['voxel_um']=19.2
    elif change=='ct':r['ct']['source']='other'
    elif change=='reference':r['reference_z']=999.
    elif change=='path':r['files']['../escape']=r['files']['map.npz']
    elif change=='hash':r['files']['map.npz']='0'*64
    elif change=='pitch':
        p=b.parent/'RENDER.json';v=json.loads(p.read_text());v['pitch_um']=9.6;atomic_json(p,v);r['files']['RENDER.json']=file_sha(p)
    atomic_json(b,r)
    with pytest.raises(ValueError):build_atlas([a,b],tmp_path/'atlas',reserve_gib=0)
    assert not (tmp_path/'atlas'/'ATLAS.json').exists()


def test_atlas_masks_tampering_and_tile_halo(tmp_path):
    from winding_field.sheet_atlas import build_atlas,query_pixel
    a=sheet(tmp_path/'a',-60);out=tmp_path/'atlas'
    atlas=build_atlas([a],out,reserve_gib=0)
    s=atlas['sheets'][0];x,y=s['offset_xy']
    # Last pixel in one tile is the first in its neighbor; both identify the same sample.
    assert query_pixel(out,x+2,y+1)['mask']==2
    render=json.loads((out/s['render']).read_text());tile=render['tiles'][0]
    p=(out/s['render']).parent/'render'/tile['directory']/'mapping.npz'
    p.write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='hash'):query_pixel(out,x+1,y+1)


def test_atlas_rejects_duplicate_w_and_no_sources(tmp_path):
    from winding_field.sheet_atlas import build_atlas
    a=sheet(tmp_path/'a',-60)
    for sources in ([],[a,a]):
        with pytest.raises(ValueError):build_atlas(sources,tmp_path/'atlas',reserve_gib=0)


def change_tile(receipt, index, edit):
    render_path=receipt.parent/'RENDER.json';render=json.loads(render_path.read_text())
    tile=render['tiles'][index];root=receipt.parent/'render'/tile['directory']
    with np.load(root/'mapping.npz') as arrays:values={k:arrays[k] for k in arrays.files}
    edit(values)
    np.savez_compressed(root/'mapping.npz',**values)
    Image.fromarray(values['mask']).save(root/'mask.png')
    with Image.open(root/'image.png') as image:pixels=np.asarray(image).copy()
    pixels[values['mask']!=2]=0
    Image.fromarray(pixels).save(root/'image.png')
    tile['files']={name:file_sha(root/name) for name in tile['files']}
    atomic_json(root/'TILE.json',tile);atomic_json(render_path,render)
    rec=json.loads(receipt.read_text());rec['files']['RENDER.json']=file_sha(render_path);atomic_json(receipt,rec)


def test_unmapped_missing_and_ambiguous_pixels_have_no_query_xyz(tmp_path):
    from winding_field.sheet_atlas import build_atlas,query_pixel
    a=sheet(tmp_path/'a',-60)
    def invalid(values):
        values['mask'][0,0]=0;values['mask'][1,0]=1;values['mask'][2,0]=3
        values['xyz'][[0,2],0]=np.nan;values['source_face_rows'][[0,2],0]=-1
    change_tile(a,0,invalid)
    # Row2 belongs to a second tile as well.
    def bottom(values):
        values['mask'][0,0]=3;values['xyz'][0,0]=np.nan;values['source_face_rows'][0,0]=-1
    change_tile(a,2,bottom)
    atlas=build_atlas([a],tmp_path/'atlas',reserve_gib=0)
    x,y=atlas['sheets'][0]['offset_xy']
    for row,mask in enumerate((0,1,3)):
        r=query_pixel(tmp_path/'atlas',x,y+row)
        assert r['mask']==mask and r['xyz'] is None and r['source_face_id'] is None


@pytest.mark.parametrize('broken',['xyz','mask','owner','missing_tile'])
def test_rejects_conflicting_halos_or_missing_tiles(tmp_path,broken):
    from winding_field.sheet_atlas import build_atlas
    a=sheet(tmp_path/'a',-60)
    if broken=='missing_tile':
        p=a.parent/'RENDER.json';r=json.loads(p.read_text());r['tiles'].pop();atomic_json(p,r)
        r=json.loads(a.read_text());r['files']['RENDER.json']=file_sha(p);atomic_json(a,r)
    else:
        def change(values):
            if broken=='xyz':values['xyz'][1,0,0]+=1
            elif broken=='mask':values['mask'][1,0]=1
            else:values['source_face_rows'][1,0]=9
        change_tile(a,1,change)
    with pytest.raises(ValueError):build_atlas([a],tmp_path/'atlas',reserve_gib=0)
    assert not (tmp_path/'atlas'/'ATLAS.json').exists()


def test_rejects_nonblack_invalid_pixels(tmp_path):
    from winding_field.sheet_atlas import build_atlas
    a=sheet(tmp_path/'a',-60)
    def missing(values):values['mask'][1,0]=1
    change_tile(a,0,missing)
    rpath=a.parent/'RENDER.json';r=json.loads(rpath.read_text());t=r['tiles'][0]
    root=a.parent/'render'/t['directory']
    with Image.open(root/'image.png') as im:pixels=np.asarray(im).copy()
    pixels[1,0]=99;Image.fromarray(pixels).save(root/'image.png')
    t['files']['image.png']=file_sha(root/'image.png');atomic_json(root/'TILE.json',t);atomic_json(rpath,r)
    rec=json.loads(a.read_text());rec['files']['RENDER.json']=file_sha(rpath);atomic_json(a,rec)
    with pytest.raises(ValueError):build_atlas([a],tmp_path/'atlas',reserve_gib=0)


def test_rejects_invalid_frame_origin(tmp_path):
    from winding_field.sheet_atlas import build_atlas
    b=sheet(tmp_path/'b',-59);rec=json.loads(b.read_text());rec['frame']['origin_xyz_um']=[0,0];atomic_json(b,rec)
    with pytest.raises(ValueError):build_atlas([b],tmp_path/'atlas2',reserve_gib=0)


def test_query_rejects_changed_atlas_placement(tmp_path):
    from winding_field.sheet_atlas import build_atlas,query_pixel
    a=sheet(tmp_path/'a',-60);root=tmp_path/'atlas';atlas=build_atlas([a],root,reserve_gib=0)
    atlas['sheets'][0]['offset_xy']=[100,100];atomic_json(root/'ATLAS.json',atlas)
    with pytest.raises(ValueError,match='metadata hash'):query_pixel(root,101,101)


def test_cli_dispatch_builds_and_queries_an_atlas(tmp_path,capsys):
    from winding_field.cli import main
    source=sheet(tmp_path/'source',-60);root=tmp_path/'atlas'
    main(['sheet-atlas','build','--sheets',str(source),'--output',str(root),'--disk-reserve-gib','0'])
    assert json.loads(capsys.readouterr().out)['status']=='VISUAL_ATLAS_READY'
    main(['sheet-atlas','query',str(root),'1','1'])
    result=json.loads(capsys.readouterr().out)
    assert result['w']==-60 and result['xyz']==[98.,222.,1006.]


@pytest.mark.parametrize('change',['ct','origin'])
def test_single_sheet_rejects_contradictory_bound_ct_identity(tmp_path,change):
    from winding_field.sheet_atlas import build_atlas
    source=sheet(tmp_path/'source',-60);record=json.loads(source.read_text())
    binding=source.parent/'CT-IDENTITY.json'
    atomic_json(binding,dict(ct=record['ct'],settings=dict(output_frame=record['frame'])))
    record['files']['CT-IDENTITY.json']=file_sha(binding);atomic_json(source,record)
    assert build_atlas([source],tmp_path/'valid',reserve_gib=0)['status']=='VISUAL_ATLAS_READY'
    if change=='ct':record['ct']['source']='different-scan'
    else:record['frame']['origin_xyz_um'][0]=9.6
    atomic_json(source,record)
    with pytest.raises(ValueError,match='CT/frame'):
        build_atlas([source],tmp_path/'invalid',reserve_gib=0)
    assert not (tmp_path/'invalid'/'ATLAS.json').exists()
