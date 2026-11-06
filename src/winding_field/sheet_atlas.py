"""Package independent constant-W CT sheets without altering their geometry."""
import argparse
import json
import math
from pathlib import Path
import re
import shutil

import numpy as np
from PIL import Image

from .patch_continue import copy_verified_file, map_array_bytes
from .util import atomic_json, digest, file_sha

MAX_PIXELS=512_000_000
MAX_MAP_BYTES=512<<20


def _path(root,relative):
    p=Path(relative)
    if not isinstance(relative,str) or not relative or p.is_absolute() or '..' in p.parts or '\\' in relative or p.as_posix()!=relative:
        raise ValueError('unsafe atlas path')
    target=root/p
    if any(q.is_symlink() for q in (root,target,*target.parents)) or not target.is_file():
        raise ValueError('regular atlas payload required: '+relative)
    return target


def _json(path):
    if path.stat().st_size>16<<20:raise ValueError('receipt too large')
    return json.loads(path.read_text())


def _verified(root,name,sha):
    path=_path(root,name)
    if not isinstance(sha,str) or not re.fullmatch('[0-9a-f]{64}',sha) or file_sha(path)!=sha:
        raise ValueError('payload hash differs: '+name)
    return path


def _arrays(path,limit=MAX_MAP_BYTES):
    if map_array_bytes(path)>limit:raise ValueError('array memory bound exceeded')
    with np.load(path,allow_pickle=False) as archive:return {k:archive[k] for k in archive.files}


def _sheet(path):
    path=Path(path).absolute();root=path.parent
    record=_json(_path(root,path.name))
    if record.get('schema')!='winding-atlas-sheet-v1' or record.get('kind')!='constant_w' or record.get('z_increases_up') is not True:
        raise ValueError('Z-oriented constant-W sheet receipt required')
    files=record['files']
    if not {'map.npz','RENDER.json','MESH.json'}<=files.keys():raise ValueError('unbound sheet inputs')
    for name,sha in files.items():_verified(root,name,sha)
    mesh=_json(root/'MESH.json');render=_json(root/'RENDER.json');frame=record['frame']
    if 'CT-IDENTITY.json' in files:
        binding=_json(root/'CT-IDENTITY.json')
        if record['ct']!=binding['ct'] or frame!=binding['settings']['output_frame']:
            raise ValueError('sheet CT/frame differs from bound rendering identity')
    origin=np.asarray(frame['origin_xyz_um'],float)
    if origin.shape!=(3,) or not np.isfinite(origin).all():raise ValueError('invalid frame origin')
    level=next((v for v in mesh['levels'] if v['id']==record['level_index']),None)
    step=mesh['metadata']['sheet_step'];w=record['w'];pitch=render['pitch_um']
    if (mesh.get('schema')!='winding-mesh-v1' or mesh.get('status')!='MESH_READY' or level is None
        or not math.isfinite(step) or step<=0 or not math.isfinite(w) or not math.isclose(level['level']/step,w,rel_tol=0,abs_tol=1e-10)):
        raise ValueError('normalized W differs from source mesh')
    if (frame['id']!=mesh['metadata']['frame'] or frame['voxel_um']!=mesh['metadata']['voxel_um']
        or not math.isfinite(frame['voxel_um']) or frame['voxel_um']<=0 or not math.isfinite(pitch) or pitch<=0
        or render.get('window')!=[0,255] or render.get('row_direction')!='decreasing UV v'):
        raise ValueError('incompatible frame/pitch/window/orientation')
    anchor=np.asarray(record['anchor_uv'],float)
    if anchor.shape!=(2,) or not np.isfinite(anchor).all() or not math.isfinite(record['reference_z']):raise ValueError('invalid reference anchor')
    mapped=_arrays(root/'map.npz');uv=mapped['uv'];world=mapped['world'];faces=mapped['faces']
    rows=mapped['source_face_rows'];ids=mapped['source_face_ids']
    if (uv.ndim!=2 or uv.shape[1]!=2 or len(uv)==0 or world.shape!=(len(uv),3)
        or not np.isfinite(uv).all() or not np.isfinite(world).all()
        or faces.ndim!=2 or faces.shape[1]!=3 or not len(faces) or faces.dtype.kind not in 'iu'
        or faces.min()<0 or faces.max()>=len(uv) or rows.shape!=(len(faces),) or ids.shape!=rows.shape
        or rows.dtype.kind not in 'iu' or ids.dtype.kind not in 'iu' or (rows<0).any()
        or len(np.unique(rows))!=len(rows) or len(np.unique(ids))!=len(ids)
        or mapped['source_vertex_ids'].shape!=(len(uv),)):
        raise ValueError('invalid source-bound map arrays')
    low,high=uv.min(0),uv.max(0);world_pitch=pitch/frame['voxel_um']
    shape=(np.ceil((high-low)/world_pitch).astype(np.int64)+1)[::-1].tolist()
    if render['shape_yx']!=shape or math.prod(shape)>MAX_PIXELS:raise ValueError('invalid or oversized sheet canvas')
    if (anchor<low).any() or (anchor>high).any():raise ValueError('anchor outside map')
    tiles=render['tiles'];positions={}
    for tile in tiles:
        c=tile['canvas'];row,col=c['row_offset'],c['column_offset']
        if any(type(n) is not int or n<0 for n in (row,col)) or (row,col) in positions:raise ValueError('invalid tile position')
        positions[row,col]=tile
        expected=dict(pitch=world_pitch,global_u0=float(low[0]),global_v_top=float(high[1]),
                      u0=float(low[0]+col*world_pitch),v_top=float(high[1]-row*world_pitch))
        if any(c[k]!=v for k,v in expected.items()):raise ValueError('tile lattice differs from map')
        if tile['directory']!=f'tile-{row:09d}-{col:09d}':raise ValueError('invalid tile directory')
    rr=sorted({r for r,c in positions});cc=sorted({c for r,c in positions})
    if not rr or rr[0]!=0 or cc[0]!=0 or set(positions)!={(r,c) for r in rr for c in cc}:raise ValueError('missing tiles')
    for starts,extent,key in ((rr,shape[0],'height'),(cc,shape[1],'width')):
        size=max(t['canvas'][key] for t in tiles)
        if type(size) is not int or not 1<=size<=4096:raise ValueError('invalid tile size')
        if len(starts)>1 and (size<2 or starts!=list(range(0,extent,size-1))):raise ValueError('tile coverage gap')
        if len(starts)==1 and size!=extent:raise ValueError('tile coverage gap')
        for (row,col),tile in positions.items():
            offset=row if key=='height' else col
            if tile['canvas'][key]!=min(size,extent-offset):raise ValueError('invalid tile extent')
    order=np.argsort(rows)
    return dict(root=root,record=record,render=render,shape=shape,origin=[float(low[0]),float(high[1])],
                pitch=world_pitch,anchor_row=float((high[1]-anchor[1])/world_pitch),
                rows=rows[order],ids=ids[order],receipt_sha=file_sha(path))


def _tile(root,tile):
    directory='render/'+tile['directory'];receipt=_json(_path(root,directory+'/TILE.json'))
    if receipt!=tile:raise ValueError('tile receipt differs from render')
    files=tile['files']
    if not {'image.png','mask.png','mapping.npz'}<=files.keys():raise ValueError('incomplete tile payloads')
    for name,sha in files.items():_verified(root,directory+'/'+name,sha)
    data=_arrays(root/directory/'mapping.npz',128<<20)
    shape=(tile['canvas']['height'],tile['canvas']['width'])
    if (data['mask'].shape!=shape or data['mask'].dtype!=np.uint8 or (data['mask']>3).any()
        or data['xyz'].shape!=(*shape,3) or data['xyz'].dtype.kind!='f'
        or data['source_face_rows'].shape!=shape or data['source_face_rows'].dtype.kind not in 'iu'):
        raise ValueError('invalid tile mapping arrays')
    for name in ('image','mask'):
        with Image.open(root/directory/(name+'.png')) as image:
            if image.mode!='L' or image.size!=shape[::-1]:raise ValueError('tile PNG format/size differs')
            values=np.asarray(image)
            if name=='mask' and not np.array_equal(values,data['mask']):raise ValueError('PNG/mapping masks differ')
            if name=='image':data['image']=values.copy()
    if data['image'][data['mask']!=2].any():raise ValueError('nonblack invalid CT pixels')
    return data


def _edge(data,index):return {k:v[index].copy() for k,v in data.items()}


def _halo(a,b):
    if any(not np.array_equal(a[k],b[k],equal_nan=True) for k in a):raise ValueError('tile halo differs')


def build_atlas(sheet_paths,output,*,reserve_gib=8):
    """Copy hash-bound sheets, align reference rows and publish READY last."""
    output=Path(output).absolute()
    if output.exists():raise FileExistsError(output)
    if not math.isfinite(reserve_gib) or reserve_gib<0:raise ValueError('invalid disk reserve')
    if not sheet_paths:raise ValueError('at least one sheet required')
    sheets=[_sheet(p) for p in sheet_paths];sheets.sort(key=lambda s:s['record']['w'])
    if len({s['record']['w'] for s in sheets})!=len(sheets):raise ValueError('duplicate W')
    first=sheets[0]
    for s in sheets:
        if any(s['record'][k]!=first['record'][k] for k in ('frame','ct','reference_z')) or s['render']['pitch_um']!=first['render']['pitch_um']:
            raise ValueError('incompatible sheet frame/CT/reference/pitch')
        source=s['root'].resolve()
        if source==output.resolve() or source in output.resolve().parents or output.resolve() in source.parents:raise ValueError('output must be disjoint from sources')
    anchor=max(s['anchor_row'] for s in sheets);x=0;height=0;entries=[]
    for i,s in enumerate(sheets):
        y=round(anchor-s['anchor_row']);h,w=s['shape'];prefix=f'sheets/sheet-{i:03d}'
        entry=dict(w=s['record']['w'],offset_xy=[x,y],shape_yx=s['shape'],uv_origin=s['origin'],pitch=s['pitch'],
                   anchor_atlas_y=y+s['anchor_row'],anchor_rounding_error_pixels=y+s['anchor_row']-anchor,
                   map=prefix+'/map.npz',render=prefix+'/RENDER.json',receipt=prefix+'/SHEET.json',
                   image=prefix+'/sheet.png',overview=prefix+'/overview.png')
        entries.append(entry);x+=w+32;height=max(height,y+h)
    width=x-32;pixels=width*height
    if pixels>MAX_PIXELS:raise ValueError('atlas exceeds 512 million pixel bound')
    reserve=int(reserve_gib*2**30)
    output.parent.mkdir(parents=True,exist_ok=True)
    def guard():
        if shutil.disk_usage(output.parent).free<reserve:raise RuntimeError('atlas disk reserve reached')
    guard()
    source_bytes=sum(sum(p.stat().st_size for p in s['root'].rglob('*') if p.is_file()) for s in sheets)
    if shutil.disk_usage(output.parent).free<reserve+source_bytes+6*pixels:raise RuntimeError('insufficient space for atlas plus disk reserve')
    output.mkdir()
    image=np.memmap(output/'.image.raw',dtype=np.uint8,mode='w+',shape=(height,width))
    mask=np.memmap(output/'.mask.raw',dtype=np.uint8,mode='w+',shape=(height,width))
    def copy(source,name,sha):
        destination=output/name;destination.parent.mkdir(parents=True,exist_ok=True)
        copy_verified_file(source,destination,guard=guard,reserve_bytes=reserve)
        if file_sha(destination)!=sha:raise ValueError('copied payload hash differs: '+name)
    try:
        for s,entry in zip(sheets,entries):
            root=s['root'];prefix=str(Path(entry['map']).parent);record=s['record'];x,y=entry['offset_xy'];h,w=entry['shape_yx']
            copy(_path(root,'SHEET.json'),prefix+'/SHEET.json',s['receipt_sha'])
            for name,sha in record['files'].items():copy(_verified(root,name,sha),prefix+'/'+name,sha)
            tops={};left=None;last_row=None;counts=np.zeros(4,np.int64)
            for tile in sorted(s['render']['tiles'],key=lambda t:(t['canvas']['row_offset'],t['canvas']['column_offset'])):
                guard();data=_tile(root,tile);c=tile['canvas'];r,col=c['row_offset'],c['column_offset'];th,tw=data['mask'].shape
                if r!=last_row:left=None;last_row=r
                if left is not None:_halo(left,_edge(data,(slice(None),0)))
                if r:_halo(tops[col],_edge(data,(0,slice(None))))
                left=_edge(data,(slice(None),-1));tops[col]=_edge(data,(-1,slice(None)))
                valid=(data['mask']==1)|(data['mask']==2);rows=data['source_face_rows'][valid];indices=np.searchsorted(s['rows'],rows)
                if ((indices>=len(s['rows'])).any() or not np.array_equal(s['rows'][np.minimum(indices,len(s['rows'])-1)],rows)
                    or not np.isfinite(data['xyz'][valid]).all()):raise ValueError('invalid mapped XYZ/source face row')
                image[y+r:y+r+th,x+col:x+col+tw]=data['image'];mask[y+r:y+r+th,x+col:x+col+tw]=data['mask']
                counts+=np.bincount(data['mask'][int(r>0):,int(col>0):].ravel(),minlength=4)
                directory='render/'+tile['directory']
                for name,sha in tile['files'].items():copy(_path(root,directory+'/'+name),prefix+'/'+directory+'/'+name,sha)
                copy(_path(root,directory+'/TILE.json'),prefix+'/'+directory+'/TILE.json',file_sha(root/directory/'TILE.json'))
            entry['mask_counts']={str(k):int(v) for k,v in enumerate(counts)}
            picture=Image.fromarray(image[y:y+h,x:x+w]);picture.save(output/entry['image'])
            picture.thumbnail((2048,2048));picture.save(output/entry['overview']);picture.close()
        guard();image.flush();mask.flush()
        Image.fromarray(image).save(output/'atlas.png');Image.fromarray(mask).save(output/'mask.png')
        overview=Image.fromarray(image);overview.thumbnail((4096,4096));overview.save(output/'overview.png');overview.close()
    finally:
        del image,mask
        (output/'.image.raw').unlink(missing_ok=True);(output/'.mask.raw').unlink(missing_ok=True)
    atlas=dict(schema='winding-sheet-atlas-v1',status='VISUAL_ATLAS_READY',shape_yx=[height,width],
               pitch_um=first['render']['pitch_um'],frame=first['record']['frame'],ct=first['record']['ct'],
               reference_z=first['record']['reference_z'],sheets=entries,window=[0,255],separator_pixels=32,
               physical_continuity_verified=False,physical_acceptance=False,
               coordinates='atlas pixels -> per-sheet UV -> original global XYZ; layout translations only')
    from .sheet_atlas_view import write_viewer
    write_viewer(output,atlas,[s['render'] for s in sheets])
    files={str(p.relative_to(output)):file_sha(p) for p in sorted(output.rglob('*')) if p.is_file()}
    atomic_json(output/'MANIFEST.json',dict(files=files,atlas_metadata_sha256=digest(atlas)))
    atlas['manifest_sha256']=file_sha(output/'MANIFEST.json');guard();atomic_json(output/'ATLAS.json',atlas)
    return atlas


def query_pixel(output,x,y):
    """Return an observed integer pixel's stored XYZ and immutable source face ID."""
    if type(x) is not int or type(y) is not int:raise ValueError('integer atlas pixel required')
    root=Path(output);atlas=_json(_path(root,'ATLAS.json'))
    manifest=_json(_verified(root,'MANIFEST.json',atlas['manifest_sha256']))
    if digest({k:v for k,v in atlas.items() if k!='manifest_sha256'})!=manifest['atlas_metadata_sha256']:raise ValueError('atlas metadata hash differs')
    result=dict(atlas_pixel=[x,y],w=None,uv=None,xyz=None,source_face_id=None,mask=0)
    for s in atlas['sheets']:
        ox,oy=s['offset_xy'];h,w=s['shape_yx'];col,row=x-ox,y-oy
        if not (0<=col<w and 0<=row<h):continue
        result.update(w=s['w'],uv=[s['uv_origin'][0]+col*s['pitch'],s['uv_origin'][1]-row*s['pitch']])
        render=_json(_verified(root,s['render'],manifest['files'][s['render']]))
        for t in render['tiles']:
            c=t['canvas'];r,co=row-c['row_offset'],col-c['column_offset']
            if not (0<=r<c['height'] and 0<=co<c['width']):continue
            relative=str(Path(s['render']).parent/'render'/t['directory']/'mapping.npz')
            arrays=_arrays(_verified(root,relative,manifest['files'][relative]),128<<20)
            result['mask']=int(arrays['mask'][r,co])
            if result['mask']==2:
                mapped=_arrays(_verified(root,s['map'],manifest['files'][s['map']]))
                indices=np.flatnonzero(mapped['source_face_rows']==arrays['source_face_rows'][r,co])
                if len(indices)!=1 or not np.isfinite(arrays['xyz'][r,co]).all():raise ValueError('invalid source-bound pixel')
                result.update(xyz=arrays['xyz'][r,co].tolist(),source_face_id=int(mapped['source_face_ids'][indices[0]]))
            return result
        raise ValueError('pixel has no tile')
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    build=sub.add_parser('build');build.add_argument('--sheets',nargs='+',type=Path,required=True)
    build.add_argument('--output',type=Path,required=True);build.add_argument('--disk-reserve-gib',type=float,default=8)
    query=sub.add_parser('query');query.add_argument('atlas',type=Path);query.add_argument('x',type=int);query.add_argument('y',type=int)
    verify=sub.add_parser('verify');verify.add_argument('--atlas',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.command=='build':
        result=build_atlas(args.sheets,args.output,reserve_gib=args.disk_reserve_gib)
        print(json.dumps(dict(status=result['status'],shape_yx=result['shape_yx'],sheets=len(result['sheets']))))
    elif args.command=='verify':
        from .sheet_atlas_verify import verify_atlas
        print(json.dumps(verify_atlas(args.atlas),allow_nan=False))
    else:print(json.dumps(query_pixel(args.atlas,args.x,args.y),allow_nan=False))
    return 0


if __name__=='__main__':main()
