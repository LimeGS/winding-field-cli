"""Full atlas PNG readback and stored XYZ/source-ID verification."""
from pathlib import Path
import time

import numpy as np
from PIL import Image

from .native_patches import ResourceGuard
from .sheet_atlas import _arrays, _json, _path, _tile, _verified, query_pixel, MAX_PIXELS
from .util import file_sha


def region(image,pixels,x,y):
    h,w=pixels.shape
    np.testing.assert_array_equal(np.asarray(image.crop((x,y,x+w,y+h))),pixels)


def verify_atlas(root):
    root=Path(root).absolute();started=time.monotonic();resources=ResourceGuard(root,6.,8.)
    atlas=_json(_path(root,'ATLAS.json'));manifest=_json(_path(root,'MANIFEST.json'))
    query_pixel(root,-1,0)  # Validates manifest and atlas metadata binding even outside the canvas.
    for name,sha in manifest['files'].items():resources();_verified(root,name,sha)
    h,w=atlas['shape_yx']
    if h*w>MAX_PIXELS:raise ValueError('atlas pixel bound exceeded')
    previous_limit=Image.MAX_IMAGE_PIXELS;Image.MAX_IMAGE_PIXELS=MAX_PIXELS
    checks=[];total_tiles=0
    try:
        with Image.open(root/'atlas.png') as full,Image.open(root/'mask.png') as masks:
            if full.mode!=masks.mode or full.mode!='L' or full.size!=masks.size or full.size!=(w,h):raise ValueError('atlas PNG format differs')
            for i,s in enumerate(atlas['sheets']):
                x,y=s['offset_xy'];sh,sw=s['shape_yx'];sheet_root=(root/s['render']).parent
                rendered=_json(_path(root,s['render']));queries=0
                with Image.open(_path(root,s['image'])) as image:
                    if image.mode!='L' or image.size!=(sw,sh):raise ValueError('sheet PNG format differs')
                    for tile in rendered['tiles']:
                        resources();c=tile['canvas'];row,col=c['row_offset'],c['column_offset'];data=_tile(sheet_root,tile)
                        pixels,mask=data['image'],data['mask']
                        region(full,pixels,x+col,y+row);region(image,pixels,col,row);region(masks,mask,x+col,y+row);total_tiles+=1
                        if queries<4:
                            rr,cc=np.where(mask==2)
                            if len(rr):
                                middle=len(rr)//2;r,co=int(rr[middle]),int(cc[middle]);result=query_pixel(root,x+col+co,y+row+r)
                                np.testing.assert_array_equal(result['xyz'],data['xyz'][r,co]);owner=int(data['source_face_rows'][r,co])
                                mapped=_arrays(root/s['map']);index=np.flatnonzero(mapped['source_face_rows']==owner)
                                if len(index)!=1 or result['source_face_id']!=int(mapped['source_face_ids'][index[0]]) or result['w']!=s['w'] or result['mask']!=2:
                                    raise ValueError('atlas query source differs')
                                queries+=1
                counts=np.bincount(np.asarray(masks.crop((x,y,x+sw,y+sh))).ravel(),minlength=4)
                if {str(j):int(n) for j,n in enumerate(counts)}!=s['mask_counts']:raise ValueError('mask counts differ')
                right=atlas['sheets'][i+1]['offset_xy'][0] if i+1<len(atlas['sheets']) else w
                for image in (full,masks):
                    for box in ((x,0,x+sw,y),(x,y+sh,x+sw,h),(x+sw,0,right,h)):
                        if np.asarray(image.crop(box)).any():raise ValueError('nonblack separator')
                checks.append(dict(w=s['w'],tiles=len(rendered['tiles']),queries=queries,shape_yx=s['shape_yx'],mask_counts=s['mask_counts']))
        resources()
        return dict(status='FULL_ATLAS_DELIVERY_VERIFIED',atlas_sha256=file_sha(root/'ATLAS.json'),manifest_sha256=file_sha(root/'MANIFEST.json'),
                    files=len(manifest['files']),tiles=total_tiles,sheets=checks,seconds=time.monotonic()-started,
                    full_png_rereads_match_every_tile=True,stored_xyz_and_source_ids_exact=True,
                    peak_tree_rss_bytes=resources.peak,physical_acceptance=False)
    finally:Image.MAX_IMAGE_PIXELS=previous_limit
