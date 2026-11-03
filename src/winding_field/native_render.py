"""One bounded XYZ raster feeds raw-CT images and optional mesh-derived TIFXYZ."""
import json
from contextlib import closing
import hashlib
from functools import lru_cache
from itertools import groupby
import math
import os
import struct
import sqlite3
import zlib

import numpy as np
from PIL import Image

from .flatten import cross2
from .preview import sample_ct
from .util import atomic_json,digest

TILE=256
MAX_PIXELS=100_000_000


def _geometry_sha(store):
    chain=hashlib.sha256(); count=0
    rows=store.db.execute("SELECT c.name,c.component,c.minx,c.maxx,c.miny,c.maxy,u.sha,m.sha "
        "FROM charts c JOIN files u ON u.name=c.uv_path JOIN files m ON m.name='charts/'||c.name||'.npz' "
        "WHERE c.status='MAPPED' ORDER BY c.name")
    for row in rows:
        chain.update(json.dumps(row,separators=(',',':'),allow_nan=False).encode()+b'\n'); count+=1
    if count!=store.db.execute("SELECT count(*) FROM charts WHERE status='MAPPED'").fetchone()[0]:
        raise ValueError('uncommitted native render geometry')
    return chain.hexdigest()


def rasterize(charts,origin,pitch,shape):
    height,width=shape
    if not 0<height<=TILE or not 0<width<=TILE or not math.isfinite(pitch) or pitch<=0:
        raise ValueError('invalid bounded raster tile')
    origin=np.asarray(origin,float)
    xyz=np.full((height*width,3),np.nan); normals=np.zeros_like(xyz)
    low=np.full_like(xyz,np.inf); high=np.full_like(xyz,-np.inf)
    mask=np.zeros(height*width,np.uint8)
    for vertices,faces,uv in charts:
        if not np.isfinite(vertices).all() or not np.isfinite(uv).all():
            raise ValueError('nonfinite raster geometry')
        tri=(uv[faces]-origin)/pitch
        starts=np.maximum(0,np.ceil(tri.min(1)-1e-9)).astype(np.int64)
        stops=np.minimum([width-1,height-1],np.floor(tri.max(1)+1e-9)).astype(np.int64)
        dims=np.maximum(0,stops-starts+1); counts=np.prod(dims,axis=1)
        selected=np.flatnonzero(counts)
        first=0
        while first<len(selected):
            cumulative=np.cumsum(counts[selected[first:]])
            count=max(1,int(np.searchsorted(cumulative,131072,side='right')))
            ids=selected[first:first+count]; first+=count
            sizes=counts[ids]; which=np.repeat(ids,sizes)
            offset=np.arange(int(sizes.sum()))-np.repeat(np.r_[0,np.cumsum(sizes[:-1])],sizes)
            cols=starts[which,0]+offset%dims[which,0]
            rows=starts[which,1]+offset//dims[which,0]
            t=tri[which]; a,b=t[:,1]-t[:,0],t[:,2]-t[:,0]
            relative=np.column_stack([cols,rows])-t[:,0]
            den=cross2(a,b)
            if np.any(den<=0):
                raise ValueError('inverted raster triangles')
            w1=cross2(relative,b)/den; w2=cross2(a,relative)/den
            inside=(w1>=-1e-9)&(w2>=-1e-9)&(w1+w2<=1+1e-9)
            which,rows,cols,w1,w2=(v[inside] for v in (which,rows,cols,w1,w2))
            source=vertices[faces[which]]
            points=source[:,0]+w1[:,None]*(source[:,1]-source[:,0])+w2[:,None]*(source[:,2]-source[:,0])
            normal=np.cross(source[:,1]-source[:,0],source[:,2]-source[:,0])
            length=np.linalg.norm(normal,axis=1)
            if np.any(length<=0):
                raise ValueError('degenerate raster geometry')
            normal/=length[:,None]
            key=rows*width+cols
            unique,pick=np.unique(key,return_index=True)
            fresh=mask[unique]==0
            xyz[unique[fresh]]=points[pick[fresh]]; normals[unique[fresh]]=normal[pick[fresh]]
            mask[unique[fresh]]=1
            np.minimum.at(low,key,points); np.maximum.at(high,key,points)
            conflict=np.linalg.norm(high[unique]-low[unique],axis=1)>max(1e-7,pitch*1e-6)
            mask[unique[conflict]]=3; xyz[unique[conflict]]=np.nan
    return xyz.reshape(height,width,3),normals.reshape(height,width,3),mask.reshape(height,width)


def _charts(store,component,origin,pitch,shape):
    high=np.asarray(origin)+pitch*(np.asarray(shape[::-1])-1)
    for name,uv_path in store.db.execute("SELECT name,uv_path FROM charts WHERE component=? AND status='MAPPED' "
            'AND minx<=? AND maxx>=? AND miny<=? AND maxy>=? ORDER BY name',
            (component,float(high[0]),float(origin[0]),float(high[1]),float(origin[1]))):
        store.check()
        chart=store.load_arrays('charts/'+name+'.npz')
        yield chart['vertices'],chart['faces'],store.load_arrays(uv_path)['uv']


def _tile_name(prefix,y,x):
    return f'{prefix}/tiles/{y:09d}-{x:09d}.npz'


def _cell_mask_name(prefix,y,x):
    # Keep interrupted mask payloads inside the Journal's accounted tile namespace.
    return f'{prefix}/tiles/tifxyz-mask-{y:09d}-{x:09d}.npz'


def _blocks(store,prefix,shape,window,quality=False):
    height,width=shape
    for y in range(0,height,TILE):
        store.check()
        block=np.zeros((min(TILE,height-y),width,1 if quality else 2),np.uint8)
        for x in range(0,width,TILE):
            tile=store.load_arrays(_tile_name(prefix,y,x)); mask=tile['mask']
            h,w=mask.shape
            if quality:
                block[:h,x:x+w,0]=mask
            else:
                values=np.nan_to_num(tile['ct'],nan=window[0])
                gray=np.rint(np.clip((values-window[0])*255/(window[1]-window[0]),0,255)).astype(np.uint8)
                block[:h,x:x+w,0]=gray; block[:h,x:x+w,1]=(mask==2)*255
        yield y,block


def _write_png(store,name,shape,blocks,quality=False):
    if store.db.execute('SELECT 1 FROM files WHERE name=?',(name,)).fetchone():
        store.validate_file(name)
        return
    path=store.path(name); path.parent.mkdir(parents=True,exist_ok=True)
    pending=store.path(str(path.relative_to(store.root).with_name('.'+path.name+'.pending')))
    height,width=shape
    with pending.open('wb') as stream:
        def chunk(kind,data):
            stream.write(struct.pack('>I',len(data))+kind+data+struct.pack('>I',zlib.crc32(kind+data)))
        stream.write(b'\x89PNG\r\n\x1a\n')
        chunk(b'IHDR',struct.pack('>IIBBBBB',width,height,8,0 if quality else 4,0,0,0))
        compressor=zlib.compressobj()
        for _,block in blocks:
            store.check(block.nbytes*2)
            for row in block:
                data=compressor.compress(b'\0'+row.tobytes())
                if data:
                    chunk(b'IDAT',data)
        chunk(b'IDAT',compressor.flush()); chunk(b'IEND',b'')
        stream.flush(); os.fsync(stream.fileno())
    pending.replace(path); store.register(name)


def _publish_record(store,name,record):
    if store.db.execute('SELECT 1 FROM files WHERE name=?',(name,)).fetchone():
        if json.loads(store.validate_file(name).read_bytes())!=record:
            raise ValueError('published native render changed')
    else:
        atomic_json(store.path(name),record); store.register(name)


class _TiffTiles:
    def __init__(self,store,prefix,grid,frame):
        self.store=store; self.prefix=prefix; self.grid=grid; self.identity={'frame':frame}
        self.root=store.root/prefix/'tiles'; self.scope=store.root
        self.frame=frame; self._binding={'producer':'native-mesh-v1','source':store.identity,'grid':grid,'component':prefix}
        self.cell_masks=False

    def guard(self,pending_bytes=0):
        self.store.check(pending_bytes)

    def read_region(self,lo,hi):
        lower=np.maximum(0,np.asarray(lo)-1)
        upper=np.minimum(self.grid['shape_yx'],np.asarray(hi)+1)
        raw=self._raw_region(lower,upper)
        valid=raw['mask']
        original=valid.copy()
        if self.cell_masks:
            self._clear_unsafe_corners(valid,lower,upper)
        cells=valid[:-1,:-1]&valid[1:,:-1]&valid[:-1,1:]&valid[1:,1:]
        used=np.zeros_like(valid)
        for dy,dx in ((0,0),(0,1),(1,0),(1,1)):
            used[dy:dy+cells.shape[0],dx:dx+cells.shape[1]]|=cells
        select=tuple(slice(int(a-b),int(c-b)) for a,b,c in zip(lo,lower,hi))
        xyz=raw['xyz'][select].copy(); mask=used[select]
        xyz[~mask]=-1
        return {'xyz':xyz,'mask':mask,'removed':(original&~valid)[select],'orphans':(valid&~used)[select]}

    def _clear_unsafe_corners(self,valid,lo,hi):
        lower=np.maximum(0,np.asarray(lo)-1)
        upper=np.minimum(np.asarray(self.grid['shape_yx'])-1,hi)
        for y in range(lower[0]//TILE*TILE,upper[0],TILE):
            for x in range(lower[1]//TILE*TILE,upper[1],TILE):
                self.store.check()
                bad=self.store.load_arrays(_cell_mask_name(self.prefix,y,x))['unsafe']!=0
                for dy,dx in ((0,0),(0,1),(1,0),(1,1)):
                    start=np.array([y+dy,x+dx])
                    a=np.maximum(lo,start); b=np.minimum(hi,start+bad.shape)
                    if (b<=a).any():
                        continue
                    src=tuple(slice(int(v-o),int(w-o)) for v,w,o in zip(a,b,start))
                    dst=tuple(slice(int(v-o),int(w-o)) for v,w,o in zip(a,b,lo))
                    valid[dst]&=~bad[src]

    def _raw_region(self,lo,hi):
        xyz=np.full((hi[0]-lo[0],hi[1]-lo[1],3),-1,np.float32)
        mask=np.zeros(xyz.shape[:2],bool)
        for y in range(lo[0]//TILE*TILE,hi[0],TILE):
            for x in range(lo[1]//TILE*TILE,hi[1],TILE):
                tile=self.store.load_arrays(_tile_name(self.prefix,y,x))
                lower=np.maximum(lo,[y,x]); upper=np.minimum(hi,np.array([y,x])+tile['mask'].shape)
                src=tuple(slice(int(a-b),int(c-b)) for a,b,c in zip(lower,[y,x],upper))
                dst=tuple(slice(int(a-b),int(c-b)) for a,b,c in zip(lower,lo,upper))
                valid=(tile['mask'][src]==1)|(tile['mask'][src]==2)
                points=(tile['xyz_um'][src]-self.frame['origin_xyz_um'])/self.frame['voxel_um']
                xyz[dst]=np.where(valid[...,None],points,-1).astype(np.float32); mask[dst]=valid
        return {'xyz':xyz,'mask':mask}


def _cell_support_check(store,component,grid):
    """Center support plus boundary crossing detects holes smaller than a raster cell."""
    @lru_cache(maxsize=128)
    def uv(path):
        return store.load_arrays(path)['uv']

    @lru_cache(maxsize=32)
    def boundaries(name,path):
        own=uv(path); result=[]
        rows=store.db.execute('SELECT e.a,e.b,e.ia,e.ib,e.orientation,p.ia,p.ib,p.orientation,c.uv_path '
            'FROM edges e LEFT JOIN edges p ON p.a=e.a AND p.b=e.b AND p.chart!=e.chart '
            "LEFT JOIN charts c ON c.name=p.chart AND c.status='MAPPED' "
            'WHERE e.chart=? ORDER BY e.a,e.b',(name,))
        for _,group in groupby(rows,key=lambda r:r[:2]):
            first=next(group); mates=[]
            if first[8] is not None:
                mates.append(first[5:])
            for row in group:
                if row[8] is not None and len(mates)<2:
                    mates.append(row[5:])
            edge=own[[first[2],first[3]]]
            if (len(mates)==1 and mates[0][2]==-first[4]
                    and np.array_equal(edge,uv(mates[0][3])[[mates[0][0],mates[0][1]]])):
                continue
            result.append(edge)
        return np.asarray(result).reshape(-1,2,2)

    def supported(x,y,active):
        origin=np.asarray(grid['uv_min_um']); pitch=grid['pitch_um']
        center_origin=origin+np.array([x+.5,y+.5])*pitch
        _,_,centers=rasterize(_charts(store,component,center_origin,pitch,active.shape),center_origin,pitch,active.shape)
        supported=active&(centers==1)
        lower=origin+np.array([x,y])*pitch; upper=lower+np.array(active.shape[::-1])*pitch
        charts=store.db.execute("SELECT name,uv_path FROM charts WHERE status='MAPPED' AND component=? "
            'AND minx<=? AND maxx>=? AND miny<=? AND maxy>=? ORDER BY name',
            (component,float(upper[0]),float(lower[0]),float(upper[1]),float(lower[1])))
        for name,path in charts:
            store.check()
            for edge in boundaries(name,path):
                a,b=(edge-lower)/pitch
                lo=np.maximum(0,np.floor(np.minimum(a,b))).astype(int)
                hi=np.minimum(np.array(active.shape[::-1])-1,np.floor(np.maximum(a,b))).astype(int)
                if (hi<lo).any():
                    continue
                rr,cc=np.mgrid[lo[1]:hi[1]+1,lo[0]:hi[0]+1]
                use=active[rr,cc]
                if not use.any():
                    continue
                cells=np.column_stack([cc[use],rr[use]])
                eps=128*np.finfo(float).eps*max(1.,float(np.abs(edge).max()/pitch))
                bottom=cells+eps; top=cells+1-eps
                enter=np.zeros(len(cells)); leave=np.ones(len(cells)); inside=np.ones(len(cells),bool)
                for axis,delta in enumerate(b-a):
                    if delta==0:
                        inside&=(a[axis]>bottom[:,axis])&(a[axis]<top[:,axis])
                    else:
                        t0=(bottom[:,axis]-a[axis])/delta; t1=(top[:,axis]-a[axis])/delta
                        enter=np.maximum(enter,np.minimum(t0,t1)); leave=np.minimum(leave,np.maximum(t0,t1))
                crossed=cells[inside&(leave>enter)]
                supported[crossed[:,1],crossed[:,0]]=False
        return supported
    return supported


def _cell_validity(xyz,valid,grid,voxel_um,support_check,x,y):
    active=valid[:-1,:-1]&valid[1:,:-1]&valid[:-1,1:]&valid[1:,1:]
    unsafe=np.zeros(active.shape,np.uint8)
    if active.any():
        xyz=xyz.astype(float)
        a,b,c,d=(points[active]*voxel_um for points in
                 (xyz[:-1,:-1],xyz[:-1,1:],xyz[1:,:-1],xyz[1:,1:]))
        jump=np.zeros(len(a),bool)
        for p,q in ((a,b),(a,c),(a,d),(b,c),(b,d),(c,d)):
            jump|=np.linalg.norm(p-q,axis=1)>8*grid['pitch_um']
        n1=np.cross(b-a,c-a); n2=np.cross(d-b,c-b)
        folded=(np.linalg.norm(n1,axis=1)<=1e-12)|(np.linalg.norm(n2,axis=1)<=1e-12)|(np.sum(n1*n2,axis=1)<=0)
        supported=np.zeros_like(active) if support_check is None else support_check(x,y,active)
        unsafe[active]=jump.astype(np.uint8)|(folded.astype(np.uint8)<<1)|((~supported[active]).astype(np.uint8)<<2)
    return dict(active=active,unsafe=unsafe)


def _tifxyz(store,prefix,grid,volume,component=None):
    from .tifxyz_store import write_segment
    spacing=np.asarray(volume.spacing_xyz)
    if not np.all(spacing==spacing[0]):
        return {'status':'UNAVAILABLE','reason':'anisotropic output frame'}
    frame=dict(id=digest(volume.identity),origin_xyz_um=list(volume.origin_xyz),voxel_um=float(spacing[0]),
               shape_zyx=list(volume.shape),source_identity_sha256=digest(volume.identity))
    tiles=_TiffTiles(store,prefix,grid,frame)
    height,width=grid['shape_yx']; cells=0; nodes=0; raw_nodes=0; removed=0; orphans=0
    support_check=None if component is None else _cell_support_check(store,component,grid)
    chain=hashlib.sha256()
    for y in range(0,height,TILE):
        for x in range(0,width,TILE):
            store.check()
            name=_tile_name(prefix,y,x); store.validate_file(name)
            row=store.db.execute('SELECT sha,bytes FROM files WHERE name=?',(name,)).fetchone()
            chain.update(json.dumps((name,*row),separators=(',',':')).encode()+b'\n')
    binding=dict(tiles._binding,frame=frame,geometry_sha256=None if component is None else _geometry_sha(store),
                 component_id=component,raster_sha256=chain.hexdigest(),normalization='unsafe-cell-corners-v1')
    _publish_record(store,prefix+'/TIFXYZ-MASK.json',binding)
    tiles._binding=dict(tiles._binding,cell_mask_identity_sha256=digest(binding))
    original_cells=0; unsafe_cells=0
    causes=dict(geometric_jump=0,collapsed_or_folded=0,unsupported_source=0)
    reasons=('local grid cell bridges a geometric jump','collapsed or folded grid cell',
             'grid cell crosses unsupported source geometry, a hole or a cut')
    # Classify the ORIGINAL delivered float32 nodes, before any corner or orphan removal.
    for y in range(0,height,TILE):
        for x in range(0,width,TILE):
            store.check()
            region=tiles._raw_region((y,x),(min(y+TILE+1,height),min(x+TILE+1,width)))
            xyz=region['xyz']; valid=region['mask']
            raw_nodes+=int(valid[:TILE,:TILE].sum())
            if not np.isfinite(xyz).all() or (xyz[valid]<0).any() or (xyz[valid]>np.array(volume.shape[::-1])-1).any():
                return {'status':'UNAVAILABLE','reason':'geometry outside output frame'}
            if y>=height-1 or x>=width-1:
                continue
            name=_cell_mask_name(prefix,y,x)
            if store.db.execute('SELECT 1 FROM files WHERE name=?',(name,)).fetchone():
                classified=store.load_arrays(name)
            else:
                classified=_cell_validity(xyz,valid,grid,spacing[0],support_check,x,y)
                store.save_arrays(name,**classified)
            original_cells+=int(classified['active'].sum())
            unsafe_cells+=int(np.count_nonzero(classified['unsafe']))
            for bit,key in enumerate(causes):
                causes[key]+=int(np.count_nonzero(classified['unsafe']&(1<<bit)))
    tiles.cell_masks=True
    # Re-run the published-geometry gates on exactly the mask the TIFF writer will read.
    for y in range(0,height,TILE):
        for x in range(0,width,TILE):
            store.check()
            region=tiles.read_region((y,x),(min(y+TILE+1,height),min(x+TILE+1,width)))
            xyz=region['xyz'].astype(float); valid=region['mask']
            nodes+=int(valid[:min(TILE,height-y),:min(TILE,width-x)].sum())
            removed+=int(region['removed'][:TILE,:TILE].sum())
            orphans+=int(region['orphans'][:TILE,:TILE].sum())
            if not np.isfinite(xyz).all() or (xyz[valid]<0).any() or (xyz[valid]>np.array(volume.shape[::-1])-1).any():
                return {'status':'UNAVAILABLE','reason':'geometry outside output frame'}
            classified=_cell_validity(xyz,valid,grid,spacing[0],support_check,x,y)
            cells+=int(classified['active'].sum())
            for bit,reason in enumerate(reasons):
                if np.any(classified['unsafe']&(1<<bit)):
                    return {'status':'UNAVAILABLE','reason':reason}
    counts=dict(partial_geometry=unsafe_cells>0,original_active_cells=original_cells,unsafe_cells=unsafe_cells,
                unsafe_cell_causes=causes,removed_grid_nodes=removed,collateral_cells_omitted=original_cells-unsafe_cells-cells,
                retained_cells=cells,surface_cells=cells,original_grid_nodes=raw_nodes,nodes=nodes,orphan_samples_omitted=orphans,
                producer='native-mesh-v1',field_coverage_certified=False,physical_acceptance=False)
    if not cells:
        reason=next((reason for key,reason in zip(causes,reasons) if causes[key]),'no four-corner surface cells')
        return dict(counts,status='UNAVAILABLE',reason=reason)
    final=store.path(prefix+'/tifxyz'); pending=store.path(prefix+'/.tifxyz.pending')
    result=write_segment(tiles,grid,frame,final if final.exists() else pending)
    if not final.exists():
        pending.replace(final)
    for name in (*result['files'],'.segment.json'):
        store.register(prefix+'/tifxyz/'+name)
    return dict(counts,status='WRITTEN',path=prefix+'/tifxyz',manifest=result)


def render(store,volume,pitch_um,export_tifxyz=False):
    if type(pitch_um) not in (int,float) or not math.isfinite(pitch_um) or pitch_um<=0:
        raise ValueError('positive physical render pitch required')
    if export_tifxyz:
        import tifffile  # Preflight before any expensive work.
    identity=dict(volume=volume.identity,pitch_um=pitch_um,export_tifxyz=export_tifxyz,geometry_sha256=_geometry_sha(store))
    old=store.get('render_identity')
    if old is not None and old!=identity:
        raise ValueError('render identity changed')
    with store.db:
        store.set('render_identity',identity)
    if volume._ledger is None:
        volume.bind_ledger(store.root/'ct-source.sqlite')
    store.phase='validate_ct'
    volume.validate_used()
    store.phase='rasterize'
    store.db.execute('CREATE TABLE IF NOT EXISTS rendered(component INTEGER,y INTEGER,x INTEGER,name TEXT,PRIMARY KEY(component,y,x))')
    records=[]
    groups=store.db.execute("SELECT component,min(minx),min(miny),max(maxx),max(maxy),sum(area) "
                            "FROM charts WHERE status='MAPPED' GROUP BY component ORDER BY component").fetchall()
    for component,minx,miny,maxx,maxy,area in groups:
        store.check()
        origin=np.floor(np.array([minx,miny])/pitch_um)*pitch_um
        wh=(np.ceil((np.array([maxx,maxy])-origin)/pitch_um)+1).astype(np.int64)
        width,height=map(int,wh)
        if min(width,height)<=0 or max(width,height)>100_000 or width*height>MAX_PIXELS:
            raise ValueError('global raster exceeds bounded100M-pixel canvas')
        prefix=f'component-{component:012d}'
        grid=dict(shape_yx=[height,width],uv_min_um=origin.tolist(),pitch_um=pitch_um,tile_pixels=TILE,
                  level=float(store.identity.get('level',0.)))
        counts=np.zeros(4,np.int64); value_min=math.inf; value_max=-math.inf
        for y in range(0,height,TILE):
            for x in range(0,width,TILE):
                store.phase='rasterize'
                store.check()
                name=_tile_name(prefix,y,x)
                committed=store.db.execute('SELECT 1 FROM rendered WHERE component=? AND y=? AND x=?',(component,y,x)).fetchone()
                if committed:
                    tile=store.load_arrays(name)
                else:
                    shape=(min(TILE,height-y),min(TILE,width-x))
                    here=origin+np.array([x,y])*pitch_um
                    xyz,normal,mask=rasterize(_charts(store,component,here,pitch_um,shape),here,pitch_um,shape)
                    valid=mask==1; values=np.full(shape,np.nan,np.float32)
                    if valid.any():
                        store.phase='sample_ct'
                        samples,observed=sample_ct(volume,xyz[valid])
                        if not np.isfinite(samples[observed]).all():
                            raise ValueError('nonfinite original CT samples')
                        values[valid]=samples
                        rows,cols=np.nonzero(valid); mask[rows[observed],cols[observed]]=2
                    tile={'xyz_um':xyz,'normal':normal,'mask':mask,'ct':values}
                    store.phase='save_tiles'
                    store.save_arrays(name,**tile)
                    with store.db:
                        store.db.execute('INSERT INTO rendered VALUES (?,?,?,?)',(component,y,x,name))
                        store.set('render_tiles',store.get('render_tiles',0)+1)
                counts+=np.bincount(tile['mask'].ravel(),minlength=4)
                values=tile['ct'][tile['mask']==2]
                if len(values):
                    value_min=min(value_min,float(values.min())); value_max=max(value_max,float(values.max()))
        if volume.dtype.kind in 'ui':
            limits=np.iinfo(volume.dtype); window=[float(limits.min),float(limits.max)]
        else:
            window=[value_min,value_max] if math.isfinite(value_min) else [0.,1.]
            if window[1]<=window[0]:
                window[1]=window[0]+1.
        store.phase='publish'
        _write_png(store,prefix+'/sheet.png',(height,width),_blocks(store,prefix,(height,width),window))
        _write_png(store,prefix+'/mask.png',(height,width),_blocks(store,prefix,(height,width),window,True),True)
        overview_name=prefix+'/overview.png'
        if not store.db.execute('SELECT 1 FROM files WHERE name=?',(overview_name,)).fetchone():
            stride=max(1,math.ceil(max(height,width)/2048))
            overview=np.zeros((math.ceil(height/stride),math.ceil(width/stride),2),np.uint8)
            for y,block in _blocks(store,prefix,(height,width),window):
                rows=np.arange(y,y+len(block)); keep=rows%stride==0
                overview[rows[keep]//stride]=block[keep,::stride]
            path=store.path(overview_name); pending=path.with_name('.overview.pending')
            with pending.open('wb') as stream:
                Image.fromarray(overview,mode='LA').save(stream,format='PNG'); stream.flush(); os.fsync(stream.fileno())
            pending.replace(path); store.register(overview_name)
        record=dict(path=prefix,grid=grid,mapped_area=area,counts=dict(zip(('background','unsupported_ct','observed_ct','ambiguous_xyz'),map(int,counts))),
                    gray_window=window,ct_value_range=[value_min,value_max] if math.isfinite(value_min) else None,
                    tifxyz_requested=export_tifxyz,tifxyz_receipt='TIFXYZ.json',
                    physical_acceptance=False,field_coverage_certified=False)
        _publish_record(store,prefix+'/RENDER.json',record)
        if export_tifxyz:
            store.phase='tifxyz'
        tif=_tifxyz(store,prefix,grid,volume,component) if export_tifxyz else {'status':'NOT_REQUESTED'}
        _publish_record(store,prefix+'/TIFXYZ.json',tif)
        records.append(dict(record,tifxyz=tif))
    store.phase='validate_ct'
    volume.validate_used()
    if not records:
        raise ValueError('no mapped native component to render')
    # A case-only name would replace ct-source.sqlite on common macOS filesystems.
    if store.db.execute("SELECT 1 FROM files WHERE name='CT-SNAPSHOT.sqlite'").fetchone():
        store.validate_file('CT-SNAPSHOT.sqlite')
    else:
        pending=store.path('.CT-SNAPSHOT.pending.sqlite')
        with closing(sqlite3.connect(pending)) as snapshot:
            volume._ledger.backup(snapshot)
        pending.replace(store.path('CT-SNAPSHOT.sqlite'))
        store.register('CT-SNAPSHOT.sqlite')
    return dict(components=records,producer='native-mesh-v1',physical_acceptance=False,field_coverage_certified=False)
