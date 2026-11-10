"""Canonical continuous W/Z ranges, with atomic geometry and domain audit.

This imposes an explicit angular hypothesis; it does not establish a physical
sheet. SQLite row IDs are local; node/face keys are global recipe identities.
"""
from contextlib import ExitStack
from collections import OrderedDict
from decimal import Decimal, InvalidOperation, localcontext
import json
import math
from pathlib import Path
import platform
import sys

import numpy as np
import field_store

from .surface_sources import FieldSource
from .surface_mesh import MeshDB, MeshReader, _manifest, _rows_sha, sha
from .util import atomic_json, digest


EPS = 128 * np.finfo(np.float64).eps
POLICY = dict(coefficient_relative_tolerance=EPS, root_relative_tolerance=1e-10,
              support='all-traversed-cell-corners', tangent='reject',
              triangulation='(00,10,11),(00,11,01)', coordinates='float64')


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _integer(value, name, low=1, high=1000000):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f'{name} must be an integer in [{low},{high}]')
    return value


def _recipe(request, source):
    allowed={'field','geometry','umbilicus','h','W_ref','theta_ref','samples_per_w','max_edge_world'}
    if not isinstance(request,dict) or set(request)-allowed:
        raise ValueError('unknown range recipe fields')
    if request.get('W_ref',-60)!=-60 or request.get('theta_ref',0)!=0:
        raise ValueError('range calibration requires W_ref=-60 and theta_ref=0')
    if type(request.get('h')) is not int or request['h'] not in (-1,1):
        raise ValueError('explicit handedness h must be -1 or +1')
    samples=_integer(request.get('samples_per_w',1000),'samples_per_w',4)
    cp=np.asarray(request.get('umbilicus'),dtype=float)
    if cp.ndim!=2 or cp.shape[1:]!=(3,) or len(cp)<2 or not np.isfinite(cp).all() or not np.all(np.diff(cp[:,2])>0):
        raise ValueError('umbilicus requires finite world XYZ points in strictly increasing Z')
    if min(source.shape)<2:
        raise ValueError('range source requires at least two samples on each axis')
    edge=request.get('max_edge_world')
    if edge is not None and (type(edge) not in (int,float) or not math.isfinite(edge) or edge<=0):
        raise ValueError('max_edge_world must be finite and positive')
    field=dict(request['field']);field.pop('path',None)
    if 'mask' in field:field['mask']={'sha256':field['mask']['sha256']}
    result = dict(field=field,geometry=source.identity['geometry'],umbilicus=cp.tolist(),
        h=request['h'],W_ref=-60,theta_ref=0,samples_per_w=samples,max_edge_world=edge,
        numeric_contract=dict(policy=POLICY,algorithm='radial-cell-quadratic-v1',
            python=platform.python_version(),implementation=sys.implementation.name,
            numpy=np.__version__,platform=platform.platform(),machine=platform.machine(),
            code={name:sha(Path(__file__).with_name(name)) for name in
                  ('range_surface.py','surface_sources.py','surface_mesh.py')},
             field_decoder_sha256=sha(field_store.__file__)))
    if field.get('kind') == 'zarr-v2':
        result['numeric_contract']['remote_source_identity'] = source.identity
    return result


def _bounds(bounds,samples):
    if not isinstance(bounds,(list,tuple)) or len(bounds)!=2:
        raise ValueError('w_range requires two decimal bounds')
    result=[]
    for value in bounds:
        try:
            # Do not let Decimal's ambient context silently round long literals.
            text=str(value)
            with localcontext() as context:
                context.prec=max(50,len(text)+20)
                tick=Decimal(text)*samples
                if not tick.is_finite() or tick!=tick.to_integral_value() or abs(tick)>2**50:
                    raise ValueError('W bound is nonfinite, outside capacity, or off lattice')
                result.append(int(tick))
        except InvalidOperation as error:
            raise ValueError('invalid decimal W bound') from error
    if result[0]>=result[1]:raise ValueError('W bounds must be strictly increasing')
    return result


def quadratic_roots(a,b,c):
    """All roots on [0,1]; flag tangency or an identically-zero segment.

    Normalize before discriminant evaluation and use the cancellation-resistant
    q formula. Numerically unresolved discriminants are rejected as degenerate.
    """
    scale=max(abs(a),abs(b),abs(c))
    if not math.isfinite(scale):raise ValueError('nonfinite radial polynomial')
    if scale==0:return [],True
    a,b,c=a/scale,b/scale,c/scale
    degenerate=False
    if abs(a)<=EPS:
        if abs(b)<=EPS:return [],abs(c)<=EPS
        roots=[-c/b]
    else:
        d=b*b-4*a*c
        tolerance=EPS*(b*b+abs(4*a*c))
        if d < -tolerance:return [],False
        if abs(d)<=tolerance:
            roots=[-b/(2*a)];degenerate=True
        else:
            q=-.5*(b+math.copysign(math.sqrt(d),b))
            roots=sorted([q/a,c/q])
    roots=[min(1.,max(0.,v)) for v in roots if -EPS<=v<=1+EPS]
    return roots,degenerate and bool(roots)


def _direction(tick,recipe):
    samples=recipe['samples_per_w']
    phase=(recipe['h']*(tick-recipe['W_ref']*samples))%samples
    # Exact axes avoid tiny spurious crossings along lattice lines.
    axes={0:(1.,0.),samples:(0.,1.),2*samples:(-1.,0.),3*samples:(0.,-1.)}
    if 4*phase in axes:return np.array(axes[4*phase])
    angle=math.tau*(phase/samples)
    return np.array([math.cos(angle),math.sin(angle)])


class _PlaneCache:
    """Owned float64/valid slabs; cache hits retain FieldSource identity checks."""

    def __init__(self,source,budget):
        self.source=source
        self.capacity=budget//(source.shape[1]*source.shape[2]*9+512)
        self.planes=OrderedDict()

    def get(self,z):
        if not self.capacity:return None
        low=[z,0,0];high=[z+1,*self.source.shape[1:]]
        if z in self.planes:
            self.source.validate_region(low,high)
            self.planes.move_to_end(z)
        else:
            if len(self.planes)>=self.capacity:self.planes.popitem(last=False)
            values,valid=self.source.read(low,high)
            values.flags.writeable=False;valid.flags.writeable=False
            self.planes[z]=(values,valid)
        return self.planes[z]


def _coefficients(source,plane,p,v,shape,cuts,target,iz):
    if plane is None:
        for lo,hi in zip(cuts,cuts[1:]):
            midpoint=p+v*((lo+hi)/2)
            ix,iy=np.minimum(np.maximum(np.floor(midpoint).astype(int),0),shape-1)
            values,valid=source.read([iz,int(iy),int(ix)],[iz+1,int(iy)+2,int(ix)+2])
            if not valid.all():
                yield lo,hi,False,0.,0.,0.
                continue
            f00,f10,f01,f11=map(float,values[0].ravel()[[0,1,2,3]])
            x,y=p+v*lo-[ix,iy];dx,dy=v*(hi-lo)
            cross=f11-f10-f01+f00
            a=cross*dx*dy
            b=(f10-f00)*dx+(f01-f00)*dy+cross*(x*dy+y*dx)
            c=f00+(f10-f00)*x+(f01-f00)*y+cross*x*y-target
            yield lo,hi,True,a,b,c
        return
    # Fixed-size workspace, never retain coefficients across rays or turns.
    values,valid=plane[0][0],plane[1][0]
    for start in range(0,len(cuts)-1,256):
        lo=np.asarray(cuts[start:start+256]);hi=np.asarray(cuts[start+1:start+257])
        lo=lo[:len(hi)]
        midpoint=p[:,None]+v[:,None]*((lo+hi)/2)
        indices=np.minimum(np.maximum(np.floor(midpoint).astype(int),0),(shape-1)[:,None])
        ix,iy=indices
        support=valid[iy,ix]&valid[iy,ix+1]&valid[iy+1,ix]&valid[iy+1,ix+1]
        ix,iy=indices[:,support]
        f00=values[iy,ix];f10=values[iy,ix+1];f01=values[iy+1,ix];f11=values[iy+1,ix+1]
        x,y=p[:,None]+v[:,None]*lo[support]-indices[:,support]
        dx,dy=v[:,None]*(hi[support]-lo[support])
        cross=f11-f10-f01+f00
        a=np.zeros(len(lo));b=np.zeros(len(lo));c=np.zeros(len(lo))
        a[support]=cross*dx*dy
        b[support]=(f10-f00)*dx+(f01-f00)*dy+cross*(x*dy+y*dx)
        c[support]=f00+(f10-f00)*x+(f01-f00)*y+cross*x*y-target
        yield from zip(lo,hi,support,a,b,c)


def _node(source,recipe,tick,iz,cache=None):
    z=float(source.origin[2]+iz*source.spacing[2]);cp=np.asarray(recipe['umbilicus'])
    if z<cp[0,2] or z>cp[-1,2]:return 'center_unsupported',[],None
    center=np.array([np.interp(z,cp[:,2],cp[:,i]) for i in range(2)])
    p=(center-source.origin[:2])/source.spacing[:2]
    shape=np.array(source.shape[:0:-1])-1
    if np.any(p<0) or np.any(p>shape):return 'center_unsupported',[],None
    direction=_direction(tick,recipe);v=direction/source.spacing[:2]
    far=min((float((n-q)/d) if d>0 else float(-q/d))
            for q,d,n in zip(p,v,shape) if d!=0)
    if far<=0:return 'no_root',[],None
    cuts=[0.,far]
    for q,d,n in zip(p,v,shape):
        if d:
            cuts.extend(float(r) for r in (np.arange(n+1)-q)/d if 0<r<far)
    cuts=sorted(set(cuts));roots=[];degenerate=False;unsupported=False;last_slope=None
    target=(tick/recipe['samples_per_w'])*source.sheet_step
    tolerance=recipe['numeric_contract']['policy']['root_relative_tolerance']*max(1.,far)
    plane=cache.get(iz) if cache is not None else None
    for lo,hi,support,a,b,c in _coefficients(source,plane,p,v,shape,cuts,target,iz):
        if not support:unsupported=True;continue
        candidates,bad=quadratic_roots(a,b,c)
        degenerate |= bad
        for u in candidates:
            radius=lo+u*(hi-lo)
            slope=2*a*u+b
            if roots and abs(radius-roots[-1])<=tolerance:
                # Proximity cannot prove equality: two roots can straddle a
                # cell boundary. Only the exact shared endpoint is deduplicated.
                if radius==lo and roots[-1]==lo:
                    if slope==0 or last_slope==0 or (slope>0)!=(last_slope>0):degenerate=True
                    last_slope=slope
                    continue
                degenerate=True
            roots.append(radius)
            last_slope=slope
    state=('unsupported' if unsupported else 'degenerate' if degenerate or (roots and roots[0]<=tolerance)
           else 'no_root' if not roots else 'multiple' if len(roots)>1 else 'unique')
    xyz=(*map(float,center+direction*roots[0]),z) if state=='unique' else None
    return state,roots,xyz


def _key(recipe_sha,tick,z):
    return f'{recipe_sha}:n:{tick}:{z}'


def _cell(recipe,recipe_sha,tick,z,nodes):
    key=f'{recipe_sha}:c:{tick}:{z}'
    keys=[_key(recipe_sha,tick+dw,z+dz) for dw,dz in ((0,0),(1,0),(1,1),(0,1))]
    if any(node[3]!='unique' for node in nodes):return tick,z,key,'node_rejected','[]'
    points=np.array([node[5:8] for node in nodes])
    edge=recipe['max_edge_world']
    if edge is not None and any(np.linalg.norm(points[a]-points[b])>edge for a,b in ((0,1),(1,2),(2,3),(3,0),(0,2))):
        return tick,z,key,'long_edge','[]'
    triangles=((0,1,2),(0,2,3))
    if any(np.linalg.norm(np.cross(points[b]-points[a],points[c]-points[a]))==0 for a,b,c in triangles):
        return tick,z,key,'degenerate','[]'
    return tick,z,key,'accepted',_json([[keys[i] for i in tri] for tri in triangles])


def _audit_sha(db,tick):
    return digest([_rows_sha(db,'SELECT * FROM range_nodes WHERE tick_w=? ORDER BY tick_z',(tick,)),
        _rows_sha(db,'SELECT * FROM range_cells WHERE tick_w=? ORDER BY tick_z',(tick-1,))])


class RangeDB(MeshDB):
    """Range-only extension; old scalar bundles and MeshDB schema are unchanged."""

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        try:
            self._ensure_schema()
        except BaseException:
            self.close();raise

    def _ensure_schema(self):
        # Base initialization holds the writer lock, including its resume callback.
        if self.lock is None or self.lock.closed:raise ValueError('range schema requires writer lock')
        schema={
            'range_nodes':'CREATE TABLE range_nodes(tick_w INTEGER,tick_z INTEGER,key TEXT UNIQUE,state TEXT,roots TEXT,x REAL,y REAL,z REAL,PRIMARY KEY(tick_w,tick_z))',
            'range_cells':'CREATE TABLE range_cells(tick_w INTEGER,tick_z INTEGER,key TEXT UNIQUE,state TEXT,triangles TEXT,PRIMARY KEY(tick_w,tick_z))',
            'range_audit':'CREATE TABLE range_audit(tick_w INTEGER PRIMARY KEY,sha256 TEXT)',
        }
        existing=dict(self.db.execute("SELECT name,sql FROM sqlite_master WHERE name IN ('range_nodes','range_cells','range_audit')"))
        if any(existing[name]!=schema[name] for name in existing):
            raise ValueError('range schema differs')
        if len(existing)==len(schema):return
        # Missing audit is recoverable only before ANY work, independently of cursor.
        pristine=(self.get('next_tile')==0 and self.get('complete') is False
            and self.get('binding')==self.binding
            and set(row[0] for row in self.db.execute('SELECT key FROM meta'))=={'binding','next_tile','complete'}
            and list(self.db.execute('SELECT id,value FROM levels ORDER BY id'))==[(0,0.)])
        tables=('vertices','faces','tiles','edges','components',*existing)
        if not pristine or any(self.db.execute('SELECT COUNT(*) FROM '+table).fetchone()[0] for table in tables):
            raise ValueError('missing range schema with non-pristine committed state')
        with self.db:
            self.db.execute('BEGIN IMMEDIATE')
            for name,sql in schema.items():
                if name not in existing:self.db.execute(sql)

    def validate_tiles(self):
        super().validate_tiles()
        self._ensure_schema()
        _validate_audit(self.db,self.binding['identity'],self.get('next_tile'))

    def column(self,tick,nodes,cells):
        for row in nodes:
            self.db.execute('INSERT INTO range_nodes VALUES(?,?,?,?,?,?,?,?)',row)
            if row[3]=='unique':
                self.add(0,[row[2]],np.array([row[5:8]]),np.empty((0,3),int))
        for row in cells:
            self.db.execute('INSERT INTO range_cells VALUES(?,?,?,?,?)',row)
            for index,keys in enumerate(json.loads(row[4])):
                points=np.array([self.db.execute('SELECT x,y,z FROM vertices WHERE level=0 AND key=?',(key,)).fetchone() for key in keys])
                before=self.last_id('faces')
                self.add(0,keys,points,np.array([[0,1,2]]))
                if self.last_id('faces')!=before+1:raise ValueError('accepted range triangle was rejected')
                self.db.execute('UPDATE faces SET key=? WHERE id=?',(row[2]+f':{index}',before+1))
        self.db.execute('INSERT INTO range_audit VALUES(?,?)',(tick,_audit_sha(self.db,tick)))
        self.checkpoint_tile(tick-self.binding['identity']['ticks'][0])


def _validate_audit(db,identity,count):
    low,high=identity['ticks'];nz=identity['recipe']['geometry']['shape_zyx'][0]
    rows=db.execute('SELECT tick_w,sha256 FROM range_audit ORDER BY tick_w')
    seen=0
    for tick,value in rows:
        if tick!=low+seen or value!=_audit_sha(db,tick):raise ValueError('range audit checksum/cursor differs')
        if db.execute('SELECT COUNT(*) FROM range_nodes WHERE tick_w=?',(tick,)).fetchone()[0]!=nz:
            raise ValueError('range node inventory differs')
        if tick>low and db.execute('SELECT COUNT(*) FROM range_cells WHERE tick_w=?',(tick-1,)).fetchone()[0]!=nz-1:
            raise ValueError('range cell inventory differs')
        seen+=1
    if seen!=count or count>high-low+1:raise ValueError('range audit cursor differs')
    if db.execute('SELECT COUNT(*) FROM range_nodes').fetchone()[0]!=count*nz or db.execute(
            'SELECT COUNT(*) FROM range_cells').fetchone()[0]!=max(0,count-1)*(nz-1):
        raise ValueError('range audit inventory differs')


def _identity(recipe,ticks):
    return dict(kind='continuous_range',recipe=recipe,recipe_sha256=digest(recipe),ticks=ticks,
                total_tiles=ticks[1]-ticks[0]+1,slot_semantics='slot0 is a range, not W=0')


def _finish(store,output):
    identity=store.binding['identity'];summary=store.finish()
    _validate_audit(store.db,identity,identity['total_tiles'])
    node_states=dict(store.db.execute('SELECT state,COUNT(*) FROM range_nodes GROUP BY state'))
    cell_states=dict(store.db.execute('SELECT state,COUNT(*) FROM range_cells GROUP BY state'))
    result=dict(schema='winding-range-v1',kind='continuous_range',status='RANGE_READY' if summary[0]['faces'] else 'EMPTY',
        recipe=identity['recipe'],recipe_sha256=identity['recipe_sha256'],ticks=identity['ticks'],
        complete=True,vertices=summary[0]['vertices'],faces=summary[0]['faces'],area_world2=summary[0]['area_world2'],
        node_states=node_states,cell_states=cell_states,physical_acceptance=False,self_intersection='UNMEASURED')
    # Keep the writer lease through publication; finish has committed all data.
    manifest=_manifest(identity['recipe']['geometry'],identity,summary,identity['total_tiles'],sha(output/'mesh/mesh.sqlite'))
    atomic_json(output/'mesh/MESH.json',manifest)
    result['mesh_manifest_sha256']=sha(output/'mesh/MESH.json')
    atomic_json(output/'RANGE.json',result)
    return result


def extract_range(recipe,w_range,output,resume=False,*,block_ticks=16,cache_bytes=32*1024**2):
    """Extract inclusive node bounds and half-open cell ownership; resume atomically.

    block_ticks controls transaction grouping only, never geometry or identity.
    Memory holds two Z columns, a bounded source-plane cache and at most 256
    cells of coefficient workspace (under 128 KiB of arrays). Slabs that
    exceed cache_bytes use the original cell reads; zero disables caching.
    Cache hits validate source identity, and source.validate before each commit
    still rejects any source mutation during cached numerical processing.
    """
    _integer(block_ticks,'block_ticks',1,256)
    _integer(cache_bytes,'cache_bytes',0,1024**3)
    if type(resume) is not bool:raise ValueError('resume must be boolean')
    if not isinstance(recipe,dict) or not {'field','geometry'}<=recipe.keys():
        raise ValueError('range recipe requires field and geometry')
    with FieldSource(recipe['field'],recipe['geometry']) as source:
        if resume and recipe['field'].get('kind') == 'zarr-v2':
            source.validate(hashes=True)
        cache=_PlaneCache(source,cache_bytes)
        normalized=_recipe(recipe,source);ticks=_bounds(w_range,normalized['samples_per_w'])
        identity=_identity(normalized,ticks);output=Path(output)
        if output.is_symlink():raise ValueError('range output cannot be a symlink')
        output.mkdir(exist_ok=resume)
        with RangeDB(output/'mesh',normalized['geometry'],identity,resume=resume) as store:
            first=ticks[0]+store.get('next_tile');nz=source.shape[0]
            previous=list(store.db.execute('SELECT * FROM range_nodes WHERE tick_w=? ORDER BY tick_z',(first-1,)))
            for start in range(first,ticks[1]+1,block_ticks):
                store.check()
                with store.db:
                    for tick in range(start,min(start+block_ticks,ticks[1]+1)):
                        nodes=[]
                        for z in range(nz):
                            state,roots,xyz=_node(source,normalized,tick,z,cache)
                            nodes.append((tick,z,_key(identity['recipe_sha256'],tick,z),state,_json(roots),*(xyz or (None,None,None))))
                        cells=[] if not previous else [_cell(normalized,identity['recipe_sha256'],tick-1,z,
                            [previous[z],nodes[z],nodes[z+1],previous[z+1]]) for z in range(nz-1)]
                        store.column(tick,nodes,cells);previous=nodes
                    source.validate()
            source.validate(hashes=True)
            return _finish(store,output)


class RangeReader:
    """Read a complete wrapper, including EMPTY, without opening original fields."""

    def __init__(self,path):
        self.root=Path(path);self.receipt=json.loads((self.root/'RANGE.json').read_bytes())
        self.receipt_sha=sha(self.root/'RANGE.json')
        r=self.receipt
        if r.get('schema')!='winding-range-v1' or r.get('kind')!='continuous_range' or r.get('complete') is not True or r.get('status') not in ('EMPTY','RANGE_READY'):
            raise ValueError('complete continuous_range wrapper required')
        self.mesh=MeshReader(self.root/'mesh')
        try:
            self.db=self.mesh.db;identity=self.mesh.manifest['identity']
            if identity!=_identity(r['recipe'],r['ticks']) or r['recipe_sha256']!=digest(r['recipe']) or r['mesh_manifest_sha256']!=self.mesh.manifest_sha:
                raise ValueError('range recipe/manifest binding differs')
            _validate_audit(self.db,identity,identity['total_tiles'])
            summary=self.mesh.manifest['levels'][0]
            if any(r[k]!=summary[k] for k in ('vertices','faces','area_world2')) or (r['status']=='EMPTY')!=(r['faces']==0):
                raise ValueError('range summary differs')
            for label,table in (('node_states','range_nodes'),('cell_states','range_cells')):
                if r[label]!=dict(self.db.execute('SELECT state,COUNT(*) FROM '+table+' GROUP BY state')):
                    raise ValueError('range state summary differs')
            self._validate_geometry()
        except BaseException:
            self.close();raise

    def _validate_geometry(self):
        if list(self.db.execute('SELECT id,value FROM levels ORDER BY id'))!=[(0,0.)]:
            raise ValueError('range storage level inventory differs')
        for table in ('vertices','faces'):
            if self.db.execute('SELECT COUNT(*) FROM '+table).fetchone()[0]!=self.receipt[table] or self.db.execute(
                    'SELECT COUNT(*) FROM '+table+' WHERE level!=0').fetchone()[0]:
                raise ValueError('range actual geometry inventory differs')
        n=0
        for tick,z,key,state,roots,x,y,zz in self.db.execute('SELECT * FROM range_nodes ORDER BY tick_w,tick_z'):
            vertex=self.db.execute('SELECT x,y,z FROM vertices WHERE level=0 AND key=?',(key,)).fetchone()
            if key!=_key(self.receipt['recipe_sha256'],tick,z) or (vertex!=(x,y,zz) if state=='unique' else vertex is not None):
                raise ValueError('range node geometry/state differs')
            n+=state=='unique'
        if n!=self.receipt['vertices']:raise ValueError('range vertex inventory differs')
        n=0
        for tick,z,key,state,triangles in self.db.execute('SELECT * FROM range_cells ORDER BY tick_w,tick_z'):
            recorded=json.loads(triangles)
            if (state=='accepted' and len(recorded)!=2) or (state!='accepted' and recorded):
                raise ValueError('range cell presence/state differs')
            for i,keys in enumerate(recorded):
                actual=self.db.execute('''SELECT a.key,b.key,c.key FROM faces f JOIN vertices a ON a.id=f.a
                    JOIN vertices b ON b.id=f.b JOIN vertices c ON c.id=f.c WHERE f.level=0 AND f.key=?''',(key+f':{i}',)).fetchall()
                if actual!=[tuple(keys)]:raise ValueError('range cell face presence/orientation differs')
                n+=1
        if n!=self.receipt['faces']:raise ValueError('range face inventory differs')

    def validate(self):
        self.mesh.validate()
        if sha(self.root/'RANGE.json')!=self.receipt_sha:raise ValueError('range wrapper changed during reading')

    def close(self):self.mesh.close()
    def __enter__(self):return self
    def __exit__(self,*unused):self.close()


def join_ranges(inputs,output):
    """Join only equal recipes and contiguous domains; compare every shared row."""
    if not inputs:raise ValueError('join requires input ranges')
    output=Path(output)
    with ExitStack() as stack:
        readers=[stack.enter_context(RangeReader(path)) for path in inputs]
        readers.sort(key=lambda r:tuple(r.receipt['ticks']))
        recipe=readers[0].receipt['recipe'];low,high=readers[0].receipt['ticks']
        for reader in readers:
            if reader.receipt['recipe']!=recipe:raise ValueError('incompatible range recipe')
            a,b=reader.receipt['ticks']
            if a>high:raise ValueError('gap between range intervals')
            high=max(high,b)
            root=reader.root.resolve();dest=output.resolve()
            if dest==root or root in dest.parents or dest in root.parents:
                raise ValueError('join output must be separate from inputs')
        identity=_identity(recipe,[low,high]);output.mkdir(exist_ok=False)
        with RangeDB(output/'mesh',recipe['geometry'],identity) as store:
            for tick in range(low,high+1):
                nodes=cells=None
                for reader in readers:
                    a,b=reader.receipt['ticks']
                    if a<=tick<=b:
                        rows=list(reader.db.execute('SELECT * FROM range_nodes WHERE tick_w=? ORDER BY tick_z',(tick,)))
                        if nodes is not None and nodes!=rows:raise ValueError('overlapping range node state/XYZ differs')
                        nodes=rows
                    if a<tick<=b:
                        rows=list(reader.db.execute('SELECT * FROM range_cells WHERE tick_w=? ORDER BY tick_z',(tick-1,)))
                        if cells is not None and cells!=rows:raise ValueError('overlapping range cell state/presence/orientation/rejection differs')
                        cells=rows
                with store.db:
                    store.column(tick,nodes,cells or [])
            for reader in readers:reader.validate()
            return _finish(store,output)


def reject_scalar_range(path):
    """Guard both a range wrapper and a detached inner mesh bundle, even incomplete."""
    path=Path(path)
    for name in ('RANGE.json','MESH.json','RUN.json','mesh/RUN.json'):
        member=path/name
        if member.is_file():
            record=json.loads(member.read_bytes())
            if record.get('kind')=='continuous_range' or record.get('identity',{}).get('kind')=='continuous_range':
                raise ValueError('continuous_range is not a scalar W level; use the range patch pipeline')
