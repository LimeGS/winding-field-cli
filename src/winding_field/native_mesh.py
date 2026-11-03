"""Batched native mesh storage; the vendored numerical extraction kernel is unchanged."""
import hashlib
import itertools
import json
import marshal
import math
from pathlib import Path
from importlib.metadata import version

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from . import surface_mesh as core
from .util import atomic_json,file_sha


def rows_sha(db,query,args=()):
    """Exactly the core's canonical JSON-lines digest, encoded in bounded batches."""
    result=hashlib.sha256(); cursor=db.execute(query,args)
    encoder=json.JSONEncoder(separators=(',',':'),allow_nan=False)
    while rows:=cursor.fetchmany(8192):
        if any(isinstance(v,str) and '],[' in v for row in rows for v in row):
            raw='\n'.join(encoder.encode(row) for row in rows)+'\n'
        else:
            raw=encoder.encode(rows)[1:-1].replace('],[',']\n[')+'\n'
        result.update(raw.encode())
    return result.hexdigest()


def _topology(ids,triangles,areas):
    n=len(ids)
    directed=np.concatenate([triangles[:,[0,1]],triangles[:,[1,2]],triangles[:,[2,0]]])
    edges,inverse=np.unique(np.sort(directed,axis=1),axis=0,return_inverse=True)
    owners=np.tile(np.arange(n),3)
    first=np.full(len(edges),n,np.int64); np.minimum.at(first,inverse,owners)
    other=first[inverse]; used=owners!=other
    graph=coo_matrix((np.ones(int(used.sum()),np.uint8),(owners[used],other[used])),shape=(n,n)).tocsr()
    count,labels=connected_components(graph,directed=False)
    roots=np.full(count,np.iinfo(np.int64).max,np.int64); np.minimum.at(roots,labels,ids)
    counts=np.bincount(inverse)
    signs=np.bincount(inverse,weights=np.where(directed[:,0]<directed[:,1],1,-1)).astype(np.int64)
    groups=(roots,np.bincount(labels),np.bincount(labels,weights=areas))
    return roots[labels],(edges,ids[first],counts,signs),groups


class MeshDB(core.MeshDB):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._tables()

    def _tables(self):
        if self.db.execute('PRAGMA temp_store').fetchone()[0]!=2:
            self.db.execute('PRAGMA temp_store=MEMORY')
        # Main-database scratch stays in the owned output and its rollback journal.
        for statement in (
            'CREATE TABLE IF NOT EXISTS _incoming(local INTEGER PRIMARY KEY,key TEXT,x REAL,y REAL,z REAL)',
            'CREATE INDEX IF NOT EXISTS _incoming_key ON _incoming(key,local)',
            'CREATE TABLE IF NOT EXISTS _edge_batch(a INTEGER,b INTEGER,face INTEGER,count INTEGER,sign INTEGER,PRIMARY KEY(a,b))',
            'CREATE TABLE IF NOT EXISTS _groups(id INTEGER PRIMARY KEY,level INTEGER,faces INTEGER,area REAL)',
            'CREATE TABLE IF NOT EXISTS _roots(id INTEGER PRIMARY KEY,root INTEGER)',
            'CREATE TABLE IF NOT EXISTS _joins(a INTEGER,b INTEGER,PRIMARY KEY(a,b))',
        ):
            self.db.execute(statement)

    def row_digest(self,table,first,last):
        fields='id,level,key,x,y,z,local' if table=='vertices' else 'id,level,a,b,c,area,key'
        return rows_sha(self.db,'SELECT '+fields+' FROM '+table+' WHERE id>? AND id<=? ORDER BY id',(first,last))

    def _connect(self,level,edge_data,groups):
        edges,owners,counts,signs=edge_data
        roots,sizes,areas=groups
        self.db.executemany('INSERT INTO _groups VALUES (?,?,?,?)',
            ((int(r),level,int(n),float(a)) for r,n,a in zip(roots,sizes,areas)))
        self.db.execute('DELETE FROM _edge_batch'); self.db.execute('DELETE FROM _joins')
        self.db.executemany('INSERT INTO _edge_batch VALUES (?,?,?,?,?)',
            ((int(a),int(b),int(f),int(n),int(s)) for (a,b),f,n,s in zip(edges,owners,counts,signs)))
        self.db.execute('INSERT OR IGNORE INTO _joins '
            'SELECT a.parent,b.parent FROM _edge_batch n JOIN edges e ON e.level=? AND e.a=n.a AND e.b=n.b '
            'JOIN faces a ON a.id=e.face JOIN faces b ON b.id=n.face',(level,))
        for a,b in self.db.execute('SELECT a,b FROM _joins'):
            a,b=self.find(a),self.find(b)
            if a!=b:
                self.db.execute('UPDATE faces SET parent=? WHERE id=?',(min(a,b),max(a,b)))
        self.db.execute('INSERT INTO edges SELECT ?,a,b,face,count,sign FROM _edge_batch WHERE 1 '
            'ON CONFLICT(level,a,b) DO UPDATE SET count=edges.count+excluded.count,sign=edges.sign+excluded.sign',
            (level,))
        self.db.execute('DELETE FROM _edge_batch'); self.db.execute('DELETE FROM _joins')

    def add(self,level,keys,vertices,faces):
        vertices=np.asarray(vertices); faces=np.asarray(faces)
        if (vertices.ndim!=2 or vertices.shape[1:]!=(3,) or not np.isfinite(vertices).all()
                or faces.ndim!=2 or faces.shape[1:]!=(3,) or faces.dtype.kind not in 'iu'
                or len(keys)!=len(vertices) or len(faces) and (faces.min()<0 or faces.max()>=len(vertices))):
            raise ValueError('invalid mesh arrays')
        self.admit()
        self.db.execute('DELETE FROM _incoming')
        self.db.executemany('INSERT INTO _incoming VALUES (?,?,?,?,?)',
            ((i,str(k),*map(float,v)) for i,(k,v) in enumerate(zip(keys,vertices))))
        if self.db.execute('SELECT 1 FROM _incoming GROUP BY key HAVING min(x)!=max(x) OR min(y)!=max(y) OR min(z)!=max(z) LIMIT 1').fetchone():
            raise ValueError('shared lattice vertex coordinates differ')
        if self.db.execute('SELECT 1 FROM _incoming i JOIN vertices v ON v.level=? AND v.key=i.key '
            'WHERE v.x!=i.x OR v.y!=i.y OR v.z!=i.z LIMIT 1',(level,)).fetchone():
            raise ValueError('shared lattice vertex coordinates differ')
        local=self.db.execute('SELECT coalesce(max(local),0)+1 FROM vertices WHERE level=?',(level,)).fetchone()[0]
        self.db.execute('INSERT INTO vertices(level,key,x,y,z,local) '
            'SELECT ?,i.key,i.x,i.y,i.z,?+row_number() OVER (ORDER BY fresh.first)-1 '
            'FROM (SELECT min(i.local) first FROM _incoming i LEFT JOIN vertices v ON v.level=? AND v.key=i.key '
            'WHERE v.id IS NULL GROUP BY i.key) fresh JOIN _incoming i ON i.local=fresh.first',
            (level,local,level))
        ids=np.fromiter((r[0] for r in self.db.execute('SELECT v.id FROM _incoming i JOIN vertices v '
                        'ON v.level=? AND v.key=i.key ORDER BY i.local',(level,))),np.int64,count=len(vertices))
        triangles=ids[faces]
        v=vertices[faces]
        areas=np.linalg.norm(np.cross(v[:,1]-v[:,0],v[:,2]-v[:,0]),axis=1)/2
        valid=(triangles[:,0]!=triangles[:,1])&(triangles[:,0]!=triangles[:,2])&(triangles[:,1]!=triangles[:,2])&(areas>0)
        triangles=triangles[valid]; areas=areas[valid]
        if not np.isfinite(areas).all():
            raise ValueError('mesh area exceeds numerical capacity')
        if len(triangles):
            first=self.last_id('faces')+1
            face_ids=np.arange(first,first+len(triangles),dtype=np.int64)
            parents,edges,groups=_topology(face_ids,triangles,areas)
            sorted_ids=np.sort(triangles,axis=1)
            self.db.executemany('INSERT INTO faces(id,level,a,b,c,parent,area,key) VALUES (?,?,?,?,?,?,?,?)',
                ((int(i),level,int(a),int(b),int(c),int(p),float(area),f'{key[0]},{key[1]},{key[2]}')
                 for i,(a,b,c),p,area,key in zip(face_ids,triangles,parents,areas,sorted_ids)))
            self._connect(level,edges,groups)
        self.db.execute('DELETE FROM _incoming')
        self.admit()

    def rebuild_topology(self):
        self._tables()
        self.db.execute('DELETE FROM edges'); self.db.execute('DELETE FROM _groups'); self.db.execute('DELETE FROM _roots')
        cursor=self.db.execute('SELECT id,level,a,b,c,area FROM faces ORDER BY id')
        while rows:=cursor.fetchmany(8192):
            values=np.array([r[:5] for r in rows],np.int64); areas=np.array([r[5] for r in rows])
            for level in np.unique(values[:,1]):
                selected=values[:,1]==level; part=values[selected]
                parents,edges,groups=_topology(part[:,0],part[:,2:5],areas[selected])
                self.db.executemany('UPDATE faces SET parent=? WHERE id=?',((int(p),int(i)) for p,i in zip(parents,part[:,0])))
                self._connect(int(level),edges,groups)
            self.admit()

    def finish(self):
        self.check()
        if self.get('complete'):
            result=self.get('summary')
            for item in result:
                self.export(item['id'])
            return result
        with self.db:
            self.db.execute('DELETE FROM _roots'); self.db.execute('DELETE FROM components')
            cursor=self.db.execute('SELECT id FROM _groups ORDER BY id')
            while rows:=cursor.fetchmany(8192):
                self.db.executemany('INSERT INTO _roots VALUES (?,?)',[(i,self.find(i)) for (i,) in rows])
                self.admit()
            self.db.execute('INSERT INTO components SELECT g.level,r.root,sum(g.faces),sum(g.area) '
                            'FROM _groups g JOIN _roots r ON r.id=g.id GROUP BY g.level,r.root')
            self.db.execute('UPDATE faces SET parent=(SELECT root FROM _roots WHERE id=faces.parent) '
                            'WHERE parent!=(SELECT root FROM _roots WHERE id=faces.parent)')
        summary=[]
        for level,value in self.db.execute('SELECT id,value FROM levels ORDER BY id'):
            n=lambda table:self.db.execute('SELECT count(*) FROM '+table+' WHERE level=?',(level,)).fetchone()[0]
            edges=self.db.execute('SELECT sum(count=1),sum(count>2),sum(count=2 AND sign!=0) FROM edges WHERE level=?',(level,)).fetchone()
            duplicates=self.db.execute('SELECT coalesce(sum(n-1),0) FROM (SELECT count(*) n FROM faces '
                'WHERE level=? GROUP BY key HAVING count(*)>1)',(level,)).fetchone()[0]
            summary.append(dict(id=level,level=value,vertices=n('vertices'),faces=n('faces'),components=n('components'),
                area_world2=self.db.execute('SELECT coalesce(sum(area),0) FROM faces WHERE level=?',(level,)).fetchone()[0],
                boundary_edges=edges[0] or 0,nonmanifold_edges=edges[1] or 0,
                inconsistent_edges=edges[2] or 0,duplicate_faces=duplicates))
        parts=[rows_sha(self.db,'SELECT id,parent FROM faces ORDER BY id'),
               rows_sha(self.db,'SELECT * FROM edges ORDER BY level,a,b'),
               rows_sha(self.db,'SELECT * FROM components ORDER BY level,root')]
        derived=hashlib.sha256(json.dumps([parts,summary],sort_keys=True,allow_nan=False).encode()).hexdigest()
        with self.db:
            self.set('summary',summary); self.set('derived_sha',derived); self.set('complete',True)
        for item in summary:
            self.export(item['id'])
        return summary


def extract_mesh(source,output,levels,*,tile=128,level_batch=4,resume=False,stop=None,reserve_bytes=0):
    tile=core._integer(tile,'tile'); level_batch=core._integer(level_batch,'level_batch')
    reserve_bytes=core._integer(reserve_bytes,'reserve_bytes',0); levels=core._levels(levels)
    if tile>256 or min(source.shape)<2 or type(resume) is not bool or stop is not None and not callable(stop):
        raise ValueError('invalid bounded native extraction settings')
    source.validate(hashes=resume)
    tile_bounds=getattr(source,'tile_bounds',None)
    if tile_bounds is not None and not callable(tile_bounds):
        raise ValueError('invalid source tile bounds provider')
    grid=tuple((n-2)//tile+1 for n in source.shape); total=math.prod(grid)
    identity=dict(source=source.identity,levels=levels,tile=tile,total_tiles=total,
        algorithm='lewiner-shifted-f32-adaptive-global-edges-v2+batched-store-v1',numpy=np.__version__,
        skimage=version('scikit-image'),storage_sha256=file_sha(__file__),
        code={Path(core.__file__).name:file_sha(core.__file__),
              'surface_sources.py':file_sha(Path(core.__file__).with_name('surface_sources.py')),
              'field_store.py':file_sha(core.field_store.__file__)},
        decoder_runtime=hashlib.sha256(marshal.dumps(core.field_store._decode.__code__)).hexdigest())
    metadata=source.identity['geometry']; output=Path(output)
    core.check_resources(output.parent,reserve_bytes=reserve_bytes)
    with MeshDB(output,metadata,identity,resume=resume,reserve_bytes=reserve_bytes) as store:
        first=store.get('next_tile')
        if type(first) is not int or not 0<=first<=total:
            raise ValueError('invalid mesh cursor')
        for index,key in enumerate(itertools.product(*(range(n) for n in grid))):
            if index<first:
                continue
            if (output/'STOP').exists() or stop and stop():
                result=dict(status='CHECKPOINTED_STOP',next_tile=index,total_tiles=total,physical_acceptance=False)
                atomic_json(output/'STATUS.json',result); return result
            store.check()
            low=np.array(key)*tile; high=np.minimum(low+tile,np.array(source.shape)-1)
            bounds=tile_bounds(low,high+1) if tile_bounds is not None else None
            if bounds is not None and (not isinstance(bounds,(tuple,list)) or len(bounds)!=2
                    or any(isinstance(v,(bool,np.bool_)) or not isinstance(v,(int,float,np.integer,np.floating))
                           or not math.isfinite(v) for v in bounds) or bounds[0]>bounds[1]):
                raise ValueError('invalid source tile bounds')
            eligible=[j for j,level in enumerate(levels) if bounds is None or bounds[0]<=level<=bounds[1]]
            if eligible:
                core.check_resources(output,reserve_bytes=reserve_bytes+int(np.prod(high-low))*128)
                values,valid=source.read(low,high+1)
            with store.db:
                for start in range(0,len(eligible),level_batch):
                    for j in eligible[start:start+level_batch]:
                        keys,vertices,faces=core._tile_mesh(values,valid,low,levels[j],source.origin,source.spacing)
                        if len(faces):
                            store.add(j,keys,vertices,faces)
                store.checkpoint_tile(index)
            atomic_json(output/'STATUS.json',dict(status='EXTRACTING',next_tile=index+1,total_tiles=total,
                                                range_excluded=not eligible))
        source.validate(); details=store.finish()
    result=core._manifest(metadata,identity,details,total,file_sha(output/'mesh.sqlite'))
    atomic_json(output/'MESH.json',result); atomic_json(output/'STATUS.json',result)
    return result
