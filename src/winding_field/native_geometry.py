"""Bounded native charts and shared-edge UV assembly, preserving source IDs."""
import json
from functools import lru_cache
from importlib.metadata import version

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from .flatten import components,cross2,overlap_pairs,parameterize


@lru_cache(maxsize=1)
def _solver():
    try:
        import igl
    except ImportError:
        return None
    return igl


def solver_identity():
    return dict(solver='scipy-lsqr' if _solver() is None else 'libigl-lscm',numpy=np.__version__,
                scipy=version('scipy'),libigl=None if _solver() is None else version('libigl'))


def _lscm(vertices,faces):
    igl=_solver()
    if igl is None:
        return parameterize(vertices,faces)
    first=int(np.argmax(np.linalg.norm(vertices-vertices.mean(0),axis=1)))
    second=int(np.argmax(np.linalg.norm(vertices-vertices[first],axis=1)))
    pins=np.array([first,second],np.int64)
    fixed=np.array([[0.,0.],[np.linalg.norm(vertices[second]-vertices[first]),0.]])
    uv,_=igl.lscm(vertices.astype(float),faces.astype(np.int64),pins,fixed)
    if uv.shape!=(len(vertices),2) or not np.isfinite(uv).all():
        raise ValueError('native LSCM did not produce finite UVs')
    tri=uv[faces]; signed=cross2(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    if signed.sum()<0:
        uv[:,1]*=-1; signed=-signed
    if np.any(signed<=1e-12):
        raise ValueError('native LSCM has flipped or collapsed faces')
    physical=vertices[faces]
    area=np.linalg.norm(np.cross(physical[:,1]-physical[:,0],physical[:,2]-physical[:,0]),axis=1)
    uv*=np.sqrt(area.sum()/signed.sum()); uv-=uv.min(0)
    return uv,dict(_validate(vertices,faces,uv),method='LSCM',solver='libigl')


def _schema(store):
    store.db.executescript('CREATE TABLE IF NOT EXISTS charts('
        'name TEXT PRIMARY KEY,component INTEGER,area REAL,faces INTEGER,status TEXT,uv_path TEXT,'
        'minx REAL,maxx REAL,miny REAL,maxy REAL,reason TEXT,quality TEXT);'
        'CREATE INDEX IF NOT EXISTS chart_state ON charts(status,component);'
        'CREATE TABLE IF NOT EXISTS edges(a INTEGER,b INTEGER,chart TEXT,ia INTEGER,ib INTEGER,orientation INTEGER,PRIMARY KEY(a,b,chart));'
        'CREATE INDEX IF NOT EXISTS chart_edges ON edges(chart);'
        'CREATE TABLE IF NOT EXISTS frontier(child TEXT,parent TEXT,n INTEGER,PRIMARY KEY(child,parent));'
        'CREATE INDEX IF NOT EXISTS frontier_order ON frontier(n DESC,child,parent);'
        'CREATE TABLE IF NOT EXISTS cuts(a INTEGER,b INTEGER,chart TEXT,parent TEXT,PRIMARY KEY(a,b,chart,parent));')


def _queue(store,name,child=False):
    store.db.execute('INSERT OR REPLACE INTO frontier '
        'SELECT e.chart,p.chart,count(*) FROM edges e JOIN edges p ON e.a=p.a AND e.b=p.b AND e.chart!=p.chart '
        'JOIN charts c ON c.name=e.chart JOIN charts d ON d.name=p.chart '
        "WHERE c.status='LOCAL' AND d.status='MAPPED' AND c.component=d.component AND "+
        ('e.chart' if child else 'p.chart')+'=? GROUP BY e.chart,p.chart',(name,))


def _pieces(v,f,face_ids,vertex_ids,minimum=64):
    # Shared edge IDs connect faces; a point contact cannot pin a separate LSCM island.
    edges=np.sort(np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1)
    _,edge_ids=np.unique(edges,axis=0,return_inverse=True)
    groups=list(components(edge_ids.reshape(3,len(f)).T))
    if len(groups)>1:
        for group in groups:
            yield from _pieces(v,f[group],face_ids[group],vertex_ids,minimum)
        return
    used,local=np.unique(f,return_inverse=True)
    vertices=v[used]; faces=local.reshape(-1,3)
    try:
        try:
            uv,quality=_project(vertices,faces)
        except ValueError:
            uv,quality=_lscm(vertices,faces)
    except (ValueError,RuntimeError) as error:
        if len(f)>minimum:
            mid=len(f)//2
            yield from _pieces(v,f[:mid],face_ids[:mid],vertex_ids,minimum)
            yield from _pieces(v,f[mid:],face_ids[mid:],vertex_ids,minimum)
            return
        uv=np.empty((0,2)); quality={'error':str(error)}
    yield vertices,faces,uv,face_ids,vertex_ids[used],quality


def _project(vertices,faces):
    centred=vertices-vertices.mean(0)
    _,basis=np.linalg.eigh(centred.T@centred)
    uv=centred@basis[:,[-1,-2]]
    tri=uv[faces]; signed=cross2(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    if signed.sum()<0:
        uv[:,1]*=-1; signed=-signed
    physical=vertices[faces]
    areas=np.linalg.norm(np.cross(physical[:,1]-physical[:,0],physical[:,2]-physical[:,0]),axis=1)
    if np.any(areas<=1e-12) or np.any(signed<.98*areas):
        raise ValueError('chart is not locally near-planar')
    uv*=np.sqrt(areas.sum()/signed.sum()); uv-=uv.min(0)
    quality=_validate(vertices,faces,uv)
    if quality['edge_stretch_min']<.98 or quality['edge_stretch_max']>1.02:
        raise ValueError('projection changes local metric')
    return uv,dict(quality,method='tangent-projection')


def flatten_mesh(reader,store,chart_faces=1024,until=None,max_batches=None):
    if type(chart_faces) is not int or not 1<=chart_faces<=8192:
        raise ValueError('chart_faces must be1..8192')
    _schema(store)
    total,area,last=store.get('source_faces'),store.get('source_area'),store.get('last_face')
    if total is None:
        total,area,last=reader.db.execute('SELECT count(*),coalesce(sum(area),0),coalesce(max(id),0) FROM faces').fetchone()
    if not total or area<=0:
        raise ValueError('empty native surface')
    with store.db:
        store.set('source_faces',total); store.set('source_area',area); store.set('last_face',last)
    store.phase='flatten'
    cursor=store.get('face_cursor',0)
    batches=0
    while max_batches is None or batches<max_batches:
        store.check(); reader.validate()
        if until is not None:
            available=store.db.execute("SELECT coalesce(sum(area),0) FROM charts WHERE status!='FAILED'").fetchone()[0]
            if available>=until*area:
                break
        rows=reader.db.execute('SELECT id,a,b,c,parent FROM faces WHERE id>? ORDER BY id LIMIT ?',
                               (cursor,chart_faces)).fetchall()
        if not rows:
            with store.db:
                store.set('flatten_complete',True)
            break
        rows=np.asarray(rows,np.int64)
        vi,faces=np.unique(rows[:,1:4],return_inverse=True)
        vertices=np.array([reader.db.execute('SELECT x,y,z FROM vertices WHERE id=?',(int(i),)).fetchone() for i in vi])
        faces=faces.reshape(-1,3)
        records=[]; edges=[]
        for component in np.unique(rows[:,4]):
            selection=np.flatnonzero(rows[:,4]==component)
            for connected in components(faces[selection]):
                subset=selection[connected]
                for v,f,uv,ids,source_ids,quality in _pieces(vertices,faces[subset],rows[subset,0],vi):
                    store.check()
                    name=f'chart-{int(ids.min()):012d}'
                    area2=np.linalg.norm(np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]]),axis=1)
                    store.save_arrays('charts/'+name+'.npz',vertices=v,faces=f,uv=uv,face_ids=ids,vertex_ids=source_ids)
                    status='LOCAL' if len(uv) else 'FAILED'
                    records.append((name,int(component),float(area2.sum()/2),len(f),status,
                                    json.dumps({'local':quality}),quality.get('error')))
                    if len(uv):
                        local=np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]])
                        source=source_ids[local]; orientation=np.where(source[:,0]<source[:,1],1,-1)
                        order=np.argsort(source,axis=1)
                        local=np.take_along_axis(local,order,axis=1); source=np.take_along_axis(source,order,axis=1)
                        _,first,counts=np.unique(source,axis=0,return_index=True,return_counts=True)
                        for index in first[counts==1]:
                            a,b=source[index]; ia,ib=local[index]
                            edges.append((int(a),int(b),name,int(ia),int(ib),int(orientation[index])))
        with store.db:
            store.db.executemany('INSERT OR IGNORE INTO charts(name,component,area,faces,status,quality,reason) VALUES (?,?,?,?,?,?,?)',records)
            store.db.executemany('INSERT OR IGNORE INTO edges VALUES (?,?,?,?,?,?)',edges)
            for record in records:
                _queue(store,record[0],child=True)
            store.set('face_cursor',int(rows[-1,0]))
            if int(rows[-1,0])==last:
                store.set('flatten_complete',True)
            store.set('elapsed_s',store.elapsed)
        cursor=int(rows[-1,0])
        batches+=1
    return dict(source_faces=total,source_area=area,processed_faces=store.db.execute('SELECT coalesce(sum(faces),0) FROM charts').fetchone()[0])


def weld(uv,faces,fixed,target,vertices):
    """Validated cotangent displacement, with one bounded SLIM recovery step."""
    centre=uv[fixed].mean(0); destination=target.mean(0)
    u,_,vt=np.linalg.svd((uv[fixed]-centre).T@(target-destination))
    if np.linalg.det(u@vt)<0:
        u[:,-1]*=-1
    reference=(uv-centre)@(u@vt)+destination
    tri=reference[faces]; rows=[]; cols=[]; values=[]
    for a,b,c in ((0,1,2),(1,2,0),(2,0,1)):
        x,y=tri[:,a]-tri[:,c],tri[:,b]-tri[:,c]
        den=np.abs(cross2(x,y))
        if np.any(den<=1e-12):
            raise ValueError('degenerate chart before welding')
        w=.5*np.sum(x*y,axis=1)/den
        i,j=faces[:,a],faces[:,b]
        rows.extend([i,j,i,j]); cols.extend([i,j,j,i]); values.extend([w,w,-w,-w])
    matrix=coo_matrix((np.concatenate(values),(np.concatenate(rows),np.concatenate(cols))),shape=(len(uv),len(uv))).tocsr()
    free=np.ones(len(uv),bool); free[fixed]=False
    delta=np.zeros_like(uv); delta[fixed]=target-reference[fixed]
    if free.any():
        delta[free]=spsolve(matrix[free][:,free].tocsc(),-(matrix[free][:,fixed]@delta[fixed]))
    result=reference+delta; result[fixed]=target
    try:
        return result,_validate(vertices,faces,result)
    except ValueError:
        igl=_solver()
        if igl is None or not free.any():
            raise
    _validate(vertices,faces,reference)
    data=igl.slim_precompute(vertices,faces.astype(np.int32),reference,igl.MappingEnergyType.SYMMETRIC_DIRICHLET,
                            fixed.astype(np.int32),target,1e12)
    result=igl.slim_solve(data,1).copy()
    # SLIM constraints are soft; restore exact shared coordinates before applying the same gates.
    result[fixed]=target
    return result,_validate(vertices,faces,result)


def _validate(vertices,faces,uv):
    if not np.isfinite(uv).all():
        raise ValueError('nonfinite assembled UV')
    tri=uv[faces]
    if np.any(cross2(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])<=1e-12) or overlap_pairs(uv,faces):
        raise ValueError('inverted or overlapping assembled chart')
    edges=np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]])
    ratio=np.linalg.norm(uv[edges[:,0]]-uv[edges[:,1]],axis=1)/np.linalg.norm(vertices[edges[:,0]]-vertices[edges[:,1]],axis=1)
    if ratio.min()<.25 or ratio.max()>4:
        raise ValueError('excessively distorted assembled chart')
    return dict(edge_stretch_min=float(ratio.min()),edge_stretch_max=float(ratio.max()))


def _coupled_seed(parent,child,puv,outer_ids,outer_target,check):
    for chart in (parent,child):
        n=len(chart['vertices']); nf=len(chart['faces'])
        if not 1<=nf<=8192 or n>3*nf or len(chart['vertex_ids'])!=n or len(chart['face_ids'])!=nf:
            raise ValueError('coupled chart exceeds bounded geometry')
        if len(np.unique(chart['vertex_ids']))!=n:
            raise ValueError('repeated source vertices in coupled chart')
    if len(np.unique(np.r_[parent['face_ids'],child['face_ids']]))!=len(parent['faces'])+len(child['faces']):
        raise ValueError('overlapping coupled source faces')
    ids,first,inverse=np.unique(np.r_[parent['vertex_ids'],child['vertex_ids']],return_index=True,return_inverse=True)
    source=np.concatenate([parent['vertices'],child['vertices']]); vertices=source[first]
    if not np.array_equal(vertices[inverse],source):
        raise ValueError('shared source vertex geometry differs')
    pv,cv=inverse[:len(parent['vertices'])],inverse[len(parent['vertices']):]
    shared,pi,ci=np.intersect1d(pv,cv,return_indices=True)
    if len(shared)<2:
        raise ValueError('coupled charts have no shared seam')
    faces=np.concatenate([pv[parent['faces']],cv[child['faces']]])
    outer_ids,first,inverse=np.unique(outer_ids,return_index=True,return_inverse=True)
    targets=np.asarray(outer_target)[first]
    if not np.array_equal(targets[inverse],outer_target):
        raise ValueError('conflicting outer seam constraints')
    outer=np.searchsorted(ids,outer_ids)
    if len(outer)<2 or np.any(outer>=len(ids)) or not np.array_equal(ids[outer],outer_ids) or not np.isfinite(targets).all():
        raise ValueError('invalid coupled outer seam')
    pins={int(i):value for i,value in zip(shared,puv[pi])}
    for i,value in zip(outer,targets):
        if int(i) in pins and not np.array_equal(pins[int(i)],value):
            raise ValueError('conflicting frozen parent constraints')
        pins[int(i)]=value
    fixed=np.array([i for i,j in enumerate(cv) if int(j) in pins],np.int32)
    target=np.array([pins[int(cv[i])] for i in fixed])
    _validate(parent['vertices'],parent['faces'],puv)
    igl=_solver()
    if igl is None:
        raise ValueError('coupled initialization requires libigl')
    check()
    a=int(np.argmax(np.linalg.norm(vertices-vertices.mean(0),axis=1)))
    b=int(np.argmax(np.linalg.norm(vertices-vertices[a],axis=1)))
    uv,_=igl.lscm(vertices,faces.astype(np.int32),np.array([a,b],np.int32),
                  np.array([[0.,0.],[np.linalg.norm(vertices[b]-vertices[a]),0.]]))
    tri=uv[faces]; signed=cross2(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    if signed.sum()<0:uv[:,1]*=-1
    centre=uv[outer].mean(0); destination=targets.mean(0)
    u,_,vt=np.linalg.svd((uv[outer]-centre).T@(targets-destination))
    if np.linalg.det(u@vt)<0:u[:,-1]*=-1
    uv=(uv-centre)@(u@vt)+destination
    tri=uv[faces]
    # Only initialization may exceed metric bounds; folds/overlaps are never a valid SLIM seed.
    if not np.isfinite(uv).all() or np.any(cross2(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])<=1e-12) or overlap_pairs(uv,faces):
        raise ValueError('invalid coupled initialization')
    check()
    data=igl.slim_precompute(vertices,faces.astype(np.int32),uv,igl.MappingEnergyType.SYMMETRIC_DIRICHLET,
                            outer.astype(np.int32),targets,1e12)
    for _ in range(2):
        check(); uv=igl.slim_solve(data,1).copy(); uv[outer]=targets
    _validate(vertices,faces,uv)
    check()
    data=igl.slim_precompute(child['vertices'],child['faces'].astype(np.int32),uv[cv],
                            igl.MappingEnergyType.SYMMETRIC_DIRICHLET,fixed,target,1e12)
    result=igl.slim_solve(data,1).copy(); result[fixed]=target
    measured=_validate(child['vertices'],child['faces'],result)
    uv[pv]=puv; uv[cv]=result
    if not np.array_equal(uv[pv],puv) or not np.array_equal(uv[outer],targets):
        raise ValueError('coupled recovery changed frozen coordinates')
    _validate(vertices,faces,uv);check()
    return result,measured


def _coupled_weld(store,name,parent,chart,puv,fixed,target):
    if _solver() is None:
        raise ValueError('coupled initialization requires libigl')
    if not 1<=store.db.execute('SELECT faces FROM charts WHERE name=?',(parent,)).fetchone()[0]<=8192 or not 1<=len(chart['faces'])<=8192:
        raise ValueError('coupled chart exceeds bounded geometry')
    store.check(); previous=store.load_arrays('charts/'+parent+'.npz')
    _,pi,ci=np.intersect1d(previous['vertex_ids'],chart['vertex_ids'],return_indices=True)
    required=np.full((len(chart['vertices']),2),np.nan); required[ci]=puv[pi]
    if not np.array_equal(required[fixed],target):
        raise ValueError('shared source IDs disagree with seam targets')
    rows=store.db.execute('SELECT e.chart,e.a,e.b,e.ia,e.ib,p.ia,p.ib,c.uv_path FROM edges e '
        'JOIN edges p ON e.a=p.a AND e.b=p.b AND e.chart!=p.chart JOIN charts c ON c.name=p.chart '
        "WHERE e.chart IN (?,?) AND p.chart NOT IN (?,?) AND c.status='MAPPED' ORDER BY c.uv_path,e.chart,e.a,e.b LIMIT 49153",
        (name,parent,name,parent))
    pins={};last=None
    for count,(own,a,b,ia,ib,ja,jb,path) in enumerate(rows):
        if count>=49152:raise ValueError('coupled outer seam exceeds bounded context')
        if path!=last:
            store.check(); neighbor=store.load_arrays(path)['uv'];last=path
        current=previous if own==parent else chart
        for sid,i,j in ((a,ia,ja),(b,ib,jb)):
            value=neighbor[j]
            if current['vertex_ids'][i]!=sid:
                raise ValueError('outer seam source IDs differ')
            if own==parent and not np.array_equal(puv[i],value) or sid in pins and not np.array_equal(pins[sid],value):
                raise ValueError('conflicting outer seam constraints')
            pins[sid]=value.copy()
    outer=sorted(pins)
    return _coupled_seed(previous,chart,puv,np.array(outer,np.int64),np.array([pins[i] for i in outer]),store.check)


def assemble(store,coverage_target=1.):
    if not 0<coverage_target<=1:
        raise ValueError('coverage target must be in(0,1]')
    _schema(store); store.phase='assemble'
    source_area=store.get('source_area'); source_faces=store.get('source_faces')
    if source_area is None:
        raise ValueError('native source census missing')
    mapped=store.db.execute("SELECT coalesce(sum(area),0) FROM charts WHERE status='MAPPED'").fetchone()[0]
    while True:
        store.check()
        if mapped>=source_area*coverage_target-1e-12*source_area:
            break
        pair=store.db.execute('SELECT child,parent,n FROM frontier ORDER BY n DESC,child,parent LIMIT 1').fetchone()
        if pair:
            name,parent,_=pair
        else:
            row=store.db.execute("SELECT name FROM charts c WHERE status='LOCAL' AND NOT EXISTS "
                "(SELECT 1 FROM charts d WHERE d.component=c.component AND d.status='MAPPED') ORDER BY area DESC,name LIMIT 1").fetchone()
            if row is None:
                break
            name,parent=row[0],None
        chart=store.load_arrays('charts/'+name+'.npz'); uv=chart['uv'].copy(); coupled=False
        try:
            if parent:
                parent_path=store.db.execute('SELECT uv_path FROM charts WHERE name=?',(parent,)).fetchone()[0]
                puv=store.load_arrays(parent_path)['uv']
                shared=np.asarray(store.db.execute('SELECT e.ia,e.ib,p.ia,p.ib FROM edges e JOIN edges p ON e.a=p.a AND e.b=p.b '
                    'WHERE e.chart=? AND p.chart=? ORDER BY e.a,e.b',(name,parent)).fetchall(),int)
                child=np.r_[shared[:,0],shared[:,1]]; other=np.r_[shared[:,2],shared[:,3]]
                fixed,first,inverse=np.unique(child,return_index=True,return_inverse=True)
                target=puv[other[first]]
                if not np.array_equal(target[inverse],puv[other]):
                    raise ValueError('conflicting shared seam constraints')
                try:
                    uv,measured=weld(uv,chart['faces'],fixed,target,chart['vertices'])
                except (ValueError,RuntimeError):
                    uv,measured=_coupled_weld(store,name,parent,chart,puv,fixed,target)
                    coupled=True
            else:
                measured=_validate(chart['vertices'],chart['faces'],uv)
        except (ValueError,RuntimeError) as error:
            with store.db:
                store.db.execute("UPDATE charts SET status='FAILED',reason=? WHERE name=?",(str(error),name))
                store.db.execute('DELETE FROM frontier WHERE child=?',(name,))
            continue
        path='uv/'+name+'.npz'; store.save_arrays(path,uv=uv)
        low,high=uv.min(0),uv.max(0)
        with store.db:
            quality=json.loads(store.db.execute('SELECT quality FROM charts WHERE name=?',(name,)).fetchone()[0])
            quality['assembled']=measured
            store.db.execute("UPDATE charts SET status='MAPPED',uv_path=?,minx=?,maxx=?,miny=?,maxy=?,reason=NULL,quality=? WHERE name=?",
                             (path,float(low[0]),float(high[0]),float(low[1]),float(high[1]),json.dumps(quality),name))
            store.db.execute('DELETE FROM frontier WHERE child=?',(name,))
            _queue(store,name)
            if parent and not coupled:
                store.db.execute("INSERT OR IGNORE INTO cuts SELECT e.a,e.b,e.chart,p.chart FROM edges e JOIN edges p "
                    "ON e.a=p.a AND e.b=p.b JOIN charts c ON c.name=p.chart WHERE e.chart=? AND p.chart!=? "
                    "AND p.chart!=e.chart AND c.status='MAPPED'",(name,parent))
            store.set('elapsed_s',store.elapsed)
        mapped+=store.db.execute('SELECT area FROM charts WHERE name=?',(name,)).fetchone()[0]
        with store.db:
            store.set('mapped_area',mapped)
    return summary(store)


def summary(store):
    source_area=store.get('source_area'); source_faces=store.get('source_faces')
    area,faces,components_count=store.db.execute("SELECT coalesce(sum(area),0),coalesce(sum(faces),0),count(DISTINCT component) "
                                              "FROM charts WHERE status='MAPPED'").fetchone()
    return dict(source_faces=source_faces,mapped_faces=faces,source_area=source_area,mapped_area=area,
                coverage_fraction=min(1.,area/source_area),components=components_count)
