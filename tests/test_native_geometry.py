import json

import numpy as np


def plane_bundle(tmp_path,nx=5,ny=4,copies=1):
    from winding_field.surface_mesh import import_mesh
    from winding_field.util import file_sha
    x,y=np.meshgrid(np.arange(float(nx)),np.arange(float(ny)))
    vertices=np.column_stack([x.ravel()+2,y.ravel()+3,np.full(x.size,4.)])
    faces=[]
    for row in range(ny-1):
        for col in range(nx-1):
            a=row*nx+col
            faces.extend([(a,a+1,a+nx),(a+1,a+nx+1,a+nx)])
    faces=np.concatenate([np.array(faces)+i*len(vertices) for i in range(copies)])
    vertices=np.concatenate([vertices+[0,0,i*2] for i in range(copies)])
    path=tmp_path/'source.npz'
    np.savez(path,vertices_xyz=vertices,faces=np.array(faces))
    root=tmp_path/'mesh'
    import_mesh({'path':str(path),'sha256':file_sha(path)},root,
                {'frame':'test','origin_xyz':[0.,0.,0.],'spacing_xyz':[1.,1.,1.],
                 'shape_zyx':[12,ny+6,nx+6],'voxel_um':1.,'sheet_step':1.})
    return root


def test_global_plane_preserves_every_face_and_shared_seams(tmp_path):
    from winding_field.surface_mesh import MeshReader
    from winding_field.native_store import Journal
    from winding_field.native_geometry import flatten_mesh,assemble
    with MeshReader(plane_bundle(tmp_path)) as reader, Journal(tmp_path/'out',{'source':reader.manifest_sha}) as store:
        flatten_mesh(reader,store,chart_faces=8)
        result=assemble(store,1.)
        assert result['source_faces']==24
        assert result['mapped_faces']==24
        assert result['coverage_fraction']==1.
        assert result['components']==1
        for (quality,) in store.db.execute("SELECT quality FROM charts WHERE status='MAPPED'"):
            measured=json.loads(quality)
            assert measured['local']['method']
            assert measured['assembled']['edge_stretch_min']>.98
            assert measured['assembled']['edge_stretch_max']<1.02
        positions={}; ids=[]
        for name,path in store.db.execute("SELECT name,uv_path FROM charts WHERE status='MAPPED' ORDER BY name"):
            chart=store.load_arrays('charts/'+name+'.npz')
            uv=store.load_arrays(path)['uv']
            ids.extend(chart['face_ids'].tolist())
            for identity,point in zip(chart['vertex_ids'],uv):
                if int(identity) in positions:
                    np.testing.assert_allclose(point,positions[int(identity)],atol=1e-7)
                positions[int(identity)]=point
        assert len(ids)==len(set(ids))==24
        before={p.name:p.read_bytes() for p in (store.root/'uv').glob('*.npz')}
        assert assemble(store,1.)==result
        assert before=={p.name:p.read_bytes() for p in (store.root/'uv').glob('*.npz')}


def test_overlap_predicate_catches_long_slivers_but_not_boundary_contacts():
    from winding_field.flatten import overlap_pairs
    uv=np.array([[0.,0.],[1000.,0.],[0.,.001],
                 [500.,-.001],[501.,.002],[499.,.002],
                 [0.,0.],[0.,.001],[-1.,0.]])
    assert overlap_pairs(uv,np.arange(9).reshape(3,3))==1


def test_large_seam_graph_does_not_rescan_every_edge_per_chart(tmp_path):
    from winding_field.surface_mesh import MeshReader
    from winding_field.native_store import Journal
    from winding_field.native_geometry import flatten_mesh,assemble
    with MeshReader(plane_bundle(tmp_path,nx=33,ny=33)) as reader, Journal(tmp_path/'out',{'source':reader.manifest_sha}) as store:
        flatten_mesh(reader,store,chart_faces=2)
        steps=0
        def budget():
            nonlocal steps
            steps+=1000
            return int(steps>5_000_000)
        store.db.set_progress_handler(budget,1000)
        try:
            result=assemble(store,1.)
        finally:
            store.db.set_progress_handler(None,0)
        assert result['mapped_faces']==2048


def test_disconnected_components_remain_separate(tmp_path):
    from winding_field.surface_mesh import MeshReader
    from winding_field.native_store import Journal
    from winding_field.native_geometry import flatten_mesh,assemble
    with MeshReader(plane_bundle(tmp_path,copies=2)) as reader, Journal(tmp_path/'out',{'source':reader.manifest_sha}) as store:
        flatten_mesh(reader,store,chart_faces=8)
        result=assemble(store,1.)
        assert result['components']==2
        assert result['source_faces']==result['mapped_faces']==48


def test_cancelled_flatten_resumes_without_replacing_committed_charts(tmp_path):
    import pytest
    from winding_field.surface_mesh import MeshReader
    from winding_field.native_store import Journal,Paused
    from winding_field.native_geometry import flatten_mesh,assemble
    source=plane_bundle(tmp_path)
    with MeshReader(source) as reader, Journal(tmp_path/'out',{'source':reader.manifest_sha}) as store:
        store.stop=lambda:store.get('face_cursor',0)>0
        with pytest.raises(Paused):
            flatten_mesh(reader,store,chart_faces=8)
        files={p.name:(p.read_bytes(),p.stat().st_mtime_ns) for p in (store.root/'charts').glob('*.npz')}
        assert files
    with MeshReader(source) as reader, Journal(tmp_path/'out',{'source':reader.manifest_sha}) as store:
        flatten_mesh(reader,store,chart_faces=8)
        assert assemble(store,1.)['mapped_faces']==24
        for name,(data,mtime) in files.items():
            assert (store.root/'charts'/name).read_bytes()==data
            assert (store.root/'charts'/name).stat().st_mtime_ns==mtime


def test_rotated_planar_patch_overlap_validation_has_bounded_cpu_work():
    import signal
    import pytest
    from winding_field.flatten import overlap_pairs
    if not hasattr(signal,'ITIMER_VIRTUAL'):
        pytest.skip('CPU timer unavailable')
    n=257; x,y=np.meshgrid(np.arange(float(n)),np.arange(float(n)))
    uv=np.column_stack([x.ravel(),y.ravel()])@np.array([[.8,-.6],[.6,.8]])
    a=(np.arange(n-1)[:,None]*n+np.arange(n-1)).ravel()
    faces=np.concatenate([np.column_stack([a,a+1,a+n]),np.column_stack([a+1,a+n+1,a+n])])
    def expired(*unused):
        raise TimeoutError('overlap validation exceeded10CPU seconds for a regular patch')
    previous=signal.signal(signal.SIGVTALRM,expired)
    timer=signal.setitimer(signal.ITIMER_VIRTUAL,10)
    try:
        assert overlap_pairs(uv,faces)==0
    finally:
        signal.setitimer(signal.ITIMER_VIRTUAL,*timer)
        signal.signal(signal.SIGVTALRM,previous)


def test_thin_planar_chart_preserves_native_area():
    from winding_field.native_geometry import _pieces
    v=np.array([[0.,0.,0.],[1e6,0.,0.],[0.,1e-4,0.],[1e6,1e-4,0.]])
    f=np.array([[0,1,2],[1,3,2]])
    pieces=list(_pieces(v,f,np.array([1,2]),np.arange(4)))
    assert len(pieces)==1
    _,faces,uv,ids,_,_=pieces[0]
    assert uv.shape==(4,2)
    tri=uv[faces]
    a,b=tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]
    area=np.sum(a[:,0]*b[:,1]-a[:,1]*b[:,0])/2
    np.testing.assert_allclose(area,100.,rtol=1e-7)


def test_recursive_face_halves_are_reconnected_before_parameterization():
    from winding_field.native_geometry import _pieces,_validate
    vertices=np.array([[1.,0.,0.],[-1.,0.,0.],[0.,1.,0.],[0.,-1.,0.],[0.,0.,1.],[0.,0.,-1.]])
    # Closed octahedron: edge-connected overall, but each face-order half has four point-touching faces.
    faces=np.array([[4,0,2],[4,1,3],[5,1,2],[5,0,3],
                    [4,2,1],[4,3,0],[5,2,0],[5,3,1]])
    face_ids=np.array([70,11,90,32,42,21,88,101]);vertex_ids=np.array([101,701,20,41,51,900])
    pieces=list(_pieces(vertices,faces,face_ids,vertex_ids,minimum=4))
    assert sum(len(f) for _,f,uv,_,_,_ in pieces if len(uv))==8
    assert len(pieces)==8
    seen=[]
    for v,f,uv,ids,source_ids,_ in pieces:
        _validate(v,f,uv);seen.extend(ids.tolist())
        for triangle,fid in zip(f,ids):
            original=faces[np.flatnonzero(face_ids==fid)[0]]
            np.testing.assert_array_equal(source_ids[triangle],vertex_ids[original])
            np.testing.assert_array_equal(v[triangle],vertices[original])
    assert sorted(seen)==sorted(face_ids.tolist()) and len(set(seen))==len(seen)


def test_point_contacts_do_not_make_one_parameterization_chart():
    from winding_field.native_geometry import _pieces
    v=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[-1.,0.,0.],[0.,-1.,0.]])
    f=np.array([[0,1,2],[0,3,4]])
    pieces=list(_pieces(v,f,np.array([9,3]),np.array([50,40,30,20,10])))
    assert len(pieces)==2
    assert sorted(int(ids[0]) for _,faces,uv,ids,_,_ in pieces if len(faces)==1 and len(uv))==[3,9]


def test_connected_planar_piece_keeps_its_geometry_ids_and_uv():
    from winding_field.native_geometry import _pieces
    v=np.array([[0.,0.,0.],[4.,0.,0.],[0.,2.,0.],[4.,2.,0.]])
    f=np.array([[0,1,2],[1,3,2]]);ids=np.array([8,2]);source_ids=np.array([90,70,50,30])
    pieces=list(_pieces(v,f,ids,source_ids))
    assert len(pieces)==1
    actual_v,actual_f,uv,actual_ids,actual_source,_=pieces[0]
    for actual,expected in ((actual_v,v),(actual_f,f),(actual_ids,ids),(actual_source,source_ids),(uv,v[:,:2])):
        np.testing.assert_array_equal(actual,expected)


def test_real_disconnected_failed_chart_retains_every_source_face():
    import os
    from pathlib import Path
    import hashlib
    import pytest
    pytest.importorskip('igl')
    from winding_field.native_geometry import _pieces,_validate
    root=os.environ.get('WINDING_NATIVE_PIECES_PROOF')
    if not root:pytest.skip('set WINDING_NATIVE_PIECES_PROOF to the frozen W50 chart witnesses')
    for name,sha in (('chart-000000005377.npz','b1a5d0489b1f766d39819cf1f860acc14781e5d9f06f85c1c0bf76fe4763fd22'),
                     ('chart-000000005441.npz','47fc4ae1f923ac91a79bd7911a781d2b3aa580abff1008b83a9574a1b7c1b8db')):
        path=Path(root)/name
        assert hashlib.sha256(path.read_bytes()).hexdigest()==sha
        with np.load(path,allow_pickle=False) as data:
            v,f,ids,source_ids=(data[k] for k in ('vertices','faces','face_ids','vertex_ids'))
        pieces=list(_pieces(v,f,ids,source_ids))
        assert sum(len(faces) for _,faces,uv,_,_,_ in pieces if len(uv))==64
        assert 1<len(pieces)<=len(f)
        all_ids=[]
        for pv,pf,uv,face_ids,vertex_ids,_ in pieces:
            _validate(pv,pf,uv);all_ids.extend(face_ids.tolist())
            for triangle,fid in zip(pf,face_ids):
                original=f[np.flatnonzero(ids==fid)[0]]
                np.testing.assert_array_equal(vertex_ids[triangle],source_ids[original])
                np.testing.assert_array_equal(pv[triangle],v[original])
        assert sorted(all_ids)==sorted(ids.tolist()) and len(all_ids)==len(set(all_ids))


def test_overlap_is_invariant_to_large_uv_translation():
    from winding_field.flatten import overlap_pairs
    uv=np.array([[0.,0.],[1.,0.],[0.,1.]])+1e8
    assert overlap_pairs(uv,np.array([[0,1,2],[0,1,2]]))==1


def test_optional_libigl_lscm_keeps_curved_strip_geometry():
    import pytest
    pytest.importorskip('igl')
    from test_meshes import curved_strip
    from winding_field.native_geometry import _lscm
    v,f=curved_strip()
    uv,quality=_lscm(v,f)
    assert uv.shape==(65,2)
    assert quality['edge_stretch_min']>.99
    assert quality['edge_stretch_max']<1.01


def failed_seam_chart():
    # Centred, rounded two-ring reduction of frozen tile7423/chart054273, with six fixed seam targets.
    vertices=np.array([
        [-8.032957,18.557912,-6.379944],[-17.335420,23.930374,-6.379944],
        [9.247043,7.933858,-6.379944],[10.667775,23.930374,-6.379944],
        [12.204961,41.210374,-6.379944],[13.756137,58.490374,-6.379944],
        [-25.312957,-45.189626,5.337475],[-8.032957,-45.189626,2.626433],
        [9.247043,-45.189626,.255339],[-25.312957,-27.909626,2.673167],
        [-8.032957,-27.909626,.259412],[9.247043,-27.909626,-1.851063],
        [17.548358,-27.909626,10.900056],[-25.312957,-10.629626,-.024574],
        [-8.032957,-10.629626,-2.176897],[9.247043,-10.629626,-4.010916],
        [18.885732,-10.629626,10.900056],[-25.312957,6.650374,-2.796924],
        [-8.032957,6.650374,-4.652240],[9.247043,6.650374,-6.215385],
        [20.232028,6.650374,10.900056],[-25.312957,23.930374,-5.644440],
        [21.586251,23.930374,10.900056],[22.948527,41.210374,10.900056]])
    faces=np.array([[7,6,10],[10,6,9],[8,7,11],[11,7,10],[11,12,8],[10,9,14],
        [14,9,13],[11,10,15],[15,10,14],[16,12,11],[15,16,11],[14,13,18],
        [18,13,17],[15,14,19],[19,14,18],[20,16,15],[19,20,15],[0,21,1],
        [0,18,21],[18,17,21],[2,18,0],[19,18,2],[19,2,3],[19,3,20],
        [20,3,22],[3,4,22],[22,4,23],[4,5,23]])
    uv=np.array([[2.711365,-9.120845],[1.463529,-13.707446],[4.724456,-.453351],
        [10.552893,-4.407810],[16.938267,-8.675413],[23.408412,-12.936723],
        [-21.962928,5.854658],[-16.368288,10.482812],[-11.216718,14.689478],
        [-17.013589,.321671],[-11.529344,5.221523],[-6.369545,10.145726],
        [-2.743066,15.300114],[-11.864308,-5.167972],[-6.383313,-.053359],
        [-1.115872,5.136833],[3.410766,11.217897],[-6.540962,-10.667832],
        [-1.042127,-5.395898],[4.310867,-.065017],[9.599932,7.024569],
        [-1.075503,-16.187863],[15.873150,2.822038],[22.231928,-1.377791]])
    fixed=np.arange(6)
    target=np.array([[.289529,8.224251],[2.570133,18.625389],[-3.493388,-11.521098],
                     [-16.805430,-2.806161],[-31.185784,6.569649],[-45.568338,15.879568]])
    return vertices,faces,uv,fixed,target


def test_failed_cotangent_seam_recovers_with_exact_targets_and_unchanged_gates():
    import pytest
    pytest.importorskip('igl')
    from winding_field.native_geometry import weld,_validate
    vertices,faces,uv,fixed,target=failed_seam_chart()
    _validate(vertices,faces,uv)
    result,quality=weld(uv,faces,fixed,target,vertices)
    _validate(vertices,faces,result)
    np.testing.assert_array_equal(result[fixed],target)
    assert quality['edge_stretch_min']>=.25 and quality['edge_stretch_max']<=4.


def test_weld_fallback_does_not_accept_collapsed_or_infeasible_seams():
    import pytest
    pytest.importorskip('igl')
    from winding_field.native_geometry import weld,_validate
    vertices,faces,uv,fixed,target=failed_seam_chart()
    stretched=target.copy(); stretched[1]=stretched[0]+[1e6,0.]
    for broken in (np.zeros_like(target),stretched):
        with pytest.raises((ValueError,RuntimeError)):
            weld(uv,faces,fixed,broken,vertices)


def test_failed_weld_stays_rejected_without_optional_solver(monkeypatch):
    import pytest
    from winding_field.native_geometry import weld
    vertices,faces,uv,fixed,target=failed_seam_chart()
    monkeypatch.setattr('winding_field.native_geometry._solver',lambda:None)
    with pytest.raises(ValueError,match='inverted or overlapping'):
        weld(uv,faces,fixed,target,vertices)


def test_valid_cotangent_weld_does_not_invoke_slim(monkeypatch):
    import pytest
    from winding_field.native_geometry import weld
    uv=np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.]])
    faces=np.array([[0,1,2],[1,3,2]]); vertices=np.column_stack([uv,np.zeros(4)])
    fixed=np.array([0,1]); target=uv[fixed]+[3.,4.]
    monkeypatch.setattr('winding_field.native_geometry._solver',lambda:pytest.fail('unnecessary SLIM'))
    result,_=weld(uv,faces,fixed,target,vertices)
    np.testing.assert_allclose(result,uv+[3.,4.],atol=1e-12)


def test_assembly_rejects_contradictory_shared_vertex_targets(tmp_path):
    from winding_field.surface_mesh import MeshReader
    from winding_field.native_store import Journal
    from winding_field.native_geometry import flatten_mesh,assemble
    with MeshReader(plane_bundle(tmp_path)) as reader, Journal(tmp_path/'out',{'source':reader.manifest_sha}) as store:
        flatten_mesh(reader,store,chart_faces=8)
        assemble(store,.1)
        child,parent=store.db.execute('SELECT child,parent FROM frontier ORDER BY n DESC,child,parent LIMIT 1').fetchone()
        shared=store.db.execute('SELECT e.a,e.b,e.ia,e.ib,p.ia,p.ib FROM edges e JOIN edges p ON e.a=p.a AND e.b=p.b '
            'WHERE e.chart=? AND p.chart=? ORDER BY e.a,e.b',(child,parent)).fetchall()
        first,second=shared[:2]
        assert first[4]!=second[4]
        with store.db:
            store.db.execute('UPDATE edges SET ia=? WHERE chart=? AND a=? AND b=?',
                             (first[2],child,second[0],second[1]))
        assemble(store,1.)
        status,reason=store.db.execute('SELECT status,reason FROM charts WHERE name=?',(child,)).fetchone()
        assert status=='FAILED'
        assert reason=='conflicting shared seam constraints'


def test_coupled_recovery_is_lazy_and_publishes_only_the_child(tmp_path,monkeypatch):
    import pytest
    import winding_field.native_geometry as geometry
    from winding_field.surface_mesh import MeshReader
    from winding_field.native_store import Journal,Paused
    monkeypatch.setattr(geometry,'_coupled_weld',lambda *args:pytest.fail('coupled recovery on fast path'),raising=False)
    with MeshReader(plane_bundle(tmp_path)) as reader, Journal(tmp_path/'out',{'source':reader.manifest_sha}) as store:
        geometry.flatten_mesh(reader,store,8);geometry.assemble(store,1.)
        name,path=store.db.execute("SELECT name,uv_path FROM charts WHERE status='MAPPED' ORDER BY name DESC LIMIT 1").fetchone()
        expected=store.load_arrays(path)['uv']
        before={n:store.path(n).read_bytes() for (n,) in store.db.execute("SELECT name FROM files WHERE name LIKE 'uv/%'")}
        cuts=store.db.execute('SELECT * FROM cuts ORDER BY a,b,chart,parent').fetchall()
        with store.db:
            store.db.execute("UPDATE charts SET status='LOCAL',uv_path=NULL WHERE name=?",(name,))
            geometry._queue(store,name,child=True)
        def rejected(*args):raise ValueError('existing weld rejected')
        interrupted=[True]
        def recovery(store,child,parent,chart,puv,fixed,target):
            if interrupted[0]:raise Paused('STOP')
            np.testing.assert_array_equal(expected[fixed],target)
            return expected,geometry._validate(chart['vertices'],chart['faces'],expected)
        monkeypatch.setattr(geometry,'weld',rejected)
        monkeypatch.setattr(geometry,'_coupled_weld',recovery,raising=False)
        with pytest.raises(Paused,match='STOP'):
            geometry.assemble(store,1.)
        assert store.db.execute('SELECT status FROM charts WHERE name=?',(name,)).fetchone()[0]=='LOCAL'
        interrupted[0]=False
        assert geometry.assemble(store,1.)['mapped_faces']==24
        assert before=={n:store.path(n).read_bytes() for n in before}
        assert cuts==store.db.execute('SELECT * FROM cuts ORDER BY a,b,chart,parent').fetchall()


def frozen_coupled_store(tmp_path):
    import hashlib
    import os
    from pathlib import Path
    import sqlite3
    import pytest
    from winding_field.native_store import Journal
    from winding_field.native_geometry import _schema,_queue
    source=os.environ.get('WINDING_NATIVE_COUPLED_PROOF')
    if not source:pytest.skip('set WINDING_NATIVE_COUPLED_PROOF to the frozen recovery journal')
    source=Path(source)
    expected='7e13b339585b33eadacea45822cedb3e7c5b872a4edcb1b80f717f873f23c7b9'
    assert hashlib.sha256((source/'work.sqlite').read_bytes()).hexdigest()==expected
    store=Journal(tmp_path/'out',{'frozen_chart_journal':expected})
    _schema(store)
    names=['chart-000000053249','chart-000000054273','chart-000000055297']
    with sqlite3.connect((source/'work.sqlite').as_uri()+'?mode=ro',uri=True) as original,store.db:
        rows=[]
        for name in names:
            row=original.execute('SELECT * FROM charts WHERE name=?',(name,)).fetchone();rows.append(row)
            for path in ('charts/'+name+'.npz',row[5]):
                if path is None:continue
                raw=(source/path).read_bytes()
                assert original.execute('SELECT sha,bytes FROM files WHERE name=?',(path,)).fetchone()==(hashlib.sha256(raw).hexdigest(),len(raw))
                destination=store.path(path);destination.parent.mkdir(exist_ok=True);destination.write_bytes(raw);store.register(path)
            store.db.execute('INSERT INTO charts VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',row)
            store.db.executemany('INSERT INTO edges VALUES (?,?,?,?,?,?)',original.execute('SELECT * FROM edges WHERE chart=?',(name,)).fetchall())
        store.set('source_faces',sum(r[3] for r in rows));store.set('source_area',sum(r[2] for r in rows))
        store.db.execute("UPDATE charts SET status='LOCAL',reason=NULL WHERE name=?",(names[-1],))
        _queue(store,names[-1],child=True)
    return store


def test_frozen_child_recovers_with_immutable_parent_and_stop_resume(tmp_path,monkeypatch):
    import pytest
    igl=pytest.importorskip('igl')
    import winding_field.native_geometry as geometry
    from winding_field.native_store import Journal,Paused
    from winding_field.util import file_sha
    original=igl.slim_solve;calls=[0]
    def counted(*args):
        calls[0]+=1
        return original(*args)
    monkeypatch.setattr(igl,'slim_solve',counted)
    outputs=[]
    for interrupted in (False,True):
        here=tmp_path/str(interrupted);here.mkdir()
        with frozen_coupled_store(here) as store:
            identity=store.identity
            parents={name:file_sha(store.path(name)) for (name,) in store.db.execute("SELECT name FROM files WHERE name LIKE 'uv/%'")}
            if interrupted:
                start=calls[0];store.stop=lambda:calls[0]>=start+2
                with pytest.raises(Paused,match='STOP'):geometry.assemble(store,1.)
                assert not store.db.execute("SELECT 1 FROM files WHERE name='uv/chart-000000055297.npz'").fetchone()
            else:
                assert geometry.assemble(store,1.)['mapped_faces']==3072
        with Journal(here/'out',identity) as store:
            assert geometry.assemble(store,1.)['mapped_faces']==3072
            assert parents=={name:file_sha(store.path(name)) for name in parents}
            assert store.db.execute('SELECT count(*) FROM cuts').fetchone()[0]==0
            outputs.append(store.path('uv/chart-000000055297.npz').read_bytes())
    assert outputs[0]==outputs[1]


def coupled_triangles():
    parent=dict(vertices=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]),faces=np.array([[0,1,2]]),
                vertex_ids=np.array([0,1,2]),face_ids=np.array([1]))
    child=dict(vertices=np.array([[1.,0.,0.],[0.,1.,0.],[1.,1.,0.]]),faces=np.array([[0,2,1]]),
               vertex_ids=np.array([1,2,3]),face_ids=np.array([2]))
    return parent,child,parent['vertices'][:,:2].copy(),np.arange(3),parent['vertices'][:,:2].copy()


def test_coupled_seed_rejects_oversized_or_conflicting_sources_before_solver(monkeypatch):
    import pytest
    import winding_field.native_geometry as geometry
    monkeypatch.setattr(geometry,'_solver',lambda:pytest.fail('invalid patch reached solver'))
    for invalid in ('oversized','unused-vertices','xyz','face-ids','outer-targets'):
        parent,child,puv,outer,target=coupled_triangles()
        if invalid=='oversized':parent['faces']=np.tile(parent['faces'],(8193,1))
        elif invalid=='unused-vertices':parent['vertices']=np.vstack([parent['vertices'],[9.,9.,9.]])
        elif invalid=='xyz':child['vertices'][0,0]+=1e-12
        elif invalid=='face-ids':child['face_ids']=parent['face_ids'].copy()
        else:outer=np.r_[outer,outer[0]];target=np.vstack([target,target[0]+[1.,0.]])
        with pytest.raises(ValueError):geometry._coupled_seed(parent,child,puv,outer,target,lambda:None)


def test_coupled_seed_bounds_iterations_and_rejects_unqualified_phases(monkeypatch):
    from types import SimpleNamespace
    import pytest
    import winding_field.native_geometry as geometry
    from winding_field.native_store import Paused
    for mode in ('valid','invalid-merged','invalid-child','invalid-initial','stop'):
        parent,child,puv,outer,target=coupled_triangles();before=puv.tobytes();calls=[]
        def lscm(v,f,pins,bc):
            # A metric-invalid but unfolded initialization must still reach the final strict gates.
            uv=v[:,:2]*10
            return (np.zeros_like(uv) if mode=='invalid-initial' else uv),None
        def precompute(v,f,uv,energy,fixed,bc,weight):return v[:,:2].copy()
        def solve(data,iterations):
            calls.append(iterations)
            assert len(calls)<=3,'unbounded coupled optimization'
            if mode=='invalid-merged' or mode=='invalid-child' and len(calls)==3:return np.zeros_like(data)
            return data
        solver=SimpleNamespace(lscm=lscm,slim_precompute=precompute,slim_solve=solve,
                               MappingEnergyType=SimpleNamespace(SYMMETRIC_DIRICHLET=0))
        monkeypatch.setattr(geometry,'_solver',lambda:solver)
        def check():
            if mode=='stop' and calls:raise Paused('STOP')
        if mode=='valid':
            uv,quality=geometry._coupled_seed(parent,child,puv,outer,target,check)
            np.testing.assert_array_equal(uv,child['vertices'][:,:2])
            assert quality['edge_stretch_min']==quality['edge_stretch_max']==1.
            assert calls==[1,1,1]
        else:
            with pytest.raises(Paused if mode=='stop' else ValueError):
                geometry._coupled_seed(parent,child,puv,outer,target,check)
            assert len(calls)=={'invalid-merged':2,'invalid-child':3,'invalid-initial':0,'stop':1}[mode]
        assert puv.tobytes()==before
