import hashlib
import json
import sqlite3

import numpy as np
import pytest

from winding_field import surface_mesh as reference


class Field:
    def __init__(self):
        z,y,x=np.indices((9,10,11))
        self.values=x-3.2+.1*y+.02*z
        self.shape=self.values.shape; self.origin=np.array([10.,20.,30.]); self.spacing=np.array([2.,3.,4.])
        self.identity={'geometry':dict(shape_zyx=list(self.shape),origin_xyz=self.origin.tolist(),
            spacing_xyz=self.spacing.tolist(),voxel_um=1.,sheet_step=1.,frame='analytic')}

    def read(self,lo,hi):
        a=self.values[tuple(slice(int(x),int(y)) for x,y in zip(lo,hi))]
        return a.copy(),np.ones(a.shape,bool)

    def validate(self,hashes=False):
        pass


def compare(reference_root,actual_root):
    a=json.loads((reference_root/'MESH.json').read_bytes())
    b=json.loads((actual_root/'MESH.json').read_bytes())
    assert len(a['levels'])==len(b['levels'])
    for x,y in zip(a['levels'],b['levels']):
        for key in x:
            if key=='area_world2':
                assert y[key]==pytest.approx(x[key],rel=1e-12)
            else:
                assert y[key]==x[key]
        assert (reference_root/f"level-{x['id']:04d}.obj").read_bytes()==(actual_root/f"level-{y['id']:04d}.obj").read_bytes()
    with reference.MeshReader(actual_root) as reader:
        assert sum(len(x) for x,_ in reader.triangles())==sum(x['faces'] for x in a['levels'])


def test_batched_mesh_matches_original_geometry_and_topology(tmp_path):
    from winding_field.native_mesh import extract_mesh
    source=Field()
    reference.extract_mesh(source,tmp_path/'reference',[0.,2.],tile=3)
    assert extract_mesh(source,tmp_path/'actual',[0.,2.],tile=3)['status']=='MESH_READY'
    compare(tmp_path/'reference',tmp_path/'actual')


class BoundedField(Field):
    def __init__(self):
        super().__init__(); self.reads=[]; self.excluded=[]
        self.identity['bounds_proof']='analytic-full-node-ranges'

    def tile_bounds(self,lo,hi):
        values=self.values[tuple(slice(int(a),int(b)) for a,b in zip(lo,hi))]
        result=float(values.min()),float(values.max())
        self.excluded.append(not any(result[0]<=level<=result[1] for level in (0.,2.)))
        return result

    def read(self,lo,hi):
        self.reads.append(tuple(map(int,lo)))
        return super().read(lo,hi)


def test_proven_empty_tiles_skip_reconstruction_without_changing_mesh(tmp_path):
    from winding_field.native_mesh import extract_mesh
    source=BoundedField()
    reference.extract_mesh(Field(),tmp_path/'reference',[0.,2.],tile=3)
    assert extract_mesh(source,tmp_path/'actual',[0.,2.],tile=3)['status']=='MESH_READY'
    compare(tmp_path/'reference',tmp_path/'actual')
    assert sum(source.excluded)>0
    assert len(source.reads)==len(source.excluded)-sum(source.excluded)


@pytest.mark.parametrize('endpoint', ['min','max'])
def test_range_endpoint_equality_still_reads_the_source(tmp_path,endpoint):
    from winding_field.native_mesh import extract_mesh
    source=BoundedField(); level=float(getattr(source.values,endpoint)())
    extract_mesh(source,tmp_path/'mesh',[level],tile=32)
    assert len(source.reads)==1


@pytest.mark.parametrize('bounds', [(2.,1.),(float('nan'),1.),(0.,float('inf')),(False,1.),(0.,)])
def test_invalid_bounds_cannot_publish_an_empty_mesh(tmp_path,bounds):
    from winding_field.native_mesh import extract_mesh
    source=Field(); source.tile_bounds=lambda lo,hi:bounds
    with pytest.raises(ValueError,match='bounds'):
        extract_mesh(source,tmp_path/'mesh',[0.],tile=3)
    assert not (tmp_path/'mesh/MESH.json').exists()


def test_bounds_none_and_cancel_resume_preserve_committed_geometry(tmp_path):
    from winding_field.native_mesh import extract_mesh
    source=BoundedField(); calls=0
    def stop():
        nonlocal calls
        calls+=1
        return calls>8
    assert extract_mesh(source,tmp_path/'mesh',[0.,2.],tile=3,stop=stop)['status']=='CHECKPOINTED_STOP'
    before=list(source.reads)
    assert extract_mesh(source,tmp_path/'mesh',[0.,2.],tile=3,resume=True)['status']=='MESH_READY'
    assert source.reads[:len(before)]==before
    assert len(source.reads)==len(set(source.reads))
    reference.extract_mesh(Field(),tmp_path/'reference',[0.,2.],tile=3)
    compare(tmp_path/'reference',tmp_path/'mesh')
    fallback=Field(); fallback.tile_bounds=lambda lo,hi:None
    extract_mesh(fallback,tmp_path/'fallback',[0.,2.],tile=3)
    compare(tmp_path/'reference',tmp_path/'fallback')


def test_batched_resume_rebuilds_derived_state_without_changing_geometry(tmp_path):
    from winding_field.native_mesh import extract_mesh
    source=Field(); calls=0
    def stop():
        nonlocal calls
        calls+=1
        return calls>3
    result=extract_mesh(source,tmp_path/'actual',[0.,2.],tile=3,stop=stop)
    assert result['status']=='CHECKPOINTED_STOP'
    with sqlite3.connect(tmp_path/'actual/mesh.sqlite') as db:
        before=db.execute('SELECT id,a,b,c,area FROM faces ORDER BY id').fetchall()
        db.execute('UPDATE edges SET count=99'); db.commit()
    assert extract_mesh(source,tmp_path/'actual',[0.,2.],tile=3,resume=True)['status']=='MESH_READY'
    with sqlite3.connect(tmp_path/'actual/mesh.sqlite') as db:
        assert before==db.execute('SELECT id,a,b,c,area FROM faces ORDER BY id LIMIT ?',(len(before),)).fetchall()
    reference.extract_mesh(source,tmp_path/'reference',[0.,2.],tile=3)
    compare(tmp_path/'reference',tmp_path/'actual')


def test_batched_resume_rejects_corrupt_committed_vertices(tmp_path):
    from winding_field.native_mesh import extract_mesh
    source=Field(); calls=0
    def stop():
        nonlocal calls
        calls+=1
        return calls>4
    extract_mesh(source,tmp_path/'actual',[0.],tile=3,stop=stop)
    with sqlite3.connect(tmp_path/'actual/mesh.sqlite') as db:
        db.execute('UPDATE vertices SET x=x+1 WHERE id=1'); db.commit()
    with pytest.raises(ValueError,match='checksum'):
        extract_mesh(source,tmp_path/'actual',[0.],tile=3,resume=True)


def test_batched_row_hash_matches_original_even_with_json_delimiters():
    from winding_field.native_mesh import rows_sha
    with sqlite3.connect(':memory:') as db:
        db.execute('CREATE TABLE data(i INTEGER,x REAL,s TEXT)')
        db.executemany('INSERT INTO data VALUES (?,?,?)',[(i,i/3,'a],[b' if i==17 else 'key:'+str(i)) for i in range(9000)])
        query='SELECT * FROM data ORDER BY i'
        assert rows_sha(db,query)==reference._rows_sha(db,query)


def test_connected_tile_assembly_does_not_query_sql_per_face(tmp_path):
    from winding_field.native_mesh import MeshDB
    x,y=np.meshgrid(np.arange(33.),np.arange(33.)); v=np.column_stack([x.ravel(),y.ravel(),np.zeros(x.size)])
    a=(np.arange(32)[:,None]*33+np.arange(32)).ravel()
    f=np.concatenate([np.column_stack([a,a+1,a+33]),np.column_stack([a+1,a+34,a+33])])
    identity={'levels':[0.],'total_tiles':1}
    with MeshDB(tmp_path/'mesh',Field().identity['geometry'],identity) as store:
        selects=0
        def trace(sql):
            nonlocal selects
            selects+=sql.lstrip().upper().startswith('SELECT')
        store.db.set_trace_callback(trace)
        with store.db:
            store.add(0,[str(i) for i in range(len(v))],v,f)
            store.checkpoint_tile(0)
        assert selects<100
        assert store.finish()[0]['faces']==2048


@pytest.mark.parametrize('faces,components,nonmanifold,duplicates',[
    ([[0,1,2],[0,3,4]],2,0,0),
    ([[0,1,2],[1,0,3],[0,1,4]],1,1,0),
    ([[0,1,2],[0,1,2]],1,0,1),
])
def test_edge_connected_topology_keeps_vertex_contacts_and_duplicates(tmp_path,faces,components,nonmanifold,duplicates):
    from winding_field.native_mesh import MeshDB
    v=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],[1.,1.,1.]])
    with MeshDB(tmp_path/'mesh',Field().identity['geometry'],{'levels':[0.],'total_tiles':1}) as store:
        with store.db:
            store.add(0,[str(i) for i in range(5)],v,np.array(faces))
            store.checkpoint_tile(0)
        result=store.finish()[0]
        assert result['components']==components
        assert result['nonmanifold_edges']==nonmanifold
        assert result['duplicate_faces']==duplicates
