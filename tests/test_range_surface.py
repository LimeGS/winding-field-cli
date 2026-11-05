"""Independent geometry and adversarial artifact contracts for continuous ranges."""
import importlib.util
import json
from pathlib import Path
import sqlite3
import subprocess
import sys

import numpy as np
import pytest

from winding_field.surface_mesh import MeshReader
from winding_field.util import file_sha


def api():
    assert importlib.util.find_spec('winding_field.range_surface'), 'continuous range engine missing'
    from winding_field import range_surface
    return range_surface


def recipe(tmp_path, *, empty=False, mask=None, step=2., samples=8):
    # L1 radial field is exactly bilinear in each grid cell. Its level curve
    # has radius (W+63)/(|cos(theta)|+|sin(theta)|), independently of solver.
    z,y,x=np.indices((3,9,9))
    values=(np.abs(x-4)+np.abs(y-4)-63.)*step
    if empty:values[:]=100.
    path=tmp_path/'field.npy';np.save(path,values)
    field=dict(kind='npy',path=str(path),sha256=file_sha(path))
    if mask is not None:
        path=tmp_path/'mask.npy';np.save(path,mask)
        field['mask']=dict(path=str(path),sha256=file_sha(path))
    return dict(field=field,geometry=dict(frame='test/world',voxel_um=2.,sheet_step=step,
        origin_xyz=[11.,-7.,19.],spacing_xyz=[2.,3.,5.],shape_zyx=[3,9,9]),
        umbilicus=[[19.,5.,19.],[19.,5.,29.]],h=1,samples_per_w=samples)


def canonical(path):
    with sqlite3.connect(path/'mesh/mesh.sqlite') as db:
        return {table:list(db.execute(query)) for table,query in {
            'nodes':'SELECT * FROM range_nodes ORDER BY tick_w,tick_z',
            'cells':'SELECT * FROM range_cells ORDER BY tick_w,tick_z',
            'vertices':'SELECT key,x,y,z FROM vertices ORDER BY key',
            'faces':'''SELECT f.key,a.key,b.key,c.key FROM faces f JOIN vertices a ON a.id=f.a
                JOIN vertices b ON b.id=f.b JOIN vertices c ON c.id=f.c ORDER BY f.key''',
        }.items()}


def test_analytic_spiral_calibration_anisotropic_world_coordinates(tmp_path):
    r=recipe(tmp_path)
    result=api().extract_range(r,[-61,-59],tmp_path/'full',block_ticks=3)
    assert result['status']=='RANGE_READY' and result['faces']>0 and result['area_world2']>0
    with MeshReader(tmp_path/'full/mesh') as reader:
        assert reader.manifest['identity']['kind']=='continuous_range'
        assert reader.manifest['levels'][0]['nonmanifold_edges']==0
        for tick,z,x,y,world_z in reader.db.execute(
                'SELECT tick_w,tick_z,x,y,z FROM range_nodes WHERE state="unique"'):
            angle=2*np.pi*(tick/8+60)
            # Ray angle is physical XY, hence anisotropic grid metric below.
            radius=(tick/8+63)/(abs(np.cos(angle))/2+abs(np.sin(angle))/3)
            np.testing.assert_allclose([x,y,world_z],
                [19+radius*np.cos(angle),5+radius*np.sin(angle),19+z*5],atol=1e-11)


@pytest.mark.parametrize('cut',[-60,-60.375])
@pytest.mark.parametrize('samples',[8,1000])
def test_exact_partition_overlap_reversed_inputs_and_duplicate_idempotence(tmp_path,cut,samples):
    r=recipe(tmp_path,samples=samples);m=api()
    m.extract_range(r,[-61,-59],tmp_path/'direct',block_ticks=1)
    m.extract_range(r,[-61,cut],tmp_path/'a',block_ticks=4)
    m.extract_range(r,[cut,-59],tmp_path/'b',block_ticks=2)
    m.extract_range(r,[-60.5,-59.5],tmp_path/'overlap')
    result=m.join_ranges([tmp_path/'b',tmp_path/'a',tmp_path/'overlap',tmp_path/'a'],tmp_path/'joined')
    assert result['faces']>0
    assert canonical(tmp_path/'direct')==canonical(tmp_path/'joined')


def test_empty_composes_and_no_center_extrapolation(tmp_path):
    r=recipe(tmp_path,empty=True);m=api()
    for name,bounds in [('a',[-61,-60]),('b',[-60,-59])]:
        assert m.extract_range(r,bounds,tmp_path/name)['status']=='EMPTY'
    result=m.join_ranges([tmp_path/'b',tmp_path/'a'],tmp_path/'empty')
    assert result['status']=='EMPTY' and result['faces']==0 and result['area_world2']==0
    assert len(canonical(tmp_path/'empty')['nodes'])==51
    r['umbilicus']=[[19.,5.,24.],[19.,5.,29.]]
    m.extract_range(r,[-61,-60],tmp_path/'cropped')
    with sqlite3.connect(tmp_path/'cropped/mesh/mesh.sqlite') as db:
        assert db.execute('SELECT DISTINCT state FROM range_nodes WHERE tick_z=0').fetchall()==[('center_unsupported',)]


@pytest.mark.parametrize('bounds',[[-61.01,-60],[-60,-61],[-60,-60],['NaN',-59]])
def test_reject_invalid_decimal_lattice_bounds_before_output(tmp_path,bounds):
    r=recipe(tmp_path)
    with pytest.raises(ValueError):api().extract_range(r,bounds,tmp_path/'bad')
    assert not (tmp_path/'bad').exists()


def test_join_rejects_gaps_and_recipe_incompatibility(tmp_path):
    r=recipe(tmp_path);m=api()
    m.extract_range(r,[-61,-60.5],tmp_path/'a')
    m.extract_range(r,[-60,-59],tmp_path/'gap')
    with pytest.raises(ValueError,match='gap'):m.join_ranges([tmp_path/'a',tmp_path/'gap'],tmp_path/'badgap')
    r['h']=-1;m.extract_range(r,[-60.5,-59],tmp_path/'other')
    with pytest.raises(ValueError,match='recipe'):m.join_ranges([tmp_path/'a',tmp_path/'other'],tmp_path/'badrecipe')


@pytest.mark.parametrize('inner',[False,True])
def test_scalar_export_guard_before_ct_or_optional_dependencies(tmp_path,inner):
    r=recipe(tmp_path);api().extract_range(r,[-61,-60],tmp_path/'range')
    from winding_field.original_pipeline import export_mesh
    path=tmp_path/'range'/('mesh' if inner else '')
    with pytest.raises(ValueError,match='continuous_range'):
        export_mesh(path,tmp_path/'export',None,None)
    assert not (tmp_path/'export').exists()


def test_all_cell_roots_two_roots_tangency_boundary_and_degeneracy():
    m=api()
    # Along x=y=t: xy-x+0.16 = (t-.2)(t-.8), no sign change at endpoints.
    roots,degenerate=m.quadratic_roots(1.,-1.,.16)
    np.testing.assert_allclose(roots,[.2,.8]);assert not degenerate
    roots,degenerate=m.quadratic_roots(1.,-1.,.25)
    assert degenerate and roots==[.5]
    assert m.quadratic_roots(0.,1.,0.)==([0.],False)
    assert m.quadratic_roots(0.,0.,0.)==([],True)


def test_abrupt_process_death_rolls_back_nodes_cells_geometry_and_cursor(tmp_path):
    r=recipe(tmp_path);m=api()
    config=tmp_path/'recipe.json';config.write_text(json.dumps(r))
    # Kill during a real MeshDB transaction, after all writes for the second
    # column but before commit. No cooperative-stop path is involved.
    script='''
import json,os,sys
from winding_field.range_surface import extract_range
from winding_field.surface_mesh import MeshDB
original=MeshDB.checkpoint_tile
def crash(self,index):
    original(self,index)
    if index==1:os._exit(91)
MeshDB.checkpoint_tile=crash
extract_range(json.load(open(sys.argv[1])),[-61,-59],sys.argv[2],block_ticks=1)
'''
    proc=subprocess.run([sys.executable,'-c',script,str(config),str(tmp_path/'crash')])
    assert proc.returncode==91
    assert not (tmp_path/'crash/RANGE.json').exists()
    with sqlite3.connect(tmp_path/'crash/mesh/mesh.sqlite') as db:
        assert json.loads(db.execute('SELECT value FROM meta WHERE key="next_tile"').fetchone()[0])==1
        assert db.execute('SELECT COUNT(DISTINCT tick_w) FROM range_nodes').fetchone()[0]==1
        assert db.execute('SELECT COUNT(*) FROM range_cells').fetchone()[0]==0
    m.extract_range(r,[-61,-59],tmp_path/'crash',resume=True,block_ticks=3)
    m.extract_range(r,[-61,-59],tmp_path/'direct',block_ticks=2)
    assert canonical(tmp_path/'crash')==canonical(tmp_path/'direct')


def test_cli_extract_join_and_scalar_wrapper_guard(tmp_path,capsys):
    from winding_field.cli import main
    r=recipe(tmp_path);config=tmp_path/'recipe.json';config.write_text(json.dumps(r))
    for name,lo,hi in [('a','-61','-60'),('b','-60','-59')]:
        assert main(['extract-range','--config',str(config),'--w-range',lo,hi,'--output',str(tmp_path/name)])==0
    assert main(['join-ranges',str(tmp_path/'b'),str(tmp_path/'a'),'--output',str(tmp_path/'joined')])==0
    assert canonical(tmp_path/'joined')['faces']
    request=tmp_path/'flatten.json';request.write_text(json.dumps(dict(output_frame=dict(id='test',origin_xyz_um=[0,0,0],voxel_um=1))))
    for path in [tmp_path/'joined',tmp_path/'joined/mesh']:
        assert main(['export-sheets','--mesh-bundle',str(path),'--source','/nonexistent-ct',
            '--output',str(tmp_path/'export'),'--flattening-config',str(request),'--flattening-method','original'])==1
        assert 'continuous_range' in capsys.readouterr().err


@pytest.mark.parametrize('cached',[False,True])
def test_radial_cell_solver_does_not_miss_same_sign_roots_or_tangencies(tmp_path,cached):
    from winding_field.surface_sources import FieldSource
    m=api();r=recipe(tmp_path,samples=8)
    r['geometry'].update(origin_xyz=[0,0,0],spacing_xyz=[1,1,1],shape_zyx=[2,2,2],sheet_step=1.)
    r['umbilicus']=[[0,0,0],[0,0,1]]
    z,y,x=np.indices((2,2,2));path=Path(r['field']['path'])
    # W=-59.875 selects pi/4, x=y=t. Both endpoints are above target.
    for constant,want in [(.16,'multiple'),(.25,'degenerate')]:
        np.save(path,x*y-x+constant-59.875);r['field']['sha256']=file_sha(path)
        with FieldSource(r['field'],r['geometry']) as source:
            state,roots,xyz=m._node(source,m._recipe(r,source),-479,0,m._PlaneCache(source,1<<20) if cached else None)
        assert state==want and xyz is None
        np.testing.assert_allclose(roots,np.array([.2,.8] if want=='multiple' else [.5])*np.sqrt(2),atol=1e-12)


def test_field_mask_and_long_edge_rejections_remain_audited(tmp_path):
    mask=np.ones((3,9,9),bool);mask[1,:,:]=False
    r=recipe(tmp_path,mask=mask);m=api()
    result=m.extract_range(r,[-61,-60],tmp_path/'masked')
    assert result['node_states']['unsupported']==9
    assert result['status']=='EMPTY' and result['cell_states']=={'node_rejected':16}
    r=recipe(tmp_path);r['max_edge_world']=.01
    result=m.extract_range(r,[-61,-60],tmp_path/'edges')
    assert result['status']=='EMPTY' and result['cell_states']=={'long_edge':16}


def reseal(path):
    # Deliberately forge outer checksums to exercise semantic join comparisons,
    # rather than merely failing at the first changed SQLite byte.
    m=api()
    with sqlite3.connect(path/'mesh/mesh.sqlite') as db:
        for (tick,) in db.execute('SELECT tick_w FROM range_audit').fetchall():
            db.execute('UPDATE range_audit SET sha256=? WHERE tick_w=?',(m._audit_sha(db,tick),tick))
    manifest=json.loads((path/'mesh/MESH.json').read_text())
    manifest['database_sha256']=file_sha(path/'mesh/mesh.sqlite')
    (path/'mesh/MESH.json').write_text(json.dumps(manifest))
    receipt=json.loads((path/'RANGE.json').read_text())
    receipt['mesh_manifest_sha256']=file_sha(path/'mesh/MESH.json')
    with sqlite3.connect(path/'mesh/mesh.sqlite') as db:
        receipt['node_states']=dict(db.execute('SELECT state,COUNT(*) FROM range_nodes GROUP BY state'))
        receipt['cell_states']=dict(db.execute('SELECT state,COUNT(*) FROM range_cells GROUP BY state'))
    (path/'RANGE.json').write_text(json.dumps(receipt))


@pytest.mark.parametrize('mutation',['rejection','orientation','presence','node'])
def test_join_rejects_resealed_conflicting_overlap_records(tmp_path,mutation):
    r=recipe(tmp_path,empty=mutation=='rejection');m=api()
    for name in ('a','b'):m.extract_range(r,[-61,-60],tmp_path/name)
    with sqlite3.connect(tmp_path/'b/mesh/mesh.sqlite') as db:
        if mutation=='rejection':
            db.execute('UPDATE range_cells SET state="long_edge" WHERE tick_w=-488 AND tick_z=0')
        elif mutation=='node':
            db.execute('UPDATE range_nodes SET roots="[99]" WHERE tick_w=-488 AND tick_z=0')
        else:
            key,triangles=db.execute('SELECT key,triangles FROM range_cells WHERE state="accepted" LIMIT 1').fetchone()
            if mutation=='presence':
                db.execute('DELETE FROM faces WHERE key=?',(key+':0',))
            else:
                triangles=json.loads(triangles);triangles[0].reverse()
                db.execute('UPDATE range_cells SET triangles=? WHERE key=?',(json.dumps(triangles),key))
                db.execute('UPDATE faces SET a=c,c=a WHERE key=?',(key+':0',))
    reseal(tmp_path/'b')
    with pytest.raises(ValueError,match='(overlapping|presence|orientation|inventory)'):
        m.join_ranges([tmp_path/'a',tmp_path/'b'],tmp_path/'bad')
    assert not (tmp_path/'bad/RANGE.json').exists()


def test_completed_resume_is_idempotent_and_audit_corruption_is_rejected(tmp_path):
    r=recipe(tmp_path);m=api()
    first=m.extract_range(r,[-61,-60],tmp_path/'a')
    assert m.extract_range(r,[-61,-60],tmp_path/'a',resume=True,block_ticks=1)==first
    with sqlite3.connect(tmp_path/'a/mesh/mesh.sqlite') as db:
        db.execute('UPDATE range_nodes SET state="no_root" WHERE tick_z=0')
    with pytest.raises(ValueError):m.extract_range(r,[-61,-60],tmp_path/'a',resume=True)


def test_default_thousand_lattice_does_not_round_decimal_literals(tmp_path):
    r=recipe(tmp_path);r.pop('samples_per_w');m=api()
    result=m.extract_range(r,['-60.375','-60.373'],tmp_path/'good')
    assert result['ticks']==[-60375,-60373] and result['faces']>0
    with pytest.raises(ValueError,match='lattice'):
        m.extract_range(r,['-60.37500000000000000000000000000000000000001','-60.373'],tmp_path/'bad')


def test_range_reader_checks_receipt_domain_inventory_not_just_db_hash(tmp_path):
    r=recipe(tmp_path);m=api();m.extract_range(r,[-61,-60],tmp_path/'a')
    receipt=json.loads((tmp_path/'a/RANGE.json').read_text())
    receipt['cell_states']={'accepted':999}
    (tmp_path/'a/RANGE.json').write_text(json.dumps(receipt))
    with pytest.raises(ValueError,match='state'):
        with m.RangeReader(tmp_path/'a'):pass


@pytest.mark.parametrize('cached',[False,True])
def test_distinct_roots_near_cell_boundary_are_not_deduplicated_as_unique(tmp_path,cached):
    from winding_field.surface_sources import FieldSource
    r=recipe(tmp_path);m=api()
    r['geometry'].update(origin_xyz=[0,0,0],spacing_xyz=[1,1,1],shape_zyx=[2,2,3],sheet_step=1.)
    r['umbilicus']=[[0,.5,0],[0,.5,1]]
    path=Path(r['field']['path'])
    np.save(path,np.broadcast_to(np.array([-59.,-60.-1e-11,-59.]),(2,2,3)))
    r['field']['sha256']=file_sha(path)
    with FieldSource(r['field'],r['geometry']) as source:
        state,roots,xyz=m._node(source,m._recipe(r,source),-480,0,m._PlaneCache(source,1<<20) if cached else None)
    assert state in ('multiple','degenerate') and xyz is None


def test_boundary_root_is_one_root_and_all_analytic_nodes_survive(tmp_path):
    r=recipe(tmp_path);m=api()
    result=m.extract_range(r,[-61,-59],tmp_path/'a')
    assert result['node_states']=={'unique':51}
    with sqlite3.connect(tmp_path/'a/mesh/mesh.sqlite') as db:
        roots,x,y=db.execute('SELECT roots,x,y FROM range_nodes WHERE tick_w=-488 AND tick_z=0').fetchone()
    assert json.loads(roots)==[4.] and (x,y)==(23.,5.)


@pytest.mark.parametrize('cached',[False,True])
def test_piecewise_tangency_at_cell_boundary_is_degenerate(tmp_path,cached):
    from winding_field.surface_sources import FieldSource
    r=recipe(tmp_path);m=api()
    r['geometry'].update(origin_xyz=[0,0,0],spacing_xyz=[1,1,1],shape_zyx=[2,2,3],sheet_step=1.)
    r['umbilicus']=[[0,.5,0],[0,.5,1]]
    path=Path(r['field']['path']);np.save(path,np.broadcast_to(np.array([-59.,-60.,-59.]),(2,2,3)))
    r['field']['sha256']=file_sha(path)
    with FieldSource(r['field'],r['geometry']) as source:
        state,roots,xyz=m._node(source,m._recipe(r,source),-480,0,m._PlaneCache(source,1<<20) if cached else None)
    assert state=='degenerate' and roots==[1.] and xyz is None


def test_writer_lease_is_held_until_final_wrapper_publication(tmp_path,monkeypatch):
    import fcntl
    r=recipe(tmp_path);m=api();write=m.atomic_json
    def checked_write(path,value):
        if Path(path).name=='RANGE.json':
            with (Path(path).parent/'mesh/.mesh.lock').open('a') as competing_writer:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(competing_writer,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write(path,value)
    monkeypatch.setattr(m,'atomic_json',checked_write)
    assert m.extract_range(r,[-61,-60],tmp_path/'a')['faces']>0


@pytest.mark.parametrize('extra',['face','vertex','level'])
def test_resealed_extra_actual_inventory_is_rejected(tmp_path,extra):
    r=recipe(tmp_path);m=api()
    for name in ('a','b'):m.extract_range(r,[-61,-60],tmp_path/name)
    with sqlite3.connect(tmp_path/'b/mesh/mesh.sqlite') as db:
        if extra=='face':
            cur=db.execute('INSERT INTO faces(level,a,b,c,area,key) SELECT level,a,b,c,area,key||":extra" FROM faces LIMIT 1')
            db.execute('UPDATE faces SET parent=id WHERE id=?',(cur.lastrowid,))
        elif extra=='vertex':
            db.execute('INSERT INTO vertices(level,key,x,y,z,local) VALUES(0,"extra",1,2,3,999999)')
        else:db.execute('INSERT INTO levels VALUES(1,42)')
    reseal(tmp_path/'b')
    with pytest.raises(ValueError,match='inventory'):
        with m.RangeReader(tmp_path/'b'):pass
    with pytest.raises(ValueError,match='inventory'):
        m.join_ranges([tmp_path/'a',tmp_path/'b'],tmp_path/'bad')
    assert not (tmp_path/'bad').exists()


@pytest.mark.parametrize('window',['base','legacy_partial','transaction_partial'])
def test_abrupt_startup_schema_crash_resumes(tmp_path,window):
    r=recipe(tmp_path);m=api();config=tmp_path/'recipe.json';config.write_text(json.dumps(r))
    script='''
import json,os,sqlite3,sys
from winding_field.range_surface import extract_range
from winding_field.surface_mesh import MeshDB
window=sys.argv[3]
if window=='transaction_partial':
    connect=sqlite3.connect
    class CrashConnection(sqlite3.Connection):
        def execute(self,sql,*args,**kwargs):
            result=super().execute(sql,*args,**kwargs)
            if sql.startswith('CREATE TABLE') and 'range_nodes' in sql:os._exit(91)
            return result
    sqlite3.connect=lambda *a,**kw:connect(*a,factory=CrashConnection,**kw)
else:
    original=MeshDB.__init__
    def crash(self,*a,**kw):
        original(self,*a,**kw)
        if window=='legacy_partial':
            self.db.execute('CREATE TABLE range_nodes(tick_w INTEGER,tick_z INTEGER,key TEXT UNIQUE,state TEXT,roots TEXT,x REAL,y REAL,z REAL,PRIMARY KEY(tick_w,tick_z))')
            self.db.commit()
        os._exit(91)
    MeshDB.__init__=crash
extract_range(json.load(open(sys.argv[1])),[-61,-60],sys.argv[2])
'''
    proc=subprocess.run([sys.executable,'-c',script,str(config),str(tmp_path/'crash'),window])
    assert proc.returncode==91
    with sqlite3.connect(tmp_path/'crash/mesh/mesh.sqlite') as db:
        tables={row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'range_%'")}
        assert tables==({'range_nodes'} if window=='legacy_partial' else set())
    m.extract_range(r,[-61,-60],tmp_path/'crash',resume=True)
    m.extract_range(r,[-61,-60],tmp_path/'direct')
    assert canonical(tmp_path/'crash')==canonical(tmp_path/'direct')


@pytest.mark.parametrize('empty',[False,True])
def test_missing_schema_with_committed_work_is_never_recreated(tmp_path,empty):
    r=recipe(tmp_path,empty=empty);m=api();m.extract_range(r,[-61,-60],tmp_path/'a')
    path=tmp_path/'a/mesh/mesh.sqlite'
    with sqlite3.connect(path) as db:db.execute('DROP TABLE range_audit')
    before=file_sha(path)
    with pytest.raises(ValueError,match='schema'):
        m.extract_range(r,[-61,-60],tmp_path/'a',resume=True)
    assert file_sha(path)==before
    with sqlite3.connect(path) as db:
        assert not db.execute("SELECT name FROM sqlite_master WHERE name='range_audit'").fetchall()


def test_schema_recovery_checks_residual_state_and_holds_writer_lock(tmp_path,monkeypatch):
    import fcntl
    from winding_field.surface_mesh import MeshDB
    r=recipe(tmp_path);m=api();base_init=MeshDB.__init__
    def residual(self,*a,**kw):
        base_init(self,*a,**kw)
        self.db.execute('INSERT INTO edges VALUES(0,1,2,1,1,1)');self.db.commit()
    monkeypatch.setattr(MeshDB,'__init__',residual)
    with pytest.raises(ValueError,match='non-pristine'):
        m.extract_range(r,[-61,-60],tmp_path/'dirty')
    with sqlite3.connect(tmp_path/'dirty/mesh/mesh.sqlite') as db:
        assert not db.execute("SELECT name FROM sqlite_master WHERE name='range_audit'").fetchall()
    monkeypatch.setattr(MeshDB,'__init__',base_init)
    connect=sqlite3.connect
    class CheckLock(sqlite3.Connection):
        def execute(self,sql,*a,**kw):
            if sql.startswith('CREATE TABLE') and 'range_' in sql:
                assert self.in_transaction
                with (tmp_path/'locked/mesh/.mesh.lock').open('a') as other:
                    with pytest.raises(BlockingIOError):fcntl.flock(other,fcntl.LOCK_EX|fcntl.LOCK_NB)
            return super().execute(sql,*a,**kw)
    monkeypatch.setattr(sqlite3,'connect',lambda *a,**kw:connect(*a,factory=CheckLock,**kw))
    assert m.extract_range(r,[-61,-60],tmp_path/'locked')['faces']>0


@pytest.mark.parametrize('budget',[0,100,1241,1<<20])
@pytest.mark.parametrize('block',[1,4])
def test_plane_cache_budget_and_block_preserve_exact_geometry(tmp_path,budget,block):
    r=recipe(tmp_path);m=api()
    m.extract_range(r,[-61,-59],tmp_path/'plain',cache_bytes=0,block_ticks=2)
    result=m.extract_range(r,[-61,-59],tmp_path/'cached',cache_bytes=budget,block_ticks=block)
    assert result['faces']>0 and result['area_world2']>0
    assert canonical(tmp_path/'cached')==canonical(tmp_path/'plain')


def test_plane_cache_reduces_real_source_reads_without_changing_geometry(tmp_path,monkeypatch):
    from winding_field.surface_sources import FieldSource
    r=recipe(tmp_path);m=api();read=FieldSource.read;calls=[]
    def counted(self,lo,hi):calls.append((lo,hi));return read(self,lo,hi)
    monkeypatch.setattr(FieldSource,'read',counted)
    m.extract_range(r,[-61,-59],tmp_path/'plain',cache_bytes=0)
    uncached=len(calls);calls.clear()
    m.extract_range(r,[-61,-59],tmp_path/'cached',cache_bytes=1<<20)
    assert len(calls)==3 and uncached>100
    assert canonical(tmp_path/'plain')==canonical(tmp_path/'cached')


@pytest.mark.parametrize('kind',['npy','chunks'])
def test_cached_planes_preserve_support_and_detect_source_mutation(tmp_path,kind):
    from winding_field.surface_sources import FieldSource
    from test_surface_sources import native,geometry,payload
    m=api()
    assert hasattr(m,'_PlaneCache'), 'bounded immutable plane cache missing'
    if kind=='npy':
        mask=np.ones((3,9,9),bool);mask[1,4,4]=False
        r=recipe(tmp_path,mask=mask);field=r['field'];meta=r['geometry'];changed=Path(field['mask']['path'])
    else:
        field,manifest=native(tmp_path);meta=geometry();changed=payload(field,manifest)
    with FieldSource(field,meta) as source:
        cache=m._PlaneCache(source,1<<20)
        for z in range(source.shape[0]):
            values,valid=cache.get(z)
            expected,support=source.read([z,0,0],[z+1,*source.shape[1:]])
            np.testing.assert_array_equal(values,expected)
            np.testing.assert_array_equal(valid,support)
            assert not values.flags.writeable and not valid.flags.writeable
        # Same bytes, changed inode: immutable cache must fail closed on a hit.
        replacement=changed.with_name('replacement'+changed.suffix)
        replacement.write_bytes(changed.read_bytes());replacement.replace(changed)
        with pytest.raises(ValueError,match='changed'):cache.get(0)


def test_cache_hit_mutation_before_commit_rolls_back_whole_column(tmp_path,monkeypatch):
    from winding_field.surface_sources import FieldSource
    r=recipe(tmp_path);m=api();validate=FieldSource.validate_region
    def mutate_after_check(self,low,high):
        validate(self,low,high)
        if low[0]==2:
            path=Path(r['field']['path'])
            replacement=tmp_path/'changed.npy';replacement.write_bytes(path.read_bytes());replacement.replace(path)
    monkeypatch.setattr(FieldSource,'validate_region',mutate_after_check)
    with pytest.raises(ValueError,match='changed'):
        m.extract_range(r,[-61,-60],tmp_path/'out',cache_bytes=1<<20,block_ticks=1)
    with sqlite3.connect(tmp_path/'out/mesh/mesh.sqlite') as db:
        assert json.loads(db.execute('SELECT value FROM meta WHERE key="next_tile"').fetchone()[0])==1
        assert db.execute('SELECT COUNT(*) FROM range_nodes').fetchone()[0]==3
        assert db.execute('SELECT COUNT(*) FROM range_cells').fetchone()[0]==0
        assert db.execute('SELECT COUNT(*) FROM faces').fetchone()[0]==0
    assert not (tmp_path/'out/RANGE.json').exists()


@pytest.mark.parametrize('target',[0.,-61.125,1e6])
@pytest.mark.parametrize('masked',[False,True])
def test_batched_coefficients_bit_exact_across_batch_boundaries(tmp_path,target,masked):
    from winding_field.surface_sources import FieldSource
    m=api()
    assert hasattr(m,'_coefficients'), 'bounded coefficient batching missing'
    r=recipe(tmp_path)
    path=Path(r['field']['path'])
    rng=np.random.default_rng(724)
    values=rng.normal(size=(3,9,9));np.save(path,values)
    r['field']['sha256']=file_sha(path)
    with FieldSource(r['field'],r['geometry']) as source:
        plane=source.read([0,0,0],[1,9,9])
        if masked:plane[1][0,3,3]=False
        # Over 256 intervals exercises batch transitions, including support gaps.
        cuts=np.linspace(0.,1.,601).tolist();p=np.array([1.25,2.125]);v=np.array([4.5,3.25]);shape=np.array([8,8])
        actual=list(m._coefficients(source,plane,p,v,shape,cuts,target,0))
        expected=[]
        for lo,hi in zip(cuts,cuts[1:]):
            midpoint=p+v*((lo+hi)/2)
            ix,iy=np.minimum(np.maximum(np.floor(midpoint).astype(int),0),shape-1)
            valid=plane[1][:,iy:iy+2,ix:ix+2].all()
            if not valid:expected.append((lo,hi,False,0.,0.,0.));continue
            f00,f10,f01,f11=map(float,plane[0][0,iy:iy+2,ix:ix+2].ravel())
            x,y=p+v*lo-[ix,iy];dx,dy=v*(hi-lo)
            cross=f11-f10-f01+f00
            a=cross*dx*dy;b=(f10-f00)*dx+(f01-f00)*dy+cross*(x*dy+y*dx)
            c=f00+(f10-f00)*x+(f01-f00)*y+cross*x*y-target
            expected.append((lo,hi,True,a,b,c))
        def bits(rows):return [(float(lo).hex(),float(hi).hex(),bool(ok),*([float(x).hex() for x in (a,b,c)] if ok else [])) for lo,hi,ok,a,b,c in rows]
        assert bits(actual)==bits(expected)
