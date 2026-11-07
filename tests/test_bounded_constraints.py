import io
import sqlite3
import numpy as np
import pytest


def test_physical_upper_boundary_survives_grid_roundoff(tmp_path):
    from winding_field.bounded_constraints import build,PairCatalog
    from winding_field.util import file_sha
    origin=np.array([50595.840000000004,41748.48,68290.56])
    spacing=np.full(3,272.16)
    high=origin+2*spacing
    assert ((high-origin)/spacing>2).any()
    points=np.array([origin+spacing,high])
    blob=io.BytesIO();np.save(blob,points)
    source=tmp_path/'fibres.sqlite'
    with sqlite3.connect(source) as db:
        db.execute('CREATE TABLE curves(id INTEGER PRIMARY KEY,points BLOB)')
        db.execute('INSERT INTO curves VALUES (?,?)',(1,blob.getvalue()))
    grid=dict(shape_zyx=[3]*3,origin_xyz=origin.tolist(),spacing_xyz=spacing.tolist())
    root=tmp_path/'pairs'
    build(root,source,file_sha(source),grid,weight=1.,quota_bytes=1<<20)
    with PairCatalog(root,grid) as provider:
        rows=provider([0]*3,[3]*3,lambda nodes:np.zeros(len(nodes)))
    z=np.indices((3,3,3))[0].ravel()
    np.testing.assert_allclose(rows.matrix@z,[-.5,-.5],atol=1e-12,rtol=0)
    np.testing.assert_allclose(rows.matrix@np.ones(27),0,atol=1e-15)

    # Even one representable step outside physical support must still be rejected.
    import json
    for endpoint in (np.nextafter(high,np.inf),np.nextafter(origin,-np.inf)):
        with sqlite3.connect(root/'pairs.sqlite') as db:
            db.execute('UPDATE pairs SET points=? WHERE id=1',
                       (np.array([origin+spacing,endpoint],dtype='<f8').tobytes(),))
        marker=root/'PAIRS.json';receipt=json.loads(marker.read_text())
        receipt['sha256']=file_sha(root/'pairs.sqlite');marker.write_text(json.dumps(receipt))
        with PairCatalog(root,grid) as provider,pytest.raises(ValueError,match='outside physical'):
            provider([0]*3,[3]*3,lambda nodes:np.zeros(len(nodes)))


@pytest.mark.parametrize('cache_bytes',[0,1,1<<20])
def test_global_pair_catalog_preserves_cross_block_row_and_exterior_values(tmp_path,cache_bytes):
    from winding_field.bounded_constraints import build,PairCatalog
    from constraint_templates import StencilConstraints
    from winding_field.util import file_sha
    source=tmp_path/'fibres.sqlite'
    points=np.array([[.5,1.,1.],[6.5,1.,1.]],'f8')
    blob=io.BytesIO();np.save(blob,points)
    with sqlite3.connect(source) as db:
        db.execute('CREATE TABLE curves(id INTEGER PRIMARY KEY,points BLOB)')
        db.execute('INSERT INTO curves VALUES (?,?)',(1,blob.getvalue()))
    grid=dict(shape_zyx=[3,3,8],origin_xyz=[0.,0.,0.],spacing_xyz=[1.,1.,1.])
    root=tmp_path/'pairs'
    assert build(root,source,file_sha(source),grid,weight=2.,quota_bytes=1<<20)=='COMPLETE'
    assert build(root,source,file_sha(source),grid,weight=2.,quota_bytes=1<<20)=='COMPLETE'
    phi=np.arange(72.).reshape(3,3,8)**2
    sampled=np.linspace(points[0],points[1],7)
    expected=StencilConstraints.pairs(sampled[:-1],sampled[1:],[0]*3,[1]*3,(3,3,8),weight=2.)
    with PairCatalog(root,grid,cache_bytes=cache_bytes) as provider:
        lo=np.array([0,0,3]);hi=np.array([3,3,5])
        q=provider(lo,hi,lambda nodes:phi.ravel()[nodes])
        reference=expected.restrict(lo,hi,lambda nodes:phi.ravel()[nodes])
        np.testing.assert_allclose(q.matrix.toarray(),reference.matrix.toarray())
        np.testing.assert_allclose(q.targets,reference.targets)
        assert np.any(q.targets!=0)
        # Changing exterior values must never be hidden by retained templates.
        later=provider(lo,hi,lambda nodes:phi.ravel()[nodes]+7)
        expected_later=expected.restrict(lo,hi,lambda nodes:phi.ravel()[nodes]+7)
        np.testing.assert_array_equal(later.targets,expected_later.targets)
        np.testing.assert_array_equal(q.targets,reference.targets)
        assert provider.stats['bytes']<=cache_bytes
        assert provider.stats['hits']==(1 if cache_bytes>1 else 0)
        from winding_field.global_residual import verify
        def read(lo,hi):return phi[tuple(slice(a,b) for a,b in zip(lo,hi))]
        def obs(lo,hi):return np.zeros((*tuple(hi-lo),3),'f4'),np.ones(tuple(hi-lo),'f4'),np.ones(tuple(hi-lo),bool),{}
        a=verify((3,3,8),2,[1]*3,read,obs,constraints=provider)
        b=verify((3,3,8),2,[1]*3,read,obs,constraints=expected.restrict)
        assert a==b
    assert provider.stats['bytes']==0
    with sqlite3.connect(root/'pairs.sqlite') as db:db.execute('UPDATE pairs SET points=zeroblob(48) WHERE id=1')
    with pytest.raises(ValueError,match='changed'):
        build(root,source,file_sha(source),grid,weight=2.,quota_bytes=1<<20)
