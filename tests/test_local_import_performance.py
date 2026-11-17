import importlib.util
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pytest

from test_lasagna_input import source_case
from test_splines import upstream_curves
from winding_field.util import file_sha


def benchmark():
    path=Path(__file__).resolve().parents[1]/'tools/bench_local_imports.py'
    assert path.is_file(), 'local import benchmark missing'
    spec=importlib.util.spec_from_file_location('bench_local_imports',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def local_case(root, *, physical=True, count=8):
    root.mkdir()
    if physical:
        volume,_,config=source_case(root,shape=(8,8,16),whole=True)
        volume.close();shape=(8,8,16)
    else:
        from test_lasagna_stream import fixture
        ct,_,config=fixture(root);shape=(9,10,11);config['source']=str(ct)
        (root/'geometry.json').write_text(json.dumps(dict(source=str(ct),
            origin_xyz_um=[100.,200.,300.],spacing_xyz_um=[5.,10.,20.])))
    fibres=root/'fibres.afv'
    upstream_curves(fibres,shape,[(1,np.array([[1.,2.,3.],[float(shape[2]-2),2.,3.]]))]*count)
    config.update(fibres=str(fibres),fibres_geometry=str(root/'geometry.json'),chunk=4,
                  threads=1,prediction='qual-v2',cache_bytes=1<<20)
    path=root/'config.json';path.write_text(json.dumps(config))
    return path


@pytest.mark.parametrize('physical',[False,True])
def test_benchmark_records_stage_costs_and_preserves_inputs(tmp_path,physical):
    config=local_case(tmp_path/'input',physical=physical)
    def snapshot():
        return {str(p):(file_sha(p),p.stat().st_mtime_ns) for p in (tmp_path/'input').rglob('*')
                if p.is_file() and 'cache' not in p.parts}
    before=snapshot()
    result=benchmark().run(tmp_path/'benchmark',config=config,repeats=1,workers=(1,),profile=True)
    assert before==snapshot()
    assert result['logical_results_equal']
    assert len(result['measurements'])==1
    record=result['measurements'][0]
    assert record['curves']>0 and record['output_bytes']>0 and record['peak_rss_bytes']>0
    assert record['decodes']>0 and record['support_loads']>0 and record['commits']>0
    assert record['hashed_bytes']>0
    assert all(record['stage_seconds'][name]>0 for name in ('prepare','fibres','export_lasagna','export_afv',
                                                           'validation','map_compression'))
    assert result['profile']['profiled'] and not record['profiled']
    assert result['profile']['pid']!=record['pid']
    assert record['completed_resume_seconds']>0
    assert result['resume']['partial_resume_seconds']>0 and result['resume']['partial_pause_seconds']>0
    assert result['resume']['logical_sha256']==record['logical_sha256']
    assert result['runtime']['core_version']==importlib.metadata.version('winding-field')


@pytest.mark.parametrize('overlap',['existing','inside','contains'])
def test_benchmark_rejects_existing_or_overlapping_output(tmp_path,overlap):
    config=local_case(tmp_path/'input')
    output={'existing':tmp_path/'input','inside':tmp_path/'input/ct/bench','contains':tmp_path}[overlap]
    with pytest.raises((ValueError,FileExistsError)):
        benchmark().run(output,config=config,repeats=1)
    assert not (tmp_path/'input/ct/bench').exists()


def test_adjacent_lasagna_reads_reuse_decoded_chunks(tmp_path):
    from contextlib import closing
    from test_lasagna_stream import fixture,geometry
    from winding_field.io import Volume
    from winding_field.lasagna_stream import LasagnaInput
    ct,_,config=fixture(tmp_path)
    with closing(Volume(ct,cache=tmp_path/'cache')) as volume, LasagnaInput(volume,tmp_path/'run',config) as reader:
        first=reader.read(geometry(volume),[0]*3,[2]*3)
        second=reader.read(geometry(volume),[1]*3,[3]*3)
        assert first['support'].all() and second['support'].all()
        assert sum(p.stats()['decodes'] for p in reader.products.values())==4
        assert sum(p._decoded_limit for p in reader.products.values())<=256<<20
        assert sum(p.stats()['decoded_cache_bytes'] for p in reader.products.values())<=256<<20


def test_cached_prediction_mutation_is_rejected_before_prepared_publication(tmp_path,monkeypatch):
    from test_lasagna_stream import fixture
    from winding_field import predict
    from winding_field.io import Volume
    from winding_field.artifacts import Tiles
    from contextlib import closing
    ct,root,config=fixture(tmp_path);config['fibres']='unused-by-prepare'
    original=Tiles.write;changed=False
    def write(self,*args,**kwargs):
        nonlocal changed
        result=original(self,*args,**kwargs)
        if not changed:
            changed=True
            (root/'nx.ome.zarr/2/0.0.0').write_bytes(bytes(64))
        return result
    monkeypatch.setattr(Tiles,'write',write)
    with closing(Volume(ct,cache=tmp_path/'cache')) as volume:
        with pytest.raises(ValueError,match='changed'):
            predict.prepare(volume,tmp_path/'run',config)
    assert not (tmp_path/'run/PREPARED.json').exists()


@pytest.mark.parametrize('embedded',[False,True])
def test_direct_calibrated_import_avoids_second_map_copy(tmp_path,embedded):
    from winding_field import lasagna_input
    volume,geometry,config=source_case(tmp_path,embedded=embedded)
    try:
        lasagna_input.prepare(volume,tmp_path/'reference',config,geometry)
        reference=lasagna_input.read_observations(tmp_path/'reference',geometry,[0]*3,[4]*3)
        with lasagna_input.CalibratedLasagnaInput(volume,tmp_path/'direct',config,geometry) as reader:
            data=reader.read(geometry,[0]*3,[4]*3)
            assert reader.identity['kind']=='external-lasagna-physical-direct-v1'
            for name in reference:np.testing.assert_array_equal(data[name],reference[name])
        assert not (tmp_path/'direct/lasagna-source/maps').exists()
        assert (tmp_path/'direct/lasagna-source/WORK.json').is_file()
        (tmp_path/'direct/lasagna-source/nx.sqlite').unlink()
        with pytest.raises((ValueError,FileNotFoundError)):
            lasagna_input.CalibratedLasagnaInput(volume,tmp_path/'direct',config,geometry)
        assert not (tmp_path/'direct/lasagna-source/nx.sqlite').exists()
    finally:volume.close()


@pytest.mark.parametrize('physical',[False,True])
@pytest.mark.parametrize('workers',[2,4])
def test_parallel_local_imports_match_serial_maps_and_fibres(tmp_path,monkeypatch,physical,workers):
    from contextlib import closing
    from test_import_parallel import resources
    from test_afv_batching import inventory
    from winding_field import workflow
    from winding_field.artifacts import Tiles
    path=local_case(tmp_path/'input',physical=physical,count=12)
    config=json.loads(path.read_bytes());resources(monkeypatch)
    workflow.run(dict(config,output=str(tmp_path/'serial'),import_workers=1),until='fibres')
    workflow.run(dict(config,output=str(tmp_path/'parallel'),import_workers=workers),until='fibres')
    assert inventory(tmp_path/'serial')==inventory(tmp_path/'parallel')
    with closing(Tiles(tmp_path/'serial/maps')) as a,closing(Tiles(tmp_path/'parallel/maps')) as b:
        for key in a.keys():
            left,right=a.load(key),b.load(key)
            for name in left:np.testing.assert_array_equal(left[name],right[name])
    root=tmp_path/'parallel'/('lasagna-source' if physical else 'lasagna-import')
    stats=json.loads((root/'PERFORMANCE.json').read_bytes())
    assert stats['workers']==workers and stats['worker_pids']
    assert not (tmp_path/'parallel/lasagna-source/maps').exists()
    assert not list((tmp_path/'parallel').glob('.import-workers-*'))


@pytest.mark.parametrize('physical',[False,True])
@pytest.mark.parametrize('mutation',[False,True])
def test_parallel_prepare_stop_or_mutation_never_certifies_partial_inputs(tmp_path,monkeypatch,physical,mutation):
    from test_import_parallel import resources
    from winding_field import workflow
    from winding_field.artifacts import Tiles
    import multiprocessing as mp
    import sqlite3
    config=json.loads(local_case(tmp_path/'input',physical=physical).read_bytes())
    config.update(output=str(tmp_path/'run'),import_workers=2);resources(monkeypatch)
    run=tmp_path/'run';original=Tiles.write;first=run/'maps/0_0_0.npz';interrupted=False
    previous={p.pid for p in mp.active_children()}
    def write(self,*args,**kwargs):
        nonlocal interrupted
        result=original(self,*args,**kwargs)
        if self.root==run/'maps' and not interrupted:
            interrupted=True
            if mutation:
                root=run/('lasagna-source' if physical else 'lasagna-import')
                with sqlite3.connect(root/'nx.sqlite') as db:
                    key=db.execute('SELECT key FROM chunks ORDER BY key LIMIT 1').fetchone()[0]
                manifest=json.loads(Path(config['lasagna_input']).read_bytes())
                channel=Path(config['lasagna_input']).parent/manifest['groups']['nx']['zarr']
                (channel/key).write_bytes(bytes(64))
            else:(run/'STOP').touch()
        return result
    with monkeypatch.context() as patch:
        patch.setattr(Tiles,'write',write)
        if mutation:
            with pytest.raises(ValueError,match='changed'):workflow.run(config,until='prepare')
        else:assert workflow.run(config,until='prepare')['status']=='PAUSED'
    assert not (run/'PREPARED.json').exists()
    assert not list(run.glob('.import-workers-*'))
    assert {p.pid for p in mp.active_children()}<=previous
    if not mutation:
        before=(file_sha(first),first.stat().st_mtime_ns)
        (run/'STOP').unlink()
        assert 'prepare' in workflow.run(config,resume=True,until='prepare')['stages']
        assert before==(file_sha(first),first.stat().st_mtime_ns)


def test_stop_while_tile_jobs_are_pending_is_paused_not_failed(tmp_path,monkeypatch):
    from concurrent.futures import ProcessPoolExecutor
    from test_import_parallel import resources
    from winding_field import workflow
    config=json.loads(local_case(tmp_path/'input').read_bytes())
    root=tmp_path/'run';config.update(output=str(root),import_workers=2);resources(monkeypatch)
    submit=ProcessPoolExecutor.submit
    def interrupted(self,*args,**kwargs):
        result=submit(self,*args,**kwargs);(root/'STOP').touch();return result
    with monkeypatch.context() as patch:
        patch.setattr(ProcessPoolExecutor,'submit',interrupted)
        assert workflow.run(config,until='prepare')['status']=='PAUSED'
    assert not (root/'PREPARED.json').exists()
    (root/'STOP').unlink()
    assert 'prepare' in workflow.run(config,resume=True,until='prepare')['stages']
