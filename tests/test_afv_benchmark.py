import importlib.util
from pathlib import Path


def tool():
    path=Path(__file__).resolve().parents[1]/'tools/bench_afv_import.py'
    assert path.is_file(), 'reproducible AFV benchmark missing'
    spec=importlib.util.spec_from_file_location('bench_afv_import',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_benchmark_compares_independent_processes_and_logical_results(tmp_path):
    from winding_field import afv
    report=tool().run(tmp_path/'benchmark',Path(afv.__file__),blocks=12,retained_every=3,repeats=1,profile=True)
    assert report['logical_results_equal']
    assert report['input']['blocks']==12
    records=report['measurements']
    assert {r['variant'] for r in records}=={'baseline','candidate'}
    assert len({r['pid'] for r in records})==2
    assert all(r['blocks']==12 and 0<r['commits']<=12 for r in records)
    assert all(r['seconds']>0 and r['peak_rss_bytes']>0 and r['commit_seconds']>=0 for r in records)
    assert all(r['synchronous']==2 and r['journal_mode']=='delete' for r in records)
    assert len({r['logical_sha256'] for r in records})==1
    assert report['profile']['top_cumulative']
    assert (tmp_path/'benchmark/RESULT.json').is_file()


def test_benchmark_refuses_existing_output(tmp_path):
    import pytest
    from winding_field import afv
    with pytest.raises(FileExistsError):tool().run(tmp_path,Path(afv.__file__),blocks=2,repeats=1)


def test_real_reference_mode_never_modifies_input_or_prepared_maps(tmp_path):
    import hashlib,json
    from test_afv_batching import many
    from winding_field import afv
    reference=tmp_path/'reference';config=many(reference,4)
    path=reference/'import.json';path.write_text(json.dumps(config))
    def snapshot():
        return {str(p.relative_to(reference)):(hashlib.sha256(p.read_bytes()).hexdigest(),p.stat().st_mtime_ns)
                for p in reference.rglob('*') if p.is_file()}
    before=snapshot()
    report=tool().run(tmp_path/'benchmark',Path(afv.__file__),prepared=reference,config_path=path,repeats=1)
    assert not report['input']['generated'] and report['input']['blocks']==4
    assert report['logical_results_equal']
    assert snapshot()==before
