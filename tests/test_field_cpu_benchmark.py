import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tarfile
from types import SimpleNamespace

import pytest


def benchmark():
    path=Path(__file__).parents[1]/'benchmarks/field_cpu.py'
    spec=importlib.util.spec_from_file_location('field_cpu_benchmark',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def record(variant='serial',repetition=0,wall=1.,total=1.,**extra):
    values=dict(input_sha256='fixed',field_sha256='same',variant=variant,
                source_sha256='source',benchmark_sha256='benchmark',config={'cache':0},
                repetition=repetition,wall_seconds=wall,process_wall_seconds=total)
    values.update(extra)
    return values


def test_benchmark_rejects_different_inputs_and_preserves_raw_repetitions():
    module=benchmark()
    records=[record(repetition=index,wall=value,total=value) for index,value in enumerate((3.,1.,2.,4.,5.))]
    result=module.summarize(records)
    assert result['raw']==records
    assert result['median_seconds']==3.
    with pytest.raises(ValueError,match='inputs'):
        module.summarize([records[0],dict(records[1],input_sha256='other')])
    with pytest.raises(ValueError,match='field'):
        module.summarize([records[0],dict(records[1],field_sha256='changed')])


@pytest.mark.parametrize('key,value',[('source_sha256','other'),('benchmark_sha256','other'),
                                      ('config',{'cache':1<<20})])
def test_benchmark_rejects_mixed_provenance_within_variant(key,value):
    module=benchmark()
    with pytest.raises(ValueError,match='provenance'):
        module.summarize([record(repetition=0),record(repetition=1,**{key:value})])


def test_benchmark_allows_distinct_baseline_candidate_provenance():
    module=benchmark()
    records=[record(variant='base',source_sha256='base-source',benchmark_sha256='base-harness',
                    config={'cache':0},repetition=0),
             record(variant='cache',source_sha256='cache-source',benchmark_sha256='cache-harness',
                    config={'cache':1<<20},repetition=0)]
    result=module.summarize(records)
    assert set(result['variants'])=={'base','cache'}
    assert result['median_seconds']==1.


def test_paired_comparison_uses_total_time():
    module=benchmark()
    records=[record(variant=variant,repetition=index,wall=0.,total=value)
             for index in range(5) for variant,value in (('baseline',10.),('candidate',5.))]
    assert module.paired_comparison(records,'baseline','candidate')==dict(
        ratios=[2.]*5,median_ratio=2.,wins=5,count=5)


@pytest.mark.parametrize('mutation',[
    pytest.param(lambda record:{key:value for key,value in record.items() if key!='repetition'},
                 id='missing-repetition'),
    pytest.param(lambda record:dict(record,repetition=0),id='duplicate-repetition'),
    pytest.param(lambda record:dict(record,process_wall_seconds=0.),id='zero-time'),
    pytest.param(lambda record:dict(record,process_wall_seconds=-1.),id='negative-time'),
    pytest.param(lambda record:dict(record,process_wall_seconds=float('nan')),id='nan-time'),
    pytest.param(lambda record:{key:value for key,value in record.items() if key!='process_wall_seconds'},
                 id='missing-time'),
])
def test_paired_comparison_rejects_invalid_pairs(mutation):
    module=benchmark()
    records=[record(variant=variant,repetition=index) for index in range(2)
             for variant in ('baseline','candidate')]
    records[-1]=mutation(dict(records[-1]))
    with pytest.raises(ValueError):
        module.paired_comparison(records,'baseline','candidate')


@pytest.mark.parametrize('key,value',[('source_sha256','other'),('benchmark_sha256','other'),
                                      ('config',{'cache':1<<20})])
def test_paired_comparison_rejects_mixed_provenance(key,value):
    module=benchmark()
    records=[record(variant=variant,repetition=index) for index in range(2)
             for variant in ('baseline','candidate')]
    records[-1]=dict(records[-1],**{key:value})
    with pytest.raises(ValueError,match='provenance'):
        module.paired_comparison(records,'baseline','candidate')


def test_paired_comparison_rejects_incomplete_repetition_sets():
    module=benchmark()
    records=[record(variant='baseline',repetition=0),record(variant='baseline',repetition=1),
             record(variant='candidate',repetition=0)]
    with pytest.raises(ValueError,match='repetitions'):
        module.paired_comparison(records,'baseline','candidate')


def test_once_records_are_self_describing(tmp_path):
    module=benchmark();root=Path(__file__).parents[1]
    args=SimpleNamespace(variant='base',output=tmp_path/'out',source_root=root,
                         cache_bytes=0,workers=0,buffer_bytes=0)
    module.run_once(args)
    record=json.loads((tmp_path/'out/result.json').read_text())
    assert record['benchmark_sha256']==hashlib.sha256(
        (root/'benchmarks/field_cpu.py').read_bytes()).hexdigest()
    assert module.summarize([dict(record,repetition=0)])['median_seconds']==record['wall_seconds']


def test_benchmark_separates_initialization_cold_and_warm_closures():
    module=benchmark()
    result=module.sweep_timings([.1,.2,.3,.4,.6,.8,1.,1.2,1.3,1.4,1.5,1.6],4)
    assert result['initialization_seconds']==.4
    assert result['first_sweep_seconds']==pytest.approx(.8)
    assert result['later_sweep_seconds']==pytest.approx([.4])
    with pytest.raises(ValueError):module.sweep_timings([.1,.2],4)


def test_sdist_contains_the_benchmark_used_by_shipped_tests(tmp_path):
    root=Path(__file__).resolve().parents[1]
    result=subprocess.run([sys.executable,'-m','build','--sdist','--no-isolation','--outdir',str(tmp_path)],
                          cwd=root,text=True,capture_output=True)
    assert result.returncode==0,result.stdout+result.stderr
    with tarfile.open(next(tmp_path.glob('*.tar.gz'))) as archive:
        member=next((name for name in archive.getnames() if name.endswith('/benchmarks/field_cpu.py')),None)
        assert member is not None,'sdist omits the documented benchmark required by its shipped tests'
        assert archive.extractfile(member).read()==(root/'benchmarks/field_cpu.py').read_bytes()
