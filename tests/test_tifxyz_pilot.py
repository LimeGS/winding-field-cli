import importlib.util
from pathlib import Path


def test_fixed_pilot_uses_sorted_quantiles_not_outcomes():
    path=Path(__file__).resolve().parents[1]/'tools/bench_direct_tifxyz.py'
    assert path.exists(),'pilot implementation missing'
    spec=importlib.util.spec_from_file_location('pilot',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    rows=[dict(tile=i,lo_zyx=[512,i*128,1024],hi_zyx=[640,i*128+128,1152],parts=[{'level':-4,'faces':10}]) for i in range(10)]
    result=m.select_rois(list(reversed(rows)),(4097,4097,4097),-4)
    assert [x['tile'] for x in result]==[0,3,5,8]
    assert result[0]['lo_zyx']==[320,0,832]
    assert result[-1]['lo_zyx']==[320,832,832]
    assert all([b-a for a,b in zip(x['lo_zyx'],x['hi_zyx'])]==[513]*3 for x in result)
