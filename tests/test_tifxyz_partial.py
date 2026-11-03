from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from test_tifxyz import engine,controlled,physical_source,settings,tree_state
from winding_field.tifxyz_contract import Blocked,validate_request
from winding_field.tifxyz_field import open_physical_field
from winding_field.util import atomic_json,file_sha,digest


def partial_plane(tmp_path,policy='allow_partial'):
    from winding_field.tifxyz import _export_source
    source=physical_source(tmp_path/'source',shape=(3,8,8));out=tmp_path/'out'
    with open_physical_field(source) as field:
        config=settings(field,pitch_um=4.,tolerance_um=3.,coverage_policy=policy)
        result=_export_source(field,out,config)
    return source,out,config,result


def test_partial_policy_is_explicit_and_frozen():
    assert validate_request({'levels':[0.]})['coverage_policy']=='strict'
    assert validate_request({'levels':[0.],'coverage_policy':'allow_partial'})['coverage_policy']=='allow_partial'
    for bad in ('partial',True,None,[]):
        with pytest.raises(ValueError):validate_request({'levels':[0.],'coverage_policy':bad})


def test_natural_incomplete_plane_exports_partial_without_ready(tmp_path,engine):
    source,out,config,result=partial_plane(tmp_path)
    assert result['status']=='PARTIAL' and result['coverage_complete'] is False
    assert not (out/'READY.json').exists() and (out/'PARTIAL.json').is_file()
    marker=json.loads((out/'PARTIAL.json').read_text());receipt=json.loads((out/'TIFXYZ.json').read_text())
    assert marker['status']==receipt['status']=='PARTIAL'
    assert marker['tifxyz_sha256']==file_sha(out/'TIFXYZ.json')
    item=receipt['levels'][0];verified=json.loads((out/item['segment']/'VERIFIED.json').read_text())
    assert item['status']==verified['status']=='PARTIAL' and item['coverage_complete'] is False
    audit=verified['audit']
    assert audit['geometry_complete'] is True and audit['blocked_phase']=='coverage'
    assert audit['source_scan_complete'] is False and audit['remaining_source_coverage']=='UNMEASURED'
    assert audit['reasons'][0]['code'] in ('SOURCE_UNCOVERED','COVERAGE_DEPTH_LIMIT')
    region=audit['reasons'][0]['region']
    assert set(region)>={'cell_zyx','global_cell_zyx','subcell_index_zyx','depth'}
    from winding_field.tifxyz_store import iter_tiff_tiles
    x=next(iter_tiff_tiles(out/'level-0000/x.tif',[3,3],9))[1]
    assert np.any(x==-1) and np.any(x>=0)
    assert result['independent_reprojection'] is False


def test_same_incomplete_plane_remains_blocked_under_strict_policy(tmp_path,engine):
    source,out,config,result=partial_plane(tmp_path,'strict')
    assert result['status']=='BLOCKED'
    assert not (out/'READY.json').exists() and not (out/'PARTIAL.json').exists()


def test_partial_fresh_verify_and_resume_are_readonly_and_stay_partial(tmp_path,engine):
    from winding_field.tifxyz import _verify_source,_export_source
    source,out,config,result=partial_plane(tmp_path)
    assert result['status']=='PARTIAL'
    before=tree_state(out);source_before=tree_state(source)
    with open_physical_field(source) as field:verified=_verify_source(field,out)
    assert verified['status']=='TIFXYZ_PARTIAL_VERIFIED' and verified['coverage_complete'] is False
    assert verified['independent_reprojection'] is True
    assert before==tree_state(out) and source_before==tree_state(source)
    with open_physical_field(source) as field:resumed=_export_source(field,out,config,True)
    assert resumed['status']=='PARTIAL' and before==tree_state(out)
    assert not (out/'READY.json').exists()


@pytest.mark.parametrize('code',['OUTPUT_ORIENTATION','OUTPUT_INTERSECTION','OUTPUT_UNSUPPORTED_STENCIL',
    'NUMERICAL_UNCERTAINTY','GPU_MEMORY_BUDGET','COVERAGE_WORK_LIMIT','IDENTICALLY_LEVEL_CELL'])
def test_allow_partial_never_hides_hard_failures(tmp_path,engine,monkeypatch,code):
    from winding_field import tifxyz_audit
    def blocked(*args,**kwargs):raise Blocked(code)
    monkeypatch.setattr(tifxyz_audit,'_coverage' if code in ('COVERAGE_WORK_LIMIT','IDENTICALLY_LEVEL_CELL','GPU_MEMORY_BUDGET') else '_output_geometry',blocked)
    source,out,config,result=partial_plane(tmp_path)
    assert result['status']=='BLOCKED'
    assert not (out/'PARTIAL.json').exists() and not (out/'READY.json').exists()


@pytest.mark.parametrize('damage',['both-markers','relabel','coverage-flag','region','tiff'])
def test_partial_corruption_never_verifies(tmp_path,engine,damage):
    from winding_field.tifxyz import _verify_source
    source,out,config,result=partial_plane(tmp_path)
    assert result['status']=='PARTIAL'
    if damage=='both-markers':(out/'READY.json').write_bytes((out/'PARTIAL.json').read_bytes())
    elif damage=='relabel':(out/'PARTIAL.json').rename(out/'READY.json')
    elif damage=='coverage-flag':
        marker=json.loads((out/'PARTIAL.json').read_text());marker['coverage_complete']=True;atomic_json(out/'PARTIAL.json',marker)
    elif damage=='region':
        p=out/'level-0000/VERIFIED.json';data=json.loads(p.read_text());data['audit']['reasons'][0]['region']['depth']=1000000;atomic_json(p,data)
    else:
        p=out/'level-0000/x.tif';raw=bytearray(p.read_bytes());raw[-1]^=1;p.write_bytes(raw)
    with open_physical_field(source) as field,pytest.raises(ValueError):_verify_source(field,out)


def test_complete_result_under_allow_partial_still_uses_ready(tmp_path,engine):
    from test_tifxyz import export_plane
    source,out=export_plane(tmp_path,coverage_policy='allow_partial')
    assert (out/'READY.json').exists() and not (out/'PARTIAL.json').exists()
    assert json.loads((out/'TIFXYZ.json').read_text())['coverage_complete'] is True


def test_partial_cleanup_failure_cannot_be_downgraded(tmp_path,engine,monkeypatch):
    from winding_field.tifxyz_predicates import Backend
    original=Backend.__exit__
    def fail(self,kind,value,trace):
        original(self,kind,value,trace)
        raise Blocked('GPU_MEMORY_BUDGET','cleanup failed')
    monkeypatch.setattr(Backend,'__exit__',fail)
    source,out,config,result=partial_plane(tmp_path)
    assert result['status']=='BLOCKED' and result['reasons'][0]['code']=='GPU_MEMORY_BUDGET'
    assert not (out/'PARTIAL.json').exists()


def test_failed_partial_marker_publication_can_resume_without_rewriting_tiles(tmp_path,engine,monkeypatch):
    from winding_field import tifxyz as code
    original=code._write
    def fail(path,*args,**kwargs):
        if path.name=='PARTIAL.json':raise OSError('publication fault')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(code,'_write',fail)
    with pytest.raises(OSError,match='publication fault'):partial_plane(tmp_path)
    out=tmp_path/'out';assert not (out/'PARTIAL.json').exists() and not (out/'READY.json').exists()
    before={p:(file_sha(p),p.stat().st_mtime_ns) for p in out.rglob('tile-*.bin')}
    config=json.loads((out/'EXPORT.json').read_text())['effective_settings']
    monkeypatch.setattr(code,'_write',original)
    with open_physical_field(tmp_path/'source') as field:result=code._export_source(field,out,config,True)
    assert result['status']=='PARTIAL' and result['independent_reprojection'] is True
    assert before=={p:(file_sha(p),p.stat().st_mtime_ns) for p in before}


def test_mixed_level_publication_and_verification_remain_partial(tmp_path,engine,monkeypatch):
    from winding_field import tifxyz as code,tifxyz_audit as audit
    source=physical_source(tmp_path/'source',shape=(4,3,3));out=tmp_path/'out'
    original=audit._coverage
    def coverage(field,store,*args,**kwargs):
        if store.grid['level']==1.:
            raise audit.CoverageFailure('SOURCE_UNCOVERED',(1,0,0),(0,0,0),0,field)
        return original(field,store,*args,**kwargs)
    monkeypatch.setattr(audit,'_coverage',coverage)
    with open_physical_field(source) as field:
        config=settings(field,levels=[0.,1.],coverage_policy='allow_partial')
        assert code._export_source(field,out,config)['status']=='PARTIAL'
    receipt=json.loads((out/'TIFXYZ.json').read_text())
    assert [x['status'] for x in receipt['levels']]==['READY','PARTIAL']
    assert not (out/'READY.json').exists()
    with open_physical_field(source) as field:assert code._verify_source(field,out)['status']=='TIFXYZ_PARTIAL_VERIFIED'


@pytest.mark.parametrize('damage',['geometry','phase','scan','remaining','region','resource'])
def test_partial_gate_cannot_accept_inconsistent_audit(tmp_path,engine,damage):
    from winding_field.tifxyz import _audit_status
    _,out,_,result=partial_plane(tmp_path)
    audit=json.loads((out/'level-0000/VERIFIED.json').read_text())['audit']
    if damage=='geometry':audit['geometry_complete']=False
    elif damage=='phase':audit['blocked_phase']='geometry'
    elif damage=='scan':audit['source_scan_complete']=True
    elif damage=='remaining':audit['remaining_source_coverage']='NONE'
    elif damage=='region':audit['reasons'][0]['region']['global_cell_zyx'][0]+=1
    else:audit['reasons'].append(dict(code='MEMORY_BUDGET'))
    with pytest.raises((Blocked,ValueError)):_audit_status(audit,'allow_partial')
