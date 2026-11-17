import json
import sqlite3

import pytest

from test_afv_batching import many
from test_local_import_performance import local_case
from winding_field import afv,lasagna_input,workflow
from winding_field.util import file_sha


def test_standalone_import_reuses_preflight_but_keeps_final_source_validation(tmp_path,monkeypatch):
    config=many(tmp_path/'run',4);calls=[];original=afv.source_identity
    def identity(*args,**kwargs):
        calls.append(1);return original(*args,**kwargs)
    monkeypatch.setattr(afv,'source_identity',identity)
    assert afv.import_fibres(tmp_path/'run',config)['status']=='FIBRES_READY'
    assert len(calls)==2


def test_workflow_reuses_only_current_invocation_afvs_and_lasagna_preflight(tmp_path,monkeypatch):
    config=json.loads(local_case(tmp_path/'input').read_bytes());config['output']=str(tmp_path/'run')
    calls=dict(afv=0,lasagna=0)
    for module,name in ((afv,'afv'),(lasagna_input,'lasagna')):
        original=module.source_identity
        def identity(*args,_name=name,_original=original,**kwargs):
            calls[_name]+=1;return _original(*args,**kwargs)
        monkeypatch.setattr(module,'source_identity',identity)
    workflow.run(config,until='fibres')
    assert calls==dict(afv=3,lasagna=2)


def test_source_mutation_after_checked_preflight_still_rejects_completion(tmp_path,monkeypatch):
    config=many(tmp_path/'run',4)
    source=afv.source_identity(config['fibres'],config.get('fibres_geometry'),
                               json.loads((tmp_path/'run/PREPARED.json').read_bytes())['source'])
    original=afv._pieces;changed=False
    def pieces(*args,**kwargs):
        nonlocal changed
        yield from original(*args,**kwargs)
        if not changed:
            changed=True
            with sqlite3.connect(config['fibres']) as db:db.execute("UPDATE fibers SET family='changed' WHERE id=1")
    monkeypatch.setattr(afv,'_pieces',pieces)
    with pytest.raises(ValueError,match='changed'):
        afv.import_fibres(tmp_path/'run',config,verified_source=source)
    assert not (tmp_path/'run/FIBRES.json').exists()


def test_completed_resume_revalidates_source_after_plan_without_rewriting_products(tmp_path,monkeypatch):
    config=json.loads(local_case(tmp_path/'input').read_bytes());root=tmp_path/'run';config['output']=str(root)
    workflow.run(config,until='fibres')
    paths=[root/name for name in ('PREPARED.json','FIBRES.json','LASAGNA.json')]
    before={p:(file_sha(p),p.stat().st_mtime_ns) for p in paths}
    plan=workflow.plan
    def changed(*args,**kwargs):
        value=plan(*args,**kwargs)
        with sqlite3.connect(config['fibres']) as db:db.execute("UPDATE fibers SET family='changed' WHERE id=1")
        return value
    monkeypatch.setattr(workflow,'plan',changed)
    with pytest.raises(ValueError,match='changed'):workflow.run(config,resume=True,until='fibres')
    assert before=={p:(file_sha(p),p.stat().st_mtime_ns) for p in paths}


def test_completed_standalone_import_cannot_trust_a_stale_checked_result(tmp_path):
    config=many(tmp_path/'run',2)
    source=afv.source_identity(config['fibres'],config.get('fibres_geometry'),
                               json.loads((tmp_path/'run/PREPARED.json').read_bytes())['source'])
    afv.import_fibres(tmp_path/'run',config,verified_source=source)
    with sqlite3.connect(config['fibres']) as db:db.execute("UPDATE fibers SET family='changed' WHERE id=1")
    with pytest.raises(ValueError,match='changed'):
        afv.import_fibres(tmp_path/'run',config,verified_source=source)


def test_checked_results_cannot_be_used_for_a_different_input_path(tmp_path):
    config=many(tmp_path/'run',2)
    source=afv.source_identity(config['fibres'],config.get('fibres_geometry'),
                               json.loads((tmp_path/'run/PREPARED.json').read_bytes())['source'])
    with pytest.raises(ValueError,match='context'):
        afv.import_fibres(tmp_path/'run',config,verified_source=dict(source,path='different.afv'))
    assert not (tmp_path/'run/fibres/IMPORT.json').exists()
