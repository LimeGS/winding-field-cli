"""Small synthetic diagnostics: no extra solves, verifications or acceptance."""
from copy import deepcopy
import hashlib
import itertools
import json

import numpy as np
import pytest

from field_store import BlockStore
from winding_field import field_advice as advice
from winding_field.util import file_sha


GEOMETRY=dict(frame='synthetic-um',voxel_um=1.,sheet_step=1.,
              shape_zyx=[7,7,9],origin_xyz=[0.,0.,0.],spacing_xyz=[2.,3.,5.])


def event(root,sweep=1):
    manifest=json.loads((root/'CHECKPOINT.json').read_bytes())
    return dict(checkpoint_path=str((root/'CHECKPOINT.json').resolve()),
                checkpoint_sha256=file_sha(root/'CHECKPOINT.json'),scientific_sha256='a'*64,
                manifest=manifest,metrics=dict(schema='field-observer-v1',scientific_sha256='a'*64,
                sweep=sweep,next_block=12,complete=True,timing_complete=True,
                observed_count=440,observed_sum_sq=4.,observed_max_update=2.,
                observed_rms_update=np.sqrt(4/440),work_s=1.,control_s=.01,active_wall_s=1.1,
                candidates=[dict(node=220,rank='a')]))


def stored(root,*,shift=0.,support=True):
    shape=GEOMETRY['shape_zyx'];chunk=4
    z,y,x=np.indices(shape);phi=x*2.+y*6.+z*15.+shift
    identity=dict(geometry=GEOMETRY,boundary='natural-zero-gauge')
    with BlockStore(root,shape,chunk,identity,resume=root.exists(),reserve_bytes=0) as store:
        for key in itertools.product(range(2),range(2),range(3)):
            lo=np.array(key)*chunk;hi=np.minimum(lo+chunk,shape)
            region=tuple(slice(a,b) for a,b in zip(lo,hi))
            mask=np.full(tuple(hi-lo),support,bool)
            store.commit(key,phi[region],dict(sweep=1,next_block=12),mask)


def test_journal_selects_only_the_exact_scientific_side_of_a_crash(tmp_path):
    root=tmp_path/'field';stored(root);snapshot=event(root)
    sink=advice.Advice(root,GEOMETRY);assert sink.restore(snapshot) is None
    state=deepcopy(snapshot['metrics'])
    first=dict(snapshot,previous_checkpoint_sha256=snapshot['checkpoint_sha256'],
               checkpoint_sha256='b'*64,scientific_bytes=1000,reserve_bytes=0)
    sink.prepare(first,state)
    assert advice.Advice(root,GEOMETRY).restore(snapshot) is None
    committed=dict(snapshot,checkpoint_sha256='b'*64)
    assert advice.Advice(root,GEOMETRY).restore(committed)==state
    second=dict(first,previous_checkpoint_sha256='b'*64,checkpoint_sha256='c'*64)
    state2=dict(state,next_block=13)
    sink.prepare(second,state2)
    assert advice.Advice(root,GEOMETRY).restore(committed)==state
    assert advice.Advice(root,GEOMETRY).restore(dict(snapshot,checkpoint_sha256='c'*64))==state2
    assert sink.path.parent==root.parent and root not in sink.path.parents


def test_quota_sink_and_identity_failures_preserve_previous_diagnostics(tmp_path,monkeypatch):
    root=tmp_path/'field';stored(root);snapshot=event(root)
    sink=advice.Advice(root,GEOMETRY);sink.restore(snapshot)
    proposal=dict(snapshot,previous_checkpoint_sha256=snapshot['checkpoint_sha256'],
                  scientific_bytes=1000,reserve_bytes=0)
    sink.prepare(proposal,snapshot['metrics'])
    before={p.name:p.read_bytes() for p in sink.path.iterdir()}
    monkeypatch.setattr(advice,'QUOTA_BYTES',sum(map(len,before.values()))+1)
    with pytest.raises(OSError):sink.prepare(dict(proposal,checkpoint_sha256='b'*64),snapshot['metrics'])
    assert before=={p.name:p.read_bytes() for p in sink.path.iterdir()}
    monkeypatch.setattr(advice,'QUOTA_BYTES',4<<20)
    def fail(*args,**kwargs):raise OSError('diagnostic sink unavailable')
    monkeypatch.setattr(advice,'_publish',fail)
    with pytest.raises(OSError):sink.prepare(dict(proposal,checkpoint_sha256='b'*64),snapshot['metrics'])
    assert before=={p.name:p.read_bytes() for p in sink.path.iterdir()}
    with pytest.raises(ValueError):advice.Advice(root,GEOMETRY).restore(dict(snapshot,scientific_sha256='d'*64))


def test_samples_are_frozen_grouped_and_anisotropic_with_null_unsupported_estimates(tmp_path):
    root=tmp_path/'field';stored(root)
    sink=advice.Advice(root,GEOMETRY);snapshot=event(root)
    sink.restore(snapshot);sink.complete(snapshot)
    baseline=sink.records()[0];plan=baseline['plan']
    assert 2<=len(plan['points'])<=128
    assert any('interface' in p['reasons'] for p in plan['points'])
    assert any('observed' in p['reasons'] for p in plan['points'])
    assert baseline['io']['payload_reads']<=64
    assert baseline['io']['decoded_bytes']>0
    centre=next(s for s in baseline['samples'] if s['gradient_h'] is not None and s['gradient_2h'] is not None)
    np.testing.assert_allclose(centre['gradient_h'],[1.,2.,3.])
    np.testing.assert_allclose(centre['gradient_2h'],[1.,2.,3.])
    sink.complete(snapshot)
    assert len(sink.records())==1
    stored(root,shift=1.,support=False);snapshot2=event(root,2)
    sink.complete(snapshot2);second=sink.records()[-1]
    assert second['plan']==plan
    assert all(s['gradient_h'] is None for s in second['samples'])
    comparison=advice.compare(baseline,second)
    assert comparison['common_observed_count']==0
    assert all(s['normal_displacement'] is None for s in comparison['points'])
    assert all(s['raw_delta_phi']==pytest.approx(1.) for s in comparison['points'])
    assert all(s['relative_delta_phi']==pytest.approx(0.) for s in comparison['points'])


def history(changes=(4.,2.),error=0.):
    result=[]
    for sweep,value in enumerate((0.,changes[0],sum(changes)),1):
        result.append(dict(sweep=sweep,checkpoint_sha256=str(sweep)*64,scientific_sha256='a'*64,
            complete=True,plan_sha256='p',gauge='zero-origin',reference=dict(phi=0.,error=0.),
            plan=dict(required_ids=[1,2],multiblock=True,
                      points=[dict(node=1,reasons=['observed'],stratum='a'),
                              dict(node=2,reasons=['observed'],stratum='b'),
                              dict(node=3,reasons=['interface'],stratum='a')]),
            samples=[dict(node=i,phi=value,error=error,observed=True,gradient_h=None,gradient_2h=None,
                          gradient_error=None,radius_um=1.) for i in (1,2,3)],
            metrics=dict(complete=True,timing_complete=True,active_wall_s=1.)))
    return result


@pytest.mark.parametrize('changes,error,available,verification,want',[
    ((4.,2.),0.,True,[],'SEGUIR_CALCULANDO'),
    ((2.,4.),0.,True,[],'POSIBLE_ESTANCAMIENTO'),
    ((2.,2.),0.,True,[],'DATOS_INSUFICIENTES'),  # float64 resolution overlaps
    ((0.,0.),0.,True,[],'CONVIENE_VERIFICAR'),
    ((0.,0.),0.,False,[],'POSIBLE_ESTANCAMIENTO'),
    ((.1,.1),1.,True,[],'CONVIENE_VERIFICAR'),
    ((4.,2.),0.,True,[dict(checkpoint_sha256='3'*64,scientific_sha256='a'*64,
                          global_converged=True,residual_l2=0.)],'DATOS_INSUFICIENTES'),
    ((0.,0.),0.,True,[dict(checkpoint_sha256='3'*64,scientific_sha256='a'*64,
                          global_converged=False,residual_l2=2.)],'POSIBLE_ESTANCAMIENTO'),
    ((4.,2.),0.,True,[dict(checkpoint_sha256='2'*64,scientific_sha256='a'*64,
                          global_converged=False,residual_l2=1.),
                     dict(checkpoint_sha256='3'*64,scientific_sha256='a'*64,
                          global_converged=False,residual_l2=2.)],'POSIBLE_ESTANCAMIENTO'),
])
def test_recommendation_precedence_without_physical_thresholds(changes,error,available,verification,want):
    result=advice.recommend(history(changes,error),verification,available)
    assert result['recommendation']==want
    assert result['reason'] and result['actions']==[]
    assert result['physical_usefulness']=='precisión útil no calibrada'
    assert result['compared_checkpoints']==['1'*64,'2'*64,'3'*64]


@pytest.mark.parametrize('breakage',['gap','identity','plan','gauge','support','incomplete','interface','one'])
def test_missing_evidence_never_shrinks_required_denominator(breakage):
    records=history()
    if breakage=='gap':records[1]['sweep']=8
    if breakage=='identity':records[1]['scientific_sha256']='b'*64
    if breakage=='plan':records[1]['plan_sha256']='q'
    if breakage=='gauge':records[1]['gauge']=None
    if breakage=='support':records[1]['samples'][0]['observed']=False
    if breakage=='incomplete':records[1]['metrics']['complete']=False
    if breakage=='interface':records[1]['samples'][2]['phi']=None
    if breakage=='one':
        for record in records:record['plan']['required_ids']=[1]
    assert advice.recommend(records,[],True)['recommendation']=='DATOS_INSUFICIENTES'


def test_real_sample_corruption_is_not_a_telemetry_failure(tmp_path):
    from block_solver import AdvisoryIntegrityError
    root=tmp_path/'field';stored(root);snapshot=event(root)
    sink=advice.Advice(root,GEOMETRY);sink.restore(snapshot)
    manifest=snapshot['manifest']
    payload=root/next(iter(manifest['chunks'].values()))['path']
    payload.write_bytes(b'corrupt')
    with pytest.raises(AdvisoryIntegrityError):sink.complete(snapshot)


def test_incomplete_first_sweep_does_not_freeze_a_partial_sampling_baseline(tmp_path):
    root=tmp_path/'field';stored(root);snapshot=event(root)
    sink=advice.Advice(root,GEOMETRY);sink.restore(snapshot)
    snapshot['metrics']['complete']=False
    sink.complete(snapshot)
    assert not (sink.path/'PLAN.json').exists()
    assert sink.records()[0]['complete'] is False
    snapshot['metrics']['complete']=True;snapshot['metrics']['sweep']=2
    stored(root,shift=1.);snapshot['checkpoint_sha256']=file_sha(root/'CHECKPOINT.json')
    snapshot['manifest']=json.loads((root/'CHECKPOINT.json').read_bytes())
    sink.complete(snapshot)
    assert (sink.path/'PLAN.json').is_file()
    assert advice.recommend(sink.records(),[],False)['recommendation']=='DATOS_INSUFICIENTES'


def test_plan_is_deterministic_and_limits_actual_neighbour_chunks():
    geometry=dict(GEOMETRY,shape_zyx=[1001,1001,1001])
    snapshot=dict(manifest=dict(chunk=4),metrics=dict(candidates=[
        dict(node=i*1001001+10,rank=str(i).zfill(3)) for i in range(32)]))
    plan=advice.make_plan(snapshot,geometry)
    assert plan==advice.make_plan(snapshot,geometry)
    touched={(0,0,0)}
    for point in plan['points']:
        touched.add(tuple(np.asarray(point['zyx'])//4))
        touched.update(tuple(np.asarray(n)//4) for n in point['stencil'])
    assert len(plan['points'])<=128 and len(touched)<=64


def test_incompatible_closure_history_is_preserved_and_rejected(tmp_path):
    root=tmp_path/'field';stored(root);snapshot=event(root)
    sink=advice.Advice(root,GEOMETRY);sink.restore(snapshot);sink.complete(snapshot)
    record=sink.records()[0];record['scientific_sha256']='b'*64
    path=next(sink.path.glob('closure-*.json'))
    advice._publish(path,dict(data=record,sha256=advice.digest(record)))
    before=path.read_bytes()
    with pytest.raises(ValueError):advice.Advice(root,GEOMETRY).restore(snapshot)
    assert path.read_bytes()==before


@pytest.mark.parametrize('gradient,error,radius,reason',[
    ([1.,2.,3.],0.,1.,None),
    ([0.,0.,0.],0.,1.,'insufficient resolved gradient'),
    ([1.,0.,0.],2.,1.,'insufficient resolved gradient'),
    ([1.,0.,0.],0.,.01,'outside local stencil radius'),
])
def test_local_shift_requires_a_resolved_gradient_and_local_domain(gradient,error,radius,reason):
    a,b=history((.1,.1))[:2]
    for record in (a,b):
        for row in record['samples']:
            row.update(gradient_h=gradient,gradient_2h=gradient,gradient_error=error,radius_um=radius)
    rows=advice.compare(a,b)['points']
    assert all(row['displacement_exclusion']==reason for row in rows)
    if reason is None:
        assert rows[0]['normal_displacement']==pytest.approx(.1/np.sqrt(14))
    else:assert all(row['normal_displacement'] is None for row in rows)
