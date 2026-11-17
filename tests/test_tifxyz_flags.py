import argparse
import json
from pathlib import Path

import pytest

from winding_field.cli import main
from winding_field.tifxyz_contract import load_settings,validate_request
from test_prepared_flags import pins
from test_tifxyz_cli import job as _job
from tifxyz_fixture import field_run

job=_job


def request_flags():
    return ['--levels','1','2','--roi-lo-zyx','0','0','0','--roi-hi-zyx','3','4','5',
        '--pitch-um','9.6','--tolerance-um','19.2','--memory-gib','3','--scratch-gib','16',
        '--reserve-disk-gib','0','--max-pixels','4096','--audit-backend','cpu','--coverage-policy','allow_partial',
        '--output-frame-id','explicit','--output-origin-xyz-um','10','20','30','--output-voxel-um','9.6',
        '--output-shape-zyx','10','11','12','--output-source-identity-sha256','a'*64]


def resolve(tokens):
    from winding_field.export_settings import tifxyz_arguments,tifxyz_request
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path)
    tifxyz_arguments(parser)
    return tifxyz_request(parser.parse_args(tokens))


def test_maximal_direct_request_matches_native_file_and_real_unit_resolution(tmp_path):
    request=dict(levels=[1.,2.],roi=dict(lo_zyx=[0,0,0],hi_zyx=[3,4,5]),pitch_um=9.6,tolerance_um=19.2,
        memory_gib=3.,scratch_gib=16.,reserve_disk_gib=0.,max_pixels=4096,audit_backend='cpu',
        coverage_policy='allow_partial',output_frame=dict(id='explicit',origin_xyz_um=[10.,20.,30.],
            voxel_um=9.6,shape_zyx=[10,11,12],source_identity_sha256='a'*64))
    path=tmp_path/'request.json';path.write_text(json.dumps(request))
    flags=resolve(request_flags());native=resolve(['--config',str(path)])
    assert flags==native==validate_request(request)
    geometry=dict(shape_zyx=[3,4,5],spacing_xyz=[2.,3.,5.],voxel_um=9.6)
    assert load_settings(flags,geometry,request['output_frame'])==load_settings(native,geometry,request['output_frame'])
    settings=load_settings(flags,geometry,request['output_frame'])
    assert settings.pitch_um==9.6 and settings.tolerance_um==19.2 and settings.reserve_disk_bytes==0


def test_zero_reserve_false_and_selected_source_survive_merge(tmp_path):
    path=tmp_path/'request.json'
    request=resolve(request_flags());path.write_text(json.dumps(request))
    result=resolve(['--config',str(path),'--pitch-um','auto','--tolerance-um','auto',
                    '--reserve-disk-gib','0','--output-frame','selected-source','--whole-field'])
    assert result['pitch_um'] is None and result['tolerance_um'] is None and result['roi'] is None
    assert result['reserve_disk_gib']==0 and result['output_frame']=='selected-source'
    assert result['levels']==[1.,2.] and result['coverage_policy']=='allow_partial'


def test_run_plan_need_no_tifxyz_or_flattening_config(tmp_path,capsys):
    from winding_field.demo import synthetic
    source=synthetic(tmp_path/'ct.zarr')
    common=['--source',str(source),'--prediction','classical','--device','cpu','--threads','1',
            '--reserve-gib','0','--cache',str(tmp_path/'cache')]
    assert main(['plan',*common,'--output',str(tmp_path/'direct'),'--surface-output','tifxyz',
        '--levels','1','--tifxyz-reserve-disk-gib','0','--tifxyz-max-pixels','4096'])==0
    result=json.loads(capsys.readouterr().out)
    assert result['stages']==['prepare','fibres','field','tifxyz'] and result['tifxyz_reserve_disk_bytes']==0
    assert main(['run',*common,'--output',str(tmp_path/'original'),'--flattening-method','original',
        '--output-frame-id','example','--output-origin-xyz-um','10','20','30','--output-voxel-um','2',
        '--pitch','auto','--original-memory-gib','5','--until','prepare'])==0
    config=json.loads((tmp_path/'original/RUN.json').read_bytes())['config']
    assert config['original_request']==dict(output_frame=dict(id='example',origin_xyz_um=[10.,20.,30.],voxel_um=2.),
                                           pitch=None,memory_gib=5.)
    assert not (tmp_path/'direct').exists()


def test_resume_accepts_equal_flags_but_rejects_changed_request(tmp_path,job):
    source=field_run(tmp_path/'producer');out=tmp_path/'export'
    common=['tifxyz',str(source),'--output',str(out)]
    assert main([*common,'--levels','1','--reserve-disk-gib','0','--max-pixels','4096'])==0
    assert main([*common,'--resume','--levels','1','--reserve-disk-gib','0','--max-pixels','4096'])==0
    before=pins(tmp_path)
    assert main([*common,'--resume','--levels','1','--pitch-um','2'])==1
    assert before==pins(tmp_path)


def test_verify_tifxyz_remains_read_only(tmp_path,job):
    source=field_run(tmp_path/'producer');out=tmp_path/'export'
    assert main(['tifxyz',str(source),'--output',str(out),'--levels','1',
                 '--reserve-disk-gib','0','--max-pixels','4096'])==0
    before=pins(tmp_path)
    assert main(['verify-tifxyz',str(out)])==0
    assert before==pins(tmp_path)


def test_tifxyz_options_reject_non_direct_run_branch(tmp_path):
    assert main(['plan','--source','absent','--output',str(tmp_path/'bad'),
                 '--tifxyz-pitch-um','2','--levels','1'])==1
    assert not (tmp_path/'bad').exists()


def test_producer_false_values_override_file_true_values(tmp_path):
    from winding_field.demo import synthetic
    source=synthetic(tmp_path/'ct.zarr')
    options=dict(source=str(source),output=str(tmp_path/'run'),prediction='classical',device='cpu',
        threads=1,reserve_gib=0,cache=str(tmp_path/'cache'),previews=True,export_tifxyz=True,
        surface_output='tifxyz',levels=[1.],tifxyz_reserve_disk_gib=0,tifxyz_max_pixels=4096,
        until='prepare')
    path=tmp_path/'options.json';path.write_text(json.dumps(options))
    assert main(['run','--cli-config',str(path),'--no-previews','--no-export-tifxyz'])==0
    config=json.loads((tmp_path/'run/RUN.json').read_bytes())['config']
    assert config['previews'] is False and config['export_tifxyz'] is False


def test_original_branch_does_not_silently_ignore_direct_frame_only_flags(tmp_path,monkeypatch):
    monkeypatch.setattr('winding_field.cli.resolve_source',lambda *a,**kw:pytest.fail('source IO before wrong-mode rejection'))
    assert main(['plan','--source','missing','--output',str(tmp_path/'bad'),'--flattening-method','original',
        '--output-frame-id','frame','--output-origin-xyz-um','0','0','0','--output-voxel-um','1',
        '--output-shape-zyx','3','4','5'])==1
    assert not (tmp_path/'bad').exists()


def test_nested_levels_override_remains_distinct_from_producer_level_check(tmp_path,capsys):
    from winding_field.demo import synthetic
    source=synthetic(tmp_path/'ct.zarr')
    path=tmp_path/'request.json';path.write_text('{"levels": [1, 2]}')
    assert main(['plan','--source',str(source),'--output',str(tmp_path/'direct'),
        '--prediction','classical','--surface-output','tifxyz','--tifxyz-config',str(path),
        '--tifxyz-levels','3','--levels','3','--tifxyz-reserve-disk-gib','0'])==0
    assert json.loads(capsys.readouterr().out)['tifxyz_reserve_disk_bytes']==0
