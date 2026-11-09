"""The README's saved-run export needs no hand-maintained configuration."""
import json
from pathlib import Path

import numpy as np
import pytest

from winding_field.cli import main
from winding_field.util import file_sha


@pytest.fixture
def saved_run(tmp_path,request):
    from ct_fixture import write_ct
    from winding_field.workflow import run
    x=np.indices((8,8,24))[2]
    ct=write_ct(tmp_path/'ct',(20+210*(np.cos(2*np.pi*x/8)>.4)).astype('u1'),
                spacing=(2.,3.,4.),origin=(10.,20.,30.),chunk=4)
    level=getattr(request,'param',0)
    if level:
        (ct/'0').rename(ct/str(level))
        attrs=json.loads((ct/'.zattrs').read_bytes())
        attrs['multiscales'][0]['datasets'][0]['path']=str(level)
        (ct/'.zattrs').write_text(json.dumps(attrs))
    root=tmp_path/'run'
    run(dict(source=str(ct),output=str(root),prediction='classical',threads=1,level=level,
             chunk=8,field_chunk=8,sweeps=1,seed_stride=4,previews=False,
             reserve_bytes=0,cache=str(tmp_path/'cache')),until='field')
    return root


@pytest.fixture
def small_raster(monkeypatch):
    pytest.importorskip('igl')
    from winding_field import original_pipeline as pipeline
    raster=pipeline.raster_sheet
    def small(v,f,uv,eligible,sample,canvas,**kwargs):
        canvas.update(width=32,height=32,pitch=canvas['pitch']*4095/31)
        return raster(v,f,uv,eligible,sample,canvas,**kwargs)
    monkeypatch.setattr(pipeline,'raster_sheet',small)
    monkeypatch.setattr(pipeline,'available_memory',lambda:16<<30)


def command(root,out,*extra):
    return ['export-sheets','--run',str(root),'--levels','.5','--export-tifxyz','--output',str(out),*extra]


def test_saved_run_exports_without_configs_and_resume_preserves_source(saved_run,tmp_path,small_raster):
    before={str(p.relative_to(saved_run)):file_sha(p) for p in saved_run.rglob('*') if p.is_file()}
    out=tmp_path/'export'
    assert main(command(saved_run,out))==0
    field=json.loads((out/'field.json').read_bytes())
    settings=json.loads((out/'flattening.json').read_bytes())
    assert field['field']['path']==str(saved_run/'field/CHECKPOINT.json')
    assert settings['output_frame']==dict(id=field['geometry']['frame'],voxel_um=1.,origin_xyz_um=[0.,0.,0.])
    level=out/'original/level-0000'
    receipt=json.loads((level/'LEVEL.json').read_bytes())
    assert receipt['tifxyz']['retained_cells']>0
    assert all((level/'tifxyz'/name).is_file() for name in ('x.tif','y.tif','z.tif'))
    generated={p.name:file_sha(p) for p in (out/'field.json',out/'flattening.json')}
    assert main(command(saved_run,out,'--resume'))==0
    assert generated=={p.name:file_sha(p) for p in (out/'field.json',out/'flattening.json')}
    assert before=={str(p.relative_to(saved_run)):file_sha(p) for p in saved_run.rglob('*') if p.is_file()}


@pytest.mark.parametrize('mutation',['field','checkpoint','geometry','ct-metadata','ct-chunk','missing-stage','missing-ledger'])
def test_invalid_saved_inputs_fail_before_output(saved_run,tmp_path,mutation,capsys):
    from winding_field.util import atomic_json
    state_path=saved_run/'RUN.json';state=json.loads(state_path.read_bytes())
    field_path=saved_run/'FIELD.json'
    if mutation=='field':field_path.write_bytes(field_path.read_bytes()+b' ')
    elif mutation=='checkpoint':
        path=saved_run/'field/CHECKPOINT.json';path.write_bytes(path.read_bytes()+b' ')
    elif mutation=='geometry':
        field=json.loads(field_path.read_bytes());field['geometry']['spacing_xyz'][0]*=2
        atomic_json(field_path,field);state['stages']['field']['sha256']=file_sha(field_path);atomic_json(state_path,state)
    elif mutation=='ct-metadata':
        path=tmp_path/'ct/.zattrs';attrs=json.loads(path.read_bytes());attrs['changed']=True;atomic_json(path,attrs)
    elif mutation=='ct-chunk':
        path=tmp_path/'ct/0/0.0.0';path.write_bytes(bytes(4**3))
    elif mutation=='missing-stage':
        state['stages'].pop('field');atomic_json(state_path,state)
    else:(saved_run/'source.sqlite').unlink()
    out=tmp_path/'export'
    assert main(command(saved_run,out))==1
    assert capsys.readouterr().err
    assert not out.exists()


@pytest.mark.parametrize('extra',[['--source','other'],['--level','0'],['--voxel-um','2'],
                                 ['--flattening-method','patches'],['--flattening-solver','lscm','--scope','sheet']])
def test_saved_run_rejects_conflicting_selectors_before_io(tmp_path,extra):
    assert main(command(tmp_path/'absent',tmp_path/'export',*extra))==1
    assert not (tmp_path/'export').exists()


@pytest.mark.parametrize('relative',['.','nested','..'])
def test_source_output_overlap_is_rejected_before_io(tmp_path,relative):
    root=tmp_path/'source'
    assert main(command(root,root/relative))==1
    assert not root.exists()


@pytest.mark.parametrize('name',['field.json','flattening.json'])
def test_edited_generated_configuration_is_never_overwritten(saved_run,tmp_path,small_raster,name):
    out=tmp_path/'export'
    assert main(command(saved_run,out))==0
    path=out/name;path.write_bytes(path.read_bytes()+b' ');edited=path.read_bytes()
    assert main(command(saved_run,out,'--resume'))==1
    assert path.read_bytes()==edited


def test_run_export_matches_equivalent_explicit_configuration(saved_run,tmp_path,small_raster):
    from PIL import Image
    out=tmp_path/'run-export';explicit=tmp_path/'explicit'
    assert main(command(saved_run,out))==0
    state=json.loads((saved_run/'RUN.json').read_bytes())
    assert main(['export-sheets','--field-config',str(out/'field.json'),'--levels','.5',
                 '--source',state['config']['source'],'--flattening-config',str(out/'flattening.json'),
                 '--flattening-method','original','--export-tifxyz','--output',str(explicit)])==0
    for name in ('image.png','mask.png','tifxyz/x.tif','tifxyz/y.tif','tifxyz/z.tif'):
        with Image.open(out/'original/level-0000'/name) as a,Image.open(explicit/'original/level-0000'/name) as b:
            np.testing.assert_array_equal(np.asarray(a),np.asarray(b))


@pytest.mark.parametrize('record,key',[('RUN.json','config'),('RUN.json','source'),('RUN.json','stages'),
                                    ('FIELD.json','geometry'),('FIELD.json','checkpoint_sha256')])
def test_missing_metadata_has_a_clear_error_without_output(saved_run,tmp_path,record,key,capsys):
    from winding_field.util import atomic_json
    path=saved_run/record;value=json.loads(path.read_bytes());value.pop(key);atomic_json(path,value)
    if record=='FIELD.json':
        state=json.loads((saved_run/'RUN.json').read_bytes())
        state['stages']['field']['sha256']=file_sha(path);atomic_json(saved_run/'RUN.json',state)
    assert main(command(saved_run,tmp_path/'export'))==1
    assert 'winding-field:' in capsys.readouterr().err
    assert not (tmp_path/'export').exists()


@pytest.mark.parametrize('saved_run',[1],indirect=True)
def test_saved_ct_pyramid_level_is_reused(saved_run,tmp_path,small_raster):
    out=tmp_path/'export'
    assert main(command(saved_run,out))==0
    receipt=json.loads((out/'EXPORT.json').read_bytes())
    assert receipt['level']==1 and receipt['run']['source']['level']==1


def test_explicit_output_frame_override_is_retained(saved_run,tmp_path,small_raster):
    from winding_field.util import atomic_json
    config=tmp_path/'coordinates.json'
    frame=dict(id='explicit',voxel_um=2.,origin_xyz_um=[10.,20.,30.])
    atomic_json(config,dict(output_frame=frame,memory_gib=5.))
    out=tmp_path/'export'
    assert main(command(saved_run,out,'--flattening-config',str(config)))==0
    assert json.loads((out/'flattening.json').read_bytes())['output_frame']==frame
    assert json.loads((out/'flattening.json').read_bytes())['memory_gib']==5.


def test_source_mutation_after_admission_cannot_enter_rendering(saved_run,tmp_path,small_raster,monkeypatch,capsys):
    from winding_field import surface_mesh
    extract=surface_mesh.extract_mesh
    def changed_after_extract(*args,**kwargs):
        result=extract(*args,**kwargs)
        for path in (tmp_path/'ct/0').iterdir():
            if not path.name.startswith('.'):path.write_bytes(bytes(4**3))
        return result
    monkeypatch.setattr(surface_mesh,'extract_mesh',changed_after_extract)
    out=tmp_path/'export'
    assert main(command(saved_run,out))==1
    assert 'chunk changed' in capsys.readouterr().err
    assert not (out/'original/level-0000/LEVEL.json').exists()


@pytest.mark.parametrize('mutation',['missing','removed-row','symlink'])
def test_producer_hashes_are_required_in_export_ledger_on_resume(saved_run,tmp_path,small_raster,mutation):
    import sqlite3
    out=tmp_path/'export'
    assert main(command(saved_run,out))==0
    ledger=out/'source.sqlite'
    if mutation=='removed-row':
        with sqlite3.connect(ledger) as db:db.execute('DELETE FROM chunks WHERE key=(SELECT key FROM chunks LIMIT 1)')
    else:
        ledger.unlink()
        if mutation=='symlink':ledger.symlink_to(saved_run/'source.sqlite')
    before=file_sha(saved_run/'source.sqlite')
    assert main(command(saved_run,out,'--resume'))==1
    assert file_sha(saved_run/'source.sqlite')==before
