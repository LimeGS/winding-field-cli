import json
from pathlib import Path
import sys

import numpy as np
import pytest

from winding_field.cli import main,TOOLS
from winding_field.util import file_sha,atomic_json
from test_prepared_flags import pins
from test_surface_dataset import surface_case as _surface_case,native_fixture
from test_ct_visuals import visual_case as _visual_case

surface_case=_surface_case
visual_case=_visual_case


def geometry_flags(meta,prefix='field-'):
    result=['--'+prefix+'frame',meta['frame'],'--'+prefix+'voxel-um',str(meta['voxel_um'])]
    for name in ('origin_xyz','spacing_xyz','shape_zyx'):
        if name in meta:result+=['--'+prefix+name.replace('_','-'),*map(str,meta[name])]
    if prefix=='field-':
        if 'field_sha256' in meta:result+=['--field-sha256',meta['field_sha256']]
        if 'step' in meta:result+=['--sheet-step',str(meta['step'])]
    return result


def dataset_flags(case):
    dataset=case['dataset']
    result=['--dataset-frame',dataset['frame'],'--dataset-voxel-um',str(dataset['voxel_um'])]
    for source,prefix in (('ct','ct-'),('surface','prior-')):
        entry=dataset['sources'][source]
        result+=['--'+prefix+'source',entry['path'],'--'+prefix+'frame',entry['frame']]
        for name in ('origin_xyz','spacing_xyz'):result+=['--'+prefix+name.replace('_','-'),*map(str,entry[name])]
    result+=['--seed-margin','1','--ct-padding-samples','0','--render-pitch-world','2',
             '--render-depth-offsets-world','0','--render-mesh-step','1','--ct-window','0','3000']
    return result


def test_surface_export_flags_match_real_ct_and_xyz_without_metadata_json(tmp_path):
    shape=(5,6,7);field=np.indices(shape)[0].astype(float)
    path=tmp_path/'field.npy';ct=tmp_path/'ct.npy';np.save(path,field);np.save(ct,field*10)
    meta=dict(frame='test/world',voxel_um=9.6,origin_xyz=[10.,20.,30.],
              spacing_xyz=[2.,3.,4.],shape_zyx=list(shape))
    native=tmp_path/'meta.json';native.write_text(json.dumps(meta))
    assert main(['surface-export','--field',str(path),'--metadata',str(native),'--ct',str(ct),
        '--ct-metadata',str(native),'--level','2.5','--pitch','.5','--output',str(tmp_path/'file')])==0
    native.unlink()
    assert main(['surface-export','--field',str(path),*geometry_flags(meta),'--ct',str(ct),
        *geometry_flags(meta,'ct-'),'--ct-display-window','0','50','--level','2.5',
        '--pitch','.5','--output',str(tmp_path/'flags')])==0
    for name in ('surface.npz','grid.npz','ct_stack.npz'):
        with np.load(tmp_path/'file'/name) as a,np.load(tmp_path/'flags'/name) as b:
            for key in a.files:np.testing.assert_array_equal(a[key],b[key])
    assert json.loads((tmp_path/'flags/receipt.json').read_bytes())['raster']['pitch_um']==[4.8,4.8]


def test_probe_dataset_and_metadata_flags_reach_actual_child(surface_case):
    case=surface_case;root=case['root'];app=native_fixture(root)
    flags=dataset_flags(case)+geometry_flags(case['metadata'])
    case['path'].unlink();(root/'field.npy.json').unlink()
    assert main(['surface-probe','--field',str(root/'field.npy'),'--work',str(root/'probe'),
        '--output',str(root/'export'),'--app-run',str(app),*flags])==0
    result=json.loads((root/'probe/result.json').read_bytes())
    assert result['seed']['seed_xyz']==[16.,26.,38.]
    assert result['export_receipt']['preview']['window']==[0.,3000.]
    with np.load(root/'export/ct_stack.npz') as stack:assert stack['valid'].any()
    assert (root/'probe/field-metadata.json').is_file()


def test_atlas_accepts_inline_settings_with_actual_frozen_seed_and_ct(surface_case):
    from winding_field.surface_probe import prepare_inputs
    case=surface_case;root=case['root'];work=root/'prepared-inputs';work.mkdir()
    _,inputs=prepare_inputs(case['dataset'],case['field'],case['metadata'],work)
    ct_meta=json.loads(Path(inputs['ct_metadata']).read_bytes())
    app=native_fixture(root)
    assert main(['surface-atlas','--field',str(root/'field.npy'),*geometry_flags(case['metadata']),
        '--seed',inputs['seed'],'--seed-sha256',inputs['seed_sha256'],
        '--ct',inputs['ct'],'--ct-sha256',inputs['ct_sha256'],*geometry_flags(ct_meta,'ct-'),
        '--ct-display-window','0','3000','--render-pitch-world','2',
        '--render-depth-offsets-world','0','--render-mesh-step','1',
        '--app-run',str(app),'--output',str(root/'atlas'),'--max-depth','0',
        '--max-charts','1','--budget-seconds','30'])==0
    result=json.loads((root/'atlas/result.json').read_bytes())
    assert result['atlas_status']=='PASS' and result['human_seed'] is False


def test_ct_visuals_need_only_frozen_archive_data_not_authored_case_or_metadata(visual_case):
    case=visual_case;meta=json.loads(case['metadata'].read_bytes())
    descriptor=json.loads(case['case'].read_bytes())
    spec=descriptor['visualization'];archive=spec['archive_manifest']
    case['case'].unlink();case['metadata'].unlink()
    assert main(['ct-visuals','--field',str(case['field']),*geometry_flags(meta),
        '--visualization-schema',spec['schema'],'--archive-manifest',archive['path'],
        '--archive-manifest-sha256',archive['sha256'],'--reference-key','coarse',
        '--reference-label','Frozen reference','--output',str(case['root']/'visual'),'--check-only'])==0
    assert json.loads((case['root']/'visual/PREFLIGHT.json').read_bytes())['status']=='PASS_PREFLIGHT'


def test_patch_visuals_flag_geometry_preserves_physical_residuals(tmp_path):
    import tifffile
    path=tmp_path/'field.npy';np.save(path,np.broadcast_to(np.arange(40.),(3,3,40)))
    patch=tmp_path/'patch';patch.mkdir()
    atomic_json(patch/'meta.json',dict(format='tifxyz',uuid='synthetic'))
    for axis,value in zip('xyz',[np.arange(40.)[None,:],np.ones((1,40)),np.ones((1,40))]):
        tifffile.imwrite(patch/(axis+'.tif'),value)
    meta=dict(frame='test/world',origin_xyz=[0,0,0],spacing_xyz=[1,1,1],
              shape_zyx=[3,3,40],voxel_um=1.,step=2.)
    assert main(['patch-visuals','--field',str(path),*geometry_flags(meta),'--patch',str(patch),
        '--patch-frame','test/world','--margin','0','--output',str(tmp_path/'scores.json')])==0
    result=json.loads((tmp_path/'scores.json').read_bytes())
    assert result['patches'][0]['covered_points']==40 and result['patches'][0]['within_1_sheet_pct']==10


@pytest.mark.skipif(sys.platform!='linux',reason='constant-sheet process-tree resource admission is Linux-only')
def test_constant_sheet_prepare_accepts_repeated_centerline_points(tmp_path):
    from test_constant_sheet import extracted
    mesh=extracted(tmp_path)
    assert main(['constant-sheet','prepare','--mesh',str(mesh),'--w','3.5',
        '--centerline-point','0','0','0','--centerline-point','0','0','6',
        '--output',str(tmp_path/'prepared')])==0
    result=json.loads((tmp_path/'prepared/PREPARED.json').read_bytes())
    assert result['centerline']==[[0.,0.,0.],[0.,0.,6.]]


def test_constant_sheet_render_flags_reach_existing_identity_guard(tmp_path,capsys):
    source=tmp_path/'validated';source.mkdir()
    for name in ('map.npz','QUALITY.json','PREPARED.json','disconnected-source.npz'):
        (source/name).write_bytes(b'not-read-before-prior-identity-guard')
    atomic_json(source/'MESH.json',dict(metadata=dict(frame='test',voxel_um=1.)))
    atomic_json(source/'VALIDATED.json',dict(status='MAP_VALIDATED_DIAGNOSTIC',overlap={'complete':True},
        files={p.name:file_sha(p) for p in source.iterdir()}))
    prior=tmp_path/'prior.json';atomic_json(prior,dict(ct={'backend':'vc_reader'}))
    with pytest.raises(SystemExit) as error:
        main(['constant-sheet','render','--validated',str(source),'--output',str(tmp_path/'out'),
            '--ct-source','/must-not-open','--ct-level','0','--output-frame-id','test',
            '--output-origin-xyz-um','0','0','0','--output-voxel-um','1','--app-run','/unused',
            '--prior-identity',str(prior)])
    assert error.value.code==2 and 'v1.5.0' in capsys.readouterr().err
    assert not (tmp_path/'out').exists()


def test_sheet_atlas_subcommands_accept_complete_config_files(tmp_path,capsys):
    from test_sheet_atlas import sheet
    source=sheet(tmp_path/'source',-60)
    options=tmp_path/'build.json';atomic_json(options,dict(sheets=[str(source)],output='atlas',disk_reserve_gib=0))
    assert main(['sheet-atlas','build','--config',str(options)])==0
    capsys.readouterr()
    options=tmp_path/'query.json';atomic_json(options,dict(atlas='atlas',x=1,y=1))
    assert main(['sheet-atlas','query','--config',str(options)])==0
    assert json.loads(capsys.readouterr().out)['mask']==2
    options=tmp_path/'verify.json';atomic_json(options,dict(atlas='atlas'))
    before=pins(tmp_path/'atlas')
    if sys.platform!='linux':
        with pytest.raises(InterruptedError,match='Linux'):
            main(['sheet-atlas','verify','--config',str(options)])
        assert before==pins(tmp_path/'atlas')
        return
    assert main(['sheet-atlas','verify','--config',str(options)])==0
    assert before==pins(tmp_path/'atlas')


@pytest.mark.parametrize('separator',[False,True])
def test_run_watch_config_does_not_consume_child_config_arguments(tmp_path,separator):
    out=tmp_path/'watch';seen=tmp_path/'seen.json'
    command=[sys.executable,'-c','import json,sys;from pathlib import Path;Path(sys.argv[1]).write_text(json.dumps(sys.argv[2:]))',
             str(seen),'--config','child.json']
    options=tmp_path/'watch.json';atomic_json(options,dict(output=str(out),timeout=10.,reserve_gib=0.))
    assert main(['run-watch','--cli-config',str(options),*(['--'] if separator else []),*command])==0
    assert json.loads(seen.read_text())==['--config','child.json']


@pytest.mark.parametrize('command',list(TOOLS))
def test_each_public_tool_advertises_config_entry(command,capsys):
    with pytest.raises(SystemExit) as error:main([command,'--help'])
    assert error.value.code==0
    assert '--cli-config' in capsys.readouterr().out


def test_run_watch_command_can_come_entirely_from_config(tmp_path):
    seen=tmp_path/'seen.txt'
    command=[sys.executable,'-c','from pathlib import Path;import sys;Path(sys.argv[1]).write_text("ran without shell")',str(seen)]
    options=tmp_path/'watch.json'
    atomic_json(options,dict(command=command,output='watch',timeout=10.,reserve_gib=0.))
    assert main(['run-watch','--config',str(options)])==0
    assert seen.read_text()=='ran without shell'


def test_mixed_surface_metadata_keeps_original_raw_file_provenance(tmp_path):
    source=tmp_path/'field.npy';np.save(source,np.indices((5,6,7))[0].astype(float))
    metadata=tmp_path/'metadata.json'
    atomic_json(metadata,dict(frame='test',voxel_um=9.6,origin_xyz=[0,0,0],
        spacing_xyz=[1,1,1],shape_zyx=[5,6,7]))
    pin=file_sha(metadata)
    assert main(['surface-export','--field',str(source),'--metadata',str(metadata),
        '--field-voxel-um','12','--level','2.5','--pitch','1','--output',str(tmp_path/'export')])==0
    receipt=json.loads((tmp_path/'export/receipt.json').read_bytes())
    assert receipt['request_sources']=={str(metadata):pin}
    assert receipt['raster']['pitch_um']==[12.,12.]
    assert file_sha(metadata)==pin


def test_atlas_cannot_relabel_a_metadata_file_changed_during_admission(surface_case,monkeypatch):
    from winding_field.surface_probe import prepare_inputs
    from winding_field import tool_settings
    case=surface_case;root=case['root'];work=root/'inputs';work.mkdir()
    _,inputs=prepare_inputs(case['dataset'],case['field'],case['metadata'],work)
    path=Path(inputs['ct_metadata']);original=tool_settings.ct_request
    def changed(args,default=None):
        path.write_bytes(path.read_bytes()+b' ')
        return original(args,default)
    monkeypatch.setattr(tool_settings,'ct_request',changed)
    assert main(['surface-atlas','--field',str(root/'field.npy'),'--metadata',str(root/'field.npy.json'),
        '--seed',inputs['seed'],'--seed-sha256',inputs['seed_sha256'],'--ct',inputs['ct'],
        '--ct-sha256',inputs['ct_sha256'],'--ct-metadata',str(path),
        '--ct-metadata-sha256',inputs['ct_metadata_sha256'],'--app-run',str(native_fixture(root)),
        '--output',str(root/'atlas'),'--max-depth','0','--max-charts','1','--budget-seconds','30'])==2
    assert not (root/'atlas').exists()


def test_probe_shared_ct_window_flag_is_not_silently_ignored(surface_case):
    case=surface_case;root=case['root']
    flags=dataset_flags(case)[:-3]+['--ct-display-window','0','1234']
    assert main(['surface-probe','--field',str(root/'field.npy'),*geometry_flags(case['metadata']),*flags,
        '--work',str(root/'probe'),'--output',str(root/'export'),'--app-run',str(native_fixture(root))])==0
    result=json.loads((root/'probe/result.json').read_bytes())
    assert result['export_receipt']['preview']['window']==[0.,1234.]


@pytest.mark.parametrize('flags',[['--ct-voxel-um','1'],['--ct-shape-zyx','3','4','5']])
def test_probe_rejects_conflicting_ct_declarations_before_output(surface_case,flags):
    case=surface_case;root=case['root']
    assert main(['surface-probe','--config',str(case['path']),'--field',str(root/'field.npy'),
        '--metadata',str(root/'field.npy.json'),'--work',str(root/'probe'),'--output',str(root/'export'),
        '--app-run',str(native_fixture(root)),*flags])==2
    assert not (root/'probe').exists() and not (root/'export').exists()


def test_human_patch_atlas_does_not_require_unused_frozen_seed(tmp_path):
    import tifffile
    root=tmp_path
    field=root/'field.npy';np.save(field,np.indices((40,40,40))[2].astype('f4'))
    meta=dict(frame='test',voxel_um=9.6,origin_xyz=[0,0,0],spacing_xyz=[1,1,1],
              shape_zyx=[40,40,40],field_sha256=file_sha(field))
    metadata=root/'metadata.json';atomic_json(metadata,meta)
    ct=root/'ct.npy';np.save(ct,np.full((40,40,40),100,'u1'))
    ct_meta=root/'ct-meta.json';atomic_json(ct_meta,meta)
    patch=root/'patch';patch.mkdir();atomic_json(patch/'meta.json',dict(format='tifxyz',frame='test',voxel_um=9.6))
    z,y=np.mgrid[16.:25.,16.:25.]
    for axis,values in zip('xyz',(np.full(y.shape,20.),y,z)):tifffile.imwrite(patch/(axis+'.tif'),values)
    inputs=dict(ct=str(ct),ct_sha256=file_sha(ct),ct_metadata=str(ct_meta),ct_metadata_sha256=file_sha(ct_meta),
        field_sha256=file_sha(field),metadata_sha256=file_sha(metadata),
        render=dict(mesh_step=1,pitch_world=2,depth_offsets_world=[0]))
    path=root/'inputs.json';atomic_json(path,inputs)
    assert main(['surface-atlas','--field',str(field),'--metadata',str(metadata),
        '--inputs',str(path),'--patch',str(patch),'--patch-frame','test',
        '--patch-voxel-um','9.6','--app-run',str(native_fixture(root)),'--output',str(root/'atlas'),
        '--max-depth','0','--max-charts','1','--budget-seconds','30'])==0
    assert json.loads((root/'atlas/result.json').read_bytes())['human_seed'] is True
