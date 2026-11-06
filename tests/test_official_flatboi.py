"""Official native qualification keeps binary-bound checkpoints and old defaults."""
import json
import os
import sys

import pytest

from winding_field import patch_pipeline as p
from winding_field.util import atomic_json, file_sha


def test_explicit_official_recipe_requires_its_own_binary(tmp_path,monkeypatch):
    app=tmp_path/'AppRun';app.write_text('launcher');app.chmod(0o755)
    binary=tmp_path/'usr/bin/flatboi';binary.parent.mkdir(parents=True);binary.write_text('binary')
    sha='d724cf0cc6940d7c959e36c73ed1a9a977d3b90bab60fc97d519fe370cc63f38'
    monkeypatch.setattr(p,'file_sha',lambda path:p.LAUNCHER_SHA if path==app else sha)
    receipt=p.check_flatboi(app,'historical')
    assert receipt['binary_sha256']==sha and receipt['historical_reproduction']
    with pytest.raises(ValueError,match='binary SHA256'):p.check_flatboi(app,'installed-2f89bfc9')
    sha='0'*64
    with pytest.raises(ValueError,match='binary SHA256'):p.check_flatboi(app,'historical')


def test_matrix_selects_only_checkpoint_qualified_recipes(tmp_path):
    from winding_field.flattening_matrix import settings
    from test_flattening_matrix import request
    req=dict(request(tmp_path),app_run='/opt/vc3d/AppRun')
    assert settings(req,'slim','sheet')['native_recipe']=='installed-2f89bfc9'
    for scope in ('patches','sheet'):
        assert settings(dict(req,native_recipe='historical'),'slim',scope)['native_recipe']=='historical'
        for bad in ('unknown',None):
            with pytest.raises(ValueError,match='recipe'):
                settings(dict(req,native_recipe=bad),'slim',scope)
    with pytest.raises(ValueError,match='unsupported'):
        settings(dict(request(tmp_path),native_recipe='historical'),'lscm','sheet')


@pytest.mark.skipif(sys.platform!='linux',reason='Linux native qualification')
def test_official_constant_sheet_checkpoint_resume_import_and_validate(tmp_path):
    app=os.environ.get('MATRIX_NATIVE_APP')
    if not app or os.environ.get('MATRIX_NATIVE_RECIPE')!='historical':
        pytest.skip('requires MATRIX_NATIVE_APP, MATRIX_NATIVE_RECIPE=historical and bounded cgroup')
    from winding_field.cli import main
    from winding_field import flatboi_checkpoint as r
    from test_constant_sheet import extracted
    mesh=extracted(tmp_path);cp=tmp_path/'centerline.json'
    atomic_json(cp,[[0,0,0],[0,0,6]])
    prepared=tmp_path/'prepared';native=tmp_path/'native'
    assert main(['constant-sheet','prepare','--mesh',str(mesh),'--w','3.5',
                 '--centerline',str(cp),'--output',str(prepared)])==0
    source_sha=file_sha(prepared/'inputmesh.obj')
    args=['constant-sheet','flatten','--prepared',str(prepared),'--output',str(native),
          '--app-run',app,'--native-recipe','historical','--batch-iterations','1']
    assert main(args+['--iterations','1'])==0
    first=json.loads((native/'LATEST.json').read_text());saved=native/first['obj'];saved_sha=file_sha(saved)
    assert first['iterations']==1 and not first['converged']
    assert main(args+['--iterations','2'])==0
    second=json.loads((native/'LATEST.json').read_text())
    assert second['iterations']==2 and file_sha(saved)==saved_sha
    assert file_sha(prepared/'inputmesh.obj')==source_sha
    assert second['identity']['binary_sha256']=='d724cf0cc6940d7c959e36c73ed1a9a977d3b90bab60fc97d519fe370cc63f38'
    before=file_sha(native/'LATEST.json')
    wrong=args.copy();wrong[wrong.index('historical')]='installed-2f89bfc9'
    with pytest.raises(SystemExit):main(wrong+['--iterations','3'])
    assert file_sha(native/'LATEST.json')==before
    imported=tmp_path/'imported'
    assert main(['constant-sheet','import-native','--prepared',str(prepared),'--native',str(native),
                 '--output',str(imported),'--app-run',app,'--native-recipe','historical',
                 '--runner-sha256',file_sha(r.__file__)])==0
    continuation=args.copy();continuation[continuation.index(str(native))]=str(imported)
    assert main(continuation+['--iterations','200'])==0
    assert main(['constant-sheet','validate','--prepared',str(prepared),'--native',str(imported),
                 '--reference-z','3','--output',str(tmp_path/'validated')])==0
    assert file_sha(saved)==saved_sha


@pytest.mark.skipif(sys.platform!='linux',reason='Linux native qualification')
def test_official_historical_patch_cli_retains_source_faces_and_rejects_resume(tmp_path,capsys):
    app=os.environ.get('MATRIX_NATIVE_APP')
    if not app or os.environ.get('MATRIX_NATIVE_RECIPE')!='historical':
        pytest.skip('requires official historical MATRIX_NATIVE_APP')
    import numpy as np
    from winding_field.cli import main
    from test_flattening_matrix import request,source_bundle
    mesh,ct=source_bundle(tmp_path)
    config=tmp_path/'patch.json'
    # Resolve the small synthetic domains after conservative TIFXYZ boundary erosion.
    atomic_json(config,dict(request(tmp_path),app_run=app,max_patch_faces=32,pitch_um=2.4,tile_size=64))
    output=tmp_path/'output'
    args=['export-sheets','--mesh-bundle',str(mesh),'--source',str(ct),'--voxel-um','9.6',
          '--flattening-method','patches','--flattening-config',str(config),'--output',str(output),'--export-tifxyz']
    assert main(args)==0
    level=output/'patches/level-0000'
    with np.load(level/'map.npz') as mapped:
        assert len(set(mapped['source_face_ids']))==121
        np.testing.assert_array_equal(mapped['field_xyz'][:,1],3.)
    assert list((level/'render').glob('*/tifxyz/VERIFIED.json'))
    before={str(path):file_sha(path) for path in output.rglob('*') if path.is_file() and path.suffix in ('.json','.npz')}
    assert main(args+['--resume'])==1
    assert 'patch export does not resume' in capsys.readouterr().err
    assert before=={name:file_sha(name) for name in before}
