import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from tifxyz_fixture import field_run,mutate_json


def test_roi_physical_sampling_and_stable_run_identity(tmp_path):
    from winding_field.tifxyz_field import open_physical_field
    root=field_run(tmp_path/'source',unit=8.64,origin=(11,23,37),spacing=(2,3,4))
    with open_physical_field(root,cache_bytes=1024,roi={'lo_zyx':[1,1,1],'hi_zyx':[5,5,5]}) as f:
        np.testing.assert_allclose(f.origin_um,[112.32,224.64,354.24])
        values,valid=f.sample(np.array([[112.32,250.56,423.36]]))
        assert valid.tolist()==[True];np.testing.assert_allclose(values,[14.])
        before=f.identity
        mutate_json(root/'RUN.json',lambda r:r.update(status='COMPLETE',updated=123))
        f.validate();assert f.identity==before
        with pytest.raises(ValueError):f.read([-1,0,0],[1,1,1])
        mutate_json(root/'RUN.json',lambda r:r['source'].update(level=1))
        with pytest.raises(ValueError,match='binding|frame'):f.validate()


def test_receipt_checkpoint_and_frame_binding(tmp_path):
    from winding_field.tifxyz_field import open_physical_field
    root=field_run(tmp_path/'source')
    mutate_json(root/'FIELD.json',lambda r:r['geometry'].update(frame='another volume'))
    with pytest.raises(ValueError,match='receipt'):open_physical_field(root,cache_bytes=0)


def test_checkpoint_change_detected_after_cache_hit(tmp_path):
    from winding_field.tifxyz_field import open_physical_field
    root=field_run(tmp_path/'source')
    with open_physical_field(root,cache_bytes=1024**2) as f:
        f.read([0,0,0],[2,2,2]);manifest=json.loads((root/'field/CHECKPOINT.json').read_text())
        payload=root/'field'/next(iter(manifest['chunks'].values()))['path']
        payload.write_bytes(b'corrupted')
        with pytest.raises(ValueError):f.read([0,0,0],[2,2,2])


def test_manifest_envelope_precedes_json_decode(tmp_path):
    from winding_field.tifxyz_field import open_physical_field
    root=field_run(tmp_path/'source')
    with (root/'field/CHECKPOINT.json').open('wb') as stream:stream.truncate(16*1024**2+1)
    with pytest.raises(ValueError,match='metadata'):open_physical_field(root,cache_bytes=0)


def test_opener_does_not_import_torch_or_change_source(tmp_path):
    from winding_field.util import file_sha
    root=field_run(tmp_path/'source');before={str(p):file_sha(p) for p in root.rglob('*') if p.is_file()}
    code='import sys; from winding_field.tifxyz_field import open_physical_field; f=open_physical_field(sys.argv[1],cache_bytes=0); f.read([0,0,0],[2,2,2]); f.close(); assert "torch" not in sys.modules'
    run=subprocess.run([sys.executable,'-B','-c',code,str(root)],capture_output=True,text=True,timeout=20)
    assert run.returncode==0,run.stderr
    assert before=={str(p):file_sha(p) for p in root.rglob('*') if p.is_file()}


def test_all_eight_support_nodes_across_chunks_required(tmp_path):
    from winding_field.tifxyz_field import open_physical_field
    root=field_run(tmp_path/'source',unsupported=(2,2,2))
    with open_physical_field(root,cache_bytes=0) as f:
        _,valid=f.sample(np.array([[1.5,1.5,1.5],[.5,.5,.5],[-.001,.5,.5]]))
        assert valid.tolist()==[False,True,False]


def test_coarse_ranges_are_streamed_and_receipt_reopens(tmp_path):
    from winding_field.tifxyz_field import build_coarse,load_coarse,iter_coarse_ranges
    class ThinField:
        shape=(2,2,5242881)
        origin_um=np.zeros(3);spacing_um=np.ones(3);identity={'fixture':'bounded-thin-field'}
        def validate(self,hashes=False):pass
        def read(self,lo,hi):
            assert max(b-a for a,b in zip(lo,hi))<=65
            values=np.broadcast_to(np.arange(lo[2],hi[2],dtype=float),(hi[0]-lo[0],hi[1]-lo[1],hi[2]-lo[2])).copy()
            return values,np.ones(values.shape,bool)
    receipt=build_coarse(ThinField(),32,tmp_path/'coarse')
    assert isinstance(receipt['tile_ranges'],dict) and receipt['tile_ranges']['count']==81920
    assert (tmp_path/'coarse/COARSE.json').stat().st_size<16384
    values,cells,axes=load_coarse(receipt)
    assert values.shape==(2,2,163841) and cells.all()
    assert sum(1 for _ in iter_coarse_ranges(receipt))==81920
    path=tmp_path/'coarse'/receipt['tile_ranges']['path'];path.write_bytes(b'corrupted ranges\n')
    with pytest.raises(ValueError,match='range'):list(iter_coarse_ranges(receipt))


def test_coarse_sidecar_budget_checked_before_read(tmp_path):
    from types import SimpleNamespace
    from winding_field.tifxyz_field import build_coarse
    from winding_field.tifxyz_contract import Blocked
    field=SimpleNamespace(shape=(2,2,5242881))
    with pytest.raises(Blocked,match='SCRATCH'):
        build_coarse(field,32,tmp_path/'coarse',budget=SimpleNamespace(scratch_bytes=1024,reserve_disk_bytes=0))
    assert not (tmp_path/'coarse').exists()
