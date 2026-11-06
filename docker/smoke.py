"""Installed CLI smoke check; --models also runs pinned Lasagna and Qual inference."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from winding_field.util import file_sha

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,required=True,help='new persistent directory')
parser.add_argument('--models',action='store_true')
parser.add_argument('--device',default='cpu',choices=('cpu','cuda:0'))
args=parser.parse_args()
root=args.output.absolute();root.mkdir(parents=True,exist_ok=False)


def cli(*arguments):
    subprocess.run([sys.executable,'-I','-m','winding_field',*map(str,arguments)],check=True,cwd=root)


cli('doctor')
demo=root/'demo';cli('demo','--output',demo)
record=json.loads((demo/'RUN.json').read_text())
assert record['status']=='COMPLETE' and record['physical_acceptance'] is False
assert list((demo/'meshes').glob('*.obj')) and list((demo/'flat').glob('*.png'))
before=file_sha(demo/'field/CHECKPOINT.json')
cli('resume',demo)
assert file_sha(demo/'field/CHECKPOINT.json')==before

roi=root/'roi'
cli('run','--source',record['config']['source'],'--output',roi,'--prediction','classical',
    '--roi',0,16,0,24,0,40,'--field-spacing-um',8,'--seed-stride',4,'--sweeps',1,
    '--until','field','--cache',root/'cache')
assert (roi/'FIELD.json').is_file()
before=file_sha(roi/'field/CHECKPOINT.json')
cli('field',roi)
assert file_sha(roi/'field/CHECKPOINT.json')==before

if args.models:
    from winding_field.bounded_lasagna import LasagnaModel
    from winding_field.lasagna import runtime_identity
    from winding_field.models import checkpoint,QualV2
    identity=runtime_identity()
    weights=checkpoint(prediction='lasagna')
    rng=np.random.default_rng(0)
    tile=rng.integers(1,255,(64,)*3,dtype=np.uint8)
    with LasagnaModel(identity['executable'],weights,args.device,root/'lasagna.log') as model:
        raw=model.predict_raw(tile);encoded=model.finalize(raw)
        assert raw.shape==(8,64,64,64) and np.isfinite(raw).all()
        assert encoded.shape==(4,64,64,64) and encoded.dtype==np.uint8
    model=QualV2(device=args.device)
    result=model.predict(rng.integers(1,255,(128,)*3,dtype=np.uint8))
    assert all(np.isfinite(value).all() for value in result)
    (root/'MODEL-RUNTIME.json').write_text(json.dumps(dict(runtime=identity,device=args.device,
        weights=dict(lasagna=file_sha(weights),qual_v2=file_sha(checkpoint(prediction='qual-v2')))),sort_keys=True))

(root/'SMOKE.json').write_text(json.dumps(dict(status='PASSED',models=args.models,device=args.device)))
print('Container smoke passed:',root)
