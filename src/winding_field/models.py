"""Pinned public fibre inference and verified checkpoint acquisition."""

import json
from pathlib import Path
import os
import tempfile
from urllib.request import urlopen

import numpy as np
import torch

from .util import file_sha

MODEL_SHA='2528a46e38844c06e591084f1dc1af9442ccb45a17aae9b2a68441e6028bbdd6'
MODEL_REV='a9aaf89cc504206f4dda0f0398a4f3e263fee25f'
MODEL_FILE='s1a_128_2_single_8x8_20260728_094259_best_25_9k.pt'
MODEL_URL=f'https://huggingface.co/scrollprize/lasagna-fiber/resolve/{MODEL_REV}/{MODEL_FILE}'
QUAL_MODEL_SHA='b8519a4ce4044c93ed98af0c278628fdcdfdd8808275b3c8becd7211f38b7967'
QUAL_MODEL_REV='2a8a26a70754a5d90aeca91f9ef8e7130e6e4f81'
QUAL_MODEL_URL=f'https://huggingface.co/Qualzz20/afv_fiber_9um_v2/resolve/{QUAL_MODEL_REV}/checkpoint_final.pth'


def model_sha(prediction):
    return {'sd2':MODEL_SHA,'qual-v2':QUAL_MODEL_SHA,'classical':None}[prediction]


def verify_checkpoint(path,*,prediction='sd2'):
    if file_sha(path)!=model_sha(prediction):
        raise ValueError('checkpoint SHA-256 mismatch; refusing to load')


def checkpoint(path=None,offline=False,*,prediction='sd2'):
    if prediction not in ('sd2','qual-v2'):
        raise ValueError('checkpoint requires sd2 or qual-v2')
    path=Path(path or f'~/.cache/winding-field/models/{prediction}.pt').expanduser()
    if not path.exists():
        if offline:
            raise FileNotFoundError(f'{prediction} model not cached; run winding-field models --prediction {prediction} first')
        path.parent.mkdir(parents=True,exist_ok=True)
        fd,name=tempfile.mkstemp(prefix='.'+path.name+'-',suffix='.download',dir=path.parent)
        temporary=Path(name)
        try:
            url=QUAL_MODEL_URL if prediction=='qual-v2' else MODEL_URL
            with os.fdopen(fd,'wb') as out, urlopen(url,timeout=120) as response:
                total=0
                while block:=response.read(1<<20):
                    total+=len(block)
                    if total>1<<30:
                        raise ValueError('checkpoint exceeds 1 GiB budget')
                    out.write(block)
                out.flush(); os.fsync(out.fileno())
            verify_checkpoint(temporary,prediction=prediction)
            os.replace(temporary,path)
        finally:
            temporary.unlink(missing_ok=True)
    verify_checkpoint(path,prediction=prediction)
    return path


class SD2:
    def __init__(self,path=None,device='cpu',offline=False):
        from ._vendor.vesuvius.neural_tracing.fiber_trace_3d.model import build_fiber_trace_3d_model
        from ._vendor.vesuvius.neural_tracing.fiber_trace_3d.direction import decode_lasagna_direction_3x2_analytic
        path=checkpoint(path,offline)
        payload=torch.load(path,map_location='cpu',weights_only=False)  # hash-verified public checkpoint only
        self.config=payload['config']
        self.model=build_fiber_trace_3d_model(self.config)
        self.model.load_state_dict(payload['model'],strict=True)
        self.model.to(device).eval()
        self.device=device; self.decode=decode_lasagna_direction_3x2_analytic

    @torch.inference_mode()
    def predict(self,ct):
        if ct.dtype!=np.uint8:
            raise ValueError('pinned SD2 expects uint8 CT; use explicit acquisition normalization before inference')
        shape=ct.shape
        # Upstream architecture uses powers-of-two strides; pad only input, then trim.
        pad=[(0,(-n)%32) for n in shape]
        values=ct.astype(np.float32)
        mode=self.config.get('image_normalization','zscore')
        if mode!='zscore':
            raise ValueError('unexpected pinned model normalization')
        array=np.pad((values-values.mean())/max(float(values.std()),1e-6),pad,mode='edge')
        output=self.model(torch.from_numpy(array)[None,None].to(self.device))
        if output.shape[1]!=7:
            raise ValueError('unexpected SD2 branch count')
        tangent=self.decode(output[:,:6].permute(0,2,3,4,1))[0].cpu().numpy()
        presence=output[0,6].cpu().numpy()
        core=tuple(slice(0,n) for n in shape)
        return presence[core].copy(),tangent[core].copy()


class QualV2:
    def __init__(self,path=None,device='cpu',offline=False):
        from ._vendor.qual_model import FiberDetector
        path=checkpoint(path,offline,prediction='qual-v2')
        plans=json.loads((Path(__file__).parent/'_vendor/qual_plans.json').read_bytes())
        self.model=FiberDetector(plans)
        payload=torch.load(path,map_location='cpu',weights_only=True)
        state={'backbone.'+k:v for k,v in payload['network_weights'].items()}
        state.update(payload['auxiliary_weights'])
        self.model.load_state_dict(state,strict=True)
        self.model.to(device).eval(); self.device=device

    @torch.inference_mode()
    def predict(self,ct):
        from ._vendor.qual_model import normalize_ct
        if ct.dtype!=np.uint8 or ct.shape!=(128,)*3:
            raise ValueError('Qual V2 expects uint8 CT in fixed 128³ windows')
        image=torch.from_numpy(ct.astype(np.float32))[None,None].to(self.device)
        output=self.model(normalize_ct(image))
        probabilities=output['seg'].softmax(1)[0]
        tangent=torch.nn.functional.normalize(output['direction'],dim=1)[0].permute(1,2,3,0)
        if not torch.isfinite(probabilities).all() or not torch.isfinite(tangent).all():
            raise ValueError('Qual V2 returned nonfinite predictions')
        return ((1-probabilities[0]).cpu().numpy().copy(),tangent.cpu().numpy().copy(),
                probabilities[1:3].permute(1,2,3,0).cpu().numpy().copy())
