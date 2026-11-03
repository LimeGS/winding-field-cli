"""Small real checkpoint fixtures, without importing the field solver or torch."""
import itertools
import json

import numpy as np

from field_store import BlockStore
from winding_field.util import atomic_json,digest,file_sha


def field_run(root,shape=(5,5,5),*,unit=1.,origin=(0,0,0),spacing=(1,1,1),unsupported=None):
    root.mkdir()
    source=dict(source='synthetic',level=0,shape=[100,100,100],origin_xyz=[0,0,0],spacing_xyz=[unit]*3,metadata_sha256='a'*64)
    geometry=dict(frame=digest(source),voxel_um=unit,sheet_step=1.,origin_xyz=list(origin),spacing_xyz=list(spacing),shape_zyx=list(shape))
    with BlockStore(root/'field',shape,2,{'geometry':geometry}) as store:
        for key in itertools.product(*(range((n+1)//2) for n in shape)):
            lo=np.array(key)*2;hi=np.minimum(lo+2,shape)
            z,y,x=np.meshgrid(*(np.arange(a,b) for a,b in zip(lo,hi)),indexing='ij')
            values=x+2*y+3*z
            support=np.ones(values.shape,bool)
            if unsupported is not None:support&=~((z==unsupported[0])&(y==unsupported[1])&(x==unsupported[2]))
            store.commit(key,values.astype(float),{},support)
    atomic_json(root/'FIELD.json',dict(status='FIELD_READY',geometry=geometry,checkpoint_sha256=file_sha(root/'field/CHECKPOINT.json')))
    atomic_json(root/'RUN.json',dict(schema='winding-field-run-v1',implementation_sha256='b'*64,source=source,
        stages={'field':{'receipt':'FIELD.json','sha256':file_sha(root/'FIELD.json'),'seconds':1}},status='PAUSED'))
    return root


def mutate_json(path,edit):
    data=json.loads(path.read_text());edit(data);atomic_json(path,data)
