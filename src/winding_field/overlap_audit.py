"""Atomic, resumable whole-sheet UV overlap audit state."""
import base64
import hashlib
import json
from pathlib import Path
import zlib

import numpy as np

from .flatten import iter_overlap_blocks
from .util import atomic_json

PREDICATE='triangle-clip-area-v1'
TRAVERSAL='self-bvh-aabb-v1'


def _array_sha(array):
    array=np.ascontiguousarray(array)
    return hashlib.sha256(f'{array.dtype.str}{array.shape}'.encode()+array.tobytes()).hexdigest()


def _identity(uv,faces):
    return dict(uv_sha256=_array_sha(uv),faces_sha256=_array_sha(faces),source_faces=int(len(faces)),
                predicate=PREDICATE,traversal=TRAVERSAL)


def _pack(bits):
    return base64.b64encode(zlib.compress(np.packbits(bits).tobytes(),9)).decode()


def _unpack(text,count):
    try:
        bits=np.unpackbits(np.frombuffer(zlib.decompress(base64.b64decode(text,validate=True)),np.uint8))
    except (ValueError,zlib.error) as error:
        raise ValueError('overlap audit face bits corrupt') from error
    if len(bits)<count or bits[count:].any():
        raise ValueError('overlap audit face bits do not match source faces')
    return bits[:count].astype(bool)


def _receipt(state,involved):
    return dict(status=state['status'],complete=state['status']=='COMPLETE',
                identity=state['identity'],predicate=state['predicate'],
                traversal=state['traversal'],cursor=state['cursor'],pairs=state['pairs'],
                narrow_tests=state['narrow_tests'],source_faces=state['source_faces'],
                overlapping_faces=int(involved.sum()),involved=involved)


def audit_uv(uv,faces,output,*,guard,commit_tests=1_000_000):
    """Resume an exact overlap audit from its atomic cursor; never infer completion."""
    uv=np.asarray(uv);faces=np.asarray(faces)
    if (uv.ndim!=2 or uv.shape[1:]!=(2,) or faces.ndim!=2 or faces.shape[1:]!=(3,)
            or faces.dtype.kind not in 'iu' or not np.isfinite(uv).all()
            or (faces.size and (faces.min()<0 or faces.max()>=len(uv)))):
        raise ValueError('invalid overlap audit arrays')
    if type(commit_tests) is not int or commit_tests<=0:raise ValueError('invalid audit commit interval')
    identity=_identity(uv,faces)
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    path=out/'AUDIT.json'
    involved=np.zeros(len(faces),bool);cursor=0;pairs=0;tests=0
    if path.exists():
        state=json.loads(path.read_text())
        if state.get('identity')!=identity:raise ValueError('overlap audit input changed')
        if any(type(state.get(key)) is not int or state[key]<0 for key in ('cursor','pairs','narrow_tests')):
            raise ValueError('overlap audit state corrupt')
        if state['cursor']>len(faces):raise ValueError('overlap audit cursor out of range')
        cursor,pairs,tests=state['cursor'],state['pairs'],state['narrow_tests']
        involved=_unpack(state['involved_bits'],len(faces))
        if state.get('status')=='COMPLETE':
            if cursor!=len(faces):raise ValueError('complete overlap audit does not cover every face')
            return _receipt(state,involved)
    def publish(status):
        state=dict(status=status,identity=identity,predicate=PREDICATE,traversal=TRAVERSAL,
                   cursor=cursor,pairs=pairs,narrow_tests=tests,source_faces=len(faces),
                   involved_bits=_pack(involved))
        atomic_json(path,state)
        return state
    publish('AUDIT_PENDING');committed=tests
    try:
        for block in iter_overlap_blocks(uv,faces,guard,start=cursor,narrow_tests=tests):
            for i,j in block['pairs']:involved[[i,j]]=True
            pairs+=len(block['pairs']);tests+=block['narrow_tests'];cursor=block['start']
            if tests-committed>=commit_tests:publish('AUDIT_PENDING');committed=tests
    except BaseException:
        publish('AUDIT_PENDING');raise
    return _receipt(publish('COMPLETE'),involved)
