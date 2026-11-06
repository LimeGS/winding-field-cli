"""Pinned-core adapter: atomically extend only an exhausted field's sweep budget."""
from copy import deepcopy
import hashlib
import math
from pathlib import Path
import re

from field_store import BlockStore,_json_bytes,_json_load,_read_file


def _digest(value):return hashlib.sha256(_json_bytes(value)).hexdigest()


def _budget(manifest,initial):
    budget=initial;history=manifest.get('sweep_extensions',[])
    if type(initial) is not int or initial<1 or not isinstance(history,list):
        raise ValueError('invalid field sweep history')
    for entry in history:
        if (not isinstance(entry,dict) or type(entry.get('from')) is not int
                or entry['from']!=budget or type(entry.get('to')) is not int or entry['to']<=budget):
            raise ValueError('invalid field sweep history')
        for key in ('previous_checkpoint_sha256','identity_sha256','progress_sha256','chunks_sha256'):
            if not isinstance(entry.get(key),str) or not re.fullmatch('[0-9a-f]{64}',entry[key]):
                raise ValueError('invalid field sweep audit digest')
        if entry['identity_sha256']!=_digest(dict(manifest['identity'],sweeps=budget)):
            raise ValueError('field problem changed across sweep extension')
        budget=entry['to']
    if type(manifest['identity'].get('sweeps')) is not int or manifest['identity']['sweeps']!=budget:
        raise ValueError('field sweep budget differs from extension history')
    return budget


def effective_budget(root,initial):
    path=Path(root)/'CHECKPOINT.json'
    return _budget(_json_load(_read_file(path)),initial) if path.exists() else initial


def extend(root,target,initial,*,reserve_bytes,max_bytes=None):
    """Caller holds the run lock and has validated the problem with solve_blocks.

    The checkpoint is the only commit point. Its embedded audit survives core
    commits; an absolute target can be retried after either side of publication.
    """
    if type(target) is not int or target<1:raise ValueError('positive field sweep target required')
    root=Path(root);path=root/'CHECKPOINT.json';saved=_json_load(_read_file(path))
    with BlockStore(root,saved['shape'],saved['chunk'],saved['identity'],resume=True,
                    reserve_bytes=reserve_bytes,max_bytes=max_bytes) as store:
        store._ensure_open()
        old=store.manifest;raw=_read_file(path)
        if _json_load(raw)!=old:raise ValueError('checkpoint changed during sweep extension')
        budget=_budget(old,initial)
        if target<budget:raise ValueError('field sweep target cannot decrease')
        if target==budget:return budget
        if any((root.parent/name).exists() for name in
               ('FIELD.json','MESHES.json','meshes','flat','original','surfaces','tifxyz')):
            raise ValueError('cannot extend a published field')
        count=math.prod((n-1)//old['chunk']+1 for n in old['shape'])
        progress=old['progress'];completed=progress.get('completed_sweeps',[])
        if (len(old['chunks'])!=count or progress.get('sweep')!=budget
                or progress.get('next_block')!=count or len(completed)!=budget
                or any(row.get('sweep')!=i or row.get('cores')!=count for i,row in enumerate(completed,1))):
            raise ValueError('field extension requires a complete exhausted sweep boundary')
        proposed=deepcopy(old);proposed['identity']['sweeps']=target
        proposed.setdefault('sweep_extensions',[]).append(dict(
            **{'from':budget,'to':target},previous_checkpoint_sha256=hashlib.sha256(raw).hexdigest(),
            identity_sha256=_digest(old['identity']),progress_sha256=_digest(progress),chunks_sha256=_digest(old['chunks'])))
        # Private pinned-core publication preserves admission, lease and fsync semantics.
        store._publish(proposed)
    return target
