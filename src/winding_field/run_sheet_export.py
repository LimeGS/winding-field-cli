"""Resolve an immutable saved field run into the existing sheet exporter."""
from contextlib import closing
import json
import hashlib
import math
from pathlib import Path
import sqlite3

from .util import digest,file_sha


def export_run(args):
    from .original_pipeline import export_existing,settings
    from .surface_sources import FieldSource
    from .workflow import open_volume

    if any(getattr(args,k,None) is not None for k in ('source','level','voxel_um','flattening_solver','scope')):
        raise ValueError('--run uses saved CT metadata; omit --source, --level, --voxel-um and solver/scope overrides')
    if args.flattening_method not in (None,'original'):
        raise ValueError('--run requires the original flattening method')
    if not args.levels or not all(math.isfinite(v) for v in args.levels):
        raise ValueError('--levels requires finite raw field isovalues with --run')
    root=Path(args.run).expanduser().resolve();out=Path(args.output).expanduser().resolve()
    if root==out or root in out.parents or out in root.parents:
        raise ValueError('export output must not overlap the source run')
    state_path=root/'RUN.json';field_path=root/'FIELD.json'
    state_sha=file_sha(state_path);state=json.loads(state_path.read_bytes())
    if not isinstance(state,dict) or not isinstance(state.get('stages'),dict):
        raise ValueError('saved run requires RUN.json with stage metadata')
    stage=state.get('stages',{}).get('field',{})
    if not isinstance(stage,dict) or stage.get('receipt')!='FIELD.json' or stage.get('sha256')!=file_sha(field_path):
        raise ValueError('saved run has no verified field stage or FIELD.json changed')
    field=json.loads(field_path.read_bytes())
    if state.get('schema')!='winding-field-run-v1' or not isinstance(field,dict) or field.get('status')!='FIELD_READY':
        raise ValueError('saved run requires a completed FIELD_READY field')
    config=state.get('config');source=state.get('source');geometry=field.get('geometry')
    if not all(isinstance(v,dict) for v in (config,source,geometry)) or not isinstance(config.get('source'),str):
        raise ValueError('saved run is missing configuration, source or field geometry metadata')
    config=dict(config)
    if digest(source)!=geometry.get('frame'):
        raise ValueError('saved field and CT coordinate frames differ')
    descriptor=dict(field=dict(kind='chunks',path=str(root/'field/CHECKPOINT.json'),
                              sha256=field.get('checkpoint_sha256'),support_policy='observed'),geometry=geometry)
    with FieldSource(descriptor['field'],geometry) as reader:
        reader.validate(hashes=True)
    request=settings(json.loads(Path(args.flattening_config).read_bytes()) if args.flattening_config else
                     dict(output_frame=dict(id=geometry['frame'],voxel_um=1.,origin_xyz_um=[0.,0.,0.])))
    ledger=root/'source.sqlite';ledger_sha=file_sha(ledger)
    # A saved cache inside the source run must not turn this read-only export into a writer there.
    if config.get('cache'):
        cache=Path(config['cache']).expanduser().resolve()
        if cache==root or root in cache.parents:config.pop('cache')
    with closing(open_volume(config)) as volume:
        if volume.identity!=source:
            raise ValueError('saved source CT metadata changed')
        with closing(sqlite3.connect(ledger.as_uri()+'?mode=ro',uri=True)) as db:
            if db.execute('PRAGMA integrity_check').fetchall()!=[('ok',)]:
                raise ValueError('saved CT source ledger is corrupt')
            for key,sha in db.execute('SELECT key,sha FROM chunks'):
                raw=volume._chunk_bytes(key)
                observed=digest('authoritative-zarr-fill') if raw is None else hashlib.sha256(raw).hexdigest()
                if observed!=sha:raise ValueError('saved source CT chunk changed: '+key)
        if file_sha(state_path)!=state_sha or file_sha(field_path)!=stage['sha256'] or file_sha(ledger)!=ledger_sha:
            raise ValueError('saved run changed during export admission')
        args.source=config['source'];args.level=config.get('level',0);args.voxel_um=config.get('voxel_um')
        resolved=dict(field=descriptor,settings=request,
                      binding=dict(path=str(root),run_sha256=state_sha,field_sha256=stage['sha256'],
                                   source_ledger_sha256=ledger_sha,source=source))
        return export_existing(args,run_export=resolved,volume=volume)
