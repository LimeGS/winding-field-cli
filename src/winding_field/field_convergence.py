"""Opt-in convergence of one fixed global problem; legacy fields are not migrated."""
import argparse
from contextlib import contextmanager,nullcontext
import fcntl
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import secrets
import shutil
import signal
import stat
import sys

import numpy as np

from block_solver import solve_blocks
from field_runtime import ResourceUnavailable
from field_store import BlockStore,StorageBudgetError

from .surface_sources import FieldSource
from .util import digest,file_sha

DEFAULTS=dict(field_chunk=64,rtol=1e-6,maxiter=2000,fibre_weight=1.,device='cpu',threads=4,
              reserve_bytes=1<<30,bounded_bytes=12<<30,prepared_cache_bytes=0,constraint_cache_bytes=0)
CONFIG_KEYS=set(DEFAULTS)|{'field_spacing_um'}


def policy(max_sweeps,global_rtol=1e-6):
    if type(max_sweeps) is not int or max_sweeps<1:
        raise ValueError('max_sweeps must be a positive finite absolute total')
    if type(global_rtol) not in (int,float) or not math.isfinite(global_rtol) or global_rtol<=0:
        raise ValueError('global_rtol must be positive and finite')
    return dict(max_sweeps=max_sweeps,global_rtol=float(global_rtol),
                stagnation_window=5,stagnation_improvement=1e-3)


def _read(path):
    from .cli_settings import read_settings
    return read_settings(path)


def _save(path,data):
    _publish(path,dict(data=data,sha256=digest(data)))


@contextmanager
def _directory(path,*,create=False):
    """Anchor every component; a swapped symlink must never redirect a write."""
    path=Path(path).absolute();fd=os.open(path.anchor,os.O_RDONLY|os.O_DIRECTORY)
    try:
        for name in path.parts[1:]:
            if name in ('.','..'):raise ValueError('unclean managed directory')
            if create:
                try:
                    os.mkdir(name,dir_fd=fd)
                    os.fsync(fd)
                except FileExistsError:pass
            child=os.open(name,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd)
            os.close(fd);fd=child
        yield fd
    finally:os.close(fd)


def _publish(path,data):
    path=Path(path)
    with _directory(path.parent,create=True) as directory:
        name='.'+path.name+'-'+secrets.token_hex(16)
        fd=os.open(name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=directory)
        try:
            with os.fdopen(fd,'w') as stream:
                json.dump(data,stream,indent=2,sort_keys=True,allow_nan=False)
                stream.write('\n');stream.flush();os.fsync(stream.fileno())
            os.replace(name,path.name,src_dir_fd=directory,dst_dir_fd=directory)
            os.fsync(directory)
        finally:
            try:os.unlink(name,dir_fd=directory)
            except FileNotFoundError:pass


def _checked(path):
    envelope=_read(path)
    if set(envelope)!={'data','sha256'} or digest(envelope['data'])!=envelope['sha256']:
        raise ValueError('convergence metadata checksum differs')
    return envelope['data']


class Controller:
    """Verify committed sweep snapshots inside the existing core's stop callback."""
    def __init__(self,run,settings,*,validate=None):
        if settings!=policy(settings['max_sweeps'],settings['global_rtol']):
            raise ValueError('invalid convergence policy')
        self.run=Path(run);self.policy=settings;self.reason=None;self.validate=validate
        self.path=self.run/'convergence/STATE.json'
        fixed={k:v for k,v in settings.items() if k!='max_sweeps'}
        self.state=(_checked(self.path) if self.path.exists() else
                    dict(schema='field-convergence-state-v1',policy=fixed,records=[]))
        if self.state.get('schema')!='field-convergence-state-v1' or self.state.get('policy')!=fixed:
            raise ValueError('convergence policy changed')
        records=self.state['records']
        if not isinstance(records,list) or any(row.get('sweep')!=i for i,row in enumerate(records)):
            raise ValueError('invalid convergence verification history')

    def _reason(self):
        records=self.state['records'];last=records[-1]
        if last['verification']['global_converged']:return 'VERIFIED_CONVERGED'
        if last['sweep']>=self.policy['max_sweeps']:return 'SWEEP_LIMIT'
        window=self.policy['stagnation_window']
        if len(records)>window:
            before=min(r['verification']['residual_l2'] for r in records[:-window])
            best=min(r['verification']['residual_l2'] for r in records[-window:])
            improvement=max(0.,before-best)/before if before else 0.
            if improvement<self.policy['stagnation_improvement']:return 'STAGNATED'
        return None

    def solve(self,root,shape,chunk,identity,load,geometry,cal,provider,options,stop):
        from . import global_residual
        checkpoint=Path(root)/'CHECKPOINT.json'
        count=math.prod((n-1)//chunk+1 for n in shape)
        records=self.state['records']

        def boundary():
            if stop and stop():
                self.reason='PAUSED';return True
            saved=_read(checkpoint);progress=saved['progress']
            completed=progress.get('completed_sweeps',[])
            if records:
                last=records[-1]
                if len(completed)<last['sweep'] or digest(completed[:last['sweep']])!=last['completed_sha256']:
                    raise ValueError('completed sweep history differs from verification journal')
            if (not progress.get('initialization_done') and not progress.get('solver')
                    or progress.get('solver') and progress['next_block']!=count):
                return False
            if self.validate:self.validate()
            sweep=len(completed);snapshot=digest(dict(chunks=saved['chunks'],progress=progress))
            if records and records[-1]['sweep']==sweep:
                if records[-1]['snapshot_sha256']!=snapshot:
                    raise ValueError('verified checkpoint changed')
            else:
                if sweep!=(records[-1]['sweep']+1 if records else 0):
                    raise ValueError('missing convergence sweep verification')
                sha=file_sha(checkpoint)
                descriptor=dict(kind='chunks',path=str(checkpoint.resolve()),sha256=sha,support_policy='stored')
                with FieldSource(descriptor,geometry,cache_bytes=64<<20) as source:
                    offset=source._load('0-0-0',tuple(min(chunk,n) for n in shape))[1]
                    def read(lo,hi):
                        values,present=source.read(lo,hi)
                        if not present.all():raise ValueError('complete stored field required')
                        return values
                    result=global_residual.verify(shape,chunk,geometry['spacing_xyz'],read,load,
                        constraints=provider,rtol=self.policy['global_rtol'],stop=stop,gauge_offset=offset)
                    source.validate(hashes=True)
                if self.validate:self.validate()
                if result['status']=='PAUSED':
                    self.reason='PAUSED';return True
                if not all(math.isfinite(result[k]) for k in ('residual_l2','rhs_l2','fixed_storage_max_error')):
                    raise RuntimeError('nonfinite global verification')
                result.update(checkpoint_sha256=sha,operator_sha256=digest(identity),pairs_sha256=provider.digest)
                records.append(dict(sweep=sweep,snapshot_sha256=snapshot,checkpoint_sha256=sha,
                    completed_sha256=digest(completed),max_core_update=completed[-1]['max_core_update'] if completed else None,
                    verification=result))
                records[-1]['reason']=self._reason() or 'CONTINUE'
                _save(self.path,self.state)
            self.reason=self._reason()
            return self.reason is not None

        solver=solve_blocks(root,shape,chunk,identity,load,resume=True,stop=boundary,**options)
        # Pin verification through publication, including the core's callback-free last sweep.
        saved=_read(checkpoint)
        with BlockStore(root,shape,chunk,saved['identity'],resume=True,
                        reserve_bytes=options['reserve_bytes'],max_bytes=options.get('max_bytes')):
            if solver['status']=='SWEEP_BUDGET_REACHED':
                boundary()
            if self.reason is None:self.reason='PAUSED'
            saved=_read(checkpoint);sha=file_sha(checkpoint)
            if self.reason!='PAUSED' and sha!=records[-1]['checkpoint_sha256']:
                raise ValueError('verified checkpoint changed before publication')
            result=dict(status='FIELD_READY' if self.reason=='VERIFIED_CONVERGED' else
                        'PAUSED' if self.reason=='PAUSED' else 'NONCONVERGED',
                        reason=self.reason,global_converged=self.reason=='VERIFIED_CONVERGED',
                        physical_acceptance=False,policy=self.policy,geometry=geometry,calibration=cal,
                        checkpoint_sha256=sha,completed_sweeps=saved['progress']['completed_sweeps'],
                        verification=records[-1]['verification'] if records else None,solver=solver)
            if result['global_converged']:
                if self.validate:self.validate()
                path=self.run/'FIELD.json'
                if not path.exists() or _read(path)!=result:_publish(path,result)
        return result


def _root(path):
    path=Path(path).expanduser().absolute()
    if any(p.is_symlink() for p in (path,*path.parents)):
        raise ValueError('symlinks are not allowed in convergence paths')
    return path.resolve()


def _relative(name):
    if (not isinstance(name,str) or not name or Path(name).is_absolute()
            or '..' in Path(name).parts or name!=Path(name).as_posix() or name=='.'):
        raise ValueError('convergence path must be a clean relative name')
    return name


@contextmanager
def _lock(path,*,create=False):
    fd=os.open(path,os.O_NOFOLLOW|(os.O_CREAT|os.O_RDWR if create else os.O_RDONLY),0o600)
    try:
        info=os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):raise ValueError('regular writer lock required')
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield
        if (path.stat().st_dev,path.stat().st_ino)!=(info.st_dev,info.st_ino):
            raise ValueError('writer lock changed')
    finally:os.close(fd)


def _inventory(root,names):
    result={}
    for name in names:
        base=root/_relative(name)
        if any(p.is_symlink() for p in (base,*base.parents) if p!=root and root in p.parents):
            raise ValueError('symlink in convergence inventory')
        for path in sorted(base.rglob('*')) if base.is_dir() else [base]:
            if path.is_symlink():raise ValueError('symlink in convergence inputs')
            if path.is_dir():continue
            if not stat.S_ISREG(path.stat().st_mode):raise ValueError('regular convergence input required')
            result[str(path.relative_to(root))]=dict(sha256=file_sha(path),bytes=path.stat().st_size)
    return result


def _inputs(root):
    from .artifacts import Tiles
    meta=_read(root/'PREPARED.json');fibres=_read(root/'FIBRES.json')
    name='coarse/maps' if meta.get('bounded_preparation') else 'maps'
    if any(p.is_symlink() for p in (root/name,*list((root/name).parents)[:1])):
        raise ValueError('symlink in observation maps')
    maps=Tiles(root/name,readonly=True)
    try:
        if maps.identity()!=meta['tiles_sha256'] or list(maps.shape)!=meta['shape_zyx']:
            raise ValueError('prepared observations differ')
    finally:maps.close()
    if file_sha(root/'fibres/fibres.sqlite')!=fibres.get('sha256'):
        raise ValueError('fibre source differs')
    return _inventory(root,['PREPARED.json','FIBRES.json',name,'fibres/fibres.sqlite'])


def _checkpoint(root,request):
    """Read-only admission against the exact core problem, not merely its shape."""
    from .artifacts import Tiles
    from .config import field_geometry
    from .field import calibration
    from .field_budget import effective_budget
    path=root/'field/CHECKPOINT.json'
    if not path.exists():
        if (root/'convergence/STATE.json').exists():raise ValueError('missing journalled checkpoint')
        return None
    meta=_read(root/'PREPARED.json');config=request['config'];bounded=meta.get('bounded_preparation',False)
    maps=Tiles(root/('coarse/maps' if bounded else 'maps'),readonly=True)
    try:cal=meta['calibration'] if bounded else calibration(maps,meta)
    finally:maps.close()
    if _read(root/'CALIBRATION.json')!=cal:raise ValueError('calibration differs')
    geometry=field_geometry(meta,None if bounded else config.get('field_spacing_um'))
    geometry.update(voxel_um=1.,sheet_step=1.)
    pairs=_read(root/'pairs/PAIRS.json')
    expected=dict(source_sha256=_read(root/'FIBRES.json')['sha256'],grid=geometry,
                  weight=config['fibre_weight'],algorithm='global-arclength-pairs-v1')
    if pairs!=dict(identity=expected,sha256=file_sha(root/'pairs/pairs.sqlite')):
        raise ValueError('global constraint inventory differs')
    identity=dict(maps=meta['tiles_sha256'],fibres=expected['source_sha256'],geometry=geometry,calibration=cal,
        fibre_weight=config['fibre_weight'],algorithm='global-arclength-pairs-coarse-prior-v1' if bounded else
        'global-arclength-pairs-full-prior-v1',pairs=file_sha(root/'pairs/PAIRS.json'),mode='automatic-convergence-v1')
    saved=_read(path);chunk=config['field_chunk'];shape=geometry['shape_zyx']
    order=list(itertools.product(*(range((n-1)//chunk+1) for n in shape)))
    budget=effective_budget(root/'field',config['sweeps'])
    if budget not in (request['policy']['max_sweeps'],request.get('previous_total')):
        raise ValueError('convergence sweep budget differs')
    import hashlib
    problem=dict(schema='winding-block-solve-v1',source_identity=identity,shape=shape,chunk=chunk,
        origin_xyz=geometry['origin_xyz'],spacing_xyz=geometry['spacing_xyz'],
        order_sha256=hashlib.sha256(np.asarray(order,dtype='<i8').tobytes()).hexdigest(),
        constraints='caller-bound-provider',boundary='natural-zero-gauge',halo=2,sweeps=budget,
        rtol=config['rtol'],maxiter=config['maxiter'],dtype='float64',device=config['device'])
    if saved['identity']!=problem or saved['shape']!=shape or saved['chunk']!=chunk:
        raise ValueError('automatic checkpoint scientific configuration differs')
    with FieldSource(dict(kind='chunks',path=str(path),sha256=file_sha(path),support_policy='stored'),geometry) as source:
        source.validate(hashes=True)
    progress=saved['progress'];completed=progress.get('completed_sweeps')
    count=len(order);sweep=progress.get('sweep');block=progress.get('next_block')
    if (type(sweep) is not int or not 1<=sweep<=budget or type(block) is not int or not 0<=block<=count
            or not isinstance(completed,list) or len(completed)!=sweep-int(block<count)
            or any(r.get('sweep')!=i or r.get('cores')!=count for i,r in enumerate(completed,1))
            or (progress.get('solver') or progress.get('initialization_done')) and len(saved['chunks'])!=count):
        raise ValueError('invalid automatic checkpoint progress')
    journal=Controller(root,request['policy']).state['records']
    if progress.get('solver') and not journal:raise ValueError('missing convergence verification journal')
    if journal:
        last=journal[-1]
        if (len(completed)<last['sweep'] or digest(completed[:last['sweep']])!=last['completed_sha256']
                or len(completed)>last['sweep']+1):
            raise ValueError('completed sweep history differs')
        if len(completed)==last['sweep'] and (not progress.get('solver') or block==count):
            if last['snapshot_sha256']!=digest(dict(chunks=saved['chunks'],progress=progress)):
                raise ValueError('verified checkpoint differs')
    return saved


def _request(root):
    from .workflow import implementation_sha
    for name in ('maps','coarse','coarse/maps','fibres','pairs','field','convergence','source'):
        path=root/name
        if path.is_symlink() or path.exists() and not path.is_dir():
            raise ValueError('invalid managed convergence directory: '+name)
    request=_checked(root/'CONVERGENCE.json')
    if request.get('schema')!='field-convergence-v1':raise ValueError('unknown convergence generation')
    if request['implementation_sha256']!=implementation_sha():raise ValueError('implementation changed; original package required')
    if request['policy']!=policy(request['policy']['max_sweeps'],request['policy']['global_rtol']):
        raise ValueError('invalid saved convergence policy')
    from .config import validate_config
    validate_config(request['config'])
    if set(request['config'])-CONFIG_KEYS-{'sweeps'}:raise ValueError('unknown saved field settings')
    for entries in (request['inputs'],request['files']):
        for name,entry in entries.items():
            _relative(name)
            if 'source' in entry:_relative(entry['source'])
            if (type(entry['bytes']) is not int or entry['bytes']<0
                    or not isinstance(entry['sha256'],str) or re.fullmatch('[0-9a-f]{64}',entry['sha256']) is None):
                raise ValueError('invalid convergence file inventory')
    return request


def _generation(root):
    request=_request(root)
    if _inputs(root)!=request['inputs']:raise ValueError('generation inputs changed')
    provenance={k:{p:v for p,v in entry.items() if p!='source'}
                for k,entry in request['files'].items() if k.startswith('source/')}
    if _inventory(root,list(provenance))!=provenance:raise ValueError('source provenance changed')
    checkpoint=_checkpoint(root,request)
    result=_checked(root/'CONVERGENCE_RESULT.json') if (root/'CONVERGENCE_RESULT.json').exists() else None
    if result is not None and result['request_sha256']!=digest(request):raise ValueError('result request differs')
    if (root/'FIELD.json').exists():
        published=_read(root/'FIELD.json')
        records=Controller(root,request['policy']).state['records']
        if (not records or not published.get('global_converged') or published.get('physical_acceptance') is not False
                or published.get('checkpoint_sha256')!=file_sha(root/'field/CHECKPOINT.json')
                or published.get('verification')!=records[-1]['verification']
                or published.get('reason')!='VERIFIED_CONVERGED' or published.get('policy')!=request['policy']
                or published.get('geometry')!=checkpoint['identity']['source_identity']['geometry']
                or published.get('calibration')!=checkpoint['identity']['source_identity']['calibration']
                or published.get('completed_sweeps')!=checkpoint['progress']['completed_sweeps']
                or published['checkpoint_sha256']!=published['verification']['checkpoint_sha256']
                or result is not None and result['result']['reason']!='PAUSED'
                   and result.get('field_sha256')!=file_sha(root/'FIELD.json')):
            raise ValueError('published automatic field differs')
        if result is None or result['result']['reason']=='PAUSED':
            # FIELD is the publication commit; finish a missing/stale summary without solving.
            result=dict(request_sha256=digest(request),result=published,
                        state_sha256=file_sha(root/'convergence/STATE.json'),field_sha256=file_sha(root/'FIELD.json'))
    elif result is not None and result.get('field_sha256'):
        raise ValueError('missing published automatic field')
    if result is not None and result['result']['reason'] in ('VERIFIED_CONVERGED','SWEEP_LIMIT','STAGNATED'):
        if (result['result']['checkpoint_sha256']!=file_sha(root/'field/CHECKPOINT.json')
                or result['state_sha256']!=file_sha(root/'convergence/STATE.json')):
            raise ValueError('completed convergence result changed')
    return request,result


def _copy(root,source,files,reserve):
    with _directory(root) as staging:
        for name,entry in files.items():
            target=Path(_relative(name))
            with _directory(root/target.parent,create=True) as directory:
                try:existing=os.open(target.name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=directory)
                except FileNotFoundError:existing=None
                if existing is not None:
                    with os.fdopen(existing,'rb') as stream:
                        info=os.fstat(stream.fileno())
                        if (not stat.S_ISREG(info.st_mode) or info.st_size!=entry['bytes']
                                or hashlib.file_digest(stream,'sha256').hexdigest()!=entry['sha256']):
                            raise ValueError('retained admission copy differs')
                        if info.st_nlink!=1:
                            aliases=[]
                            for candidate in os.listdir(staging):
                                if re.fullmatch(r'\.copy-[0-9a-f]{32}',candidate):
                                    other=os.stat(candidate,dir_fd=staging,follow_symlinks=False)
                                    if (other.st_dev,other.st_ino)==(info.st_dev,info.st_ino):aliases.append(candidate)
                            if info.st_nlink!=1+len(aliases):
                                raise ValueError('admission copy has an external writable alias')
                            # Recover only our exact staging inode after a link/unlink crash.
                            for candidate in aliases:os.unlink(candidate,dir_fd=staging)
                            os.fsync(staging)
                            if os.fstat(stream.fileno()).st_nlink!=1:
                                raise ValueError('admission copy alias changed')
                    continue
                if shutil.disk_usage(root).free<reserve+entry['bytes']+65536:
                    raise OSError('insufficient space for private convergence copy and reserve')
                # Orphan temporaries live outside immutable input inventories.
                temporary='.copy-'+secrets.token_hex(16)
                fd=os.open(temporary,os.O_RDWR|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=staging)
                try:
                    with os.fdopen(fd,'w+b') as stream:
                        with (source/entry['source']).open('rb') as original:shutil.copyfileobj(original,stream)
                        stream.flush();os.fsync(stream.fileno());stream.seek(0)
                        if hashlib.file_digest(stream,'sha256').hexdigest()!=entry['sha256']:
                            raise ValueError('source changed during admission copy')
                    # Exclusive publication must not overwrite a newly appeared destination.
                    os.link(temporary,target.name,src_dir_fd=staging,dst_dir_fd=directory,follow_symlinks=False)
                    os.fsync(directory)
                finally:
                    os.unlink(temporary,dir_fd=staging)
                    os.fsync(staging)


def _admit(source,max_sweeps,global_rtol,config):
    from .workflow import implementation_sha
    if (source/'CONVERGENCE.json').exists():
        old,result=_generation(source)
        if result is None or result['result']['reason']!='SWEEP_LIMIT':
            raise ValueError('new generation requires an exhausted automatic field; paused work uses --resume')
        settings=policy(max_sweeps,old['policy']['global_rtol'] if global_rtol is None else global_rtol)
        if (settings['max_sweeps']<=old['policy']['max_sweeps']
                or {k:v for k,v in settings.items() if k!='max_sweeps'}!=
                   {k:v for k,v in old['policy'].items() if k!='max_sweeps'}
                or any(old['config'].get(k)!=v for k,v in config.items())):
            raise ValueError('continuation requires a higher total budget and unchanged scientific settings')
        effective=old['config'];inputs=old['inputs']
        checkpoint=_read(source/'field/CHECKPOINT.json')
        names=list(inputs)+['CALIBRATION.json','pairs','convergence/STATE.json','field/CHECKPOINT.json']
        names+=['field/'+r['path'] for r in checkpoint['chunks'].values()]
        copies=_inventory(source,names)
        provenance=['CONVERGENCE.json','CONVERGENCE_RESULT.json']
        previous=old['policy']['max_sweeps']
    else:
        if (source/'field').exists() or (source/'FIELD.json').exists():
            raise ValueError('legacy field is not migrated; keep using its existing route')
        settings=policy(max_sweeps,1e-6 if global_rtol is None else global_rtol)
        state=_read(source/'RUN.json')
        if state.get('schema')!='winding-field-run-v1':raise ValueError('prepared workflow RUN required; recovery migration unsupported')
        for stage,name in (('prepare','PREPARED.json'),('fibres','FIBRES.json')):
            entry=state.get('stages',{}).get(stage,{})
            if entry.get('receipt')!=name or entry.get('sha256')!=file_sha(source/name):
                raise ValueError('completed prepared/fibres receipt required')
        effective=dict(DEFAULTS,**{k:v for k,v in state['config'].items() if k in CONFIG_KEYS})
        effective.update(config);effective['sweeps']=settings['max_sweeps']
        inputs=_inputs(source);copies=dict(inputs);provenance=['RUN.json'];previous=None
    from .config import validate_config
    validate_config(effective)
    if set(config)-CONFIG_KEYS:raise ValueError('unknown field settings')
    files={name:dict(entry,source=name) for name,entry in copies.items()}
    for name in provenance:
        files['source/'+name]=dict(_inventory(source,[name])[name],source=name)
    return dict(schema='field-convergence-v1',implementation_sha256=implementation_sha(),source=str(source),
                inputs=inputs,files=files,config=effective,policy=settings,previous_total=previous)


def converge(source,output=None,*,resume=False,max_sweeps=None,global_rtol=None,config=None,stop=None):
    """Start from prepared inputs, resume automatic work, or fork an exhausted generation."""
    from .field import solve
    from .field_budget import effective_budget,extend
    config=config or {};source=_root(source)
    root=source if resume else _root(output) if output is not None else None
    if root is None:raise ValueError('new automatic field requires --output')
    if resume and output is not None:raise ValueError('--resume takes the generation as SOURCE, without --output')
    if not resume:
        if root==source or source in root.parents or root in source.parents:
            raise ValueError('output must be separate from its source')
        if root.exists():raise FileExistsError('output exists; use --resume or a new output')
        with _lock(source/'.run.lock'):
            with _lock(source/'field/.writer.lock') if (source/'CONVERGENCE.json').exists() else nullcontext():
                request=_admit(source,max_sweeps,global_rtol,config)
                root.mkdir(parents=True,exist_ok=False)
                with _lock(root/'.run.lock',create=True):
                    _save(root/'CONVERGENCE.json',request)
                    _copy(root,source,request['files'],request['config']['reserve_bytes'])
                    if (root/'field').exists():
                        with _lock(root/'field/.writer.lock',create=True):pass
                    _save(root/'INPUTS_READY.json',dict(request_sha256=digest(request)))
    with _lock(root/'.run.lock'):
        request=_request(root)
        if (max_sweeps is not None and max_sweeps!=request['policy']['max_sweeps']
                or global_rtol is not None and global_rtol!=request['policy']['global_rtol']
                or any(request['config'].get(k)!=v for k,v in config.items())):
            raise ValueError('saved convergence policy/configuration changed; resume cannot reset limits')
        if not (root/'INPUTS_READY.json').exists():
            # A crash during admission copies no solver work; retry only the pinned bytes.
            original=_root(request['source'])
            with _lock(original/'.run.lock'):
                _copy(root,original,request['files'],request['config']['reserve_bytes'])
            if (root/'field').exists():
                with _lock(root/'field/.writer.lock',create=True):pass
            _save(root/'INPUTS_READY.json',dict(request_sha256=digest(request)))
        if _checked(root/'INPUTS_READY.json')!=dict(request_sha256=digest(request)):
            raise ValueError('admission identity differs')
        with _lock(root/'field/.writer.lock') if (root/'field').exists() else nullcontext():
            request,previous=_generation(root)
        if previous is not None and previous['result']['reason'] in ('VERIFIED_CONVERGED','SWEEP_LIMIT','STAGNATED'):
            path=root/'CONVERGENCE_RESULT.json'
            if not path.exists() or _checked(path)!=previous:_save(path,previous)
            return previous['result']
        if request['previous_total'] is not None:
            budget=effective_budget(root/'field',request['config']['sweeps'])
            if budget<request['policy']['max_sweeps']:
                extend(root/'field',request['policy']['max_sweeps'],request['config']['sweeps'],
                       reserve_bytes=request['config']['reserve_bytes'])
        def validate():
            if _inputs(root)!=request['inputs']:raise ValueError('generation inputs changed during solve')
        controller=Controller(root,request['policy'],validate=validate)
        import torch
        torch.set_num_threads(request['config']['threads'])
        cancelled=lambda:(root/'STOP').exists() or (stop is not None and stop())
        try:
            result=solve(root,request['config'],cancelled,convergence=controller)
        except (ValueError,RuntimeError,OSError) as error:
            reason=('INTEGRITY_ERROR' if isinstance(error,ValueError) else 'RESOURCE_ERROR'
                    if isinstance(error,(OSError,ResourceUnavailable,StorageBudgetError)) else 'NUMERICAL_ERROR')
            _save(root/'CONVERGENCE_ERROR.json',dict(reason=reason,error=str(error),
                  global_converged=False,physical_acceptance=False))
            raise
        if result.get('reason') is None:
            result.update(reason='PAUSED',global_converged=False,physical_acceptance=False,policy=request['policy'])
        _save(root/'CONVERGENCE_RESULT.json',dict(request_sha256=digest(request),result=result,
              state_sha256=file_sha(controller.path) if controller.path.exists() else None,
              field_sha256=file_sha(root/'FIELD.json') if (root/'FIELD.json').exists() else None))
        return result


def main(argv=None):
    from .cli_settings import parse_options
    from .cli import resolve_device
    parser=argparse.ArgumentParser(prog='winding-field field-converge',
        description='Opt-in globally verified field-only solves. Legacy fields are not migrated. No meshes/flattening run.')
    parser.add_argument('source',help='prepared workflow without a field, or automatic generation')
    parser.add_argument('--output',help='fresh private generation outside SOURCE; omit with --resume')
    parser.add_argument('--resume',action='store_true',help='resume SOURCE with unchanged total limits; remove SOURCE/STOP first')
    parser.add_argument('--max-sweeps',type=int,help='required for new generation; absolute total including inherited sweeps')
    parser.add_argument('--global-rtol',type=float,help='global stored-field normal-equation tolerance (new default: 1e-6)')
    group=parser.add_argument_group('new field overrides (otherwise inherited from source)',argument_default=argparse.SUPPRESS)
    for name,kind in (('field_chunk',int),('field_spacing_um',float),('rtol',float),('maxiter',int),
                      ('fibre_weight',float),('device',str),('threads',int),('reserve_bytes',int),
                      ('bounded_bytes',int),('prepared_cache_bytes',int),('constraint_cache_bytes',int)):
        group.add_argument('--'+name.replace('_','-'),type=kind)
    args=parse_options(parser,argv,paths=('source','output'))
    values=vars(args);config={k:values[k] for k in CONFIG_KEYS if k in values}
    if config.get('device')=='auto':config['device']=resolve_device('auto')
    interrupted=False
    def handle(signum,frame):
        nonlocal interrupted
        interrupted=True
    previous={sig:signal.signal(sig,handle) for sig in (signal.SIGINT,signal.SIGTERM)}
    try:
        result=converge(args.source,args.output,resume=args.resume,max_sweeps=args.max_sweeps,
                        global_rtol=args.global_rtol,config=config,stop=lambda:interrupted)
        print(json.dumps(result,indent=2,sort_keys=True,allow_nan=False))
        return 0 if result['reason']=='VERIFIED_CONVERGED' else 130 if result['reason']=='PAUSED' else 2
    except (ValueError,RuntimeError,OSError,KeyError,TypeError) as error:
        print(str(error),file=sys.stderr);return 1
    finally:
        for sig,handler in previous.items():signal.signal(sig,handler)
