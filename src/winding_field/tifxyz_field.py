"""Physical, ROI-bounded view of the unchanged SHA-bound field reader. No torch."""
from copy import deepcopy
import hashlib
import itertools
import math
import os
from pathlib import Path
import stat
import shutil

import numpy as np

from .surface_sources import FieldSource
from .tifxyz_contract import Blocked,Paused,parse_json,triple,validate_frame
from .tifxyz_runtime import metadata_charge
from .util import atomic_json,digest,file_sha

MAX_METADATA=16<<20


def read_metadata(path):
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    with os.fdopen(fd,'rb') as stream:
        before=os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size>MAX_METADATA:raise ValueError('metadata envelope exceeded')
        raw=stream.read(MAX_METADATA+1);after=os.fstat(stream.fileno())
    if len(raw)>MAX_METADATA or (before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(after.st_size,after.st_mtime_ns,after.st_ctime_ns):
        raise ValueError('metadata changed')
    value=parse_json(raw)
    if not isinstance(value,dict):raise ValueError('metadata object required')
    return value,dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


def source_binding(root,checkpoint_record=None):
    run,run_record=read_metadata(root/'RUN.json');field,field_record=read_metadata(root/'FIELD.json')
    stage=run.get('stages',{}).get('field',{})
    if stage.get('receipt')!='FIELD.json' or stage.get('sha256')!=field_record['sha256'] or field.get('status')!='FIELD_READY':
        raise ValueError('field stage receipt binding differs')
    geometry=field['geometry'];source=run['source']
    if digest(source)!=geometry['frame']:raise ValueError('source frame binding differs')
    charge=0
    if checkpoint_record is None:
        checkpoint,cp_record=read_metadata(root/'field/CHECKPOINT.json')
        charge=metadata_charge(cp_record['bytes'],len(checkpoint['chunks']))
    else:
        cp_record=checkpoint_record
    if cp_record['sha256']!=field['checkpoint_sha256']:raise ValueError('checkpoint binding differs')
    stable=dict(producer_schema=run['schema'],producer_implementation=run['implementation_sha256'],source=source,
                field_receipt=dict(receipt='FIELD.json',sha256=field_record['sha256']),geometry=geometry,
                checkpoint=cp_record,support_policy='observed')
    return stable,dict(run=run,record=run_record),charge


def selected_frame(binding):
    source=binding['source'];spacing=triple(source['spacing_xyz'],'source spacing',minimum=0)
    if not spacing[0]>0 or spacing!=(spacing[0],)*3:raise ValueError('selected output frame is not isotropic')
    return validate_frame(dict(id=binding['geometry']['frame'],origin_xyz_um=source['origin_xyz'],
        voxel_um=spacing[0],shape_zyx=source['shape'],source_identity_sha256=digest(source)))


class PhysicalField:
    def __init__(self,source,*,roi=None,binding_check=None,identity=None):
        self.source=source;self._binding_check=binding_check
        shape=tuple(source.shape)
        self.lo=(0,0,0) if roi is None else triple(roi['lo_zyx'],'ROI low',integer=True,minimum=0)
        hi=shape if roi is None else triple(roi['hi_zyx'],'ROI high',integer=True,minimum=0)
        if any(b-a<2 or b>n for a,b,n in zip(self.lo,hi,shape)):raise ValueError('ROI outside field')
        self.shape=tuple(b-a for a,b in zip(self.lo,hi))
        self.spacing_um=np.asarray(source.spacing,float)*source.voxel_um
        self.origin_um=np.asarray(source.origin,float)*source.voxel_um+np.array(self.lo[::-1])*self.spacing_um
        self._identity=dict(source=deepcopy(source.identity if identity is None else identity),
                            roi=dict(lo_zyx=list(self.lo),hi_zyx=list(hi)))
        self.stats=dict(read_calls=0,crop_bytes=0)

    @property
    def identity(self):return deepcopy(self._identity)

    def validate(self,hashes=False):
        if self._binding_check:self._binding_check()
        self.source.validate(hashes=hashes)

    def read(self,lo_zyx,hi_zyx):
        lo=triple(list(lo_zyx),'read low',integer=True,minimum=0);hi=triple(list(hi_zyx),'read high',integer=True,minimum=0)
        if any(b<=a or b>n for a,b,n in zip(lo,hi,self.shape)):raise ValueError('read outside ROI')
        if self._binding_check:self._binding_check()
        values,valid=self.source.read([a+b for a,b in zip(lo,self.lo)],[a+b for a,b in zip(hi,self.lo)])
        self.stats['read_calls']+=1;self.stats['crop_bytes']+=values.nbytes+valid.nbytes
        return values,valid

    def read_cells(self,cells):
        cells=np.asarray(cells)
        if cells.ndim!=2 or cells.shape[1]!=3 or cells.dtype.kind not in 'iu' or len(cells)>16384:
            raise ValueError('bounded integer cell batch required')
        if (cells<0).any() or (cells>=np.array(self.shape)-1).any():raise ValueError('cell outside ROI')
        values=np.empty((len(cells),2,2,2),float);support=np.empty(len(cells),bool)
        groups,inverse=np.unique(cells//64,axis=0,return_inverse=True)
        for i,key in enumerate(groups):
            selected=np.flatnonzero(inverse==i);lo=key*64;hi=np.minimum(lo+65,self.shape)
            crop,valid=self.read(lo.tolist(),hi.tolist());local=cells[selected]-lo
            supported=np.ones(len(selected),bool)
            for bits in itertools.product((0,1),repeat=3):
                index=(local+bits).T
                values[(selected,*bits)]=crop[tuple(index)]
                supported&=valid[tuple(index)]
            support[selected]=supported
        return values,support

    def sample(self,xyz_um):
        xyz=np.asarray(xyz_um,float)
        if xyz.ndim!=2 or xyz.shape[1]!=3 or len(xyz)>16384 or not np.isfinite(xyz).all():raise ValueError('bounded finite XYZ batch required')
        grid=((xyz-self.origin_um)/self.spacing_um)[:,::-1]
        rounding=8*np.finfo(float).eps*np.maximum(np.maximum(abs(xyz),abs(self.origin_um)),1)/self.spacing_um
        inside=((grid>=-rounding[:,::-1])&(grid<=np.array(self.shape)-1+rounding[:,::-1])).all(1)
        grid=np.clip(grid,0,np.array(self.shape)-1)
        result=np.zeros(len(xyz));valid=np.zeros(len(xyz),bool);ids=np.flatnonzero(inside)
        if len(ids):
            cells=np.minimum(np.floor(grid[ids]).astype(np.int64),np.array(self.shape)-2)
            values,support=self.read_cells(cells);local=grid[ids]-cells
            for bits in itertools.product((0,1),repeat=3):
                weight=np.prod(np.where(bits,local,1-local),axis=1)
                result[ids]+=weight*values[(slice(None),*bits)]
            valid[ids]=support&np.isfinite(result[ids])
        return result,valid

    def close(self):self.source.close()
    def __enter__(self):return self
    def __exit__(self,*unused):self.close()


def open_physical_field(field_run,cache_bytes=128<<20,*,roi=None):
    root=Path(field_run)
    if root.is_symlink():raise ValueError('symlink field run')
    root=root.resolve(strict=True);binding,snapshot,charge=source_binding(root)
    def check():
        current,_,_=source_binding(root,binding['checkpoint'])
        if current!=binding:raise ValueError('stable source binding changed')
    source=FieldSource(dict(kind='chunks',path=str(root/'field/CHECKPOINT.json'),
        sha256=binding['checkpoint']['sha256'],support_policy='observed'),binding['geometry'],cache_bytes=cache_bytes)
    try:
        result=PhysicalField(source,roi=roi,binding_check=check,identity=binding)
        result.producer_snapshot=snapshot;result.binding=binding;result.metadata_bytes=charge
        return result
    except BaseException:
        source.close();raise


def build_coarse(field,stride,out,stop=None,*,budget=None):
    if type(stride) is not int or stride<2:raise ValueError('guide stride must be >=2')
    shape=tuple((n-2)//stride+2 for n in field.shape)
    if math.prod(shape)>4_000_000:raise Blocked('GUIDE_VOXEL_LIMIT')
    tile_shape=tuple((n-2)//64+1 for n in field.shape);tile_count=math.prod(tile_shape)
    array_bytes=8*math.prod(shape)+math.prod(tuple(n-1 for n in shape))+8*sum(shape)+65536
    cap=array_bytes+512*tile_count+65536
    limit=(16<<30) if budget is None else budget.scratch_bytes
    reserve=0 if budget is None else budget.reserve_disk_bytes
    if cap>limit:raise Blocked('SCRATCH_BUDGET','coarse arrays and range sidecar exceed allowance')
    out=Path(out)
    if shutil.disk_usage(out.parent).free<reserve+cap:raise Blocked('DISK_HEADROOM')
    axes=[np.r_[np.arange(0,n-1,stride,dtype=np.int64),n-1] for n in field.shape]
    values=np.full(shape,np.nan);cells=np.ones(tuple(n-1 for n in shape),bool)
    out.mkdir(exist_ok=False);field.validate();written=count=0;checksum=hashlib.sha256()
    import json
    with (out/'ranges.jsonl').open('xb',buffering=65536) as ranges:
        for key in itertools.product(*(range(n) for n in tile_shape)):
            if stop and stop():raise Paused()
            if shutil.disk_usage(out).free<reserve+cap-written:raise Blocked('DISK_HEADROOM')
            lo=np.array(key)*64;hi=np.minimum(lo+64,np.array(field.shape)-1)
            data,valid=field.read(lo.tolist(),(hi+1).tolist())
            valid=valid&np.isfinite(data)
            first=[int(np.searchsorted(a,b)) for a,b in zip(axes,lo)]
            last=[int(np.searchsorted(a,b,side='right')) for a,b in zip(axes,hi)]
            if all(b>a for a,b in zip(first,last)):
                samples=[axis[a:b]-l for axis,a,b,l in zip(axes,first,last,lo)]
                values[tuple(slice(a,b) for a,b in zip(first,last))]=data[np.ix_(*samples)]
            bad=np.argwhere(~valid)+lo
            if len(bad):
                lower=np.column_stack([np.searchsorted(a,bad[:,i],side='left')-1 for i,a in enumerate(axes)])
                upper=np.column_stack([np.searchsorted(a,bad[:,i],side='right')-1 for i,a in enumerate(axes)])
                for bits in itertools.product((0,1),repeat=3):
                    index=np.where(bits,upper,lower);inside=((index>=0)&(index<np.array(cells.shape))).all(1)
                    cells[tuple(index[inside].T)]=False
            row=dict(lo_zyx=lo.tolist(),hi_zyx=(hi+1).tolist(),minimum=float(data[valid].min()) if valid.any() else None,
                     maximum=float(data[valid].max()) if valid.any() else None,all_supported=bool(valid.all()))
            raw=(json.dumps(row,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
            if len(raw)>512:raise ValueError('coarse range row exceeds declared envelope')
            ranges.write(raw);checksum.update(raw);written+=len(raw);count+=1
        ranges.flush();os.fsync(ranges.fileno())
    field.validate()
    artifacts={}
    for name,array in [('values',values),('cells',cells),*[(f'axis_{i}',a) for i,a in enumerate(axes)]]:
        path=out/(name+'.npy')
        with path.open('xb') as stream:np.save(stream,array,allow_pickle=False);stream.flush();os.fsync(stream.fileno())
        artifacts[name]=dict(bytes=path.stat().st_size,sha256=file_sha(path),shape=list(array.shape),dtype=array.dtype.str)
    receipt=dict(root=str(out.resolve()),stride=stride,shape_zyx=list(shape),origin_um=field.origin_um.tolist(),
                 spacing_um=field.spacing_um.tolist(),source=field.identity,artifacts=artifacts,fine_shape_zyx=list(field.shape),
                 tile_ranges=dict(path='ranges.jsonl',bytes=written,count=count,sha256=checksum.hexdigest()),
                 tail_policy='explicit final native node; shortened last coarse interval')
    atomic_json(out/'COARSE.json',receipt)
    return receipt


def load_coarse(receipt):
    root=Path(receipt['root'])
    recorded,_=read_metadata(root/'COARSE.json')
    if recorded!=receipt:raise ValueError('coarse receipt differs')
    for _ in iter_coarse_ranges(receipt):pass
    arrays=[]
    for name in ('values','cells','axis_0','axis_1','axis_2'):
        record=receipt['artifacts'][name];path=root/(name+'.npy')
        if path.is_symlink() or path.stat().st_size!=record['bytes'] or file_sha(path)!=record['sha256']:raise ValueError('coarse artifact differs')
        with path.open('rb') as stream:
            version=np.lib.format.read_magic(stream)
            if version!=(1,0):raise ValueError('unexpected coarse array format')
            shape,fortran,dtype=np.lib.format.read_array_header_1_0(stream)
            if (list(shape)!=record['shape'] or dtype.str!=record['dtype'] or fortran
                or math.prod(shape)>4_000_000 or dtype.hasobject or math.prod(shape)*dtype.itemsize+stream.tell()!=record['bytes']):
                raise ValueError('invalid coarse header')
        arrays.append(np.load(path,allow_pickle=False))
    values,cells,*axes=arrays
    if values.shape!=tuple(receipt['shape_zyx']) or cells.shape!=tuple(n-1 for n in values.shape) or cells.dtype!=bool:
        raise ValueError('invalid coarse geometry')
    return values,cells,axes


def iter_coarse_ranges(receipt):
    descriptor=receipt['tile_ranges'];shape=tuple(receipt['fine_shape_zyx'])
    counts=tuple((n-2)//64+1 for n in shape)
    if (not isinstance(descriptor,dict) or descriptor.get('path')!='ranges.jsonl'
        or descriptor.get('count')!=math.prod(counts) or type(descriptor.get('bytes')) is not int
        or not 0<descriptor['bytes']<=512*descriptor['count']):raise ValueError('invalid coarse range descriptor')
    path=Path(receipt['root'])/'ranges.jsonl'
    if path.is_symlink() or not path.is_file() or path.stat().st_size!=descriptor['bytes']:raise ValueError('coarse range size differs')
    with path.open('rb') as stream:
        before=os.fstat(stream.fileno());signature=(before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)
        if hashlib.file_digest(stream,'sha256').hexdigest()!=descriptor['sha256']:raise ValueError('coarse range checksum differs')
        stream.seek(0)
        for key in itertools.product(*(range(n) for n in counts)):
            raw=stream.readline(513)
            if not raw or len(raw)>512:raise ValueError('invalid coarse range row')
            row=parse_json(raw);low=[k*64 for k in key];high=[min(a+65,n) for a,n in zip(low,shape)]
            if (set(row)!={'lo_zyx','hi_zyx','minimum','maximum','all_supported'} or row['lo_zyx']!=low or row['hi_zyx']!=high
                or type(row['all_supported']) is not bool):raise ValueError('coarse range ownership differs')
            a,b=row['minimum'],row['maximum']
            if a is None or b is None:
                if a is not None or b is not None or row['all_supported']:raise ValueError('invalid unsupported range')
            elif type(a) not in (int,float) or type(b) not in (int,float) or not math.isfinite(a) or not math.isfinite(b) or a>b:
                raise ValueError('invalid coarse range extrema')
            yield row
        if stream.read(1):raise ValueError('extra coarse range rows')
        after=os.fstat(stream.fileno());current=path.stat()
        if any((s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)!=signature for s in (after,current)):
            raise ValueError('coarse range changed')
