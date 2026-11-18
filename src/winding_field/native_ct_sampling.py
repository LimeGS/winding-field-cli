"""Float64 native interpolation over verified chunks with exact support masks."""
from itertools import product
import numpy as np


class NativeCTSampler:
    def __init__(self,shape,chunks,dtype,chunk):
        from vc_reader.reader import ZarrReader
        self.shape=np.asarray(shape);self.chunks=np.asarray(chunks);self.chunk=chunk
        # The caller owns the bounded, hash-verified cache; native reads use it too.
        self.reader=ZarrReader.computed(chunk,shape=shape,chunks=chunks,dtype=np.dtype(dtype).str,
                                       memory_cache=0,threads=1)

    def sample(self,xyz,*,prefetch=None):
        xyz=np.asarray(xyz)
        if xyz.ndim!=2 or xyz.shape[1]!=3 or len(xyz)>100000:raise ValueError('expected <=100000 XYZ points')
        points=xyz[:,::-1].astype(np.float64)
        valid=np.isfinite(points).all(1)&(points>=0).all(1)&(points<=self.shape-1).all(1)
        ids=np.flatnonzero(valid);lower=np.floor(points[ids]).astype(np.int64);fraction=points[ids]-lower
        grid=(self.shape+self.chunks-1)//self.chunks;seen={}

        def check(selected,indices):
            keys=np.ravel_multi_index(indices.T,grid);unique=np.unique(keys)
            unseen=[int(k) for k in unique if int(k) not in seen]
            for start in range(0,len(unseen),4):
                batch=unseen[start:start+4]
                indices=[tuple(map(int,np.unravel_index(k,grid))) for k in batch]
                if prefetch is not None:prefetch(indices)
                for key,index in zip(batch,indices):
                    data=self.chunk(index)
                    if data is not None and data.shape!=tuple(self.chunks):raise ValueError('malformed CT chunk')
                    seen[key]=data is not None
            missing=[int(k) for k in unique if not seen[int(k)]]
            if missing:valid[selected[np.isin(keys,missing)]]=False

        check(ids,lower//self.chunks)
        crossing=(lower%self.chunks==self.chunks-1)&(fraction>0)
        border=np.flatnonzero(crossing.any(1))
        lower,fraction,crossing,ids=lower[border],fraction[border],crossing[border],ids[border]
        for corner in product((0,1),repeat=3):
            active=np.flatnonzero((crossing&corner).any(1)&(np.prod(np.where(corner,fraction,1-fraction),axis=1)>0))
            if len(active):check(ids[active],(lower[active]+corner)//self.chunks)
        values=np.full(len(points),np.nan,np.float32)
        if valid.any():
            selected=np.flatnonzero(valid)
            sampled=self.reader.sample(points[selected],dtype=np.float64)
            observed=np.isfinite(sampled)&(np.abs(sampled)<=np.finfo(np.float32).max)
            valid[selected[~observed]]=False
            values[selected[observed]]=sampled[observed]
        return values,valid

    def close(self):
        self.reader.close()
