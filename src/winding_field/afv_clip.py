"""Bounded, invocation-owned support snapshots for exact AFV clipping."""
from collections import OrderedDict


class SupportTiles:
    def __init__(self,maps,*,cache_bytes):
        if type(cache_bytes) is not int or cache_bytes<0:raise ValueError('invalid support cache budget')
        self.maps=maps;self.shape=maps.shape;self.chunk=maps.chunk
        self.limit=cache_bytes;self.bytes=0;self.cache=OrderedDict()

    def load(self,key,names=('support',)):
        if names!=('support',):raise ValueError('support cache only reads support')
        key=tuple(key)
        if key in self.cache:
            self.cache.move_to_end(key);return self.cache[key]
        data=self.maps.load(key,names)
        for array in data.values():array.flags.writeable=False
        charge=512+sum(a.nbytes for a in data.values())
        if charge<=self.limit:
            while self.bytes+charge>self.limit:
                _,old=self.cache.popitem(last=False);self.bytes-=512+sum(a.nbytes for a in old.values())
            self.cache[key]=data;self.bytes+=charge
        return data

    def close(self):
        self.cache.clear();self.bytes=0
