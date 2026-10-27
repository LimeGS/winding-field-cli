import itertools
import json

import numpy as np


def write_ct(root,data,spacing=(1.,1.,1.),origin=(0.,0.,0.),chunk=32):
    root.mkdir(parents=True)
    (root/'0').mkdir()
    (root/'.zattrs').write_text(json.dumps({'multiscales':[{'axes':[
        {'name':a,'type':'space','unit':'micrometer'} for a in 'zyx'],
        'datasets':[{'path':'0','coordinateTransformations':[
            {'type':'scale','scale':list(spacing)[::-1]},
            {'type':'translation','translation':list(origin)[::-1]}]}]}]}))
    (root/'0/.zarray').write_text(json.dumps(dict(zarr_format=2,shape=list(data.shape),chunks=[chunk]*3,
        dtype=data.dtype.str,compressor=None,fill_value=0,filters=None,order='C',dimension_separator='.')))
    for key in itertools.product(*(range((n-1)//chunk+1) for n in data.shape)):
        lo=np.array(key)*chunk; hi=np.minimum(lo+chunk,data.shape)
        array=np.zeros((chunk,)*3,data.dtype)
        array[tuple(slice(0,int(n)) for n in hi-lo)]=data[tuple(slice(int(a),int(b)) for a,b in zip(lo,hi))]
        (root/'0'/'.'.join(map(str,key))).write_bytes(array.tobytes())
    return root
