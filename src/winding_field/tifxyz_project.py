"""Bounded native-cell ray projection; no nearest-root or sign-only shortcut."""
from fractions import Fraction as Q
import heapq
import itertools
import json
import math

import numpy as np

from .tifxyz_contract import Blocked,Paused,to_output_xyz


def _q(value):return value if isinstance(value,Q) else Q(float(value))


def _fraction_bounds(value):
    rounded=float(value);represented=Q(rounded)
    if represented==value:return rounded,rounded
    if represented<value:return rounded,math.nextafter(rounded,math.inf)
    return math.nextafter(rounded,-math.inf),rounded


def _bounds(value):
    if isinstance(value,tuple):return value
    if isinstance(value,Q):return _fraction_bounds(value)
    return float(value),float(value)


def _add(a,b):
    if a==(0.,0.):return b
    if b==(0.,0.):return a
    return math.nextafter(a[0]+b[0],-math.inf),math.nextafter(a[1]+b[1],math.inf)


def _sub(a,b):
    if a[0]==a[1]==b[0]==b[1]:return (0.,0.)
    return _add(a,(-b[1],-b[0]))


def _mul(a,b):
    if a==(0.,0.) or b==(0.,0.):return (0.,0.)
    products=[x*y for x in a for y in b]
    return math.nextafter(min(products),-math.inf),math.nextafter(max(products),math.inf)


def _value_interval(values,point,level):
    pairs=[_sub((float(v),float(v)),(level,level)) for v in values.ravel()]
    for t in point[::-1]:
        t=_bounds(t)
        if t==(0.,0.):pairs=pairs[::2]
        elif t==(1.,1.):pairs=pairs[1::2]
        else:
            other=_sub((1.,1.),t)
            pairs=[_add(_mul(a,other),_mul(b,t)) for a,b in zip(pairs[::2],pairs[1::2])]
    return pairs[0]


def segment_value_interval(values,a,b,t,level):
    point=[]
    for start,end in zip(a,b):
        delta=_fraction_bounds(end-start) if isinstance(start,Q) and isinstance(end,Q) else _sub(_bounds(end),_bounds(start))
        value=_add(_bounds(start),_mul((t,t),delta))
        point.append((max(0.,value[0]),min(1.,value[1])))
    return _value_interval(values,point,level)


def _slope_interval(values,a,b):
    result=(0.,0.)
    for axis in range(3):
        low=np.take(values,0,axis=axis).ravel();high=np.take(values,1,axis=axis).ravel()
        differences=[_sub((float(y),float(y)),(float(x),float(x))) for x,y in zip(low,high)]
        grad=(min(v[0] for v in differences),max(v[1] for v in differences))
        direction=_fraction_bounds(b[axis]-a[axis]) if isinstance(a[axis],Q) else _sub(_bounds(b[axis]),_bounds(a[axis]))
        result=_add(result,_mul(grad,direction))
    return result


def _sign(interval):
    return 1 if interval[0]>0 else -1 if interval[1]<0 else 0


def _monotone_roots(values,a,b,level,tolerance):
    slope=_slope_interval(values,a,b);sign=_sign(slope)
    if not sign:return None
    first=_value_interval(values,a,level);last=_value_interval(values,b,level)
    if first==(0.,0.):return [(0.,0.,sign)]
    if last==(0.,0.):return [(1.,1.,sign)]
    sa,sb=_sign(first),_sign(last)
    if not sa or not sb:return None
    if sa==sb:return []
    lo,hi=0.,1.;minimum=slope[0] if sign>0 else -slope[1]
    for _ in range(128):
        if hi-lo<=tolerance:return [(lo,hi,sign)]
        mid=(lo+hi)/2
        value=segment_value_interval(values,a,b,mid,level);sm=_sign(value)
        if not sm:
            radius=math.nextafter(max(abs(v) for v in value)/minimum,math.inf)
            left=max(lo,math.nextafter(mid-radius,-math.inf));right=min(hi,math.nextafter(mid+radius,math.inf))
            if right-left<=tolerance:return [(left,right,sign)]
            return None
        if sm==sa:lo=mid
        else:hi=mid
    return None


def _trim(p):
    while len(p)>1 and not p[-1]:p.pop()
    return p


def _eval(p,x):
    value=Q(0)
    for c in reversed(p):value=value*x+c
    return value


def _derivative(p):return [i*c for i,c in enumerate(p)][1:] or [Q(0)]


def _divide(p,q):
    remainder=p.copy();quotient=[Q(0)]*max(1,len(p)-len(q)+1)
    while any(remainder) and len(remainder)>=len(q):
        i=len(remainder)-len(q);factor=remainder[-1]/q[-1];quotient[i]=factor
        for j,c in enumerate(q):remainder[i+j]-=factor*c
        _trim(remainder)
    return _trim(quotient),_trim(remainder)


def _sturm(p):
    sequence=[p,_derivative(p)]
    while any(sequence[-1]):
        _,remainder=_divide(sequence[-2],sequence[-1])
        if not any(remainder):break
        remainder=[-x for x in remainder];factor=abs(remainder[-1])
        sequence.append([x/factor for x in remainder])
    return sequence


def _variations(sequence,x):
    signs=[1 if value>0 else -1 for p in sequence if (value:=_eval(p,x))]
    return sum(a!=b for a,b in zip(signs,signs[1:]))


def _exact_polynomial(values,a,b,level):
    p=[Q(0)]*4
    for bits in itertools.product((0,1),repeat=3):
        weight=[Q(float(values[bits]))-Q(level)]
        for i,bit in enumerate(bits):
            start=_q(a[i]);delta=_q(b[i])-start
            factor=[start,delta] if bit else [1-start,-delta]
            product=[Q(0)]*(len(weight)+1)
            for j,c in enumerate(weight):
                product[j]+=c*factor[0];product[j+1]+=c*factor[1]
            weight=product
        p=[x+y for x,y in zip(p,weight)]
    return _trim(p)


def _exact_roots(values,a,b,level,tolerance):
    # Exact binary-rational fallback avoids inventing an epsilon for degree/sign tests.
    p=_exact_polynomial(values,a,b,level)
    if len(p)==1:return None if p[0]==0 else []
    original=p.copy();derivative=_derivative(p);found=[]
    for x in (Q(0),Q(1)):
        if not _eval(p,x):
            slope=_eval(derivative,x)
            if not slope:return None
            found.append((float(x),float(x),1 if slope>0 else -1))
            p,_=_divide(p,[-x,Q(1)])
    while len(p)>1:
        sequence=_sturm(p)
        common=sequence[-1]
        if len(common)>1:
            shared=_sturm(common)
            if not _eval(common,Q(0)) or not _eval(common,Q(1)) or _variations(shared,Q(0))!=_variations(shared,Q(1)):
                return None
        stack=[(Q(0),Q(1))];intervals=[];restart=False;steps=0
        while stack:
            lo,hi=stack.pop();steps+=1
            if steps>4096:raise Blocked('ROOT_ALGEBRA_LIMIT')
            count=_variations(sequence,lo)-_variations(sequence,hi)
            if count==0:continue
            if count==1 and hi-lo<=Q(float(tolerance)):
                intervals.append((math.nextafter(float(lo),-math.inf),math.nextafter(float(hi),math.inf),0));continue
            mid=(lo+hi)/2
            if not _eval(p,mid):
                slope=_eval(derivative,mid)
                left,right=_fraction_bounds(mid)
                found.append((left,right,1 if slope>0 else -1))
                p,_=_divide(p,[-mid,Q(1)]);restart=True;break
            stack.extend([(mid,hi),(lo,mid)])
        if restart:continue
        found.extend(intervals);break
    return sorted(found)


def _segments(field,origin,direction,t_interval):
    origin=np.asarray(origin,float);direction=np.asarray(direction,float)
    if origin.shape!=(3,) or direction.shape!=(3,) or not np.isfinite([origin,direction]).all():raise ValueError('finite XYZ ray required')
    length=float(np.linalg.norm(direction))
    if not length or not math.isfinite(length):raise Blocked('DEGENERATE_NORMAL')
    if len(t_interval)!=2 or not all(math.isfinite(x) for x in t_interval) or t_interval[1]<=t_interval[0]:raise ValueError('invalid ray interval')
    direction=direction/length
    # Never certify a rounded grid ray in place of the supplied physical ray.
    gf=[(_q(p)-_q(o))/_q(h) for p,o,h in zip(origin[::-1],field.origin_um[::-1],field.spacing_um[::-1])]
    df=[_q(n)/_q(h) for n,h in zip(direction[::-1],field.spacing_um[::-1])]
    maximum=[int(n)-1 for n in field.shape]
    lower,upper=map(lambda x:Q(float(x)),t_interval)
    for i in range(3):
        if not df[i]:
            if gf[i]<0 or gf[i]>maximum[i]:return []
        else:
            a,b=sorted([(Q(0)-gf[i])/df[i],(Q(int(maximum[i]))-gf[i])/df[i]])
            lower=max(lower,a);upper=min(upper,b)
    if lower>=upper:return []
    events={lower:[],upper:[]};ranges=[];crossings=0
    for axis in range(3):
        if not df[axis]:continue
        low,high=sorted([gf[axis]+df[axis]*lower,gf[axis]+df[axis]*upper])
        first=max(0,math.ceil(low));last=min(maximum[axis],math.floor(high))
        crossings+=max(0,last-first+1);ranges.append((axis,first,last))
        if crossings>3*4097:raise Blocked('RAY_WORK_LIMIT')
    for axis,first,last in ranges:
        for k in range(first,last+1):
            t=(Q(k)-gf[axis])/df[axis]
            if lower<=t<=upper:events.setdefault(t,[]).append((axis,k))
    times=sorted(events)
    if len(times)-1>4096:raise Blocked('RAY_WORK_LIMIT')
    mid=(times[0]+times[1])/2
    cell=[min(math.floor(gf[i]+df[i]*mid),maximum[i]-1) for i in range(3)]
    segments=[]
    for i,(ta,tb) in enumerate(zip(times,times[1:])):
        if any(not 0<=c<m for c,m in zip(cell,maximum)):raise ValueError('invalid exact DDA cell')
        roi=lambda t:any(k in (0,maximum[axis]) for axis,k in events[t])
        segments.append(dict(cell=tuple(cell),grid_origin=gf,grid_direction=df,t0=ta,t1=tb,event0=i,event1=i+1,
            last_event=len(times)-1,roi0=roi(ta),roi1=roi(tb),face0=bool(events[ta]),face1=bool(events[tb])))
        for axis,k in events[tb]:cell[axis]=k if df[axis]>0 else k-1
    return segments


def _root_record(field,segment,values,interval,level):
    lo,hi,slope=interval;middle=(_q(lo)+_q(hi))/2
    point_exact=[a+middle*(b-a) for a,b in zip(segment['a'],segment['b'])]
    world_exact=[_q(o)+(p+c)*_q(h) for o,p,c,h in zip(field.origin_um,point_exact[::-1],segment['cell'][::-1],field.spacing_um)]
    xyz=np.array([float(x) for x in world_exact]);point=np.array([float(x) for x in point_exact])
    rounding=sum(_fraction_bounds(abs(_q(float(x))-exact))[1] for x,exact in zip(xyz,world_exact))
    width=_fraction_bounds(_q(hi)-_q(lo))[1]
    error=math.nextafter(width*segment['length_upper_um']/2+rounding,math.inf)
    residual=0.
    for bits in itertools.product((0,1),repeat=3):residual+=float(values[bits])*float(np.prod(np.where(bits,point,1-point)))
    endpoint=0 if lo==hi==0 else 1 if lo==hi==1 else None
    event=None if endpoint is None else segment['event0' if endpoint==0 else 'event1']
    one_sided=bool(endpoint is not None and segment['roi0' if endpoint==0 else 'roi1'])
    return dict(xyz_um=xyz.tolist(),error_um=error,residual=residual-level,
                t_bounds=[_fraction_bounds(segment['t0']+_q(lo)*(segment['t1']-segment['t0']))[0],
                          _fraction_bounds(segment['t0']+_q(hi)*(segment['t1']-segment['t0']))[1]],
                event=event,slope=slope,one_sided=one_sided,incident_segments=1,
                requires_other_side=bool(endpoint is not None and not one_sided and segment['face0' if endpoint==0 else 'face1']))


def ray_batch(field,origins,directions,t_intervals,xtol_um,*,level,stop=None):
    if not math.isfinite(xtol_um) or xtol_um<=0 or not math.isfinite(level):raise ValueError('invalid level/root tolerance')
    if not (len(origins)==len(directions)==len(t_intervals)) or len(origins)>16384:raise ValueError('bounded matching ray batch required')
    states=[dict(roots=[],status=None,reason=None) for _ in origins];queue=[]
    def flush():
        if not queue:return
        if stop and stop():raise Paused()
        cells=np.array([segment['cell'] for _,segment in queue],np.int64)
        values,valid=field.read_cells(cells)
        active=(values.min(axis=(1,2,3))<=level)&(values.max(axis=(1,2,3))>=level)
        for j,(i,segment) in enumerate(queue):
            state=states[i]
            if not valid[j]:state.update(status='UNSUPPORTED',reason='UNSUPPORTED_STENCIL');continue
            if state['status'] or not active[j]:continue
            segment['a']=tuple(g+d*segment['t0']-c for g,d,c in zip(segment['grid_origin'],segment['grid_direction'],segment['cell']))
            segment['b']=tuple(g+d*segment['t1']-c for g,d,c in zip(segment['grid_origin'],segment['grid_direction'],segment['cell']))
            if any(not 0<=q<=1 for q in segment['a']+segment['b']):raise ValueError('exact DDA endpoint outside cell')
            segment_length=math.nextafter(sum(_fraction_bounds(abs((b-a)*_q(h)))[1]
                for a,b,h in zip(segment['a'],segment['b'],field.spacing_um[::-1])),math.inf)
            segment['length_upper_um']=segment_length
            if not segment_length:state.update(status='AMBIGUOUS',reason='DEGENERATE_SEGMENT');continue
            tolerance=min(.25,math.nextafter(xtol_um/segment_length,0.))
            roots=_monotone_roots(values[j],segment['a'],segment['b'],float(level),tolerance)
            if roots is None:roots=_exact_roots(values[j],segment['a'],segment['b'],float(level),tolerance)
            if roots is None:state.update(status='AMBIGUOUS',reason='TANGENT_OR_PLATEAU');continue
            for interval in roots:
                r=_root_record(field,segment,values[j],interval,float(level))
                old=next((x for x in state['roots'] if r['event'] is not None and x['event']==r['event']),None)
                if old is not None:
                    if old['slope']!=r['slope']:state.update(status='AMBIGUOUS',reason='NONTRANSVERSE_BOUNDARY')
                    old['one_sided']=False;old['incident_segments']+=1
                elif len(state['roots'])<4:state['roots'].append(r)
        queue.clear()
    for i,(origin,direction,interval) in enumerate(zip(origins,directions,t_intervals)):
        for segment in _segments(field,origin,direction,interval):
            queue.append((i,segment))
            if len(queue)==16384:flush()
    flush()
    for state in states:
        if state['status'] is None and any(r['requires_other_side'] and r['incident_segments']<2 for r in state['roots']):
            state.update(status='AMBIGUOUS',reason='UNCHECKED_INTERNAL_SEARCH_BOUNDARY')
        if state['status'] is None:
            n=len(state['roots']);state.update(status='UNIQUE' if n==1 else 'ABSENT' if n==0 else 'AMBIGUOUS',
                reason='ONE_SIMPLE_ROOT' if n==1 else 'NO_ROOT' if n==0 else 'MULTIPLE_ROOTS')
    return states


def ray_roots(field,origin_um,direction,t_interval,xtol_um,*,level):
    return ray_batch(field,[origin_um],[direction],[t_interval],xtol_um,level=level)[0]


def make_grid(guide,pitch_um,max_pixels,*,level):
    if not math.isfinite(pitch_um) or pitch_um<=0:raise ValueError('positive UV pitch required')
    extent=np.ptp(guide['uv'],axis=0)
    with np.errstate(over='ignore'):dimensions=np.ceil(extent/pitch_um)+1
    if not np.isfinite(dimensions).all() or (dimensions>max_pixels).any():raise Blocked('PIXEL_LIMIT')
    width,height=(max(2,int(n)) for n in dimensions)
    if width*height>max_pixels:raise Blocked('PIXEL_LIMIT')
    return dict(shape_yx=[height,width],uv_min_um=guide['uv'].min(0).tolist(),pitch_um=pitch_um,level=level,tile_pixels=256)


def _uv_samples(guide,grid,low,high):
    import igl
    rows,cols=np.mgrid[low[0]:high[0],low[1]:high[1]]
    queries=np.column_stack([cols.ravel(),rows.ravel()])*grid['pitch_um']+grid['uv_min_um']
    uv=guide['uv'];faces=guide['faces'];flat=np.column_stack([uv,np.zeros(len(uv))])
    tree=igl.AABB();tree.init(flat,faces)
    ids=np.empty(rows.size,np.int64);weights=np.empty((rows.size,3),np.longdouble)
    for first in range(0,len(queries),16384):
        q=queries[first:first+16384];_,found,_=tree.squared_distance(flat,faces,np.column_stack([q,np.zeros(len(q))]))
        t=uv[faces[found]].astype(np.longdouble);a,b=t[:,1]-t[:,0],t[:,2]-t[:,0];delta=q.astype(np.longdouble)-t[:,0]
        cross=lambda a,b:a[:,0]*b[:,1]-a[:,1]*b[:,0]
        determinant=cross(a,b)
        if not (determinant>0).all():raise ValueError('invalid guide UV Jacobian')
        ids[first:first+len(q)]=found
        weights[first:first+len(q)]=np.column_stack([cross(t[:,1]-q,t[:,2]-q)/determinant,cross(delta,b)/determinant,cross(a,delta)/determinant])
    inside=(weights>=0).all(1).reshape(rows.shape)
    cells=inside[:-1,:-1]&inside[1:,:-1]&inside[:-1,1:]&inside[1:,1:]
    used=np.zeros_like(inside)
    for dy,dx in ((0,0),(0,1),(1,0),(1,1)):used[dy:dy+len(cells),dx:dx+cells.shape[1]]|=cells
    return rows,cols,ids.reshape(rows.shape),weights.reshape((*rows.shape,3)),used


def precheck_projection(field,guide,grid,stop=None):
    """Find counterexamples among 64 mandatory boundary nodes; never certify a level.

    The exit-distance ranking only selects rays. Rejection uses the same native
    solver, origins, normals, radius and tolerance as project_tile.
    """
    size=grid.get('tile_pixels',256)
    if size not in (128,256):raise ValueError('invalid precheck tile size')
    height,width=grid['shape_yx'];heap=[];visible=windows=0
    upper=field.origin_um+(np.array(field.shape[::-1])-1)*field.spacing_um
    for r in range(0,height,size):
        for c in range(0,width,size):
            if stop and stop():raise Paused()
            low=[max(0,r-2),max(0,c-2)];high=[min(height,r+size+2),min(width,c+size+2)]
            rows,cols,ids,weights,used=_uv_samples(guide,grid,low,high);windows+=1
            selected=np.flatnonzero(used&(rows>=r)&(rows<r+size)&(cols>=c)&(cols<c+size))
            visible+=len(selected)
            tri=guide['faces'][ids.ravel()[selected]];w=weights.reshape(-1,3)[selected,:,None]
            points=np.sum(w*guide['world'][tri],axis=1).astype(float)
            directions=np.sum(w*guide['normals'][tri],axis=1).astype(float)
            with np.errstate(divide='ignore',invalid='ignore'):
                unit=directions/np.linalg.norm(directions,axis=1)[:,None]
                margin=np.minimum(points-field.origin_um,upper-points)
                distance=np.min(np.where(abs(unit)>1e-15,margin/abs(unit),np.inf),axis=1)
            distance=np.where(np.isnan(distance),np.inf,distance)
            indices=rows.ravel()[selected]*width+cols.ravel()[selected]
            # Sorting is tile-bounded; the retained global heap never exceeds 64.
            for j in np.lexsort((indices,distance))[:64]:
                priority=(-float(distance[j]),-int(indices[j]))
                if len(heap)==64 and priority<=heap[0][:2]:continue
                k=selected[j]
                sample=dict(pixel_yx=[int(rows.ravel()[k]),int(cols.ravel()[k])],origin_um=points[j].tolist(),
                            direction=directions[j].tolist(),exit_distance_um=float(distance[j]) if math.isfinite(distance[j]) else None)
                if len(heap)<64:heapq.heappush(heap,(*priority,sample))
                else:heapq.heapreplace(heap,(*priority,sample))
    queries=[entry[2] for entry in sorted(heap,reverse=True)]
    radius=2*float(np.linalg.norm(guide['stride']*field.spacing_um));xtol=float(np.min(field.spacing_um))/1000
    receipt=dict(status='NO_REJECTION_WITNESS',criterion='64 mandatory reader-visible nodes nearest ROI ray exit',
                 visible_nodes=visible,uv_windows=windows,selected_nodes=len(queries),evaluated_nodes=0,
                 radius_um=radius,xtol_um=xtol,samples=[])
    for start in range(0,len(queries),8):
        if stop and stop():raise Paused()
        group=queries[start:start+8]
        results=ray_batch(field,[q['origin_um'] for q in group],[q['direction'] for q in group],
                          [(-radius,radius)]*len(group),xtol,level=grid['level'],stop=stop)
        receipt['evaluated_nodes']+=len(group)
        receipt['samples'].extend(dict(query,**result) for query,result in zip(group,results))
        failure=next((s for s in receipt['samples'] if s['status']!='UNIQUE'),None)
        if failure is not None:
            receipt.update(status='BLOCKED',failure_pixel_yx=failure['pixel_yx'])
            raise Blocked('PROJECTION_'+failure['status'],json.dumps(receipt,allow_nan=False))
    return receipt


def _encoded_roi_bounds(field,frame):
    """Directed float32 bounds avoid rounding a certified root outside its ROI."""
    low=[];high=[];unit=_q(frame['voxel_um'])
    for i,(origin,spacing,count,target_origin) in enumerate(zip(field.origin_um,field.spacing_um,field.shape[::-1],frame['origin_xyz_um'])):
        a=max(Q(0),(_q(origin)-_q(target_origin))/unit)
        b=min(Q(frame['shape_zyx'][::-1][i]-1),(_q(origin)+(count-1)*_q(spacing)-_q(target_origin))/unit)
        x=np.float32(float(a));y=np.float32(float(b))
        if not np.isfinite([x,y]).all():raise Blocked('FLOAT32_ROI_UNREPRESENTABLE')
        if _q(float(x))<a:x=np.nextafter(x,np.float32(np.inf))
        if _q(float(y))>b:y=np.nextafter(y,np.float32(-np.inf))
        if x>y:raise Blocked('FLOAT32_ROI_UNREPRESENTABLE')
        low.append(x);high.append(y)
    return np.asarray(low,np.float32),np.asarray(high,np.float32)


def project_tile(field,guide,grid,tile_id,settings,stop=None):
    size=grid.get('tile_pixels',256);height,width=grid['shape_yx'];ty,tx=tile_id
    if size not in (128,256) or type(ty) is not int or type(tx) is not int or min(ty,tx)<0:raise ValueError('invalid projection tile')
    r0,c0=ty*size,tx*size;r1,c1=min(r0+size,height),min(c0+size,width)
    if r0>=height or c0>=width:raise ValueError('tile outside grid')
    low=[max(0,r0-1),max(0,c0-1)];high=[min(height,r1+1),min(width,c1+1)]
    # A second UV-only halo determines participation of the delivered first halo.
    outer_low=[max(0,r0-2),max(0,c0-2)];outer_high=[min(height,r1+2),min(width,c1+2)]
    crop=tuple(slice(a-o,b-o) for a,b,o in zip(low,high,outer_low))
    rows,cols,ids,weights,used=[a[crop] for a in _uv_samples(guide,grid,outer_low,outer_high)]
    faces=guide['faces'];ids=ids.ravel();weights=weights.reshape(-1,3);visible=used.ravel()
    xyz=np.full((*rows.shape,3),-1,np.float32);mask=np.zeros(rows.shape,bool);errors=np.zeros(rows.shape,float)
    radius=2*float(np.linalg.norm(guide['stride']*field.spacing_um));xtol=float(np.min(field.spacing_um))/1000
    for first in range(0,rows.size,16384):
        if stop and stop():raise Paused()
        good=np.flatnonzero(visible[first:first+16384])
        for start in range(0,len(good),16384):
            selected=good[start:start+16384];tri=faces[ids[first+selected]];w=weights[first+selected,:,None]
            origins=np.sum(w*guide['world'][tri],axis=1).astype(float)
            directions=np.sum(w*guide['normals'][tri],axis=1).astype(float)
            roots=ray_batch(field,origins,directions,[(-radius,radius)]*len(selected),xtol,level=grid['level'],stop=stop)
            points=[];bounds=[]
            for j,result in enumerate(roots):
                if result['status']!='UNIQUE':
                    index=first+int(selected[j]);witness=dict(pixel_yx=[int(rows.ravel()[index]),int(cols.ravel()[index])],
                        reason=result['reason'],roots=result['roots'],origin_um=origins[j].tolist(),direction=directions[j].tolist(),radius_um=radius)
                    raise Blocked('PROJECTION_'+result['status'],json.dumps(witness,allow_nan=False))
                points.append(result['roots'][0]['xyz_um']);bounds.append(result['roots'][0]['error_um'])
            if not points:continue
            points=np.asarray(points);encoded=to_output_xyz(points,settings.output_frame)
            encoded=np.clip(encoded,*_encoded_roi_bounds(field,settings.output_frame))
            decoded=encoded.astype(float)*settings.output_frame['voxel_um']+settings.output_frame['origin_xyz_um']
            index=first+selected;xyz.reshape(-1,3)[index]=encoded;mask.ravel()[index]=True
            roundoff=32*np.finfo(float).eps*np.maximum(1,np.maximum(abs(decoded),abs(points)).max(1))
            errors.ravel()[index]=np.asarray(bounds)+np.linalg.norm(decoded-points,axis=1)+roundoff
    return dict(tile_id=list(tile_id),owned_box_yx=[[r0,c0],[r1,c1]],box_yx=[low,high],xyz=xyz,mask=mask,root_error_um=errors)
