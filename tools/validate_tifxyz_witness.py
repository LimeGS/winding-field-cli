"""Independent exact-algebra check of captured native-field ray witnesses.

Uses physical-coordinate rational DDA and SymPy root counts, not the production
projector's grid DDA, interval filter or Sturm implementation. Diagnostic only.
"""
import argparse
from fractions import Fraction as Q
import hashlib
import itertools
import json
import math
from pathlib import Path


def exact_ray_segments(origin,direction,field_origin,spacing,shape,radius):
    if (len(shape)!=3 or any(type(n) is not int or n<2 for n in shape)
        or any(len(v)!=3 or not all(type(x) in (int,float) and math.isfinite(x) for x in v)
               for v in (origin,direction,field_origin,spacing))
        or min(spacing)<=0 or not math.isfinite(radius) or radius<=0):raise ValueError('invalid ray geometry')
    g=list(map(Q,origin[::-1]));d=list(map(Q,direction[::-1]));o=list(map(Q,field_origin[::-1]));h=list(map(Q,spacing[::-1]))
    low,high=-Q(radius),Q(radius)
    for i in range(3):
        a=o[i];b=o[i]+(shape[i]-1)*h[i]
        if not d[i]:
            if not a<=g[i]<=b:return []
        else:
            first,last=sorted(((a-g[i])/d[i],(b-g[i])/d[i]));low=max(low,first);high=min(high,last)
    if low>=high:return []
    times={low,high};ranges=[];crossings=0
    for i in range(3):
        if d[i]:
            a,b=sorted(((g[i]+low*d[i]-o[i])/h[i],(g[i]+high*d[i]-o[i])/h[i]))
            first=max(0,math.ceil(a));last=min(shape[i]-1,math.floor(b))
            crossings+=max(0,last-first+1);ranges.append((i,first,last))
    if crossings>3*4097:raise ValueError('ray work limit')
    for i,first,last in ranges:
            for k in range(first,last+1):
                t=(o[i]+k*h[i]-g[i])/d[i]
                if low<t<high:times.add(t)
    times=sorted(times)
    if len(times)>4097:raise ValueError('witness exceeds bounded DDA')
    result=[]
    for first,last in zip(times,times[1:]):
        mid=(first+last)/2
        cell=[min(math.floor((g[i]+mid*d[i]-o[i])/h[i]),shape[i]-2) for i in range(3)]
        a=[(g[i]+first*d[i]-o[i])/h[i]-cell[i] for i in range(3)]
        b=[(g[i]+last*d[i]-o[i])/h[i]-cell[i] for i in range(3)]
        assert all(0<=x<=1 for x in a+b)
        pack=lambda x:[x.numerator,x.denominator]
        result.append(dict(cell=cell,t0=pack(first),t1=pack(last),a=[pack(x) for x in a],b=[pack(x) for x in b]))
    assert sum((Q(*r['t1'])-Q(*r['t0']) for r in result),Q(0))==high-low
    return result


def validate(path,source_replay=None):
    import sympy as sp
    path=Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size>16<<20:raise ValueError('invalid bounded witness file')
    raw=path.read_bytes();data=json.loads(raw);s=sp.Symbol('s');counts=[]
    interior=0;endpoints={};nontransverse=False;plateau=False
    expected=exact_ray_segments(data['origin_um'],data['direction_unit'],data['field_origin_um'],
                                data['spacing_um'],data['shape_zyx'],data['radius_um'])
    if not expected or len(expected)!=len(data['segments']):raise ValueError('incomplete claimed ray segments')
    for row,wanted in zip(data['segments'],expected):
        if any(row.get(k)!=v for k,v in wanted.items()):raise ValueError('segment geometry differs from claimed ray')
        values=row['values']
        if (not isinstance(values,list) or len(values)!=2 or any(not isinstance(plane,list) or len(plane)!=2
            or any(not isinstance(line,list) or len(line)!=2 or any(type(v) not in (int,float) or not math.isfinite(v) for v in line)
                   for line in plane) for plane in values)):raise ValueError('invalid segment field values')
        a=[sp.Rational(*x) for x in row['a']];b=[sp.Rational(*x) for x in row['b']]
        p=-sp.Rational(data['level'])
        for bits in itertools.product((0,1),repeat=3):
            weight=sp.Integer(1)
            for i,bit in enumerate(bits):
                q=a[i]+s*(b[i]-a[i]);weight*=q if bit else 1-q
            p+=sp.Rational(row['values'][bits[0]][bits[1]][bits[2]])*weight
        polynomial=sp.Poly(p,s)
        if polynomial.is_zero:
            plateau=True;counts.append(None);continue
        squarefree=polynomial.sqf_part();count=int(squarefree.count_roots(0,1));counts.append(count)
        zero0=squarefree.eval(0)==0;zero1=squarefree.eval(1)==0
        interior+=count-int(zero0)-int(zero1)
        repeated=sp.gcd(polynomial,polynomial.diff())
        if repeated.degree()>0 and repeated.count_roots(0,1):nontransverse=True
        for point,is_zero,key in ((0,zero0,'t0'),(1,zero1,'t1')):
            if is_zero:
                derivative=polynomial.diff().eval(point)
                sign=1 if derivative>0 else -1 if derivative<0 else 0
                endpoints.setdefault(Q(*row[key]),[]).append(sign)
    if any(0 in signs or (min(signs)<0<max(signs)) for signs in endpoints.values()):nontransverse=True
    distinct=None if plateau else interior+len(endpoints)
    source_bound=False;witness_sha=hashlib.sha256(raw).hexdigest()
    if source_replay is not None:
        replay=json.loads(Path(source_replay).read_text())
        if (replay.get('status')!='SOURCE_AND_GUIDE_REPLAY_VERIFIED' or replay.get('witness_sha256')!=witness_sha
            or replay.get('field_binding')!=data['field_binding']
            or any(replay.get(k)!=data[k] for k in ('candidate','stride','pixel_yx'))):raise ValueError('source replay binding differs')
        source_bound=True
    reason='MULTIPLE_ROOTS' if plateau or distinct>1 else 'NONTRANSVERSE' if nontransverse else 'ABSENCE' if distinct==0 else None
    status=('EXACT_'+reason+('_CONFIRMED' if source_bound else '_IN_SUPPLIED_DATA')) if reason else 'ABSENCE_NOT_CONFIRMED'
    return dict(status=status,source_bound=source_bound,
                witness_sha256=hashlib.sha256(raw).hexdigest(),segments=len(counts),roots_per_segment=counts,
                distinct_roots=distinct,plateau=plateau,nontransverse=nontransverse,
                sympy_version=sp.__version__,field_binding=data['field_binding'],pixel_yx=data['pixel_yx'],
                candidate=data['candidate'],stride=data['stride'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('input',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source-replay',type=Path)
    args=p.parse_args();result=validate(args.input,args.source_replay)
    with args.output.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='roots_per_segment'},indent=2))
