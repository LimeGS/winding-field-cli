"""Self-contained offline CT demo and a local results gallery."""

import html
import itertools
import json
from pathlib import Path

import numpy as np

from .util import atomic_bytes,atomic_json


def synthetic(path):
    path=Path(path); path.mkdir(parents=True,exist_ok=False)
    z,y,x=np.indices((24,32,56))
    coordinate=x-.003*(y-16)**2-.04*z
    distance=(coordinate+4)%8-4
    ridge=np.exp(-distance**2/1.5)
    ct=np.clip(25+ridge*(180+25*np.sin(y*.7)),0,255).astype(np.uint8)
    chunk=32
    atomic_json(path/'.zgroup',dict(zarr_format=2))
    atomic_json(path/'.zattrs',{'multiscales':[{'version':'0.4','axes':[
        {'name':a,'type':'space','unit':'micrometer'} for a in 'zyx'],
        'datasets':[{'path':'0','coordinateTransformations':[{'type':'scale','scale':[4.,4.,4.]},
                                                            {'type':'translation','translation':[80.,40.,20.]}]}]}]})
    atomic_json(path/'0/.zarray',dict(zarr_format=2,shape=list(ct.shape),chunks=[chunk]*3,dtype='|u1',
                compressor=None,filters=None,fill_value=0,order='C',dimension_separator='.'))
    for key in itertools.product(*(range((n-1)//chunk+1) for n in ct.shape)):
        lo=np.array(key)*chunk; hi=np.minimum(lo+chunk,ct.shape)
        tile=np.zeros((chunk,)*3,np.uint8)
        tile[tuple(slice(0,int(n)) for n in hi-lo)]=ct[tuple(slice(int(a),int(b)) for a,b in zip(lo,hi))]
        atomic_bytes(path/'0'/'.'.join(map(str,key)),tile.tobytes())
    atomic_json(path/'TRUTH.json',dict(kind='synthetic curved layers',period_input_voxels=8,
                spacing_um=4.,physical_validation='analytic example, not an ancient scroll'))
    return str(path.resolve())


def gallery(run):
    run=Path(run)
    report=json.loads((run/'RUN.json').read_bytes())
    flat=json.loads((run/'FLAT.json').read_bytes())
    meshes=json.loads((run/'MESHES.json').read_bytes())
    cards=[]; triangles=[]
    for p in sorted((run/'flat').glob('chart-*.npz')):
        if len(cards)<48:
            name=p.stem
            image=f'<img loading="lazy" src="flat/{name}.png" alt="Original CT sampled on flat chart">' if p.with_suffix('.png').exists() else ''
            cards.append(f'<article>{image}<p>{name}</p><a href="flat/{name}.obj">3D OBJ</a> · '
                         f'<a href="flat/{name}-flat.obj">flat OBJ</a> · <a href="flat/{name}.npz">correspondence</a></article>')
        if len(triangles)<20000:
            with np.load(p) as chart:
                triangles.extend(chart['vertices'][chart['faces']].tolist())
    payload=json.dumps(triangles[:20000],separators=(',',':'))
    title=html.escape(Path(report['config']['source']).name)
    content=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Winding field — {title}</title><style>
body{{font:16px system-ui;background:#11151b;color:#e8edf2;max-width:1200px;margin:32px auto;padding:0 20px}}
a{{color:#84c7ff}} .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:16px}}
article{{background:#202833;padding:14px}}img{{width:100%;height:210px;object-fit:contain}}canvas{{width:100%;height:420px;background:#080c10;touch-action:none}}
.note{{border-left:4px solid #d4a848;padding:12px;background:#28251d}}</style>
<h1>CT → fibres → field → meshes → flat charts</h1><p>{title}</p>
<p class="note">Pipeline execution complete. Physical sheet identity and whole-scroll continuity are unvalidated.
The classical option is a CT baseline. Flat charts form a linked atlas with explicit cuts.</p>
<p>{sum(x['faces'] for x in meshes['levels']):,} mesh faces · {flat['charts']} charts ·
{flat['coverage_fraction']:.1%} mesh-face coverage. <a href="RUN.json">Run receipt</a> ·
<a href="FIELD.json">Numerical status</a> · <a href="FLAT.json">Atlas receipt</a></p>
<canvas id="view" width="1100" height="420" aria-label="Rotatable mesh preview"></canvas>
<p>Drag to rotate the 3D preview (up to 20,000 faces). Full per-level OBJs are in <code>meshes/</code>.</p>
<h2>Flattened original CT</h2><p>Showing up to 48 charts. All charts and correspondence are in <code>flat/</code>.</p>
<div class="grid">{''.join(cards)}</div><script>
const triangles={payload}; const canvas=document.getElementById('view'),ctx=canvas.getContext('2d');
let ax=.35,ay=.7,drag=null;const centre=[0,0,0];let n=0;
for(const t of triangles)for(const p of t){{for(let i=0;i<3;i++)centre[i]+=p[i];n++;}}
centre.forEach((_,i)=>centre[i]/=Math.max(n,1));let radius=1;
for(const t of triangles)for(const p of t)radius=Math.max(radius,Math.hypot(...p.map((v,i)=>v-centre[i])));
function project(p){{let [x,y,z]=p.map((v,i)=>(v-centre[i])*180/radius);
let qx=x*Math.cos(ay)+z*Math.sin(ay),qz=-x*Math.sin(ay)+z*Math.cos(ay);
return [550+qx,210+y*Math.cos(ax)-qz*Math.sin(ax),y*Math.sin(ax)+qz*Math.cos(ax)];}}
function draw(){{ctx.clearRect(0,0,1100,420);let ts=triangles.map(t=>t.map(project));
ts.sort((a,b)=>a.reduce((s,p)=>s+p[2],0)-b.reduce((s,p)=>s+p[2],0));
for(const t of ts){{ctx.beginPath();ctx.moveTo(t[0][0],t[0][1]);ctx.lineTo(t[1][0],t[1][1]);ctx.lineTo(t[2][0],t[2][1]);ctx.closePath();
ctx.fillStyle=`hsl(35 45% ${{45+(t[0][2]/radius)%15}}%)`;ctx.fill();ctx.strokeStyle='#0002';ctx.stroke();}}}}
canvas.onpointerdown=e=>{{drag=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId);}};
canvas.onpointermove=e=>{{if(drag){{ay+=(e.clientX-drag[0])*.01;ax+=(e.clientY-drag[1])*.01;drag=[e.clientX,e.clientY];draw();}}}};
canvas.onpointerup=()=>drag=null;draw();</script></html>'''
    atomic_bytes(run/'index.html',content.encode())
