"""Dependency-free local-file canvas viewer for immutable sheet-atlas tiles."""
import json


def write_viewer(root,atlas,renders):
    sheets=[]
    for s,r in zip(atlas['sheets'],renders):
        prefix=s['render'].rsplit('/',1)[0]
        sheets.append(dict(s,tiles=[dict(x=t['canvas']['column_offset'],y=t['canvas']['row_offset'],
            width=t['canvas']['width'],height=t['canvas']['height'],src=prefix+'/render/'+t['directory']+'/image.png') for t in r['tiles']]))
    data=json.dumps(dict(shape=atlas['shape_yx'],pitch=atlas['pitch_um'],sheets=sheets),allow_nan=False).replace('<','\\u003c')
    (root/'index.html').write_text(HTML.replace('__DATA__',data))


HTML='''<!doctype html><html lang="es"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Atlas de hojas W constantes</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#13171b;color:#e2e6ea;font:14px system-ui}
header{display:flex;align-items:center;flex-wrap:wrap;gap:8px;padding:10px 14px;background:#20262d}
button,a{color:inherit;background:#303a44;border:1px solid #596673;border-radius:5px;padding:6px 10px}
button{cursor:pointer}button:focus-visible,canvas:focus-visible{outline:2px solid #9dd6ff}
small{color:#acb7c3}canvas{display:block;width:100%;height:calc(100dvh - 112px);touch-action:none;cursor:grab}
footer{padding:7px 14px;min-height:42px}#position{display:block;font-variant-numeric:tabular-nums}
</style><header><strong>Atlas W constante</strong><button id="fit">Encajar todo</button>
<button id="minus" aria-label="Alejar">−</button><button id="plus" aria-label="Acercar">+</button>
<span id="sheets"></span><a href="atlas.png">PNG completo</a><a href="ATLAS.json">Metadatos</a>
<small>Arrastrar · rueda para zoom · flechas / + / − / Inicio</small></header>
<canvas id="view" tabindex="0" aria-label="Atlas navegable de hojas W, usar flechas para desplazarse y más o menos para zoom"></canvas>
<footer><span id="position">Cargando…</span><small>Escala común; los bordes separan hojas independientes. Una misma fila no garantiza igual Z. XYZ exacto: comando sheet-atlas query.</small></footer>
<script>
(()=>{
'use strict';const data=__DATA__,canvas=document.getElementById('view'),ctx=canvas.getContext('2d'),status=document.getElementById('position');
let scale=1,tx=0,ty=0,cw=1,ch=1,drag=null,queued=false,failed=0;const cache=new Map();
function schedule(){if(!queued){queued=true;requestAnimationFrame(draw)}}
function image(src){if(cache.has(src)){const value=cache.get(src);cache.delete(src);cache.set(src,value);return value}
 const im=new Image();im.onload=schedule;im.onerror=()=>{failed++;status.textContent='No se pudo cargar '+src};im.src=src;cache.set(src,im);
 while(cache.size>64){const key=cache.keys().next().value;const old=cache.get(key);old.onload=null;cache.delete(key)}return im}
function fit(sheet){const [h,w]=sheet?sheet.shape_yx:data.shape,[x,y]=sheet?sheet.offset_xy:[0,0];
 scale=Math.min(32,(cw-40)/w,(ch-40)/h);tx=(cw-w*scale)/2-x*scale;ty=(ch-h*scale)/2-y*scale;schedule()}
function zoom(f,x=cw/2,y=ch/2){const next=Math.max(.005,Math.min(32,scale*f));tx=x-(x-tx)*next/scale;ty=y-(y-ty)*next/scale;scale=next;schedule()}
function draw(){queued=false;ctx.setTransform(1,0,0,1,0,0);ctx.fillStyle='#13171b';ctx.fillRect(0,0,canvas.width,canvas.height);
 const d=canvas.width/cw;ctx.setTransform(d*scale,0,0,d*scale,d*tx,d*ty);ctx.imageSmoothingEnabled=scale<1;
 for(const s of data.sheets){const [x,y]=s.offset_xy,[h,w]=s.shape_yx;if((x+w)*scale+tx<0||x*scale+tx>cw||(y+h)*scale+ty<0||y*scale+ty>ch)continue;
 const low=image(s.overview);if(low.complete&&low.naturalWidth)ctx.drawImage(low,x,y,w,h);
 if(scale>=.5)for(const t of s.tiles){const px=x+t.x,py=y+t.y;if((px+t.width)*scale+tx<0||px*scale+tx>cw||(py+t.height)*scale+ty<0||py*scale+ty>ch)continue;
 const im=image(t.src);if(im.complete&&im.naturalWidth)ctx.drawImage(im,px,py,t.width,t.height)}
 ctx.strokeStyle='#8797a680';ctx.lineWidth=1/scale;ctx.strokeRect(x,y,w,h);ctx.font=`${13/scale}px system-ui`;ctx.fillStyle='#dce9f5';ctx.fillText('W='+s.w,x+6/scale,Math.max(y*scale+ty+17,18)/scale-ty/scale)}
 if(!drag)status.textContent=`${(scale*100).toFixed(1)}% · ${data.pitch} µm/píxel · atlas ${data.shape[1]} × ${data.shape[0]}${failed?' · Hay imágenes que no cargaron':''}`}
function location(e){const r=canvas.getBoundingClientRect();return [e.clientX-r.left,e.clientY-r.top]}
canvas.addEventListener('wheel',e=>{e.preventDefault();zoom(Math.exp(-e.deltaY*.001),...location(e))},{passive:false});
canvas.addEventListener('pointerdown',e=>{canvas.focus();canvas.setPointerCapture(e.pointerId);drag=location(e);canvas.style.cursor='grabbing'});
canvas.addEventListener('pointermove',e=>{const [x,y]=location(e);if(drag){tx+=x-drag[0];ty+=y-drag[1];drag=[x,y];schedule()}
 const px=Math.floor((x-tx)/scale),py=Math.floor((y-ty)/scale),s=data.sheets.find(s=>px>=s.offset_xy[0]&&px<s.offset_xy[0]+s.shape_yx[1]&&py>=s.offset_xy[1]&&py<s.offset_xy[1]+s.shape_yx[0]);
 status.textContent=`Atlas (${px}, ${py})`+(s?` · W=${s.w} · UV=(${(s.uv_origin[0]+(px-s.offset_xy[0])*s.pitch).toFixed(2)}, ${(s.uv_origin[1]-(py-s.offset_xy[1])*s.pitch).toFixed(2)})`:' · Separación / exterior')});
function stop(){drag=null;canvas.style.cursor='grab'}canvas.addEventListener('pointerup',stop);canvas.addEventListener('pointercancel',stop);
canvas.addEventListener('keydown',e=>{if(e.key==='+'||e.key==='=')zoom(1.4);else if(e.key==='-')zoom(1/1.4);else if(e.key==='Home')fit();
 else if(e.key.startsWith('Arrow')){tx+=e.key==='ArrowLeft'?60:e.key==='ArrowRight'?-60:0;ty+=e.key==='ArrowUp'?60:e.key==='ArrowDown'?-60:0;schedule()}else return;e.preventDefault()});
document.getElementById('fit').onclick=()=>fit();document.getElementById('plus').onclick=()=>zoom(1.4);document.getElementById('minus').onclick=()=>zoom(1/1.4);
for(const s of data.sheets){const b=document.createElement('button');b.textContent='W='+s.w;b.onclick=()=>fit(s);document.getElementById('sheets').append(b)}
function resize(){const r=canvas.getBoundingClientRect(),d=Math.min(devicePixelRatio||1,2);cw=r.width;ch=r.height;canvas.width=Math.round(cw*d);canvas.height=Math.round(ch*d);schedule()}
window.addEventListener('resize',resize);resize();fit();
})();
</script></html>'''
