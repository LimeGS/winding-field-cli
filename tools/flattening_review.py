"""Private static blind CT review; original method identities stay outside the served tree."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import shutil
import uuid

from winding_field.util import atomic_json, digest, file_sha
from bench_flattening import locked, read_json, safe, seal, validate_freeze, verify


HTML = """<!doctype html><meta charset="utf-8"><title>CT texture comparison</title>
<style>body{font:15px system-ui;background:#202124;color:#eee}select,button,input{margin:6px}
.panels{display:flex;gap:12px}article{min-width:350px;flex:1;border:1px solid #777;padding:8px}
.viewport{width:100%;height:600px;overflow:auto;background:#555}.tiles{position:relative}
.tiles img{position:absolute;image-rendering:auto}label{display:block}.absent{color:#ffc266}</style>
<h1>Textura CT — comparación ciega</h1><p>Misma región física, pitch y profundidad entre candidatos.
Gris/transparente significa sin soporte o fuera de región, no papiro negro.
Evaluar fibras, continuidad y deformación; no elegir por brillo o tamaño del panel.</p>
<select id="group"></select><select id="region"></select><input id="depth" type="range" min="0" max="0" value="0">
<span id="offset"></span><button id="save">Guardar voto de esta región</button>
<button id="download">Descargar votos</button><input id="restore" type="file" accept=".json">
<div id="panels" class="panels"></div><p id="status"></p>
<section id="references"></section>
<script>
let manifest, reviewHash, entries=[], current=[], groups=[];
const $=s=>document.getElementById(s);
const sha=async b=>[...new Uint8Array(await crypto.subtle.digest('SHA-256',b))].map(x=>x.toString(16).padStart(2,'0')).join('');
async function init(){
 const r=await fetch('manifest.json');const bytes=await r.arrayBuffer();reviewHash=await sha(bytes);
 manifest=JSON.parse(new TextDecoder().decode(bytes));
 groups=[...new Set(manifest.panels.map(p=>JSON.stringify([p.case_id,p.track,p.scope])))];
 groups.forEach((g,i)=>{const o=new Option(JSON.parse(g).join(' / '),i);$('group').add(o)});
 $('group').onchange=chooseGroup;$('region').onchange=chooseRegion;$('depth').oninput=draw;chooseGroup();
}
function chooseGroup(){
 const g=JSON.parse(groups[$('group').value||0]);current=manifest.panels.filter(p=>JSON.stringify([p.case_id,p.track,p.scope])===JSON.stringify(g));
  $('region').replaceChildren();const reference=(manifest.references||[]).find(r=>r.case_id===g[0]);
  const regions=[...new Set([...current.flatMap(p=>p.regions.map(r=>r.id)),...(reference?.regions||[]).map(r=>r.id)])];
 regions.forEach(r=>$('region').add(new Option(r,r)));chooseRegion();
}
function chooseRegion(){
 const r=current.flatMap(p=>p.regions).find(r=>r.id===$('region').value);
 $('depth').max=Math.max(0,(r?.offsets_um.length||1)-1);$('depth').value=r?Math.max(0,r.offsets_um.indexOf(0)):0;draw();
}
function draw(){
 $('panels').replaceChildren();const depth=Number($('depth').value);
 current.forEach((p,i)=>{
  const article=document.createElement('article');const title=document.createElement('h3');title.textContent='Candidato '+(i+1);article.append(title);
  const region=p.regions.find(r=>r.id===$('region').value);
  if(!region||region.missing||p.status!=='COMPLETE'){const t=document.createElement('p');t.className='absent';t.textContent='Sin resultado válido: '+p.status;article.append(t)}
  else{
   $('offset').textContent=region.offsets_um[depth]+' µm — '+region.pitch_um+' µm/píxel';
   const vp=document.createElement('div');vp.className='viewport';const tiles=document.createElement('div');tiles.className='tiles';
   const lo=Math.min(...region.tiles.map(t=>t.canvas.u0));const top=Math.max(...region.tiles.map(t=>t.canvas.v_top));
   let width=0,height=0;region.tiles.forEach(t=>{
    const x=(t.canvas.u0-lo)/region.pitch_um,y=(top-t.canvas.v_top)/region.pitch_um;
    const image=new Image();image.loading='lazy';image.src=t.planes[depth];image.style.left=x+'px';image.style.top=y+'px';
    image.width=t.canvas.width;image.height=t.canvas.height;tiles.append(image);width=Math.max(width,x+image.width);height=Math.max(height,y+image.height);
   });tiles.style.width=width+'px';tiles.style.height=height+'px';vp.append(tiles);article.append(vp);
  }
  const label=document.createElement('label');label.textContent='Preferido (varios = empate)';const input=document.createElement('input');input.type='checkbox';input.dataset.token=p.token;
  input.disabled=!region||region.missing||p.status!=='COMPLETE';label.prepend(input);article.append(label);
  for(const name of ['textura','continuidad','deformacion']){
   const lab=document.createElement('label');lab.textContent=name+' (0 ilegible, 4 preservada)';const sel=document.createElement('select');sel.dataset.token=p.token;sel.dataset.score=name;
   sel.add(new Option('Sin evaluar',''));for(let j=0;j<=4;j++)sel.add(new Option(j,j));lab.append(sel);article.append(lab);
   }$('panels').append(article);
  });
  $('references').replaceChildren();
  const control=(manifest.references||[]).find(r=>r.case_id===current[0]?.case_id)?.regions.find(r=>r.id===$('region').value);
  if(control){
   const heading=document.createElement('h2');heading.textContent='Controles locales CT a 0 µm — no son candidatos';$('references').append(heading);
   const note=document.createElement('p');note.textContent=control.pitch_um+' µm/píxel; planos físicos, no cobertura ni aceptación anatómica.';$('references').append(note);
   const planes=document.createElement('div');planes.className='panels';
   control.planes.forEach((src,i)=>{
    const panel=document.createElement('div');const title=document.createElement('p');title.textContent=control.plane_names[i];panel.append(title);
    const image=new Image();image.src=src;image.width=control.width;image.height=control.height;image.alt='Plano CT '+control.plane_names[i];panel.append(image);
    if(i===0&&control.tangent_missing_reason){const absent=document.createElement('p');absent.textContent=control.tangent_missing_reason;panel.append(absent)}
    planes.append(panel);
   });$('references').append(planes);
   const raw=document.createElement('a');raw.href=control.raw;raw.textContent='Descargar CT, soporte y XYZ originales';$('references').append(raw);
  }
}
$('save').onclick=()=>{
 const region=current.flatMap(p=>p.regions).find(r=>r.id===$('region').value);if(!region)return;
 const quality={};document.querySelectorAll('[data-score]').forEach(s=>{if(s.value!==''){quality[s.dataset.token]??={};quality[s.dataset.token][s.dataset.score]=Number(s.value)}});
 const entry={case_id:current[0].case_id,track:current[0].track,scope:current[0].scope,roi_id:region.id,
 preferred:[...document.querySelectorAll('input[data-token]:checked')].map(x=>x.dataset.token),
 common_depth_um:region.offsets_um[Number($('depth').value)],quality};
 entries=entries.filter(e=>JSON.stringify([e.case_id,e.track,e.scope,e.roi_id])!==JSON.stringify([entry.case_id,entry.track,entry.scope,entry.roi_id]));entries.push(entry);
 $('status').textContent=entries.length+' regiones guardadas; ningún preferido = ninguno aceptable. Descargá los votos antes de cerrar.';
};
$('download').onclick=()=>{const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify({review_sha256:reviewHash,entries},null,2)],{type:'application/json'}));a.download='CT-votes.json';a.click();URL.revokeObjectURL(a.href)};
$('restore').onchange=async e=>{const data=JSON.parse(await e.target.files[0].text());if(data.review_sha256!==reviewHash)throw Error('Votos de otra revisión');entries=data.entries;$('status').textContent='Votos restaurados: '+entries.length};
init().catch(e=>$('status').textContent='Error: '+e.message);
</script>"""


def review_rows(run):
    """Import only explicitly pinned, physically matching historical trials."""
    if any(p.is_symlink() for p in (run, *run.parents)):
        raise ValueError('private run symlink refused')
    queue = read_json(run / 'QUEUE.json')
    stage = queue.get('active_stage')
    def selected(row):
        return not stage or row['stage'] == stage or (stage == 'main' and row['stage'] == 'pilot')
    rows = [dict(r, _source_run=str(run)) for r in queue['rows'] if selected(r)]
    if any(r['stage'] == 'holdout' for r in rows):
        validate_freeze(run)
    config = read_json(run / 'RUN.json').get('config', {})
    references = [('baseline', config['baseline_reference'])] if config.get('baseline_reference') else []
    references += [('trial', reference) for reference in config.get('trial_references', [])]
    for kind, source in references:
        prior = Path(source['run']).absolute()
        if prior == run or any(p.is_symlink() for p in (prior, *prior.parents)):
            raise ValueError('invalid private ' + kind + ' run')
        for name, key in [('RUN.json', 'run_sha256'), ('INVENTORY.json', 'inventory_sha256'), ('QUEUE.json', 'queue_sha256')]:
            if file_sha(prior / name) != source[key]:
                raise ValueError(kind + ' source identity changed: ' + name)
        current_cases = {e['case']['id']: e['case'] for e in read_json(run / 'INVENTORY.json')['entries'] if e['status'] == 'READY'}
        prior_cases = {e['case']['id']: e['case'] for e in read_json(prior / 'INVENTORY.json')['entries'] if e['status'] == 'READY'}
        prior_rows = {r['output']: r for r in read_json(prior / 'QUEUE.json')['rows']}
        for reference in source['rows']:
            row = prior_rows.get(reference.get('output'))
            fields = ('case_id', 'status') if kind == 'baseline' else (
                'case_id', 'field_id', 'resolution_id', 'method', 'scope', 'track', 'stage', 'status')
            if (row is None or any(row[k] != reference.get(k) for k in fields)
                    or (kind == 'baseline' and row['method'] != 'legacy-lscm')):
                raise ValueError(kind + ' row identity changed')
            if kind == 'trial' and row['status'] not in (
                    'COMPLETE', 'RESOURCE_LIMIT', 'FAILED', 'UNAVAILABLE', 'INCOMPATIBLE_GEOMETRY', 'MISSING_INPUT'):
                raise ValueError('trial reference must be a terminal outcome')
            if not selected(row):
                continue
            if row['stage'] == 'holdout':
                validate_freeze(run)
                validate_freeze(prior)
            current, old = current_cases.get(row['case_id']), prior_cases.get(row['case_id'])
            if (not current or not old or current['source_sha256'] != old['source_sha256']
                    or digest(current['ct_profile']) != digest(old['ct_profile'])):
                raise ValueError(kind + ' physical source/CT profile differs')
            if kind == 'trial' and row['status'] == 'COMPLETE' and not all(
                    reference.get(k) for k in ('cell_sha256', 'ct_sha256')):
                raise ValueError('complete trial requires pinned CELL and CT receipts')
            for key, name in [('map_sha256', 'map/MAP.json'), ('metrics_sha256', 'metrics/METRICS.json'),
                              ('cell_sha256', 'CELL.json'), ('ct_sha256', 'ct/CT.json')]:
                if key in reference and file_sha(safe(prior, row['output'] + '/' + name)) != reference[key]:
                    raise ValueError(kind + ' artifact identity changed')
            rows.append(dict(row, _source_run=str(prior)))
    candidates = [(r['case_id'], r['track'], r['scope'], r['method']) for r in rows]
    if len(set(candidates)) != len(candidates):
        raise ValueError('duplicate comparison candidate; select exactly one trial per recipe')
    return rows


def review_snapshot(run, rows):
    snapshot = dict(run=file_sha(run / 'RUN.json'), queue=file_sha(run / 'QUEUE.json'),
                ct={file_sha(Path(r['_source_run']) / 'RUN.json') + '/' + r['output']:
                    file_sha(safe(Path(r['_source_run']), r['output'] + '/ct/CT.json'))
                    for r in rows if r['status'] == 'COMPLETE'})
    if any(r['stage'] == 'holdout' for r in rows):
        roots = {run, *[Path(r['_source_run']) for r in rows if r['stage'] == 'holdout']}
        snapshot['freezes'] = {file_sha(p / 'RUN.json'): file_sha(p / 'FREEZE.json') for p in roots}
    return snapshot


def build_review(run, output, *, controls=None):
    run, output = Path(run).absolute(), Path(output).absolute()
    rows = review_rows(run)
    config = read_json(run / 'RUN.json').get('config', {})
    private_runs = [run, *[Path(r['_source_run']) for r in rows]]
    references = list(config.get('trial_references', []))
    if config.get('baseline_reference'):
        references.append(config['baseline_reference'])
    private_runs.extend(Path(r['run']).absolute() for r in references)
    if any(p.is_relative_to(output) for p in private_runs):
        raise ValueError('served output must be separate from the private run/key location')
    identity = review_snapshot(run, rows)
    if controls is not None:
        controls = Path(controls).absolute()
        if output.is_relative_to(controls) or controls.is_relative_to(output):
            raise ValueError('served output must be separate from reference source controls')
        control_run = verify(controls, 'CONTROL-RUN.json')
        source_runs = {identity['run'], *[file_sha(Path(r['_source_run']) / 'RUN.json') for r in rows]}
        if control_run['identity']['source_run_sha256'] not in source_runs:
            raise ValueError('reference controls belong to another source run')
        identity['controls'] = file_sha(controls / 'CONTROL-RUN.json')
    identity = dict(snapshot=digest(identity))
    with locked(output):
        if (output / 'REVIEW.json').exists():
            return verify(output, 'REVIEW.json', identity)['report']
        panels, keys, files = [], {}, []
        def publish(source):
            name = 'assets/' + uuid.uuid4().hex + source.suffix
            dest = safe(output, name)
            dest.parent.mkdir(exist_ok=True)
            shutil.copyfile(source, dest)
            files.append(name)
            return name
        for row in rows:
            token = uuid.uuid4().hex
            panel = {k: row[k] for k in ('case_id', 'field_id', 'resolution_id', 'track', 'scope', 'stage', 'status')}
            panel.update(token=token, regions=[])
            keys[token] = row
            if row['status'] == 'COMPLETE':
                ctroot = safe(Path(row['_source_run']), row['output'] + '/ct')
                record = verify(ctroot, 'CT.json')
                for region in record['report']['regions']:
                    public = {k: region[k] for k in ('id', 'pitch_um', 'offsets_um', 'missing')}
                    public['tiles'] = []
                    for tile in region['tiles']:
                        published = dict(canvas=tile['canvas'], planes=[])
                        for kind, sources in [('planes', tile['planes']), ('raw', [tile['raw']])]:
                            for original in sources:
                                source = safe(ctroot, original)
                                name = publish(source)
                                if kind == 'planes':
                                    published['planes'].append(name)
                                else:
                                    published['raw'] = name
                        public['tiles'].append(published)
                    panel['regions'].append(public)
            panels.append(panel)
        panels.sort(key=lambda p: p['token'])
        manifest = dict(snapshot=identity['snapshot'], panels=panels, physical_acceptance=False)
        if controls is not None:
            manifest['references'] = []
            for row in control_run['report']['rows']:
                if row['status'] != 'COMPLETE' or not any(
                        p['case_id'] == row['case_id'] and p['field_id'] == row['field_id'] for p in panels):
                    continue
                receipt = safe(controls, row['references'])
                reference = verify(receipt.parent, receipt.name)['report']
                if not reference['diagnostic_only'] or reference['physical_acceptance'] or reference['case_id'] != row['case_id']:
                    raise ValueError('invalid source reference control')
                public = dict(case_id=row['case_id'], field_id=row['field_id'], regions=[])
                for region in reference['regions']:
                    data = {k: region[k] for k in ('id', 'plane_names', 'pitch_um', 'offset_um', 'width', 'height',
                                                  'source_face_row', 'center_xyz_um', 'tangent_missing_reason')}
                    data['raw'] = publish(safe(receipt.parent, region['raw']))
                    data['planes'] = [publish(safe(receipt.parent, name)) for name in region['planes']]
                    public['regions'].append(data)
                manifest['references'].append(public)
        atomic_json(output / 'manifest.json', manifest)
        (output / 'index.html').write_text(HTML)
        atomic_json(run / 'PRIVATE-review-key.json',
                    dict(identity=identity, output=str(output), manifest_sha256=file_sha(output / 'manifest.json'), keys=keys,
                         controls=str(controls) if controls is not None else None))
        seal(output, 'REVIEW.json', identity, manifest, ['manifest.json', 'index.html', *files])
    return manifest


def aggregate_preferences(rows):
    per_case = defaultdict(list)
    for row in rows:
        per_case[(row['field_id'], row['case_id'], row['method'] + '/' + row['scope'])].append(row['score'])
    per_field = defaultdict(lambda: defaultdict(list))
    for (field, case, method), scores in per_case.items():
        per_field[field][method].append(sum(scores) / len(scores))
    fields = {field: {method: sum(scores) / len(scores) for method, scores in methods.items()}
              for field, methods in per_field.items()}
    missing = sorted({'175a', 'paris4'} - fields.keys())
    balanced = {} if missing else {
        method: (fields['175a'][method] + fields['paris4'][method]) / 2
        for method in fields['175a'].keys() & fields['paris4'].keys()}
    return dict(fields=fields, balanced=balanced, missing_fields=missing)


def summarize(run, votes, output):
    run, output = Path(run).absolute(), Path(output).absolute()
    key = read_json(run / 'PRIVATE-review-key.json')
    public = Path(key['output'])
    if output.is_relative_to(public) or public.is_relative_to(output):
        raise ValueError('private unblinded report must not overlap the served gallery')
    verify(public, 'REVIEW.json', key['identity'])
    manifest = read_json(public / 'manifest.json')
    vote = read_json(votes)
    if vote['review_sha256'] != key['manifest_sha256']:
        raise ValueError('vote review identity/hash changed')
    # Re-check current source artifacts too: a sealed gallery must not mask a changed CT run.
    current = review_snapshot(run, review_rows(run))
    if key.get('controls'):
        controls = Path(key['controls'])
        verify(controls, 'CONTROL-RUN.json')
        current['controls'] = file_sha(controls / 'CONTROL-RUN.json')
    if digest(current) != manifest['snapshot']:
        raise ValueError('review source identity changed')
    scores, seen, missing_candidate_regions = [], set(), 0
    for entry in vote['entries']:
        group = (entry['case_id'], entry.get('track', 'A'), entry.get('scope', 'sheet'), entry['roi_id'])
        if group in seen:
            raise ValueError('duplicate ROI vote')
        seen.add(group)
        panels = [p for p in manifest['panels'] if (p['case_id'], p['track'], p['scope']) == group[:3]]
        available = {}
        for panel in panels:
            region = next((r for r in panel['regions'] if r['id'] == entry['roi_id']), None)
            if region and not region['missing'] and panel['status'] == 'COMPLETE':
                if entry['common_depth_um'] not in region['offsets_um']:
                    raise ValueError('common ROI depth differs')
                available[panel['token']] = panel
        if not available:
            raise ValueError('unknown or absent ROI')
        preferred = entry['preferred']
        if len(set(preferred)) != len(preferred) or not set(preferred) <= available.keys():
            raise ValueError('ROI preference contains an unavailable candidate')
        for token, values in entry.get('quality', {}).items():
            if token not in available or any(type(v) is not int or not 0 <= v <= 4 for v in values.values()):
                raise ValueError('invalid ROI quality vote')
        for panel in panels:
            token = panel['token']
            if token not in available:
                missing_candidate_regions += 1
            original = key['keys'][token]
            scores.append(dict(field_id=original['field_id'], case_id=original['case_id'],
                               method=original['method'], scope=original['scope'],
                               score=1 / len(preferred) if token in preferred else 0.))
    result = dict(status='CT_REVIEW_RECORDED_NOT_FINAL', review_sha256=vote['review_sha256'],
                  reviewed_regions=len(seen), missing_candidate_regions=missing_candidate_regions,
                  preferences=aggregate_preferences(scores), physical_acceptance=False,
                  conclusion='No automatic winner; require both-field holdout and human texture judgment.')
    with locked(output):
        atomic_json(output / 'DECISION.json', result)
        with (output / 'RESULTS.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=['field_id', 'case_id', 'method', 'scope', 'score'])
            writer.writeheader()
            writer.writerows(scores)
        (output / 'REPORT.md').write_text('# CT review\n\n' + result['conclusion'] + '\n\n```json\n'
                                         + json.dumps(result, indent=2) + '\n```\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['build', 'summarize'])
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--votes', type=Path)
    parser.add_argument('--controls', type=Path, help='versioned native CT controls for this source run')
    args = parser.parse_args()
    if args.action == 'summarize' and args.votes is None:
        parser.error('--votes required')
    result = build_review(args.run, args.output, controls=args.controls) if args.action == 'build' else summarize(args.run, args.votes, args.output)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
