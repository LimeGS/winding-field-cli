#!/usr/bin/env bash
# Cross-host resume probe: run a stage on "host A", move the run like a server copy, resume.
set -uo pipefail
W=/tmp/xh
rm -rf $W /tmp/xhB; mkdir -p $W
python - <<'PY'
from winding_field.demo import synthetic
synthetic("/tmp/xh/ct.zarr")
PY
COMMON=(--source $W/ct.zarr --cache $W/cache --reserve-gib 0 --prediction classical
  --chunk 64 --field-chunk 16 --sweeps 1 --field-coarse-factor 2 --seed-stride 4
  --chart-faces 1024 --nonconvergence-policy continue)
step(){ echo; echo "=== $1"; shift; "$@" > /tmp/out.json 2>/tmp/err.txt; rc=$?; echo "rc=$rc"; tail -3 /tmp/err.txt; python -c "import json;d=json.load(open('/tmp/out.json'));print('status',d.get('status'),'stages',list(d.get('stages',{})))" 2>/dev/null; return 0; }

# 0. Reference: one process, same host
step "REF run to field" winding-field run "${COMMON[@]}" --output $W/ref --device cpu --until field
python -c "import json;print('REF ckpt',json.load(open('$W/ref/FIELD.json'))['checkpoint_sha256'])"

# 1. Host A: prepare+fibres; then simulate copy (tar with fresh inodes, then cp without mtimes)
step "A run until fibres" winding-field run "${COMMON[@]}" --output $W/a --device cpu --until fibres
tar -C / -cf /tmp/a.tar tmp/xh/a tmp/xh/cache && rm -rf $W/a $W/cache && tar -C / -xf /tmp/a.tar
step "B field after tar round trip (same paths)" winding-field field $W/a
python -c "import json;print('A->B ckpt',json.load(open('$W/a/FIELD.json'))['checkpoint_sha256'])"

# 2. Copy without preserving mtimes (cp -r, new mtime/ctime/inode)
step "A2 run until fibres" winding-field run "${COMMON[@]}" --output $W/a2 --device cpu --until fibres
cp -r $W/a2 $W/a2.copy && rm -rf $W/a2 && mv $W/a2.copy $W/a2
step "B2 field after cp -r without mtimes" winding-field field $W/a2 --offline

# 3. Continue a field checkpoint on the other host: 1 sweep, copy, resume --field-sweeps 2
step "A3 run to field 1 sweep" winding-field run "${COMMON[@]}" --output $W/a3 --device cpu --until field
cp -a $W/a3 $W/a3.copy && rm -rf $W/a3 && mv $W/a3.copy $W/a3
step "B3 resume --field-sweeps 2 after copy" winding-field resume $W/a3 --field-sweeps 2
step "REF2 resume --field-sweeps 2 in place" winding-field resume $W/ref --field-sweeps 2
python -c "import json;print('B3 ckpt',json.load(open('$W/a3/FIELD.json'))['checkpoint_sha256']);print('REF2 ckpt',json.load(open('$W/ref/FIELD.json'))['checkpoint_sha256'])"

# 4. Different absolute path (run moved): expected refusal or wrong path
step "A4 run until fibres" winding-field run "${COMMON[@]}" --output $W/a4 --device cpu --until fibres
mkdir -p /tmp/xhB && cp -a $W/a4 /tmp/xhB/a4 && mv $W/a4 $W/a4.gone
step "B4 field at a new path" winding-field field /tmp/xhB/a4
mv $W/a4.gone $W/a4

# 5. Device frozen at run start: auto on a CPU-only host, and explicit cuda on a CPU-only host
step "A5 auto device until fibres" winding-field run "${COMMON[@]}" --output $W/a5 --until fibres
python -c "import json;print('A5 frozen device:',json.load(open('$W/a5/RUN.json'))['config']['device'])"
step "A6 explicit cuda until fibres on CPU host" winding-field run "${COMMON[@]}" --output $W/a6 --device cuda --until fibres
step "A6 field with cuda on CPU host" winding-field field $W/a6
ls $W/a6/field 2>/dev/null | head; python -c "import json;p=json.load(open('$W/a6/field/CHECKPOINT.json'))['progress'];print('A6 init progress',p)" 2>&1 | tail -1
# a cpu-made field checkpoint cannot change device: identity includes device
python - <<'PY'
import json
s=json.load(open('/tmp/xh/a3/field/CHECKPOINT.json'))['identity']
print('identity device:', s.get('device'), 'prior keys:', [k for k in s.get('source_identity',{})])
PY
echo; echo "=== RUN.json absolute paths"; python - <<'PY'
import json
c=json.load(open('/tmp/xh/a/RUN.json'))['config']
print({k:v for k,v in c.items() if isinstance(v,str) and v.startswith('/')})
PY
