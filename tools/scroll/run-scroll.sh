#!/usr/bin/env bash
# One scroll, one machine: counted -> coarse -> plan -> tiles -> merge -> gate -> meshes -> renders.
# Every stage is resumable: rerun the script and it continues where it stopped.
#
#   ROOT=/workspace/175A ./run-scroll.sh            # all stages
#   ROOT=/workspace/175A ./run-scroll.sh tiles      # one stage
#
# Configuration comes from $ROOT/scroll.env (see scroll.env.example).
#
# CT is read from LOCAL caches only during runs (--offline): prefetch level 2 (tiles,
# TILE_CACHE) and level 0 (renders, $ROOT/ct-cache-l0) from S3 in parallel beforehand.
# Online mode re-verifies every cached chunk against S3 one at a time (tens of minutes).
#
# Two scrolls on one machine (e.g. 128 cores / 256 GB): give each its share and GPU:
#   CUDA_VISIBLE_DEVICES=0 RENDER_JOBS=48 CACHE_BYTES=20000000000 ROOT=/workspace/A ./run-scroll.sh
#   CUDA_VISIBLE_DEVICES=1 RENDER_JOBS=48 CACHE_BYTES=20000000000 ROOT=/workspace/B ./run-scroll.sh
# Tiles are sized to fill each GPU and run one per GPU (JOBS_PER_GPU to share a GPU).
# CLEANUP=1 deletes tile inputs after the merge. Full meshes are always dropped per batch.
set -euo pipefail

ROOT=${ROOT:?set ROOT to the scroll work directory}
# shellcheck disable=SC1091
source "$ROOT/scroll.env"
: "${CT:?}" "${VOXEL_UM:?}" "${AFV:?}" "${LASAGNA:?}" "${BOUNDS_L2:?}"
CODE=${CODE:-/workspace/wf-cli-dev}
CORES=${CORES:-$(nproc)}
# Tiles at once: default one per visible GPU (times JOBS_PER_GPU), decided by scroll tiles.
TILE_JOBS=${TILE_JOBS:-}
RENDER_JOBS=${RENDER_JOBS:-$((CORES / 2))}
MERGE_WORKERS=${MERGE_WORKERS:-$((CORES < 32 ? CORES : 32))}
CLEANUP=${CLEANUP:-0}
COARSE_CHUNK=${COARSE_CHUNK:-256}
COARSE_SWEEPS=${COARSE_SWEEPS:-16}
# Blocks exchange information only through their overlap. With the default 2-node halo a
# whole-scroll coarse stays windings away from the solution (PHerc0175A: median 11 after
# 8 sweeps); halo 32 reaches 0.38 after 16 sweeps. Prefer a one-block coarse solved on a
# large GPU (COARSE_CHUNK >= the grid, COARSE_SWEEPS=1), or give its run as COARSE_DIR.
COARSE_HALO=${COARSE_HALO:-32}
# Largest fraction of seam nodes that may differ by more than half a winding (gate stage).
SEAM_LIMIT=${SEAM_LIMIT:-0.05}
# Render mesh decimation (1/N of the faces; 1 disables).
DECIMATE=${DECIMATE:-9}
# Slab rendered around each level, in micrometres (0: the single surface layer).
THICKNESS_UM=${THICKNESS_UM:-70}
# The coarse prior: built here by the coarse stage, or an existing run (e.g. a one-block
# global coarse solved on a larger GPU) given as COARSE_DIR with .done-coarse present.
COARSE_DIR=${COARSE_DIR:-$ROOT/coarse}
# Optional prefetched CT cache shared by every tile (e.g. level 2 in /dev/shm). Tiles then
# run --offline with a budget above the cache size, so none of them evicts it.
TILE_CACHE=${TILE_CACHE:-}
TILE_CACHE_GIB=${TILE_CACHE_GIB:-20}
FINE_SWEEPS=${FINE_SWEEPS:-1}
# Sheet density counted on Lasagna cos (October 2026 recipe) instead of Lasagna grad_mag,
# which reads 1.4-1.7x the real sheets on PHerc0175A. COUNTED=0 keeps grad_mag.
COUNTED=${COUNTED:-1}
COUNTED_JOBS=${COUNTED_JOBS:-1}
# Fibre weight of the fine tiles. 1000 keeps levels on one sheet at 34.56 um; the coarse
# keeps weight 1 (at 138 um strong fibres compress the winding count).
FIBRE_WEIGHT=${FIBRE_WEIGHT:-1000}
# Fine tiles are solved as one block each (sweeps of 64-node blocks drift ~1 sheet with
# strong fibres). TILE_NODES=auto picks the largest tile the GPUs hold with AMG (6 GB:
# 128-node cores, 24 GB: 256, 80 GB: 384), Jacobi below a 192-node tile;
# GPU_MEMORY_GIB plans for GPUs other than this machine's.
TILE_NODES=${TILE_NODES:-auto}
OVERLAP_NODES=${OVERLAP_NODES:-32}
# auto: AMG above a 192-node tile, Jacobi below (amg or jacobi force one).
TILE_PRECONDITIONER=${TILE_PRECONDITIONER:-auto}
JOBS_PER_GPU=${JOBS_PER_GPU:-1}
GPU_MEMORY_GIB=${GPU_MEMORY_GIB:-}
# Solver caches: the first sweep builds, later sweeps reuse (~2.5x cheaper on 175A).
CACHE_BYTES=${CACHE_BYTES:-40000000000}
# Native install: PYTHON=/workspace/venv/bin/python (CLI built from the branch).
# Development/container: PYTHON=python with the checkout first on PYTHONPATH.
PYTHON=${PYTHON:-/workspace/venv/bin/python}
[[ -x "$PYTHON" ]] || PYTHON=python
# The branch source always wins over the installed wheel, so a code fix synced into
# $CODE reaches the next stage without reinstalling.
export PYTHONPATH="$CODE/src${PYTHONPATH:+:$PYTHONPATH}"
WF=("$PYTHON" -m winding_field)
LOG="$ROOT/logs"
mkdir -p "$LOG"
read -r Z0 Z1 Y0 Y1 X0 X1 <<<"$BOUNDS_L2"
# Fine field at the level-2 CT spacing (4 voxels); coarse at stride 4 of level 2 (16 voxels).
FINE_UM=$("$PYTHON" -c "from decimal import Decimal; print(Decimal('$VOXEL_UM') * 4)")
COARSE_UM=$("$PYTHON" -c "from decimal import Decimal; print(Decimal('$VOXEL_UM') * 16)")

say() { echo "[$(date -Is)] $*" | tee -a "$LOG/run-scroll.log"; }
done_stage() { [[ -f "$ROOT/.done-$1" ]]; }
mark() { touch "$ROOT/.done-$1"; say "stage $1 done"; }

# The Lasagna manifest every field stage reads: the counted one once it exists.
lasagna() {
  if [[ "$COUNTED" == 1 ]]; then echo "$ROOT/counted/counted.lasagna.json"; else echo "$LASAGNA"; fi
}

counted() {
  [[ "$COUNTED" == 1 ]] || return 0
  done_stage counted && return
  say "counted: sheet density on Lasagna cos, $COUNTED_JOBS processes"
  local pids=() k
  for ((k = 0; k < COUNTED_JOBS; k++)); do
    local gpu=()
    [[ -n "${COUNTED_GPUS:-}" ]] && gpu=("CUDA_VISIBLE_DEVICES=$(cut -d, -f$((k + 1)) <<<"$COUNTED_GPUS")")
    env "${gpu[@]}" "${WF[@]}" scroll counted-density --lasagna-input "$LASAGNA" --output "$ROOT/counted" \
      --bounds "$((Z0 * 4))" "$((Z1 * 4))" "$((Y0 * 4))" "$((Y1 * 4))" "$((X0 * 4))" "$((X1 * 4))" \
      --part "$k" "$COUNTED_JOBS" >>"$LOG/counted-$k.log" 2>&1 &
    pids+=($!)
  done
  for k in "${pids[@]}"; do wait "$k" || true; done
  # The last part to finish writes the manifest; a rerun of one part completes it.
  [[ -f "$ROOT/counted/counted.lasagna.json" ]] \
    || "${WF[@]}" scroll counted-density --lasagna-input "$LASAGNA" --output "$ROOT/counted" \
      --bounds "$((Z0 * 4))" "$((Z1 * 4))" "$((Y0 * 4))" "$((Y1 * 4))" "$((X0 * 4))" "$((X1 * 4))" \
      >>"$LOG/counted.log" 2>&1
  [[ -f "$ROOT/counted/counted.lasagna.json" ]] || { say "counted failed; see $LOG/counted*.log"; exit 1; }
  mark counted
}

coarse() {
  done_stage coarse && return
  if [[ -f "$COARSE_DIR/RUN.json" ]]; then
    say "coarse: continuing"
    # Local cache only: an online continuation re-verifies every used chunk against S3.
    "${WF[@]}" field "$COARSE_DIR" --offline >>"$LOG/coarse.log" 2>&1
  else
    say "coarse: level 2 stride 4, chunk $COARSE_CHUNK, halo $COARSE_HALO, $COARSE_SWEEPS sweeps"
    "${WF[@]}" run --source "$CT" --voxel-um "$VOXEL_UM" --level 2 --stride 4 \
      --roi "$Z0" "$Z1" "$Y0" "$Y1" "$X0" "$X1" --field-spacing-um "$COARSE_UM" \
      --fibres "$AFV" --lasagna-input "$(lasagna)" --device cuda \
      --sweeps "$COARSE_SWEEPS" --field-chunk "$COARSE_CHUNK" --field-halo "$COARSE_HALO" \
      --constraint-cache-bytes "$CACHE_BYTES" --observation-cache-bytes "$CACHE_BYTES" \
      --prepared-cache-bytes "$((CACHE_BYTES / 4))" \
      --threads 8 --import-workers 0 --until field --cache-gib "${CT_CACHE_GIB:-20}" \
      --cache "$ROOT/ct-cache-l2" --output "$COARSE_DIR" >>"$LOG/coarse.log" 2>&1
  fi
  [[ -f "$COARSE_DIR/FIELD.json" ]] || { say "coarse failed; see $LOG/coarse.log"; exit 1; }
  mark coarse
}

plan() {
  done_stage plan && return
  rm -rf "$ROOT/plan"
  local extra=("--run-arg=--fibre-weight=$FIBRE_WEIGHT" --jobs-per-gpu "$JOBS_PER_GPU"
    --tile-preconditioner "$TILE_PRECONDITIONER")
  [[ -n "$GPU_MEMORY_GIB" ]] && extra+=(--gpu-memory-gib "$GPU_MEMORY_GIB")
  [[ -n "$TILE_CACHE" ]] && extra+=(--run-arg=--offline "--run-arg=--cache-gib=$TILE_CACHE_GIB")
  say "plan: $FINE_UM um tiles on coarse $COARSE_DIR${TILE_CACHE:+ (tiles offline on $TILE_CACHE)}"
  "${WF[@]}" scroll plan --source "$CT" --voxel-um "$VOXEL_UM" --level 2 \
    --field-spacing-um "$FINE_UM" --bounds "$Z0" "$Z1" "$Y0" "$Y1" "$X0" "$X1" \
    --tile-nodes "$TILE_NODES" --overlap-nodes "$OVERLAP_NODES" --coarse "$COARSE_DIR" "${extra[@]}" \
    --fibres "$AFV" --lasagna-input "$(lasagna)" --sweeps "$FINE_SWEEPS" \
    --threads 4 --import-workers 4 --output "$ROOT/plan" | tee -a "$LOG/plan.log"
  mark plan
}

tiles() {
  done_stage tiles && return
  say "tiles: ${TILE_JOBS:-one per GPU} at a time"
  local cache=()
  [[ -n "$TILE_CACHE" ]] && cache=(--cache "$TILE_CACHE")
  [[ -n "$TILE_JOBS" ]] && cache+=(--jobs "$TILE_JOBS")
  "${WF[@]}" scroll tiles "$ROOT/plan" "${cache[@]}" | tee -a "$LOG/tiles.log"
  grep -q '"TILES_DONE"' <(tail -1 "$LOG/tiles.log") || { say "tiles incomplete"; exit 1; }
  mark tiles
}

merge() {
  done_stage merge && return
  rm -rf "$ROOT/merged"
  say "merge"
  "${WF[@]}" scroll merge "$ROOT/plan" --output "$ROOT/merged" --workers "$MERGE_WORKERS" \
    | tee -a "$LOG/merge.log"
  [[ -f "$ROOT/merged/FIELD.json" ]] || { say "merge failed"; exit 1; }
  if [[ "$CLEANUP" == 1 ]]; then
    # Keep each tile's field, receipts and logs; drop its prepared inputs (maps, imports).
    say "cleanup: tile inputs"
    find "$ROOT/plan/tiles" -mindepth 2 -maxdepth 2 \( -name maps -o -name lasagna \
      -o -name lasagna-import -o -name fibres -o -name products.sqlite \) -exec rm -rf {} +
    rm -rf "$ROOT/plan/cache"
  fi
  mark merge
}

gate() {
  done_stage gate && return
  # Tiles that disagree in their overlaps make level surfaces jump between sheets.
  "$PYTHON" "$(dirname "$(readlink -f "$0")")/seam_gate.py" "$ROOT/merged/SEAMS.json" "$SEAM_LIMIT" \
    | tee -a "$LOG/run-scroll.log" || { say "seam gate failed: surfaces not started"; exit 3; }
  mark gate
}

# Surfaces in batches: mesh BATCH levels, render them, drop their full-resolution meshes
# (the decimated copy stays with each render). Full meshes of every level at once would
# need ~245 bytes per face, ~1.6 TB for PHerc0175A.
BATCH=${BATCH:-$RENDER_JOBS}

render_one() {
  local bundle=$1 out=$2
  local WF=("$PYTHON" -m winding_field)
  [[ -f "$out/LEVEL.json" ]] && return 0
  "${WF[@]}" scroll render-ct --mesh "$bundle" --source "$CT" --voxel-um "$VOXEL_UM" \
    --level 0 --render-pitch-um "$VOXEL_UM" --decimate "$DECIMATE" \
    --thickness-um "$THICKNESS_UM" --offline \
    --cache "$ROOT/ct-cache-l0" --cache-gib "${RENDER_CACHE_GIB:-900}" --output "$out" \
    >"$LOG/render-$(basename "$out").log" 2>&1 \
    || echo "render $out failed" >>"$LOG/render-failures.log"
  # Keep the tile PNGs and overview; the per-tile CT arrays were only for publishing.
  [[ -f "$out/LEVEL.json" ]] && find "$out" -path '*/tiles/*.npz' -delete
  return 0
}
export -f render_one
export ROOT CT VOXEL_UM LOG CODE PYTHON PYTHONPATH RENDER_CACHE_GIB DECIMATE THICKNESS_UM

surfaces() {
  done_stage surfaces && return
  mkdir -p "$ROOT/renders" "$ROOT/meshes"
  local total
  total=$("$PYTHON" -c "import json,sys; print(len(json.load(open(sys.argv[1]))['levels']))" \
    "$ROOT/merged/surface-run.json")
  total=$((total < ${RENDER_LIMIT:-100000} ? total : ${RENDER_LIMIT:-100000}))
  say "surfaces: $total levels, $BATCH per batch, $RENDER_JOBS renders at a time"
  for ((first = 0; first < total; first += BATCH)); do
    local batch="$ROOT/meshes/batch-$(printf %06d "$first")"
    if [[ ! -f "$batch.done" ]]; then
      "$PYTHON" - "$ROOT/merged/surface-run.json" "$first" "$BATCH" "$batch.json" <<'PY2'
import json, sys
config = json.load(open(sys.argv[1]))
first, size = int(sys.argv[2]), int(sys.argv[3])
config["levels"] = config["levels"][first:first + size]
json.dump(config, open(sys.argv[4], "w"))
PY2
      [[ -d "$batch" ]] && resume=(--resume) || resume=()
      "${WF[@]}" prepared mesh --config "$batch.json" --output "$batch" --parallel-levels \
        "${resume[@]}" >>"$LOG/meshes.log" 2>&1
      find "$batch" -maxdepth 1 -name 'level-*' -type d | sort | while read -r level; do
        echo "$level $ROOT/renders/level-$(printf %06d $((first + 10#${level##*-})))"
      done | xargs -P "$RENDER_JOBS" -L1 bash -c 'render_one "$@"' _
      touch "$batch.done"
    fi
    # Full-resolution meshes of rendered levels are no longer needed.
    rm -rf "$batch"
  done
  mark surfaces
}

stages=("${@:-counted coarse plan tiles merge gate surfaces}")
for stage in ${stages[*]}; do "$stage"; done
say "finished: ${stages[*]}"
