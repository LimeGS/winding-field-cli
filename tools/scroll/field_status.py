#!/usr/bin/env python3
"""Readable status of a field being solved (blocked or one-block).

    watch -n 30 -t python3 field_status.py /workspace/SCROLL/coarse

Shows the phase (validating inputs, assembling a block on CPU, solving on GPU), a progress
bar over blocks and sweeps, an ETA from the observed block rate, and the largest change of
each completed sweep. The per-sweep change is a convergence trend, not a distance from the
solution: compare against a one-block solve with compare_fields.py.
"""

import json
import os
import subprocess
import sys
import time

RUN = sys.argv[1].rstrip("/")
NAME = os.path.basename(RUN)
HISTORY = f"/tmp/field-status-{NAME}.json"  # block completion times seen, for the ETA


def sh(command):
    try:
        return subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=10
        ).stdout.strip()
    except Exception:
        return ""


def load(path):
    try:
        with open(path) as stream:
            return json.load(stream)
    except Exception:
        return None


def duration(seconds):
    seconds = int(max(0, seconds))
    hours, minutes = divmod(seconds // 60, 60)
    return f"{hours} h {minutes:02d} min" if hours else f"{minutes} min {seconds % 60:02d} s"


def bar(done, total, width=30):
    total = max(total, 1)
    filled = int(width * min(done, total) / total)
    return "[" + "#" * filled + "." * (width - filled) + f"] {100 * done / total:4.0f}%"


now = time.time()
config = (load(RUN + "/RUN.json") or {}).get("config", {})
chunk, halo = config.get("field_chunk", 64), config.get("field_halo", 2)
checkpoint = load(RUN + "/field/CHECKPOINT.json")
status = load(RUN + "/field/STATUS.json") or {}
# An extended field (resume --field-sweeps N) records its budget in the checkpoint.
sweeps = ((checkpoint or {}).get("identity") or {}).get("sweeps") or config.get("sweeps", 4)
pid = sh(f"pgrep -f '(resume|field|run) .*{RUN}( |$)' | head -1")
elapsed = sh(f"ps -o etimes= -p {pid}") if pid else ""
cpu = sh(f"ps -o pcpu= -p {pid}") if pid else ""
gpu = sh(
    "nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu "
    "--format=csv,noheader,nounits | head -1"
)
gpu_name, gpu_used, gpu_total, gpu_use = ([x.strip() for x in gpu.split(",")] + ["?"] * 4)[:4]
ram = sh("free -g | awk 'NR==2{print $3 \"/\" $2}'")

one_block = checkpoint is not None and all(n <= chunk for n in checkpoint["shape"])
mode = "ONE BLOCK (exact global solve)" if one_block else f"blocks of {chunk}, halo {halo}"
print(f"=== {NAME} ===   {time.strftime('%H:%M:%S', time.gmtime(now))} UTC")
print(f"Mode: {mode} | {sweeps} sweep(s) | rtol {config.get('rtol')}")
print(
    f"Process: {'running for ' + duration(int(elapsed)) if elapsed else 'NOT RUNNING'}"
    + (f" | CPU {float(cpu):.0f}%" if cpu else "")
)
if gpu_used.isdigit():
    print(
        f"GPU: {gpu_name} | memory {int(gpu_used) / 1024:.1f}/{int(gpu_total) / 1024:.0f} GB "
        f"| use {gpu_use}%   RAM {ram} GB"
    )
print()

if os.path.exists(RUN + "/FIELD.json") and status.get("status") != "RUNNING":
    print("FIELD READY (FIELD.json written).")
elif not checkpoint:
    print("Validating inputs (map and fibre hashes, CT metadata); the GPU is not used yet.")
    print("With --offline and a prefetched cache this takes a few minutes.")
else:
    blocks = 1
    for n in checkpoint["shape"]:
        blocks *= (n - 1) // chunk + 1
    progress = checkpoint["progress"]
    sweep = status.get("sweep") or progress.get("sweep") or 1
    block = progress.get("next_block", 0) or 0
    done, total = (sweep - 1) * blocks + block, sweeps * blocks

    history = load(HISTORY) or {"seen": []}
    key = [sweep, block]
    if not history["seen"] or history["seen"][-1]["k"] != key:
        history["seen"].append(
            {"k": key, "done": done, "t": os.path.getmtime(RUN + "/field/CHECKPOINT.json")}
        )
        history["seen"] = history["seen"][-200:]
        try:
            with open(HISTORY, "w") as stream:
                json.dump(history, stream)
        except OSError:
            pass

    on_gpu = gpu_use.isdigit() and int(gpu_use) > 5
    if blocks == 1:
        print("Solving the global system on the GPU" if on_gpu else "Assembling the block on CPU")
    else:
        what = "solving on GPU" if on_gpu else "assembling the block on CPU"
        print(f"Sweep {sweep} of {sweeps} | block {min(block + 1, blocks)} of {blocks}: {what}")
    print(f"Progress: {bar(done, total)}  ({done}/{total} blocks)")

    seen = [s for s in history["seen"] if s["k"][0] == sweep] or history["seen"]
    if len(seen) >= 2 and seen[-1]["done"] > seen[0]["done"]:
        rate = (seen[-1]["t"] - seen[0]["t"]) / (seen[-1]["done"] - seen[0]["done"])
        print(f"Rate: ~{duration(rate)} per block | ETA ~ {duration((total - done) * rate)}")
    else:
        print("ETA: after a couple of blocks finish.")

    solver = progress.get("solver") or {}
    if solver.get("iterations"):
        print(
            f"Last block solve: {solver['iterations']} iterations in "
            f"{solver.get('elapsed_s', 0):.0f} s ({solver.get('reason')})"
        )
    age = now - os.path.getmtime(RUN + "/field/CHECKPOINT.json")
    print(f"Last checkpoint: {duration(age)} ago")

    changes = [row["max_core_update"] for row in progress.get("completed_sweeps", [])]
    if changes:
        print()
        print("Largest field change per completed sweep (windings; tends to 0 as it converges)")
        for index, change in enumerate(changes, 1):
            print(f"  sweep {index}: {change:8.3f}  {'#' * min(40, max(1, int(change)))}")
        if len(changes) >= 2 and changes[-2] > 0:
            ratio = 100 * changes[-1] / changes[-2]
            print(f"  the last sweep left {ratio:.0f}% of the previous change")
