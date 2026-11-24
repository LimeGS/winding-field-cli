"""Prefetch CT chunks from the source into a local winding-field CT cache, in parallel.

Runs read CT with ``--offline`` from this cache: an online run re-verifies every cached
chunk against the source one connection at a time (tens of minutes per resume).

    prefetch_ct.py SOURCE CACHE --voxel-um 8.64 --level 2                  # whole level
    prefetch_ct.py SOURCE CACHE --voxel-um 8.64 --level 0 --bounds Z0 Z1 Y0 Y1 X0 X1
    prefetch_ct.py SOURCE CACHE --voxel-um 8.64 --level 2 --ledger RUN     # chunks a run pinned

Give ``--cache-gib`` above the final cache size (the reader evicts down to its budget) and
pass the same or a larger ``--cache-gib`` to every run that uses the cache.
"""

import argparse
import itertools
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from winding_field.workflow import open_volume

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("source")
parser.add_argument("cache")
parser.add_argument("--voxel-um", type=float, required=True)
parser.add_argument("--level", type=int, default=0)
parser.add_argument("--bounds", type=int, nargs=6, metavar=("Z0", "Z1", "Y0", "Y1", "X0", "X1"))
parser.add_argument("--ledger", help="run directory: fetch the chunks in its source.sqlite")
parser.add_argument("--workers", type=int, default=32)
parser.add_argument("--cache-gib", type=float, default=900.0)
args = parser.parse_args()

volume = open_volume(
    dict(
        source=args.source,
        voxel_um=args.voxel_um,
        level=args.level,
        cache=args.cache,
        cache_bytes=int(args.cache_gib * 2**30),
    )
)
# Runs also read the metadata of the finer levels (level 0 sets the frame); an offline run
# refuses a cache without it, so fetch those small documents too.
for finer in range(args.level):
    other = open_volume(
        dict(
            source=args.source,
            voxel_um=args.voxel_um,
            level=finer,
            cache=args.cache,
            cache_bytes=int(args.cache_gib * 2**30),
        )
    )
    getattr(other, "close", lambda: None)()
if args.ledger:
    ledger = sqlite3.connect(f"file:{args.ledger}/source.sqlite?mode=ro", uri=True)
    keys = [key for (key,) in ledger.execute("SELECT key FROM chunks")]
else:
    low = args.bounds[0::2] if args.bounds else (0, 0, 0)
    high = args.bounds[1::2] if args.bounds else volume.shape
    separator = volume.metadata.get("dimension_separator", ".")
    prefix = volume.array_path + "/" if volume.array_path else ""
    keys = [
        prefix + separator.join(map(str, index))
        for index in itertools.product(
            *(range(a // c, (b - 1) // c + 1) for a, b, c in zip(low, high, volume.chunks))
        )
    ]
start, done, present, data = time.time(), 0, 0, 0
print(f"level {args.level} shape {volume.shape}: {len(keys)} chunks", flush=True)
with ThreadPoolExecutor(args.workers) as pool:
    for future in as_completed([pool.submit(volume._native_bytes, key) for key in keys]):
        raw = future.result()
        done += 1
        if raw is not None:
            present += 1
            data += len(raw)
        if done % 2000 == 0 or done == len(keys):
            print(
                f"{done}/{len(keys)} present {present} {data / 2**30:.1f} GiB "
                f"{time.time() - start:.0f}s",
                flush=True,
            )
print("DONE", flush=True)
