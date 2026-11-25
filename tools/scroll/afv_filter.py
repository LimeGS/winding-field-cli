"""Write a new AFV with only the fibres strictly longer than a minimum length.

usage: afv-filter.py <source.afv> <dest.afv> <min_mm>

Fibres, their blocks and block bounds are copied unchanged (geometry untouched); the
header counts, uuid and provenance are updated, as Studio's "filter_strictly_greater_than_1mm"
derivative does. Fibre length is the AFV's own `fibers.length` (source voxels).
"""

import json
import os
import sqlite3
import sys
import time
import uuid

src, dst, min_mm = sys.argv[1], sys.argv[2], float(sys.argv[3])
if os.path.exists(dst):
    sys.exit("destination exists: " + dst)
t0 = time.time()
s = sqlite3.connect(f"file:{src}?mode=ro&immutable=1", uri=True)
header = {k: json.loads(v) for k, v in s.execute("SELECT key, value FROM metadata")}
root = header["root"]
voxel_mm = root.get("voxel_size_mm") or root["vc_open_data_source_original_resolution"] / 1000.0
threshold = min_mm / voxel_mm  # in source voxels
print(f"voxel {voxel_mm} mm -> keep fibres with length > {threshold:.3f} voxels", flush=True)
schema = [
    (name, sql)
    for name, sql in s.execute("SELECT name, sql FROM sqlite_master WHERE sql IS NOT NULL")
    if not name.startswith("block_bounds_")  # rtree shadow tables come with the virtual table
]
app_id = s.execute("PRAGMA application_id").fetchone()[0]
user_version = s.execute("PRAGMA user_version").fetchone()[0]
page_size = s.execute("PRAGMA page_size").fetchone()[0]
s.close()

tmp = dst + ".part"
if os.path.exists(tmp):
    os.remove(tmp)
d = sqlite3.connect(tmp, isolation_level=None)
d.execute(f"PRAGMA page_size={page_size}")
d.execute("PRAGMA journal_mode=OFF")
d.execute("PRAGMA synchronous=OFF")
d.execute(f"PRAGMA application_id={app_id}")
d.execute(f"PRAGMA user_version={user_version}")
d.execute("ATTACH DATABASE ? AS s", (f"file:{src}?mode=ro&immutable=1",))
d.execute("BEGIN")
for name, sql in schema:
    if not sql.upper().startswith("CREATE INDEX"):
        d.execute(sql)
d.execute("INSERT INTO metadata SELECT key, value FROM s.metadata")
d.execute("INSERT INTO fibers SELECT * FROM s.fibers WHERE length > ? ORDER BY id", (threshold,))
print(f"fibres copied {time.time() - t0:.0f}s", flush=True)
d.execute(
    "INSERT INTO blocks SELECT b.* FROM s.blocks b WHERE b.fiber_id IN (SELECT id FROM fibers) ORDER BY b.id"
)
print(f"blocks copied {time.time() - t0:.0f}s", flush=True)
d.execute(
    "INSERT INTO block_bounds SELECT bb.* FROM s.block_bounds bb WHERE bb.id IN (SELECT id FROM blocks) ORDER BY bb.id"
)
print(f"bounds copied {time.time() - t0:.0f}s", flush=True)
for name, sql in schema:
    if sql.upper().startswith("CREATE INDEX"):
        d.execute(sql)
fibres, points = d.execute("SELECT count(*), coalesce(sum(point_count), 0) FROM fibers").fetchone()
blocks = d.execute("SELECT count(*) FROM blocks").fetchone()[0]
length_mm = d.execute("SELECT coalesce(sum(length), 0) FROM fibers").fetchone()[0] * voxel_mm


def put(key, value):
    d.execute("INSERT OR REPLACE INTO metadata VALUES (?, ?)", (key, json.dumps(value)))


put("fiber_count", fibres)
put("point_count", points)
put("block_count", blocks)
put("uuid", str(uuid.uuid4()))
put(
    "provenance",
    {
        "operation": f"filter_strictly_greater_than_{min_mm:g}mm",
        "minimumLengthMm": min_mm,
        "minimumLengthVoxels": threshold,
        "lengthSource": "fibers.length (source voxels) x voxel_size_mm",
        "parentFile": os.path.basename(src),
        "parentUUID": header.get("uuid"),
        "parentFiberCount": header.get("fiber_count"),
        "parentPointCount": header.get("point_count"),
        "parentProvenance": header.get("provenance"),
        "keptFiberCount": fibres,
        "keptPointCount": points,
        "keptBlockCount": blocks,
        "keptLengthMm": length_mm,
        "geometryModified": False,
    },
)
d.execute("COMMIT")
d.execute("DETACH DATABASE s")
ok = d.execute("PRAGMA integrity_check").fetchall()
d.close()
if ok != [("ok",)]:
    sys.exit(f"integrity check failed: {ok[:3]}")
os.replace(tmp, dst)
print(
    f"DONE kept {fibres:,} fibres ({fibres / header['fiber_count']:.1%}), {points:,} points, "
    f"{blocks:,} blocks, {length_mm / 1000:.1f} m of {header['fiber_count']:,} fibres; "
    f"{os.path.getsize(dst) / 2**30:.1f} GiB in {time.time() - t0:.0f}s",
    flush=True,
)
