"""Mirror public S3 prefixes to a local directory: parallel listing and parallel GETs.

usage: s3-mirror.py <bucket-url> <prefix> <dest> <workers> <subpath> [<subpath> ...]
Each subpath (e.g. PHerc0257_cos.ome.zarr/1) is listed per top-level child directory in
parallel; files already present with the listed size are kept. Writes are atomic.
"""

import os
import sys
import threading
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import quote

import requests

bucket, prefix, dest, workers = (
    sys.argv[1].rstrip("/"),
    sys.argv[2].strip("/"),
    sys.argv[3],
    int(sys.argv[4]),
)
subpaths = sys.argv[5:]
NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"
local = threading.local()


def session():
    s = getattr(local, "s", None)
    if s is None:
        s = local.s = requests.Session()
        s.mount(
            "https://",
            requests.adapters.HTTPAdapter(pool_connections=4, pool_maxsize=4, max_retries=5),
        )
    return s


def list_objects(pfx, delimiter=None):
    """Yield (key, size) and common prefixes under pfx."""
    token, keys, dirs = None, [], []
    while True:
        params = {"list-type": "2", "prefix": pfx, "max-keys": "1000"}
        if delimiter:
            params["delimiter"] = delimiter
        if token:
            params["continuation-token"] = token
        for attempt in range(8):
            try:
                r = session().get(bucket + "/", params=params, timeout=60)
                r.raise_for_status()
                break
            except requests.RequestException:
                time.sleep(2**attempt)
        root = ET.fromstring(r.content)
        for c in root.findall(NS + "Contents"):
            keys.append((c.find(NS + "Key").text, int(c.find(NS + "Size").text)))
        for p in root.findall(NS + "CommonPrefixes"):
            dirs.append(p.find(NS + "Prefix").text)
        if root.find(NS + "IsTruncated").text != "true":
            return keys, dirs
        token = root.find(NS + "NextContinuationToken").text


stats = dict(listed=0, kept=0, fetched=0, bytes=0, failed=0)
lock = threading.Lock()


def fetch(key, size):
    rel = key[len(prefix) + 1 :]
    path = os.path.join(dest, rel)
    if os.path.exists(path) and os.path.getsize(path) == size:
        with lock:
            stats["kept"] += 1
        return
    for attempt in range(8):
        try:
            r = session().get(bucket + "/" + quote(key), timeout=120)
            r.raise_for_status()
            data = r.content
            if len(data) != size:
                raise IOError("size mismatch")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = path + ".part"
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, path)
            with lock:
                stats["fetched"] += 1
                stats["bytes"] += size
            return
        except (requests.RequestException, IOError):
            time.sleep(min(30, 2**attempt))
    with lock:
        stats["failed"] += 1
    print("FAILED", key, flush=True)


start = time.time()
done = threading.Event()


def report():
    while not done.wait(30):
        t = time.time() - start
        st = dict(stats)
        print(
            f"{t:5.0f}s listed {st['listed']} kept {st['kept']} fetched {st['fetched']} "
            f"({st['bytes'] / 2**30:.1f} GiB, {st['bytes'] / 2**20 / t:.0f} MiB/s) failed {st['failed']}",
            flush=True,
        )


threading.Thread(target=report, daemon=True).start()
with ThreadPoolExecutor(workers) as pool:
    futures = []
    # Small metadata files at the top of the prediction (manifest, .zattrs, .zgroup).
    keys, dirs = list_objects(prefix + "/", "/")
    for key, size in keys:
        futures.append(pool.submit(fetch, key, size))
    for d in dirs:
        k2, _ = list_objects(d, "/")
        futures += [pool.submit(fetch, k, s) for k, s in k2]
    for sub in subpaths:
        base = f"{prefix}/{sub}/"
        top_keys, top_dirs = list_objects(base, "/")
        futures += [pool.submit(fetch, k, s) for k, s in top_keys]
        with ThreadPoolExecutor(32) as lister:
            for keys, _ in lister.map(lambda d: list_objects(d), top_dirs):
                with lock:
                    stats["listed"] += len(keys)
                futures += [pool.submit(fetch, k, s) for k, s in keys]
        print(f"listed {sub}: {stats['listed']} objects so far", flush=True)
    for f in as_completed(futures):
        f.result()
done.set()
print(
    ("DONE" if not stats["failed"] else "INCOMPLETE"),
    stats,
    f"{time.time() - start:.0f}s",
    flush=True,
)
