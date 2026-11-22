"""Download every Lasagna chunk pinned in the run ledgers from its Hugging Face source,
verifying each against the pinned SHA-256; keep local files that already match."""

import hashlib
import json
import os
import sqlite3
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

run, workers = sys.argv[1], int(sys.argv[2])
cfg = json.load(open(run + "/RUN.json"))["config"]
manifest = json.load(open(cfg["lasagna_input"]))
FILL = hashlib.sha256(
    json.dumps("authoritative-zarr-fill", sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
jobs = []
for name, group in manifest["groups"].items():
    db = sqlite3.connect(f"file:{run}/lasagna-import/{name}.sqlite?mode=ro", uri=True)
    for key, sha in db.execute("select key, sha from chunks"):
        if sha != FILL:
            jobs.append(
                (os.path.join(group["zarr"], key), group["missing_source"] + "/" + key, sha)
            )
print(len(jobs), "chunks to have locally", flush=True)

local = threading.local()
stats = dict(kept=0, fetched=0, bytes=0, retries=0, failed=0)
lock = threading.Lock()


def sha_of(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def one(job):
    path, url, sha = job
    if os.path.exists(path) and sha_of(path) == sha:
        with lock:
            stats["kept"] += 1
        return
    s = getattr(local, "s", None)
    if s is None:
        s = requests.Session()
        if os.environ.get("HF_TOKEN"):
            s.headers["Authorization"] = "Bearer " + os.environ["HF_TOKEN"]
        local.s = s
    attempt = 0
    while attempt < 8:
        attempt += 1
        try:
            r = s.get(url, timeout=60)
            if r.status_code == 429:
                # Fixed-window quota: wait for the window HF names, without using an attempt.
                with lock:
                    stats["retries"] += 1
                time.sleep(int(r.headers.get("retry-after", "30")) + 1)
                attempt = 0
                continue
            if r.status_code >= 500:
                with lock:
                    stats["retries"] += 1
                time.sleep(min(60, 2**attempt))
                continue
            r.raise_for_status()
            data = r.content
            if hashlib.sha256(data).hexdigest() != sha:
                raise ValueError("sha mismatch " + url)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = path + ".part"
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, path)
            with lock:
                stats["fetched"] += 1
                stats["bytes"] += len(data)
            return
        except requests.RequestException:
            with lock:
                stats["retries"] += 1
            time.sleep(min(60, 2**attempt))
    with lock:
        stats["failed"] += 1
    print("FAILED", url, flush=True)


start = time.time()
done = threading.Event()


def report():
    while not done.is_set():
        done.wait(30)
        t = max(time.time() - start, 1)
        st = dict(stats)
        n = st["kept"] + st["fetched"] + st["failed"]
        gib = st["bytes"] / 2**30
        rate = st["bytes"] / 2**20 / t
        print(
            f"{n}/{len(jobs)} kept {st['kept']} fetched {st['fetched']} ({gib:.1f} GiB, "
            f"{rate:.0f} MiB/s, {st['fetched'] / t:.0f} files/s) retries {st['retries']} "
            f"failed {st['failed']} {t:.0f}s",
            flush=True,
        )


threading.Thread(target=report, daemon=True).start()
with ThreadPoolExecutor(workers) as pool:
    list(pool.map(one, jobs))
done.set()
print(
    "DONE" if not stats["failed"] else "INCOMPLETE",
    stats,
    f"{time.time() - start:.0f}s",
    flush=True,
)
