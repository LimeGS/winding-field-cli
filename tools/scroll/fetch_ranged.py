"""Download one large file with parallel HTTP range requests (HF_TOKEN used if set).

usage: ranged-get.py <url> <dest> <parts> [<expected-size>]
"""

import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

url, dest, parts = sys.argv[1], sys.argv[2], int(sys.argv[3])
headers = (
    {"Authorization": "Bearer " + os.environ["HF_TOKEN"]} if os.environ.get("HF_TOKEN") else {}
)
# Resolve the redirect once (the CDN URL is pre-signed); the token is not sent to the CDN.
head = requests.head(url, headers=headers, allow_redirects=False, timeout=60)
target = head.headers.get("location", url) if head.status_code in (301, 302, 303, 307, 308) else url
size = (
    int(sys.argv[4])
    if len(sys.argv) > 4
    else int(requests.head(target, timeout=60, allow_redirects=True).headers["content-length"])
)
print("size", size, "parts", parts, flush=True)
tmp = dest + ".part"
with open(tmp, "wb") as f:
    f.truncate(size)
step = -(-size // parts)
done_bytes = [0]
lock = threading.Lock()


def get(i):
    lo, hi = i * step, min(size, (i + 1) * step) - 1
    pos = lo
    for attempt in range(20):
        try:
            with requests.get(
                target, headers={"Range": f"bytes={pos}-{hi}"}, stream=True, timeout=120
            ) as r:
                if r.status_code != 206:
                    raise IOError(f"status {r.status_code}")
                with open(tmp, "r+b") as f:
                    f.seek(pos)
                    for chunk in r.iter_content(8 << 20):
                        f.write(chunk)
                        pos += len(chunk)
                        with lock:
                            done_bytes[0] += len(chunk)
            if pos == hi + 1:
                return
        except (requests.RequestException, IOError) as e:
            print("retry part", i, e, flush=True)
            time.sleep(min(30, 2**attempt))
    raise RuntimeError(f"part {i} failed")


start = time.time()
stop = threading.Event()


def report():
    while not stop.wait(30):
        t = time.time() - start
        print(
            f"{t:5.0f}s {done_bytes[0] / 2**30:.1f}/{size / 2**30:.1f} GiB {done_bytes[0] / 2**20 / t:.0f} MiB/s",
            flush=True,
        )


threading.Thread(target=report, daemon=True).start()
with ThreadPoolExecutor(parts) as pool:
    list(pool.map(get, range(parts)))
stop.set()
assert os.path.getsize(tmp) == size
os.replace(tmp, dest)
print(f"DONE {size} bytes {time.time() - start:.0f}s", flush=True)
