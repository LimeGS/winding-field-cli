#!/usr/bin/env python3
"""Check the installed numerical core against provenance/core.json.

    python tools/verify-core.py provenance/core.json
    python tools/verify-core.py --install-url provenance/core.json   # print the pinned URL

The core is installed from its git repository at an exact commit (a GitHub
archive cannot be fetched everywhere and a git source has no archive hash).
Integrity therefore rests on the git commit object id recorded by the
installer plus the per-file SHA-256 of the numerical sources.  Exit status is 1
on any mismatch.
"""

import hashlib
import importlib.util
import json
import sys
from importlib import metadata
from pathlib import Path


def problems(provenance: dict) -> list[str]:
    found = []
    try:
        dist = metadata.distribution(provenance["distribution"])
    except metadata.PackageNotFoundError:
        return [f"{provenance['distribution']} is not installed"]
    if dist.version != provenance["version"]:
        found.append(f"version {dist.version} != {provenance['version']}")
    direct = json.loads(dist.read_text("direct_url.json") or "{}")
    vcs = direct.get("vcs_info") or {}
    if (
        vcs.get("vcs") != "git"
        or direct.get("url", "").removesuffix(".git") != provenance["origin"]
    ):
        found.append(f"not installed from {provenance['origin']} by git: {direct}")
    if vcs.get("commit_id") != provenance["revision"]:
        found.append(f"commit {vcs.get('commit_id')} != {provenance['revision']}")
    for name, expected in provenance["source_sha256"].items():
        spec = importlib.util.find_spec(Path(name).stem)
        if spec is None or spec.origin is None:
            found.append(f"missing core source {name}")
            continue
        actual = hashlib.sha256(Path(spec.origin).read_bytes()).hexdigest()
        if actual != expected:
            found.append(f"{name} sha256 {actual} != {expected}")
    return found


def main(argv: list[str]) -> int:
    if len(argv) == 3 and argv[1] == "--install-url":
        print(json.loads(Path(argv[2]).read_text())["install_url"])
        return 0
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    provenance = json.loads(Path(argv[1]).read_text())
    found = problems(provenance)
    for problem in found:
        print(f"core verification failed: {problem}", file=sys.stderr)
    if not found:
        print(
            f"core {provenance['distribution']} {provenance['version']} @ {provenance['revision']} verified"
        )
    return 1 if found else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
