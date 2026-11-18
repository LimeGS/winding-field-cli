"""Share identical immutable libraries while assembling an image, before final COPY."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile


def same_bytes(a, b):
    with a.open('rb') as left, b.open('rb') as right:
        while True:
            chunk = left.read(1 << 20)
            if chunk != right.read(1 << 20):
                return False
            if not chunk:
                return True


def deduplicate(roots):
    roots = [Path(root).absolute() for root in roots]
    for root in roots:
        if any(path.is_symlink() for path in (root, *root.parents)):
            raise ValueError(f'symlink root is not allowed: {root}')
        if not root.is_dir():
            raise ValueError(f'not a directory: {root}')
    groups = {}
    visited = set()
    for root in roots:
        for directory, _, names in os.walk(root, followlinks=False):
            for name in names:
                if not (name.endswith(('.so', '.a')) or '.so.' in name):
                    continue
                path = Path(directory) / name
                if path in visited:
                    continue
                visited.add(path)
                info = path.lstat()
                if stat.S_ISREG(info.st_mode) and info.st_size:
                    key = (info.st_dev, info.st_size, info.st_mode, info.st_uid, info.st_gid)
                    groups.setdefault(key, []).append(path)
    result = dict(files=0, bytes=0)
    for paths in groups.values():
        if len(paths) < 2:
            continue
        canonical = {}
        # Keep existing hardlink groups canonical regardless of traversal order.
        for path in sorted(paths, key=lambda path: -path.stat().st_nlink):
            with path.open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').digest()
            original = canonical.setdefault(digest, path)
            if path.samefile(original):
                continue
            if not same_bytes(original, path):
                raise ValueError(f'library digest collision: {path}')
            info = path.stat()
            with tempfile.TemporaryDirectory(prefix='.deduplicate-', dir=path.parent) as temporary:
                link = Path(temporary) / 'library'
                os.link(original, link)
                os.replace(link, path)
            result['files'] += 1
            if info.st_nlink == 1:
                result['bytes'] += info.st_size
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('roots', nargs='+', type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps(deduplicate(args.roots), sort_keys=True))
    except (OSError, ValueError) as error:
        parser.exit(1, f'{error}\n')
