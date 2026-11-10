"""Bounded independent prepared-level processes; each child owns a normal mesh bundle."""

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import re
import select
import signal
import shutil
import subprocess
import sys
import time
import uuid

from field_runtime import ResourceUnavailable, check_resources
from field_store import atomic_json

from .surface_mesh import MeshReader, sha
from .surface_sources import FieldSource
from .surface_workflow import _invalid, _object, load_surface_config, run_surfaces


def _mounted_v1(controller, proc, groups):
    entries = [line.split(':', 2) for line in (proc / 'self/cgroup').read_text().splitlines()]
    relative = Path(next(row[2] for row in entries if len(row) == 3 and controller in row[1].split(',')))
    if '..' in relative.parts or not relative.is_absolute():
        raise ValueError('invalid controller cgroup')
    candidates = []
    for line in (proc / 'self/mountinfo').read_text().splitlines():
        before, after = line.split(' - ', 1)
        fields, options = before.split(), after.split()
        if options[0] != 'cgroup' or controller not in options[2].split(','):
            continue
        decode = lambda text: re.sub(r'\\([0-7]{3})', lambda match: chr(int(match[1], 8)), text)
        mounted_root, mount = Path(decode(fields[3])), Path(decode(fields[4]))
        if mount.is_relative_to(groups) and relative.is_relative_to(mounted_root):
            candidates.append((len(mounted_root.parts), mount, mount / relative.relative_to(mounted_root)))
    if not candidates:
        return None
    _, root, leaf = max(candidates, key=lambda item: item[0])
    return root, leaf


def _v1_memory(proc=Path('/proc'), groups=Path('/sys/fs/cgroup')):
    try:
        mounted = _mounted_v1('memory', proc, groups)
        if mounted is None:
            return dict(known=False)
        root, directory = mounted
        match = re.search(r'^MemAvailable:\s+(\d+)\s+kB\s*$', (proc / 'meminfo').read_text(), re.M)
        if match is None:
            raise ValueError('missing host memory headroom')
        host = int(match[1]) * 1024
        available = [host]
        leaf = True
        while True:
            if leaf or (directory / 'memory.limit_in_bytes').exists():
                cap = int((directory / 'memory.limit_in_bytes').read_text())
                used = int((directory / 'memory.usage_in_bytes').read_text())
                if min(cap, used) < 0:
                    raise ValueError('invalid memory cgroup')
                if cap < (1 << 63) - 4096:
                    available.append(max(0, cap - used))
            if directory == root:
                break
            directory = directory.parent
            leaf = False
        return dict(known=True, host_available_bytes=host, effective_headroom_bytes=min(available))
    except (OSError, ValueError, StopIteration, IndexError):
        return dict(known=False)


def cpu_limit(proc=Path('/proc'), groups=Path('/sys/fs/cgroup'), *, affinity=None):
    affinity = affinity or (len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count() or 1)
    if not sys.platform.startswith('linux') and proc == Path('/proc'):
        return affinity
    try:
        entries = (proc / 'self/cgroup').read_text().splitlines()
        unified = next((line.split('::', 1)[1] for line in entries if line.startswith('0::')), None)
        relative = unified if unified is not None else next(
            line.split(':', 2)[2] for line in entries if 'cpu' in line.split(':', 2)[1].split(','))
        root = groups if unified is not None else groups / 'cpu'
        if '..' in Path(relative).parts:
            raise ValueError('invalid CPU cgroup')
        directory = root / relative.lstrip('/')
        if unified is None or not (directory / 'cpu.max').exists():
            mounted = _mounted_v1('cpu', proc, groups)
            if mounted is not None:
                root, directory = mounted
                unified = None
        limits = [affinity]
        leaf = True
        while True:
            name = 'cpu.max' if unified is not None else 'cpu.cfs_quota_us'
            if leaf or (directory / name).exists():
                if unified is not None:
                    quota, period = (directory / name).read_text().split()
                    quota = None if quota == 'max' else int(quota)
                else:
                    quota = int((directory / name).read_text())
                    quota = None if quota == -1 else quota
                    period = (directory / 'cpu.cfs_period_us').read_text()
                period = int(period)
                if period <= 0 or quota is not None and quota <= 0:
                    raise ValueError('invalid CPU quota')
                if quota is not None:
                    limits.append(max(1, quota // period))
            if directory == root:
                break
            if root not in directory.parents:
                raise ValueError('CPU cgroup outside mount')
            directory = directory.parent
            leaf = False
        return min(limits)
    except (OSError, ValueError, StopIteration, IndexError):
        return 1


def resources(config, render, output, active=()):
    from .tifxyz_runtime import memory_report, process_rss

    execution = config['execution']
    cells = math.prod(min(execution['tile'], n - 1) + 1 for n in config['geometry']['shape_zyx'])
    pixels = config['render']['width'] * config['render']['height'] if render else 0
    # shortcut: conservative tile/workspace estimates, calibrate for unusually dense tiles before increasing concurrency.
    work = (256 << 20) + cells * 1024 + execution['cache_bytes'] + pixels * 64
    staging = (32 << 20) + cells * 640 + pixels * 16
    memory = memory_report()
    if not memory['known']:
        memory = _v1_memory()
    cpus = cpu_limit()
    slots = max(0, cpus - len(active))
    if memory['known']:
        unallocated = 0
        for process in active:
            try:
                rss = process_rss(process.pid)
            except RuntimeError:
                rss = 0
            unallocated += max(0, work - rss)
        host = memory['host_available_bytes'] - (2 << 30) - unallocated
        effective = memory['effective_headroom_bytes'] - (128 << 20) - unallocated
        slots = min(slots, max(0, min(host, effective) // work))
    else:
        slots = min(slots, max(0, 1 - len(active)))
    disk = shutil.disk_usage(output).free - execution['reserve_bytes'] - staging * len(active)
    slots = min(slots, max(0, disk // staging))
    return dict(slots=slots, cpus=cpus, memory_known=memory['known'],
                worker_bytes=work, staging_bytes=staging)


def _safe(path):
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('parallel collection paths cannot be symlinks')


def _read(path):
    _safe(path)
    with path.open('rb') as stream:
        raw = stream.read((16 << 20) + 1)
    if len(raw) > 16 << 20:
        raise ValueError('parallel inventory exceeds 16 MiB')
    return json.loads(raw, object_pairs_hook=_object, parse_constant=_invalid)


def _products(root):
    result = {}
    for path in sorted(root.rglob('*')):
        _safe(path)
        if path.is_file() and not any(p.startswith('.') for p in path.relative_to(root).parts) and path.name != 'STOP':
            result[str(path.relative_to(root))] = sha(path)
    return result


def _mesh_identity(identity, level, tile):
    from . import native_mesh as native

    return dict(source=identity, levels=[level], tile=tile,
                total_tiles=math.prod((n - 2) // tile + 1 for n in identity['geometry']['shape_zyx']),
                algorithm='lewiner-shifted-f32-adaptive-global-edges-v2+batched-store-v1',
                numpy=native.np.__version__, skimage=native.version('scikit-image'),
                storage_sha256=sha(native.__file__),
                code={'surface_mesh.py': sha(native.core.__file__),
                      'surface_sources.py': sha(Path(__file__).with_name('surface_sources.py')),
                      'field_store.py': sha(native.core.field_store.__file__)},
                decoder_runtime=native.hashlib.sha256(native.marshal.dumps(native.core.field_store._decode.__code__)).hexdigest())


def _verify(root, level, identity, tile, *, rendered=False, products=None):
    _safe(root)
    with MeshReader(root) as reader:
        manifest = reader.manifest
        if (manifest['identity'] != _mesh_identity(identity, level, tile)
                or [r['level'] for r in manifest['levels']] != [level]):
            raise ValueError('parallel child source/level binding differs')
    if not (root / 'level-0000.obj').is_file():
        raise ValueError('parallel child OBJ missing')
    if rendered and not (root / 'renders/RENDER.json').is_file():
        raise ValueError('parallel child render receipt missing')
    actual = _products(root)
    if products is not None and actual != products:
        raise ValueError('parallel child products changed')
    return actual


def _seal(root, config_sha, render, identity, level, tile):
    seal = _read(root / '.parallel-ready.json')
    if (not isinstance(seal, dict) or seal.get('config_sha256') != config_sha
            or seal.get('render') is not render or not isinstance(seal.get('products'), dict)):
        raise ValueError('parallel completion seal differs')
    return _verify(root, level, identity, tile, rendered=render, products=seal['products'])


def _child_config(config, level):
    child = {key: config[key] for key in ('schema', 'field', 'geometry', 'execution', 'render')}
    child['levels'] = [level]
    child['render'] = dict(config['render'])
    if 'levels' in child['render']:
        child['render']['levels'] = [level]
    return child


def run_parallel_levels(config_path, output, *, resume=False, render=False, stop=None):
    if type(resume) is not bool or type(render) is not bool or stop is not None and not callable(stop):
        raise ValueError('invalid parallel execution controls')
    config = load_surface_config(config_path)
    if 'field' not in config:
        raise ValueError('parallel levels require a prepared field')
    selected = config['render'].get('levels', config['levels'])
    if render and any(level not in config['levels'] for level in selected):
        raise ValueError('requested render level absent from collection')
    output = Path(output).absolute()
    _safe(output)
    check_resources(output.parent, reserve_bytes=config['execution']['reserve_bytes'] + (1 << 20))
    with FieldSource(config['field'], config['geometry'], cache_bytes=0) as source:
        source.validate(hashes=True)
        identity = source.identity
    binding = dict(config_sha256=config['config_sha256'], config_path=config['config_path'], render=render,
                   source=identity, code={name: sha(Path(__file__).with_name(name)) for name in
                   ('surface_parallel.py', 'surface_workflow.py', 'native_mesh.py', 'surface_mesh.py', 'surface_sources.py')})
    child_configs = [_child_config(config, level) for level in config['levels']]
    if not resume:
        output.mkdir(exist_ok=False)
    elif not output.is_dir():
        raise ValueError('parallel resume requires an existing collection')
    lock_path = output / '.levels.lock'
    _safe(lock_path)
    with os.fdopen(os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW, 0o600), 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        inventory = output / 'LEVELS.json'
        if resume:
            result = _read(inventory)
            if (not isinstance(result, dict) or result.get('schema') != 'winding-parallel-levels-v1'
                    or result.get('binding') != binding or not isinstance(result.get('levels'), list)
                    or len(result['levels']) != len(child_configs)):
                raise ValueError('parallel collection identity differs')
            for i, (record, child) in enumerate(zip(result['levels'], child_configs)):
                if (not isinstance(record, dict) or record.get('id') != i or record.get('level') != config['levels'][i]
                        or record.get('path') != f'level-{i:04d}'
                        or record.get('state') not in {'pending', 'running', 'paused', 'ready', 'failed'}):
                    raise ValueError('invalid parallel level inventory')
                path = output / 'configs' / f'level-{i:04d}.json'
                if _read(path) != child or sha(path) != record.get('config_sha256'):
                    raise ValueError('parallel child configuration changed')
                root = output / record['path']
                _safe(root)
                if record['state'] == 'ready':
                    if not isinstance(record.get('products'), dict):
                        raise ValueError('missing parallel product inventory')
                    _verify(root, record['level'], identity, config['execution']['tile'], rendered=render and record['level'] in selected,
                            products=record['products'])
                elif record['state'] != 'pending':
                    if not root.is_dir() or not (root / 'RUN.json').is_file() or not (root / 'mesh.sqlite').is_file():
                        raise ValueError('missing/ambiguous started parallel child; preserve and use a new output')
                    if (root / '.parallel-ready.json').exists():
                        record['products'] = _seal(root, record['config_sha256'], render and record['level'] in selected,
                                                   identity, record['level'], config['execution']['tile'])
                        record['state'] = 'ready'
                    elif (root / 'MESH.json').exists():
                        if not render or record['level'] not in selected or (root / 'renders/RENDER.json').exists():
                            raise ValueError('ambiguous completed parallel child; completion seal missing')
                        _verify_mesh_seal(root, record['config_sha256'])
                elif root.exists():
                    raise ValueError('untracked parallel child output')
            if all(record['state'] == 'ready' for record in result['levels']):
                if result['status'] != 'LEVELS_READY':
                    result['status'] = 'LEVELS_READY'
                    atomic_json(inventory, result)
                return result
        else:
            (output / 'configs').mkdir()
            (output / 'logs').mkdir()
            result = dict(schema='winding-parallel-levels-v1', binding=binding, status='EXTRACTING',
                          peak_workers=0, physical_acceptance=False, levels=[])
            for i, (level, child) in enumerate(zip(config['levels'], child_configs)):
                path = output / 'configs' / f'level-{i:04d}.json'
                atomic_json(path, child)
                result['levels'].append(dict(id=i, level=level, path=f'level-{i:04d}', state='pending',
                                             config_sha256=sha(path), products=None))
            atomic_json(inventory, result)
        return _supervise(config, output, result, identity, selected, render, stop)


def _supervise(config, output, result, identity, selected, render, stop):
    active = {}
    cancelled = False
    failed = False
    resource_blocked = False
    cancel_read, cancel_write = os.pipe()
    handlers = {}

    def interrupted(signum, frame):
        nonlocal cancelled
        cancelled = True

    def changed(record=None):
        if sha(config['config_path']) != config['config_sha256']:
            raise ValueError('surface config changed during parallel execution')
        if record is not None:
            path = output / 'configs' / f"level-{record['id']:04d}.json"
            _safe(path)
            if sha(path) != record['config_sha256']:
                raise ValueError('parallel child config changed during execution')

    def request_stop():
        nonlocal cancel_write
        if cancel_write is not None:
            os.close(cancel_write)
            cancel_write = None

    def save():
        changed()
        atomic_json(output / 'LEVELS.json', result)

    try:
        import threading
        if threading.current_thread() is threading.main_thread():
            for sig in (signal.SIGINT, signal.SIGTERM):
                handlers[sig] = signal.signal(sig, interrupted)
        todo = [record for record in result['levels'] if record['state'] != 'ready']
        while todo or active:
            changed()
            cancelled = cancelled or (output / 'STOP').exists() or (stop is not None and bool(stop()))
            if cancelled or failed:
                request_stop()
            if not cancelled and not failed:
                while todo:
                    if any(p.poll() is not None for p in active):
                        break
                    if cancelled or (output / 'STOP').exists() or (stop is not None and bool(stop())):
                        cancelled = True
                        request_stop()
                        break
                    observation = resources(config, render, output, tuple(active))
                    result['resources'] = observation
                    if observation['slots'] < 1:
                        if not active:
                            resource_blocked = True
                        break
                    record = todo.pop(0)
                    changed(record)
                    child = output / record['path']
                    was_started = record['state'] != 'pending'
                    record['state'] = 'running'
                    save()
                    command = [sys.executable, '-m', 'winding_field.surface_parallel', '--child',
                               str(output / 'configs' / f"level-{record['id']:04d}.json"),
                               '--output', str(child), '--cancel-fd', str(cancel_read), '--collection', str(output),
                               '--child-sha', record['config_sha256'],
                               '--original-config', config['config_path'], '--original-sha', config['config_sha256']]
                    if was_started:
                        command.append('--resume')
                    if render and record['level'] in selected:
                        command.append('--render')
                    environment = os.environ.copy()
                    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                                'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS'):
                        environment[key] = '1'
                    log = output / 'logs' / f"level-{record['id']:04d}-{uuid.uuid4().hex}.log"
                    _safe(log)
                    with log.open('x') as stream:
                        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                                   env=environment, start_new_session=True, pass_fds=(cancel_read,))
                    active[process] = record
                    changed(record)
                    result['peak_workers'] = max(result['peak_workers'], len(active))
                    save()
            for process, record in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                process.wait()
                del active[process]
                root = output / record['path']
                record['exit_code'] = code
                if code == 0:
                    changed(record)
                    record['products'] = _seal(root, record['config_sha256'], render and record['level'] in selected,
                                               identity, record['level'], config['execution']['tile'])
                    record['state'] = 'ready'
                elif code == 3:
                    record['state'] = 'paused'
                    cancelled = True
                else:
                    record['state'] = 'failed'
                    failed = True
                    resource_blocked = resource_blocked or code == 75
                save()
            if (cancelled or failed or resource_blocked) and not active:
                break
            if active:
                time.sleep(0.05)
        changed()
        for record in result['levels']:
            changed(record)
        with FieldSource(config['field'], config['geometry'], cache_bytes=0) as source:
            source.validate(hashes=True)
        result['status'] = ('LEVELS_READY' if all(r['state'] == 'ready' for r in result['levels']) else
                            'RESOURCE_UNAVAILABLE' if resource_blocked else 'LEVELS_FAILED' if failed else 'CHECKPOINTED_STOP')
        save()
        return result
    finally:
        try:
            request_stop()
            for process in active:
                process.wait()
        finally:
            os.close(cancel_read)
            for sig, handler in handlers.items():
                signal.signal(sig, handler)


def _verify_mesh_seal(root, config_sha):
    seal = _read(root / '.parallel-mesh.json')
    if not isinstance(seal, dict) or seal.get('config_sha256') != config_sha or not isinstance(seal.get('products'), dict):
        raise ValueError('parallel mesh completion seal differs')
    for name, digest in seal['products'].items():
        path = Path(name)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('invalid parallel mesh product path')
        _safe(root / path)
        if sha(root / path) != digest:
            raise ValueError('parallel mesh products changed')


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--child', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cancel-fd', type=int, required=True)
    parser.add_argument('--child-sha', required=True)
    parser.add_argument('--collection', type=Path, required=True)
    parser.add_argument('--original-config', type=Path, required=True)
    parser.add_argument('--original-sha', required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()

    def stopped():
        if sha(args.original_config) != args.original_sha:
            raise ValueError('original parallel config changed')
        if sha(args.child) != args.child_sha:
            raise ValueError('parallel child config changed')
        return bool(select.select([args.cancel_fd], [], [], 0)[0]) or (args.collection / 'STOP').exists()

    try:
        stopped()
        config = load_surface_config(args.child)
        if args.resume and (args.output / 'MESH.json').exists():
            _verify_mesh_seal(args.output, args.child_sha)
            result = _read(args.output / 'MESH.json')
        else:
            result = run_surfaces(args.child, args.output, resume=args.resume, render=False, stop=stopped)
            if result['status'] == 'MESH_READY':
                atomic_json(args.output / '.parallel-mesh.json', dict(config_sha256=args.child_sha, products=_products(args.output)))
        if result['status'] == 'MESH_READY':
            if args.render:
                from .surface_workflow import render_bundle

                render_bundle(args.output, args.output / 'renders',
                              settings={**config['render'], 'reserve_bytes': config['execution']['reserve_bytes']})
            stopped()
            atomic_json(args.output / '.parallel-ready.json', dict(config_sha256=args.child_sha,
                        render=args.render, products=_products(args.output)))
        print(json.dumps(result, allow_nan=False))
        return 0 if result['status'] in {'MESH_READY', 'SURFACES_READY'} else 3
    except ResourceUnavailable as error:
        print(str(error), file=sys.stderr)
        return 75


if __name__ == '__main__':
    raise SystemExit(_main())
