"""Real compiler/ABI tests: native answers must agree with exact predicates."""
import ctypes as C
from fractions import Fraction as Q
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
import tomllib

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / 'src/winding_field/native'
COMPILER = shutil.which('c++')


def build(output, *args, cwd=None, env=None):
    env = dict(os.environ, PYTHONPATH=str(ROOT / 'src')) if env is None else env
    return subprocess.run([sys.executable, '-m', 'winding_field.tifxyz_build',
                           '--backend', 'cpu', '--output', str(output), *args],
                          cwd=cwd, env=env, text=True, capture_output=True)


@pytest.fixture(scope='module')
def artifact(tmp_path_factory):
    assert importlib.util.find_spec('winding_field.tifxyz_build') is not None, 'native builder missing'
    if not COMPILER:
        pytest.skip('C++ compiler unavailable')
    output = tmp_path_factory.mktemp('native').resolve() / 'artifact'
    result = build(output, '--compiler', COMPILER)
    assert result.returncode == 0, result.stdout + result.stderr
    return output


def load_library(output):
    manifest = json.loads((output / 'accelerator.json').read_text())
    lib = C.CDLL(str(output / manifest['library']))
    lib.wf_accel_abi.argtypes = lib.wf_accel_kind.argtypes = []
    lib.wf_accel_abi.restype = lib.wf_accel_kind.restype = C.c_int
    lib.wf_cpu_filter.argtypes = [C.c_int, C.c_int, C.POINTER(C.c_double), C.POINTER(C.c_int8)]
    lib.wf_cpu_filter.restype = C.c_int
    return lib


@pytest.fixture(scope='module')
def native(artifact):
    return load_library(artifact)


def classify(lib, mode, rows):
    values = (C.c_double * (24 * len(rows)))(*(v for row in rows for v in row))
    output = (C.c_int8 * len(rows))(*([99] * len(rows)))
    assert lib.wf_cpu_filter(mode, len(rows), values, output) == 0
    assert set(output) <= {-1, 0, 1}
    return list(output)


def bounds(q):
    q = Q(q); x = float(q)
    return (math.nextafter(x, -math.inf) if Q(x) > q else x,
            math.nextafter(x, math.inf) if Q(x) < q else x)


def coverage(point=(0., 0., 0.), origin=(0., 0., 0.), unit=1.,
             center=(0., 0., 0.), half2=Q(1, 4), tau=1.):
    intervals = [bounds(q) for q in center]
    row = [*point, *origin, unit, *(b[0] for b in intervals),
           *(b[1] for b in intervals), *bounds(half2), tau, *([0.] * 8)]
    distance2 = sum((Q(c) - (Q(p) * Q(unit) + Q(o))) ** 2
                    for p, o, c in zip(point, origin, center))
    r = Q(tau)**2 - distance2 - Q(half2)
    return row, int(r >= 0 and r*r >= 4*distance2*Q(half2))


def orientation(triangle=((0., 0., 0.), (1., 0., 0.), (0., 1., 0.)),
                unit=1., spacing=(1., 1., 1.), corners=(0., 0., 0., 0., 1., 1., 1., 1.)):
    row = [*(v for p in triangle for v in p), unit, *spacing, *corners, 0., 0., 0.]
    a, b, c = [tuple(Q(v)*Q(unit) for v in p) for p in triangle]
    u, v = [tuple(p[i]-a[i] for i in range(3)) for p in (b, c)]
    normal = [u[(i+1)%3]*v[(i+2)%3] - u[(i+2)%3]*v[(i+1)%3] for i in range(3)]
    lower = sum(min((Q(corners[j+(1 << axis)])-Q(corners[j])) / Q(spacing[axis]) * normal[axis]
                    for j in range(8) if not j & (1 << axis)) for axis in range(3))
    return row, int(lower > 0)


def check_exact(lib, mode, cases, minimum_resolved=0):
    answers = classify(lib, mode, [row for row, _ in cases])
    for answer, (row, expected) in zip(answers, cases):
        assert answer == -1 or answer == expected, (mode, row, answer, expected)
    assert sum(answer != -1 for answer in answers) >= minimum_resolved
    return answers


def test_real_cpu_build_manifest_and_abi(artifact, native):
    manifest = json.loads((artifact / 'accelerator.json').read_text())
    assert set(manifest) == {'schema', 'abi', 'backend', 'max_rows', 'library', 'library_sha256',
                             'library_bytes', 'sources', 'platform', 'compiler'}
    assert manifest['schema'] == 'winding-field-tifxyz-accelerator-v1'
    assert manifest['abi'] == native.wf_accel_abi() == 1
    assert manifest['backend'] == 'cpu' and native.wf_accel_kind() == 0
    assert manifest['max_rows'] == 65536
    assert Path(manifest['library']).name == manifest['library']
    binary = (artifact / manifest['library']).read_bytes()
    assert manifest['library_bytes'] == len(binary)
    assert manifest['library_sha256'] == hashlib.sha256(binary).hexdigest()
    assert manifest['sources'] == {name: hashlib.sha256((SOURCES / name).read_bytes()).hexdigest()
                                   for name in ('predicates.h', 'predicates.cpp', 'predicates.cu')}
    assert manifest['platform'] == dict(system=platform.system(), machine=platform.machine())
    assert isinstance(manifest['compiler']['command'], list)
    assert 0 < len(manifest['compiler']['version']) <= 4096


@pytest.mark.parametrize('key',['NVCC_APPEND_FLAGS','NVCC_PREPEND_FLAGS'])
def test_builder_refuses_ambient_cuda_precision_overrides(tmp_path,key):
    output=tmp_path/'unsafe'
    env=dict(os.environ,PYTHONPATH=str(ROOT/'src'),**{key:'--use_fast_math'})
    run=subprocess.run([sys.executable,'-m','winding_field.tifxyz_build','--backend','cuda',
        '--host-compiler','/absent/compiler','--output',str(output)],env=env,capture_output=True,text=True)
    assert run.returncode!=0 and key in run.stderr
    assert not output.exists()


def test_cpu_validation_precedes_pointer_access(native):
    for mode, n in [(-1, 1), (3, 1), (0, 0), (0, -1), (0, 65537), (2, 2**31-1)]:
        assert native.wf_cpu_filter(mode, n, None, None) < 0
    output = (C.c_int8 * 1)(77)
    assert native.wf_cpu_filter(0, 1, None, output) < 0
    assert output[0] == 77
    row = (C.c_double * 24)(*coverage()[0])
    assert native.wf_cpu_filter(0, 1, row, None) < 0
    for mode in (-1, 3):
        assert native.wf_cpu_filter(mode, 1, row, output) < 0
        assert output[0] == 77
    script = '''
import ctypes as C, sys
lib = C.CDLL(sys.argv[1])
lib.wf_cpu_filter.argtypes = [C.c_int, C.c_int, C.c_void_p, C.c_void_p]
for mode, size in [(-1, 1), (3, 1), (0, 0), (1, -1), (2, 65537)]:
    assert lib.wf_cpu_filter(mode, size, 1, 1) < 0
'''
    result = subprocess.run([sys.executable, '-c', script, native._name], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert classify(native, 0, [coverage()[0]] * 65536) == [1] * 65536


def test_coverage_rational_bounds_equalities_and_dynamic_range(native):
    cases = [coverage(), coverage(point=(2., 0., 0.)),
             coverage(center=(Q(1, 3), 0., 0.), half2=Q(1, 9), tau=2/3),
             coverage(point=(.5, 0., 0.)), coverage(half2=0., tau=1.),
             coverage(point=(-3., 4., 0.), half2=4., tau=7.),
             coverage(point=(1e20, -1e20, 0.), unit=1e20, tau=1e20),
             coverage(point=(1., 0., 0.), origin=(-1e20, 0., 0.), unit=1e20,
                      center=(Q(1, 3), 0., 0.), half2=Q(1, 16)),
             coverage(point=(1e-200, 0., 0.), half2=0., tau=1e-20)]
    for tau in (math.nextafter(1., 0.), 1., math.nextafter(1., math.inf)):
        cases.append(coverage(point=(.5, 0., 0.), tau=tau))
    rng = random.Random(71)
    for _ in range(350):
        cases.append(coverage(point=tuple(rng.uniform(-5, 5) for _ in range(3)),
            origin=tuple(rng.uniform(-5, 5) for _ in range(3)), unit=10**rng.uniform(-8, 8),
            center=tuple(Q(rng.randrange(-30, 30), 3) for _ in range(3)),
            half2=Q(rng.randrange(20), 7), tau=10**rng.uniform(-8, 8)))
    answers = check_exact(native, 0, cases, minimum_resolved=340)
    assert answers[:2] == [1, 0]
    assert answers[3] == -1  # Exact equality must not become a rounded false result.


def test_orientation_axis_order_tiny_gradients_and_cancellation(native):
    cases = [orientation(), orientation(corners=(1.,)*4+(0.,)*4),
             orientation(corners=(0.,)*8), orientation(corners=(0.,)*4+(1e-280,)*4),
             orientation(unit=1e-20, spacing=(1e20, 1e20, 1e20)),
             orientation(triangle=((1e20, 1e20, 1e20),)*3)]
    for axis in range(3):
        triangle = tuple(tuple(float(i == ((axis+k)%3)) if k else 0. for i in range(3))
                         for k in range(3))
        corners = tuple(float(bool(j & (1 << axis))) for j in range(8))
        cases.append(orientation(triangle=triangle, spacing=(.5, 3., 7.), corners=corners))
    rng = random.Random(109)
    for _ in range(350):
        translation = rng.choice((0., 2**40, -2**40))
        triangle = tuple(tuple(translation+rng.uniform(-8, 8) for _ in range(3)) for _ in range(3))
        cases.append(orientation(triangle, unit=10**rng.uniform(-15, 15),
            spacing=tuple(10**rng.uniform(-15, 15) for _ in range(3)),
            corners=tuple(rng.uniform(-1e10, 1e10) for _ in range(8))))
    answers = check_exact(native, 1, cases, minimum_resolved=340)
    assert answers[:2] == [1, 0]
    assert answers[3] == 1


@pytest.mark.parametrize('mode,index,value', [
    (0, 0, math.inf), (0, 1, math.nan), (0, 0, 1e21), (0, 6, 0.), (0, 6, 1e-21),
    (0, 13, -1.), (0, 13, 2.), (0, 7, 1.), (0, 15, -1.), (0, 15, 1e21),
    (1, 9, 0.), (1, 10, 1e-21), (1, 11, -1.), (1, 12, math.inf), (1, 13, 1e21),
])
def test_unsupported_interval_rows_are_unknown(native, mode, index, value):
    row = (coverage if mode == 0 else orientation)()[0]
    row[index] = value
    assert classify(native, mode, [row]) == [-1]


def contact_row(a, b, mask=(0, 0, 0)):
    return [*(float(v) for p in (*a, *b) for v in p), *mask, 0., 0., 0.]


def test_contact_exact_random_and_adversarial(native):
    from winding_field.tifxyz_audit import triangle_contact
    from winding_field.tifxyz_contract import Blocked
    a = ((0., 0., 0.), (2., 0., 0.), (0., 2., 0.))
    fixtures = [
        (a, ((0., 0., 0.), (2., 0., 0.), (0., -2., 0.)), (1, 1, 0)),
        (a, ((0., 0., 0.), (2., 0., 0.), (0., 1., 0.)), (1, 1, 0)),
        (a, ((0., 0., 0.), (2., 0., 0.), (0., 0., 2.)), (1, 1, 0)),
        (a, ((0., 0., 0.), (-2., 0., 0.), (0., -2., 0.)), (1, 0, 0)),
        (a, ((0., 0., 0.), (-2., 0., 0.), (0., -2., 0.)), (0, 0, 0)),
        (a, a, (0, 0, 0)),
        (a, ((.5, .5, -1.), (.5, .5, 1.), (1., .5, 0.)), (0, 0, 0)),
        (a, ((3., 0., 0.), (4., 0., 0.), (3., 1., 0.)), (0, 0, 0)),
    ]
    rng = random.Random(2301)
    for _ in range(420):
        points = np.array([[rng.randint(-8, 8) for _ in range(3)] for _ in range(6)], np.float32)
        if rng.randrange(3) == 0: points[:, 2] = 0
        points *= 2.**rng.randrange(-12, 13)
        shared = rng.randrange(3)
        points[3:3+shared] = points[:shared]
        fixtures.append((points[:3], points[3:], tuple(int(i < shared) for i in range(3))))
    for _ in range(320):
        # General signed float32 mantissas stress the scale/range rejection boundary.
        points = np.array([[rng.uniform(-8, 8) for _ in range(3)] for _ in range(6)], np.float32)
        if rng.randrange(2): points[:, 2] = 0
        fixtures.append((points[:3], points[3:], (0, 0, 0)))
    cases = []
    for a, b, mask in fixtures:
        try:
            expected = int(triangle_contact(a, b, [p for p, bit in zip(a, mask) if bit]))
        except Blocked as error:
            assert error.code == 'DEGENERATE_TRIANGLE'
            expected = -1
        cases.append((contact_row(a, b, mask), expected))
    answers = check_exact(native, 2, cases, minimum_resolved=300)
    assert answers[:6] == [0, 1, 0, 0, 1, 1]
    assert answers[6] == -1  # This noncoplanar crossing needs rational clipping.
    assert all(answer == -1 for answer, (_, expected) in zip(answers, cases) if expected == -1)


def test_contact_bounds_scaling_and_invalid_shared_masks(native):
    a = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0]], np.float32)
    b = a + np.array([0, 0, 1], np.float32)
    good = [contact_row(a, b), contact_row(a*2.**-80, b*2.**-80),
            contact_row(a*2.**60, b*2.**60), contact_row(a*1024+2.**32, b*1024+2.**32)]
    assert classify(native, 2, good) == [0]*len(good)
    bad = [contact_row(a, b, (1, 0, 0)), contact_row(a, a, (1, 1, 1)),
           contact_row(a, a, (.5, 0, 0)), contact_row(a, np.zeros((3, 3))),
           contact_row(a, b*2.**100), contact_row(a, b*2.**-140)]
    for value in (.1, math.inf, math.nan, 1e300, 2.**-149):
        row = contact_row(a, b); row[0] = value; bad.append(row)
    assert classify(native, 2, bad) == [-1]*len(bad)


def test_changed_rounding_environment_is_rejected(native):
    # Use the platform C library's fenv rather than platform-specific FE_* literals.
    process = C.CDLL(None)
    if not all(hasattr(process, name) for name in ('fegetround', 'fesetround')):
        pytest.skip('fenv unavailable')
    original = process.fegetround()
    changed = False
    try:
        for mode in (0x400, 0x800, 0xc00, 0x400000, 0x800000, 0xc00000):
            if process.fesetround(mode) == 0 and process.fegetround() != original:
                changed = True
                row = (C.c_double * 24)(*coverage()[0]); output = (C.c_int8 * 1)(77)
                assert native.wf_cpu_filter(0, 1, row, output) < 0
                assert output[0] == 77
                break
    finally:
        assert process.fesetround(original) == 0
    assert changed


def test_flush_to_zero_environment_is_rejected(native, tmp_path):
    if platform.machine().lower() not in ('arm64', 'aarch64', 'x86_64', 'amd64'):
        pytest.skip('no floating-control register fixture for this architecture')
    source = tmp_path / 'fenv.cpp'
    source.write_text('''
#include <cstdint>
#ifdef __aarch64__
extern "C" uint64_t get_control() { uint64_t x; asm volatile("mrs %0, fpcr" : "=r"(x)); return x; }
extern "C" void set_control(uint64_t x) { asm volatile("msr fpcr, %0" : : "r"(x)); }
#else
#include <xmmintrin.h>
extern "C" uint64_t get_control() { return _mm_getcsr(); }
extern "C" void set_control(uint64_t x) { _mm_setcsr(x); }
#endif
''')
    library = tmp_path / 'fenv.so'
    result = subprocess.run([COMPILER, '-shared', '-fPIC', str(source), '-o', str(library)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    control = C.CDLL(str(library))
    control.get_control.argtypes, control.get_control.restype = [], C.c_uint64
    control.set_control.argtypes, control.set_control.restype = [C.c_uint64], None
    original = control.get_control()
    masks = (1 << 24,) if platform.machine().lower() in ('arm64', 'aarch64') else (1 << 15, 1 << 6)
    row = (C.c_double * 24)(*coverage()[0]); output = (C.c_int8 * 1)(77)
    try:
        for mask in masks:
            control.set_control(original | mask)
            assert native.wf_cpu_filter(0, 1, row, output) < 0
            assert output[0] == 77
    finally:
        control.set_control(original)
    assert classify(native, 0, [coverage()[0]]) == [1]


def test_builder_new_directory_ownership_and_failed_logs(tmp_path, artifact):
    tmp_path = tmp_path.resolve()
    marker = tmp_path / 'existing'; marker.mkdir(); (marker / 'keep').write_text('owned by caller')
    assert build(marker).returncode != 0
    assert (marker / 'keep').read_text() == 'owned by caller'
    link = tmp_path / 'link'; link.symlink_to(tmp_path / 'missing', target_is_directory=True)
    assert build(link).returncode != 0 and not (tmp_path / 'missing').exists()
    parent_link = tmp_path / 'parent-link'; parent_link.symlink_to(marker, target_is_directory=True)
    assert build(parent_link / 'child').returncode != 0 and not (marker / 'child').exists()
    failed = tmp_path / 'failed'
    result = build(failed, '--compiler', str(tmp_path / 'no-compiler'))
    assert result.returncode != 0
    assert failed.is_dir() and (failed / 'build.log').read_text()
    assert not (failed / 'accelerator.json').exists()
    assert build(artifact).returncode != 0


def test_builder_does_not_publish_partial_manifest(tmp_path, monkeypatch):
    from winding_field import tifxyz_build
    if not COMPILER: pytest.skip('C++ compiler unavailable')
    def interrupted_write(manifest, stream, **kwargs):
        stream.write('{"schema":')
        raise OSError('injected interrupted manifest write')
    monkeypatch.setattr(tifxyz_build.json, 'dump', interrupted_write)
    output = tmp_path.resolve() / 'interrupted'
    with pytest.raises(SystemExit):
        tifxyz_build.main(['--backend', 'cpu', '--output', str(output), '--compiler', COMPILER])
    assert not (output / 'accelerator.json').exists()
    assert 'injected interrupted manifest write' in (output / 'build.log').read_text()


@pytest.mark.parametrize('args', [('--unknown', 'yes'), ('--backend', 'bogus'),
                                 ('--host-compiler', '/usr/bin/c++')])
def test_builder_rejects_bad_arguments_before_creating_output(tmp_path, args):
    output = tmp_path.resolve() / 'new'
    assert build(output, *args).returncode != 0
    assert not output.exists()


def test_builder_lazy_imports_and_packaged_assets():
    script = "import sys; import winding_field.tifxyz_build; assert 'numpy' not in sys.modules and 'torch' not in sys.modules"
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True,
                            env=dict(os.environ, PYTHONPATH=str(ROOT / 'src')))
    assert result.returncode == 0, result.stderr
    config = tomllib.loads((ROOT / 'pyproject.toml').read_text())
    patterns = config['tool']['setuptools']['package-data']['winding_field']
    included = {p.name for pattern in patterns for p in (ROOT / 'src/winding_field').glob(pattern)}
    assert {'predicates.h', 'predicates.cpp', 'predicates.cu'} <= included
    assert not any(p.endswith(('.so', '.dylib', '.dll')) for p in included)


def test_builder_uses_installed_package_not_cwd(tmp_path):
    if not COMPILER: pytest.skip('C++ compiler unavailable')
    package = tmp_path.resolve() / 'installed/winding_field'; package.mkdir(parents=True)
    shutil.copy2(ROOT / 'src/winding_field/__init__.py', package)
    shutil.copy2(ROOT / 'src/winding_field/tifxyz_build.py', package)
    shutil.copytree(SOURCES, package / 'native')
    elsewhere = tmp_path.resolve() / 'unrelated'; elsewhere.mkdir()
    output = tmp_path.resolve() / 'built'
    result = build(output, '--compiler', COMPILER, cwd=elsewhere,
                   env=dict(os.environ, PYTHONPATH=str(package.parent)))
    assert result.returncode == 0, result.stdout + result.stderr
    assert classify(load_library(output), 0, [coverage()[0]]) == [1]
    inside = package / 'artifact'
    result = build(inside, cwd=elsewhere, env=dict(os.environ, PYTHONPATH=str(package.parent)))
    assert result.returncode != 0 and not inside.exists()
    with (package / 'native/predicates.cpp').open('a') as stream:
        stream.write('\n#error injected compile failure\n')
    failed = tmp_path.resolve() / 'failed-compile'
    result = build(failed, '--compiler', COMPILER, cwd=elsewhere,
                   env=dict(os.environ, PYTHONPATH=str(package.parent)))
    assert result.returncode != 0 and not (failed / 'accelerator.json').exists()
    assert 'injected compile failure' in (failed / 'build.log').read_text()
