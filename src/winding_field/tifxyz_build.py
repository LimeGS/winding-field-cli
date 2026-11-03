"""Explicit, standard-library-only native build; never called by the audit runtime."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess


def _sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--backend', choices=('cpu', 'cuda'), required=True)
    parser.add_argument('--output', type=Path, required=True, help='new directory outside the package')
    parser.add_argument('--compiler', help='C++ compiler or nvcc executable')
    parser.add_argument('--host-compiler', help='explicit host C++ compiler (required for CUDA)')
    args = parser.parse_args(argv)
    if args.backend == 'cuda' and not args.host_compiler:
        parser.error('CUDA requires --host-compiler')
    if args.backend == 'cpu' and args.host_compiler:
        parser.error('--host-compiler requires --backend cuda')
    if args.backend == 'cuda':
        for name in ('NVCC_APPEND_FLAGS','NVCC_PREPEND_FLAGS'):
            if os.environ.get(name):parser.error(name+' can override the required numerical flags; unset it')
    package = Path(__file__).resolve().parent
    output = args.output.absolute()
    if output.is_symlink() or output.exists():
        parser.error('--output must be a new directory, not an existing path or symlink')
    if any(parent.is_symlink() for parent in output.parents):
        parser.error('--output parents must not be symlinks')
    if output.resolve().is_relative_to(package):
        parser.error('--output must be outside the installed package')
    if not output.parent.is_dir():
        parser.error('--output parent must exist')
    try:
        output.mkdir(mode=0o700)
    except OSError as error:
        parser.exit(1, f'cannot own output directory: {error}\n')
    try:
        sources = {name: package / 'native' / name for name in ('predicates.h', 'predicates.cpp', 'predicates.cu')}
        hashes = {name: _sha(path) for name, path in sources.items()}
        compiler = args.compiler or ('c++' if args.backend == 'cpu' else 'nvcc')
        compiler = shutil.which(compiler) or compiler
        library = output / 'predicates.so'
        flags = ['-O3', '-std=c++17', '-shared']
        if args.backend == 'cpu':
            command = [compiler, *flags, '-fPIC', '-fno-fast-math', '-ffp-contract=off',
                       str(sources['predicates.cpp']), '-o', str(library)]
        else:
            host = shutil.which(args.host_compiler) or args.host_compiler
            command = [compiler, *flags, '-ccbin', host, '-DWF_WITH_CUDA=1',
                       '-gencode=arch=compute_75,code=sm_75', '-gencode=arch=compute_75,code=compute_75',
                       '--fmad=false', '--ftz=false', '--prec-div=true', '--prec-sqrt=true',
                       '--Werror=cross-execution-space-call',
                       '-Xcompiler=-fPIC,-fno-fast-math,-ffp-contract=off',
                       str(sources['predicates.cpp']), str(sources['predicates.cu']), '-o', str(library)]
        with (output / 'compiler.log').open('wb') as log:
            subprocess.run([compiler, '--version'], stdout=log, stderr=subprocess.STDOUT, check=True)
        with (output / 'compiler.log').open('r', errors='replace') as log:
            version = log.read(4096)
        with (output / 'build.log').open('wb') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        if hashes != {name: _sha(path) for name, path in sources.items()}:
            raise ValueError('native sources changed during compilation')
        lib = ctypes.CDLL(str(library))
        for name in ('wf_accel_abi', 'wf_accel_kind'):
            function = getattr(lib, name)
            function.argtypes, function.restype = [], ctypes.c_int
        if lib.wf_accel_abi() != 1 or lib.wf_accel_kind() != int(args.backend == 'cuda'):
            raise ValueError('compiled library ABI/backend mismatch')
        lib.wf_cpu_filter.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int8)]
        lib.wf_cpu_filter.restype = ctypes.c_int
        row = (ctypes.c_double * 24)(0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., .25, .25, 1.)
        result = (ctypes.c_int8 * 1)(-1)
        if lib.wf_cpu_filter(0, 1, row, result) != 0 or result[0] != 1:
            raise ValueError('compiled library numerical self-check failed')
        manifest = dict(schema='winding-field-tifxyz-accelerator-v1', abi=1, backend=args.backend,
                        max_rows=65536, library=library.name, library_sha256=_sha(library),
                        library_bytes=library.stat().st_size, sources=hashes,
                        platform=dict(system=platform.system(), machine=platform.machine()),
                        compiler=dict(command=command, version=version))
        temporary = output / 'accelerator.json.tmp'
        with temporary.open('x') as stream:
            json.dump(manifest, stream, sort_keys=True, indent=2)
            stream.write('\n')
        temporary.replace(output / 'accelerator.json')
    except (OSError, subprocess.SubprocessError, ValueError, AttributeError) as error:
        with (output / 'build.log').open('a') as log:
            log.write(f'\nBuild failed: {error}\n')
        parser.exit(1, f'build failed; logs preserved in {output}: {error}\n')
    print(output / 'accelerator.json')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
