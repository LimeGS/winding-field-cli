#!/usr/bin/env bash
# Shared host/container core; bootstrap, OS setup and activation belong to callers.
set -euo pipefail

backend=${1:-auto}
cli=${2:-"$PWD/.venv"}
lasagna=${3:-"$HOME/.cache/winding-field/lasagna-venv"}
phase=${4:-all}
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
root=${root%/} # A checkout at / needs an empty prefix before /docker.
uv=${UV:-uv}
case "$backend:$phase" in
    auto:all|auto:dependencies|auto:application|cpu:all|cpu:dependencies|cpu:application|cu128:all|cu128:dependencies|cu128:application) ;;
    *) echo 'Usage: install-runtime.sh [auto|cpu|cu128] [CLI_ENV] [LASAGNA_ENV] [all|dependencies|application] [WHEEL_DIR]' >&2; exit 2 ;;
esac
# Resolve missing parents too; macOS realpath cannot canonicalize absent paths.
environment_path() {
    local path=$1 parent
    if [[ -d "$path" ]]; then
        (cd -- "$path" && pwd -P)
    else
        parent=$(environment_path "$(dirname -- "$path")") || return
        case $(basename -- "$path") in
            .) printf '%s\n' "$parent" ;;
            ..) dirname -- "$parent" ;;
            *) printf '%s/%s\n' "${parent%/}" "$(basename -- "$path")" ;;
        esac
    fi
}
cli=$(environment_path "$cli")
lasagna=$(environment_path "$lasagna")
if [[ "$cli" == "$lasagna" || "$cli" -ef "$lasagna" ]]; then
    echo 'CLI and Lasagna require separate environments' >&2; exit 1
fi
for environment in "$cli" "$lasagna"; do
    if [[ -e "$environment" || -L "$environment" ]] && [[ ! -x "$environment/bin/python" ]]; then
        echo "Refusing to replace incomplete environment: $environment" >&2; exit 1
    fi
    if [[ "$phase" == application && ! -x "$environment/bin/python" ]]; then
        echo 'Install runtime dependencies before the application' >&2; exit 1
    fi
done
export CMAKE_ARGS=-DVC_READER_ENABLE_CUDA=OFF CMAKE_BUILD_PARALLEL_LEVEL=2
constraints="$root/docker/build-constraints.txt"

if [[ "$phase" != application ]]; then
    if [[ ! -x "$cli/bin/python" ]]; then
        "$uv" python install 3.12.12
        "$uv" venv --python 3.12.12 "$cli"
    fi
    if [[ ! -x "$lasagna/bin/python" ]]; then
        "$uv" python install 3.14.0
        "$uv" venv --python 3.14.0 "$lasagna"
    fi
    options=(--torch-backend "$backend" --build-constraints "$constraints")
    if [[ "$backend" == auto ]]; then
        # Additive installs preserve user packages; image-only sync uses frozen locks.
        "$uv" pip install --python "$cli/bin/python" "${options[@]}" \
            --constraint "$root/docker/cli.in" --requirements "$PWD/pyproject.toml" --extra plot --extra tifxyz
        "$uv" pip install --python "$lasagna/bin/python" "${options[@]}" \
            --constraint "$root/docker/cli.in" --requirements "$root/docker/lasagna.in"
    else
        "$uv" pip sync --python "$cli/bin/python" --require-hashes "${options[@]}" "$root/docker/cli-$backend.txt"
        "$uv" pip sync --python "$lasagna/bin/python" --require-hashes "${options[@]}" "$root/docker/lasagna-$backend.txt"
    fi
fi

if [[ "$phase" != dependencies ]]; then
    wheels=${5:-}
    if [[ -z "$wheels" ]]; then
        wheels=$(mktemp -d "${TMPDIR:-/tmp}/winding-field-wheels.XXXXXX")
        trap 'rm -rf "$wheels"' EXIT
    fi
    "$uv" build --wheel --python "$cli/bin/python" --build-constraints "$constraints" --out-dir "$wheels" "$PWD"
    "$uv" pip install --python "$cli/bin/python" --no-deps "$wheels"/*.whl
    "$uv" pip check --python "$cli/bin/python"
    "$uv" pip check --python "$lasagna/bin/python"
fi
