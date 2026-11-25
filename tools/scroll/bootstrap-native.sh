#!/usr/bin/env bash
# Native install (no Docker) on a stock Ubuntu 24.04 + NVIDIA host, e.g. a Vast instance.
# Uses the repository's own recipe: tools/ubuntu-packages.txt and tools/install-runtime.sh
# with the hash-locked cu128 dependency sets, the verified core commit, and the CLI wheel
# built from the copied checkout (feat/scroll-tiles). Then provisions Flatboi (SLIM).
#
#   CODE=/workspace/wf-cli-dev ./bootstrap-native.sh
set -euo pipefail
CODE=${CODE:-/workspace/wf-cli-dev}
VENV=${VENV:-/workspace/venv}
LASAGNA_VENV=${LASAGNA_VENV:-/workspace/lasagna-venv}
BOOT=${BOOT:-/workspace/bootstrap}
export DEBIAN_FRONTEND=noninteractive
SUDO=; [[ $(id -u) == 0 ]] || SUDO=sudo

. /etc/os-release
[[ "$ID" == ubuntu && ( "$VERSION_ID" == 24.04 || "$VERSION_ID" == 22.04 ) ]] || {
  echo "needs Ubuntu 22.04 or 24.04 (found $PRETTY_NAME)" >&2; exit 1; }

# Build headers/libraries for vc_reader; Python itself comes from uv (3.12.12 / 3.14.0),
# so the distribution's python3.12 packages (absent on 22.04) are not needed.
$SUDO apt-get update
$SUDO apt-get install -y --no-install-recommends ca-certificates rsync python3 python3-venv \
  $(grep -v '^python3.12' "$CODE/tools/ubuntu-packages.txt")

# uv from PyPI into a throwaway venv (the README's bootstrap), not a piped installer.
if [[ ! -x "$BOOT/bin/uv" ]]; then
  python3 -m venv "$BOOT"
  "$BOOT/bin/python" -m pip install --upgrade pip 'uv==0.9.17'
fi
export UV="$BOOT/bin/uv"
# Keep uv's Python interpreters and cache next to the environments: the venvs link to
# the interpreter, so it must survive with /workspace (not only in ~/.local).
export UV_PYTHON_INSTALL_DIR="$BOOT/python" UV_CACHE_DIR="$BOOT/uv-cache"

cd "$CODE"
bash tools/install-runtime.sh cu128 "$VENV" "$LASAGNA_VENV" all
"$VENV/bin/python" -m winding_field.flattening.flatboi --provision
"$VENV/bin/python" -m winding_field doctor
"$VENV/bin/python" -c "import torch; assert torch.cuda.is_available(), 'CUDA not visible'; \
print('cuda', torch.cuda.get_device_name(0))"
echo "native install ready: $VENV/bin/python -m winding_field"
