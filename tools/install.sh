#!/usr/bin/env bash
# Keep strict setup in a child Bash; only activation touches the caller's shell.
if bash -s -- "$@" <<'WINDING_FIELD_SETUP'
set -euo pipefail

# Parse the complete setup before executing it when downloaded through a pipe.
main() {
    if [[ $(uname -s) != Linux ]] || ! command -v apt-get >/dev/null; then
        echo 'Quick install requires Ubuntu with apt-get (tested on 24.04).' >&2
        exit 1
    fi

    local repo=https://github.com/LimeGS/winding-field-cli.git
    local checkout="$PWD/winding-field-cli"
    local lasagna="$HOME/.cache/winding-field/lasagna-venv"
    local environment
    if [[ -e "$checkout" || -L "$checkout" ]]; then
        if [[ ! -e "$checkout/.git" ]] || [[ $(git -C "$checkout" config --get remote.origin.url) != "$repo" ]]; then
            echo "Refusing to overwrite unrelated path: $checkout" >&2
            exit 1
        fi
    fi
    for environment in "$checkout/.venv" "$lasagna"; do
        if [[ -e "$environment" || -L "$environment" ]] && [[ ! -x "$environment/bin/python" ]]; then
            echo "Refusing to replace incomplete environment: $environment" >&2
            exit 1
        fi
    done

    local -a elevate=()
    if (( EUID != 0 )); then
        sudo -v
        elevate=(sudo)
    fi
    "${elevate[@]}" apt-get update
    "${elevate[@]}" apt-get install -y \
        python3.12-venv python3.12-dev git build-essential cmake ninja-build pkg-config \
        nlohmann-json3-dev libcurl4-openssl-dev libsqlite3-dev \
        libblosc-dev libzstd-dev liblz4-dev zlib1g-dev \
        libtiff-dev libpng-dev libjpeg-dev

    if [[ ! -e "$checkout" ]]; then
        git clone "$repo" "$checkout"
    fi
    cd "$checkout"
    if [[ ! -x .venv/bin/python ]]; then
        python3.12 -m venv .venv
    fi
    .venv/bin/python -m pip install 'uv>=0.9.17'
    .venv/bin/uv pip install --python .venv/bin/python --constraint docker/cli.in .
    .venv/bin/uv python install 3.14
    if [[ ! -x "$lasagna/bin/python" ]]; then
        .venv/bin/uv venv --python 3.14 "$lasagna"
    fi
    .venv/bin/uv pip install --python "$lasagna/bin/python" --constraint docker/cli.in \
        'vesuvius-lasagna @ https://github.com/ScrollPrize/villa/archive/b29f310957c7b29e7d332f9876e0e487819cfaa3.tar.gz#subdirectory=lasagna'
    .venv/bin/winding-field doctor
}

main "$@"
WINDING_FIELD_SETUP
then
    case "${BASH_SOURCE[0]:-}" in
        ''|"$0") printf '\nInstalled. Activate the CLI in your shell:\n  source %q\n' "$PWD/winding-field-cli/.venv/bin/activate" ;;
        *) source "$PWD/winding-field-cli/.venv/bin/activate" && printf '\nCLI installed and activated in this shell.\n' ;;
    esac
else
    case "${BASH_SOURCE[0]:-}" in
        ''|"$0") exit "$?" ;;
        *) return "$?" ;;
    esac
fi
