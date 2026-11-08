from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def test_uv_selects_hardware_without_manual_backend_exports():
    path = ROOT / 'uv.toml'
    assert path.is_file(), 'automatic installation defaults are missing'
    config = tomllib.loads(path.read_text())
    assert config['pip']['torch-backend'] == 'auto'
    reader = config['extra-build-variables']['vc-reader']
    assert reader['CMAKE_ARGS'] == '-DVC_READER_ENABLE_CUDA=OFF'
    assert reader['CMAKE_BUILD_PARALLEL_LEVEL'] == '2'


def test_installation_does_not_require_build_or_backend_exports():
    install = (ROOT / 'README.md').read_text().split(
        '<summary><strong>1. Install</strong></summary>', 1
    )[1].split('</details>', 1)[0]
    for setting in ('TORCH_BACKEND', 'CMAKE_BUILD_PARALLEL_LEVEL', 'CMAKE_ARGS'):
        assert setting not in install
    assert 'python -m pip install --upgrade pip uv' in install
    assert 'torch.cuda.is_available()' not in install
