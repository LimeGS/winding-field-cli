import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_uv_selects_hardware_without_manual_backend_exports():
    path = ROOT / "uv.toml"
    assert path.is_file(), "automatic installation defaults are missing"
    config = tomllib.loads(path.read_text())
    assert config["pip"]["torch-backend"] == "auto"
    reader = config["extra-build-variables"]["vc-reader"]
    assert reader["CMAKE_ARGS"] == "-DVC_READER_ENABLE_CUDA=OFF"
    assert reader["CMAKE_BUILD_PARALLEL_LEVEL"] == "2"


def test_installation_does_not_require_build_or_backend_exports():
    install = (
        (ROOT / "README.md")
        .read_text()
        .split("<summary><strong>1. Install</strong></summary>", 1)[1]
        .split("</details>", 1)[0]
    )
    for setting in ("TORCH_BACKEND", "CMAKE_BUILD_PARALLEL_LEVEL", "CMAKE_ARGS"):
        assert setting not in install
    assert "torch.cuda.is_available()" not in install


def test_readme_custom_options_are_collapsed_with_tables_and_help():
    readme = (ROOT / "README.md").read_text()
    heading = "<summary><strong>7. Custom options</strong></summary>"
    assert heading in readme
    before, section = readme.split(heading, 1)
    assert before.rstrip().endswith("<details>")
    section = section.split("</details>", 1)[0]
    assert "Options and results at a glance" in section
    assert "| Option | Meaning |" in section and "| Output | Where to look |" in section
    for command in ["run", "plan", "resume", "export-sheets", "tifxyz", "verify-tifxyz"]:
        assert f"wf {command} --help" in section
    assert "Native output does not create the atlas gallery." not in readme


def test_readme_extra_docs_are_collapsed_without_research_footer():
    readme = (ROOT / "README.md").read_text()
    heading = "<summary><strong>8. Extra docs</strong></summary>"
    assert heading in readme
    before, section = readme.split(heading, 1)
    assert before.rstrip().endswith("<details>")
    section = section.split("</details>", 1)[0]
    for title in [
        "Flattening",
        "Constant-W visual atlas",
        "Lasagna and AFV",
        "Streaming",
        "Prepared workflows",
        "Surface tools",
        "Attribution",
    ]:
        assert f"[{title}](" in section
    assert "Research output: physical sheet correctness" not in readme


def test_readme_badges_credits_and_code_license_are_explicit():
    readme = (ROOT / "README.md").read_text()
    assert (
        "[![Audit](https://github.com/LimeGS/winding-field-cli/actions/workflows/tests.yml/badge.svg?branch=main&event=push)]"
        in readme
    )
    assert "[![Coverage (local):" in readme
    assert "[![License: MIT]" in readme
    credits = readme.split("## Credits", 1)[1].split("## License", 1)[0]
    for tool in ["Lasagna", "Qual", "vc_reader", "Flatboi", "Villa", "libigl", "tifffile"]:
        assert tool in credits
    license = readme.split("## License", 1)[1]
    assert "[MIT](LICENSE)" in license
    assert "repository-authored code" in license and "own licenses" in license


def _wf_function(text):
    start = text.index("wf() {")
    return text[start : text.index("\n}", start) + 2]


def test_readme_installs_with_docker_by_default_and_host_as_custom():
    install = (
        (ROOT / "README.md")
        .read_text()
        .split("<summary><strong>1. Install</strong></summary>", 1)[1]
        .split("</details>", 1)[0]
    )
    docker = install.index("### 1A. Docker (default)")
    host = install.index("<summary><strong>1B. Custom host install</strong>")
    assert install[:host].rstrip().endswith("<details>"), "host install must be collapsible"
    assert (
        docker < host < install.index("#### Quick install") < install.index("#### Manual install")
    )
    default = install[docker:host]
    for command in ("docker buildx bake cpu --load", "wf doctor", "wf demo --output /outputs/demo"):
        assert command in default
    # The README's wrapper is the documented container invocation, not a variant of it.
    assert _wf_function(default) == _wf_function((ROOT / "docs/DOCKER.md").read_text())


def test_readme_examples_use_the_docker_wrapper_outside_the_host_install():
    readme = (ROOT / "README.md").read_text()
    host = readme.index("<summary><strong>1B. Custom host install</strong>")
    outside = readme[:host] + readme[readme.index("</details>", host) :]
    blocks = re.findall(r"```bash\n(.*?)```", outside, re.DOTALL)
    host_form = [
        line for block in blocks for line in block.splitlines() if line.startswith("winding-field ")
    ]
    assert host_form == [], (
        "README examples use `wf` with /data and /outputs; host form lives in 1B"
    )
    assert "wf run --source" in outside and "--output /outputs/" in outside
