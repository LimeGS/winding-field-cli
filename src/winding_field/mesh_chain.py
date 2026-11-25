"""Chain a flattening export after mesh extraction as a separate, second step.

The mesh step keeps its own receipts and status. Only a MESH_READY bundle starts the
second step, which is an ordinary ``export-sheets`` process with its own output,
receipts and exit code. ``CHAIN.json`` in the mesh output records both steps.
"""

import json
from pathlib import Path
import subprocess
import sys
import time

from .util import atomic_json

SCHEMA = "winding-field-mesh-chain-v1"


FRAME_KEYS = ("output_frame_id", "flattening_config")


def _frame_flags(mesh_output, cli_config):
    """The mesh publication frame, as Studio uses it, unless the settings choose one."""
    if cli_config is not None:
        from .cli_settings import read_settings

        if any(key in read_settings(cli_config) for key in FRAME_KEYS):
            return []
    receipt = Path(mesh_output) / "MESH.json"
    if not receipt.is_file():
        return []
    metadata = json.loads(receipt.read_text()).get("metadata", {})
    frame, origin, voxel = (metadata.get(k) for k in ("frame", "origin_xyz", "voxel_um"))
    if not isinstance(frame, str) or not isinstance(origin, list) or len(origin) != 3:
        return []
    return [
        "--output-frame-id",
        frame,
        "--output-origin-xyz-um",
        *map(str, origin),
        "--output-voxel-um",
        str(voxel),
    ]


def flatten_command(mesh_output, flatten_output, cli_config=None):
    command = [sys.executable, "-m", "winding_field", "export-sheets"]
    if cli_config is not None:
        command += ["--cli-config", str(Path(cli_config).absolute())]
    command += _frame_flags(mesh_output, cli_config)
    command += ["--mesh-bundle", str(mesh_output), "--output", str(flatten_output)]
    if Path(flatten_output).exists():
        command.append("--resume")
    return command


def _last_json(text):
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except ValueError:
                return None
    return None


def check_chain(mesh_output, flatten_output, cli_config=None):
    """Refuse a chain that cannot start, before the mesh step runs."""
    mesh_output = Path(mesh_output).absolute()
    flatten_output = Path(flatten_output).absolute()
    if cli_config is not None and not Path(cli_config).is_file():
        raise ValueError("--flatten-cli-config is not a file: " + str(cli_config))
    if flatten_output == mesh_output or mesh_output in flatten_output.parents:
        raise ValueError("--then-flatten must be outside the mesh output")
    return mesh_output, flatten_output


def chain_flatten(mesh_output, mesh_result, flatten_output, cli_config=None):
    """Run the flattening step after a ready mesh; returns the chain record."""
    mesh_output, flatten_output = check_chain(mesh_output, flatten_output, cli_config)
    record = dict(
        schema=SCHEMA,
        steps=[
            dict(step="mesh", output=str(mesh_output), status=mesh_result.get("status")),
            dict(step="flatten", output=str(flatten_output), status="NOT_STARTED"),
        ],
    )
    flatten = record["steps"][1]
    if mesh_result.get("status") != "MESH_READY":
        flatten["reason"] = "mesh step is not MESH_READY"
        atomic_json(mesh_output / "CHAIN.json", record)
        return record
    command = flatten_command(mesh_output, flatten_output, cli_config)
    flatten.update(status="RUNNING", command=command, started_epoch=time.time())
    atomic_json(mesh_output / "CHAIN.json", record)
    completed = subprocess.run(command, capture_output=True, text=True)
    sys.stderr.write(completed.stderr)
    result = _last_json(completed.stdout)
    flatten.update(
        returncode=completed.returncode,
        finished_epoch=time.time(),
        status=(result or {}).get("status") or ("DONE" if completed.returncode == 0 else "FAILED"),
    )
    if completed.returncode != 0:
        flatten["stderr_tail"] = completed.stderr[-2000:]
    atomic_json(mesh_output / "CHAIN.json", record)
    return record
