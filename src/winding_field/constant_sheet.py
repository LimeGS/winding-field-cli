"""Installed stages for the source-preserving constant-W sheet recipe."""

import argparse
import json
from pathlib import Path

from . import flatboi_checkpoint
from .patch_pipeline import CHECKPOINT_RECIPES
from .sheet_atlas import _json
from .sheet_finish import render, validate
from .sheet_prepare import load_prepared, prepare


def main(argv=None):
    parser = argparse.ArgumentParser(prog="winding-field constant-sheet", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser(
        "prepare", help="select constant W, cut exact mesh and initialize positive UV"
    )
    prep.add_argument("--mesh", type=Path, required=True)
    prep.add_argument("--w", type=float, required=True)
    prep.add_argument(
        "--centerline", type=Path, help="optional JSON array of XYZ points with increasing Z"
    )
    prep.add_argument(
        "--centerline-point", type=float, nargs=3, action="append", default=argparse.SUPPRESS
    )
    prep.add_argument("--output", type=Path, required=True)
    for name in ("flatten", "import-native"):
        cmd = commands.add_parser(
            name,
            help="checkpointed Flatboi"
            if name == "flatten"
            else "copy a verified historical checkpoint into a fresh run",
        )
        cmd.add_argument("--prepared", type=Path, required=True)
        cmd.add_argument("--output", type=Path, required=True)
        cmd.add_argument("--app-run", type=Path, required=True)
        cmd.add_argument(
            "--native-recipe", choices=CHECKPOINT_RECIPES, default="installed-2f89bfc9"
        )
        if name == "flatten":
            cmd.add_argument("--iterations", type=int, default=200)
            cmd.add_argument("--batch-iterations", type=int, default=10)
            cmd.add_argument("--hours", type=float, default=12.0)
        else:
            cmd.add_argument("--native", type=Path, required=True)
            cmd.add_argument("--runner-sha256", required=True)
    valid = commands.add_parser(
        "validate", help="rebind native XYZ/source IDs; calibrate UV and audit overlaps"
    )
    valid.add_argument("--prepared", type=Path, required=True)
    valid.add_argument("--native", type=Path, required=True)
    valid.add_argument("--output", type=Path, required=True)
    valid.add_argument("--reference-z", type=float, required=True)
    valid.add_argument(
        "--centerline",
        type=Path,
        help="required for legacy prepared artifacts; must match new preparation",
    )
    valid.add_argument(
        "--centerline-point", type=float, nargs=3, action="append", default=argparse.SUPPRESS
    )
    valid.add_argument(
        "--mesh", type=Path, help="relocated original mesh; original manifest hash must match"
    )
    ct = commands.add_parser("render", help="render19.2um CT and publish a verified atlas sheet")
    ct.add_argument("--validated", type=Path, required=True)
    ct.add_argument("--output", type=Path, required=True)
    ct.add_argument("--config", type=Path, help="optional CT source, frame and cache settings JSON")
    from .export_settings import flattening_arguments

    flattening_arguments(ct)
    group = ct.add_argument_group("CT and prior data", argument_default=argparse.SUPPRESS)
    group.add_argument("--ct-source")
    group.add_argument("--ct-level", type=int)
    group.add_argument("--ct-voxel-um", type=float)
    for name in ("prior-identity", "prior-ledger"):
        group.add_argument("--" + name, type=Path)
    group.add_argument("--prior-ledger-sha256")
    status = commands.add_parser(
        "status", help="read-only native progress (use watch for a live monitor)"
    )
    status.add_argument("--native", type=Path, required=True)
    status.add_argument("--json", action="store_true")
    from .cli_settings import parse_options

    args = parse_options(parser, argv, paths=("ct_source",))
    try:
        centerline = getattr(args, "centerline_point", None)
        if centerline is None and getattr(args, "centerline", None):
            centerline = _json(args.centerline)
        if args.command == "prepare":
            result = prepare(args.mesh, args.output, args.w, centerline)
        elif args.command in ("flatten", "import-native"):
            load_prepared(args.prepared)
            args.source_map = args.prepared / "initial-map.npz"
            args.input = args.prepared / "inputmesh.obj"
            if args.command == "flatten":
                result = flatboi_checkpoint.run(args)
            else:
                result = flatboi_checkpoint.import_checkpoint(
                    args.native,
                    args.output,
                    args.source_map,
                    args.input,
                    args.app_run,
                    args.runner_sha256,
                    args.native_recipe,
                )
        elif args.command == "validate":
            result = validate(
                args.prepared,
                args.native,
                args.output,
                centerline,
                args.reference_z,
                mesh_source=args.mesh,
            )
        elif args.command == "render":
            from .tool_settings import constant_render_request, check_sources

            result = render(args.validated, args.output, constant_render_request(args))
            check_sources(args)
        else:
            result = flatboi_checkpoint.observe(args.native)
            if not args.json:
                print(
                    f"FLATBOI — {result['status']}\nCalculadas: {result['calculated']} / {result['target']}\n"
                    f"Guardadas para retomar: {result['saved']}\nEnergía: {result['energy']}\nConvergencia: {result['converged']}"
                )
                return 0
        print(json.dumps(result, allow_nan=False), flush=True)
        return 0
    except (ValueError, KeyError, OSError, AssertionError, RuntimeError) as error:
        parser.exit(75 if isinstance(error, InterruptedError) else 2, f"constant-sheet: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
