"""Typed export request assembly; existing mode validators own scientific rules."""

import argparse
from copy import deepcopy
from pathlib import Path

from .cli_settings import field_overrides, merge_explicit, read_settings


def auto_float(value):
    return None if value == "auto" else float(value)


_FLOATS = (
    "pitch_um",
    "memory_gib",
    "process_tree_gib",
    "disk_reserve_gib",
    "cache_gib",
    "timeout_seconds",
    "reference_z",
)
_INTS = (
    "tile_size",
    "cache_mib",
    "ct_workers",
    "raster_workers",
    "max_patch_faces",
    "max_charts",
    "max_depth",
    "workers",
)
_PATHS = ("cache_dir", "app_run", "reuse_export")


def output_frame_arguments(parser):
    group = parser.add_argument_group("output coordinate frame", argument_default=argparse.SUPPRESS)
    group.add_argument("--output-frame-id")
    group.add_argument(
        "--output-origin-xyz-um", type=float, nargs=3, help="XYZ origin in micrometres"
    )
    group.add_argument(
        "--output-voxel-um", type=float, help="output coordinate unit in micrometres"
    )


def output_frame_overrides(args):
    return {
        key: vars(args)[flag]
        for flag, key in (
            ("output_frame_id", "id"),
            ("output_origin_xyz_um", "origin_xyz_um"),
            ("output_voxel_um", "voxel_um"),
        )
        if flag in vars(args)
    }


def flattening_arguments(parser):
    output_frame_arguments(parser)
    group = parser.add_argument_group(
        "flattening settings (mode-specific)", argument_default=argparse.SUPPRESS
    )
    group.add_argument(
        "--pitch", type=auto_float, help="original WORLD pitch; auto retains automatic sampling"
    )
    for name in _FLOATS:
        group.add_argument(
            "--" + name.replace("_", "-"),
            type=float,
            help="micrometres" if name == "pitch_um" else "GiB" if name.endswith("_gib") else None,
        )
    for name in _INTS:
        group.add_argument("--" + name.replace("_", "-"), type=int)
    for name in _PATHS:
        group.add_argument("--" + name.replace("_", "-"), type=Path)
    group.add_argument("--native-recipe")
    group.add_argument("--native-cpus", type=int, nargs="+")
    group.add_argument("--auto-native-cpus", dest="native_cpus", action="store_const", const=None)
    group.add_argument("--reuse-identity-sha256")
    group.add_argument("--no-reuse-export", dest="reuse_export", action="store_const", const=None)
    group.add_argument(
        "--no-reuse-identity", dest="reuse_identity_sha256", action="store_const", const=None
    )
    group.add_argument(
        "--centerline-point",
        type=float,
        nargs=3,
        action="append",
        help="WORLD XYZ; replaces file centerline",
    )


def flattening_request(args, default=None):
    if hasattr(args, "resolved_flattening"):
        return deepcopy(args.resolved_flattening)
    request = deepcopy(default or {})
    path = getattr(args, "flattening_config", None)
    if path:
        path = Path(path).expanduser().resolve()
        document = read_settings(path)
        for name in _PATHS:
            if document.get(name) is not None:
                document[name] = str((path.parent / Path(document[name]).expanduser()).resolve())
        request = merge_explicit(request, document)
    values = vars(args)
    overrides = {
        name: values[name]
        for name in (
            *_FLOATS,
            *_INTS,
            *_PATHS,
            "pitch",
            "native_recipe",
            "native_cpus",
            "reuse_identity_sha256",
        )
        if name in values
    }
    for name in _PATHS:
        if isinstance(overrides.get(name), Path):
            overrides[name] = str(overrides[name].expanduser().resolve())
    frame = output_frame_overrides(args)
    if frame:
        overrides["output_frame"] = frame
    if "centerline_point" in values:
        overrides["centerline"] = args.centerline_point
    return merge_explicit(request, overrides)


def field_request(args):
    """Normalize only known local paths; never treat arbitrary provenance as paths."""
    if hasattr(args, "resolved_field"):
        return deepcopy(args.resolved_field)
    request = {}
    path = getattr(args, "field_config", None)
    if path:
        from .tifxyz_field import read_metadata

        path = Path(path).expanduser().resolve()
        request, record = read_metadata(path)
        args.field_config_record = record
        descriptor = request.get("field")
        if isinstance(descriptor, dict):
            if "path" in descriptor:
                descriptor["path"] = str((path.parent / descriptor["path"]).resolve())
            if isinstance(descriptor.get("mask"), dict) and "path" in descriptor["mask"]:
                descriptor["mask"]["path"] = str(
                    (path.parent / descriptor["mask"]["path"]).resolve()
                )
    request = merge_explicit(request, field_overrides(args))
    if request and not all(isinstance(request.get(key), dict) for key in ("field", "geometry")):
        raise ValueError("descriptor requires field and geometry objects")
    return request or None


def range_request(args):
    request = read_settings(args.config) if args.config else {}
    if args.config:
        descriptor = field_request(argparse.Namespace(field_config=args.config))
        request.update({k: descriptor[k] for k in ("field", "geometry") if k in descriptor})
    request = merge_explicit(request, field_overrides(args))
    for flag, key in (
        ("umbilicus_point", "umbilicus"),
        ("handedness", "h"),
        ("w_ref", "W_ref"),
        ("theta_ref", "theta_ref"),
        ("samples_per_w", "samples_per_w"),
        ("max_edge_world", "max_edge_world"),
    ):
        if flag in vars(args):
            request[key] = vars(args)[flag]
    return request


_TIFXYZ_FLOATS = ("pitch_um", "tolerance_um", "memory_gib", "scratch_gib", "reserve_disk_gib")
_TIFXYZ_OPTIONS = (
    *_TIFXYZ_FLOATS,
    "roi_lo_zyx",
    "roi_hi_zyx",
    "whole_field",
    "max_pixels",
    "audit_backend",
    "coverage_policy",
)
_FRAME_OPTIONS = (
    "output_frame",
    "output_frame_id",
    "output_origin_xyz_um",
    "output_voxel_um",
    "output_shape_zyx",
    "output_source_identity_sha256",
)


def tifxyz_arguments(parser, *, prefix="", include_levels=True):
    output_frame_arguments(parser)
    group = parser.add_argument_group("direct TIFXYZ request", argument_default=argparse.SUPPRESS)
    if include_levels:
        group.add_argument("--levels", type=float, nargs="+", help="raw field isovalues")
    elif prefix:
        group.add_argument(
            "--" + prefix + "levels",
            type=float,
            nargs="+",
            help="direct export raw isovalues; must agree with producer --levels when supplied",
        )
    for name in _TIFXYZ_FLOATS:
        group.add_argument(
            "--" + prefix + name.replace("_", "-"),
            type=auto_float if name in ("pitch_um", "tolerance_um") else float,
            help="micrometres; auto uses existing defaults"
            if name in ("pitch_um", "tolerance_um")
            else "GiB",
        )
    for name in ("roi_lo_zyx", "roi_hi_zyx"):
        group.add_argument(
            "--" + prefix + name.replace("_", "-"),
            type=int,
            nargs=3,
            help="half-open field node bounds, ZYX",
        )
    group.add_argument("--" + prefix + "whole-field", action="store_true")
    group.add_argument("--" + prefix + "max-pixels", type=int)
    group.add_argument("--" + prefix + "audit-backend", choices=("numpy", "cpu", "cuda"))
    group.add_argument("--" + prefix + "coverage-policy", choices=("strict", "allow_partial"))
    group.add_argument("--output-frame", choices=("selected-source",))
    group.add_argument("--output-shape-zyx", type=int, nargs=3)
    group.add_argument("--output-source-identity-sha256")


def tifxyz_request(args, *, path=None, prefix=""):
    from .tifxyz_contract import validate_request

    request = read_settings(path) if path else {}
    if path is None and not prefix and getattr(args, "config", None):
        request = read_settings(args.config)
    values = vars(args)
    overrides = {}
    nested_levels = prefix.replace("-", "_") + "levels"
    if prefix and nested_levels in values:
        overrides["levels"] = values[nested_levels]
    elif values.get("levels") is not None and (not prefix or "levels" not in request):
        overrides["levels"] = values["levels"]
    for name in _TIFXYZ_OPTIONS:
        flag = prefix.replace("-", "_") + name
        if flag not in values:
            continue
        value = values[flag]
        if name in ("roi_lo_zyx", "roi_hi_zyx"):
            overrides.setdefault("roi", {})[name[4:]] = value
        elif name != "whole_field":
            overrides[name] = value
    if values.get(prefix.replace("-", "_") + "whole_field"):
        if "roi" in overrides:
            raise ValueError("whole-field conflicts with explicit ROI")
        overrides["roi"] = None
    frame = output_frame_overrides(args)
    for flag, key in (
        ("output_shape_zyx", "shape_zyx"),
        ("output_source_identity_sha256", "source_identity_sha256"),
    ):
        if flag in values:
            frame[key] = values[flag]
    if "output_frame" in values:
        if frame:
            raise ValueError("selected-source conflicts with explicit output frame")
        overrides["output_frame"] = values["output_frame"]
    elif frame:
        overrides["output_frame"] = frame
    if not request and not overrides:
        return None
    return validate_request(merge_explicit(request, overrides))


def pop_tifxyz_options(config, *, prefix=""):
    """Do not leak nested output options into the producer's saved configuration."""
    for name in (*_FRAME_OPTIONS, *(prefix.replace("-", "_") + n for n in _TIFXYZ_OPTIONS)):
        config.pop(name, None)
    if prefix:
        config.pop(prefix.replace("-", "_") + "levels", None)
