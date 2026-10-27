"""Small offline complete-cycle probe; reports timings, never a universal speedup."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples"))

from block_field import case_options, create_case, read_result, sources

from block_solver import solve_blocks
from field_solver import PreparedField, solve_field
from field_store import atomic_json


def probe(output, repetitions=2):
    if type(repetitions) is not int or repetitions < 1:
        raise ValueError("positive repetitions required")
    output = Path(output)
    output.mkdir(exist_ok=False)
    started = time.perf_counter()
    source_identity = sources()
    create_case(output / "inputs")
    preparation = time.perf_counter() - started
    runs, baseline = [], None
    for repetition in range(repetitions):
        variants = ["serial-uncached", "bounded-parallel"]
        if repetition % 2:
            variants.reverse()
        for variant in variants:
            begin = time.perf_counter()
            cached = variant == "bounded-parallel"
            options = case_options(
                output / "inputs",
                cache_bytes=(1 << 20) if cached else 0,
                read_workers=2 if cached else 1,
                pair_cache_bytes=(1 << 20) if cached else 0,
            )
            folder = output / (str(repetition) + "-" + variant)
            receipt = solve_blocks(folder, **options)
            result = read_result(folder)
            elapsed = time.perf_counter() - begin
            if baseline is None:
                baseline = result
            np.testing.assert_array_equal(result, baseline)
            runs.append(
                dict(
                    variant=variant,
                    repetition=repetition,
                    complete_cycle_seconds=elapsed,
                    cache_bytes=options["cache_bytes"],
                    read_workers=options["read_workers"],
                    field_cache=receipt["cache_stats"],
                    constraint_cache=options["constraints"].stats,
                )
            )

    # A changing RHS is a genuine reuse case; single-use block solves skip snapshots.
    options = case_options(output / "inputs")
    shape = options["shape"]
    target, weights, _, _ = options["load_observations"]([0] * 3, shape)
    target, weights = torch.as_tensor(target, dtype=torch.float64), torch.as_tensor(weights, dtype=torch.float64)
    rows = options["constraints"].restrict([0] * 3, shape, lambda nodes: np.zeros(nodes.shape))
    workspace = PreparedField()
    prepared_runs = []
    for index, offset in enumerate((0.0, 0.125)):
        constraints = rows.with_targets(rows.targets + offset)
        values, timings = {}, {}
        for prepared in (False, True) if index == 0 else (True, False):
            begin = time.perf_counter()
            field, info = solve_field(
                target,
                weights,
                options["spacing_xyz"],
                constraints=constraints,
                dtype="float64",
                rtol=1e-10,
                prepared=workspace if prepared else None,
            )
            assert info["converged"], info
            timings[str(prepared)] = time.perf_counter() - begin
            values[prepared] = field
        np.testing.assert_array_equal(values[False], values[True])
        prepared_runs.append(dict(target_offset=offset, seconds=timings))
    if sources() != source_identity:
        raise ValueError("benchmark library sources changed during execution")
    report = dict(
        schema="synthetic-block-performance-v1",
        parity=True,
        prepared_parity=True,
        input_preparation_seconds=preparation,
        complete_probe_seconds=time.perf_counter() - started,
        runs=runs,
        prepared_runs=prepared_runs,
        code_sha256=source_identity,
        scope="CPU synthetic smoke measurement; no real-data/GPU speedup claim; cache bytes are retention only",
    )
    atomic_json(output / "REPORT.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=2)
    args = parser.parse_args()
    print(json.dumps(probe(args.output, args.repetitions), allow_nan=False))
