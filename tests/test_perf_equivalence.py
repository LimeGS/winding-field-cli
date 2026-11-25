"""Performance settings change speed and memory, never results.

Each knob has its own equivalence test; this one turns them all on at once, because the
interactions (cached observations read by pool workers, cached fields across sweeps) are
where a combined regression would hide. REVIEW.md R6: a new knob is added here.
"""

import json

import numpy as np

from test_field import prepared_run

KNOBS = dict(
    prepared_cache_bytes=64 << 20,
    constraint_cache_bytes=1 << 20,
    observation_cache_bytes=1 << 20,
    field_cache_bytes=64 << 20,
    field_cpu_workers=2,
    field_cpu_buffer_bytes=16 << 20,
    threads=2,
    # Whole field in RAM with one background checkpoint per sweep (off: "disk").
    field_memory="field",
)
OFF = dict({k: 0 for k in KNOBS}, threads=1, field_memory="disk")


def _solve(root, config):
    from winding_field import field

    run = prepared_run(root, shape=(6, 6, 32), variable=True)
    receipt = field.solve(run, config)
    checkpoint = json.loads((run / "field/CHECKPOINT.json").read_bytes())
    progress = checkpoint["progress"]
    solver = {k: v for k, v in progress["solver"].items() if k not in ("elapsed_s", "cache_stats")}
    with field.open_field(run) as source:
        values = source.read([0] * 3, [6, 6, 32])
    chunks = [r["sha256"] for _, r in sorted(checkpoint["chunks"].items())]
    return chunks, solver, progress["completed_sweeps"], values, receipt


def test_all_performance_knobs_together_preserve_field_bytes(tmp_path, monkeypatch):
    from winding_field import field_cpu

    # Admission is resource policy, not numerics: admit the pool on any runner.
    monkeypatch.setattr(field_cpu, "headroom_bytes", lambda: 8 << 30)
    monkeypatch.setattr(field_cpu, "cpu_limit", lambda: 8)
    base = dict(sweeps=3, field_chunk=8)
    off = _solve(tmp_path / "off", dict(base, **OFF))
    on = _solve(tmp_path / "on", dict(base, **KNOBS))
    assert on[0] == off[0]
    assert on[1] == off[1] and on[2] == off[2]
    for a, b in zip(on[3], off[3]):
        np.testing.assert_array_equal(a, b)
    # The knobs were really active, so the equality above is not vacuous.
    receipt = on[4]
    assert receipt["field_cpu_stats"]["workers"] == 2
    assert receipt["observation_cache_stats"]["hits"] > 0
    assert receipt["solver"]["cache_stats"]["hits"] > 0
    assert receipt["solver"]["checkpoint_policy"]["policy"] == "sweep"
    assert off[4]["solver"]["checkpoint_policy"]["policy"] == "block"


def test_every_validated_cache_budget_is_covered():
    import inspect
    import re

    from winding_field import config

    declared = set(re.findall(r'"(\w+_cache_bytes)"', inspect.getsource(config)))
    assert declared and declared <= set(KNOBS)
