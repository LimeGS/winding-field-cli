"""Which code may continue an existing run.

A run records the compatibility epoch of every workflow stage when it is created. A
later version of the CLI may continue the run when, for every stage the run has not
completed yet, its epoch is unchanged and the numerical core has the same major
version. Completed stages stay protected by the hashes of their outputs, which every
resume verifies.

Bump a stage's epoch in STAGE_EPOCHS whenever a change alters what that stage
produces (its numbers, formats or identities), and release such a change as a new
major version of the CLI. Changes that never alter outputs (parallelism, caches,
scheduling, the ``scroll`` tools, messages, documentation) leave every epoch alone.
Runs created before this rule have no record and keep the exact-implementation check.
"""

from importlib.metadata import version

SCHEMA = "winding-run-compat-v1"

STAGE_EPOCHS = {
    "prepare": 1,
    "fibres": 1,
    "field": 1,
    "mesh": 1,
    "flatten": 1,
    "surfaces": 1,
    "original": 1,
    "tifxyz": 1,
}


def core_major():
    return int(version("winding-field").split(".")[0])


def record():
    """The compatibility record a new run stores in RUN.json."""
    return dict(schema=SCHEMA, stages=dict(STAGE_EPOCHS), core_major=core_major())


def incompatibilities(saved, pending):
    """Reasons the current code cannot continue ``pending`` stages of a run, or []."""
    if not isinstance(saved, dict) or saved.get("schema") != SCHEMA:
        return ["unknown run compatibility record"]
    reasons = []
    if saved.get("core_major") != core_major():
        reasons.append(f"numerical core major version {saved.get('core_major')} -> {core_major()}")
    stages = saved.get("stages", {})
    for stage in pending:
        then, now = stages.get(stage), STAGE_EPOCHS.get(stage)
        if then is None or now is None or then != now:
            reasons.append(f"stage {stage} epoch {then} -> {now}")
    return reasons
