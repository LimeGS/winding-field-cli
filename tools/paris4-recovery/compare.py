"""Compare the frozen Paris4 failure census against every receipt we have produced.

Read-only: the baseline snapshot and the new receipts are inputs; the report is the only
output. Usage: compare.py --baseline CASES.json --evidence <root> --output report.json
"""

import argparse
import json
from pathlib import Path


def level_of(w):
    return f"level-{w + 137:04d}"


def load_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def receipts(evidence):
    """Collect every per-case status we can defend from a receipt on disk."""
    found = {}

    def record(w, stage, status, cause=None, source=None):
        entry = found.setdefault(int(w), {})
        entry[stage] = dict(status=status, cause=cause, source=source)

    root = Path(evidence)
    for path in sorted(root.glob("init-w*/INITIALIZATION.json")):
        name = path.parent.name
        w = int(name.removeprefix("init-w").replace("m", "-").replace("p", ""))
        data = load_json(path) or {}
        failed = [item for item in data.get("results", []) if item.get("status") != "INITIALIZED"]
        record(
            w,
            "initialization",
            "PASSED" if data.get("status") == "INITIALIZED" else "FAILED",
            None if not failed else failed[0].get("error"),
            str(path),
        )
    audits = load_json(root / "TASK2-AUDITS.json")
    for entry in audits or []:
        if entry.get("returncode") is not None and entry.get("w") is not None:
            record(
                entry["w"],
                "audit",
                "COMPLETED" if entry["returncode"] == 0 else "FAILED",
                None,
                "TASK2-AUDITS.json",
            )
    experiment = load_json(root / "TASK3-W40-CONTINUATION.json")
    if experiment:
        record(
            -40,
            "mvc-solve",
            "CONVERGED" if max(experiment["true_residuals"]) <= 1e-10 else "FAILED",
            None,
            "TASK3-W40-CONTINUATION.json",
        )
    facts = load_json(root / "wm20-accuracy/ACCURACY.json")
    if facts:
        record(
            -20,
            "geometry",
            "PINCHED" if facts["mvc_direct"]["nonpositive"] else "CLEAN",
            f"{facts['mvc_direct']['nonpositive']} nonpositive faces with a direct solve",
            "wm20-accuracy/ACCURACY.json",
        )
    comparison = load_json(root / "native-triple-w60/COMPARISON.json")
    if comparison:
        record(
            -60,
            "native",
            "MATCHES_BASELINE" if comparison["pairs"] else "UNKNOWN",
            None,
            "native-triple-w60/COMPARISON.json",
        )
    extraction = load_json(root / "extraction-1.5.9/EXTRACTION.json") or load_json(
        root / "EXTRACTION.json"
    )
    for entry in (extraction or {}).get("bundles", []) if isinstance(extraction, dict) else []:
        w = entry.get("w")
        if w is None:
            continue
        record(w, "pipeline", entry.get("status", "UNKNOWN"), entry.get("cause"), "EXTRACTION.json")
    return found


CLASSIFICATION = {
    "ValueError: GMRES did not converge with a verified true residual": "initialization",
    "RuntimeError: overlap narrow-test budget exceeded": "audit",
    "RuntimeError: overlap candidate materialization bound exceeded": "audit",
    "ValueError: patch geometry admission exceeds 6 GiB (1051165 vertices, 2102206 faces)": "admission",
    "ValueError: bounded float64 XYZ and integer triangles required": "admission",
    "ValueError: inverted UV triangle": "geometry",
    "ValueError: collapsed uniform initialization": "geometry",
    "RuntimeError: native negative diagonal": "native",
    "ValueError: checkpoint contains inverted/degenerate UV faces": "checkpoint",
    "ValueError: warm-start energy differs": "warm-start",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text())
    groups = baseline["failed_groups"]
    observed = receipts(args.evidence)
    total = sum(len(cases) for cases in groups.values())
    report = dict(
        snapshot=baseline.get("snapshot_utc"),
        baseline_failed=total,
        families=[],
        counts=dict(baseline=total, cause_removed=0, unresolved=0, pending=0),
    )
    for cause, cases in groups.items():
        family = CLASSIFICATION.get(cause, "unknown")
        rows = []
        for w in cases:
            stages = observed.get(int(w), {})
            if family in ("initialization", "audit", "admission"):
                seen = [stages[k] for k in ("initialization", "audit", "pipeline") if k in stages]
                if any(
                    item["status"] in ("PASSED", "COMPLETED", "UV_READY_DIAGNOSTIC")
                    for item in seen
                ):
                    state = "cause_removed_verified"
                elif seen:
                    state = "still_failing"
                else:
                    state = "cause_removed_unverified" if family != "admission" else "not_rerun"
            elif family == "native":
                state = "unresolved_native"
            elif family == "geometry":
                failing = stages.get("initialization", {}).get("status") == "FAILED"
                state = (
                    "unresolved_geometry" if (stages.get("geometry") or failing) else "not_rerun"
                )
            elif family == "checkpoint":
                state = "unresolved_checkpoint"
            elif family == "warm-start":
                state = "unresolved_warm_start"
            else:
                state = "unknown"
            rows.append(dict(w=w, level=level_of(w), state=state, observed=stages))
            key = {
                "cause_removed_verified": "cause_removed",
                "cause_removed_unverified": "cause_removed",
                "not_rerun": "pending",
            }.get(state, "unresolved")
            report["counts"][key] += 1
        report["families"].append(dict(cause=cause, family=family, cases=cases, rows=rows))
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["counts"], indent=1))
    for family in report["families"]:
        states = {}
        for row in family["rows"]:
            states[row["state"]] = states.get(row["state"], 0) + 1
        print(f"{family['family']:15s} {len(family['cases']):3d} {states}")


if __name__ == "__main__":
    main()
