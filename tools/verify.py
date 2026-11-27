#!/usr/bin/env python3
"""Verify a change end to end, against the previous version as the oracle.

    python tools/verify.py                      # all steps, baseline origin/main
    python tools/verify.py --baseline v1.5.8    # another oracle
    python tools/verify.py --steps e2e-demo,golden

Steps (see .claude/skills/verify/SKILL.md):

- ``flags``: every public command/option/choice of the baseline still exists.
- ``golden``: flattening and field outputs are byte-identical to the baseline.
- ``e2e-demo``: ``winding-field demo`` completes; ``resume`` leaves it byte-identical.
- ``e2e-native``: native run stopped after ``field``, resumed to COMPLETE (SLIM through
  Flatboi when the runtime is available, otherwise the default LSCM path).
- ``cross-version``: a run made by the baseline either resumes or is refused cleanly
  (never modified, never a traceback), and its imported stages are reusable via
  ``--reuse-from``.
- ``review``: lists the REVIEW.md checklist items whose triggers match the diff.

Intended differences go in ``tools/verify-approved.json`` with a reason. The receipt is
written to ``.verify/VERIFY.json``; exit status is 1 when any step FAILs.
"""

import argparse
import fnmatch
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

DEFAULT_REPO = Path(__file__).resolve().parents[1]
REPO = DEFAULT_REPO  # CI runs the base branch's verifier on the checkout via --repo
STEPS = ("review", "flags", "golden", "e2e-demo", "e2e-native", "cross-version")


# --------------------------------------------------------------------------------------
# Probes run in a subprocess with ``<tree>/src`` and ``<tree>/tests`` first on sys.path,
# so the same probe code measures the baseline and the working tree.
# --------------------------------------------------------------------------------------


def _module(*names):
    from importlib import import_module

    for name in names:
        try:
            return import_module(name)
        except ImportError:
            continue
    raise ImportError(names)


def _digest(*arrays):
    import numpy as np

    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(np.asarray(a))
        h.update(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes())
    return h.hexdigest()


def probe_flags(work):
    """Public CLI inventory: commands, options and choices."""
    import argparse as ap

    class Captured(Exception):
        pass

    def capture(self, *args, **kwargs):
        raise Captured(self)

    ap.ArgumentParser.parse_args = capture
    try:  # typed --cli-config pre-parsing calls parse_known_args on helper parsers first
        settings = _module("winding_field.cli_settings")
        settings.parse_options = lambda parser, *args, **kwargs: capture(parser)
    except ImportError:
        pass
    cli = _module("winding_field.cli")
    entries = {"": [], "prepared": ["prepared"], **{n: [n] for n in cli.TOOLS}}
    # ``prepared mesh|render|surfaces`` dispatch to the surface workflow's own parser.
    entries.update({f"prepared {n}": ["prepared", n] for n in ("mesh", "render", "surfaces")})
    found = set()

    def walk(parser, path):
        for action in parser._actions:
            if isinstance(action, ap._SubParsersAction):
                for name, child in action.choices.items():
                    found.add(f"{path} {name}".strip())
                    walk(child, f"{path} {name}".strip())
            elif not isinstance(action, (ap._HelpAction, ap._VersionAction)):
                names = action.option_strings or [f"<{action.dest}>"]
                for name in names:
                    found.add(f"{path} {name}".strip())
                for choice in action.choices or ():
                    found.add(f"{path} {names[-1]}={choice}".strip())

    for path, argv in entries.items():
        try:
            cli.main(argv)
        except Captured as captured:
            walk(captured.args[0], path)
    return sorted(found)


def probe_golden(work):
    """Hashes of every flattening path and of field checkpoints."""
    import numpy as np

    results = {}
    from test_meshes import curved_strip

    v, f = curved_strip()
    legacy = _module("winding_field.flattening.legacy_charts", "winding_field.flatten")
    results["legacy.parameterize"] = _digest(legacy.parameterize(v, f)[0])
    charts = _module("winding_field.flattening.native_charts", "winding_field.native_geometry")
    results["native._lscm"] = _digest(charts._lscm(v, f)[0])
    pieces = charts._pieces(v, f, np.arange(len(f)), np.arange(len(v)), minimum=8)
    results["native._pieces"] = _digest(*[a for p in pieces for a in p[:5]])
    original = _module("winding_field.flattening.original", "winding_field.original_recipe")
    made = original.make_map(v, f)
    results["original.make_map"] = _digest(
        *[made[k] for k in sorted(made) if isinstance(made[k], np.ndarray)]
    )
    from test_flattening_matrix import geometry, matrix, request

    m, (gv, gf) = matrix(), geometry()
    for scope in ("patches", "sheet"):
        root = Path(tempfile.mkdtemp(dir=work))
        req = request(root)
        if scope == "patches":
            req["max_patch_faces"] = 32
        r = m.build_map(gv, gf, root / "map", m.settings(req, "lscm", scope))
        keys = ("uv", "faces", "parent", "world", "source_face_rows")
        results[f"matrix.lscm.{scope}"] = _digest(*[r[k] for k in keys if k in r])
    sheet = _module("winding_field.flattening.native_sheet", "winding_field.native_global")
    cut, _ = sheet.cut_component(v, f)
    uv, _ = sheet.mean_value_map(cut, lambda: None, diagnostics=Path(work) / "mv")
    results["native_sheet.mean_value_map"] = _digest(uv)

    from test_native_runtime import field_case

    from winding_field.native_pipeline import work as native

    methods = ["default"]
    try:
        _module("winding_field.flattening.flatboi").resolve_app_run(offline=True)
        methods.append("slim")
    except Exception:  # no runtime: the receipt shows the key as missing on both sides
        pass
    for method in methods:
        run, config = field_case(Path(tempfile.mkdtemp(dir=work)), levels=(0.5,))
        config["flattening_method"] = method
        assert native(run, config)["status"] == "COMPLETE"
        arrays = []
        for path in sorted((run / "surfaces/level-0000/charts").glob("*.npz")):
            with np.load(path) as z:
                arrays += [z[k] for k in sorted(z.files)]
        results[f"native.pipeline.{method}.charts"] = _digest(*arrays)

    from test_field import prepared_run

    from winding_field import field

    run = prepared_run(Path(tempfile.mkdtemp(dir=work)) / "run")
    field.solve(run, dict(sweeps=3, field_chunk=8))
    checkpoint = json.loads((run / "field/CHECKPOINT.json").read_bytes())
    results["field.solve.chunks"] = hashlib.sha256(
        "".join(r["sha256"] for _, r in sorted(checkpoint["chunks"].items())).encode()
    ).hexdigest()
    return results


def probe_donor(work):
    """Imported Lasagna + AFV stages, made by this tree, for cross-version reuse."""
    import numpy as np
    from test_lasagna_input import source_case
    from test_splines import upstream_curves

    from winding_field import workflow

    work = Path(work)
    volume, _, config = source_case(work, shape=(9, 10, 11), whole=True)
    volume.close()
    afv = work / "external.afv"
    upstream_curves(afv, (9, 10, 11), [(1, np.array([[1.0, 2.0, 3.0], [5.0, 2.0, 3.0]]))])
    config.update(
        output=str(work / "donor"),
        fibres=str(afv),
        fibres_geometry=config["lasagna_geometry"],
        maxiter=2,
        sweeps=1,
        field_chunk=16,
        threads=1,
        previews=False,
    )
    workflow.run(config, until="fibres")
    return config


def probe_reuse(work):
    """Reuse the donor's imported stages with this tree and solve the field."""
    from winding_field import workflow
    from winding_field.util import file_sha

    donor = json.loads(Path(work, "donor.json").read_text())
    root = Path(donor["output"])
    config = dict(
        donor,
        output=str(Path(work) / "reuse"),
        reuse_from=str(root),
        reuse_from_sha256=file_sha(root / "RUN.json"),
        nonconvergence_policy="continue",
    )
    result = workflow.run(config, until="field")
    return {name: stage.get("mode", "run") for name, stage in result["stages"].items()}


PROBES = dict(flags=probe_flags, golden=probe_golden, donor=probe_donor, reuse=probe_reuse)


def run_probe(name, tree, work):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(tree / "src"), str(tree / "tests")]))
    command = [sys.executable, str(Path(__file__).resolve()), "--probe", name, "--work", str(work)]
    done = subprocess.run(command, cwd=work, env=env, capture_output=True, text=True)
    if done.returncode:
        return None, (done.stderr or done.stdout)[-3000:]
    return json.loads(done.stdout.strip().splitlines()[-1]), None


# --------------------------------------------------------------------------------------
# Steps
# --------------------------------------------------------------------------------------


class Step:
    def __init__(self, name):
        self.name, self.status, self.evidence, self.start = name, "PASS", {}, time.monotonic()

    def fail(self, reason, **evidence):
        self.status = "FAIL"
        self.evidence.setdefault("failures", []).append(reason)
        self.evidence.update(evidence)

    def warn(self, reason):
        if self.status == "PASS":
            self.status = "WARN"
        self.evidence.setdefault("warnings", []).append(reason)

    def record(self):
        return dict(
            status=self.status, seconds=round(time.monotonic() - self.start, 1), **self.evidence
        )


def git(*args):
    return subprocess.run(
        ["git", *args], cwd=REPO, check=True, capture_output=True, text=True
    ).stdout.strip()


def snapshot(root):
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(Path(root).rglob("*"))
        if p.is_file() and not p.is_symlink()
    }


def cli(tree, *args, cwd):
    env = dict(os.environ, PYTHONPATH=str(tree / "src"))
    return subprocess.run(
        [sys.executable, "-m", "winding_field", *map(str, args)],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
    )


def flatboi_available():
    sys.path.insert(0, str(REPO / "src"))
    try:
        from winding_field.flattening import flatboi

        flatboi.resolve_app_run(offline=True)
        return True
    except Exception:
        return False
    finally:
        sys.path.remove(str(REPO / "src"))


def changed_lines(diff, untracked=()):
    """``{path: changed lines}`` from a unified diff; untracked files count as all-added."""
    files, current = {}, None
    for line in diff.splitlines():
        if line.startswith("diff --git "):
            current = line.split(" b/", 1)[-1]
            files[current] = []
        elif current and line[:1] in "+-" and not line.startswith(("+++", "---")):
            files[current].append(line)
    for path in untracked:
        try:
            files[path] = ["+" + line for line in (REPO / path).read_text().splitlines()]
        except (OSError, UnicodeDecodeError):
            files[path] = []
    return files


def step_review(step, ctx):
    """An item triggers when a changed file matches its Paths and, if it has a Diff regex,
    a changed (+/-) line of such a file matches it. Context lines never trigger."""
    # The checklist that judges a change is read from the baseline, never from the change.
    source = ctx["base_tree"] / "REVIEW.md"
    if not source.is_file():
        step.warn("REVIEW.md is not on the baseline yet; using the working tree copy")
        source = REPO / "REVIEW.md"
    files = ctx["lines"]
    matched = []
    for row in review_items(source):
        paths = [p.strip(" `") for p in row["paths"].split(",") if p.strip(" `")]
        selected = [f for f in files if any(fnmatch.fnmatch(f, p) for p in paths)]
        pattern = row["pattern"].strip(" `")
        if pattern and pattern != "\u2014":
            text = "\n".join(line for f in selected for line in files[f])
            selected = selected if re.search(pattern, text, re.MULTILINE) else []
        if selected:
            matched.append(f"{row['id']}: {row['check']}")
    step.evidence.update(changed_files=len(files), checklist=matched)


def review_items(path):
    if not path.is_file():
        return []
    items = []
    for line in path.read_text().splitlines():
        cells = [
            c.strip().replace("\\|", "|") for c in re.split(r"(?<!\\)\|", line.strip().strip("|"))
        ]
        if len(cells) == 5 and re.fullmatch(r"R\d+", cells[0]):
            items.append(dict(zip(("id", "paths", "pattern", "check", "origin"), cells)))
    return items


def step_flags(step, ctx):
    base, error = run_probe("flags", ctx["base_tree"], ctx["work"])
    head, head_error = run_probe("flags", REPO, ctx["work"])
    if head_error:
        return step.fail("working tree inventory failed", error=head_error)
    if error:
        return step.warn(f"baseline inventory failed: {error[-500:]}")
    approved = ctx["approved"].get("flags", {})
    removed = sorted(set(base) - set(head) - set(approved))
    step.evidence.update(
        options=len(head), added=sorted(set(head) - set(base)), approved_removals=sorted(approved)
    )
    if removed:
        step.fail(f"{len(removed)} public flags/commands/choices removed", removed=removed)


def step_golden(step, ctx):
    base, error = run_probe("golden", ctx["base_tree"], ctx["work"])
    head, head_error = run_probe("golden", REPO, ctx["work"])
    if head_error:
        return step.fail("working tree golden probe failed", error=head_error)
    if error:
        return step.warn(f"baseline golden probe failed: {error[-500:]}")
    approved = ctx["approved"].get("golden", {})
    changed = sorted(k for k in base.keys() & head.keys() if base[k] != head[k])
    step.evidence.update(identical=sorted(k for k in base if head.get(k) == base[k]))
    for key in sorted(base.keys() ^ head.keys()):
        step.warn(f"{key} measured on one side only")
    unapproved = [k for k in changed if k not in approved]
    if changed:
        step.evidence["approved_changes"] = sorted(set(changed) - set(unapproved))
    if unapproved:
        step.fail("outputs differ from the baseline", changed=unapproved)


def _complete(run):
    return json.loads((Path(run) / "RUN.json").read_bytes()).get("status") == "COMPLETE"


DEMO = ("--device", "cpu")


def step_e2e_demo(step, ctx):
    work = ctx["work"]
    run = work / "demo"
    done = cli(REPO, "demo", "--output", run, *DEMO, cwd=work)
    if done.returncode or not _complete(run):
        return step.fail("demo did not complete", stderr=done.stderr[-2000:])
    if not (run / "index.html").is_file():
        step.fail("demo did not publish index.html")
    before = snapshot(run)
    done = cli(REPO, "resume", run, cwd=work)
    if done.returncode:
        return step.fail("resume of a complete run failed", stderr=done.stderr[-2000:])
    after = snapshot(run)
    changed = sorted(k for k in before.keys() | after.keys() if before.get(k) != after.get(k))
    step.evidence.update(files=len(before), resume_changed=changed)
    if any(not k.endswith(("RUN.json", ".sqlite")) for k in changed):
        step.fail("resume of a complete run rewrote outputs", changed=changed)


def step_e2e_native(step, ctx):
    work = ctx["work"]
    ct = work / "demo-ct.zarr"
    if not ct.exists():
        sys.path.insert(0, str(REPO / "src"))
        from winding_field.demo import synthetic

        synthetic(ct)
    method = "slim" if flatboi_available() else "default"
    if method == "default":
        message = "Flatboi runtime unavailable: native SLIM not exercised"
        if os.environ.get("WINDING_REQUIRE_FLATBOI") == "1":
            return step.fail(message)
        step.warn(message)
    run = work / f"native-{method}"
    common = [
        "--source", ct, "--output", run, "--cache", work / "cache", "--reserve-gib", "0",
        "--prediction", "classical", "--chunk", "64", "--field-chunk", "64", "--sweeps", "1",
        "--seed-stride", "4", "--chart-faces", "1024", "--surface-output", "native",
        "--flattening-method", method, *DEMO,
    ]  # fmt: skip
    done = cli(REPO, "run", *common, "--until", "field", cwd=work)
    if done.returncode:
        return step.fail("native run until field failed", stderr=done.stderr[-2000:])
    done = cli(REPO, "resume", run, cwd=work)
    if done.returncode or not _complete(run):
        return step.fail("native resume did not complete", stderr=done.stderr[-2000:])
    native = json.loads((run / "surfaces/NATIVE.json").read_bytes())
    levels = sorted(p.name for p in (run / "surfaces").glob("level-*"))
    solver = native["settings"]["parameterizer"].get("solver")
    step.evidence.update(method=method, solver=solver, levels=len(levels))
    if not levels:
        step.fail("no native level published")
    if method == "slim" and solver != "flatboi-slim":
        step.fail(f"SLIM run recorded solver {solver!r}")


def step_cross_version(step, ctx):
    work, base = ctx["work"], ctx["base_tree"]
    old = work / "baseline-demo"
    done = cli(base, "demo", "--output", old, *DEMO, cwd=work)
    if done.returncode:
        step.warn(f"baseline demo failed: {done.stderr[-500:]}")
    else:
        before = snapshot(old)
        done = cli(REPO, "resume", old, cwd=work)
        after = snapshot(old)
        text = done.stdout + done.stderr
        if "Traceback" in text:
            step.fail("resuming a baseline run raised a traceback", output=text[-2000:])
        elif done.returncode:
            outcome = "refused-cleanly"
            if before != after:
                step.fail("refused resume modified the baseline run")
            step.evidence["refusal"] = done.stderr.strip()[-300:]
        else:
            outcome = "resumed" if _complete(old) else "resume-incomplete"
            if outcome != "resumed":
                step.fail("baseline run did not stay COMPLETE after resume")
        step.evidence["resume_baseline_run"] = outcome
    donor_work = Path(tempfile.mkdtemp(prefix="donor-", dir=work))
    donor, error = run_probe("donor", base, donor_work)
    if error:
        return step.warn(f"baseline donor failed: {error[-500:]}")
    Path(work, "donor.json").write_text(json.dumps(donor))
    before = snapshot(donor["output"])
    modes, error = run_probe("reuse", REPO, work)
    if error:
        return step.fail("--reuse-from of baseline stages failed", error=error)
    step.evidence["reuse_modes"] = modes
    if modes.get("prepare") != "reused" or modes.get("fibres") != "reused" or "field" not in modes:
        step.fail("baseline stages were not reused", modes=modes)
    if snapshot(donor["output"]) != before:
        step.fail("--reuse-from modified the donor run")


RUNNERS = {
    "review": step_review,
    "flags": step_flags,
    "golden": step_golden,
    "e2e-demo": step_e2e_demo,
    "e2e-native": step_e2e_native,
    "cross-version": step_cross_version,
}


def core_receipt():
    """The pinned core revision and whether the core modules this Python imports are its
    sources (provenance/core.json source_sha256): a verify against another core proves
    nothing about the pin."""
    import importlib.util

    pin = json.loads((REPO / "provenance/core.json").read_text())
    mismatched = {}
    for name, expected in sorted(pin["source_sha256"].items()):
        spec = importlib.util.find_spec(Path(name).stem)
        origin = None if spec is None else spec.origin
        actual = None if origin is None else hashlib.sha256(Path(origin).read_bytes()).hexdigest()
        if actual != expected:
            mismatched[name] = origin or "not importable"
    return dict(pinned=pin["revision"], matches_pin=not mismatched, mismatched=mismatched)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--baseline", default="origin/main", help="oracle git ref")
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO, help="checkout to verify")
    parser.add_argument("--steps", default=",".join(STEPS), help="comma-separated subset")
    parser.add_argument("--out", type=Path, help="keep work files here (default: temporary)")
    parser.add_argument(
        "--receipt", type=Path, help="receipt path (default: <repo>/.verify/VERIFY.json)"
    )
    parser.add_argument("--probe", choices=sorted(PROBES), help=argparse.SUPPRESS)
    parser.add_argument("--work", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.probe:
        sys.path[:0] = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
        print(json.dumps(PROBES[args.probe](args.work), default=str))
        return 0
    global REPO
    REPO = args.repo.resolve()
    approved = REPO / "tools/verify-approved.json"
    steps = [s for s in args.steps.split(",") if s]
    unknown = set(steps) - set(STEPS)
    if unknown:
        parser.error(f"unknown steps: {sorted(unknown)}")
    work = (args.out or Path(tempfile.mkdtemp(prefix="verify-"))).resolve()
    work.mkdir(parents=True, exist_ok=True)
    base_sha = git("rev-parse", args.baseline)
    base_tree = work / "baseline"
    if not base_tree.exists():
        git("worktree", "add", "--detach", str(base_tree), base_sha)
    ctx = dict(
        work=work,
        base_tree=base_tree,
        approved=json.loads(approved.read_text()) if approved.is_file() else {},
        lines=changed_lines(
            git("diff", base_sha),
            git("ls-files", "--others", "--exclude-standard").splitlines(),
        ),
    )
    receipt = dict(
        head=git("rev-parse", "HEAD"),
        dirty=bool(git("status", "--porcelain")),
        baseline=dict(ref=args.baseline, sha=base_sha),
        core=core_receipt(),
        steps={},
    )
    if not receipt["core"]["matches_pin"]:
        print(
            "FAIL  core: the imported core is not the pinned revision "
            f"{receipt['core']['pinned']}: {receipt['core']['mismatched']}",
            flush=True,
        )
    try:
        for name in steps:
            step = Step(name)
            try:
                RUNNERS[name](step, ctx)
            except Exception as error:  # a crashing step is a failure, never a skip
                step.fail(f"{type(error).__name__}: {error}")
            receipt["steps"][name] = step.record()
            print(f"{step.status:4}  {name}  ({receipt['steps'][name]['seconds']} s)", flush=True)
            for line in step.evidence.get("failures", []) + step.evidence.get("warnings", []):
                print(f"      {line}")
            for line in step.evidence.get("checklist", []):
                print(f"      review {line}")
    finally:
        subprocess.run(["git", "worktree", "remove", "--force", str(base_tree)], cwd=REPO)
    receipt["status"] = (
        "FAIL"
        if not receipt["core"]["matches_pin"]
        or any(s["status"] == "FAIL" for s in receipt["steps"].values())
        else "PASS"
    )
    args.receipt = args.receipt or REPO / ".verify/VERIFY.json"
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=1, sort_keys=True))
    print(f"{receipt['status']}  receipt: {args.receipt}")
    if not args.out:
        shutil.rmtree(work, ignore_errors=True)
    return 1 if receipt["status"] == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(main())
