"""The recovery comparison must classify every frozen failure exactly once."""
import json
from pathlib import Path
import subprocess
import sys


def run(tmp_path, baseline, evidence_files):
    evidence = tmp_path / 'evidence'
    evidence.mkdir()
    for name, payload in evidence_files.items():
        path = evidence / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))
    source = Path(__file__).resolve().parents[1] / 'tools/paris4-recovery/compare.py'
    target = tmp_path / 'comparison.json'
    subprocess.run([sys.executable, str(source), '--baseline', str(baseline),
                    '--evidence', str(evidence), '--output', str(target)], check=True,
                   capture_output=True)
    return json.loads(target.read_text())


def test_every_frozen_failure_is_classified_once(tmp_path):
    baseline = tmp_path / 'CASES.json'
    baseline.write_text(json.dumps(dict(snapshot_utc='t', failed_groups={
        'ValueError: GMRES did not converge with a verified true residual': [-40, -24],
        'RuntimeError: native negative diagonal': [-41],
        'ValueError: inverted UV triangle': [-20],
    })))
    report = run(tmp_path, baseline, {
        'init-wm40/INITIALIZATION.json': dict(status='INITIALIZED',
                                              results=[dict(status='INITIALIZED', domain=0)]),
        'wm20-accuracy/ACCURACY.json': dict(mvc_direct=dict(nonpositive=7)),
    })
    states = {row['w']: row['state'] for family in report['families'] for row in family['rows']}
    assert states == {-40: 'cause_removed_verified', -24: 'cause_removed_unverified',
                      -41: 'unresolved_native', -20: 'unresolved_geometry'}
    counts = report['counts']
    assert counts['baseline'] == sum(counts[key] for key in ('cause_removed', 'unresolved', 'pending'))
