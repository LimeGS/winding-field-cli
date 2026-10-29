import json
import sys


def test_watcher_preserves_success_and_error(tmp_path):
    from winding_field.run_watch import supervise

    for code in (0, 7):
        out = tmp_path / str(code)
        result = supervise(
            [sys.executable, "-c", f'print("progress",flush=True);exit({code})'],
            out,
            timeout=10,
            reserve_bytes=0,
            interval=0.02,
        )
        assert result["returncode"] == code
        assert result["state"] == ("completed" if code == 0 else "failed")
        assert "progress" in (out / "stdout.log").read_text()
        assert json.loads((out / "status.json").read_text())["state"] == result["state"]


def test_watcher_terminates_owned_job_on_deadline(tmp_path):
    from winding_field.run_watch import supervise

    result = supervise(
        [sys.executable, "-c", "import time;time.sleep(60)"],
        tmp_path / "timeout",
        timeout=0.1,
        reserve_bytes=0,
        interval=0.02,
    )
    assert result["state"] == "timed_out"
    assert result["returncode"] is not None
