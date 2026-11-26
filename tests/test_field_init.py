import numpy as np


def test_initial_values_from_workers_equal_the_one_process_loop(tmp_path, monkeypatch):
    from test_field import prepared_run
    from winding_field import global_pairs
    from winding_field.field import open_field, solve

    def field_after(name, workers):
        monkeypatch.setattr(global_pairs, "default_workers", lambda: workers)
        run = prepared_run(tmp_path / name, shape=(6, 6, 32), chunk=8, variable=True)
        result = solve(run, dict(sweeps=1, field_chunk=8, field_memory="disk", rtol=1e-8))
        assert result["status"] == "FIELD_READY"
        with open_field(run) as source:
            return source.read([0, 0, 0], [6, 6, 32])

    inline = field_after("inline", 1)
    pooled = field_after("pooled", 3)
    for a, b in zip(inline, pooled):
        np.testing.assert_array_equal(a, b)
