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


def test_observations_from_workers_equal_the_one_process_loop(tmp_path, monkeypatch):
    """Every observation miss of a multi-block solve (no observation cache, two-level, two
    sweeps) comes from the pool, and the field is the one of the in-process loads."""
    from test_field import prepared_run
    from winding_field import field_init, global_pairs
    from winding_field.field import open_field, solve

    served = []
    get = field_init.ObservationPool.get

    def counted(self, lo, hi, **kwargs):
        result = get(self, lo, hi, **kwargs)
        served.append(result is not None)
        return result

    monkeypatch.setattr(field_init.ObservationPool, "get", counted)

    def field_after(name, workers):
        monkeypatch.setattr(global_pairs, "default_workers", lambda: workers)
        run = prepared_run(tmp_path / name, shape=(6, 6, 32), chunk=8, variable=True)
        config = dict(
            sweeps=2,
            field_chunk=8,
            field_halo=2,
            field_coarse_factor=2,
            field_memory="disk",
            observation_cache_bytes=0,
            rtol=1e-8,
        )
        result = solve(run, config)
        assert result["status"] == "FIELD_READY"
        assert set(result["setup_seconds"]) <= {"pair_catalog", "initial_values"}
        assert "initial_values" in result["setup_seconds"]
        assert ("observation_pool_stats" in result) == (workers > 1)
        with open_field(run) as source:
            return source.read([0, 0, 0], [6, 6, 32])

    inline = field_after("inline", 1)
    assert not served
    # Four workers: two build pair templates, two sample observations.
    pooled = field_after("pooled", 4)
    assert served and all(served)
    for a, b in zip(inline, pooled):
        np.testing.assert_array_equal(a, b)
