from pathlib import Path

import numpy as np
import pytest


def test_cli_uses_the_installed_numerical_library():
    import block_solver
    import field_observations
    from winding_field import field, flatten, meshes, predict, surface_mesh

    assert field.solve_blocks is block_solver.solve_blocks
    assert predict.density_normals is field_observations.density_normals
    assert meshes.extract_mesh is surface_mesh.extract_mesh
    assert flatten.MeshReader is surface_mesh.MeshReader


def test_changed_external_solver_rejects_resume_without_rewriting_products(tmp_path, monkeypatch):
    import field_solver
    from ct_fixture import write_ct
    from winding_field import workflow

    source = write_ct(tmp_path / 'ct.zarr', np.full((8, 8, 16), 200, np.uint8))
    output = tmp_path / 'run'
    config = dict(source=str(source), output=str(output), prediction='classical',
                  cache=str(tmp_path / 'cache'), reserve_bytes=0)
    state = workflow.run(config, until='prepare')
    before = {name: (output / name).read_bytes() for name in ('RUN.json', 'PREPARED.json')}
    original = workflow.file_sha
    solver = Path(field_solver.__file__).resolve()
    monkeypatch.setattr(workflow, 'file_sha',
                        lambda path: '0' * 64 if Path(path).resolve() == solver else original(path))
    with pytest.raises(ValueError, match='implementation changed'):
        workflow.run(state['config'], resume=True, until='prepare')
    assert all((output / name).read_bytes() == contents for name, contents in before.items())


def test_prepared_command_reports_input_errors_without_creating_output(tmp_path, capsys):
    from winding_field.cli import main

    output = tmp_path / 'field'
    assert main(['prepared', 'build', '--config', str(tmp_path / 'missing.json'), '--output', str(output)]) == 2
    assert 'configuration/input error' in capsys.readouterr().err
    assert not output.exists()
