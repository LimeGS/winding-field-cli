import json

import numpy as np
import tifffile
from PIL import Image

from winding_field.cli import main


def test_patch_command_reports_physical_residuals_and_writes_picture(tmp_path):
    field = tmp_path / "field.npy"
    np.save(field, np.broadcast_to(np.arange(40.0), (3, 3, 40)))
    metadata = tmp_path / "field.json"
    metadata.write_text(json.dumps(dict(frame="test/world", origin_xyz=[0, 0, 0],
        spacing_xyz=[1, 1, 1], shape_zyx=[3, 3, 40], step=2)))
    patch = tmp_path / "patch"
    patch.mkdir()
    (patch / "meta.json").write_text(json.dumps(dict(format="tifxyz", uuid="synthetic")))
    for axis, values in zip("xyz", [np.arange(40.0)[None, :], np.ones((1, 40)), np.ones((1, 40))]):
        tifffile.imwrite(patch / f"{axis}.tif", values)
    output = tmp_path / "scores.json"
    assert main(["patch-visuals", "--field", str(field), "--metadata", str(metadata),
                 "--patch", str(patch), "--patch-frame", "test/world", "--margin", "0",
                 "--output", str(output)]) == 0
    result = json.loads(output.read_text())
    assert result["patches"][0]["covered_points"] == 40
    assert result["patches"][0]["within_1_sheet_pct"] == 10
    with Image.open(tmp_path / "scores_synthetic.png") as image:
        image.verify()
