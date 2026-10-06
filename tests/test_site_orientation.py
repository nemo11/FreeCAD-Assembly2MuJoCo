import mujoco
import numpy as np
import pytest
from freecad import app


@pytest.mark.parametrize(
    "rotation",
    [
        (0, 0, 0),
        (90, 0, 0),
        (0, 90, 0),
        (0, 0, 90),
        (25, 35, 45),
    ],
)
def test_site_pose_matches_freecad_placement(
    native_assembly,
    make_part,
    export_model,
    rotation,
):
    part = make_part("Base", grounded=True)
    part.Placement = app.Placement(app.Vector(20, 30, 40), app.Rotation(*rotation))
    expected = np.array(part.Placement.Rotation.Matrix.A).reshape(4, 4)[:3, :3]

    _, model = export_model(native_assembly)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    site_id = model.site(part.Label + " site").id
    np.testing.assert_allclose(data.site_xpos[site_id], [0.02, 0.03, 0.04], atol=1e-12)
    np.testing.assert_allclose(
        data.site_xmat[site_id].reshape(3, 3), expected, atol=1e-12
    )
