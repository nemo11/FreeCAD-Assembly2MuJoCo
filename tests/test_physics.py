import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from freecad.assembly2mujoco.core.physics import apply_physics, inertia_matrix


def fixture_xml():
    return ET.fromstring(
        '<mujoco><compiler inertiafromgeom="true"/><worldbody>'
        '<body name="Base"><geom type="box" size="1 1 .1"/>'
        '<body name="Tool"><joint name="Z" type="slide"/>'
        '<geom type="box" size=".01 .01 .01"/></body></body>'
        "</worldbody><contact/></mujoco>"
    )


def inertial(mass):
    return dict(mass=mass, pos=[0, 0, 0], fullinertia=[0.01, 0.01, 0.01, 0, 0, 0])


def test_explicit_inertia_is_independent_of_geom_density():
    xml = fixture_xml()
    apply_physics(
        xml,
        dict(
            require_inertials=True, inertials=dict(Base=inertial(2), Tool=inertial(0.5))
        ),
    )
    model = mujoco.MjModel.from_xml_string(ET.tostring(xml).decode())
    assert model.body("Tool").mass[0] == pytest.approx(0.5)
    assert model.body("Tool").inertia == pytest.approx([0.01, 0.01, 0.01])
    assert xml.find("compiler").get("inertiafromgeom") == "false"


def test_missing_inertials_fail_in_physical_mode():
    with pytest.raises(ValueError, match="Missing explicit"):
        apply_physics(fixture_xml(), dict(require_inertials=True, inertials={}))


@pytest.mark.parametrize(
    "values", [[-1, 1, 1, 0, 0, 0], [1, 1, 5, 0, 0, 0], [1, 1, 1, np.nan, 0, 0]]
)
def test_invalid_inertia_rejected(values):
    with pytest.raises(ValueError):
        inertia_matrix(values)


def test_explicit_contact_between_parent_and_child():
    xml = fixture_xml()
    boxes = [
        dict(
            body=body,
            name=name,
            pos=[0, 0, 0],
            quat=[1, 0, 0, 0],
            size=[0.01, 0.01, 0.01],
        )
        for body, name in [("Base", "a"), ("Tool", "b")]
    ]
    apply_physics(xml, dict(collisions=boxes, pairs=[["a", "b"]]))
    model = mujoco.MjModel.from_xml_string(ET.tostring(xml).decode())
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert data.ncon > 0
    assert np.all(model.geom_contype == 0)
