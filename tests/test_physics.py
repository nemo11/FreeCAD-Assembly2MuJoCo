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


def test_coupled_motor_mapping_and_rotor_inertia():
    from freecad.assembly2mujoco.core.physics import add_transmissions

    xml = ET.fromstring(
        '<mujoco><worldbody><body name="Moving">'
        '<joint name="X" type="slide" axis="1 0 0"/>'
        '<joint name="Y" type="slide" axis="0 1 0"/>'
        '<geom type="box" size=".01 .01 .01"/></body></worldbody>'
        '<actuator><position name="X" joint="X"/></actuator>'
        "<tendon/><sensor/></mujoco>"
    )
    add_transmissions(
        xml,
        [
            dict(name="A", coefficients=dict(X=100, Y=100), rotor_inertia=5e-6),
            dict(name="B", coefficients=dict(X=100, Y=-100), rotor_inertia=5e-6),
        ],
    )
    model = mujoco.MjModel.from_xml_string(ET.tostring(xml).decode())
    data = mujoco.MjData(model)
    assert model.nu == 2
    data.qpos[:] = [0.01, 0.02]
    data.ctrl[:] = [0.3, 0.1]
    mujoco.mj_forward(model, data)
    assert data.ten_length == pytest.approx([3, -1])
    assert data.qfrc_actuator == pytest.approx([40, 20])
    mass = np.zeros((2, 2))
    mujoco.mj_fullM(model, data, mass)
    # Coupled rotor contributions add .1 kg to each translational coordinate.
    assert mass[0, 0] - model.body("Moving").mass[0] == pytest.approx(0.1)


def test_cylinder_collision_primitive():
    xml = fixture_xml()
    apply_physics(
        xml,
        dict(
            collisions=[
                dict(
                    body="Base",
                    name="shaft",
                    type="cylinder",
                    pos=[0, 0, 0],
                    quat=[1, 0, 0, 0],
                    size=[0.004, 0.084],
                )
            ]
        ),
    )
    model = mujoco.MjModel.from_xml_string(ET.tostring(xml).decode())
    assert model.geom("shaft").type[0] == mujoco.mjtGeom.mjGEOM_CYLINDER


def test_convex_collision_mesh_uses_si_vertices():
    xml = fixture_xml()
    vertices = [
        [x, y, z]
        for x in (-0.002, 0.002)
        for y in (-0.003, 0.003)
        for z in (-0.004, 0.004)
    ]
    apply_physics(
        xml,
        dict(
            collisions=[
                dict(
                    body="Base",
                    name="hull",
                    type="mesh",
                    pos=[0, 0, 0],
                    quat=[1, 0, 0, 0],
                    size=[0.002, 0.003, 0.004],
                    vertices=vertices,
                )
            ]
        ),
    )
    model = mujoco.MjModel.from_xml_string(ET.tostring(xml).decode())
    assert model.geom("hull").type[0] == mujoco.mjtGeom.mjGEOM_MESH
    assert model.geom_aabb[model.geom("hull").id, 3:] == pytest.approx(
        [0.002, 0.003, 0.004], abs=1e-8
    )
