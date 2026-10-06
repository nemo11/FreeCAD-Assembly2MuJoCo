import math

import pytest


@pytest.mark.parametrize(
    "kind, property_name, lower, upper, expected",
    [
        ("Slider", "Length", "-215 mm", "21.5 cm", (-0.215, 0.215)),
        ("Revolute", "Angle", "-90 deg", "90 deg", (-math.pi / 2, math.pi / 2)),
    ],
)
def test_quantity_limits_compile_in_si_units(
    native_assembly,
    make_part,
    make_joint,
    export_model,
    kind,
    property_name,
    lower,
    upper,
    expected,
):
    parent = make_part("Base", grounded=True)
    child = make_part("Moving")
    joint = make_joint(kind, parent, child)
    for bound, value in (("Min", lower), ("Max", upper)):
        setattr(joint, "Enable" + property_name + bound, True)
        setattr(joint, property_name + bound, value)

    xml, model = export_model(native_assembly)
    assert model.nq == model.nv == model.nu == 1
    assert model.joint(joint.Label).range == pytest.approx(expected)
    actuator = model.actuator(joint.Label)
    assert actuator.ctrlrange == pytest.approx(expected)
    assert model.joint(joint.Label).limited[0]
    assert actuator.ctrllimited[0]
    joint_xml = xml.find(f".//body/joint[@name='{joint.Label}']")
    actuator_xml = xml.find(f"actuator/position[@joint='{joint.Label}']")
    assert joint_xml.get("range") == actuator_xml.get("ctrlrange")
    assert [float(value) for value in joint_xml.get("range").split()] == pytest.approx(
        expected
    )


@pytest.mark.parametrize(
    "kind, property_name", [("Slider", "Length"), ("Revolute", "Angle")]
)
def test_disabled_limits_remain_unlimited(
    native_assembly,
    make_part,
    make_joint,
    export_model,
    kind,
    property_name,
):
    parent = make_part("Base", grounded=True)
    child = make_part("Moving")
    joint = make_joint(kind, parent, child)
    for bound in ("Min", "Max"):
        setattr(joint, "Enable" + property_name + bound, False)

    xml, model = export_model(native_assembly)
    assert not model.joint(joint.Label).limited[0]
    assert not model.actuator(joint.Label).ctrllimited[0]
    assert xml.find(f".//body/joint[@name='{joint.Label}']").get("range") is None


@pytest.mark.parametrize(
    "kind, property_name", [("Slider", "Length"), ("Revolute", "Angle")]
)
@pytest.mark.parametrize("enabled_bound", ["Min", "Max"])
def test_one_sided_limits_are_not_silently_discarded(
    native_assembly,
    make_part,
    make_joint,
    export_model,
    kind,
    property_name,
    enabled_bound,
):
    parent = make_part("Base", grounded=True)
    child = make_part("Moving")
    joint = make_joint(kind, parent, child)
    for bound in ("Min", "Max"):
        setattr(joint, "Enable" + property_name + bound, bound == enabled_bound)

    with pytest.raises(NotImplementedError, match="One-sided joint limits"):
        export_model(native_assembly)
