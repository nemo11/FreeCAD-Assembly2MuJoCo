import mujoco
import numpy as np
import pytest
from freecad import app

from freecad.assembly2mujoco.core.assembly import AssemblyGraph
from freecad.assembly2mujoco.core.mujoco import MuJoCoExporter


@pytest.mark.parametrize("grounded", [True, False])
def test_single_part_has_only_intended_degrees_of_freedom(
    native_assembly,
    make_part,
    export_model,
    grounded,
):
    part = make_part("Base", grounded=grounded)
    xml, model = export_model(native_assembly)
    assert model.nq == (0 if grounded else 7)
    assert model.nv == (0 if grounded else 6)
    body = xml.find(f".//body[@name='{part.Label}']")
    assert (body.find("freejoint") is None) == grounded


def test_disconnected_grounded_part_stays_fixed(
    native_assembly,
    make_part,
    export_model,
):
    fixed = make_part("Fixed", grounded=True)
    fixed.Placement.Base = app.Vector(20, 30, 40)
    make_part("Free")
    _, model = export_model(native_assembly)
    assert model.nq == 7 and model.nv == 6
    data = mujoco.MjData(model)
    for _ in range(10):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(
        data.site_xpos[model.site(fixed.Label + " site").id],
        [0.02, 0.03, 0.04],
        atol=1e-12,
    )


@pytest.mark.parametrize("grounded", [True, False])
def test_recursive_single_part_path_respects_grounding(
    native_assembly,
    make_part,
    grounded,
):
    make_part("Base", grounded=grounded)
    native_assembly.Document.recompute()
    graph = AssemblyGraph.from_assembly(native_assembly)
    exporter = MuJoCoExporter()
    body = exporter.process_tree(graph.get_nodes()[0], graph)
    assert (body.find("freejoint") is None) == grounded
