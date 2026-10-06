import os
import xml.etree.ElementTree as ET
from pathlib import Path

import JointObject
import mujoco
import Part
import pytest
from freecad import app  # type: ignore


@pytest.fixture(scope="session")
def examples_dir() -> Path:
    examples_dir = Path(__file__).resolve().parents[1] / "examples"
    assert examples_dir.is_dir()
    return examples_dir


@pytest.fixture(scope="session")
def universal_joint_assembly(examples_dir: Path) -> app.DocumentObject:
    assembly_file = examples_dir / "universal_joint" / "universal_joint.FCStd"
    assert assembly_file.is_file()
    document = app.openDocument(os.fspath(assembly_file))
    assemblies = list(
        filter(lambda x: x.TypeId == "Assembly::AssemblyObject", document.Objects)
    )
    assert len(assemblies) == 1
    yield assemblies[0]


@pytest.fixture(scope="session")
def crank_and_slider_assembly(examples_dir: Path) -> app.DocumentObject:
    assembly_file = examples_dir / "crank_and_slider" / "crank_and_slider.FCStd"
    assert assembly_file.is_file()
    document = app.openDocument(os.fspath(assembly_file))
    assemblies = list(
        filter(lambda x: x.TypeId == "Assembly::AssemblyObject", document.Objects)
    )
    assert len(assemblies) == 1
    yield assemblies[0]


@pytest.fixture(scope="session")
def pan_tilt_assembly(examples_dir: Path) -> app.DocumentObject:
    assembly_file = examples_dir / "pan_tilt" / "pan_tilt.FCStd"
    assert assembly_file.is_file()
    document = app.openDocument(os.fspath(assembly_file))
    assemblies = list(
        filter(lambda x: x.TypeId == "Assembly::AssemblyObject", document.Objects)
    )
    assert len(assemblies) == 1
    yield assemblies[0]


@pytest.fixture
def new_document() -> app.Document:
    return app.newDocument()


@pytest.fixture
def new_document_with_assembly(new_document: app.Document) -> app.Document:
    new_document.addObject("Assembly::AssemblyObject")
    return new_document


@pytest.fixture
def native_assembly():
    document = app.newDocument()
    assembly = document.addObject("Assembly::AssemblyObject", "Assembly")
    assembly.Type = "Assembly"
    assembly.newObject("Assembly::JointGroup", "Joints")
    try:
        yield assembly
    finally:
        app.closeDocument(document.Name)


@pytest.fixture
def make_part(native_assembly):
    def create(name, *, grounded=False):
        body = native_assembly.newObject("PartDesign::Body", name)
        feature = body.newObject("PartDesign::Feature", name + "Shape")
        feature.Shape = Part.makeBox(10, 10, 10)
        if grounded:
            joint = native_assembly.Document.Joints.newObject(
                "App::FeaturePython", "Ground" + name
            )
            JointObject.GroundedJoint(joint, body)
        return body

    return create


@pytest.fixture
def make_joint(native_assembly):
    def create(kind, parent, child):
        joint = native_assembly.Document.Joints.newObject("App::FeaturePython", "Joint")
        JointObject.Joint(joint, JointObject.JointTypes.index(kind))
        joint.Detach1 = joint.Detach2 = True
        joint.Reference1 = (parent, ["", ""])
        joint.Reference2 = (child, ["", ""])
        return joint

    return create


@pytest.fixture
def export_model(tmp_path):
    from freecad.assembly2mujoco.core.assembly import AssemblyGraph
    from freecad.assembly2mujoco.core.mujoco import MuJoCoExporter

    def export(assembly):
        assembly.Document.recompute()
        graph = AssemblyGraph.from_assembly(assembly)
        exporter = MuJoCoExporter(mesh_export_format="STL")
        xml = exporter.export_assembly(graph, export_dir=tmp_path)
        path = tmp_path / "model.xml"
        ET.ElementTree(xml).write(path, encoding="utf-8", xml_declaration=True)
        return xml, mujoco.MjModel.from_xml_path(str(path))

    return export
