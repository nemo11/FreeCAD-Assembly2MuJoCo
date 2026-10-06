"""Optional physical export data, expressed in SI units and body-local frames.

The exporter does not infer collision shapes, material density or motor capability.
Clients supply these alongside the assembly graph; visual-only export is unchanged.
"""

import math
import xml.etree.ElementTree as ET

import numpy as np


def numbers(values):
    return " ".join(format(float(value), ".17g") for value in values)


def inertia_matrix(values):
    xx, yy, zz, xy, xz, yz = values
    matrix = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
    eigenvalues = np.linalg.eigvalsh(matrix)
    if not np.isfinite(matrix).all() or eigenvalues[0] <= 0:
        raise ValueError("Inertia must be finite and positive definite")
    if eigenvalues[2] > eigenvalues[0] + eigenvalues[1] + 1e-12:
        raise ValueError("Inertia violates the triangle inequality")
    return matrix


def apply_physics(xml, specification):
    """Add collision geometry and, optionally, explicit inertials to MJCF.

    specification: {collisions: [{body, name, pos, quat, size, ...}],
    inertials: {body: {mass, pos, fullinertia}}, pairs: [[geom, geom]],
    require_inertials: bool}. `size` holds primitive half-extents.
    Only explicitly listed geom pairs collide, including parent/child bodies.
    """
    bodies = {body.get("name"): body for body in xml.iter("body")}
    for geom in xml.iter("geom"):
        geom.set("contype", "0")
        geom.set("conaffinity", "0")
    if specification.get("require_inertials"):
        missing = set(bodies) - set(specification.get("inertials", {}))
        if missing:
            raise ValueError(f"Missing explicit inertials: {sorted(missing)}")
        xml.find("compiler").set("inertiafromgeom", "false")
        for geom in xml.iter("geom"):
            geom.set("mass", "0")
        for joint in xml.iter("joint"):
            if joint.get("type") in ("slide", "hinge"):
                joint.set("damping", "0")
                joint.set("armature", "0")
    for name, inertial in specification.get("inertials", {}).items():
        if name not in bodies:
            raise ValueError(f"Unknown inertial body: {name}")
        if not math.isfinite(inertial["mass"]) or inertial["mass"] <= 0:
            raise ValueError("Body mass must be finite and positive")
        inertia_matrix(inertial["fullinertia"])
        if len(inertial["pos"]) != 3 or not np.isfinite(inertial["pos"]).all():
            raise ValueError("Invalid centre of mass")
        existing = bodies[name].find("inertial")
        if existing is not None:
            bodies[name].remove(existing)
        ET.SubElement(
            bodies[name],
            "inertial",
            mass=str(inertial["mass"]),
            pos=numbers(inertial["pos"]),
            fullinertia=numbers(inertial["fullinertia"]),
        )
    names = set()
    for geom in specification.get("collisions", []):
        if geom["body"] not in bodies or geom["name"] in names:
            raise ValueError("Unknown collision body or duplicate collision name")
        names.add(geom["name"])
        kind = geom.get("type", "box")
        if kind not in ("box", "cylinder"):
            raise ValueError(
                "Only box/cylinder collision primitives are currently supported"
            )
        if len(geom["size"]) != (3 if kind == "box" else 2) or min(geom["size"]) <= 0:
            raise ValueError("Invalid collision half-extents")
        if not np.isfinite(geom["size"] + geom["pos"] + geom["quat"]).all():
            raise ValueError("Nonfinite collision geometry")
        if abs(np.linalg.norm(geom["quat"]) - 1) > 1e-8:
            raise ValueError("Collision quaternion must have unit length")
        ET.SubElement(
            bodies[geom["body"]],
            "geom",
            name=geom["name"],
            type=kind,
            pos=numbers(geom["pos"]),
            quat=numbers(geom["quat"]),
            size=numbers(geom["size"]),
            mass="0",
            group="3",
            contype="0",
            conaffinity="0",
            rgba="0.2 0.7 0.9 0.35",
        )
    contact = xml.find("contact")
    for pair in specification.get("pairs", []):
        if len(pair) != 2 or pair[0] == pair[1] or not set(pair) <= names:
            raise ValueError("Invalid explicit collision pair")
        ET.SubElement(
            contact,
            "pair",
            geom1=pair[0],
            geom2=pair[1],
            condim="3",
            friction="0.5 0.5 0.001 0.0001 0.0001",
        )
    if specification.get("motors"):
        add_transmissions(xml, specification["motors"])


def add_transmissions(xml, motors):
    """Fixed-tendon motor coordinates in radians, effort in N.m."""
    joints = {joint.get("name") for joint in xml.iter("joint") if joint.get("name")}
    driven = {name for motor in motors for name in motor["coefficients"]}
    if not driven <= joints:
        raise ValueError("Transmission references unknown joints")
    actuator = xml.find("actuator")
    for item in list(actuator):
        if item.get("joint") in driven:
            actuator.remove(item)
    tendon = xml.find("tendon")
    sensor = xml.find("sensor")
    names = set()
    for motor in motors:
        name = motor["name"]
        if name in names or not motor["coefficients"]:
            raise ValueError("Duplicate motor name or empty transmission")
        names.add(name)
        coefficients = list(motor["coefficients"].values())
        rotor = motor.get("rotor_inertia", 0)
        if (
            not np.isfinite(coefficients + [rotor]).all()
            or rotor < 0
            or all(c == 0 for c in coefficients)
        ):
            raise ValueError("Invalid transmission coefficients/inertia")
        fixed = ET.SubElement(
            tendon, "fixed", name=name + "_shaft", armature=str(rotor)
        )
        for joint, coefficient in motor["coefficients"].items():
            ET.SubElement(fixed, "joint", joint=joint, coef=str(coefficient))
        ET.SubElement(actuator, "motor", name=name, tendon=name + "_shaft", gear="1")
        ET.SubElement(sensor, "tendonpos", name=name + "_angle", tendon=name + "_shaft")
        ET.SubElement(
            sensor, "tendonvel", name=name + "_velocity", tendon=name + "_shaft"
        )
        ET.SubElement(sensor, "actuatorfrc", name=name + "_torque", actuator=name)
