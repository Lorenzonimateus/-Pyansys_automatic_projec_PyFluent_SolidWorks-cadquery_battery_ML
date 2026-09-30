# Python Script, API Version = V252
# Runs INSIDE SpaceClaim (IronPython 2.7), started in batch by autocfd/spaceclaim.py.
# Arguments come from a JSON file whose path is in the environment variable
# AUTOCFD_SC_ARGS; the result (bodies, named selections, errors) is written to
# args["report"] as JSON.
#
# Steps: open the rebuilt .SLDPRT -> check the body names -> recreate the named
# selections of the fluid body's faces at the ends of the flow axis -> save .scdocx.

import json
import os
import traceback

args = json.loads(open(os.environ["AUTOCFD_SC_ARGS"]).read())
report = {"ok": False, "bodies": [], "named_selections": {}, "error": None}
AXIS = {"x": 0, "y": 1, "z": 2}[args["flow_axis"]]
TOL = 1e-6  # m (SpaceClaim works in SI internally)


def coord(point, axis):
    return (point.X, point.Y, point.Z)[axis]


def face_extent(face, axis):
    box = face.Shape.GetBoundingBox(Matrix.Identity)
    lo, hi = coord(box.MinCorner, axis), coord(box.MaxCorner, axis)
    return lo, hi


def name_face(face, name):
    result = NamedSelection.Create(FaceSelection.Create(face), Selection.Empty())
    NamedSelection.Rename(result.CreatedNamedSelection.GetName(), name)


try:
    DocumentOpen.Execute(args["source"])
    bodies = list(GetRootPart().GetAllBodies())
    names = [b.GetName() for b in bodies]
    report["bodies"] = names
    for required in args["required_bodies"]:
        if required not in names:
            raise Exception("Body %r not found; bodies: %s" % (required, names))

    fluid = [b for b in bodies if b.GetName() == args["fluid_body"]][0]
    # Planar faces perpendicular to the flow axis: zero extent along it
    flat = []
    for face in fluid.Faces:
        lo, hi = face_extent(face, AXIS)
        if hi - lo < TOL:
            flat.append((lo, face))
    if len(flat) < 2:
        raise Exception("Fluid body has fewer than 2 faces normal to the flow axis")
    flat.sort(key=lambda t: t[0])
    ends = {args["inlet_name"]: flat[0], args["outlet_name"]: flat[-1]}
    for name, (pos, face) in ends.items():
        name_face(face, name)
        report["named_selections"][name] = pos

    DocumentSave.Execute(args["target"])
    report["ok"] = True
except Exception:
    report["error"] = traceback.format_exc()

with open(args["report"], "w") as f:
    f.write(json.dumps(report, indent=2))
