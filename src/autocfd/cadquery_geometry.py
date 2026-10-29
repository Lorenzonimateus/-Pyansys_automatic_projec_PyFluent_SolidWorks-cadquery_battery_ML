"""CadQuery stage: the same part as the SolidWorks one, built in Python and saved as STEP.

Replica of geometria_parametrizada.SLDPRT (Sketch6 extruded along +Y), with the same
global variables ([cad] D, L1, P1), body names and coordinates, so that SpaceClaim,
the Fluent Meshing workflow (.wft) and the Fluent base case work unchanged:

  volume   air duct, x in [0, duct_length], y in [0, cell_height], z in [-H, 0]
  cell_N   9 cylinders (3 x 3) along +Y, diameter D
           x = first_cell_x + i (D + P1)        columns, i = 0, 1, 2 (flow direction)
           z = -(first_cell_depth + j (D + L1)) rows, j = 0, 1, 2
  H = first_cell_depth + 2 (D + L1) + D/2 + L1  (the bottom wall is L1 below the last row)

The cell numbering follows the SolidWorks part (CELLS). The duct length is fixed
([geometry] duct_length_mm): in the SolidWorks sketch it is not set by any dimension
and moves a few mm when D or L1 change.

Output: output/cases/<key>/geometry.step, one part with a named solid per body.
SpaceClaim's STEP reader drops those names, so spaceclaim.py names the bodies again
by their bounding boxes (body_boxes).
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

from .config import Config

log = logging.getLogger(__name__)

# body name: (column i, row j), as numbered in the SolidWorks part
CELLS = {
    "cell_1": (0, 0), "cell_2": (1, 0), "cell_3": (2, 0),
    "cell_4": (2, 1), "cell_5": (1, 1), "cell_6": (0, 1),
    "cell_7": (2, 2), "cell_8": (1, 2), "cell_9": (0, 2),
}
MIN_GAP_MM = 0.1


def dimensions(cfg: Config) -> tuple[float, float, float]:
    """(D, L1, P1) of the configuration: diameter, gap between rows, gap between columns."""
    g = cfg.geometry
    try:
        return cfg.cad[g.diameter], cfg.cad[g.row_gap], cfg.cad[g.column_gap]
    except KeyError as e:
        raise KeyError(f"[geometry] names a variable that is not in [cad]: {e}") from None


def cell_centres(cfg: Config) -> dict[str, tuple[float, float]]:
    """{body name: (x, z)} of the cell axes, in mm."""
    g = cfg.geometry
    d, l1, p1 = dimensions(cfg)
    return {name: (g.first_cell_x_mm + i * (d + p1), -(g.first_cell_depth_mm + j * (d + l1)))
            for name, (i, j) in CELLS.items()}


def duct_height(cfg: Config) -> float:
    d, l1, _ = dimensions(cfg)
    return cfg.geometry.first_cell_depth_mm + 2 * (d + l1) + d / 2 + l1


def body_boxes(cfg: Config) -> dict[str, list[float]]:
    """{body name: [xmin, ymin, zmin, xmax, ymax, zmax]} in mm, fluid first."""
    g = cfg.geometry
    r = dimensions(cfg)[0] / 2
    boxes = {cfg.spaceclaim.fluid_body: [0.0, 0.0, -duct_height(cfg), g.duct_length_mm, g.cell_height_mm, 0.0]}
    for name, (x, z) in cell_centres(cfg).items():
        boxes[name] = [x - r, 0.0, z - r, x + r, g.cell_height_mm, z + r]
    return boxes


def check(cfg: Config) -> None:
    """Fail early on a combination the solid model would accept but the CFD would not
    (cells touching each other or a wall)."""
    g = cfg.geometry
    d, l1, p1 = dimensions(cfg)
    r = d / 2
    xs = [x for x, _ in cell_centres(cfg).values()]
    clearances = {
        "gap between columns (P1)": p1,
        "gap between rows (L1)": l1,
        "top wall to first row": g.first_cell_depth_mm - r,
        "inlet to first column": min(xs) - r,
        "last column to outlet": g.duct_length_mm - max(xs) - r,
    }
    bad = {k: round(v, 4) for k, v in clearances.items() if v <= MIN_GAP_MM}
    if d <= 0 or bad:
        raise ValueError(f"Invalid geometry for D={d}, L1={l1}, P1={p1}: clearances [mm] {bad}")


def write_step(path: Path, solids: dict) -> None:
    """One part with named solids, as SolidWorks writes it (MANIFOLD_SOLID_BREP('cell_1', ...)).

    A cq.Assembly would write one product per solid instead: SpaceClaim then puts the
    names on components and calls every body "Solid", and the body names are lost."""
    import cadquery as cq
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.Interface import Interface_Static
    from OCP.STEPCAFControl import STEPCAFControl_Controller, STEPCAFControl_Writer
    from OCP.STEPControl import STEPControl_AsIs
    from OCP.TCollection import TCollection_ExtendedString
    from OCP.TDataStd import TDataStd_Name
    from OCP.TDocStd import TDocStd_Document
    from OCP.XCAFDoc import XCAFDoc_DocumentTool

    doc = TDocStd_Document(TCollection_ExtendedString("XmlOcaf"))
    tool = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
    part = tool.AddShape(cq.Compound.makeCompound(list(solids.values())).wrapped, False)
    TDataStd_Name.Set_s(part, TCollection_ExtendedString("geometry"))
    for name, solid in solids.items():
        TDataStd_Name.Set_s(tool.AddSubShape(part, solid.wrapped), TCollection_ExtendedString(name))
    STEPCAFControl_Controller.Init_s()  # registers the parameter below
    Interface_Static.SetIVal_s("write.stepcaf.subshapes.name", 1)
    Interface_Static.SetCVal_s("write.step.unit", "MM")
    writer = STEPCAFControl_Writer()
    writer.SetNameMode(True)
    writer.Transfer(doc, STEPControl_AsIs)
    if writer.Write(str(path)) != IFSelect_RetDone:
        raise RuntimeError(f"Could not write {path}")


def build(cfg: Config, case_dir: Path) -> Path:
    """Build the fluid and the cells, save them as STEP and check it. Returns the .step."""
    import cadquery as cq

    check(cfg)
    g = cfg.geometry
    d, l1, p1 = dimensions(cfg)
    h, length, height = g.cell_height_mm, g.duct_length_mm, duct_height(cfg)
    cells = {name: cq.Solid.makeCylinder(d / 2, h, cq.Vector(x, 0.0, z), cq.Vector(0, 1, 0))
             for name, (x, z) in cell_centres(cfg).items()}
    box = cq.Solid.makeBox(length, h, height, cq.Vector(0.0, 0.0, -height))
    fluid = box.cut(*cells.values()).Solids()  # cut returns a compound
    if len(fluid) != 1:
        raise RuntimeError(f"The fluid is {len(fluid)} solids, expected one")
    fluid = fluid[0]

    case_dir.mkdir(parents=True, exist_ok=True)
    step = case_dir / "geometry.step"
    write_step(step, {cfg.spaceclaim.fluid_body: fluid, **cells})

    solids = cq.importers.importStep(str(step)).solids().vals()
    expected_fluid = length * h * height - len(cells) * math.pi * (d / 2) ** 2 * h
    if len(solids) != len(cells) + 1:
        raise RuntimeError(f"{step.name} has {len(solids)} solids, expected {len(cells) + 1}")
    if abs(max(s.Volume() for s in solids) - expected_fluid) > 1e-6 * expected_fluid:
        raise RuntimeError(f"{step.name}: fluid volume differs from the expected {expected_fluid:.1f} mm3")
    log.info("CadQuery: %s with D=%g, L1=%g, P1=%g (duct %.3f x %.3f x %.3f mm)",
             step.name, d, l1, p1, length, h, height)
    return step
