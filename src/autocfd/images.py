"""Contour images of temperature, pressure and velocity on the three mid-planes.

For each solved point, three cutting planes through the CENTRE of the domain
(bounding box of all boundary surfaces) are created in Fluent as iso-surfaces of
x/y/z-coordinate:
    xy  (z = z_mid)      yz  (x = x_mid)      xz  (y = y_mid)
Temperature is cut through fluid + solids (the cells' temperature is the point);
pressure and velocity only through the fluid (the cells are left blank).

The field values come from PyFluent field data, always in SI (K, Pa, m/s) whatever
the display units of the case, and are converted here: temperature is ALWAYS
plotted in °C. The images are drawn with matplotlib, one PNG per field and plane:
<images_dir>/<field>_<plane>.png.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

log = logging.getLogger(__name__)

KELVIN = 273.15

# Colors (project palette): ink, grid, surface; sequential ramps light -> dark
INK, INK_2, MUTED, SURFACE = "#0b0b0b", "#52514e", "#898781", "#fcfcfb"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
ORANGE_RAMP = ["#fde3d6", "#f9c3a8", "#f4a07a", "#eb6834", "#c9501f", "#9c3a12", "#6e270a"]
# Diverging blue <-> red around a neutral gray midpoint (0 Pa gauge)
DIVERGING = ["#0d366b", "#256abf", "#6da7ec", "#cde2fb", "#f0efec", "#f9d3d2", "#f09a99", "#e34948", "#9e2423"]


@dataclass(frozen=True)
class Field:
    fluent_name: str
    title: str
    unit: str
    convert: Callable[[np.ndarray], np.ndarray]
    fluid_only: bool
    colors: list[str]
    diverging: bool = False


FIELDS = {
    "temperature": Field("temperature", "Temperature", "°C", lambda v: v - KELVIN, False, ORANGE_RAMP),
    "pressure": Field("pressure", "Static pressure (gauge)", "Pa", lambda v: v, True, DIVERGING, True),
    "velocity": Field("velocity-magnitude", "Velocity magnitude", "m/s", lambda v: v, True, BLUE_RAMP),
}

# plane -> (normal axis, horizontal axis, vertical axis); y (the height) is vertical
# whenever it is in the plane, so xy and yz read the same way
PLANES = {"xy": (2, 0, 1), "yz": (0, 2, 1), "xz": (1, 0, 2)}
AXIS_NAMES = "xyz"


# ------------------------------------------------------------------ geometry
def domain_bounds(field_data, surfaces: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """(min, max) corner in metres of the vertices of the given surfaces."""
    from ansys.fluent.core import SurfaceDataType, SurfaceFieldDataRequest

    req = SurfaceFieldDataRequest(surfaces=surfaces, data_types=[SurfaceDataType.Vertices])
    data = field_data.get_field_data(req)
    pts = np.vstack([np.asarray(data[s].vertices).reshape(-1, 3) for s in surfaces])
    return pts.min(axis=0), pts.max(axis=0)


def triangulate(connectivity) -> np.ndarray:
    """Polygons (lists of vertex indices) -> triangles by fans."""
    tris = []
    for poly in connectivity:
        poly = np.asarray(poly, dtype=int)
        for k in range(1, len(poly) - 1):
            tris.append((poly[0], poly[k], poly[k + 1]))
    return np.asarray(tris, dtype=int)


# ------------------------------------------------------------------ Fluent side
class PlaneCutter:
    """Creates the cutting planes in an open Fluent session and reads the fields on them."""

    def __init__(self, session):
        self.session = session
        self.fd = session.fields.field_data
        czc = session.settings.setup.cell_zone_conditions
        self.fluid_zones = list(czc.fluid.get_object_names())
        self.all_zones = self.fluid_zones + list(czc.solid.get_object_names())
        boundaries = [s for s in self.fd.surfaces() if not s.startswith("autocfd_")]
        self.lo, self.hi = domain_bounds(self.fd, boundaries)
        self.mid = (self.lo + self.hi) / 2
        self._created: list[str] = []

    def _surface(self, plane: str, fluid_only: bool) -> str:
        normal = PLANES[plane][0]
        name = f"autocfd_{plane}_{'fluid' if fluid_only else 'all'}"
        if name not in self._created:
            iso = self.session.settings.results.surfaces.iso_surface
            iso[name] = {"field": f"{AXIS_NAMES[normal]}-coordinate",
                         "iso_values": [float(self.mid[normal])],
                         "zones": self.fluid_zones if fluid_only else self.all_zones}
            self._created.append(name)
        return name

    def cut(self, plane: str, field: Field) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(vertices (n,3) in m, triangles (m,3), values (n,) in SI) of a field on a plane."""
        from ansys.fluent.core import (ScalarFieldDataRequest, SurfaceDataType,
                                       SurfaceFieldDataRequest)

        name = self._surface(plane, field.fluid_only)
        geo = self.fd.get_field_data(SurfaceFieldDataRequest(
            surfaces=[name], data_types=[SurfaceDataType.Vertices, SurfaceDataType.FacesConnectivity]))[name]
        values = self.fd.get_field_data(ScalarFieldDataRequest(
            field_name=field.fluent_name, surfaces=[name], node_value=True, boundary_value=True))[name]
        verts = np.asarray(geo.vertices, dtype=float).reshape(-1, 3)
        return verts, triangulate(geo.connectivity), np.asarray(values, dtype=float)

    def cleanup(self) -> None:
        """Remove the planes, so they are not written into the saved cases."""
        iso = self.session.settings.results.surfaces.iso_surface
        for name in self._created:
            try:
                del iso[name]
            except Exception as e:  # noqa: BLE001
                log.debug("Could not delete surface %s: %s", name, e)
        self._created.clear()


# ------------------------------------------------------------------ plotting
def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)


def plot_plane(verts_m: np.ndarray, tris: np.ndarray, values: np.ndarray, plane: str, field: Field,
               plane_pos_mm: float, title: str, path: Path, dpi: int = 150,
               value_range: tuple[float, float] | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.tri import Triangulation

    _, h, v = PLANES[plane]
    x, y = verts_m[:, h] * 1000, verts_m[:, v] * 1000
    data = field.convert(values)
    if value_range is not None:
        vmin, vmax = value_range
    elif field.diverging:  # symmetric around 0 so that the gray midpoint is 0
        vmax = float(np.nanmax(np.abs(data))) or 1.0
        vmin = -vmax
    else:
        vmin, vmax = float(np.nanmin(data)), float(np.nanmax(data))
        if vmax - vmin < 1e-12:
            vmax = vmin + 1e-12
    cmap = LinearSegmentedColormap.from_list(field.fluent_name, field.colors)

    span_h, span_v = np.ptp(x) or 1.0, np.ptp(y) or 1.0
    width = 9.0
    fig, ax = plt.subplots(figsize=(width, max(2.4, width * span_v / span_h * 0.85 + 1.2)), facecolor=SURFACE)
    tpc = ax.tripcolor(Triangulation(x, y, tris), data, cmap=cmap, norm=Normalize(vmin, vmax),
                       shading="gouraud", rasterized=True)
    ax.set_aspect("equal")
    ax.set_xlabel(f"{AXIS_NAMES[h]} [mm]", fontsize=9, color=INK_2)
    ax.set_ylabel(f"{AXIS_NAMES[v]} [mm]", fontsize=9, color=INK_2)
    normal = AXIS_NAMES[PLANES[plane][0]]
    lo, hi = float(np.nanmin(data)), float(np.nanmax(data))
    ax.set_title(f"{field.title} — plane {plane} ({normal} = {plane_pos_mm:.1f} mm)", fontsize=10,
                 color=INK, loc="left", pad=22)
    ax.text(0.0, 1.02, f"{title}    min {lo:.4g}  max {hi:.4g} {field.unit}", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=8, color=INK_2)
    _style(ax)
    cb = fig.colorbar(tpc, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label(field.unit, fontsize=9, color=INK_2)
    cb.ax.tick_params(labelsize=8, colors=INK_2, length=0)
    cb.outline.set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, facecolor=SURFACE)
    plt.close(fig)


def images_from_case(cfg, case_file: Path, images_dir: Path | None = None) -> list[Path]:
    """Open a saved case + data in Fluent and write its images
    (default folder: images/<case name>/ next to the case)."""
    from .meshing import launch

    case_file = Path(case_file)
    if not case_file.with_name(case_file.name.replace(".cas.h5", ".dat.h5")).exists():
        raise FileNotFoundError(f"No .dat.h5 next to {case_file}: the images need a solution")
    stem = case_file.name.replace(".cas.h5", "")
    images_dir = Path(images_dir) if images_dir else case_file.parent / "images" / stem
    im = cfg.images
    session = launch(cfg, "solver", cfg.solver.processors, case_file.parent)
    try:
        session.settings.file.read_case_data(file_name=str(case_file.resolve()))
        return save_images(session, images_dir, stem, im.fields, im.planes, im.dpi, im.ranges)
    finally:
        session.exit()


def save_images(session, images_dir: Path, title: str, fields=("temperature", "pressure", "velocity"),
                planes=("xy", "yz", "xz"), dpi: int = 150,
                ranges: dict[str, tuple[float, float]] | None = None) -> list[Path]:
    """Cut the current solution and write one PNG per field and plane."""
    cutter = PlaneCutter(session)
    written = []
    try:
        for fname in fields:
            field = FIELDS[fname]
            for plane in planes:
                verts, tris, values = cutter.cut(plane, field)
                if len(tris) == 0:
                    log.warning("Plane %s has no faces for %s; skipped", plane, fname)
                    continue
                path = Path(images_dir) / f"{fname}_{plane}.png"
                pos_mm = float(cutter.mid[PLANES[plane][0]]) * 1000
                plot_plane(verts, tris, values, plane, field, pos_mm, title, path, dpi,
                           (ranges or {}).get(fname))
                written.append(path)
    finally:
        cutter.cleanup()
    log.info("%d images written to %s", len(written), images_dir)
    return written
