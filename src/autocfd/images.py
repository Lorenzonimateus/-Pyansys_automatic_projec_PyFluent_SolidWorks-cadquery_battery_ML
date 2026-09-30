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

Figure style (Fluent-like contours):
- colormap "jet" (blue = minimum, red = maximum);
- vertical colorbar on the LEFT, with "<Variable> [unit]" above it;
- 11 equally spaced ticks from the minimum to the maximum, format "%.1f";
- white background, no grid; solids outside the plotted domain stay white;
- titles use commas/colons, never dashes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

log = logging.getLogger(__name__)

KELVIN = 273.15
CMAP = "jet"
N_TICKS = 11
TICK_FORMAT = "%.1f"
INK, INK_2 = "#000000", "#333333"
BACKGROUND = "#ffffff"


@dataclass(frozen=True)
class Field:
    fluent_name: str
    title: str
    unit: str
    convert: Callable[[np.ndarray], np.ndarray]
    fluid_only: bool

    @property
    def label(self) -> str:
        return f"{self.title} [{self.unit}]"


FIELDS = {
    "temperature": Field("temperature", "Static Temperature", "C", lambda v: v - KELVIN, False),
    "pressure": Field("pressure", "Static Pressure", "Pa", lambda v: v, True),
    "velocity": Field("velocity-magnitude", "Velocity Magnitude", "m/s", lambda v: v, True),
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
def color_limits(data: np.ndarray, value_range: tuple[float, float] | None) -> tuple[float, float]:
    """Min/max of the color scale: the fixed range if given, else the field's own."""
    if value_range is not None:
        return float(value_range[0]), float(value_range[1])
    vmin, vmax = float(np.nanmin(data)), float(np.nanmax(data))
    if vmax - vmin < 1e-12:
        vmax = vmin + 1e-12
    return vmin, vmax


def colorbar_ticks(vmin: float, vmax: float) -> np.ndarray:
    return np.linspace(vmin, vmax, N_TICKS)


def plot_plane(verts_m: np.ndarray, tris: np.ndarray, values: np.ndarray, plane: str, field: Field,
               plane_pos_mm: float, title: str, path: Path, dpi: int = 150,
               value_range: tuple[float, float] | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.ticker import FormatStrFormatter
    from matplotlib.tri import Triangulation

    _, h, v = PLANES[plane]
    x, y = verts_m[:, h] * 1000, verts_m[:, v] * 1000
    data = field.convert(values)
    vmin, vmax = color_limits(data, value_range)

    # Layout in inches: [colorbar + its labels] [plot], title block on top
    span_h, span_v = np.ptp(x) or 1.0, np.ptp(y) or 1.0
    cbar_left, cbar_w, plot_left, right = 0.45, 0.25, 2.1, 0.35
    plot_w = 10.0 - plot_left - right
    plot_h = plot_w * span_v / span_h
    if plot_h > 6.0:  # tall planes: limit the height, narrow the plot
        plot_h, plot_w = 6.0, 6.0 * span_h / span_v
    cbar_h = max(plot_h, 3.2)
    top, bottom = 1.15, 0.65
    fig_w, fig_h = plot_left + plot_w + right, cbar_h + top + bottom

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=BACKGROUND)
    ax = fig.add_axes([plot_left / fig_w, (bottom + (cbar_h - plot_h) / 2) / fig_h,
                       plot_w / fig_w, plot_h / fig_h])
    cax = fig.add_axes([cbar_left / fig_w, bottom / fig_h, cbar_w / fig_w, cbar_h / fig_h])

    tpc = ax.tripcolor(Triangulation(x, y, tris), data, cmap=CMAP, norm=Normalize(vmin, vmax),
                       shading="gouraud", rasterized=True)
    ax.set_facecolor(BACKGROUND)
    ax.set_aspect("equal")
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    ax.grid(False)
    ax.set_xlabel(f"{AXIS_NAMES[h]} [mm]", fontsize=9, color=INK_2)
    ax.set_ylabel(f"{AXIS_NAMES[v]} [mm]", fontsize=9, color=INK_2)
    ax.tick_params(labelsize=8, colors=INK_2)

    cb = fig.colorbar(tpc, cax=cax, orientation="vertical")
    cb.set_ticks(colorbar_ticks(vmin, vmax))
    cb.ax.yaxis.set_major_formatter(FormatStrFormatter(TICK_FORMAT))
    cb.ax.tick_params(labelsize=8, colors=INK)
    fig.text(cbar_left / fig_w, (bottom + cbar_h + 0.12) / fig_h, field.label,
             ha="left", va="bottom", fontsize=9, color=INK)

    normal = AXIS_NAMES[PLANES[plane][0]]
    lo, hi = float(np.nanmin(data)), float(np.nanmax(data))
    fig.text(cbar_left / fig_w, 1 - 0.25 / fig_h,
             f"{field.title}, plane {plane} ({normal} = {plane_pos_mm:.1f} mm)",
             ha="left", va="top", fontsize=11, color=INK)
    fig.text(cbar_left / fig_w, 1 - 0.55 / fig_h,
             f"{title};  min: {lo:.1f} {field.unit}, max: {hi:.1f} {field.unit}",
             ha="left", va="top", fontsize=8, color=INK_2)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, facecolor=BACKGROUND)
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
