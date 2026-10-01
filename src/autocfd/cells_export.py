"""Export, from a solved CFD case, what the cell-only models need (cell_thermal.py, pinn.py).

For every solid zone cell_*:
  - geometry: axis (xc, zc), radius, height (from the vertices of its walls);
  - h(phi, y) = q''(phi, y) / (Ts(phi, y) - T_ref) on the side wall, on a regular
    (phi, y) grid, with T_ref = mean inlet temperature; q'' and Ts are the wall heat
    flux and temperature of the cell-fluid interface (face values, area-weighted
    per grid bin). The sign of q'' is fixed so that the integral is the heat that
    LEAVES the cell, and that integral is checked against the heat generated;
  - steady CFD temperatures of the cell (max, volume mean) for validation;
  - material (density, cp, orthotropic conductivity in global axes), read from the
    case file itself, so the reduced model uses exactly what Fluent used.
Saved with cell_thermal.save_cells to <dir>/<case>.npz.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np

from .cell_thermal import CellModel, save_cells

log = logging.getLogger(__name__)

HEAT_FLUX_FIELDS = ("heat-flux", "total-surface-heat-flux", "wall-heat-flux", "total-heat-flux")


# ------------------------------------------------------------------ material from the .cas.h5
def case_materials(case_file: Path) -> dict[str, dict]:
    """{zone: {"material", "rho", "cp", "kx", "ky", "kz"}} for the solid zones of a case."""
    from .inspect_case import _settings_text
    import h5py

    text = _settings_text(case_file)
    with h5py.File(case_file, "r") as f:
        raw = f["settings"]["Thread Variables"][()]
    threads = raw.decode(errors="ignore") if isinstance(raw, bytes) else b"".join(raw).decode(errors="ignore")
    out = {}
    # e.g. "(40908 solid cell_1 1)(\n (material . cell_p50b)"
    for zone, material in re.findall(r"\(\d+ solid ([\w.-]+) \d+\)\(\s*\(material \. ([\w.-]+)\)", threads):
        out[zone] = {"material": material, **solid_properties(text, material)}
    return out


def solid_properties(text: str, material: str) -> dict[str, float]:
    i = text.index(f"({material} solid")
    depth, end = 0, i
    for end in range(i, len(text)):  # the material's own s-expression only
        depth += {"(": 1, ")": -1}.get(text[end], 0)
        if depth == 0:
            break
    block = text[i:end + 1]
    num = r"([-+0-9.eE]+)"
    rho = float(re.search(r"\(density \(constant \. " + num, block).group(1))
    cp = float(re.search(r"\(specific-heat \(constant \. " + num, block).group(1))
    iso = re.search(r"\(thermal-conductivity \(constant \. " + num, block)
    if iso:
        k = float(iso.group(1))
        return {"rho": rho, "cp": cp, "kx": k, "ky": k, "kz": k}
    m = re.search(r"\(orthotropic \(direction-0 ([-0-9. ]+)\) \(direction-1 ([-0-9. ]+)\) "
                  r"\(k0 \(constant \. " + num + r"\)\) \(k1 \(constant \. " + num + r"\)\) "
                  r"\(k2 \(constant \. " + num + r"\)\)", block)
    if not m:
        raise ValueError(f"Thermal conductivity of {material} not understood (isotropic or orthotropic only)")
    d0, d1 = np.array(m.group(1).split(), float), np.array(m.group(2).split(), float)
    d2 = np.cross(d0, d1)
    ks = (float(m.group(3)), float(m.group(4)), float(m.group(5)))
    k_axes = np.zeros(3)
    for d, k in zip((d0, d1, d2), ks):
        axis = int(np.argmax(np.abs(d)))
        if abs(abs(d[axis]) - np.linalg.norm(d)) > 1e-6:
            raise ValueError(f"{material}: orthotropic directions must be aligned with x, y, z")
        k_axes[axis] = k
    return {"rho": rho, "cp": cp, "kx": k_axes[0], "ky": k_axes[1], "kz": k_axes[2]}


# ------------------------------------------------------------------ field data helpers
def _faces(fd, surface: str):
    """Face centroids (m), areas (m2) and the field request name of a surface."""
    from ansys.fluent.core import SurfaceDataType, SurfaceFieldDataRequest

    geo = fd.get_field_data(SurfaceFieldDataRequest(
        surfaces=[surface], data_types=[SurfaceDataType.Vertices, SurfaceDataType.FacesConnectivity]))[surface]
    verts = np.asarray(geo.vertices, float).reshape(-1, 3)
    cents, areas = [], []
    for poly in geo.connectivity:
        p = verts[np.asarray(poly, int)]
        c = p.mean(axis=0)
        a = 0.0
        for k in range(len(p)):
            a += 0.5 * np.linalg.norm(np.cross(p[k] - c, p[(k + 1) % len(p)] - c))
        cents.append(c)
        areas.append(a)
    return np.asarray(cents), np.asarray(areas), verts


def _face_values(fd, surface: str, field: str) -> np.ndarray:
    from ansys.fluent.core import ScalarFieldDataRequest

    return np.asarray(fd.get_field_data(ScalarFieldDataRequest(
        field_name=field, surfaces=[surface], node_value=False, boundary_value=True))[surface], float)


def _heat_flux_field(fd, surface: str) -> str:
    for name in HEAT_FLUX_FIELDS:
        try:
            _face_values(fd, surface, name)
            return name
        except Exception:  # noqa: BLE001
            continue
    raise RuntimeError(f"No wall heat flux field among {HEAT_FLUX_FIELDS}")


def _zone_temperatures(session, zone: str) -> tuple[float, float]:
    """(max, volume-weighted mean) cell temperature of a zone, K."""
    svd = session.fields.solution_variable_data
    t = np.asarray(svd.get_data(variable_name="SV_T", zone_names=[zone], domain_name="mixture")[zone], float)
    try:
        v = np.asarray(svd.get_data(variable_name="SV_VOLUME", zone_names=[zone], domain_name="mixture")[zone],
                       float)
        mean = float(np.average(t, weights=np.abs(v)))
    except Exception:  # noqa: BLE001
        mean = float(t.mean())
    return float(t.max()), mean


def _bin(phi, y, values, weights, nphi, ny, height):
    """Area-weighted mean per (phi, y) bin; empty bins filled from the nearest filled one."""
    j = np.clip((np.mod(phi, 2 * np.pi) / (2 * np.pi) * nphi).astype(int), 0, nphi - 1)
    k = np.clip((y / height * ny).astype(int), 0, ny - 1)
    s = np.zeros((nphi, ny))
    w = np.zeros((nphi, ny))
    np.add.at(s, (j, k), values * weights)
    np.add.at(w, (j, k), weights)
    out = np.where(w > 0, s / np.where(w > 0, w, 1), np.nan)
    if np.isnan(out).any():
        jj, kk = np.nonzero(~np.isnan(out))
        for a, b in zip(*np.nonzero(np.isnan(out))):
            dj = np.minimum(np.abs(jj - a), nphi - np.abs(jj - a)) / nphi
            dk = np.abs(kk - b) / ny
            n = np.argmin(dj ** 2 + dk ** 2)
            out[a, b] = out[jj[n], kk[n]]
    return out


# ------------------------------------------------------------------ export
def export_cells(session, case_file: Path, path: Path, q_cell_w: float | None = None,
                 nphi: int = 72, ny: int = 28, inlet: str = "inlet") -> list[CellModel]:
    """Write <path> (.npz) with one CellModel per solid zone of the solved case."""
    fd = session.fields.field_data
    materials = case_materials(case_file)
    surfaces = fd.surfaces()
    c_in, a_in, _ = _faces(fd, inlet)
    t_ref = float(np.average(_face_values(fd, inlet, "temperature"), weights=a_in))

    cells, flux_field = [], None
    for zone in sorted(materials):
        side = [s for s in surfaces if s.startswith(f"{zone}-")]
        if not side:
            raise RuntimeError(f"No interface wall found for {zone} among {surfaces}")
        surf = next((s for s in side if not s.endswith("-shadow")), side[0])
        flux_field = flux_field or _heat_flux_field(fd, surf)
        cents, areas, verts = _faces(fd, surf)
        qf = _face_values(fd, surf, flux_field)
        tf = _face_values(fd, surf, "temperature")
        q_total = float(np.sum(qf * areas))
        if q_total < 0:  # Fluent's sign is per side of the wall: make "leaving the cell" positive
            qf, q_total = -qf, -q_total

        xc, zc = float(verts[:, 0].mean()), float(verts[:, 2].mean())
        radius = float(np.mean(np.hypot(verts[:, 0] - xc, verts[:, 2] - zc)))
        y0, y1 = float(verts[:, 1].min()), float(verts[:, 1].max())
        phi = np.arctan2(cents[:, 2] - zc, cents[:, 0] - xc)
        yy = cents[:, 1] - y0
        qmap = _bin(phi, yy, qf, areas, nphi, ny, y1 - y0)
        tmap = _bin(phi, yy, tf, areas, nphi, ny, y1 - y0)
        hmap = qmap / np.maximum(tmap - t_ref, 1e-3)
        t_max, t_mean = _zone_temperatures(session, zone)
        q_gen = q_cell_w if q_cell_w is not None else q_total
        if q_cell_w is not None and abs(q_total - q_cell_w) > 0.05 * q_cell_w:
            log.warning("%s: heat through the side wall %.3f W vs generated %.3f W (%.1f%%)", zone,
                        q_total, q_cell_w, 100 * (q_total / q_cell_w - 1))
        m = materials[zone]
        cells.append(CellModel(
            name=zone, radius_m=radius, height_m=y1 - y0, kx=m["kx"], ky=m["ky"], kz=m["kz"],
            rho=m["rho"], cp=m["cp"],
            phi=(np.arange(nphi) + 0.5) * 2 * np.pi / nphi, y=(np.arange(ny) + 0.5) * (y1 - y0) / ny,
            h=hmap, t_ref_k=t_ref, center_m=(xc, zc), q_cell_cfd_w=q_gen, t_max_cfd_k=t_max,
            t_mean_cfd_k=t_mean, ts_cfd_k=tmap,
            meta={"q_side_w": q_total, "flux_field": flux_field, "material": m["material"]}))
        log.info("%s: R %.2f mm, H %.2f mm, h %.1f..%.1f W/m2K (mean %.1f), side heat %.3f W, "
                 "T max %.2f C", zone, radius * 1e3, (y1 - y0) * 1e3, hmap.min(), hmap.max(), hmap.mean(),
                 q_total, t_max - 273.15)
    save_cells(path, cells, {"case": str(case_file), "t_ref_k": t_ref, "flux_field": flux_field})
    return cells
