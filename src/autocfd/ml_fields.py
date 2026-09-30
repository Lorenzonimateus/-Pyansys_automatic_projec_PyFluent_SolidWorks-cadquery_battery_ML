"""Physical fields on a FIXED regular grid, for training CNNs (U-Net and similar).

The colored images (jet) are for people. For a model they are a poor format:
one value becomes three non-linear RGB channels, 8-bit quantization loses
precision, and with a per-image scale the same color means different values.
So, for each solved point and each mid-plane (xy, yz, xz), this module writes a
.npz with the raw fields:

    T, p, U          float32, °C / Pa (gauge) / m/s; NaN where not defined
                     (outside the domain; p and U also inside the solids)
    T_norm, ...      the same, normalized with FIXED limits for the whole dataset
                     ([ml_fields.normalization]); 0 where not defined
    mask             float32, 1 = fluid, 0 = solid or outside the domain
    sdf              float32, signed distance to the nearest solid, wall or grid edge in mm,
                     > 0 in the fluid, < 0 in the solids / outside
    params           float32 vector of the point's parameters (param_names)
    h, v             grid coordinates in mm (horizontal, vertical axes of the plane)

Grid: same domain and resolution for EVERY case ([ml_fields] x_mm, y_mm, z_mm,
resolution); changing the geometry moves the solids inside the grid, never the
grid. Interpolation is linear on Fluent's own triangulation of the cut, so it
never "fills" the cells with fluid values the way a convex-hull griddata would.

Layout: <dataset>/<plane>/<case>.npz plus <dataset>/normalization.json.
Typical model: input = [mask, sdf] + params (constant channels or a vector),
output = T_norm (and/or p_norm, U_norm). See load_dataset().
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from .images import AXIS_NAMES, FIELDS, PLANES, PlaneCutter

log = logging.getLogger(__name__)

# npz key of each field
KEYS = {"temperature": "T", "pressure": "p", "velocity": "U"}


# ------------------------------------------------------------------ grid
def grid_axes(ml) -> list[np.ndarray]:
    """Grid coordinates in mm along x, y, z ([ml_fields] bounds and resolution)."""
    bounds = (ml.x_mm, ml.y_mm, ml.z_mm)
    return [np.linspace(lo, hi, n) for (lo, hi), n in zip(bounds, ml.resolution)]


def plane_grid(plane: str, axes: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """(horizontal, vertical) 1-D coordinates in mm of the grid of a plane.
    The 2-D arrays have shape (len(vertical), len(horizontal))."""
    _, h, v = PLANES[plane]
    return axes[h], axes[v]


def rasterize(px: np.ndarray, py: np.ndarray, tris: np.ndarray, values: np.ndarray,
              h: np.ndarray, v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Linear (barycentric) interpolation of triangle data onto a uniform grid.

    Each grid point takes the value of the triangle that contains it; NaN where no
    triangle does. Unlike matplotlib's interpolators it accepts what Fluent's cuts
    contain: duplicated vertices at zone interfaces and degenerate triangles."""
    out = np.full((len(v), len(h)), np.nan)
    if len(tris) == 0:
        return out
    dh, dv = h[1] - h[0], v[1] - v[0]
    tx, ty = px[tris], py[tris]                                          # (m, 3)
    i0 = np.clip(np.ceil((tx.min(1) - h[0]) / dh - eps), 0, len(h)).astype(int)
    i1 = np.clip(np.floor((tx.max(1) - h[0]) / dh + eps), -1, len(h) - 1).astype(int)
    j0 = np.clip(np.ceil((ty.min(1) - v[0]) / dv - eps), 0, len(v)).astype(int)
    j1 = np.clip(np.floor((ty.max(1) - v[0]) / dv + eps), -1, len(v) - 1).astype(int)
    ni, nj = np.clip(i1 - i0 + 1, 0, None), np.clip(j1 - j0 + 1, 0, None)
    count = ni * nj                                                      # grid points per triangle bbox
    t = np.repeat(np.arange(len(tris)), count)
    k = np.arange(count.sum()) - np.repeat(np.cumsum(count) - count, count)
    ii, jj = i0[t] + k % ni[t], j0[t] + k // ni[t]
    x, y = h[ii], v[jj]
    (x1, x2, x3), (y1, y2, y3) = tx[t].T, ty[t].T
    det = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
    ok = np.abs(det) > 1e-12
    det = np.where(ok, det, 1.0)
    l1 = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / det
    l2 = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / det
    l3 = 1.0 - l1 - l2
    ok &= (l1 >= -eps) & (l2 >= -eps) & (l3 >= -eps)
    vt = values[tris][t]
    out[jj[ok], ii[ok]] = (l1 * vt[:, 0] + l2 * vt[:, 1] + l3 * vt[:, 2])[ok]
    return out


def interpolate(verts_m: np.ndarray, tris: np.ndarray, values: np.ndarray, plane: str,
                h: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Linear interpolation on the cut's triangles (grid in mm); NaN outside them."""
    _, ih, iv = PLANES[plane]
    return rasterize(verts_m[:, ih] * 1000, verts_m[:, iv] * 1000, np.asarray(tris, int),
                     np.asarray(values, float), h, v)


def inside(verts_m: np.ndarray, tris: np.ndarray, plane: str, h: np.ndarray, v: np.ndarray) -> np.ndarray:
    """1 where the grid point lies on the cut's triangles, else 0."""
    return np.isfinite(interpolate(verts_m, tris, np.zeros(len(verts_m)), plane, h, v)).astype(np.float32)


def signed_distance(mask: np.ndarray, dh: float, dv: float) -> np.ndarray:
    """Signed distance (mm) to the fluid boundary: > 0 in the fluid, < 0 outside it.

    The grid is padded with one ring of "solid", so the edges of the grid count as
    boundaries too: walls that coincide with the grid edge (here z = 0, x = 0) are
    treated like the ones inside it (z = z_min, x = x_max of each geometry)."""
    from scipy.ndimage import distance_transform_edt

    fluid = np.pad(mask > 0.5, 1, constant_values=False)
    if not fluid.any():
        return np.zeros_like(mask, dtype=np.float32)
    d_in = distance_transform_edt(fluid, sampling=(dv, dh))
    d_out = distance_transform_edt(~fluid, sampling=(dv, dh))
    return (d_in - d_out)[1:-1, 1:-1].astype(np.float32)


def normalize(a: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """(a - lo) / (hi - lo) with the dataset's fixed limits; NaN -> 0."""
    return np.nan_to_num((a - lo) / (hi - lo), nan=0.0).astype(np.float32)


# ------------------------------------------------------------------ export
def write_plane(path: Path, plane: str, plane_pos_mm: float, fields_si: dict[str, tuple],
                fluid_geom: tuple[np.ndarray, np.ndarray], ml, params: dict[str, float]) -> Path:
    """fields_si: {"temperature": (verts, tris, values in SI), ...}; fluid_geom: (verts, tris)."""
    h, v = plane_grid(plane, grid_axes(ml))
    mask = inside(*fluid_geom, plane, h, v)
    out: dict[str, np.ndarray] = {
        "mask": mask,
        "sdf": signed_distance(mask, float(h[1] - h[0]), float(v[1] - v[0])),
        "params": np.array(list(params.values()), dtype=np.float32),
        "param_names": np.array(list(params), dtype=str),
        "h": h.astype(np.float32), "v": v.astype(np.float32),
        "axes": np.array([AXIS_NAMES[PLANES[plane][1]], AXIS_NAMES[PLANES[plane][2]]]),
        "plane": np.array(plane), "plane_pos_mm": np.float32(plane_pos_mm),
    }
    for fname, (verts, tris, values) in fields_si.items():
        key, field = KEYS[fname], FIELDS[fname]
        phys = interpolate(verts, tris, field.convert(values), plane, h, v)
        if fname != "temperature":
            phys[mask < 0.5] = np.nan  # p and U only exist in the fluid
        lo, hi = ml.normalization[fname]
        norm = normalize(phys, lo, hi)
        finite = phys[np.isfinite(phys)]
        tol = 1e-3 * (hi - lo)  # float32 field data + interpolation round-off is not a violation
        if finite.size and (finite.min() < lo - tol or finite.max() > hi + tol):
            log.warning("%s %s: values %.4g..%.4g outside the normalization limits [%g, %g]; "
                        "widen [ml_fields.normalization] %s", path.name, fname, finite.min(),
                        finite.max(), lo, hi, fname)
        out[key] = phys.astype(np.float32)
        out[f"{key}_norm"] = norm
        out[f"{key}_limits"] = np.array([lo, hi], dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **out)
    return path


def write_normalization(dataset_dir: Path, ml, param_names: list[str]) -> None:
    """Grid and limits shared by every file of the dataset (normalization.json)."""
    info = {
        "grid_mm": {"x": list(ml.x_mm), "y": list(ml.y_mm), "z": list(ml.z_mm)},
        "resolution": {"x": ml.resolution[0], "y": ml.resolution[1], "z": ml.resolution[2]},
        "normalization": {KEYS[k]: {"min": lo, "max": hi, "unit": FIELDS[k].unit}
                          for k, (lo, hi) in ml.normalization.items()},
        "param_names": param_names,
        "mask": "1 = fluid, 0 = solid or outside the domain",
        "sdf": "signed distance in mm to the nearest solid, wall or grid edge, > 0 in the fluid",
    }
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "normalization.json").write_text(json.dumps(info, indent=2))


def write_fields(cutter: PlaneCutter, dataset_dir: Path, name: str, ml, params: dict[str, float]) -> list[Path]:
    """All planes of one point: <dataset_dir>/<plane>/<name>.npz."""
    lo_mm, hi_mm = cutter.lo * 1000, cutter.hi * 1000
    grid_lo = np.array([ml.x_mm[0], ml.y_mm[0], ml.z_mm[0]])
    grid_hi = np.array([ml.x_mm[1], ml.y_mm[1], ml.z_mm[1]])
    if (lo_mm < grid_lo - 1e-3).any() or (hi_mm > grid_hi + 1e-3).any():
        log.warning("Domain of %s (%s..%s mm) exceeds the ML grid (%s..%s mm): part of the case is "
                    "cut off; widen [ml_fields] x_mm/y_mm/z_mm", name, lo_mm.round(2), hi_mm.round(2),
                    grid_lo, grid_hi)
    write_normalization(dataset_dir, ml, list(params))
    written = []
    for plane in ml.planes:
        fields_si = {f: cutter.cut(plane, FIELDS[f]) for f in KEYS}
        fluid = cutter.geometry(plane, fluid_only=True)
        written.append(write_plane(dataset_dir / plane / f"{name}.npz", plane,
                                   cutter.plane_position_mm(plane), fields_si, fluid, ml, params))
    log.info("%d ML field files written to %s", len(written), dataset_dir)
    return written


# ------------------------------------------------------------------ training side
def load_dataset(dataset_dir: str | Path, plane: str, targets=("T_norm",)) -> dict[str, np.ndarray]:
    """Stack every case of a plane for training.

    Returns names (N,), geometry (N, 2, H, W) = [mask, sdf], params (N, P),
    targets (N, len(targets), H, W), plus param_names and the grid (h, v).
    Checks that all files share the same grid, limits and parameters."""
    files = sorted(Path(dataset_dir, plane).glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz in {Path(dataset_dir, plane)}")
    geo, par, tgt, names = [], [], [], []
    ref = None
    for f in files:
        d = np.load(f)
        sig = (d["h"].tobytes(), d["v"].tobytes(), tuple(d["param_names"]),
               tuple(d[f"{k}_limits"].tobytes() for k in KEYS.values()))
        if ref is None:
            ref = sig
        elif sig != ref:
            raise ValueError(f"{f.name} has a different grid, limits or parameters than {files[0].name}")
        geo.append(np.stack([d["mask"], d["sdf"]]))
        par.append(d["params"])
        tgt.append(np.stack([d[t] for t in targets]))
        names.append(f.stem)
    d0 = np.load(files[0])
    return {"names": np.array(names), "geometry": np.stack(geo), "params": np.stack(par),
            "targets": np.stack(tgt), "param_names": d0["param_names"], "h": d0["h"], "v": d0["v"]}
