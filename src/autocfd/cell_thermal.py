"""Transient conduction in ONE battery cell, cooled through a CFD-derived h map.

Reduced model (the "cell-only" problem solved by the PINN and, as a reference,
by the finite-volume solver below):

    rho cp dT/dt = d/dx(kx dT/dx) + d/dy(ky dT/dy) + d/dz(kz dT/dz) + q(t)   in the cell
    -k grad(T) . n = h(phi, y) (T - T_ref)                                   on the side
    dT/dy = 0                                                                at y = 0, H

The cell is a cylinder of radius R and height H with its axis along y (as in the
CFD geometry), centred at (xc, zc); phi is the angle in the x-z plane
(x = xc + r cos phi, z = zc + r sin phi). kx, ky, kz are the GLOBAL orthotropic
conductivities of the Fluent material, so a misoriented material is reproduced
as it is in the CFD.

h(phi, y) = q''(phi, y) / (Ts(phi, y) - T_ref) comes from the steady CFD of the
case (cells_export.py), with T_ref = inlet air temperature. With constant
properties the air side is linear in the heat, so the same h holds for any
C-rate; using it in a transient assumes the air responds quasi-steadily (its
heat capacity is tiny next to the cells') and that all cells heat up together.
At steady state the model reproduces the CFD by construction (validate_steady).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

log = logging.getLogger(__name__)

KELVIN = 273.15


# ------------------------------------------------------------------ model data
@dataclass
class CellModel:
    name: str
    radius_m: float
    height_m: float
    kx: float
    ky: float
    kz: float
    rho: float
    cp: float
    phi: np.ndarray           # (nphi,) angles of the h map, [0, 2 pi)
    y: np.ndarray             # (ny,) heights of the h map, [0, H]
    h: np.ndarray             # (nphi, ny) W/m2K
    t_ref_k: float            # air reference temperature (inlet), K
    center_m: tuple[float, float] = (0.0, 0.0)  # (xc, zc)
    # steady CFD of the case, for validation
    q_cell_cfd_w: float = float("nan")
    t_max_cfd_k: float = float("nan")
    t_mean_cfd_k: float = float("nan")
    ts_cfd_k: np.ndarray | None = None          # (nphi, ny) surface temperature
    meta: dict = field(default_factory=dict)

    @property
    def volume_m3(self) -> float:
        return np.pi * self.radius_m ** 2 * self.height_m

    @property
    def side_area_m2(self) -> float:
        return 2 * np.pi * self.radius_m * self.height_m

    def h_at(self, phi: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Bilinear interpolation of the h map, periodic in phi, clamped in y."""
        return interp_map(self.phi, self.y, self.h, phi, y)

    def h_mean(self) -> float:
        return float(np.mean(self.h))


def interp_map(phi_g: np.ndarray, y_g: np.ndarray, values: np.ndarray, phi: np.ndarray,
               y: np.ndarray) -> np.ndarray:
    phi = np.mod(np.asarray(phi, float), 2 * np.pi)
    y = np.clip(np.asarray(y, float), y_g[0], y_g[-1])
    nphi = len(phi_g)
    dphi = 2 * np.pi / nphi
    fj = (phi - phi_g[0]) / dphi
    j0 = np.floor(fj).astype(int)
    wj = fj - j0
    j0 %= nphi
    j1 = (j0 + 1) % nphi
    k1 = np.clip(np.searchsorted(y_g, y), 1, len(y_g) - 1)
    k0 = k1 - 1
    wk = (y - y_g[k0]) / (y_g[k1] - y_g[k0])
    v00, v01 = values[j0, k0], values[j0, k1]
    v10, v11 = values[j1, k0], values[j1, k1]
    return (1 - wj) * ((1 - wk) * v00 + wk * v01) + wj * ((1 - wk) * v10 + wk * v11)


def save_cells(path: Path, cells: list[CellModel], meta: dict) -> None:
    out = {"names": np.array([c.name for c in cells]), "meta_keys": np.array(list(meta)),
           "meta_values": np.array([str(v) for v in meta.values()])}
    for i, c in enumerate(cells):
        p = f"c{i}_"
        out.update({p + "scalars": np.array([c.radius_m, c.height_m, c.kx, c.ky, c.kz, c.rho, c.cp,
                                             c.t_ref_k, c.center_m[0], c.center_m[1], c.q_cell_cfd_w,
                                             c.t_max_cfd_k, c.t_mean_cfd_k]),
                    p + "phi": c.phi, p + "y": c.y, p + "h": c.h,
                    p + "ts": c.ts_cfd_k if c.ts_cfd_k is not None else np.full_like(c.h, np.nan)})
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **out)


def load_cells(path: str | Path) -> list[CellModel]:
    d = np.load(path)
    meta = dict(zip(d["meta_keys"].tolist(), d["meta_values"].tolist()))
    cells = []
    for i, name in enumerate(d["names"].tolist()):
        p = f"c{i}_"
        s = d[p + "scalars"]
        cells.append(CellModel(name=name, radius_m=s[0], height_m=s[1], kx=s[2], ky=s[3], kz=s[4],
                               rho=s[5], cp=s[6], t_ref_k=s[7], center_m=(s[8], s[9]),
                               q_cell_cfd_w=s[10], t_max_cfd_k=s[11], t_mean_cfd_k=s[12],
                               phi=d[p + "phi"], y=d[p + "y"], h=d[p + "h"], ts_cfd_k=d[p + "ts"],
                               meta=meta))
    return cells


# ------------------------------------------------------------------ load
@dataclass
class Load:
    """Heat per cell over time: Q(t) = (C(t) * capacity_ah)^2 * r_dc_ohm."""
    times_s: np.ndarray       # breakpoints of C(t) (linear interpolation)
    c_rates: np.ndarray
    capacity_ah: float
    r_dc_ohm: float

    @classmethod
    def constant(cls, c_rate: float, duration_s: float, capacity_ah: float, r_dc_ohm: float) -> "Load":
        return cls(np.array([0.0, duration_s]), np.array([c_rate, c_rate]), capacity_ah, r_dc_ohm)

    @classmethod
    def from_csv(cls, path: str | Path, capacity_ah: float, r_dc_ohm: float) -> "Load":
        """CSV with columns time_s, c_rate."""
        import pandas as pd

        df = pd.read_csv(path)
        return cls(df["time_s"].to_numpy(float), df["c_rate"].to_numpy(float), capacity_ah, r_dc_ohm)

    @property
    def duration_s(self) -> float:
        return float(self.times_s[-1])

    def c_rate(self, t: np.ndarray) -> np.ndarray:
        return np.interp(t, self.times_s, self.c_rates)

    def heat_w(self, t) -> np.ndarray:
        return (self.c_rate(np.asarray(t, float)) * self.capacity_ah) ** 2 * self.r_dc_ohm

    def peak_heat_w(self) -> float:
        return float(np.max(self.heat_w(self.times_s)))


def temperature_scale(cell: CellModel, load: Load) -> float:
    """Typical temperature rise (K) used to make the problem dimensionless:
    the smaller of the adiabatic rise over the load and the steady rise."""
    q = load.peak_heat_w()
    adiabatic = q * load.duration_s / (cell.rho * cell.cp * cell.volume_m3)
    k_r = 0.5 * (cell.kx + cell.kz)
    steady = q / (cell.h_mean() * cell.side_area_m2) + q / (4 * np.pi * k_r * cell.height_m)
    return max(1.0, min(adiabatic, steady))


# ------------------------------------------------------------------ finite-volume reference
class FVSolver:
    """Implicit finite volumes on a polar grid (r, phi, y), with the full
    orthotropic tensor (cross r-phi terms included when kx != kz)."""

    def __init__(self, cell: CellModel, nr: int = 20, nphi: int = 48, ny: int = 28):
        self.cell, self.nr, self.nphi, self.ny = cell, nr, nphi, ny
        R, H = cell.radius_m, cell.height_m
        self.dr, self.dphi, self.dy = R / nr, 2 * np.pi / nphi, H / ny
        self.r = (np.arange(nr) + 0.5) * self.dr
        self.phi = (np.arange(nphi) + 0.5) * self.dphi
        self.y = (np.arange(ny) + 0.5) * self.dy
        rf = np.arange(nr + 1) * self.dr
        self.vol = (0.5 * (rf[1:] ** 2 - rf[:-1] ** 2) * self.dphi * self.dy)[:, None, None] \
            * np.ones((1, nphi, ny))
        self.L, self.c = self._assemble()

    def idx(self, i, j, k):
        return (np.asarray(i) * self.nphi + np.mod(j, self.nphi)) * self.ny + np.asarray(k)

    def _assemble(self):
        """Net conductive heat into each cell = L @ T + c (c: Robin terms with T_ref)."""
        from scipy.sparse import coo_matrix

        cell, nr, nphi, ny = self.cell, self.nr, self.nphi, self.ny
        dr, dphi, dy = self.dr, self.dphi, self.dy
        rows, cols, vals = [], [], []
        c = np.zeros(nr * nphi * ny)
        I, J, K = np.meshgrid(np.arange(nr), np.arange(nphi), np.arange(ny), indexing="ij")

        def k_tensor(phi):
            cs, sn = np.cos(phi), np.sin(phi)
            krr = cell.kx * cs ** 2 + cell.kz * sn ** 2
            kpp = cell.kx * sn ** 2 + cell.kz * cs ** 2
            krp = (cell.kz - cell.kx) * sn * cs
            return krr, kpp, krp

        # r-faces between i and i+1 (outward flux = -(krr dT/dr + krp/r dT/dphi) * area)
        i, j, k = I[:-1].ravel(), J[:-1].ravel(), K[:-1].ravel()
        phi = (j + 0.5) * dphi
        rface = (i + 1) * dr
        krr, _, krp = k_tensor(phi)
        area = rface * dphi * dy
        a, b = self.idx(i, j, k), self.idx(i + 1, j, k)
        g = krr * area / dr
        x = krp * area / (rface * 4 * dphi)  # cross term: (T(j+1) - T(j-1)) averaged over i, i+1
        terms = [(a, g), (b, -g),
                 (self.idx(i, j + 1, k), -x), (self.idx(i + 1, j + 1, k), -x),
                 (self.idx(i, j - 1, k), x), (self.idx(i + 1, j - 1, k), x)]
        self._add_vectorized(rows, cols, vals, a, b, terms)

        # phi-faces between j and j+1 at r_i
        i, j, k = I.ravel(), J.ravel(), K.ravel()
        phi = (j + 1.0) * dphi
        r = (i + 0.5) * dr
        _, kpp, krp = k_tensor(phi)
        area = dr * dy
        a, b = self.idx(i, j, k), self.idx(i, j + 1, k)
        g = kpp * area / (r * dphi)
        ip, im = np.minimum(i + 1, nr - 1), np.maximum(i - 1, 0)
        span = (ip - im) * dr
        x = krp * area / (2 * span)  # cross term: dT/dr averaged over j, j+1
        terms = [(a, g), (b, -g),
                 (self.idx(ip, j, k), -x), (self.idx(ip, j + 1, k), -x),
                 (self.idx(im, j, k), x), (self.idx(im, j + 1, k), x)]
        self._add_vectorized(rows, cols, vals, a, b, terms)

        # y-faces between k and k+1
        i, j, k = I[:, :, :-1].ravel(), J[:, :, :-1].ravel(), K[:, :, :-1].ravel()
        area = 0.5 * (((i + 1) * dr) ** 2 - (i * dr) ** 2) * dphi
        a, b = self.idx(i, j, k), self.idx(i, j, k + 1)
        g = cell.ky * area / dy
        self._add_vectorized(rows, cols, vals, a, b, [(a, g), (b, -g)])

        # Robin on the side (i = nr - 1): out = (T - T_ref) / (1/h + (dr/2)/krr) * area
        j, k = J[-1].ravel(), K[-1].ravel()
        phi = (j + 0.5) * dphi
        krr, _, _ = k_tensor(phi)
        h = cell.h_at(phi, (k + 0.5) * dy)
        area = cell.radius_m * dphi * dy
        u = area / (1 / h + 0.5 * dr / krr)
        a = self.idx(nr - 1, j, k)
        rows.extend(a.tolist())
        cols.extend(a.tolist())
        vals.extend((-u).tolist())
        np.add.at(c, a, u * cell.t_ref_k)
        self._robin = (a, u)

        n = nr * nphi * ny
        L = coo_matrix((np.concatenate([np.atleast_1d(v) for v in vals]),
                        (np.concatenate([np.atleast_1d(r) for r in rows]),
                         np.concatenate([np.atleast_1d(cc) for cc in cols]))), shape=(n, n)).tocsc()
        return L, c

    @staticmethod
    def _add_vectorized(rows, cols, vals, src, dst, terms):
        """Flux from src to dst (W) = sum(w * T[m]): src loses it, dst gains it."""
        for m, w in terms:
            w = np.broadcast_to(w, src.shape)
            rows.append(src)
            cols.append(m)
            vals.append(-w)
            rows.append(dst)
            cols.append(m)
            vals.append(w)

    def heat_out_w(self, T: np.ndarray) -> float:
        a, u = self._robin
        return float(np.sum(u * (T[a] - self.cell.t_ref_k)))

    def steady(self, q_cell_w: float) -> np.ndarray:
        from scipy.sparse.linalg import spsolve

        src = q_cell_w / self.cell.volume_m3 * self.vol.ravel()
        return spsolve(-self.L, self.c + src).reshape(self.nr, self.nphi, self.ny)

    def transient(self, load: Load, t0_k: float, dt_s: float = 1.0,
                  callback: Callable[[float, np.ndarray], None] | None = None) -> dict[str, np.ndarray]:
        """Backward Euler from a uniform T0; returns time series of T_max, T_mean, T at the centre."""
        from scipy.sparse import diags
        from scipy.sparse.linalg import splu

        cell = self.cell
        m = cell.rho * cell.cp * self.vol.ravel()
        lu = splu((diags(m / dt_s) - self.L).tocsc())
        T = np.full(m.shape, t0_k)
        n_steps = int(np.ceil(load.duration_s / dt_s))
        times = np.arange(n_steps + 1) * dt_s
        series = {"time_s": times, "t_max_k": [T.max()], "t_mean_k": [np.average(T, weights=m)],
                  "q_cell_w": [float(load.heat_w(0.0))]}
        for n in range(1, n_steps + 1):
            q = float(load.heat_w(times[n]))
            src = q / cell.volume_m3 * self.vol.ravel()
            T = lu.solve(m / dt_s * T + self.c + src)
            series["t_max_k"].append(T.max())
            series["t_mean_k"].append(np.average(T, weights=m))
            series["q_cell_w"].append(q)
            if callback:
                callback(times[n], T.reshape(self.nr, self.nphi, self.ny))
        self.last_field = T.reshape(self.nr, self.nphi, self.ny)
        return {k: np.asarray(v) for k, v in series.items()}

    def points_xyz(self) -> np.ndarray:
        """Cell centres in the cell's local Cartesian frame (x, y, z) relative to the axis, m."""
        R, P, Y = np.meshgrid(self.r, self.phi, self.y, indexing="ij")
        return np.stack([R * np.cos(P), Y, R * np.sin(P)], axis=-1)


def validate_steady(cell: CellModel, **grid) -> dict[str, float]:
    """Steady FV solution with the CFD heat vs the CFD itself (checks the h map export)."""
    fv = FVSolver(cell, **grid)
    T = fv.steady(cell.q_cell_cfd_w)
    m = cell.rho * cell.cp * fv.vol
    return {"cell": cell.name, "q_cell_w": cell.q_cell_cfd_w, "heat_out_w": fv.heat_out_w(T.ravel()),
            "t_max_fv_c": float(T.max() - KELVIN), "t_max_cfd_c": float(cell.t_max_cfd_k - KELVIN),
            "t_mean_fv_c": float(np.average(T, weights=m) - KELVIN),
            "t_mean_cfd_c": float(cell.t_mean_cfd_k - KELVIN)}
