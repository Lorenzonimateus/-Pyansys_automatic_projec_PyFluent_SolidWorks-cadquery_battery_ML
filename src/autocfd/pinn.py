"""Physics-informed neural networks (PINN) for the transient temperature of a cell.

Solves the reduced model of cell_thermal.py (orthotropic conduction + volumetric
Joule heat + Robin cooling with the CFD h map) with NO simulation data: the
network is trained only on the residuals of the equations.

Dimensionless form (xi = x/R, zeta = z/R, eta = 2y/H - 1, tau = t/t_end,
theta = (T - T0)/dT):
    theta_tau = Fo_x theta_xixi + Fo_y theta_etaeta + Fo_z theta_zetazeta + S
    Fo_i = k_i t_end / (rho cp L_i^2),   S = q t_end / (rho cp dT)
    side (xi^2 + zeta^2 = 1):  kx theta_xi n_x + kz theta_zeta n_z + h R (theta + (T0 - Tref)/dT) = 0
    ends (eta = +-1):          theta_eta = 0
The initial condition is exact by construction: T = T0 + dT * tau * N(...).

Two networks:
- CellPINN: one load (constant C-rate or a C(t) profile), inputs (xi, eta, zeta, tau);
- ParametricCellPINN: constant-current discharges for ANY C-rate in a range, inputs
  (xi, eta, zeta, tau, c). Each C-rate has its own duration (3600/C for a full
  discharge, or a fixed one) and its own temperature scale, so every case is O(1)
  for the network; one training answers every C-rate in the range.

Validation: against the finite-volume solution of the same model (FVSolver) and,
at steady state, against the CFD (validate_steady).
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .cell_thermal import KELVIN, CellModel, FVSolver, Load, temperature_scale

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PINNSettings:
    width: int = 128
    depth: int = 6
    iterations: int = 20000
    lr: float = 1e-3
    lr_final: float = 1e-5
    n_interior: int = 8192
    n_side: int = 4096
    n_ends: int = 2048
    w_pde: float = 1.0          # initial weights; adapted during training when adaptive_weights
    w_side: float = 1.0
    w_ends: float = 1.0
    adaptive_weights: bool = True
    weight_update_every: int = 100
    weight_ema: float = 0.9
    # cap of each weight relative to the equation's: a term that is already satisfied has a
    # tiny gradient, and without a cap its weight explodes (seen: 5e7 for the adiabatic ends)
    # and the network stops fitting the equation
    max_weight_ratio: float = 1e3
    # parametric PINN: train on a range wider than the one used (fraction of the half-range on
    # each side), so that the ends of c_rate_range are inside the trained domain (seen: 5.6 K
    # error at 10C, the upper end, when training exactly on [1, 10])
    c_margin: float = 0.1
    seed: int = 0
    device: str = "cuda"
    log_every: int = 500


def _torch():
    import torch

    return torch


def _mlp(n_in: int, width: int, depth: int):
    torch = _torch()
    layers = []
    for _ in range(depth):
        lin = torch.nn.Linear(n_in, width)
        torch.nn.init.xavier_normal_(lin.weight)
        torch.nn.init.zeros_(lin.bias)
        layers += [lin, torch.nn.Tanh()]
        n_in = width
    out = torch.nn.Linear(n_in, 1)
    torch.nn.init.xavier_normal_(out.weight)
    torch.nn.init.zeros_(out.bias)
    return torch.nn.Sequential(*layers, out)


# ------------------------------------------------------------------ shared machinery
class _CellPINNBase:
    """Training loop, sampling, h-map interpolation and losses; subclasses define the
    extra inputs and the (possibly per-point) coefficients."""

    n_extra = 0

    def __init__(self, cell: CellModel, t0_k: float, settings: PINNSettings):
        torch = _torch()
        self.cell, self.t0_k, self.s = cell, t0_k, settings
        self.device = torch.device(settings.device if (settings.device != "cuda" or torch.cuda.is_available())
                                   else "cpu")
        torch.manual_seed(settings.seed)
        self.net = _mlp(4 + self.n_extra, settings.width, settings.depth).to(self.device)
        self.bc_scale = 1.0 / max(cell.kx, cell.kz, float(np.max(cell.h)) * cell.radius_m)
        self._phi_g = torch.tensor(cell.phi, dtype=torch.float32, device=self.device)
        self._y_g = torch.tensor(cell.y, dtype=torch.float32, device=self.device)
        self._h_g = torch.tensor(cell.h, dtype=torch.float32, device=self.device)
        self.history: list[dict] = []

    # subclasses ------------------------------------------------------------
    def _extra(self, n: int):
        """Extra input columns (n, n_extra) for n sampled points, or None."""
        return None

    def coeffs(self, x) -> dict:
        """Per-point (or scalar) coefficients: fo (3), S, dT, offset."""
        raise NotImplementedError

    # common ----------------------------------------------------------------
    def theta(self, x):
        """x: (n, 4 + n_extra) = xi, eta, zeta, tau, ... -> theta (n, 1); hard initial condition."""
        return x[:, 3:4] * self.net(x)

    def _h(self, phi, y):
        torch = _torch()
        phi_g, y_g, h = self._phi_g, self._y_g, self._h_g
        nphi = len(phi_g)
        dphi = 2 * math.pi / nphi
        fj = (torch.remainder(phi, 2 * math.pi) - phi_g[0]) / dphi
        j0 = torch.floor(fj)
        wj = fj - j0
        j0 = torch.remainder(j0.long(), nphi)
        j1 = torch.remainder(j0 + 1, nphi)
        y = torch.clamp(y, y_g[0], y_g[-1])
        k1 = torch.clamp(torch.searchsorted(y_g, y.contiguous()), 1, len(y_g) - 1)
        k0 = k1 - 1
        wk = (y - y_g[k0]) / (y_g[k1] - y_g[k0])
        return ((1 - wj) * ((1 - wk) * h[j0, k0] + wk * h[j0, k1])
                + wj * ((1 - wk) * h[j1, k0] + wk * h[j1, k1]))

    def _with_extra(self, cols):
        torch = _torch()
        x = torch.stack(cols, 1)
        extra = self._extra(len(x))
        return x if extra is None else torch.cat([x, extra], 1)

    def _sample(self):
        torch = _torch()
        s, dev = self.s, self.device
        rnd = lambda n: torch.rand(n, device=dev)  # noqa: E731
        r, ph = torch.sqrt(rnd(s.n_interior)), 2 * math.pi * rnd(s.n_interior)
        xi = self._with_extra([r * torch.cos(ph), 2 * rnd(s.n_interior) - 1, r * torch.sin(ph),
                               rnd(s.n_interior)])
        ph_s = 2 * math.pi * rnd(s.n_side)
        eta_s = 2 * rnd(s.n_side) - 1
        xs = self._with_extra([torch.cos(ph_s), eta_s, torch.sin(ph_s), rnd(s.n_side)])
        h_s = self._h(ph_s, (eta_s + 1) / 2 * self.cell.height_m)
        r_e, ph_e = torch.sqrt(rnd(s.n_ends)), 2 * math.pi * rnd(s.n_ends)
        eta_e = torch.where(rnd(s.n_ends) < 0.5, -torch.ones(s.n_ends, device=dev),
                            torch.ones(s.n_ends, device=dev))
        xe = self._with_extra([r_e * torch.cos(ph_e), eta_e, r_e * torch.sin(ph_e), rnd(s.n_ends)])
        return xi, (xs, ph_s, h_s), xe

    def losses(self):
        torch = _torch()
        xi, (xs, ph_s, h_s), xe = self._sample()
        grad = lambda f, x: torch.autograd.grad(f, x, torch.ones_like(f), create_graph=True)[0]  # noqa: E731

        xi.requires_grad_(True)
        g = grad(self.theta(xi), xi)
        k = self.coeffs(xi.detach())
        lap = sum(f * grad(g[:, d:d + 1], xi)[:, d:d + 1] for d, f in zip((0, 1, 2), k["fo"]))
        res = (g[:, 3:4] - lap - k["S"]) / torch.clamp(torch.as_tensor(k["S"]), min=1.0)
        l_pde = torch.mean(res ** 2)

        xs.requires_grad_(True)
        th_s = self.theta(xs)
        gs = grad(th_s, xs)
        ks = self.coeffs(xs.detach())
        c = self.cell
        flux = c.kx * gs[:, 0:1] * torch.cos(ph_s)[:, None] + c.kz * gs[:, 2:3] * torch.sin(ph_s)[:, None]
        robin = flux + h_s[:, None] * c.radius_m * (th_s + ks["offset"])
        l_side = torch.mean((self.bc_scale * robin) ** 2)

        xe.requires_grad_(True)
        ge = grad(self.theta(xe), xe)
        l_ends = torch.mean(ge[:, 1] ** 2)
        return l_pde, l_side, l_ends

    def train(self) -> list[dict]:
        torch = _torch()
        s = self.s
        opt = torch.optim.Adam(self.net.parameters(), lr=s.lr)
        gamma = (s.lr_final / s.lr) ** (1 / max(1, s.iterations))
        sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma)
        t_start = time.time()
        w = torch.tensor([s.w_pde, s.w_side, s.w_ends], device=self.device)
        params = [p for p in self.net.parameters() if p.requires_grad]
        for it in range(1, s.iterations + 1):
            terms = self.losses()
            if s.adaptive_weights and (it == 1 or it % s.weight_update_every == 0):
                new = self._balanced_weights(terms, params)
                new = torch.clamp(new, max=s.max_weight_ratio * new[0])
                w = new if it == 1 else s.weight_ema * w + (1 - s.weight_ema) * new
            loss = sum(wi * li for wi, li in zip(w, terms))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            if it % s.log_every == 0 or it == 1 or it == s.iterations:
                l_pde, l_side, l_ends = (t.item() for t in terms)
                row = {"iteration": it, "loss": loss.item(), "pde": l_pde, "side": l_side, "ends": l_ends,
                       "w_pde": w[0].item(), "w_side": w[1].item(), "w_ends": w[2].item(),
                       "lr": sched.get_last_lr()[0], "time_s": time.time() - t_start}
                self.history.append(row)
                log.info("PINN it %d: loss %.3e (pde %.2e, side %.2e, ends %.2e; weights %.2g/%.2g/%.2g)",
                         it, row["loss"], l_pde, l_side, l_ends, *w.tolist())
        return self.history

    @staticmethod
    def _balanced_weights(terms, params):
        """Gradient-norm balancing (Wang, Teng & Perdikaris 2021): w_i = sum_j |g_j| / |g_i|,
        so every term pulls on the network with the same strength."""
        torch = _torch()
        norms = []
        for t in terms:
            g = torch.autograd.grad(t, params, retain_graph=True, allow_unused=True)
            norms.append(torch.sqrt(sum((gi ** 2).sum() for gi in g if gi is not None)) + 1e-12)
        norms = torch.stack(norms).detach()
        return norms.sum() / norms

    def _local_inputs(self, points_m: np.ndarray) -> np.ndarray:
        c = self.cell
        p = np.asarray(points_m, float).reshape(-1, 3)
        return np.stack([p[:, 0] / c.radius_m, 2 * p[:, 1] / c.height_m - 1, p[:, 2] / c.radius_m], 1)

    def _eval(self, x: np.ndarray, dT) -> np.ndarray:
        torch = _torch()
        with torch.no_grad():
            th = self.theta(torch.tensor(x, dtype=torch.float32, device=self.device)).cpu().numpy()[:, 0]
        return self.t0_k + dT * th

    def save(self, path: Path, **extra) -> None:
        torch = _torch()
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state": self.net.state_dict(), "settings": asdict(self.s), "t0_k": self.t0_k,
                    "cell": self.cell.name, **extra}, path)


# ------------------------------------------------------------------ one load
class CellPINN(_CellPINNBase):
    """One load: constant C-rate or a C(t) profile."""

    def __init__(self, cell: CellModel, load: Load, t0_k: float, settings: PINNSettings = PINNSettings()):
        super().__init__(cell, t0_k, settings)
        torch = _torch()
        self.load = load
        self.t_end = load.duration_s
        self.dT = temperature_scale(cell, load)
        rc = cell.rho * cell.cp
        self.fo = (cell.kx * self.t_end / (rc * cell.radius_m ** 2),
                   cell.ky * self.t_end * 4 / (rc * cell.height_m ** 2),
                   cell.kz * self.t_end / (rc * cell.radius_m ** 2))
        self.offset = (t0_k - cell.t_ref_k) / self.dT
        self._lt = torch.tensor(load.times_s / self.t_end, dtype=torch.float32, device=self.device)
        self._lq = torch.tensor(load.heat_w(load.times_s) / cell.volume_m3 * self.t_end / (rc * self.dT),
                                dtype=torch.float32, device=self.device)

    def coeffs(self, x) -> dict:
        torch = _torch()
        tau = x[:, 3]
        lt, lq = self._lt, self._lq
        k1 = torch.clamp(torch.searchsorted(lt, tau.contiguous()), 1, len(lt) - 1)
        k0 = k1 - 1
        w = (tau - lt[k0]) / (lt[k1] - lt[k0])
        S = ((1 - w) * lq[k0] + w * lq[k1])[:, None]
        return {"fo": self.fo, "S": S, "offset": self.offset}

    def predict(self, points_m: np.ndarray, times_s) -> np.ndarray:
        """Temperature (K) at local points (n, 3) = (x, y, z) relative to the cell axis
        (y from the bottom), for each time: (len(times), n)."""
        base = self._local_inputs(points_m)
        return np.array([self._eval(np.column_stack([base, np.full(len(base), t / self.t_end)]), self.dT)
                         for t in np.atleast_1d(times_s)])

    def save(self, path: Path, **extra) -> None:
        super().save(path, dT=self.dT, **extra)


# ------------------------------------------------------------------ parametric in C-rate
class ParametricCellPINN(_CellPINNBase):
    """Constant-current discharges for any C-rate in [c_min, c_max]: inputs
    (xi, eta, zeta, tau, c_hat), c_hat = 2 (C - c_min)/(c_max - c_min) - 1.
    Duration: duration_s if > 0, else a full discharge 3600 / C."""

    n_extra = 1

    def __init__(self, cell: CellModel, c_range: tuple[float, float], capacity_ah: float, r_dc_ohm: float,
                 t0_k: float, duration_s: float = 0.0, settings: PINNSettings = PINNSettings()):
        super().__init__(cell, t0_k, settings)
        self.c_min, self.c_max = map(float, c_range)
        self.capacity_ah, self.r_dc_ohm, self.duration_s = capacity_ah, r_dc_ohm, duration_s

    # physical helpers (numpy or torch, element-wise)
    def t_end(self, c):
        return self.duration_s + 0 * c if self.duration_s > 0 else 3600.0 / c

    def load(self, c: float) -> Load:
        return Load.constant(c, float(self.t_end(c)), self.capacity_ah, self.r_dc_ohm)

    def temperature_scale(self, c):
        """Same rule as cell_thermal.temperature_scale, element-wise in c."""
        cell = self.cell
        q = (c * self.capacity_ah) ** 2 * self.r_dc_ohm
        adiabatic = q * self.t_end(c) / (cell.rho * cell.cp * cell.volume_m3)
        k_r = 0.5 * (cell.kx + cell.kz)
        steady = q / (cell.h_mean() * cell.side_area_m2) + q / (4 * np.pi * k_r * cell.height_m)
        lib = _torch() if not isinstance(c, (float, np.ndarray, np.floating)) else np
        return lib.clip(lib.minimum(adiabatic, steady), 1.0, None) if lib is np else \
            lib.clamp(lib.minimum(adiabatic, steady), min=1.0)

    def c_from_hat(self, c_hat):
        return self.c_min + (c_hat + 1) / 2 * (self.c_max - self.c_min)

    def c_hat(self, c):
        return 2 * (c - self.c_min) / (self.c_max - self.c_min) - 1

    def _extra(self, n: int):
        torch = _torch()
        m = self.s.c_margin
        return (2 + 2 * m) * torch.rand(n, 1, device=self.device) - 1 - m   # c_hat in [-1-m, 1+m]

    def coeffs(self, x) -> dict:
        cell = self.cell
        c = self.c_from_hat(x[:, 4:5])
        t_end = self.t_end(c)
        dT = self.temperature_scale(c)
        rc = cell.rho * cell.cp
        q = (c * self.capacity_ah) ** 2 * self.r_dc_ohm / cell.volume_m3
        fo = (cell.kx * t_end / (rc * cell.radius_m ** 2),
              cell.ky * t_end * 4 / (rc * cell.height_m ** 2),
              cell.kz * t_end / (rc * cell.radius_m ** 2))
        return {"fo": fo, "S": q * t_end / (rc * dT), "offset": (self.t0_k - cell.t_ref_k) / dT}

    def predict(self, points_m: np.ndarray, times_s, c_rate: float) -> np.ndarray:
        """Temperature (K) at local points for each time of a discharge at c_rate."""
        base = self._local_inputs(points_m)
        t_end, dT = float(self.t_end(c_rate)), float(self.temperature_scale(float(c_rate)))
        ch = np.full(len(base), self.c_hat(c_rate))
        return np.array([self._eval(np.column_stack([base, np.full(len(base), t / t_end), ch]), dT)
                         for t in np.atleast_1d(times_s)])

    def predict_t_max_end(self, c_rates, points_m: np.ndarray) -> np.ndarray:
        """T max (K) at the end of the discharge for many C-rates (fast: one call each)."""
        return np.array([self.predict(points_m, [float(self.t_end(c))], c).max() for c in c_rates])

    def save(self, path: Path, **extra) -> None:
        super().save(path, c_range=(self.c_min, self.c_max), capacity_ah=self.capacity_ah,
                     r_dc_ohm=self.r_dc_ohm, duration_s=self.duration_s, **extra)


# ------------------------------------------------------------------ run + validation
def _fv_series(cell: CellModel, load: Load, t0_k: float, fv_grid: dict | None):
    fv = FVSolver(cell, **(fv_grid or {}))
    t1 = time.time()
    series = fv.transient(load, t0_k, dt_s=max(0.5, load.duration_s / 600))
    return fv, series, time.time() - t1


def _compare(fv: FVSolver, series: dict, T_pinn: np.ndarray, load: Load, t0_k: float) -> tuple:
    import pandas as pd

    weights = (fv.cell.rho * fv.cell.cp * fv.vol).ravel()
    times = series["time_s"]
    comp = pd.DataFrame({
        "time_s": times, "c_rate": load.c_rate(times), "q_cell_w": series["q_cell_w"],
        "t_max_fv_c": series["t_max_k"] - KELVIN, "t_max_pinn_c": T_pinn.max(axis=1) - KELVIN,
        "t_mean_fv_c": series["t_mean_k"] - KELVIN,
        "t_mean_pinn_c": np.average(T_pinn, axis=1, weights=weights) - KELVIN})
    field_err = T_pinn[-1] - fv.last_field.ravel()
    rise = float(fv.last_field.max() - t0_k) or 1.0
    metrics = {
        "duration_s": load.duration_s,
        "t_max_end_fv_c": float(comp["t_max_fv_c"].iloc[-1]),
        "t_max_end_pinn_c": float(comp["t_max_pinn_c"].iloc[-1]),
        "t_max_series_max_abs_err_k": float(np.max(np.abs(comp["t_max_pinn_c"] - comp["t_max_fv_c"]))),
        "t_mean_series_max_abs_err_k": float(np.max(np.abs(comp["t_mean_pinn_c"] - comp["t_mean_fv_c"]))),
        "field_end_max_abs_err_k": float(np.max(np.abs(field_err))),
        "field_end_rmse_k": float(np.sqrt(np.mean(field_err ** 2))),
        "field_end_rel_err_of_rise": float(np.max(np.abs(field_err)) / rise),
    }
    return comp, metrics


def run_cell_pinn(cell: CellModel, load: Load, t0_k: float, out_dir: Path,
                  settings: PINNSettings = PINNSettings(), fv_grid: dict | None = None) -> dict:
    """Train the PINN for one load, solve the same problem with finite volumes,
    compare and write the figures (pinn_plots.py)."""
    import pandas as pd

    from . import pinn_plots

    out_dir.mkdir(parents=True, exist_ok=True)
    pinn = CellPINN(cell, load, t0_k, settings)
    log.info("PINN %s: t_end %.0f s, dT scale %.2f K, Fo = %s, device %s", cell.name, pinn.t_end,
             pinn.dT, tuple(round(float(f), 3) for f in pinn.fo), pinn.device)
    t0 = time.time()
    pinn.train()
    train_s = time.time() - t0
    pinn.save(out_dir / f"{cell.name}_pinn.pt")
    pd.DataFrame(pinn.history).to_csv(out_dir / f"{cell.name}_history.csv", index=False)

    fv, series, fv_s = _fv_series(cell, load, t0_k, fv_grid)
    T_pinn = pinn.predict(fv.points_xyz().reshape(-1, 3), series["time_s"])
    comp, metrics = _compare(fv, series, T_pinn, load, t0_k)
    comp.to_csv(out_dir / f"{cell.name}_pinn_vs_fv.csv", index=False)
    metrics = {"cell": cell.name, **metrics, "train_time_s": train_s, "fv_time_s": fv_s,
               "final_loss": pinn.history[-1]["loss"]}
    pd.DataFrame([metrics]).to_csv(out_dir / f"{cell.name}_metrics.csv", index=False)

    predict = pinn.predict
    pinn_plots.comparison(comp, fv, T_pinn[-1], cell, out_dir / f"{cell.name}_pinn_vs_fv.png")
    pinn_plots.sections_over_time(predict, cell, load.duration_s, f"{cell.name}, {_load_label(load)}",
                                  out_dir / f"{cell.name}_sections_time.png")
    pinn_plots.axial_section(predict, fv, cell, load.duration_s, out_dir / f"{cell.name}_axial_section.png")
    pinn_plots.surface_map(predict, cell, load.duration_s, out_dir / f"{cell.name}_surface_and_h.png")
    pinn_plots.training(pinn.history, f"{cell.name}, {_load_label(load)}",
                        out_dir / f"{cell.name}_training.png")
    log.info("PINN vs FV for %s: %s", cell.name, metrics)
    return metrics


def _load_label(load: Load) -> str:
    if np.allclose(load.c_rates, load.c_rates[0]):
        return f"{load.c_rates[0]:g}C for {load.duration_s:.0f} s"
    return f"C-rate profile, {load.duration_s:.0f} s"


def run_parametric(cell: CellModel, c_range: tuple[float, float], capacity_ah: float, r_dc_ohm: float,
                   t0_k: float, out_dir: Path, validate_c_rates, duration_s: float = 0.0,
                   settings: PINNSettings = PINNSettings(), fv_grid: dict | None = None) -> list[dict]:
    """Train one PINN for every C-rate in c_range; validate against finite volumes at
    validate_c_rates; write curves T(t) per C-rate and T max at the end vs C-rate."""
    import pandas as pd

    from . import pinn_plots

    out_dir.mkdir(parents=True, exist_ok=True)
    pinn = ParametricCellPINN(cell, c_range, capacity_ah, r_dc_ohm, t0_k, duration_s, settings)
    log.info("Parametric PINN %s: C-rate %g..%g, %s, device %s", cell.name, *c_range,
             f"{duration_s:.0f} s" if duration_s > 0 else "full discharge 3600/C", pinn.device)
    t0 = time.time()
    pinn.train()
    train_s = time.time() - t0
    pinn.save(out_dir / f"{cell.name}_parametric_pinn.pt")
    pd.DataFrame(pinn.history).to_csv(out_dir / f"{cell.name}_parametric_history.csv", index=False)

    rows, comps = [], {}
    fv_pts = None
    for c in validate_c_rates:
        load = pinn.load(float(c))
        fv, series, fv_s = _fv_series(cell, load, t0_k, fv_grid)
        fv_pts = fv.points_xyz().reshape(-1, 3)
        t1 = time.time()
        T_pinn = pinn.predict(fv_pts, series["time_s"], float(c))
        pinn_s = time.time() - t1
        comp, metrics = _compare(fv, series, T_pinn, load, t0_k)
        comp.to_csv(out_dir / f"{cell.name}_parametric_{c:g}C_vs_fv.csv", index=False)
        comps[float(c)] = (comp, fv, T_pinn[-1])
        rows.append({"cell": cell.name, "c_rate": float(c), **metrics, "fv_time_s": fv_s,
                     "pinn_eval_time_s": pinn_s, "train_time_s": train_s})
        log.info("Parametric PINN at %gC vs FV: %s", c, metrics)
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / f"{cell.name}_parametric_metrics.csv", index=False)

    sweep_c = np.linspace(pinn.c_min, pinn.c_max, 46)
    t1 = time.time()
    sweep = pinn.predict_t_max_end(sweep_c, fv_pts) - KELVIN
    log.info("PINN sweep of %d C-rates in %.2f s", len(sweep_c), time.time() - t1)
    pd.DataFrame({"c_rate": sweep_c, "t_max_end_pinn_c": sweep}).to_csv(
        out_dir / f"{cell.name}_parametric_sweep.csv", index=False)

    title = f"{cell.name}, C-rate {pinn.c_min:g} to {pinn.c_max:g}"
    pinn_plots.parametric_curves(comps, sweep_c, sweep, summary, cell, title,
                                 out_dir / f"{cell.name}_parametric_curves.png")
    pinn_plots.parametric_end_sections(pinn.predict, pinn, cell,
                                       [pinn.c_min, 0.5 * (pinn.c_min + pinn.c_max), pinn.c_max],
                                       out_dir / f"{cell.name}_parametric_end_sections.png")
    pinn_plots.training(pinn.history, f"{title}, parametric", out_dir / f"{cell.name}_parametric_training.png")
    return rows


def run_from_config(cfg, study_root: Path, parametric: bool | None = None) -> list[dict]:
    """[pinn] of the config: h map of point `pinn.point`, load from [operating]
    (capacity, R_DC) and [pinn]. Writes to <study>/pinn/<point>/."""
    import pandas as pd

    from .cell_thermal import load_cells, validate_steady

    p, op = cfg.pinn, cfg.operating
    parametric = p.parametric if parametric is None else parametric
    cells_file = study_root / "cells" / f"{p.point}.npz"
    if not cells_file.exists():
        raise FileNotFoundError(f"{cells_file} not found: run `postprocess --only cells` (or a study) first")
    cells = load_cells(cells_file)
    out_dir = study_root / "pinn" / p.point
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) the reduced model must reproduce the CFD at steady state (checks the h map export)
    steady = pd.DataFrame([validate_steady(c) for c in cells])
    steady.to_csv(out_dir / "steady_vs_cfd.csv", index=False)
    log.info("Steady reduced model vs CFD:\n%s", steady.to_string(index=False))

    if p.cell == "hottest":
        chosen = [max(cells, key=lambda c: c.t_max_cfd_k)]
    elif p.cell == "all":
        chosen = cells
    else:
        chosen = [c for c in cells if c.name == p.cell]
        if not chosen:
            raise KeyError(f"[pinn] cell {p.cell!r} not in {[c.name for c in cells]}")

    t0 = p.initial_temperature_c + KELVIN
    if parametric:
        settings = PINNSettings(iterations=p.parametric_iterations, width=p.width, depth=p.depth, device=p.device)
        results = []
        for c in chosen:
            results += run_parametric(c, tuple(p.c_rate_range), op.capacity_ah, op.r_dc_ohm, t0, out_dir,
                                      p.validate_c_rates, p.duration_s, settings)
        pd.DataFrame(results).to_csv(out_dir / "parametric_summary.csv", index=False)
        return results

    if p.profile_csv:
        load = Load.from_csv(p.profile_csv, op.capacity_ah, op.r_dc_ohm)
    else:
        load = Load.constant(p.c_rate, p.duration_s or 3600.0 / p.c_rate, op.capacity_ah, op.r_dc_ohm)
    settings = PINNSettings(iterations=p.iterations, width=p.width, depth=p.depth, device=p.device)
    results = [run_cell_pinn(c, load, t0, out_dir, settings) for c in chosen]
    pd.DataFrame(results).to_csv(out_dir / "summary.csv", index=False)
    return results


def load_cell_pinn(path: Path, cell: CellModel, load: Load, device: str = "cpu") -> CellPINN:
    """Rebuild a trained one-load PINN from its .pt (e.g. to redo the figures)."""
    torch = _torch()
    ckpt = torch.load(path, map_location=device, weights_only=False)
    settings = PINNSettings(**{**ckpt["settings"], "device": device})
    pinn = CellPINN(cell, load, ckpt["t0_k"], settings)
    pinn.net.load_state_dict(ckpt["state"])
    pinn.net.eval()
    return pinn


def replot_cell_pinn(cfg, study_root: Path) -> None:
    """Redo the figures of the one-load PINN of [pinn] from the saved model (no training)."""
    import pandas as pd

    from . import pinn_plots
    from .cell_thermal import load_cells

    p, op = cfg.pinn, cfg.operating
    out_dir = study_root / "pinn" / p.point
    cells = {c.name: c for c in load_cells(study_root / "cells" / f"{p.point}.npz")}
    load = (Load.from_csv(p.profile_csv, op.capacity_ah, op.r_dc_ohm) if p.profile_csv else
            Load.constant(p.c_rate, p.duration_s or 3600.0 / p.c_rate, op.capacity_ah, op.r_dc_ohm))
    for pt in sorted(out_dir.glob("*_pinn.pt")):
        if pt.name.endswith("_parametric_pinn.pt"):
            continue
        cell = cells[pt.name.replace("_pinn.pt", "")]
        pinn = load_cell_pinn(pt, cell, load)
        fv, series, _ = _fv_series(cell, load, pinn.t0_k, None)
        T_pinn = pinn.predict(fv.points_xyz().reshape(-1, 3), series["time_s"])
        comp, _ = _compare(fv, series, T_pinn, load, pinn.t0_k)
        hist = pd.read_csv(out_dir / f"{cell.name}_history.csv").to_dict("records")
        predict = pinn.predict  # bound to this iteration's network
        pinn_plots.comparison(comp, fv, T_pinn[-1], cell, out_dir / f"{cell.name}_pinn_vs_fv.png")
        pinn_plots.sections_over_time(predict, cell, load.duration_s, f"{cell.name}, {_load_label(load)}",
                                      out_dir / f"{cell.name}_sections_time.png")
        pinn_plots.axial_section(predict, fv, cell, load.duration_s, out_dir / f"{cell.name}_axial_section.png")
        pinn_plots.surface_map(predict, cell, load.duration_s, out_dir / f"{cell.name}_surface_and_h.png")
        pinn_plots.training(hist, f"{cell.name}, {_load_label(load)}", out_dir / f"{cell.name}_training.png")
        log.info("Figures of %s redone in %s", cell.name, out_dir)


def load_parametric_pinn(path: Path, cell: CellModel, device: str = "cpu") -> ParametricCellPINN:
    """Rebuild a trained parametric PINN from its .pt."""
    torch = _torch()
    ckpt = torch.load(path, map_location=device, weights_only=False)
    settings = PINNSettings(**{**ckpt["settings"], "device": device})
    pinn = ParametricCellPINN(cell, ckpt["c_range"], ckpt["capacity_ah"], ckpt["r_dc_ohm"], ckpt["t0_k"],
                              ckpt["duration_s"], settings)
    pinn.net.load_state_dict(ckpt["state"])
    pinn.net.eval()
    return pinn


def replot_parametric(cfg, study_root: Path) -> None:
    """Redo the maps and the training figure of the saved parametric PINNs (no training)."""
    import pandas as pd

    from . import pinn_plots
    from .cell_thermal import load_cells

    out_dir = study_root / "pinn" / cfg.pinn.point
    cells = {c.name: c for c in load_cells(study_root / "cells" / f"{cfg.pinn.point}.npz")}
    for pt in sorted(out_dir.glob("*_parametric_pinn.pt")):
        cell = cells[pt.name.replace("_parametric_pinn.pt", "")]
        pinn = load_parametric_pinn(pt, cell)
        title = f"{cell.name}, C-rate {pinn.c_min:g} to {pinn.c_max:g}"
        pinn_plots.parametric_end_sections(pinn.predict, pinn, cell,
                                           [pinn.c_min, 0.5 * (pinn.c_min + pinn.c_max), pinn.c_max],
                                           out_dir / f"{cell.name}_parametric_end_sections.png")
        hist = pd.read_csv(out_dir / f"{cell.name}_parametric_history.csv").to_dict("records")
        pinn_plots.training(hist, f"{title}, parametric", out_dir / f"{cell.name}_parametric_training.png")
        log.info("Parametric figures of %s redone in %s", cell.name, out_dir)
