"""Physics-informed neural network (PINN) for the transient temperature of a cell.

Solves the reduced model of cell_thermal.py (orthotropic conduction + volumetric
Joule heat + Robin cooling with the CFD h map) with NO simulation data: the
network is trained only on the residuals of the equations.

Dimensionless form (xi = x/R, zeta = z/R, eta = 2y/H - 1, tau = t/t_end,
theta = (T - T0)/dT):
    theta_tau = Fo_x theta_xixi + Fo_y theta_etaeta + Fo_z theta_zetazeta + S(tau)
    Fo_i = k_i t_end / (rho cp L_i^2),   S = q(t) t_end / (rho cp dT)
    side (xi^2 + zeta^2 = 1):  kx theta_xi n_x + kz theta_zeta n_z + h R (theta + (T0 - Tref)/dT) = 0
    ends (eta = +-1):          theta_eta = 0
The initial condition is exact by construction: T = T0 + dT * tau * N(xi, eta, zeta, tau).

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
    seed: int = 0
    device: str = "cuda"
    log_every: int = 500


def _torch():
    import torch

    return torch


class _Net:
    """MLP (xi, eta, zeta, tau) -> N, built lazily so that importing this module
    does not require torch."""

    @staticmethod
    def build(width: int, depth: int):
        torch = _torch()
        layers, n_in = [], 4
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


class CellPINN:
    def __init__(self, cell: CellModel, load: Load, t0_k: float, settings: PINNSettings = PINNSettings()):
        torch = _torch()
        self.cell, self.load, self.t0_k, self.s = cell, load, t0_k, settings
        self.device = torch.device(settings.device if (settings.device != "cuda" or torch.cuda.is_available())
                                   else "cpu")
        torch.manual_seed(settings.seed)
        self.net = _Net.build(settings.width, settings.depth).to(self.device)
        c = cell
        self.t_end = load.duration_s
        self.dT = temperature_scale(cell, load)
        rc = c.rho * c.cp
        self.fo = (c.kx * self.t_end / (rc * c.radius_m ** 2),
                   c.ky * self.t_end * 4 / (rc * c.height_m ** 2),
                   c.kz * self.t_end / (rc * c.radius_m ** 2))
        s_max = load.peak_heat_w() / c.volume_m3 * self.t_end / (rc * self.dT)
        # theta_tau and S are O(1) by the choice of dT; do NOT divide by the Fourier
        # numbers (Fo_z = 25 with kz = 25 would make the equation ~600x weaker than the BCs)
        self.pde_scale = 1.0 / max(1.0, s_max)
        self.bc_scale = 1.0 / max(c.kx, c.kz, float(np.max(c.h)) * c.radius_m)
        self.offset = (t0_k - c.t_ref_k) / self.dT
        # h map and load as tensors for interpolation on the GPU
        self._phi_g = torch.tensor(c.phi, dtype=torch.float32, device=self.device)
        self._y_g = torch.tensor(c.y, dtype=torch.float32, device=self.device)
        self._h_g = torch.tensor(c.h, dtype=torch.float32, device=self.device)
        self._lt = torch.tensor(load.times_s / self.t_end, dtype=torch.float32, device=self.device)
        self._lq = torch.tensor(load.heat_w(load.times_s) / c.volume_m3 * self.t_end / (rc * self.dT),
                                dtype=torch.float32, device=self.device)
        self.history: list[dict] = []

    # -------------------------------------------------------------- helpers
    def theta(self, x):
        """x: (n, 4) = xi, eta, zeta, tau -> theta (n, 1); hard initial condition."""
        return x[:, 3:4] * self.net(x)

    def _source(self, tau):
        torch = _torch()
        lt, lq = self._lt, self._lq
        k1 = torch.clamp(torch.searchsorted(lt, tau.contiguous()), 1, len(lt) - 1)
        k0 = k1 - 1
        w = (tau - lt[k0]) / (lt[k1] - lt[k0])
        return (1 - w) * lq[k0] + w * lq[k1]

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

    def _sample(self):
        torch = _torch()
        s, dev = self.s, self.device
        rnd = lambda n: torch.rand(n, device=dev)  # noqa: E731
        # interior: uniform in the cylinder x time
        r, ph = torch.sqrt(rnd(s.n_interior)), 2 * math.pi * rnd(s.n_interior)
        xi = torch.stack([r * torch.cos(ph), 2 * rnd(s.n_interior) - 1, r * torch.sin(ph),
                          rnd(s.n_interior)], 1)
        # side
        ph_s = 2 * math.pi * rnd(s.n_side)
        eta_s = 2 * rnd(s.n_side) - 1
        xs = torch.stack([torch.cos(ph_s), eta_s, torch.sin(ph_s), rnd(s.n_side)], 1)
        h_s = self._h(ph_s, (eta_s + 1) / 2 * self.cell.height_m)
        # ends
        r_e, ph_e = torch.sqrt(rnd(s.n_ends)), 2 * math.pi * rnd(s.n_ends)
        eta_e = torch.where(rnd(s.n_ends) < 0.5, -torch.ones(s.n_ends, device=dev),
                            torch.ones(s.n_ends, device=dev))
        xe = torch.stack([r_e * torch.cos(ph_e), eta_e, r_e * torch.sin(ph_e), rnd(s.n_ends)], 1)
        return xi, (xs, ph_s, h_s), xe

    def losses(self):
        torch = _torch()
        xi, (xs, ph_s, h_s), xe = self._sample()
        grad = lambda f, x: torch.autograd.grad(f, x, torch.ones_like(f), create_graph=True)[0]  # noqa: E731

        xi.requires_grad_(True)
        th = self.theta(xi)
        g = grad(th, xi)
        lap = sum(f * grad(g[:, d:d + 1], xi)[:, d:d + 1] for d, f in zip((0, 1, 2), self.fo))
        res = g[:, 3:4] - lap - self._source(xi[:, 3]).unsqueeze(1)
        l_pde = torch.mean((self.pde_scale * res) ** 2)

        xs.requires_grad_(True)
        gs = grad(self.theta(xs), xs)
        c = self.cell
        flux = c.kx * gs[:, 0] * torch.cos(ph_s) + c.kz * gs[:, 2] * torch.sin(ph_s)
        robin = flux + h_s * c.radius_m * (self.theta(xs)[:, 0] + self.offset)
        l_side = torch.mean((self.bc_scale * robin) ** 2)

        xe.requires_grad_(True)
        ge = grad(self.theta(xe), xe)
        l_ends = torch.mean(ge[:, 1] ** 2)
        return l_pde, l_side, l_ends

    # -------------------------------------------------------------- train / predict
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
                w = s.weight_ema * w + (1 - s.weight_ema) * self._balanced_weights(terms, params) \
                    if it > 1 else self._balanced_weights(terms, params)
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

    def predict(self, points_m: np.ndarray, times_s: np.ndarray) -> np.ndarray:
        """Temperature (K) at local points (n, 3) = (x, y, z) relative to the cell axis
        (y from the bottom), for each time: returns (len(times), n)."""
        torch = _torch()
        c = self.cell
        p = np.asarray(points_m, float).reshape(-1, 3)
        base = np.stack([p[:, 0] / c.radius_m, 2 * p[:, 1] / c.height_m - 1, p[:, 2] / c.radius_m], 1)
        out = []
        with torch.no_grad():
            for t in np.atleast_1d(times_s):
                x = np.column_stack([base, np.full(len(base), t / self.t_end)])
                th = self.theta(torch.tensor(x, dtype=torch.float32, device=self.device))
                out.append(self.t0_k + self.dT * th.cpu().numpy()[:, 0])
        return np.array(out)

    def save(self, path: Path) -> None:
        torch = _torch()
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state": self.net.state_dict(), "settings": asdict(self.s), "dT": self.dT,
                    "t0_k": self.t0_k, "cell": self.cell.name}, path)


# ------------------------------------------------------------------ run + validation
def run_cell_pinn(cell: CellModel, load: Load, t0_k: float, out_dir: Path,
                  settings: PINNSettings = PINNSettings(), fv_grid: dict | None = None) -> dict:
    """Train the PINN, solve the same problem with finite volumes, compare and plot."""
    import pandas as pd

    out_dir.mkdir(parents=True, exist_ok=True)
    pinn = CellPINN(cell, load, t0_k, settings)
    log.info("PINN %s: t_end %.0f s, dT scale %.2f K, Fo = %s, device %s", cell.name, pinn.t_end,
             pinn.dT, tuple(round(f, 3) for f in pinn.fo), pinn.device)
    t0 = time.time()
    pinn.train()
    train_s = time.time() - t0
    pinn.save(out_dir / f"{cell.name}_pinn.pt")
    pd.DataFrame(pinn.history).to_csv(out_dir / f"{cell.name}_history.csv", index=False)

    fv = FVSolver(cell, **(fv_grid or {}))
    t1 = time.time()
    series = fv.transient(load, t0_k, dt_s=max(0.5, load.duration_s / 600))
    fv_s = time.time() - t1

    pts = fv.points_xyz().reshape(-1, 3)
    weights = (cell.rho * cell.cp * fv.vol).ravel()
    times = series["time_s"]
    T_pinn = pinn.predict(pts, times)
    comp = pd.DataFrame({
        "time_s": times, "c_rate": load.c_rate(times), "q_cell_w": series["q_cell_w"],
        "t_max_fv_c": series["t_max_k"] - KELVIN, "t_max_pinn_c": T_pinn.max(axis=1) - KELVIN,
        "t_mean_fv_c": series["t_mean_k"] - KELVIN,
        "t_mean_pinn_c": np.average(T_pinn, axis=1, weights=weights) - KELVIN})
    comp.to_csv(out_dir / f"{cell.name}_pinn_vs_fv.csv", index=False)

    field_fv = fv.last_field.ravel()
    field_err = T_pinn[-1] - field_fv
    rise = float(field_fv.max() - t0_k) or 1.0
    metrics = {
        "cell": cell.name, "duration_s": load.duration_s, "train_time_s": train_s, "fv_time_s": fv_s,
        "t_max_end_fv_c": float(comp["t_max_fv_c"].iloc[-1]),
        "t_max_end_pinn_c": float(comp["t_max_pinn_c"].iloc[-1]),
        "t_max_series_max_abs_err_k": float(np.max(np.abs(comp["t_max_pinn_c"] - comp["t_max_fv_c"]))),
        "t_mean_series_max_abs_err_k": float(np.max(np.abs(comp["t_mean_pinn_c"] - comp["t_mean_fv_c"]))),
        "field_end_max_abs_err_k": float(np.max(np.abs(field_err))),
        "field_end_rmse_k": float(np.sqrt(np.mean(field_err ** 2))),
        "field_end_rel_err_of_rise": float(np.max(np.abs(field_err)) / rise),
        "final_loss": pinn.history[-1]["loss"],
    }
    pd.DataFrame([metrics]).to_csv(out_dir / f"{cell.name}_metrics.csv", index=False)
    _plot(comp, fv, T_pinn[-1], field_fv, cell, out_dir / f"{cell.name}_pinn_vs_fv.png")
    log.info("PINN vs FV for %s: %s", cell.name, metrics)
    return metrics


def _plot(comp, fv: FVSolver, T_pinn_end: np.ndarray, T_fv_end: np.ndarray, cell: CellModel,
          path: Path) -> None:
    """Time series (PINN vs FV) + mid-height section at the end: PINN, FV, difference.
    Contours in the project's Fluent style (jet, colorbar on the left, 11 ticks, %.1f)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter

    fig = plt.figure(figsize=(15, 9.5), facecolor="white")
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.15], hspace=0.38, wspace=0.45)
    ax = fig.add_subplot(gs[0, :])
    blue, orange = "#2a78d6", "#eb6834"
    for col, label, color in (("t_max", "T max", orange), ("t_mean", "T mean", blue)):
        ax.plot(comp["time_s"], comp[f"{col}_fv_c"], color=color, lw=2, label=f"{label}, finite volumes")
        ax.plot(comp["time_s"], comp[f"{col}_pinn_c"], color=color, lw=2, ls="--", label=f"{label}, PINN")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("cell temperature [C]")
    ax.set_title(f"{cell.name}: transient temperature, PINN vs finite volumes", loc="left")
    ax.legend(frameon=False, ncol=2)
    ax.grid(False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    k_mid = fv.ny // 2
    shape = (fv.nr, fv.nphi, fv.ny)
    sec = lambda a: a.reshape(shape)[:, :, k_mid] - KELVIN  # noqa: E731
    R, P = np.meshgrid(np.append(np.arange(fv.nr), fv.nr) * fv.dr * 1000,
                       np.append(fv.phi - fv.dphi / 2, 2 * np.pi), indexing="ij")
    X, Z = R * np.cos(P), R * np.sin(P)
    pinn_s, fv_s = sec(T_pinn_end), sec(T_fv_end)
    lo, hi = min(pinn_s.min(), fv_s.min()), max(pinn_s.max(), fv_s.max())
    panels = [(pinn_s, "PINN", "Static Temperature [C]", (lo, hi)),
              (fv_s, "finite volumes", "Static Temperature [C]", (lo, hi)),
              (pinn_s - fv_s, "PINN minus finite volumes", "Difference [K]", None)]
    for n, (data, title, label, rng) in enumerate(panels):
        a = fig.add_subplot(gs[1, n])
        vmin, vmax = rng if rng else (float(data.min()), float(data.max()) + 1e-9)
        m = a.pcolormesh(X, Z, data, cmap="jet", vmin=vmin, vmax=vmax, shading="flat")
        a.set_aspect("equal")
        a.set_title(f"{title}, y = {fv.y[k_mid] * 1000:.1f} mm, t = {comp['time_s'].iloc[-1]:.0f} s",
                    fontsize=10, loc="left")
        a.set_xlabel("x [mm]")
        a.set_ylabel("z [mm]")
        cb = fig.colorbar(m, ax=a, location="left", pad=0.28, fraction=0.05)
        cb.set_ticks(np.linspace(vmin, vmax, 11))
        cb.ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        cb.ax.set_title(label, fontsize=8, loc="left")
    fig.savefig(path, dpi=130, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def run_from_config(cfg, study_root: Path) -> list[dict]:
    """[pinn] of the config: h map of point `pinn.point`, load from [operating]
    (capacity, R_DC) and [pinn] (C-rate or profile). Writes to <study>/pinn/<point>/."""
    import pandas as pd

    from .cell_thermal import load_cells, validate_steady

    p, op = cfg.pinn, cfg.operating
    cells_file = study_root / "cells" / f"{p.point}.npz"
    if not cells_file.exists():
        raise FileNotFoundError(f"{cells_file} not found: run `postprocess --only cells` (or a study) first")
    cells = load_cells(cells_file)
    out_dir = study_root / "pinn" / p.point

    # 1) the reduced model must reproduce the CFD at steady state (checks the h map export)
    steady = pd.DataFrame([validate_steady(c) for c in cells])
    out_dir.mkdir(parents=True, exist_ok=True)
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

    if p.profile_csv:
        load = Load.from_csv(p.profile_csv, op.capacity_ah, op.r_dc_ohm)
    else:
        load = Load.constant(p.c_rate, p.duration_s or 3600.0 / p.c_rate, op.capacity_ah, op.r_dc_ohm)
    settings = PINNSettings(iterations=p.iterations, width=p.width, depth=p.depth, device=p.device)
    t0 = p.initial_temperature_c + KELVIN
    results = [run_cell_pinn(c, load, t0, out_dir, settings) for c in chosen]
    pd.DataFrame(results).to_csv(out_dir / "summary.csv", index=False)
    return results
