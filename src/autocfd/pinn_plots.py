"""Figures of the cell PINN.

Style: plot_style.py (Fluent-style maps, fixed-order line colours).

Every function takes a `predict(points_m, times_s) -> T [K] (len(times), n)` callable
in the cell's local frame (x, y, z) relative to the axis, y from the bottom.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .cell_thermal import KELVIN, CellModel, FVSolver
from .plot_style import INK, INK_2, SERIES, contour  # noqa: F401  (contour re-exported)
from .plot_style import plt_agg as _plt
from .plot_style import style_lines as _style_lines


def _disk_grid(cell: CellModel, n: int = 121):
    """Cartesian grid over the cell section (mm) and its points (m) at mid-height."""
    R = cell.radius_m
    g = np.linspace(-R, R, n)
    X, Z = np.meshgrid(g, g)
    inside = X ** 2 + Z ** 2 <= R ** 2
    pts = np.column_stack([X.ravel(), np.full(X.size, cell.height_m / 2), Z.ravel()])
    return X * 1000, Z * 1000, inside, pts


def comparison(comp, fv: FVSolver, T_pinn_end: np.ndarray, cell: CellModel, path: Path) -> None:
    """Time series (PINN vs finite volumes) + mid-height section at the end: PINN, FV, difference."""
    plt = _plt()
    fig = plt.figure(figsize=(17, 10.5), facecolor="white")
    gs = fig.add_gridspec(2, 3, height_ratios=[0.8, 1.4], hspace=0.22, wspace=0.12)
    ax = fig.add_subplot(gs[0, :])
    for (col, label), color in zip((("t_max", "T max"), ("t_mean", "T mean")), (SERIES[1], SERIES[0])):
        ax.plot(comp["time_s"], comp[f"{col}_fv_c"], color=color, lw=2, label=f"{label}, finite volumes")
        ax.plot(comp["time_s"], comp[f"{col}_pinn_c"], color=color, lw=2, ls="--", label=f"{label}, PINN")
    _style_lines(ax, "time [s]", "cell temperature [C]",
                 f"{cell.name}: transient temperature, PINN vs finite volumes")
    ax.legend(frameon=False, ncol=2, fontsize=9)

    k_mid = fv.ny // 2
    shape = (fv.nr, fv.nphi, fv.ny)
    sec = lambda a: a.reshape(shape)[:, :, k_mid] - KELVIN  # noqa: E731
    Rg, P = np.meshgrid(np.append(np.arange(fv.nr), fv.nr) * fv.dr * 1000,
                        np.append(fv.phi - fv.dphi / 2, 2 * np.pi), indexing="ij")
    X, Z = Rg * np.cos(P), Rg * np.sin(P)
    pinn_s, fv_s = sec(T_pinn_end), sec(fv.last_field.ravel())
    lo, hi = min(pinn_s.min(), fv_s.min()), max(pinn_s.max(), fv_s.max())
    t_end = comp["time_s"].iloc[-1]
    for n, (data, title, label, rng) in enumerate([
            (pinn_s, "PINN", "Static Temperature [C]", (lo, hi)),
            (fv_s, "finite volumes", "Static Temperature [C]", (lo, hi)),
            (pinn_s - fv_s, "PINN minus finite volumes", "Difference [K]", (None, None))]):
        a = fig.add_subplot(gs[1, n])
        contour(fig, a, X, Z, data, label, f"{title}, y = {fv.y[k_mid] * 1000:.1f} mm, t = {t_end:.0f} s", *rng)
    fig.savefig(path, dpi=130, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def sections_over_time(predict, cell: CellModel, t_end: float, title: str, path: Path,
                       fractions=(0.25, 0.5, 0.75, 1.0)) -> None:
    """Mid-height section of the PINN temperature at several times, one colour scale."""
    plt = _plt()
    X, Z, inside, pts = _disk_grid(cell)
    times = [f * t_end for f in fractions]
    T = predict(pts, times) - KELVIN
    maps = [np.where(inside, t.reshape(X.shape), np.nan) for t in T]
    lo = min(np.nanmin(m) for m in maps)
    hi = max(np.nanmax(m) for m in maps)
    fig, axes = plt.subplots(1, len(times), figsize=(4.8 * len(times), 4.6), facecolor="white")
    for a, m, t in zip(np.atleast_1d(axes), maps, times):
        contour(fig, a, X, Z, m, "Static Temperature [C]", f"PINN, t = {t:.0f} s", lo, hi)
    fig.suptitle(f"{title}: mid-height section (y = {cell.height_m * 500:.1f} mm) over time",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def axial_section(predict, fv: FVSolver, cell: CellModel, t_end: float, path: Path) -> None:
    """Section through the axis (x-y plane, z = 0) at the end: PINN, finite volumes, difference."""
    plt = _plt()
    R, H = cell.radius_m, cell.height_m
    xs, ys = np.linspace(-R, R, 81), np.linspace(0, H, 141)
    Xg, Yg = np.meshgrid(xs, ys)
    T_p = predict(np.column_stack([Xg.ravel(), Yg.ravel(), np.zeros(Xg.size)]), [t_end])[0]
    pinn_map = T_p.reshape(Xg.shape) - KELVIN

    # finite volumes: the two phi columns closest to +x (phi = 0) and -x (phi = pi)
    T = fv.last_field
    j_pos, j_neg = int(np.argmin(np.abs(np.angle(np.exp(1j * fv.phi))))), int(np.argmin(np.abs(fv.phi - np.pi)))
    x_fv = np.concatenate([-fv.r[::-1], fv.r])
    fv_map = np.concatenate([T[::-1, j_neg, :], T[:, j_pos, :]], axis=0).T - KELVIN   # (ny, 2 nr)
    Xf, Yf = np.meshgrid(x_fv, fv.y)
    T_at_fv = predict(np.column_stack([Xf.ravel(), Yf.ravel(), np.zeros(Xf.size)]), [t_end])[0]
    diff = T_at_fv.reshape(Xf.shape) - KELVIN - fv_map

    lo, hi = min(pinn_map.min(), fv_map.min()), max(pinn_map.max(), fv_map.max())
    fig, axes = plt.subplots(1, 3, figsize=(13, 7.5), facecolor="white")
    contour(fig, axes[0], Xg * 1000, Yg * 1000, pinn_map, "Static Temperature [C]",
            f"PINN, z = 0, t = {t_end:.0f} s", lo, hi, ylabel="y [mm]")
    contour(fig, axes[1], Xf * 1000, Yf * 1000, fv_map, "Static Temperature [C]",
            f"finite volumes, z = 0, t = {t_end:.0f} s", lo, hi, ylabel="y [mm]")
    contour(fig, axes[2], Xf * 1000, Yf * 1000, diff, "Difference [K]", "PINN minus finite volumes",
            ylabel="y [mm]")
    fig.suptitle(f"{cell.name}: section through the cell axis (flow along +x)", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def surface_map(predict, cell: CellModel, t_end: float, path: Path) -> None:
    """Unwrapped side surface: PINN temperature at the end, and the CFD h map that cools it."""
    plt = _plt()
    phi = np.linspace(0, 2 * np.pi, 145)
    y = np.linspace(0, cell.height_m, 71)
    P, Y = np.meshgrid(phi, y)
    pts = np.column_stack([cell.radius_m * np.cos(P.ravel()), Y.ravel(), cell.radius_m * np.sin(P.ravel())])
    Ts = predict(pts, [t_end])[0].reshape(P.shape) - KELVIN
    h = cell.h_at(P.ravel(), Y.ravel()).reshape(P.shape)
    fig, axes = plt.subplots(2, 1, figsize=(11, 9), facecolor="white")
    deg, ymm = np.degrees(P), Y * 1000
    contour(fig, axes[0], deg, ymm, Ts, "Static Temperature [C]",
            f"PINN surface temperature, t = {t_end:.0f} s", xlabel="angle from +x [deg]", ylabel="y [mm]")
    contour(fig, axes[1], deg, ymm, h, "Heat Transfer Coef. [W/m2K]",
            "h from the CFD (input of the PINN): 0 deg = wake side, 180 deg = facing the inlet",
            xlabel="angle from +x [deg]", ylabel="y [mm]")
    for a in axes:
        a.set_aspect("auto")
        a.set_xticks(np.arange(0, 361, 45))
    fig.suptitle(f"{cell.name}: unwrapped side surface", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def training(history: list[dict], title: str, path: Path) -> None:
    """Loss terms and adaptive weights over the iterations (two panels, log scale)."""
    import pandas as pd

    plt = _plt()
    h = pd.DataFrame(history)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True, facecolor="white")
    for col, label, color in (("pde", "equation residual", SERIES[0]), ("side", "cooling (Robin) on the side",
                                                                       SERIES[1]),
                              ("ends", "adiabatic ends", SERIES[2]), ("loss", "total (weighted)", INK)):
        a1.semilogy(h["iteration"], h[col], color=color, lw=2 if col != "loss" else 1.2, label=label)
    _style_lines(a1, "", "loss", f"PINN training: {title}")
    a1.legend(frameon=False, fontsize=9)
    for col, label, color in (("w_pde", "equation", SERIES[0]), ("w_side", "side", SERIES[1]),
                              ("w_ends", "ends", SERIES[2])):
        a2.semilogy(h["iteration"], h[col], color=color, lw=2, label=label)
    _style_lines(a2, "iteration", "weight", "Adaptive weights (gradient-norm balancing)")
    a2.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def parametric_curves(comps: dict, sweep_c: np.ndarray, sweep_t: np.ndarray, summary, cell: CellModel,
                      title: str, path: Path) -> None:
    """Left: T max over the discharge for each validated C-rate (PINN dashed, FV solid).
    Right: T max at the end of the discharge vs C-rate (PINN curve, FV points)."""
    plt = _plt()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5.8), facecolor="white",
                                 gridspec_kw={"width_ratios": [1.25, 1]})
    for (c, (comp, _, _)), color in zip(sorted(comps.items()), SERIES):
        dod = comp["time_s"] / comp["time_s"].iloc[-1] * 100
        a1.plot(dod, comp["t_max_fv_c"], color=color, lw=2, label=f"{c:g}C, finite volumes")
        a1.plot(dod, comp["t_max_pinn_c"], color=color, lw=2, ls="--", label=f"{c:g}C, PINN")
    _style_lines(a1, "depth of discharge [%]", "cell T max [C]", f"{title}: T max during the discharge")
    a1.legend(frameon=False, fontsize=8, ncol=2)

    a2.plot(sweep_c, sweep_t, color=SERIES[0], lw=2, label="PINN (one network, any C-rate)")
    a2.plot(summary["c_rate"], summary["t_max_end_fv_c"], "o", ms=8, color=SERIES[1],
            markeredgecolor="white", markeredgewidth=2, label="finite volumes")
    _style_lines(a2, "C-rate", "cell T max at the end of the discharge [C]", "End of discharge vs C-rate")
    a2.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def parametric_end_sections(predict_c, pinn, cell: CellModel, c_rates, path: Path) -> None:
    """Mid-height section at the end of the discharge for a few C-rates (each its own scale)."""
    plt = _plt()
    X, Z, inside, pts = _disk_grid(cell)
    fig, axes = plt.subplots(1, len(c_rates), figsize=(4.8 * len(c_rates), 4.6), facecolor="white")
    for a, c in zip(np.atleast_1d(axes), c_rates):
        t_end = float(pinn.t_end(c))
        T = predict_c(pts, [t_end], float(c))[0].reshape(X.shape) - KELVIN
        contour(fig, a, X, Z, np.where(inside, T, np.nan), "Static Temperature [C]",
                f"{c:g}C, end of discharge (t = {t_end:.0f} s)")
    fig.suptitle(f"{cell.name}: parametric PINN, mid-height section at the end of the discharge",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor="white", bbox_inches="tight")
    plt.close(fig)
