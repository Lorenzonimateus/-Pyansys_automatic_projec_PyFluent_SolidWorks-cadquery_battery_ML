"""Figures and tables of reports/final_report.md, built from the study outputs.

Run from the project root after the study, `pinn` (one load and --parametric, for the
worst and the recommended points) and `geometry-report`:

    python reports/make_report_figures.py

Writes reports/figures/*.png and reports/figures/numbers.json.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from autocfd.cell_thermal import KELVIN, load_cells  # noqa: E402
from autocfd.config import load_config  # noqa: E402
from autocfd.plot_style import SERIES, contour, plt_agg, style_lines  # noqa: E402

OLD, NEW = "doe_crate", "doe_crate_kfix"
WORST, BEST, EXTRA = "dp021", "dp030", "dp010"   # highest T max, closest to L1 = P1 = 2, first PINN case
FIG = ROOT / "reports" / "figures"


def study(name: str) -> Path:
    return ROOT / "output" / "studies" / name


def results(name: str) -> pd.DataFrame:
    df = pd.read_csv(study(name) / "results.csv")
    df["converged"] = df["converged"].astype(str) == "True"
    return df


def fig_fixes(old: pd.DataFrame, new: pd.DataFrame) -> dict:
    """T max of every point before and after the fixes (k axis, cell heat, iterations)."""
    m = old.merge(new, on="id", suffixes=("_old", "_new"))
    m = m[m.converged_old & m.converged_new]
    plt = plt_agg()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.5), facecolor="white")
    lim = [m[["T_max_C_old", "T_max_C_new"]].min().min() - 5, m[["T_max_C_old", "T_max_C_new"]].max().max() + 5]
    a1.plot(lim, lim, color="#c3c2b7", lw=1)
    a1.scatter(m["T_max_C_old"], m["T_max_C_new"], s=40, color=SERIES[0], edgecolors="white", linewidths=1)
    style_lines(a1, "T max before the fixes [C]", "T max after the fixes [C]",
                "Cell T max of the 40 points, before vs after")
    rise_ratio = (m["T_max_C_new"] - 20) / (m["T_max_C_old"] - 20)
    a2.scatter(m["operating.c_rate_new"], (rise_ratio - 1) * 100, s=40, color=SERIES[1], edgecolors="white",
               linewidths=1)
    a2.axhline(0, color="#c3c2b7", lw=1)
    style_lines(a2, "C-rate", "change of the temperature rise [%]",
                "Temperature rise (T max - T in): change after the fixes")
    fig.tight_layout()
    fig.savefig(FIG / "fixes_t_max.png", dpi=125, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return {"n_compared": int(len(m)), "rise_change_pct_median": float((rise_ratio.median() - 1) * 100),
            "rise_change_pct_min": float((rise_ratio.min() - 1) * 100),
            "rise_change_pct_max": float((rise_ratio.max() - 1) * 100),
            "t_max_change_K_max": float((m["T_max_C_new"] - m["T_max_C_old"]).max())}


def hottest(cells):
    return max(cells, key=lambda c: c.t_max_cfd_k)


def fig_h_maps(cells_w, cells_b) -> None:
    """Unwrapped h map of the hottest cell of each point, one colour scale."""
    plt = plt_agg()
    maps = []
    for cells in (cells_w, cells_b):
        c = hottest(cells)
        phi = np.linspace(0, 2 * np.pi, 145)
        y = np.linspace(0, c.height_m, 71)
        P, Y = np.meshgrid(phi, y)
        maps.append((c, np.degrees(P), Y * 1000, c.h_at(P.ravel(), Y.ravel()).reshape(P.shape)))
    lo = min(m[3].min() for m in maps)
    hi = max(m[3].max() for m in maps)
    fig, axes = plt.subplots(2, 1, figsize=(11, 9), facecolor="white")
    for ax, (c, X, Y, H), name in zip(axes, maps, (WORST, BEST)):
        contour(fig, ax, X, Y, H, "Heat Transfer Coef. [W/m2K]",
                f"{name}, {c.name}: h from the CFD, mean {c.h_mean():.1f} W/m2K "
                f"(0 deg = wake side, 180 deg = facing the inlet)", lo, hi,
                xlabel="angle from +x [deg]", ylabel="y [mm]")
        ax.set_aspect("auto")
        ax.set_xticks(np.arange(0, 361, 45))
    fig.tight_layout()
    fig.savefig(FIG / "h_maps_worst_vs_best.png", dpi=125, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def fig_pinn(name_w: str, name_b: str) -> dict:
    """PINN of both points: T max during a 10C discharge and T max at the end vs C-rate."""
    plt = plt_agg()
    out = {}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5.6), facecolor="white")
    for (name, label), color in zip(((name_w, f"{WORST} (worst)"), (name_b, f"{BEST} (recommended)")), SERIES):
        d = study(NEW) / "pinn" / name
        comp = pd.read_csv(next(f for f in d.glob("*_pinn_vs_fv.csv") if "parametric" not in f.name))
        a1.plot(comp["time_s"], comp["t_max_fv_c"], color=color, lw=2, label=f"{label}, finite volumes")
        a1.plot(comp["time_s"], comp["t_max_pinn_c"], color=color, lw=2, ls="--", label=f"{label}, PINN")
        sweep = pd.read_csv(next(d.glob("*_parametric_sweep.csv")))
        pm = pd.read_csv(next(d.glob("*_parametric_metrics.csv")))
        a2.plot(sweep["c_rate"], sweep["t_max_end_pinn_c"], color=color, lw=2, label=f"{label}, parametric PINN")
        a2.plot(pm["c_rate"], pm["t_max_end_fv_c"], "o", ms=8, color=color, markeredgecolor="white",
                markeredgewidth=2, label=f"{label}, finite volumes")
        one = pd.read_csv(next(f for f in d.glob("*_metrics.csv") if "parametric" not in f.name))
        out[name] = {"one_load": one.iloc[0].to_dict(), "parametric": pm.to_dict("records")}
    style_lines(a1, "time [s]", "cell T max [C]", "10C discharge (360 s), hottest cell")
    a1.legend(frameon=False, fontsize=8)
    style_lines(a2, "C-rate", "cell T max at the end of the discharge [C]", "Full discharge at each C-rate")
    a2.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "pinn_worst_vs_best.png", dpi=125, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return out


def copy_figures() -> None:
    s = study(NEW)
    for point in (WORST, BEST):
        for f in ("temperature_xz", "velocity_xz", "temperature_xy"):
            src = s / "images" / point / f"{f}.png"
            if src.exists():
                shutil.copy2(src, FIG / f"cfd_{point}_{f}.png")
        for f in (s / "pinn" / point).glob("*_parametric_curves.png"):
            shutil.copy2(f, FIG / f"pinn_{point}_parametric_curves.png")
        for f in (s / "pinn" / point).glob("*_axial_section.png"):
            shutil.copy2(f, FIG / f"pinn_{point}_axial_section.png")
    for f in ("geometry_tradeoff.png", "geometry_maps.png"):
        if (s / "geometry" / f).exists():
            shutil.copy2(s / "geometry" / f, FIG / f)
    if (s / "sensitivity_sobol.png").exists():
        shutil.copy2(s / "sensitivity_sobol.png", FIG / "sensitivity_sobol.png")


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    cfg = load_config(ROOT / "configs" / "default.toml")
    old, new = results(OLD), results(NEW)
    numbers = {"fixes": fig_fixes(old, new)}
    ok = new[new.converged]
    numbers["study"] = {"points": int(len(new)), "ok": int((new.status == "ok").sum()),
                        "converged": int(new.converged.sum()),
                        "worst_id": int(ok.loc[ok["T_max_C"].idxmax(), "id"]),
                        "t_max_range_C": [float(ok["T_max_C"].min()), float(ok["T_max_C"].max())],
                        "dp_range_Pa": [float(ok["dP_Pa"].min()), float(ok["dP_Pa"].max())]}
    pts = {}
    for name in (WORST, BEST, EXTRA):
        r = new.set_index("id").loc[int(name[2:])]
        cells = load_cells(study(NEW) / "cells" / f"{name}.npz")
        c = hottest(cells)
        pts[name] = {"L1": float(r["cad.L1"]), "P1": float(r["cad.P1"]), "c_rate": float(r["operating.c_rate"]),
                     "velocity": float(r["fluent.parameter_velocity_inlet"]), "T_max_C": float(r["T_max_C"]),
                     "dT_cells_K": float(r["dT_cells_K"]), "dP_Pa": float(r["dP_Pa"]),
                     "Q_cell_W": float(r["Q_cell_W"]), "R_th": float((r["T_max_C"] - 20) / r["Q_cell_W"]),
                     "mass_flow_g_s": float(abs(r["m_out"]) * 1000), "hottest_cell": c.name,
                     "h_mean": float(c.h_mean()), "h_min": float(c.h.min()), "h_max": float(c.h.max()),
                     "cell_t_max_cfd_C": float(c.t_max_cfd_k - KELVIN)}
    numbers["points"] = pts
    fig_h_maps(load_cells(study(NEW) / "cells" / f"{WORST}.npz"), load_cells(study(NEW) / "cells" / f"{BEST}.npz"))
    numbers["pinn"] = fig_pinn(WORST, BEST)
    extra = study(NEW) / "pinn" / EXTRA
    if extra.exists():
        numbers["pinn"][EXTRA] = {
            "one_load": pd.read_csv(next(f for f in extra.glob("*_metrics.csv") if "parametric" not in f.name)).iloc[0].to_dict(),
            "parametric": pd.read_csv(next(extra.glob("*_parametric_metrics.csv"))).to_dict("records")}
    copy_figures()
    rec = study(NEW) / "geometry" / "recommended_geometry.csv"
    if rec.exists():
        numbers["recommended"] = pd.read_csv(rec).to_dict("records")
    numbers["config"] = {"study": cfg.study.name, "c_rate_report": cfg.geometry_report.c_rate}
    (FIG / "numbers.json").write_text(json.dumps(numbers, indent=2, default=float))
    print(json.dumps(numbers, indent=2, default=float))


if __name__ == "__main__":
    main()
