"""Which geometry is best? A fair comparison of the study's geometric parameters.

The C-rate dominates the cell temperature, so it hides the geometry. With constant
properties the problem is linear in the heat, so the thermal resistance

    R_th = (T max - T inlet) / Q_cell    [K/W]

depends only on the geometry and the air velocity. Two laws are fitted on the
converged CFD points (least squares in log form, standard errors, leave-one-out):

    log R_th = a0 + b log v + sum_i c_i g_i          log dP = d0 + e log v + sum_i f_i g_i

They give the effect of each parameter (and whether it is significant). The comparison
itself uses Gaussian processes on the same inputs (log v, g): a cell has an internal
conduction resistance in series with the convection, so R_th need not be monotonic
in a gap at the same fan power, which the log-linear law cannot represent. The inlet
area A(g) = m_dot / (rho v) is fitted linearly in the geometric parameters (a gap may
also widen the duct, so at the same velocity it gets more air). The geometries are
then compared in three ways, all relative to a reference point:

  same inlet velocity   v = v_ref                                 (how the DOE was run)
  same mass flow        v = v_ref A(g_ref) / A(g)
  same fan power        v such that v A(g) dP(v, g) = fan power of the reference

Outputs in <study>/geometry/: figures, tables and report.md (a short report).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .plot_style import SERIES, contour, plt_agg, style_lines

log = logging.getLogger(__name__)

CRITERIA = {"same_velocity": "same inlet velocity", "same_mass_flow": "same mass flow",
            "same_fan_power": "same fan power"}


# ------------------------------------------------------------------ fitting
@dataclass
class LogLaw:
    """log y = c0 + c1 log v + sum c_i g_i, fitted by least squares."""
    names: list[str]          # ["log v", geometric params...]
    coef: np.ndarray          # (1 + len(names),) with the intercept first
    stderr: np.ndarray
    resid_std: float
    loo_r2: float

    def __call__(self, v, G) -> np.ndarray:
        G = np.atleast_2d(G)
        return np.exp(self.coef[0] + self.coef[1] * np.log(v) + G @ self.coef[2:])


def _design(v, G):
    return np.column_stack([np.ones(len(v)), np.log(v), G])


def fit_log_law(v: np.ndarray, G: np.ndarray, y: np.ndarray, geo_names: list[str]) -> LogLaw:
    A = _design(v, G)
    ly = np.log(y)
    coef, *_ = np.linalg.lstsq(A, ly, rcond=None)
    res = ly - A @ coef
    dof = max(1, len(ly) - A.shape[1])
    cov = res @ res / dof * np.linalg.pinv(A.T @ A)
    h = np.einsum("ij,ji->i", A, np.linalg.pinv(A))           # hat matrix diagonal: exact LOO
    loo = res / np.clip(1 - h, 1e-9, None)
    loo_r2 = 1 - float(np.sum(loo ** 2) / np.sum((ly - ly.mean()) ** 2))
    return LogLaw(["log v", *geo_names], coef, np.sqrt(np.diag(cov)), float(res.std(ddof=A.shape[1])), loo_r2)


def _fit_gp(X: np.ndarray, y: np.ndarray):
    import warnings

    from sklearn.exceptions import ConvergenceWarning

    from .ml import make_model

    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    return make_model(X.shape[1]).fit(X, y)


def gp_loo_r2(X: np.ndarray, y: np.ndarray) -> float:
    """Leave-one-out R2 of a Gaussian process on the same inputs (checks the law's form)."""
    import warnings

    from sklearn.exceptions import ConvergenceWarning
    from sklearn.model_selection import LeaveOneOut, cross_val_predict

    from .ml import make_model

    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    pred = cross_val_predict(make_model(X.shape[1]), X, y, cv=LeaveOneOut())
    return 1 - float(np.sum((pred - y) ** 2) / np.sum((y - y.mean()) ** 2))


# ------------------------------------------------------------------ analysis
@dataclass
class Analysis:
    cfg: Config
    df: pd.DataFrame
    geo: list[str]
    velocity: str
    r_law: LogLaw
    dp_law: LogLaw
    area_coef: np.ndarray        # A = a0 + sum a_i g_i  (m2)
    area_resid: float
    r_gp_loo: float
    dp_gp_loo: float
    r_gp: object                 # Gaussian process: (log v, g) -> log R_th
    dp_gp: object                # Gaussian process: (log v, g) -> log dP
    ref: dict[str, float]
    q_cell_w: float              # heat per cell at the report's C-rate
    fan_ref_w: float = 0.0
    m_ref: float = 0.0

    def area(self, G) -> np.ndarray:
        return self.area_coef[0] + np.atleast_2d(G) @ self.area_coef[1:]

    def _x(self, v, G) -> np.ndarray:
        return np.column_stack([np.log(v), np.atleast_2d(G)])

    def log_rth(self, v, G):
        """Mean and standard deviation of log R_th (Gaussian process)."""
        return self.r_gp.predict(self._x(v, G), return_std=True)

    def dp(self, v, G) -> np.ndarray:
        return np.exp(self.dp_gp.predict(self._x(v, G)))

    def velocity_for(self, criterion: str, G) -> np.ndarray:
        G = np.atleast_2d(G)
        v_ref = self.ref[self.velocity]
        g_ref = np.array([[self.ref[g] for g in self.geo]])
        if criterion == "same_velocity":
            return np.full(len(G), v_ref)
        if criterion == "same_mass_flow":
            return v_ref * self.area(g_ref)[0] / self.area(G)
        # same fan power: solve v A(g) dP(v, g) = fan_ref on the study's velocity range
        lo, hi = self.cfg.study.parameters[self.velocity]
        vg = np.linspace(lo, hi, 241)
        Gr = np.repeat(G, len(vg), axis=0)
        fan = (np.tile(vg, len(G)) * self.area(Gr) * self.dp(np.tile(vg, len(G)), Gr)).reshape(len(G), len(vg))
        out = np.full(len(G), np.nan)                     # NaN: needs a velocity outside the study
        for i, f in enumerate(fan):
            if f[0] <= self.fan_ref_w <= f[-1] and np.all(np.diff(f) > 0):
                out[i] = np.interp(self.fan_ref_w, f, vg)
        return out

    def evaluate(self, criterion: str, G) -> pd.DataFrame:
        G = np.atleast_2d(G)
        v = self.velocity_for(criterion, G)
        ok = np.isfinite(v)
        m, s = np.full(len(G), np.nan), np.full(len(G), np.nan)
        dp = np.full(len(G), np.nan)
        if ok.any():
            m[ok], s[ok] = self.log_rth(v[ok], G[ok])
            dp[ok] = self.dp(v[ok], G[ok])
        lo, hi = self.cfg.study.parameters[self.velocity]
        rth = np.exp(m)
        out = pd.DataFrame(G, columns=self.geo)
        out["velocity"] = v
        out["velocity_in_study_range"] = ok & (v >= lo - 1e-9) & (v <= hi + 1e-9)
        out["R_th_K_per_W"] = rth
        out["T_max_C"] = self.cfg.geometry_report.inlet_temperature_c + rth * self.q_cell_w
        out["T_max_std_K"] = rth * self.q_cell_w * s       # delta method on log R_th
        out["dP_Pa"] = dp
        out["mass_flow_g_s"] = self.cfg.geometry_report.air_density * v * self.area(G) * 1000
        out["fan_power_W"] = v * self.area(G) * dp
        return out


def load_points(cfg: Config) -> pd.DataFrame:
    from .doe import study_dir
    from .sensitivity import load_valid_points

    gr = cfg.geometry_report
    df = load_valid_points(study_dir(cfg) / "results.csv")
    missing = [c for c in (gr.velocity, gr.temperature, gr.pressure, gr.heat, gr.mass_flow) if c not in df]
    if missing:
        raise KeyError(f"Columns missing in results.csv for the geometry report: {missing}")
    df = df.copy()
    df["R_th"] = (df[gr.temperature] - gr.inlet_temperature_c) / df[gr.heat]
    return df


def analyse(cfg: Config) -> Analysis:
    gr = cfg.geometry_report
    geo = [p for p in cfg.study.parameters if p.startswith("cad.")]
    if not geo:
        raise ValueError("The study has no geometric (cad.*) parameter to compare")
    if gr.velocity not in cfg.study.parameters:
        raise ValueError(f"[geometry_report] velocity {gr.velocity!r} is not a study parameter")
    df = load_points(cfg)
    v, G = df[gr.velocity].to_numpy(float), df[geo].to_numpy(float)
    r_law = fit_log_law(v, G, df["R_th"].to_numpy(float), geo)
    dp_law = fit_log_law(v, G, df[gr.pressure].to_numpy(float), geo)
    area = df[gr.mass_flow].abs().to_numpy(float) / (gr.air_density * v)
    Aa = np.column_stack([np.ones(len(G)), G])
    area_coef, *_ = np.linalg.lstsq(Aa, area, rcond=None)
    area_resid = float(np.max(np.abs(Aa @ area_coef - area)))
    X = np.column_stack([np.log(v), G])
    y_r, y_p = np.log(df["R_th"].to_numpy(float)), np.log(df[gr.pressure].to_numpy(float))
    r_gp_loo, dp_gp_loo = gp_loo_r2(X, y_r), gp_loo_r2(X, y_p)
    r_gp, dp_gp = _fit_gp(X, y_r), _fit_gp(X, y_p)

    ref = {p: float(np.mean(cfg.study.parameters[p])) for p in [*geo, gr.velocity]}
    ref.update({k: float(x) for k, x in gr.reference.items()})
    q = cfg.with_overrides({"operating.c_rate": gr.c_rate}).heat_inputs().get("Q_cell_W")
    if q is None:
        raise ValueError("[operating] heat_source must be set: the report needs the heat per cell")
    g_ref = np.array([[ref[g] for g in geo]])
    v_ref = ref[gr.velocity]
    a = Analysis(cfg, df, geo, gr.velocity, r_law, dp_law, area_coef, area_resid, r_gp_loo, dp_gp_loo,
                 r_gp, dp_gp, ref, q)
    a.fan_ref_w = float(v_ref * a.area(g_ref)[0] * a.dp(np.array([v_ref]), g_ref)[0])
    a.m_ref = float(gr.air_density * v_ref * a.area(g_ref)[0])
    return a


# ------------------------------------------------------------------ figures
def _label(cfg: Config, p: str) -> str:
    return cfg.geometry_report.labels.get(p, p)


def plot_tradeoff(a: Analysis, path: Path) -> None:
    """Per geometric parameter (others at the reference): R_th for the three criteria,
    CFD points brought to the reference, and dP for the three criteria."""
    plt = plt_agg()
    n = len(a.geo)
    fig, axes = plt.subplots(n, 3, figsize=(19, 5.4 * n), facecolor="white", squeeze=False)
    v_ref = a.ref[a.velocity]
    for row, g in enumerate(a.geo):
        lo, hi = a.cfg.study.parameters[g]
        x = np.linspace(lo, hi, 81)
        G = np.tile([a.ref[k] for k in a.geo], (len(x), 1))
        G[:, a.geo.index(g)] = x
        ax1, ax2, ax3 = axes[row]
        for (crit, label), color in zip(CRITERIA.items(), SERIES):
            e = a.evaluate(crit, G)
            ax1.plot(x, e["R_th_K_per_W"], color=color, lw=2, label=label)
            ax3.semilogy(x, e["dP_Pa"], color=color, lw=2, label=label)
        others = ", ".join(f"{k.split('.', 1)[1]} = {a.ref[k]:g}" for k in a.geo if k != g)
        style_lines(ax1, _label(a.cfg, g), "R_th = (T max - T in) / Q_cell [K/W]",
                    f"Cooling vs {g.split('.', 1)[1]}" + (f" ({others})" if others else ""))
        ax1.legend(frameon=False, fontsize=9)
        # CFD points brought to v_ref and to the reference of the other parameters
        c = a.r_law.coef
        r = a.df["R_th"] * (v_ref / a.df[a.velocity]) ** c[1]
        for k in a.geo:
            if k != g:
                r = r * np.exp(-c[2 + a.geo.index(k)] * (a.df[k] - a.ref[k]))
        colour_by = next((k for k in a.geo if k != g), a.velocity)
        sc = ax2.scatter(a.df[g], r, c=a.df[colour_by], cmap="jet", s=40, edgecolors="white", linewidths=1)
        e = a.evaluate("same_velocity", G)
        ax2.plot(x, e["R_th_K_per_W"], color=SERIES[0], lw=2, label=f"Gaussian process at {v_ref:g} m/s")
        cb = fig.colorbar(sc, ax=ax2, location="left", pad=0.2)
        cb.ax.set_title(colour_by.split(".", 1)[1], fontsize=8, loc="left")
        style_lines(ax2, _label(a.cfg, g), f"R_th brought to {v_ref:g} m/s [K/W]",
                    f"CFD points (colour: {colour_by.split('.', 1)[1]})")
        ax2.legend(frameon=False, fontsize=9)
        style_lines(ax3, _label(a.cfg, g), "pressure drop [Pa]", f"Pressure drop vs {g.split('.', 1)[1]}")
        ax3.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=125, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def plot_maps(a: Analysis, path: Path) -> None:
    """Two geometric parameters: T max at the report's C-rate for the three criteria (one
    colour scale, so a flat map looks flat) and dP at the same velocity."""
    plt = plt_agg()
    gx, gy = a.geo
    X, Y = np.meshgrid(np.linspace(*a.cfg.study.parameters[gx], 41), np.linspace(*a.cfg.study.parameters[gy], 41))
    G = np.column_stack([X.ravel(), Y.ravel()])
    evals = {c: a.evaluate(c, G) for c in CRITERIA}
    T = {c: e["T_max_C"].to_numpy().reshape(X.shape) for c, e in evals.items()}
    lo = min(np.nanmin(t) for t in T.values())
    hi = max(np.nanmax(t) for t in T.values())
    fig, axes = plt.subplots(1, 4, figsize=(25, 6.4), facecolor="white")
    for ax, (crit, label) in zip(axes[:3], CRITERIA.items()):
        contour(fig, ax, X, Y, T[crit], "Cell T max [C]", f"T max at {a.cfg.geometry_report.c_rate:g}C, {label}",
                lo, hi, xlabel=_label(a.cfg, gx), ylabel=_label(a.cfg, gy))
        ax.scatter(a.df[gx], a.df[gy], s=12, c="black", label="CFD points")
        i = int(np.nanargmin(T[crit]))
        ax.plot(X.ravel()[i], Y.ravel()[i], marker="*", ms=18, color="white", markeredgecolor="black",
                label="lowest T max")
        ax.legend(frameon=True, fontsize=7, loc="upper right")
    dp = evals["same_velocity"]["dP_Pa"].to_numpy().reshape(X.shape)
    contour(fig, axes[3], X, Y, dp, "Pressure drop [Pa]", "Pressure drop, same inlet velocity",
            xlabel=_label(a.cfg, gx), ylabel=_label(a.cfg, gy))
    axes[3].scatter(a.df[gx], a.df[gy], s=12, c="black")
    for ax in axes:
        ax.set_aspect("auto")
    fig.tight_layout()
    fig.savefig(path, dpi=120, facecolor="white", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ recommendation
def array_size(cfg: Config, a: Analysis) -> dict | None:
    """Size of the cell array (bounding box of the cells, mm) as a linear function of the
    geometric parameters, measured from cells/<point>.npz; None when they do not exist."""
    from .cell_thermal import load_cells
    from .doe import study_dir

    rows = []
    for _, r in a.df.iterrows():
        f = study_dir(cfg) / "cells" / f"dp{int(r['id']):03d}.npz"
        if not f.exists():
            continue
        cells = load_cells(f)
        c = np.array([cc.center_m for cc in cells]) * 1000
        rad = cells[0].radius_m * 1000
        rows.append([*(r[g] for g in a.geo), np.ptp(c[:, 0]) + 2 * rad, np.ptp(c[:, 1]) + 2 * rad,
                     cells[0].height_m * 1000])
    if len(rows) < len(a.geo) + 2:
        return None
    d = np.array(rows)
    A = np.column_stack([np.ones(len(d)), d[:, :len(a.geo)]])
    fit = {}
    for k, col in (("length_mm", -3), ("width_mm", -2)):
        coef, *_ = np.linalg.lstsq(A, d[:, col], rcond=None)
        fit[k] = coef
    fit["height_mm"] = float(np.mean(d[:, -1]))
    return fit


def array_volume(size: dict, G) -> pd.DataFrame:
    G = np.atleast_2d(G)
    A = np.column_stack([np.ones(len(G)), G])
    out = pd.DataFrame({"length_mm": A @ size["length_mm"], "width_mm": A @ size["width_mm"]})
    out["height_mm"] = size["height_mm"]
    out["volume_cm3"] = out.length_mm * out.width_mm * out.height_mm / 1000
    return out


def recommend(cfg: Config, a: Analysis, effects: pd.DataFrame) -> pd.DataFrame:
    """One value per geometric parameter. Judged at the same fan power (the fair comparison):
    no significant effect on R_th, or an effect below tie_tolerance_k over the range ->
    indifferent_choice (by default the smallest value: most compact module); otherwise the
    value with the lowest T max."""
    gr = cfg.geometry_report
    sig = effects[effects.output == "R_th"].set_index("parameter")["significant"]
    rows = []
    for g in a.geo:
        lo, hi = cfg.study.parameters[g]
        rule = {"min": lo, "max": hi, "reference": a.ref[g]}[gr.indifferent_choice]
        rule_txt = {"min": "menor valor: módulo mais compacto", "max": "maior valor",
                    "reference": "valor de referência"}[gr.indifferent_choice]
        x = np.linspace(lo, hi, 81)
        G = np.tile([a.ref[k] for k in a.geo], (len(x), 1))
        G[:, a.geo.index(g)] = x
        e = a.evaluate("same_fan_power", G)
        T, S = e["T_max_C"].to_numpy(), e["T_max_std_K"].to_numpy()
        valid = np.isfinite(T)
        if not bool(sig[g]) or not valid.any():
            rows.append({"parameter": g, "value": rule, "spread_same_fan_power_K": np.nan,
                         "reason": f"efeito na refrigeração não significativo (±2σ) → {rule_txt}"})
            continue
        i_best = int(np.nanargmin(T))
        # tied with the best: closer than the tolerance or than 2 sigma of the difference
        margin = np.maximum(gr.tie_tolerance_k, 2 * np.sqrt(S ** 2 + S[i_best] ** 2))
        tied = valid & (T - T[i_best] <= margin)
        xt = x[tied]
        value = {"min": xt.min(), "max": xt.max(),
                 "reference": xt[np.argmin(np.abs(xt - a.ref[g]))]}[gr.indifferent_choice]
        spread = float(np.nanmax(T) - np.nanmin(T))
        j = int(np.argmin(np.abs(x - value)))
        if xt.min() == x[valid].min() and xt.max() == x[valid].max():
            why = (f"na mesma potência de ventilador todos os valores empatam (diferença máx. {spread:.1f} K, "
                   f"dentro da incerteza) → {rule_txt}")
        else:
            why = (f"na mesma potência de ventilador: menor T max em {x[i_best]:.1f} ({T[i_best]:.1f} °C); "
                   f"empatam com ele (dentro de {gr.tie_tolerance_k:g} K ou 2σ) os valores de {xt.min():.1f} a "
                   f"{xt.max():.1f} → {rule_txt} desse intervalo ({T[j]:.1f} ± {2 * S[j]:.1f} °C); "
                   f"fora dele até {spread:.1f} K mais quente")
        rows.append({"parameter": g, "value": float(value), "reason": why, "spread_same_fan_power_K": spread,
                     "tied_from": float(xt.min()), "tied_to": float(xt.max()), "best_at": float(x[i_best])})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ report
def _md_table(df: pd.DataFrame, floatfmt: dict[str, str] | None = None) -> str:
    floatfmt = floatfmt or {}
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            x = r[c]
            if isinstance(x, (float, np.floating)):
                cells.append(format(x, floatfmt.get(c, ".2f")))
            else:
                cells.append(str(x))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_report(cfg: Config) -> Path:
    from .doe import study_dir
    from .sensitivity import analyze

    sd = study_dir(cfg)
    out = sd / "geometry"
    out.mkdir(parents=True, exist_ok=True)
    all_rows = pd.read_csv(sd / "results.csv")

    sens = analyze(sd / "results.csv", cfg.study.parameters, list(cfg.study.outputs))
    a = analyse(cfg)
    plot_tradeoff(a, out / "geometry_tradeoff.png")
    if len(a.geo) == 2:
        plot_maps(a, out / "geometry_maps.png")

    # comparison table: each geometric parameter at its min / reference / max
    v_ref = a.ref[a.velocity]
    rows = []
    for g in a.geo:
        lo, hi = cfg.study.parameters[g]
        for val in (lo, a.ref[g], hi):
            G = np.array([[val if k == g else a.ref[k] for k in a.geo]])
            for crit, label in CRITERIA.items():
                e = a.evaluate(crit, G).iloc[0]
                rows.append({"parameter": g, "value": val, "criterion": label, "v_m_s": e["velocity"],
                             "T_max_C": e["T_max_C"], "dP_Pa": e["dP_Pa"], "mass_flow_g_s": e["mass_flow_g_s"],
                             "fan_power_W": e["fan_power_W"], "in_range": bool(e["velocity_in_study_range"])})
    table = pd.DataFrame(rows)
    table.to_csv(out / "geometry_comparison.csv", index=False)

    # effects over each parameter's range (same velocity), with 2-sigma
    eff = []
    for name, law in (("R_th", a.r_law), ("dP", a.dp_law)):
        for i, g in enumerate(a.geo):
            lo, hi = cfg.study.parameters[g]
            c, s = law.coef[2 + i], law.stderr[2 + i]
            eff.append({"output": name, "parameter": g,
                        "change_over_range_%": (np.exp(c * (hi - lo)) - 1) * 100,
                        "plus_minus_2sigma_%": (np.exp(2 * s * (hi - lo)) - 1) * 100,
                        "significant": bool(abs(c) > 2 * s)})
    effects = pd.DataFrame(eff)
    effects.to_csv(out / "geometry_effects.csv", index=False)

    # Best geometry and spread at the same fan power, over the parameters with a SIGNIFICANT
    # effect on R_th only: the others are noise-level and stay at the reference (indifferent).
    significant = effects[(effects.output == "R_th") & effects.significant]["parameter"].tolist()
    axes = [np.linspace(*cfg.study.parameters[g], 41) if g in significant else np.array([a.ref[g]])
            for g in a.geo]
    Gg = np.column_stack([x.ravel() for x in np.meshgrid(*axes)])
    fan_eval = a.evaluate("same_fan_power", Gg)
    fan_spread = float(fan_eval["T_max_C"].max() - fan_eval["T_max_C"].min())
    best = {crit: a.evaluate(crit, Gg).sort_values("T_max_C").iloc[0] for crit in CRITERIA}

    rec = recommend(cfg, a, effects)
    rec.to_csv(out / "recommended_geometry.csv", index=False)
    size = array_size(cfg, a)
    path = out / "report.md"
    text = _report_text(cfg, a, all_rows, sens, table, effects, fan_spread, best, v_ref, significant)
    head, _, rest = text.partition("## 1. Estudo")
    path.write_text(head + _recommendation_text(cfg, a, rec, size) + "## 1. Estudo" + rest, encoding="utf-8")
    log.info("Geometry report written to %s", path)
    return path


def _recommendation_text(cfg, a: Analysis, rec: pd.DataFrame, size: dict | None) -> str:
    gr = cfg.geometry_report
    G = np.array([[float(rec.set_index("parameter").loc[g, "value"]) for g in a.geo]])
    geo_txt = ", ".join(f"{g.split('.', 1)[1]} = {G[0, i]:g} mm" for i, g in enumerate(a.geo))
    rec_md = _md_table(pd.DataFrame({"parâmetro": [_label(cfg, g) for g in rec.parameter],
                                     "valor recomendado": rec["value"], "motivo": rec["reason"]}),
                       {"valor recomendado": "g"})
    ops = []
    for crit, label in (("same_fan_power", "mesma potência de ventilador da referência"),
                        ("same_velocity", "mesma velocidade da referência"),
                        ("same_mass_flow", "mesma vazão da referência")):
        e = a.evaluate(crit, G).iloc[0]
        ops.append({"se o ventilador mantiver a": label, "v [m/s]": e["velocity"], "vazão [g/s]": e["mass_flow_g_s"],
                    "ΔP [Pa]": e["dP_Pa"], "pot. vent. [W]": e["fan_power_W"], "T max [°C]": e["T_max_C"]})
    ops_md = _md_table(pd.DataFrame(ops), {"v [m/s]": ".2f", "vazão [g/s]": ".1f", "ΔP [Pa]": ".0f",
                                           "pot. vent. [W]": ".2f", "T max [°C]": ".1f"})
    vol = ""
    if size is not None:
        g_ref = np.array([[a.ref[g] for g in a.geo]])
        g_max = np.array([[cfg.study.parameters[g][1] for g in a.geo]])
        v = array_volume(size, np.vstack([G, g_ref, g_max]))
        v.insert(0, "geometria", ["recomendada", "referência", "maior da faixa"])
        v["volume vs recomendada"] = v["volume_cm3"] / v["volume_cm3"].iloc[0]
        vol = ("\nTamanho do arranjo das 9 células (caixa que as envolve; medido nas posições das células "
               "de cada ponto):\n\n" + _md_table(
                   v.rename(columns={"length_mm": "comprimento [mm]", "width_mm": "largura [mm]",
                                     "height_mm": "altura [mm]", "volume_cm3": "volume [cm³]"}),
                   {"comprimento [mm]": ".1f", "largura [mm]": ".1f", "altura [mm]": ".2f", "volume [cm³]": ".0f",
                    "volume vs recomendada": ".2f"}) + "\n")
    return f"""## Recomendação

**Geometria recomendada: {geo_txt}.**

{rec_md}

Regra: a comparação justa entre geometrias é na **mesma potência de ventilador** (o mesmo gasto de
energia para mover o ar). Para cada parâmetro, os valores cujo T max fica a menos de
{gr.tie_tolerance_k:g} K ou de 2σ (incerteza do modelo) do melhor são considerados empatados, e entre
eles escolhe-se o menor, que dá o módulo mais compacto; um parâmetro sem efeito significativo
fica no menor valor da faixa.
{vol}
Operação dessa geometria a {gr.c_rate:g}C ({a.q_cell_w:.1f} W por célula):

{ops_md}

A refrigeração é a mesma das outras geometrias para a mesma potência de ventilador, mas o
ventilador precisa vencer uma queda de pressão maior com menos vazão: confira a curva do
ventilador (pressão estática disponível) na vazão da primeira linha.

"""


def _report_text(cfg, a: Analysis, all_rows, sens, table, effects, fan_spread, best, v_ref, significant) -> str:
    gr = cfg.geometry_report
    ok = all_rows[all_rows["status"] == "ok"] if "status" in all_rows else all_rows
    n_conv = len(a.df)
    params = "\n".join(f"| `{p}` | {lo:g} a {hi:g} | {_label(cfg, p)} |" for p, (lo, hi) in cfg.study.parameters.items())
    fixed = ", ".join(f"`cad.{k}` = {v:g}" for k, v in cfg.cad.items() if f"cad.{k}" not in cfg.study.parameters)

    st = sens.pivot_table(index="parameter", columns="output", values="ST").round(2).reset_index()
    st["parameter"] = st["parameter"].map(lambda p: _label(cfg, p))
    cop = sens.groupby("output")["CoP"].first()
    st_md = _md_table(st.rename(columns={"parameter": "parâmetro"}))
    cop_md = ", ".join(f"{o}: {v:.0%}" for o, v in cop.items())

    c = a.r_law.coef
    law_r = f"R_th ∝ v^{c[1]:.2f} · " + " · ".join(
        f"e^({c[2 + i]:+.4f}·{g.split('.', 1)[1]})" for i, g in enumerate(a.geo))
    d = a.dp_law.coef
    law_dp = f"ΔP ∝ v^{d[1]:.2f} · " + " · ".join(
        f"e^({d[2 + i]:+.4f}·{g.split('.', 1)[1]})" for i, g in enumerate(a.geo))
    area = f"A = {a.area_coef[0] * 1e6:.0f} mm² " + " ".join(
        f"{a.area_coef[1 + i] * 1e6:+.1f} mm²/mm · {g.split('.', 1)[1]}" for i, g in enumerate(a.geo))

    eff = effects.copy()
    eff["significant"] = eff["significant"].map({True: "sim", False: "não"})
    eff["output"] = eff["output"].map({"R_th": "R_th (refrigeração)", "dP": "ΔP"})
    eff_md = _md_table(eff.rename(columns={"output": "saída", "parameter": "parâmetro",
                                           "change_over_range_%": "variação na faixa [%]",
                                           "plus_minus_2sigma_%": "± 2σ [%]", "significant": "significativo"}),
                       {"variação na faixa [%]": "+.1f", "± 2σ [%]": ".1f"})
    tab = table.copy()
    tab["criterion"] = tab["criterion"].map({"same inlet velocity": "mesma velocidade", "same mass flow":
                                             "mesma vazão", "same fan power": "mesma potência"})
    tab = tab.rename(columns={"parameter": "parâmetro", "value": "valor", "criterion": "critério",
                              "v_m_s": "v [m/s]", "T_max_C": "T max [°C]", "dP_Pa": "ΔP [Pa]",
                              "mass_flow_g_s": "vazão [g/s]", "fan_power_W": "pot. vent. [W]",
                              "in_range": "v dentro da faixa"})
    tab["v dentro da faixa"] = tab["v dentro da faixa"].map({True: "sim", False: "não"})
    tab_md = _md_table(tab, {"valor": "g", "v [m/s]": ".2f", "T max [°C]": ".1f", "ΔP [Pa]": ".0f",
                             "vazão [g/s]": ".1f", "pot. vent. [W]": ".2f"})

    flat = fan_spread < 0.02 * (best["same_fan_power"]["T_max_C"] - gr.inlet_temperature_c)
    weak = [g for g in a.geo if g not in significant]
    weak_txt = (", ".join(f"`{p}`" for p in weak) + " não tem efeito significativo na refrigeração "
                "(dentro de ±2σ): escolha pelo tamanho do módulo.") if weak else \
        "Todos os parâmetros geométricos têm efeito significativo na refrigeração."
    best_lines = "\n".join(
        f"- **{ {'same_velocity': 'Mesma velocidade', 'same_mass_flow': 'Mesma vazão', 'same_fan_power': 'Mesma potência de ventilador'}[k] }:** "
        + ", ".join(f"{g.split('.', 1)[1]} = {b[g]:.1f}" if g in significant else
                    f"{g.split('.', 1)[1]} indiferente" for g in a.geo)
        + f" → T max {b['T_max_C']:.1f} °C, ΔP {b['dP_Pa']:.0f} Pa, v {b['velocity']:.2f} m/s"
        + ("" if b["velocity_in_study_range"] else " (velocidade fora da faixa do estudo: extrapolação)")
        for k, b in best.items())
    ref_txt = ", ".join(f"{k.split('.', 1)[1]} = {v:g}" for k, v in a.ref.items())
    sig_txt = ", ".join(f"`{g}`" for g in significant) or "nenhum parâmetro"
    maps = "\n![Mapas](geometry_maps.png)\n" if len(a.geo) == 2 else ""
    verdict = (f"Na **mesma potência de ventilador**, {sig_txt} muda o T max em só {fan_spread:.1f} K "
               f"em toda a faixa: **as geometrias empatam**. A escolha depende do ventilador (a pressão "
               f"que ele entrega: veja ΔP na tabela da seção 4) e do tamanho do módulo." if flat else
               f"Na mesma potência de ventilador, {sig_txt} muda o T max em até {fan_spread:.1f} K. O menor "
               f"T max acima é a estimativa central; a recomendação no topo trata como empatados os "
               f"valores dentro da incerteza do modelo (2σ) e escolhe entre eles o mais compacto.")

    return f"""# Relatório: escolha da geometria, estudo `{cfg.study.name}`

Gerado por `autocfd geometry-report` em {time.strftime('%Y-%m-%d %H:%M')}. Os números saem dos
resultados de CFD do estudo; refaça o comando depois de mudar o estudo.

## 1. Estudo

- {len(all_rows)} pontos no plano, {len(ok)} resolvidos, **{n_conv} convergidos** (usados aqui).
- Parâmetros variados (LHS):

| parâmetro | faixa | significado |
|---|---|---|
{params}

- Fixos: {fixed or '-'}.

## 2. Sensibilidade (todos os parâmetros juntos)

Índice de Sobol total S_T (fração da variância explicada, incluindo interações):

{st_md}

Qualidade do modelo de superfície (CoP): {cop_md}.

A **taxa C domina o T max**: ela esconde o efeito da geometria. Por isso a comparação de
geometria abaixo usa a resistência térmica, que não depende da taxa C.

## 3. Resistência térmica e queda de pressão

R_th = (T max − T entrada) / Q_célula [K/W]. Com propriedades constantes o problema é linear
no calor, então R_th depende só da geometria e da velocidade do ar. Leis ajustadas nos
{n_conv} pontos (mínimos quadrados em log):

- {law_r} — validação deixando um ponto de fora: R² = {a.r_law.loo_r2:.3f}
- {law_dp} — R² = {a.dp_law.loo_r2:.3f}

As leis servem para ler o efeito de cada parâmetro (tabela abaixo). **A comparação das seções
4 e 5 e a recomendação usam processos gaussianos** nas mesmas entradas (R² deixando um ponto de
fora: R_th {a.r_gp_loo:.3f}, ΔP {a.dp_gp_loo:.3f}), porque a célula tem uma resistência de condução
interna em série com a convecção: na mesma potência de ventilador, R_th pode ter um mínimo no
meio da faixa de uma folga, o que uma lei log-linear (monotônica) não representa.
- Área de entrada do duto, da vazão medida: {area} (erro máx. {a.area_resid * 1e6:.1f} mm²)

Efeito de cada parâmetro sobre toda a sua faixa, na mesma velocidade:

{eff_md}

## 4. Comparação justa

Um parâmetro que alarga o duto recebe mais ar na mesma velocidade, então "mesma velocidade" não
é uma comparação justa. Três critérios, relativos à referência ({ref_txt};
vazão {a.m_ref * 1000:.1f} g/s, potência de ventilador {a.fan_ref_w:.2f} W), a
**{gr.c_rate:g}C ({a.q_cell_w:.1f} W por célula)**:

{tab_md}

![Compromisso](geometry_tradeoff.png)

**Como ler as curvas.** Cada curva responde: "trocando a geometria de referência por outra,
mantendo X igual ao da referência, o que acontece?". Na referência ({ref_txt}) as três condições
são idênticas (mesma velocidade, mesma vazão e mesma potência ao mesmo tempo), por isso **as três
curvas se cruzam no valor de referência por construção**: o cruzamento não é um ótimo nem um
compromisso; com outra referência elas se cruzariam em outro ponto. O que importa é a forma de
cada curva: na mesma velocidade ou vazão a geometria muda muito a temperatura, mas à custa de
gastar mais (ou menos) energia no ventilador; **na mesma potência de ventilador** (a comparação
justa) a curva plana indica que as geometrias resfriam igual e diferem só na combinação
vazão × pressão que o ventilador precisa entregar e no tamanho do módulo.
{maps}
## 5. Detalhe: menor T max em cada critério

Otimizado só nos parâmetros com efeito significativo ({sig_txt}); os demais ficam na referência.
A recomendação (no topo) usa a mesma potência de ventilador, com empates resolvidos pela compacidade.

{best_lines}

{verdict}

{weak_txt}

## 6. Cuidados

- Resultados em regime permanente: a {gr.c_rate:g}C a descarga dura minutos e o T max real é menor
  (ver o modelo transiente das células, `autocfd pinn`).
- Verifique a orientação da condutividade do material das células no caso base (eixo das
  células vs direção de k axial) e o diâmetro real das células: mudam os valores de R_th; a
  comparação entre geometrias depende sobretudo do lado do ar.
- Pontos não convergidos ficam de fora; fora da faixa de velocidade do estudo, os valores são
  extrapolação (coluna "v dentro da faixa").
"""
