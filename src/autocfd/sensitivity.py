"""Sensitivity analysis of the DOE results (replaces optiSLang's MOP).

1. Polynomial response surface (quadratic with interactions; linear if there are
   few points) fitted by least squares on normalized inputs.
2. CoP (Coefficient of Prognosis), as in optiSLang: 1 - PRESS/SS_tot, with the
   leave-one-out residuals computed exactly (r_i / (1 - h_ii)).
3. First-order and total Sobol indices, evaluated on the surrogate
   (Saltelli 2010 / Jansen estimators).
4. Spearman correlation on the raw data, as a model-free check.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

log = logging.getLogger(__name__)

# Colors (project reference palette): ink, grid, sequential blue
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb"
BLUE = "#2a78d6"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


# ------------------------------------------------------------------ surrogate
def _terms(n_inputs: int, order: int) -> list[tuple[int, ...]]:
    """List of monomials: () constant, (i,) linear, (i,i) square, (i,j) interaction."""
    terms: list[tuple[int, ...]] = [()] + [(i,) for i in range(n_inputs)]
    if order >= 2:
        terms += list(itertools.combinations_with_replacement(range(n_inputs), 2))
    return terms


def _design_matrix(x: np.ndarray, terms) -> np.ndarray:
    return np.column_stack([np.prod(x[:, list(t)], axis=1) if t else np.ones(len(x)) for t in terms])


@dataclass
class Surrogate:
    names: list[str]
    lo: np.ndarray
    hi: np.ndarray
    terms: list[tuple[int, ...]]
    coef: np.ndarray
    cop: float
    r2: float

    def _norm(self, x: np.ndarray) -> np.ndarray:
        return 2 * (x - self.lo) / (self.hi - self.lo) - 1

    def predict(self, x: np.ndarray) -> np.ndarray:
        return _design_matrix(self._norm(np.atleast_2d(x)), self.terms) @ self.coef


def fit_surrogate(x: np.ndarray, y: np.ndarray, names: list[str],
                  bounds: dict[str, tuple[float, float]] | None = None) -> Surrogate:
    lo = np.array([bounds[n][0] for n in names]) if bounds else x.min(axis=0)
    hi = np.array([bounds[n][1] for n in names]) if bounds else x.max(axis=0)
    xn = 2 * (x - lo) / (hi - lo) - 1

    # Quadratic if there are enough points (margin for leave-one-out)
    order = 2 if len(x) >= len(_terms(len(names), 2)) + 3 else 1
    terms = _terms(len(names), order)
    a = _design_matrix(xn, terms)
    coef, *_ = np.linalg.lstsq(a, y, rcond=None)

    resid = y - a @ coef
    h = np.einsum("ij,ji->i", a, np.linalg.pinv(a))       # diagonal of the hat matrix
    loo = resid / np.clip(1 - h, 1e-9, None)
    ss_tot = float(np.sum((y - y.mean()) ** 2)) or 1e-30
    cop = 1 - float(np.sum(loo**2)) / ss_tot
    r2 = 1 - float(np.sum(resid**2)) / ss_tot
    if order == 1:
        log.warning("Too few points (%d) for a quadratic model; using a linear model", len(x))
    return Surrogate(names, lo, hi, terms, coef, cop, r2)


def sobol_indices(model: Surrogate, n: int = 2**14, seed: int = 0) -> pd.DataFrame:
    """Sobol indices (S1, ST) with uniform inputs over the model ranges."""
    rng = np.random.default_rng(seed)
    d = len(model.names)
    span = model.hi - model.lo
    a = model.lo + rng.random((n, d)) * span
    b = model.lo + rng.random((n, d)) * span
    fa, fb = model.predict(a), model.predict(b)
    var = np.var(np.concatenate([fa, fb]))
    rows = []
    for i, name in enumerate(model.names):
        abi = a.copy()
        abi[:, i] = b[:, i]
        fabi = model.predict(abi)
        s1 = np.mean(fb * (fabi - fa)) / var if var > 0 else 0.0
        st = 0.5 * np.mean((fa - fabi) ** 2) / var if var > 0 else 0.0
        rows.append({"parameter": name, "S1": float(s1), "ST": float(st)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ analysis
def load_valid_points(results_csv: Path) -> pd.DataFrame:
    """Read results.csv and keep only the computed and converged points."""
    df = pd.read_csv(results_csv)
    if "status" in df:
        df = df[df["status"] == "ok"]
    if "converged" in df:
        bad = ~df["converged"].astype(str).str.lower().eq("true")
        if bad.any():
            log.warning("%d non-converged point(s) excluded from the analysis", int(bad.sum()))
            df = df[~bad]
    return df


def analyze(results_csv: Path, bounds: dict[str, tuple[float, float]],
            outputs: list[str], out_dir: Path | None = None) -> pd.DataFrame:
    df = load_valid_points(results_csv)
    names = list(bounds)
    outputs = [o for o in outputs if o in df]
    if len(df) < len(names) + 2:
        raise RuntimeError(f"Only {len(df)} valid points; too few for {len(names)} parameters")

    x = df[names].to_numpy(float)
    table, models = [], {}
    for out in outputs:
        y = df[out].to_numpy(float)
        model = fit_surrogate(x, y, names, bounds)
        models[out] = model
        sob = sobol_indices(model)
        for _, r in sob.iterrows():
            rho = spearmanr(df[r["parameter"]], y).statistic
            table.append({"output": out, "parameter": r["parameter"], "S1": r["S1"],
                          "ST": r["ST"], "spearman": float(rho), "CoP": model.cop, "R2": model.r2})
    res = pd.DataFrame(table)

    out_dir = out_dir or results_csv.parent
    res.to_csv(out_dir / "sensitivity.csv", index=False)
    _plot_indices(res, out_dir / "sensitivity_sobol.png")
    if len(names) >= 2:
        _plot_surfaces(df, models, res, out_dir / "response_surfaces.png")
    _print_report(res, len(df))
    return res


# ------------------------------------------------------------------ plots
def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)


def _plot_indices(res: pd.DataFrame, path: Path) -> None:
    """Small multiples: one panel per output, bars of the total index ST."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outs = list(dict.fromkeys(res["output"]))
    fig, axes = plt.subplots(1, len(outs), figsize=(4.2 * len(outs), 0.6 * res["parameter"].nunique() + 1.6),
                             sharey=True, facecolor=SURFACE, squeeze=False)
    for ax, out in zip(axes[0], outs):
        sub = res[res["output"] == out].sort_values("ST")
        ax.barh(sub["parameter"], sub["ST"], color=BLUE, height=0.55)
        for yv, v in enumerate(sub["ST"]):
            ax.text(v + 0.02, yv, f"{v:.2f}", va="center", fontsize=9, color=INK)
        cop = sub["CoP"].iloc[0]
        ax.set_title(f"{out}  (CoP {cop:.0%})", fontsize=10, color=INK, loc="left")
        ax.set_xlim(0, 1.15)
        ax.xaxis.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        _style(ax)
    axes[0][0].set_ylabel("")
    fig.supxlabel("Total Sobol index (ST)", fontsize=9, color=INK_2)
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor=SURFACE)
    plt.close(fig)


def _plot_surfaces(df: pd.DataFrame, models: dict[str, Surrogate], res: pd.DataFrame, path: Path) -> None:
    """Surrogate of each output over the 2 most influential parameters (others at mid-range)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("blue_seq", BLUE_RAMP)
    fig, axes = plt.subplots(1, len(models), figsize=(4.6 * len(models), 3.9),
                             facecolor=SURFACE, squeeze=False)
    for ax, (out, m) in zip(axes[0], models.items()):
        top = res[res["output"] == out].nlargest(2, "ST")["parameter"].tolist()
        i, j = m.names.index(top[0]), m.names.index(top[1])
        g1 = np.linspace(m.lo[i], m.hi[i], 80)
        g2 = np.linspace(m.lo[j], m.hi[j], 80)
        gg1, gg2 = np.meshgrid(g1, g2)
        pts = np.tile((m.lo + m.hi) / 2, (gg1.size, 1))
        pts[:, i], pts[:, j] = gg1.ravel(), gg2.ravel()
        z = m.predict(pts).reshape(gg1.shape)
        cf = ax.contourf(gg1, gg2, z, levels=14, cmap=cmap)
        ax.contour(gg1, gg2, z, levels=7, colors="white", linewidths=0.6, alpha=0.7)
        ax.scatter(df[top[0]], df[top[1]], s=22, c=INK, edgecolors=SURFACE, linewidths=1.5,
                   zorder=3, label="CFD points")
        cb = fig.colorbar(cf, ax=ax)
        cb.ax.tick_params(labelsize=8, colors=INK_2)
        cb.outline.set_visible(False)
        ax.set_title(out, fontsize=10, color=INK, loc="left")
        ax.set_xlabel(top[0], fontsize=9, color=INK_2)
        ax.set_ylabel(top[1], fontsize=9, color=INK_2)
        _style(ax)
    axes[0][0].legend(fontsize=8, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor=SURFACE)
    plt.close(fig)


def _print_report(res: pd.DataFrame, n_points: int) -> None:
    print(f"\nSensitivity ({n_points} valid points)")
    for out, sub in res.groupby("output", sort=False):
        print(f"\n  {out}   CoP={sub['CoP'].iloc[0]:.1%}  R2={sub['R2'].iloc[0]:.3f}")
        print(f"    {'parameter':<24}{'S1':>7}{'ST':>7}{'Spearman':>10}")
        for _, r in sub.sort_values("ST", ascending=False).iterrows():
            print(f"    {r['parameter']:<24}{r['S1']:>7.2f}{r['ST']:>7.2f}{r['spearman']:>10.2f}")
    low = res.groupby("output")["CoP"].first()
    for out in low[low < 0.8].index:
        print(f"\n  Warning: CoP of {out} < 80%: the surrogate is weak; add points (study.samples or optimize).")
