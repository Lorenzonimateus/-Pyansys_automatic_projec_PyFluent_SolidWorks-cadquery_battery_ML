"""Machine Learning model trained on the DOE results.

One Gaussian process (scikit-learn) per output: with few simulations (~20) it is
the most suitable model, because it interpolates the CFD points, captures
non-linearities and gives the uncertainty of each prediction.

- Inputs scaled to [0, 1]; Matern 5/2 kernel with one length scale per input
  (ARD) + small white noise (converged CFD has some numerical noise).
- Quality measured by leave-one-out cross-validation: each point is predicted by
  a model trained without it ("honest" R2 and RMSE, like optiSLang's CoP).
- Outputs in output/studies/<name>/: ml_model.joblib, ml_metrics.csv, ml_parity.png.

Usage once trained:
    from autocfd.ml import predict
    predict("output/studies/doe_v1/ml_model.joblib",
            {"cad.D": 20.0, "cad.P1": 10.0, "fluent.parameter_velocity_inlet": 4.0})
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .sensitivity import BLUE, GRID, INK, INK_2, SURFACE, _style, load_valid_points

log = logging.getLogger(__name__)

MODEL_FILE = "ml_model.joblib"


def make_model(n_inputs: int):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import MinMaxScaler

    kernel = (ConstantKernel(1.0, (1e-3, 1e3))
              * Matern(length_scale=np.ones(n_inputs), length_scale_bounds=(1e-2, 1e2), nu=2.5)
              + WhiteKernel(1e-4, (1e-8, 1e-1)))
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=5, random_state=0)
    return Pipeline([("scale", MinMaxScaler()), ("gp", gp)])


def train(results_csv: Path, inputs: list[str], outputs: list[str],
          out_dir: Path | None = None) -> pd.DataFrame:
    """Train one model per output, evaluate by leave-one-out and save everything."""
    import joblib
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.model_selection import LeaveOneOut, cross_val_predict

    # The optimizer warns when the noise hits its lower bound (CFD is nearly noise-free): harmless
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    df = load_valid_points(results_csv)
    missing = [o for o in outputs if o not in df]
    if missing:
        raise KeyError(f"Outputs not in {results_csv.name}: {missing}")
    if len(df) < len(inputs) + 3:
        raise RuntimeError(f"Only {len(df)} valid points; too few to train with {len(inputs)} inputs")
    x = df[inputs].to_numpy(float)

    models, rows, loo_pred = fit_all(x, df, outputs), [], {}
    for out in outputs:
        y = df[out].to_numpy(float)
        pred = cross_val_predict(make_model(len(inputs)), x, y, cv=LeaveOneOut())
        err = pred - y
        ss_tot = float(np.sum((y - y.mean()) ** 2)) or 1e-30
        rows.append({"output": out, "n_points": len(y),
                     "R2_loo": 1 - float(np.sum(err**2)) / ss_tot,
                     "RMSE_loo": float(np.sqrt(np.mean(err**2))),
                     "MAE_loo": float(np.mean(np.abs(err))),
                     "max_abs_err_loo": float(np.max(np.abs(err)))})
        loo_pred[out] = (y, pred)
    metrics = pd.DataFrame(rows)

    out_dir = out_dir or results_csv.parent
    joblib.dump({"inputs": inputs, "models": models, "metrics": metrics}, out_dir / MODEL_FILE)
    metrics.to_csv(out_dir / "ml_metrics.csv", index=False)
    _plot_parity(loo_pred, metrics, out_dir / "ml_parity.png")
    _print_report(metrics, out_dir)
    return metrics


def fit_all(x: np.ndarray, df: pd.DataFrame, outputs: list[str]) -> dict:
    """One Gaussian process per output, trained on all the points."""
    import warnings
    from sklearn.exceptions import ConvergenceWarning

    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    return {out: make_model(x.shape[1]).fit(x, df[out].to_numpy(float)) for out in outputs}


def predict(model_path: str | Path, points: dict[str, float] | pd.DataFrame) -> pd.DataFrame:
    """Prediction (mean and standard deviation) of each output for one or more points."""
    import joblib

    bundle = joblib.load(model_path)
    df = pd.DataFrame([points]) if isinstance(points, dict) else points.copy()
    x = df[bundle["inputs"]].to_numpy(float)
    for out, model in bundle["models"].items():
        mean, std = model.predict(x, return_std=True)
        df[out], df[f"{out}_std"] = mean, std
    return df


def _plot_parity(loo_pred: dict[str, tuple[np.ndarray, np.ndarray]], metrics: pd.DataFrame,
                 path: Path) -> None:
    """Small multiples: CFD value vs leave-one-out prediction, one panel per output."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(loo_pred), figsize=(3.8 * len(loo_pred), 3.9),
                             facecolor=SURFACE, squeeze=False)
    r2 = metrics.set_index("output")["R2_loo"]
    for ax, (out, (y, pred)) in zip(axes[0], loo_pred.items()):
        lo, hi = min(y.min(), pred.min()), max(y.max(), pred.max())
        pad = 0.05 * (hi - lo or 1)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=INK_2, lw=1, ls="--", zorder=1)
        ax.scatter(y, pred, s=36, color=BLUE, edgecolors=SURFACE, linewidths=1.5, zorder=3)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_aspect("equal")
        ax.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.set_title(f"{out}  (R² LOO {r2[out]:.3f})", fontsize=10, color=INK, loc="left")
        ax.set_xlabel("CFD", fontsize=9, color=INK_2)
        ax.set_ylabel("predicted (leave-one-out)", fontsize=9, color=INK_2)
        _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor=SURFACE)
    plt.close(fig)


def _print_report(metrics: pd.DataFrame, out_dir: Path) -> None:
    print(f"\nML model (Gaussian process), leave-one-out validation, {metrics['n_points'].iloc[0]} points")
    print(f"    {'output':<14}{'R2':>8}{'RMSE':>12}{'max error':>12}")
    for _, r in metrics.iterrows():
        print(f"    {r['output']:<14}{r['R2_loo']:>8.3f}{r['RMSE_loo']:>12.3g}{r['max_abs_err_loo']:>12.3g}")
    for out in metrics.loc[metrics["R2_loo"] < 0.9, "output"]:
        print(f"\n  Warning: R2 of {out} < 0.9: add simulations (study.samples) or run optimize.")
    print(f"\n  Model saved to {out_dir / MODEL_FILE}")
