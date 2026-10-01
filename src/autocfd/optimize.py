"""Surrogate-based optimization (Bayesian optimization) on top of the DOE.

Each iteration:
  1. trains one Gaussian process per output (objective + constrained outputs)
     on every valid point of results.csv;
  2. picks the point that maximizes the constrained expected improvement,
     EI(objective) x P(all constraints satisfied), by differential evolution;
  3. runs that point in the full chain (SolidWorks -> ... -> Fluent) and
     appends it to results.csv, so the next surrogate is better exactly where
     the optimum is.
The answer is always a CFD-verified point, never just a surrogate prediction.
Log of the iterations: output/studies/<name>/optimization.csv.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from scipy.stats import norm

from .config import Config
from .doe import run_points, study_dir
from .pipeline import append_csv
from .sensitivity import load_valid_points

log = logging.getLogger(__name__)

_CONSTRAINT_RE = re.compile(r"^\s*([A-Za-z_]\w*)\s*(<=|>=)\s*([-+0-9.eE]+)\s*$")


@dataclass(frozen=True)
class Constraint:
    output: str
    op: str
    limit: float

    def satisfied(self, value: float) -> bool:
        return value <= self.limit if self.op == "<=" else value >= self.limit

    def probability(self, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        z = (self.limit - mean) / np.maximum(std, 1e-12)
        return norm.cdf(z) if self.op == "<=" else norm.cdf(-z)


def parse_constraints(items) -> list[Constraint]:
    out = []
    for text in items:
        m = _CONSTRAINT_RE.match(text)
        if not m:
            raise ValueError(f"Constraint {text!r} must look like 'output <= value' or 'output >= value'")
        out.append(Constraint(m.group(1), m.group(2), float(m.group(3))))
    return out


def feasible(df: pd.DataFrame, constraints: list[Constraint]) -> pd.Series:
    ok = pd.Series(True, index=df.index)
    for c in constraints:
        ok &= df[c.output].map(c.satisfied)
    return ok


def expected_improvement(mean: np.ndarray, std: np.ndarray, best: float) -> np.ndarray:
    """EI for MINIMIZATION of the (sign-adjusted) objective."""
    std = np.maximum(std, 1e-12)
    z = (best - mean) / std
    return (best - mean) * norm.cdf(z) + std * norm.pdf(z)


def acquisition(models: dict, objective: str, sign: float, constraints: list[Constraint],
                best: float | None):
    """Function x (n, d) -> score (n,); higher is better."""
    def score(x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        pof = np.ones(len(x))
        for c in constraints:
            mean, std = models[c.output].predict(x, return_std=True)
            pof *= c.probability(mean, std)
        if best is None:  # no feasible point yet: first find the feasible region
            return pof
        mean, std = models[objective].predict(x, return_std=True)
        return expected_improvement(sign * mean, std, best) * pof
    return score


def propose(models: dict, cfg: Config, df: pd.DataFrame, seed: int) -> tuple[dict, float]:
    """Next point to evaluate, and its acquisition score."""
    opt = cfg.optimize
    names = list(cfg.study.parameters)
    bounds = [cfg.study.parameters[n] for n in names]
    constraints = parse_constraints(opt.constraints)
    sign = 1.0 if opt.sense == "min" else -1.0
    ok = feasible(df, constraints)
    best = float((sign * df.loc[ok, opt.objective]).min()) if ok.any() else None

    score = acquisition(models, opt.objective, sign, constraints, best)
    res = differential_evolution(lambda x: -score(x)[0], bounds, seed=seed, tol=1e-8,
                                 maxiter=200, popsize=20, polish=True)
    x = res.x
    # Avoid re-running an existing point (EI is ~0 there, but DE can end on one)
    span = np.array([hi - lo for lo, hi in bounds])
    dist = np.min(np.linalg.norm((df[names].to_numpy(float) - x) / span, axis=1))
    if dist < 1e-3:
        log.warning("Proposed point coincides with an existing one; the optimum is probably found")
    return dict(zip(names, map(float, x))), float(-res.fun)


def optimize(cfg: Config) -> pd.DataFrame:
    from .ml import fit_all

    opt = cfg.optimize
    out = study_dir(cfg)
    results_csv = out / "results.csv"
    if not results_csv.exists():
        raise FileNotFoundError(f"{results_csv} not found: run the study (DOE) first")
    constraints = parse_constraints(opt.constraints)
    outputs = list(dict.fromkeys([opt.objective, *(c.output for c in constraints)]))
    names = list(cfg.study.parameters)

    for it in range(opt.iterations):
        df = load_valid_points(results_csv)
        missing = [o for o in outputs if o not in df]
        if missing:
            raise KeyError(f"Outputs not in results.csv: {missing}")
        models = fit_all(df[names].to_numpy(float), df, outputs)
        point, score = propose(models, cfg, df, seed=opt.seed + it)
        pred = {}
        for o in outputs:
            mean, std = models[o].predict(np.array([[point[n] for n in names]]), return_std=True)
            pred[f"pred_{o}"], pred[f"pred_{o}_std"] = float(mean[0]), float(std[0])
        log.info("Iteration %d/%d: %s  (acquisition %.3g, %s)", it + 1, opt.iterations, point, score, pred)

        all_ids = pd.read_csv(results_csv)["id"]
        pid = int(all_ids.max()) + 1
        run_points(cfg, [{"id": pid, **point}], results_csv)
        row = pd.read_csv(results_csv).set_index("id").loc[pid]
        append_csv(out / "optimization.csv", {"iteration": it + 1, "id": pid, **point, "acquisition": score,
                                              **pred, "status": row["status"],
                                              **{o: row.get(o, np.nan) for o in outputs}})
    return report_best(cfg)


def report_best(cfg: Config) -> pd.DataFrame:
    opt = cfg.optimize
    df = load_valid_points(study_dir(cfg) / "results.csv")
    constraints = parse_constraints(opt.constraints)
    ok = df[feasible(df, constraints)]
    if ok.empty:
        print("\nNo CFD point satisfies the constraints yet; run more optimize iterations "
              "or relax the constraints.")
        return ok
    best = ok.sort_values(opt.objective, ascending=opt.sense == "min").head(5)
    cols = ["id", *cfg.study.parameters, opt.objective, *(c.output for c in constraints)]
    print(f"\nBest CFD points ({opt.sense} {opt.objective}, constraints {list(opt.constraints)}):")
    print(best[list(dict.fromkeys(cols))].to_string(index=False))
    return best
