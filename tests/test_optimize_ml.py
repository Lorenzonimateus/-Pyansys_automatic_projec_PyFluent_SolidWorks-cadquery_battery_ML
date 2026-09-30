import numpy as np
import pandas as pd
import pytest

from autocfd.config import Config, Optimize, Study
from autocfd.ml import fit_all, train
from autocfd.optimize import expected_improvement, feasible, parse_constraints, propose


def _toy(n=25, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform([0, 0], [1, 1], (n, 2))
    obj = (x[:, 0] - 0.3) ** 2 + (x[:, 1] - 0.7) ** 2   # minimum at (0.3, 0.7)
    con = x[:, 0] + x[:, 1]                              # constraint: <= 1.2
    return pd.DataFrame({"a.x": x[:, 0], "a.y": x[:, 1], "obj": obj, "con": con, "status": "ok"})


def test_parse_constraints():
    c = parse_constraints(["dP_Pa <= 25", "T >= -1e3"])
    assert c[0].output == "dP_Pa" and c[0].satisfied(20) and not c[0].satisfied(30)
    assert c[1].op == ">=" and c[1].limit == -1000.0
    with pytest.raises(ValueError):
        parse_constraints(["dP < 3"])


def test_expected_improvement_prefers_low_mean_and_high_std():
    assert expected_improvement(np.array([0.0]), np.array([0.1]), 1.0) > \
           expected_improvement(np.array([0.9]), np.array([0.1]), 1.0)
    assert expected_improvement(np.array([1.0]), np.array([0.5]), 1.0) > \
           expected_improvement(np.array([1.0]), np.array([0.01]), 1.0)


def test_propose_moves_towards_constrained_optimum():
    df = _toy()
    cfg = Config(study=Study(parameters={"a.x": (0.0, 1.0), "a.y": (0.0, 1.0)}),
                 optimize=Optimize(objective="obj", sense="min", constraints=("con <= 1.2",)))
    models = fit_all(df[["a.x", "a.y"]].to_numpy(), df, ["obj", "con"])
    point, score = propose(models, cfg, df, seed=0)
    assert score >= 0
    assert point["a.x"] + point["a.y"] <= 1.3
    assert abs(point["a.x"] - 0.3) < 0.25 and abs(point["a.y"] - 0.7) < 0.25
    assert feasible(df, parse_constraints(["con <= 1.2"])).any()


def test_train_writes_model(tmp_path):
    df = _toy(20)
    csv = tmp_path / "results.csv"
    df.to_csv(csv, index=False)
    metrics = train(csv, ["a.x", "a.y"], ["obj"], out_dir=tmp_path)
    assert (tmp_path / "ml_model.joblib").exists()
    assert metrics.loc[0, "R2_loo"] > 0.9
