import dataclasses

import numpy as np
import pandas as pd
import pytest

from autocfd.config import load_config
from tests.test_config import CONFIG


def _study(tmp_path, n=40, seed=0):
    """Synthetic study with KNOWN laws: R_th = 3 v^-0.6 e^(0.06 (L1-6)), P1 has no effect;
    dP = 100 (v/3)^1.8 e^(-0.3 (L1-6)); inlet area grows with L1."""
    cfg = load_config(CONFIG)
    cfg = dataclasses.replace(cfg, paths=dataclasses.replace(cfg.paths, output_dir=str(tmp_path)))
    rng = np.random.default_rng(seed)
    L1, P1 = rng.uniform(2, 10, n), rng.uniform(2, 10, n)
    v, c = rng.uniform(1, 7, n), rng.uniform(1, 10, n)
    q = (c * 5.0) ** 2 * 0.0128
    rth = 3 * v ** -0.6 * np.exp(0.06 * (L1 - 6)) * np.exp(rng.normal(0, 0.01, n))
    dp = 100 * (v / 3) ** 1.8 * np.exp(-0.3 * (L1 - 6))
    area = (4700 + 240 * L1) * 1e-6
    df = pd.DataFrame({"id": range(n), "cad.L1": L1, "cad.P1": P1, "operating.c_rate": c,
                       "fluent.parameter_velocity_inlet": v, "status": "ok", "converged": True,
                       "Q_cell_W": q, "T_max_C": 20 + rth * q, "dT_cells_K": 0.3 * rth * q, "dP_Pa": dp,
                       "m_out": -1.177 * v * area})
    out = tmp_path / "studies" / cfg.study.name
    out.mkdir(parents=True)
    df.to_csv(out / "results.csv", index=False)
    return cfg, out


def test_laws_recovered_and_fan_power_is_equal(tmp_path):
    from autocfd.geometry_report import analyse

    cfg, _ = _study(tmp_path)
    a = analyse(cfg)
    assert a.r_law.coef[1] == pytest.approx(-0.6, abs=0.02)
    assert a.r_law.coef[2] == pytest.approx(0.06, abs=0.005)          # L1
    assert abs(a.r_law.coef[3]) < 2 * a.r_law.stderr[3] + 1e-3          # P1: no effect
    assert a.dp_law.coef[1] == pytest.approx(1.8, abs=1e-6)
    assert a.area_coef[1] * 1e6 == pytest.approx(240, rel=1e-6)
    G = np.array([[2.0, 6.0], [10.0, 6.0]])
    e = a.evaluate("same_fan_power", G)
    assert e["fan_power_W"].to_numpy() == pytest.approx([a.fan_ref_w] * 2, rel=1e-9)
    m = a.evaluate("same_mass_flow", G)
    assert m["mass_flow_g_s"].to_numpy() == pytest.approx([a.m_ref * 1000] * 2, rel=1e-9)


def test_report_written_with_figures(tmp_path):
    from autocfd.geometry_report import build_report

    cfg, out = _study(tmp_path)
    path = build_report(cfg)
    text = path.read_text(encoding="utf-8")
    assert "# Relatório: escolha da geometria" in text
    assert "P1 indiferente" in text                        # P1 has no effect in the synthetic data
    assert (out / "geometry" / "geometry_tradeoff.png").exists()
    assert (out / "geometry" / "geometry_maps.png").exists()
    eff = pd.read_csv(out / "geometry" / "geometry_effects.csv")
    sig = eff.set_index(["output", "parameter"])["significant"]
    assert bool(sig[("R_th", "cad.L1")]) and not bool(sig[("R_th", "cad.P1")])
