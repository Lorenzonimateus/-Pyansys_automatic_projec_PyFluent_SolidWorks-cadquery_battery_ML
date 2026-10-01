from pathlib import Path

import numpy as np
import pytest

from autocfd.cell_thermal import CellModel, Load
from autocfd.config import load_config
from tests.test_config import CONFIG

torch = pytest.importorskip("torch")
ROOT = Path(__file__).parents[1]


def _cell():
    nphi, ny = 24, 8
    phi = (np.arange(nphi) + 0.5) * 2 * np.pi / nphi
    y = np.linspace(0, 0.07015, ny)
    h = 25 + 55 * np.clip(-np.cos(phi), 0, None)[:, None] + 0 * y[None, :]
    return CellModel("t", 0.011, 0.07015, 0.8, 0.8, 25.0, 2775.0, 1050.0, phi, y, h, t_ref_k=293.15)


def test_pinn_initial_condition_is_exact_and_training_reduces_the_loss():
    from autocfd.pinn import CellPINN, PINNSettings

    load = Load.constant(10.0, 360.0, 5.0, 0.0128)
    s = PINNSettings(width=32, depth=3, iterations=60, n_interior=512, n_side=256, n_ends=128,
                     device="cpu", log_every=30)
    pinn = CellPINN(_cell(), load, 293.15, s)
    pts = np.array([[0.0, 0.035, 0.0], [0.01, 0.01, 0.0]])
    assert np.allclose(pinn.predict(pts, [0.0]), 293.15)        # hard initial condition
    hist = pinn.train()
    assert hist[-1]["loss"] < hist[0]["loss"]
    assert np.all(np.isfinite(pinn.predict(pts, [180.0, 360.0])))


def test_h_interpolation_on_torch_matches_numpy():
    from autocfd.pinn import CellPINN, PINNSettings

    cell = _cell()
    pinn = CellPINN(cell, Load.constant(1, 100, 5, 0.0128), 293.15, PINNSettings(device="cpu"))
    phi = np.linspace(0, 2 * np.pi, 50, endpoint=False)
    y = np.linspace(0, 0.07, 50)
    ht = pinn._h(torch.tensor(phi, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)).numpy()
    assert np.allclose(ht, cell.h_at(phi, y), rtol=1e-4)


@pytest.mark.skipif(not (ROOT / "inputs/geometria_parametrizada_9_cells.cas.h5").exists(),
                    reason="base case not present")
def test_material_read_from_case_file():
    from autocfd.cells_export import case_materials

    m = case_materials(ROOT / "inputs/geometria_parametrizada_9_cells.cas.h5")
    assert sorted(m) == [f"cell_{i}" for i in range(1, 10)]
    assert m["cell_1"]["rho"] == 2775.0 and m["cell_1"]["cp"] == 1050.0
    # as set in the base case: k = 25 along z (the cells' axis is y, see README)
    assert (m["cell_1"]["kx"], m["cell_1"]["ky"], m["cell_1"]["kz"]) == (0.8, 0.8, 25.0)


def test_pinn_config():
    p = load_config(CONFIG).pinn
    assert p.cell == "hottest" and p.c_rate == 10.0 and p.device == "cuda"


def test_parametric_pinn_scales_and_initial_condition():
    from autocfd.pinn import ParametricCellPINN, PINNSettings

    s = PINNSettings(width=32, depth=3, iterations=40, n_interior=512, n_side=256, n_ends=128,
                     device="cpu", log_every=20)
    p = ParametricCellPINN(_cell(), (1.0, 10.0), 5.0, 0.0128, 293.15, settings=s)
    assert p.t_end(10.0) == pytest.approx(360.0) and p.t_end(1.0) == pytest.approx(3600.0)
    assert p.c_from_hat(-1.0) == 1.0 and p.c_from_hat(1.0) == 10.0
    # the numpy and torch temperature scales agree, and match cell_thermal's rule
    from autocfd.cell_thermal import temperature_scale
    for c in (1.0, 4.0, 10.0):
        expected = temperature_scale(_cell(), p.load(c))
        assert float(p.temperature_scale(c)) == pytest.approx(expected, rel=1e-6)
        assert float(p.temperature_scale(torch.tensor([c]))) == pytest.approx(expected, rel=1e-5)
    pts = np.array([[0.0, 0.035, 0.0]])
    assert np.allclose(p.predict(pts, [0.0], 3.0), 293.15)
    hist = p.train()
    assert hist[-1]["loss"] < hist[0]["loss"]
    t_end = p.predict_t_max_end([1.0, 10.0], pts)
    assert np.all(np.isfinite(t_end))


def test_pinn_figures(tmp_path):
    from autocfd import pinn_plots
    from autocfd.cell_thermal import FVSolver

    cell = _cell()
    load = Load.constant(10.0, 60.0, 5.0, 0.0128)
    fv = FVSolver(cell, nr=6, nphi=12, ny=6)
    series = fv.transient(load, 293.15, dt_s=6.0)
    # stand-in "PINN": the lumped heating, enough to exercise the plots
    predict = lambda pts, times: np.array([np.full(len(pts), 293.15 + 0.2 * t) for t in times])  # noqa: E731
    import pandas as pd
    comp = pd.DataFrame({"time_s": series["time_s"], "t_max_fv_c": series["t_max_k"] - 273.15,
                         "t_max_pinn_c": series["t_max_k"] - 273.15, "t_mean_fv_c": series["t_mean_k"] - 273.15,
                         "t_mean_pinn_c": series["t_mean_k"] - 273.15})
    pinn_plots.comparison(comp, fv, fv.last_field.ravel(), cell, tmp_path / "a.png")
    pinn_plots.sections_over_time(predict, cell, 60.0, "t", tmp_path / "b.png")
    pinn_plots.axial_section(predict, fv, cell, 60.0, tmp_path / "c.png")
    pinn_plots.surface_map(predict, cell, 60.0, tmp_path / "d.png")
    pinn_plots.training([{"iteration": i, "loss": 1 / i, "pde": 1 / i, "side": 1 / i, "ends": 1 / i,
                          "w_pde": 1, "w_side": 2, "w_ends": 3} for i in (1, 2, 3)], "t", tmp_path / "e.png")
    for f in "abcde":
        assert (tmp_path / f"{f}.png").stat().st_size > 5000


def test_save_and_reload_pinns(tmp_path):
    from autocfd.pinn import (CellPINN, ParametricCellPINN, PINNSettings, load_cell_pinn,
                              load_parametric_pinn)

    s = PINNSettings(width=16, depth=2, iterations=5, n_interior=64, n_side=32, n_ends=16, device="cpu")
    load = Load.constant(5.0, 720.0, 5.0, 0.0128)
    pts = np.array([[0.005, 0.03, -0.002]])
    one = CellPINN(_cell(), load, 293.15, s)
    one.train()
    one.save(tmp_path / "one.pt")
    assert np.allclose(load_cell_pinn(tmp_path / "one.pt", _cell(), load).predict(pts, [300.0]),
                       one.predict(pts, [300.0]))
    par = ParametricCellPINN(_cell(), (1.0, 10.0), 5.0, 0.0128, 293.15, settings=s)
    par.train()
    par.save(tmp_path / "par.pt")
    back = load_parametric_pinn(tmp_path / "par.pt", _cell())
    assert (back.c_min, back.c_max) == (1.0, 10.0)
    assert np.allclose(back.predict(pts, [100.0], 4.0), par.predict(pts, [100.0], 4.0))
