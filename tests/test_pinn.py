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
