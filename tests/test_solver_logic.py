import numpy as np
import pytest

from autocfd.config import load_config
from autocfd.solver import block_converged, derived_outputs, window_average
from tests.test_config import CONFIG


def _samples(values):
    return [{"p": v, "t": 300.0} for v in values]


def test_oscillating_signal_converges_on_average():
    # +-1 % oscillation around a constant mean: instantaneous values never settle
    sig = 34.4 * (1 + 0.01 * np.sin(np.arange(40) * 1.3))
    ok, change = block_converged(_samples(sig), ["p", "t"], per_block=5, rel_tol=5e-3)
    assert ok, change


def test_drifting_signal_not_converged():
    sig = np.linspace(30, 40, 40)
    ok, change = block_converged(_samples(sig), ["p"], per_block=5, rel_tol=2e-3)
    assert not ok and change["p"] > 2e-3


def test_too_few_samples():
    ok, _ = block_converged(_samples([1, 2, 3]), ["p"], per_block=5, rel_tol=1.0)
    assert not ok


def test_window_average():
    mean, osc = window_average(_samples([0, 0, 10, 20]), 2)
    assert mean["p"] == 15 and osc["t"] == 0


def test_derived_outputs():
    cfg = load_config(CONFIG)
    rep = {"t_max": 310.0, "t_min": 300.0, "t_avg": 305.0, "t_out": 299.0,
           "p_in": 30.0, "p_out": 0.0, "m_out": 0.01}
    out = derived_outputs(cfg, rep)
    assert out["dP_Pa"] == 30.0 and out["dT_cells_K"] == 10.0
    # reports come in K from PyFluent: the *_C outputs must be Celsius
    assert out["T_max_C"] == pytest.approx(36.85) and out["T_out_C"] == pytest.approx(25.85)
