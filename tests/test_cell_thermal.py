import numpy as np
import pytest

from autocfd.cell_thermal import (CellModel, FVSolver, Load, interp_map, load_cells, save_cells,
                                  temperature_scale)

R, H = 0.011, 0.07015


def _cell(kx=0.8, ky=25.0, kz=0.8, h=50.0, nphi=36, ny=15, h_fn=None):
    phi = (np.arange(nphi) + 0.5) * 2 * np.pi / nphi
    y = np.linspace(0, H, ny)
    hmap = np.full((nphi, ny), h) if h_fn is None else h_fn(phi[:, None], y[None, :])
    return CellModel("cell_t", R, H, kx, ky, kz, 2775.0, 1050.0, phi, y, hmap, t_ref_k=293.15)


def test_steady_matches_analytical_radial_solution():
    # isotropic, uniform h, adiabatic ends: T(r) = Tref + qR/(2h) + q (R^2 - r^2) / (4k)
    k, h, Q = 0.8, 50.0, 10.0
    cell = _cell(kx=k, ky=k, kz=k, h=h)
    q = Q / cell.volume_m3
    fv = FVSolver(cell, nr=30, nphi=24, ny=6)
    T = fv.steady(Q)
    exact = cell.t_ref_k + q * R / (2 * h) + q * (R ** 2 - fv.r ** 2) / (4 * k)
    assert np.allclose(T.mean(axis=(1, 2)), exact, rtol=0, atol=0.02 * (exact[0] - cell.t_ref_k))
    assert fv.heat_out_w(T.ravel()) == pytest.approx(Q, rel=1e-9)   # energy conserved exactly


def test_steady_conserves_energy_with_anisotropy_and_varying_h():
    cell = _cell(kx=0.8, ky=0.8, kz=25.0, h_fn=lambda p, y: 30 + 40 * (1 + np.cos(p)) * (1 + y / H))
    fv = FVSolver(cell, nr=16, nphi=32, ny=10)
    T = fv.steady(32.0)
    assert fv.heat_out_w(T.ravel()) == pytest.approx(32.0, rel=1e-9)
    # strong conduction along z: the temperature varies less along z than along x
    mid = T[:, :, 5]
    x_spread = abs(mid[-1, 0] - mid[-1, 16])         # phi = 0 vs pi   (x direction)
    z_spread = abs(mid[-1, 8] - mid[-1, 24])         # phi = pi/2 vs 3pi/2 (z direction)
    assert z_spread < x_spread


def test_transient_lumped_limit():
    # very conductive cell (Bi << 1): T = Tref + Q/(hA) (1 - exp(-t/tau))
    cell = _cell(kx=1e4, ky=1e4, kz=1e4, h=20.0)
    load = Load.constant(5.0, 600.0, capacity_ah=5.0, r_dc_ohm=0.0128)
    Q = load.peak_heat_w()
    fv = FVSolver(cell, nr=6, nphi=12, ny=4)
    s = fv.transient(load, t0_k=cell.t_ref_k, dt_s=1.0)
    tau = cell.rho * cell.cp * cell.volume_m3 / (20.0 * cell.side_area_m2)
    exact = cell.t_ref_k + Q / (20.0 * cell.side_area_m2) * (1 - np.exp(-s["time_s"] / tau))
    assert np.allclose(s["t_mean_k"], exact, atol=0.01 * (exact[-1] - cell.t_ref_k))


def test_load_profiles():
    load = Load.constant(10.0, 360.0, capacity_ah=5.0, r_dc_ohm=0.0128)
    assert load.heat_w(100.0) == pytest.approx(32.0)
    ramp = Load(np.array([0.0, 100.0]), np.array([0.0, 10.0]), 5.0, 0.0128)
    assert ramp.heat_w(50.0) == pytest.approx(8.0)       # (5 * 5)^2 * 0.0128


def test_interp_map_periodic_and_save_load(tmp_path):
    cell = _cell(h_fn=lambda p, y: 10 + np.cos(p) + 0 * y)
    # periodic: values just below 2 pi and just above 0 agree with cos
    v = interp_map(cell.phi, cell.y, cell.h, np.array([0.0, 2 * np.pi - 1e-9]), np.array([0.01, 0.01]))
    assert np.allclose(v, 11.0, atol=0.01)
    save_cells(tmp_path / "c.npz", [cell], {"case": "x"})
    back = load_cells(tmp_path / "c.npz")[0]
    assert back.name == "cell_t" and np.allclose(back.h, cell.h) and back.meta["case"] == "x"
    assert temperature_scale(cell, Load.constant(10, 360, 5.0, 0.0128)) > 1
