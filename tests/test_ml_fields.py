import json

import numpy as np
import pytest
from matplotlib.tri import Triangulation

from autocfd.config import MLFields, load_config
from autocfd.images import KELVIN
from autocfd.ml_fields import (grid_axes, inside, interpolate, load_dataset, normalize, plane_grid,
                               signed_distance, write_normalization, write_plane)
from tests.test_config import CONFIG

# xz plane of a 100 x 50 mm duct (y = 0.035 m) with a 20 mm "cell" at (50, -25) mm
ML = MLFields(planes=("xz",), x_mm=(0.0, 100.0), y_mm=(0.0, 70.0), z_mm=(-50.0, 0.0),
              resolution=(64, 16, 32),
              normalization={"temperature": (20.0, 60.0), "pressure": (-10.0, 10.0), "velocity": (0.0, 5.0)})


def _plane(with_hole: bool):
    xs, zs = np.meshgrid(np.linspace(0, 100, 51), np.linspace(-50, 0, 26))
    pts = np.column_stack([xs.ravel(), zs.ravel()])
    tri = Triangulation(pts[:, 0], pts[:, 1])
    tris = tri.triangles
    if with_hole:
        c = pts[tris].mean(axis=1)
        tris = tris[np.hypot(c[:, 0] - 50, c[:, 1] + 25) > 10]
    verts = np.column_stack([pts[:, 0], np.full(len(pts), 35.0), pts[:, 1]]) / 1000  # metres
    return verts, tris


def test_grid_is_fixed_by_config():
    h, v = plane_grid("xz", grid_axes(ML))
    assert (len(h), len(v)) == (64, 32)
    assert h[0] == 0 and h[-1] == 100 and v[0] == -50 and v[-1] == 0


def test_interpolation_is_exact_for_linear_fields_and_nan_in_holes():
    verts, tris = _plane(with_hole=True)
    h, v = plane_grid("xz", grid_axes(ML))
    field = 2.0 * verts[:, 0] * 1000 + 1.0  # linear in x (mm)
    out = interpolate(verts, tris, field, "xz", h, v)
    hg, _ = np.meshgrid(h, v)
    ok = np.isfinite(out)
    assert np.allclose(out[ok], 2.0 * hg[ok] + 1.0)
    assert np.isnan(out[np.argmin(abs(v + 25)), np.argmin(abs(h - 50))])  # centre of the cell


def test_mask_and_signed_distance():
    verts, tris = _plane(with_hole=True)
    h, v = plane_grid("xz", grid_axes(ML))
    mask = inside(verts, tris, "xz", h, v)
    i, j = np.argmin(abs(v + 25)), np.argmin(abs(h - 50))
    assert mask[i, j] == 0 and mask[0, 0] == 1
    sdf = signed_distance(mask, h[1] - h[0], v[1] - v[0])
    assert sdf[i, j] < -5 and sdf[i, 5] > 5    # inside the cell / in the fluid, in mm
    assert (np.sign(sdf[mask > 0.5]) >= 0).all()


def test_normalize_fixed_limits():
    a = np.array([20.0, 40.0, 60.0, np.nan])
    assert normalize(a, 20, 60).tolist() == [0.0, 0.5, 1.0, 0.0]
    assert normalize(a, 20, 60).dtype == np.float32


def test_write_plane_and_load_dataset(tmp_path):
    fluid = _plane(with_hole=True)
    solid_and_fluid = _plane(with_hole=False)
    temp_k = KELVIN + 20.0 + 0.1 * solid_and_fluid[0][:, 0] * 1000   # 20..30 °C along x
    fields = {"temperature": (*solid_and_fluid, temp_k),
              "pressure": (*fluid, np.full(len(fluid[0]), 3.0)),
              "velocity": (*fluid, np.full(len(fluid[0]), 2.5))}
    for i, d in enumerate((20.0, 21.0)):
        params = {"cad.D": d, "fluent.parameter_velocity_inlet": 2.5}
        write_plane(tmp_path / "xz" / f"dp{i:03d}.npz", "xz", 35.0, fields, fluid, ML, params)
    write_normalization(tmp_path, ML, list(params))

    f = np.load(tmp_path / "xz" / "dp000.npz")
    assert f["T"].dtype == np.float32 and f["T"].shape == (32, 64)
    assert np.nanmin(f["T"]) == pytest.approx(20.0, abs=1e-4)          # °C, not K
    assert f["T_limits"].tolist() == [20.0, 60.0]
    assert np.isnan(f["p"][f["mask"] < 0.5]).all()                     # p only in the fluid
    assert np.isfinite(f["T"][f["mask"] < 0.5]).any()                  # T also in the cell
    assert f["U_norm"][f["mask"] > 0.5] == pytest.approx(0.5)
    assert list(f["param_names"]) == ["cad.D", "fluent.parameter_velocity_inlet"]

    ds = load_dataset(tmp_path, "xz", targets=("T_norm", "U_norm"))
    assert ds["geometry"].shape == (2, 2, 32, 64)
    assert ds["targets"].shape == (2, 2, 32, 64)
    assert ds["params"][:, 0].tolist() == [20.0, 21.0]
    info = json.loads((tmp_path / "normalization.json").read_text())
    assert info["normalization"]["T"] == {"min": 20.0, "max": 60.0, "unit": "C"}


def test_default_config_ml_fields():
    ml = load_config(CONFIG).ml_fields
    assert ml.resolution == (256, 64, 128)
    assert ml.normalization["temperature"] == (20.0, 250.0)


def test_rasterize_tolerates_duplicates_and_degenerate_triangles():
    from autocfd.ml_fields import rasterize

    px = np.array([0.0, 10.0, 0.0, 10.0, 0.0, 5.0])
    py = np.array([0.0, 0.0, 10.0, 10.0, 0.0, 0.0])
    tris = np.array([[0, 1, 2], [1, 3, 2], [0, 1, 2], [0, 4, 5]])   # duplicate + degenerate
    vals = px + 2 * py
    h = v = np.linspace(0, 10, 11)
    out = rasterize(px, py, tris, vals, h, v)
    hg, vg = np.meshgrid(h, v)
    assert np.isfinite(out).all() and np.allclose(out, hg + 2 * vg)
