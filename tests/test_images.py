import numpy as np
import pytest

from autocfd.config import load_config
from autocfd.images import FIELDS, KELVIN, PLANES, plot_plane, triangulate
from tests.test_config import CONFIG


def test_triangulate_polygons():
    tris = triangulate([[0, 1, 2], [2, 3, 4, 5], [5, 6, 7, 8, 9]])
    assert tris.shape == (1 + 2 + 3, 3)
    assert tris[1].tolist() == [2, 3, 4]


def test_temperature_is_plotted_in_celsius():
    assert FIELDS["temperature"].unit == "°C"
    assert FIELDS["temperature"].convert(np.array([KELVIN + 25.0]))[0] == pytest.approx(25.0)
    assert not FIELDS["temperature"].fluid_only          # cells are shown
    assert FIELDS["pressure"].fluid_only and FIELDS["velocity"].fluid_only


def test_planes_cover_the_three_directions():
    assert {n for n, _, _ in PLANES.values()} == {0, 1, 2}
    for normal, h, v in PLANES.values():
        assert len({normal, h, v}) == 3


@pytest.mark.parametrize("field", ["temperature", "pressure", "velocity"])
def test_plot_plane_writes_png(tmp_path, field):
    # unit square in the xz plane (y = const), two triangles
    verts = np.array([[0, 0.5, 0], [0.1, 0.5, 0], [0.1, 0.5, -0.1], [0, 0.5, -0.1]], float)
    tris = np.array([[0, 1, 2], [0, 2, 3]])
    values = np.array([300.0, 301.0, 302.0, 299.0]) if field == "temperature" else np.array([-5.0, 3, 8, 0])
    path = tmp_path / f"{field}_xz.png"
    plot_plane(verts, tris, values, "xz", FIELDS[field], 500.0, "test", path)
    assert path.exists() and path.stat().st_size > 1000


def test_images_config():
    cfg = load_config(CONFIG)
    assert set(cfg.images.fields) == {"temperature", "pressure", "velocity"}
    assert set(cfg.images.planes) == {"xy", "yz", "xz"}
