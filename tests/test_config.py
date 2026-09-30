from pathlib import Path

import pytest

from autocfd.config import is_geometric, load_config

CONFIG = Path(__file__).parents[1] / "configs" / "default.toml"


def test_load_default():
    cfg = load_config(CONFIG)
    assert "D" in cfg.cad
    assert "parameter_velocity_inlet" in cfg.fluent
    assert set(cfg.solver.monitor) <= set(cfg.solver.reports)


def test_overrides_and_geometry_key():
    cfg = load_config(CONFIG)
    same_geo = cfg.with_overrides({"fluent.parameter_velocity_inlet": 7.0})
    new_geo = cfg.with_overrides({"cad.D": 19.0})
    new_mesh = cfg.with_overrides({"mesh.bl_layers": 8})
    assert same_geo.fluent["parameter_velocity_inlet"] == 7.0
    assert same_geo.geometry_key() == cfg.geometry_key()
    assert new_geo.geometry_key() != cfg.geometry_key()
    assert new_mesh.geometry_key() != cfg.geometry_key()
    # processors do not change the mesh
    assert cfg.with_overrides({"mesh.processors": 4}).geometry_key() == cfg.geometry_key()
    assert isinstance(new_mesh.mesh.bl_layers, int)


def test_unknown_parameter_rejected():
    cfg = load_config(CONFIG)
    with pytest.raises(KeyError):
        cfg.with_overrides({"cad.DOES_NOT_EXIST": 1.0})
    with pytest.raises(KeyError):
        cfg.with_overrides({"nosection.x": 1.0})


def test_fluent_expression_has_unit():
    cfg = load_config(CONFIG).with_overrides({"fluent.parameter_velocity_inlet": 4.5})
    assert cfg.fluent_expression("parameter_velocity_inlet") == "4.5 [m/s]"


def test_is_geometric():
    assert is_geometric("cad.D") and is_geometric("mesh.max_size_mm")
    assert not is_geometric("fluent.parameter_velocity_inlet")
