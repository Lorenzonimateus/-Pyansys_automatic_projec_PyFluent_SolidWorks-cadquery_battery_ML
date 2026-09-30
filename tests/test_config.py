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


def test_single_value_study_parameter_is_fixed(tmp_path):
    text = CONFIG.read_text(encoding="utf-8")
    start = text.index("[study.parameters]")
    end = text.index("[optimize]")
    params = '[study.parameters]\n"cad.D" = [22.0]\n"cad.P1" = [6.0, 14.0]\n\n'
    path = tmp_path / "c.toml"
    path.write_text(text[:start] + params + text[end:], encoding="utf-8")
    cfg = load_config(path)
    assert cfg.cad["D"] == 22.0
    assert list(cfg.study.parameters) == ["cad.P1"]


def test_c_rate_drives_fluent_heat_source_in_w_per_m3():
    cfg = load_config(CONFIG)
    assert cfg.operating.heat_source == "parameter_heat_source_wperm3"
    p = cfg.with_overrides({"operating.c_rate": 10.0, "cad.D": 22.0})
    q_cell = (10.0 * 5.0) ** 2 * 0.0128                        # 32 W
    vol = 3.141592653589793 * 0.011 ** 2 * 0.07015
    assert p.heat_inputs()["Q_cell_W"] == pytest.approx(q_cell)
    assert p.fluent["parameter_heat_source_wperm3"] == pytest.approx(q_cell / vol)
    assert p.fluent_expression("parameter_heat_source_wperm3").endswith("[W/m^3]")
    # the diameter changes the volume, so the W/m^3 at the same C-rate
    bigger = p.with_overrides({"cad.D": 24.0})
    assert bigger.fluent["parameter_heat_source_wperm3"] < p.fluent["parameter_heat_source_wperm3"]
    # C-rate does not change the geometry (the mesh is reused)
    assert p.geometry_key() == p.with_overrides({"operating.c_rate": 3.0}).geometry_key()


def test_heat_source_cannot_be_a_study_parameter(tmp_path):
    text = CONFIG.read_text(encoding="utf-8")
    start, end = text.index("[study.parameters]"), text.index("[optimize]")
    params = '[study.parameters]\n"fluent.parameter_heat_source_wperm3" = [1e4, 1e6]\n\n'
    path = tmp_path / "c.toml"
    path.write_text(text[:start] + params + text[end:], encoding="utf-8")
    with pytest.raises(ValueError, match="c_rate"):
        load_config(path)


def test_study_of_the_40_point_doe():
    cfg = load_config(CONFIG)
    assert cfg.study.samples == 40
    assert cfg.cad["D"] == 22.0 and "cad.D" not in cfg.study.parameters
    assert set(cfg.study.parameters) == {"cad.L1", "cad.P1", "operating.c_rate",
                                         "fluent.parameter_velocity_inlet"}


def test_is_geometric():
    assert is_geometric("cad.D") and is_geometric("mesh.max_size_mm")
    assert not is_geometric("fluent.parameter_velocity_inlet")


def test_lhs_plan_rounded_to_one_decimal():
    from autocfd.doe import latin_hypercube

    pts = latin_hypercube({"a": (2.0, 10.0), "b": (1.0, 7.0)}, 40, seed=42)
    for p in pts:
        for k, (lo, hi) in {"a": (2.0, 10.0), "b": (1.0, 7.0)}.items():
            assert p[k] == round(p[k], 1) and lo <= p[k] <= hi
