from pathlib import Path

import pytest

from autocfd import cadquery_geometry as cqg
from autocfd.config import load_config

CONFIG = Path(__file__).parents[1] / "configs" / "default.toml"


def nominal(**cad):
    values = {"D": 21.55, "L1": 10.0, "P1": 10.0, **cad}
    return load_config(CONFIG).with_overrides({f"cad.{k}": v for k, v in values.items()})


def test_layout_matches_the_solidworks_part():
    # cell axes and duct height read from geometria_parametrizada.SLDPRT (D=21.55, L1=P1=10)
    cfg = nominal()
    c = cqg.cell_centres(cfg)
    assert c["cell_1"] == pytest.approx((100.0, -15.775))
    assert c["cell_3"] == pytest.approx((163.1, -15.775))
    assert c["cell_4"] == pytest.approx((163.1, -47.325))
    assert c["cell_9"] == pytest.approx((100.0, -78.875))
    assert cqg.duct_height(cfg) == pytest.approx(99.65)


def test_layout_follows_the_variables():
    # SolidWorks with D=20, L1=12, P1=8: columns 28 mm apart, rows 32 mm, bottom wall L1 below
    cfg = nominal(D=20.0, L1=12.0, P1=8.0)
    c = cqg.cell_centres(cfg)
    assert c["cell_2"][0] - c["cell_1"][0] == pytest.approx(28.0)
    assert c["cell_1"][1] - c["cell_6"][1] == pytest.approx(32.0)
    assert cqg.duct_height(cfg) == pytest.approx(101.775)


def test_body_boxes_have_every_body():
    cfg = nominal()
    boxes = cqg.body_boxes(cfg)
    assert list(boxes) == ["volume", *(f"cell_{n}" for n in range(1, 10))]
    assert boxes["volume"] == pytest.approx([0.0, 0.0, -99.65, 309.0967, 70.15, 0.0])
    assert boxes["cell_1"] == pytest.approx([89.225, 0.0, -26.55, 110.775, 70.15, -5.0])


@pytest.mark.parametrize("cad", [{"P1": 0.0}, {"L1": -1.0}, {"D": 32.0}, {"P1": 90.0}])
def test_invalid_combinations_rejected(cad):
    # touching columns or rows, a cell through the top wall, the last column past the outlet
    with pytest.raises(ValueError):
        cqg.check(nominal(**cad))


def test_engine_changes_only_the_cadquery_key():
    cfg = nominal()
    assert "geometry" not in cfg.geometry_params()   # SolidWorks keys stay as before
    cq_cfg = cfg.with_overrides({"geometry.engine": "cadquery"})
    assert cq_cfg.geometry_key() != cfg.geometry_key()
    assert cq_cfg.with_overrides({"geometry.duct_length_mm": 300.0}).geometry_key() != cq_cfg.geometry_key()


def test_unknown_engine_rejected(tmp_path):
    text = CONFIG.read_text(encoding="utf-8").replace('engine = "solidworks"', 'engine = "catia"')
    path = tmp_path / "c.toml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)


def test_step_has_one_named_solid_per_body(tmp_path):
    pytest.importorskip("cadquery")
    cfg = nominal(D=20.0, L1=12.0, P1=8.0)
    step = cqg.build(cfg, tmp_path)
    text = step.read_text()
    for name in cqg.body_boxes(cfg):
        assert f"MANIFOLD_SOLID_BREP('{name}'" in text
    assert text.count("PRODUCT(") == 1   # one part, not an assembly
