import dataclasses

import pandas as pd
import pytest

from autocfd import doe, meshing, pipeline
from autocfd.config import load_config
from tests.test_config import CONFIG


def _cfg(tmp_path):
    cfg = load_config(CONFIG)
    return dataclasses.replace(
        cfg, paths=dataclasses.replace(cfg.paths, output_dir=str(tmp_path)),
        mesh=dataclasses.replace(cfg.mesh, attempts=3, retry_wait_s=0))


def test_meshing_is_retried_after_a_transient_cad_import_failure(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(pipeline, "ensure_geometry", lambda c: tmp_path / "geometry.scdocx")
    calls = []

    def flaky(c, scdocx, mesh):
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("Exception encountered in S_ExecuteUpstreamNonExecutedAndThisTask: Error in CAD Import")
        mesh.parent.mkdir(parents=True, exist_ok=True)
        mesh.write_text("mesh")

    monkeypatch.setattr(meshing, "build_mesh", flaky)
    assert pipeline.ensure_mesh(cfg).exists()
    assert len(calls) == 3


def test_mesh_error_after_all_attempts(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(pipeline, "ensure_geometry", lambda c: tmp_path / "geometry.scdocx")
    monkeypatch.setattr(meshing, "build_mesh", lambda *a: (_ for _ in ()).throw(RuntimeError("CAD Import")))
    with pytest.raises(pipeline.MeshError, match="after 3 attempts"):
        pipeline.ensure_mesh(cfg)


def test_geometry_error_is_not_retried(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(pipeline, "ensure_geometry",
                        lambda c: (_ for _ in ()).throw(RuntimeError("SolidWorks rebuild has 2 error(s)")))
    with pytest.raises(pipeline.GeometryError):
        pipeline.ensure_mesh(cfg)


def test_study_stops_after_consecutive_identical_failures(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(doe, "ensure_mesh", lambda c: (_ for _ in ()).throw(
        pipeline.MeshError("after 3 attempts: Error in CAD Import")))
    points = [{"id": i, "cad.L1": 2.0 + i, "cad.P1": 5.0} for i in range(6)]
    results = tmp_path / "results.csv"
    with pytest.raises(RuntimeError, match="in a row"):
        doe.run_points(cfg, points, results)
    df = pd.read_csv(results)
    assert len(df) == 3 and set(df["status"]) == {"mesh_failed"}   # stopped after 3, not 6


def test_retry_failed_drops_failed_rows_and_keeps_a_copy(tmp_path):
    results = tmp_path / "results.csv"
    pd.DataFrame({"id": [0, 1, 2], "status": ["ok", "mesh_failed", "geometry_failed"]}).to_csv(results, index=False)
    assert doe.drop_failed(results) == [1, 2]
    assert pd.read_csv(results)["id"].tolist() == [0]
    backups = list(tmp_path.glob("results_failed_*.csv"))
    assert len(backups) == 1 and len(pd.read_csv(backups[0])) == 2
