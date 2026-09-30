from pathlib import Path

import pytest

from autocfd.config import load_config
from autocfd.meshing import task_updates, workflow_tasks
from tests.test_config import CONFIG

ROOT = Path(__file__).parents[1]
cfg = load_config(CONFIG)


@pytest.mark.skipif(not (ROOT / cfg.paths.mesh_workflow).exists(), reason="workflow file not present")
def test_workflow_tasks_and_updates():
    tasks = workflow_tasks(ROOT / cfg.paths.mesh_workflow)
    assert "Import Geometry" in tasks and "Generate the Volume Mesh" in tasks
    assert "Enclose Fluid Regions (Capping)" not in tasks  # inactive task left out
    up = task_updates(cfg, Path("geo.scdocx"), tasks)
    assert up["Import Geometry"]["FileName"].endswith("geo.scdocx")
    assert up["Generate the Surface Mesh"]["CFDSurfaceMeshControls"]["MaxSize"] == cfg.mesh.max_size_mm


@pytest.mark.skipif(not (ROOT / cfg.paths.base_case).exists(), reason="base case not present")
def test_case_info_matches_config():
    from autocfd.inspect_case import case_info

    info = case_info(ROOT / cfg.paths.base_case)
    names = {e["name"] for e in info["named_expressions"]}
    assert set(cfg.fluent) <= names
    assert set(cfg.solver.reports) <= set(info["report_definitions"])
