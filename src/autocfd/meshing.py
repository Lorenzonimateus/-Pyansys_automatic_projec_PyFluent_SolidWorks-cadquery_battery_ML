"""Fluent Meshing stage: replay the saved watertight workflow (.wft) on a new .scdocx.

The .wft is the meshing "recipe" set up once by hand in the GUI (sizes, share
topology, boundary types, prism layers, poly-hexcore). Here it is loaded, the
geometry file and the [mesh] values are replaced, and every task up to the
volume mesh is executed. Fluent is ALWAYS closed at the end (license released).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .config import Config

log = logging.getLogger(__name__)


def launch(cfg: Config, mode: str, processors: int, cwd: Path):
    import ansys.fluent.core as pyfluent

    extra = f"-mpi={cfg.solver.mpi}" if cfg.solver.mpi else ""
    Path(cwd).mkdir(parents=True, exist_ok=True)
    return pyfluent.launch_fluent(
        product_version=pyfluent.FluentVersion.v252,
        mode=mode,
        precision="double",
        processor_count=processors,
        ui_mode="no_gui_or_graphics",
        start_timeout=cfg.solver.start_timeout_s,
        additional_arguments=extra,
        cwd=str(Path(cwd).resolve()),
    )


def workflow_tasks(wft: Path) -> dict[str, dict]:
    """{task display name: task} of the ACTIVE tasks of the saved workflow, in order
    (read without Fluent; skipped tasks such as capping are left out)."""
    root = json.loads(Path(wft).read_text())["workflow"]["ROOT"]
    objs = {k.split(":", 1)[1]: v for k, v in root.items() if k.startswith("TaskObject:")}
    out = {}
    for tid in root["Workflow"]["TaskList"]:
        task = objs[tid]
        out[task["_name_"]] = task
        children = task["TaskList"] if isinstance(task["TaskList"], list) else []
        for cid in children:
            if cid not in (task["InactiveTaskList"] or []):
                out[objs[cid]["_name_"]] = objs[cid]
    return out


def _find(tasks: dict[str, dict], command: str, child: bool = False) -> str:
    """Name of the task running a command (the compound child for boundary layers)."""
    for name, t in tasks.items():
        if t["CommandName"] == command and (t["TaskType"] == "Compound Child") == child:
            return name
    raise KeyError(f"Task {command!r} not found in the workflow")


def task_updates(cfg: Config, scdocx: Path, tasks: dict[str, dict]) -> dict[str, dict]:
    """Argument changes applied to the loaded workflow, per task name."""
    m = cfg.mesh
    updates = {
        _find(tasks, "ImportGeometry"): {"FileName": str(scdocx.resolve())},
        _find(tasks, "GenerateTheSurfaceMeshWTM"): {
            "CFDSurfaceMeshControls": {"MinSize": m.min_size_mm, "MaxSize": m.max_size_mm}},
        _find(tasks, "GenerateTheVolumeMeshWTM"): {
            "VolumeFillControls": {"HexMaxSize": m.hex_max_size_mm}},
    }
    try:
        updates[_find(tasks, "AddBoundaryLayers", child=True)] = {"NumberOfLayers": m.bl_layers}
    except KeyError:
        log.warning("The workflow has no boundary-layer control; mesh.bl_layers is ignored")
    return updates


def build_mesh(cfg: Config, scdocx: Path, mesh_file: Path) -> Path:
    wft = Path(cfg.paths.mesh_workflow)
    tasks = workflow_tasks(wft)
    updates = task_updates(cfg, scdocx, tasks)
    volume_task = _find(tasks, "GenerateTheVolumeMeshWTM")

    mesh_file.parent.mkdir(parents=True, exist_ok=True)
    session = launch(cfg, "meshing", cfg.mesh.processors, mesh_file.parent)
    try:
        wf = session.workflow
        wf.LoadWorkflow(FilePath=str(wft.resolve()))
        for name, args in updates.items():
            wf.TaskObject[name].Arguments.update_dict(args)
            log.info("Workflow: %s <- %s", name, args)
        # The command's return value is False even on success (v252): check the task states
        wf.TaskObject[volume_task].ExecuteUpstreamNonExecutedAndThisTask()
        states = {n: wf.TaskObject[n].State() for n in tasks}
        bad = {n: s for n, s in states.items() if s not in ("Up-to-date", "Forced-up-to-date")}
        if bad:
            raise RuntimeError(f"Meshing workflow did not complete; task states: {states}")
        session.tui.file.write_mesh(str(mesh_file.resolve()))
        log.info("Mesh written to %s", mesh_file)
        return mesh_file
    finally:
        try:
            session.exit()
            log.info("Fluent Meshing session closed, license released.")
        except Exception as e:  # noqa: BLE001
            log.warning("Failed to close the Fluent Meshing session: %s", e)
