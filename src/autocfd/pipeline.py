"""Chains SolidWorks -> SpaceClaim -> Fluent Meshing -> Fluent, caching per geometry.

output/cases/<key>/  geom_<key>.SLDPRT, geometry.scdocx, mesh.msh.h5, params.json
The <key> is a hash of [cad] + [mesh] (Config.geometry_key): the geometry and
mesh are only rebuilt when something that affects them changes; changing only
[fluent] values reuses them.
"""

from __future__ import annotations

import csv
import json
import logging
import shutil
from pathlib import Path

from .config import Config

log = logging.getLogger(__name__)


def ensure_geometry(cfg: Config) -> Path:
    """SolidWorks + SpaceClaim: return the prepared .scdocx for this geometry."""
    from . import cad, spaceclaim

    case_dir = cfg.case_dir()
    scdocx = case_dir / "geometry.scdocx"
    if scdocx.exists():
        return scdocx
    sldprt = cad.build(cfg, case_dir)
    spaceclaim.prepare(cfg, sldprt, scdocx)
    (case_dir / "params.json").write_text(json.dumps(cfg.geometry_params(), indent=2))
    return scdocx


def ensure_mesh(cfg: Config) -> Path:
    """Return the mesh for this geometry, building it if needed."""
    from . import meshing

    mesh = cfg.case_dir() / "mesh.msh.h5"
    if mesh.exists():
        log.info("Reusing existing mesh: %s", mesh)
        return mesh
    scdocx = ensure_geometry(cfg)
    meshing.build_mesh(cfg, scdocx, mesh)
    for tmp in mesh.parent.glob("FM_*"):  # Fluent Meshing scratch folders
        shutil.rmtree(tmp, ignore_errors=True)
    return mesh


def append_csv(path: Path, row: dict) -> None:
    """Append a row; if new columns appear, the file is rewritten with the union."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows, fields = [], list(row)
    if path.exists():
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            fields = list(reader.fieldnames or []) + [k for k in row if k not in (reader.fieldnames or [])]
    rows.append(row)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def run_single(cfg: Config, use_base_mesh: bool = False) -> dict:
    """One point with the values in the config; appends to output/single_runs.csv.

    use_base_mesh=True solves the base case as it is (no SolidWorks/SpaceClaim/meshing),
    useful to check the solver settings."""
    from .postprocess import point_params
    from .solver import FluentRunner

    mesh = None if use_base_mesh else ensure_mesh(cfg)
    work = cfg.output_dir / "single"
    tag = "base" if use_base_mesh else cfg.geometry_key()
    tag += "_" + "_".join(f"{v:g}" for v in cfg.fluent.values())
    with FluentRunner(cfg, mesh, work) as f:
        result = f.run(cfg, save_as=work / f"{tag}.cas.h5", images_dir=work / "images" / tag,
                       fields_dir=work / "ml_fields", name=tag)
    append_csv(cfg.output_dir / "single_runs.csv",
               {"geometry": tag.split("_")[0], **point_params(cfg), **result})
    return result
