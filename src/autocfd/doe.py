"""Design of experiments and batch execution of design points.

- Latin Hypercube sampling (scipy.stats.qmc) over [study.parameters].
- Points are grouped by geometry: each geometry is built once (SolidWorks ->
  SpaceClaim -> Fluent Meshing, cached) and a single Fluent session runs all the
  points on that mesh, each starting from the previous solution.
- Resumable: results.csv stores each point as soon as it finishes; running again
  only computes what is missing. Points whose geometry fails (invalid parameter
  combination in SolidWorks, meshing failure) are recorded with their status and
  skipped, they do not stop the study.
"""

from __future__ import annotations

import csv
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import qmc

from .config import Config
from .pipeline import GeometryError, append_csv, ensure_mesh

log = logging.getLogger(__name__)


def study_dir(cfg: Config) -> Path:
    return cfg.output_dir / "studies" / cfg.study.name


def latin_hypercube(bounds: dict[str, tuple[float, float]], n: int, seed: int) -> list[dict[str, float]]:
    names = list(bounds)
    lo = np.array([bounds[k][0] for k in names])
    hi = np.array([bounds[k][1] for k in names])
    unit = qmc.LatinHypercube(d=len(names), seed=seed).random(n)
    pts = np.round(qmc.scale(unit, lo, hi), 1)  # 1 decimal place: readable plan, CAD-friendly values
    return [dict(zip(names, map(float, row))) for row in pts]


def load_or_create_design(cfg: Config) -> list[dict]:
    """Read design.csv if it exists (resume), otherwise sample and save it."""
    path = study_dir(cfg) / "design.csv"
    params = list(cfg.study.parameters)
    if path.exists():
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        if rows and set(params) == set(rows[0]) - {"id"}:
            return [{"id": int(r["id"]), **{p: float(r[p]) for p in params}} for r in rows]
        raise RuntimeError(f"{path} has different parameters; change study.name or delete the file")

    if not params:
        raise ValueError("Define at least one parameter in [study.parameters]")
    pts = latin_hypercube(cfg.study.parameters, cfg.study.samples, cfg.study.seed)
    design = [{"id": i, **p} for i, p in enumerate(pts)]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", *params])
        w.writeheader()
        w.writerows(design)
    log.info("LHS plan with %d points saved to %s", len(design), path)
    return design


def finished_ids(results_csv: Path) -> set[int]:
    """Points already computed (successfully or with a recorded failure)."""
    if not results_csv.exists():
        return set()
    with open(results_csv, newline="") as f:
        return {int(r["id"]) for r in csv.DictReader(f)}


def group_by_geometry(cfg: Config, points: list[dict]) -> dict[str, list[tuple[dict, Config]]]:
    groups: dict[str, list[tuple[dict, Config]]] = defaultdict(list)
    for d in points:
        pcfg = cfg.with_overrides({k: v for k, v in d.items() if k != "id"})
        groups[pcfg.geometry_key()].append((d, pcfg))
    return groups


def run_points(cfg: Config, points: list[dict], results_csv: Path) -> None:
    """Compute points ({"id": .., "section.name": value, ...}) and append them to results_csv."""
    from .solver import FluentRunner

    out = results_csv.parent
    groups = group_by_geometry(cfg, points)
    log.info("%d point(s) in %d distinct geometry(ies)", len(points), len(groups))
    streak: list[str] = []  # errors of the consecutive failed geometries
    for key, pts in groups.items():
        try:
            mesh = ensure_mesh(pts[0][1])
            streak.clear()
        except Exception as e:  # noqa: BLE001 - one bad geometry does not stop the study
            status = "geometry_failed" if isinstance(e, GeometryError) else "mesh_failed"
            message = (str(e).splitlines() or [repr(e)])[0][:300]
            log.exception("Geometry %s: %s: %s", key, status, message)
            for d, _ in pts:
                append_csv(results_csv, {**d, "status": status, "geometry": key, "error": message})
            streak.append(f"{status}: {message}")
            limit = cfg.study.max_consecutive_failures
            if limit and len(streak) >= limit and len(set(streak[-limit:])) == 1:
                raise RuntimeError(
                    f"{limit} geometries in a row failed with the same error ({streak[-1]}). This points "
                    "to the environment (license, CAD reader, a hung Ansys process), not to the "
                    "parameters: check it and run `study --retry-failed`.") from e
            continue

        # Sort by the Fluent inputs: each point starts from a nearby solution
        pts.sort(key=lambda t: tuple(t[1].fluent.values()))
        runner = FluentRunner(pts[0][1], mesh, out / "work").__enter__()
        try:
            for d, pcfg in pts:
                try:
                    tag = f"dp{d['id']:03d}"
                    res = runner.run(pcfg, save_as=out / "points" / f"{tag}.cas.h5",
                                     images_dir=out / "images" / tag, fields_dir=out / "ml_fields", name=tag,
                                     cells_dir=out / "cells")
                    append_csv(results_csv, {**d, "status": "ok", "geometry": key, **res})
                except Exception as e:  # noqa: BLE001 - one failed point does not stop the study
                    log.exception("Point %d failed: %s", d["id"], e)
                    append_csv(results_csv, {**d, "status": "solve_failed", "geometry": key,
                                             "error": str(e).splitlines()[0][:300]})
                    # Session possibly corrupted: restart it for the next points
                    runner.close()
                    runner = FluentRunner(pcfg, mesh, out / "work").__enter__()
        finally:
            runner.close()


def drop_failed(results_csv: Path) -> list[int]:
    """Remove the failed rows of results.csv (kept in results_failed_<time>.csv) so
    that those points are computed again. Returns their ids."""
    import time

    import pandas as pd

    if not results_csv.exists():
        return []
    df = pd.read_csv(results_csv)
    failed = df[df["status"] != "ok"]
    if failed.empty:
        return []
    backup = results_csv.with_name(f"results_failed_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    failed.to_csv(backup, index=False)
    df[df["status"] == "ok"].to_csv(results_csv, index=False)
    log.info("%d failed point(s) removed from %s (saved in %s)", len(failed), results_csv.name, backup.name)
    return failed["id"].astype(int).tolist()


def run_study(cfg: Config, dry_run: bool = False, retry_failed: bool = False) -> Path:
    design = load_or_create_design(cfg)
    results_csv = study_dir(cfg) / "results.csv"
    if retry_failed and not dry_run:
        drop_failed(results_csv)
    done = finished_ids(results_csv)
    pending = [d for d in design if d["id"] not in done]
    log.info("%d points in the plan, %d done, %d to compute", len(design), len(done), len(pending))
    if dry_run:
        for key, pts in group_by_geometry(cfg, pending).items():
            log.info("  geometry %s: points %s", key, [d["id"] for d, _ in pts])
        return results_csv
    run_points(cfg, pending, results_csv)
    log.info("Study finished: %s", results_csv)
    return results_csv
