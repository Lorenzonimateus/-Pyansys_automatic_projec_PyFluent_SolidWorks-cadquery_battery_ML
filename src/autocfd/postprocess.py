"""Post-processing of a solved point: images for people, fields for ML and the
cell data (h map, geometry, material) for the cell-only transient models (PINN).

Images and ML fields use the same cutting planes (created once in the open Fluent session, and
removed at the end so they are not saved in the cases). A failure here is
logged and never loses the point's result.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .config import Config

log = logging.getLogger(__name__)


def point_params(cfg: Config) -> dict[str, float]:
    """Parameters that define a point: [cad] + [fluent] values (+ C-rate when
    [operating] drives the heat source)."""
    extra = {"operating.c_rate": cfg.operating.c_rate} if cfg.operating.heat_source else {}
    return {**{f"cad.{k}": v for k, v in cfg.cad.items()},
            **{f"fluent.{k}": v for k, v in cfg.fluent.items()}, **extra}


OUTPUTS = ("images", "ml_fields", "cells")


def postprocess_saved(cfg: Config, single: bool = False, only: set[str] | None = None) -> int:
    """Redo images, ML fields and cell data of every saved point (study results.csv,
    or single_runs.csv with single=True) in one Fluent session; `only` restricts it
    to some of OUTPUTS. The parameters of each point come from its row, so the
    files match those of a live run."""
    import pandas as pd

    from .doe import study_dir
    from .meshing import launch

    if single:
        csv, root = cfg.output_dir / "single_runs.csv", cfg.output_dir / "single"
    else:
        csv, root = study_dir(cfg) / "results.csv", study_dir(cfg)
    df = pd.read_csv(csv)
    if "status" in df:
        df = df[df["status"] == "ok"]
    df = df[df["case_data"].notna()] if "case_data" in df else df.iloc[0:0]
    df = df[df["case_data"].map(lambda p: Path(p).exists())].drop_duplicates("case_data", keep="last")
    if df.empty:
        raise FileNotFoundError(f"No saved case + data listed in {csv}")

    param_cols = [c for c in df.columns if c.split(".", 1)[0] in ("cad", "fluent", "operating")]
    if cfg.operating.heat_source:  # derived from the C-rate, recomputed by with_overrides
        param_cols = [c for c in param_cols if c != f"fluent.{cfg.operating.heat_source}"]
    session = launch(cfg, "solver", cfg.solver.processors, root / "work")
    try:
        for _, row in df.iterrows():
            case = Path(row["case_data"])
            name = case.name.replace(".cas.h5", "")
            pcfg = cfg.with_overrides({c: row[c] for c in param_cols if pd.notna(row[c])})
            session.settings.file.read_case_data(file_name=str(case.resolve()))
            want = set(OUTPUTS) if only is None else set(only)
            done = postprocess(session, pcfg, name,
                               root / "images" / name if "images" in want else None,
                               root / "ml_fields" if "ml_fields" in want else None,
                               cells_dir=root / "cells" if "cells" in want else None)
            log.info("%s: %s", name, done or "nothing written")
    finally:
        session.exit()
    return len(df)


def postprocess(session, cfg: Config, name: str, images_dir: Path | None,
                fields_dir: Path | None, params: dict[str, float] | None = None,
                cells_dir: Path | None = None) -> dict[str, str]:
    """Write <images_dir>/<field>_<plane>.png, <fields_dir>/<plane>/<name>.npz and
    <cells_dir>/<name>.npz."""
    from .images import PlaneCutter, write_images
    from .ml_fields import write_fields

    out = {}
    if cfg.pinn.export_cells and cells_dir is not None:
        from .cells_export import export_cells
        try:
            export_cells(session, Path(cfg.paths.base_case), Path(cells_dir) / f"{name}.npz",
                         q_cell_w=cfg.heat_inputs().get("Q_cell_W"))
            out["cells"] = str(cells_dir)
        except Exception as e:  # noqa: BLE001
            log.warning("Cell data not exported for %s: %s", name, e)
    do_images = cfg.images.enabled and images_dir is not None
    do_fields = cfg.ml_fields.enabled and fields_dir is not None
    if not (do_images or do_fields):
        return out
    params = params if params is not None else point_params(cfg)
    try:
        cutter = PlaneCutter(session)
    except Exception as e:  # noqa: BLE001
        log.warning("Post-processing of %s skipped: %s", name, e)
        return out
    try:
        if do_images:
            try:
                im = cfg.images
                title = ", ".join(f"{k.split('.', 1)[1]} = {v:g}" for k, v in params.items())
                write_images(cutter, images_dir, title, im.fields, im.planes, im.dpi, im.ranges)
                out["images"] = str(images_dir)
            except Exception as e:  # noqa: BLE001
                log.warning("Images not generated for %s: %s", name, e)
        if do_fields:
            try:
                write_fields(cutter, fields_dir, name, cfg.ml_fields, params)
                out["ml_fields"] = str(fields_dir)
            except Exception as e:  # noqa: BLE001
                log.warning("ML fields not generated for %s: %s", name, e)
    finally:
        cutter.cleanup()
    return out
