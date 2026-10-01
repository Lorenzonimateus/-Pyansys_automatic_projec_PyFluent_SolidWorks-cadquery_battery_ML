"""Command line: python -m autocfd <command> [options]

  inspect    list the SolidWorks global variables and the base case's named
             expressions / report definitions (to fill [cad], [fluent], [solver])
  geometry   SolidWorks (global variables) + SpaceClaim (named selections) -> .scdocx
  mesh       replay the Fluent Meshing workflow (.wft) on that geometry (cached)
  solve      one point with the values in the config (--base: base case as it is)
  study      run the DOE defined in [study] (resumable)
  postprocess  redo images, ML fields and cell data of every saved point of the study (--single: of solve)
  pinn       transient temperature of a cell (PINN), cooled with the CFD h map of a point ([pinn])
  images     temperature/pressure/velocity on the mid-planes of saved cases (.cas.h5 + .dat.h5)
  analyze    sensitivity analysis of the study results (no Ansys)
  geometry-report  which geometry is best: fair comparison of the cad.* parameters + report.md (no Ansys)
  train      train the Machine Learning model on the study results (no Ansys)
  optimize   surrogate-based optimization with CFD verification ([optimize])
  best       show the best CFD points found so far (no Ansys)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .config import load_config


def _parse_set(items: list[str]) -> dict[str, float | str]:
    out = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"--set expects key=value, got {item!r}")
        try:
            out[key] = float(value)
        except ValueError:
            out[key] = value
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="autocfd", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-c", "--config", default="configs/default.toml", type=Path)
    parser.add_argument("--set", action="append", default=[], metavar="SECTION.NAME=VALUE",
                        help="override a config value, e.g. --set cad.D=20")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("inspect")
    sub.add_parser("geometry")
    sub.add_parser("mesh")
    p_solve = sub.add_parser("solve")
    p_solve.add_argument("--base", action="store_true", help="solve the base case mesh (no geometry/meshing)")
    p_study = sub.add_parser("study")
    p_study.add_argument("--dry-run", action="store_true", help="only show the plan and the geometry groups")
    p_study.add_argument("--retry-failed", action="store_true",
                         help="compute again the points whose status is not ok")
    p_post = sub.add_parser("postprocess")
    p_post.add_argument("--single", action="store_true", help="points of single_runs.csv instead of the study")
    p_post.add_argument("--only", nargs="+", choices=["images", "ml_fields", "cells"],
                        help="redo only these outputs")
    p_pinn = sub.add_parser("pinn")
    p_pinn.add_argument("--parametric", action="store_true",
                        help="one network for every C-rate in [pinn] c_rate_range")
    p_pinn.add_argument("--replot", action="store_true",
                        help="only redo the figures of the saved PINNs (no training)")
    p_img = sub.add_parser("images")
    p_img.add_argument("cases", nargs="+", type=Path, help=".cas.h5 files (with their .dat.h5)")
    sub.add_parser("analyze")
    sub.add_parser("geometry-report")
    sub.add_parser("train")
    sub.add_parser("optimize")
    sub.add_parser("best")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S")
    cfg = load_config(args.config).with_overrides(_parse_set(args.set))

    if args.command == "inspect":
        from .cad import list_parameters
        from .inspect_case import print_case_info
        print(f"\nSolidWorks global variables of {cfg.paths.sldprt} ([cad] section):")
        for name, value in list_parameters(Path(cfg.paths.sldprt)).items():
            print(f"      {name:<36} = {value:g}")
        print_case_info(cfg.paths.base_case)
    elif args.command == "geometry":
        from .pipeline import ensure_geometry
        print(ensure_geometry(cfg))
    elif args.command == "mesh":
        from .pipeline import ensure_mesh
        print(ensure_mesh(cfg))
    elif args.command == "solve":
        from .pipeline import run_single
        run_single(cfg, use_base_mesh=args.base)
    elif args.command == "study":
        from .doe import run_study
        run_study(cfg, dry_run=args.dry_run, retry_failed=args.retry_failed)
    elif args.command == "postprocess":
        from .postprocess import postprocess_saved
        only = set(args.only) if args.only else None
        print(f"{postprocess_saved(cfg, single=args.single, only=only)} point(s) post-processed")
    elif args.command == "pinn":
        from .doe import study_dir
        from .pinn import run_from_config
        if args.replot:
            from .pinn import replot_cell_pinn, replot_parametric
            replot_cell_pinn(cfg, study_dir(cfg))
            replot_parametric(cfg, study_dir(cfg))
        else:
            run_from_config(cfg, study_dir(cfg), parametric=args.parametric or None)
    elif args.command == "images":
        from .images import images_from_case
        for case in args.cases:
            for path in images_from_case(cfg, case):
                print(path)
    elif args.command == "geometry-report":
        from .geometry_report import build_report
        print(build_report(cfg))
    elif args.command == "analyze":
        from .doe import study_dir
        from .sensitivity import analyze
        analyze(study_dir(cfg) / "results.csv", cfg.study.parameters, list(cfg.study.outputs))
    elif args.command == "train":
        from .doe import study_dir
        from .ml import train
        train(study_dir(cfg) / "results.csv", list(cfg.study.parameters), list(cfg.study.outputs))
    elif args.command == "optimize":
        from .optimize import optimize
        optimize(cfg)
    elif args.command == "best":
        from .optimize import report_best
        report_best(cfg)
