"""Click Run on this file to execute the pipeline.

Edit STEPS and OVERRIDES below to choose what runs.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Steps, in order:
#   "inspect"   list SolidWorks global variables and the base case's parameters
#   "geometry"  SolidWorks + SpaceClaim           "mesh"   Fluent Meshing (.wft replay)
#   "solve"     one point with the config values  "study"  DOE (resumable)
#   "analyze"   sensitivity (no Ansys)            "train"  ML model (no Ansys)
#   "optimize"  Bayesian optimization with CFD    "best"   best CFD points (no Ansys)
#
#   check one geometry:        ["geometry", "mesh", "solve"]
#   DOE + sensitivity + ML:    ["study", "analyze", "train"]
#   then optimize:             ["optimize", "train"]
STEPS = ["study", "analyze", "train"]

CONFIG = "configs/default.toml"

# Values that override the config, e.g. {"cad.D": 20.0, "fluent.parameter_velocity_inlet": 5}
OVERRIDES = {}


def run() -> None:
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT / "src"))
    from autocfd.cli import main

    extra = []
    for key, value in OVERRIDES.items():
        extra += ["--set", f"{key}={value}"]

    for step in STEPS:
        print(f"\n===== {step} =====", flush=True)
        main(["-c", CONFIG, *extra, step])
    print("\n===== done =====")


if __name__ == "__main__":
    run()
