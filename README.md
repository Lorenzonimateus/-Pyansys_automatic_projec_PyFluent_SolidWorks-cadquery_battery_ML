# autocfd

Parametric CFD pipeline for studies that are too complex to script the geometry
in Python: the geometry lives in **SolidWorks**, it is prepared in **SpaceClaim**,
meshed with a **Fluent Meshing** workflow and solved with a **Fluent** base case,
all set up and tuned by hand once. Python (PyAnsys + COM) then drives the whole
chain for every design point and feeds the results to sensitivity analysis,
Machine Learning and optimization.

```
            set up ONCE by hand (GUI)                      driven by Python for each point
 ┌───────────────────────────────────────────┐   ┌─────────────────────────────────────────────┐
 SolidWorks  global variables (D, L1, P1, ...)  ──► copy of the part, new values, rebuild   (COM)
 SpaceClaim  body names + named selections       ──► batch script: inlet/outlet by position  (IronPython)
 Fluent Meshing  watertight workflow  (.wft)     ──► replay the .wft on the new .scdocx      (PyFluent)
 Fluent      base case: physics, named           ──► replace mesh, set named expressions,    (PyFluent)
             expressions, report definitions         iterate until the reports settle
                                                     │
                                                     ▼
                                        results.csv ──► sensitivity (Sobol, CoP)
                                                    ──► ML (Gaussian process per output)
                                                    ──► optimization (GP + expected
                                                        improvement, each candidate
                                                        verified in CFD)
```

## How to run

Open `main.py` in PyCharm and click **Run**. The steps are listed in `STEPS` at the
top of the file; values can be overridden in `OVERRIDES`.

Or from the terminal (`python -m autocfd ...` is equivalent to `autocfd ...`):

```powershell
autocfd inspect                        # what can be parametrized (SolidWorks + base case)
autocfd geometry --set cad.D=20        # SolidWorks + SpaceClaim -> .scdocx
autocfd mesh --set cad.D=20            # Fluent Meshing (cached per geometry)
autocfd solve --set cad.D=20 --set fluent.parameter_velocity_inlet=5
autocfd solve --base                   # base case as it is (checks the solver settings)
autocfd study --dry-run                # LHS plan and geometry groups
autocfd study                          # run the DOE (resumable)
autocfd analyze                        # sensitivity: Sobol indices, CoP, response surfaces
autocfd train                          # ML model, leave-one-out validation
autocfd optimize                       # Bayesian optimization with CFD verification
autocfd best                           # best CFD points so far
```

## Requirements

- Windows, Python ≥ 3.11
- SolidWorks (tested with 2025, API rev 33) — only for `geometry`
- Ansys 2025 R2 (v252): SpaceClaim + Fluent licenses — for `geometry`, `mesh`, `solve`,
  `study`, `optimize`. `analyze`, `train`, `best` and `inspect` of the case run without Ansys.

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

## Setting up a new project (the "by hand" part)

1. **SolidWorks.** Put every dimension you want to vary under *Tools > Equations* as a
   **global variable** and link the sketch/feature dimensions to it. Give the bodies
   names (they become the Fluent zone names: here `cell_1..9` and `volume`).
2. **SpaceClaim.** Import the part and create the named selections the meshing needs.
   The batch script recreates them for each geometry by **position**: the faces of the
   fluid body (`spaceclaim.fluid_body`) at the minimum/maximum of `spaceclaim.flow_axis`
   become `inlet_name`/`outlet_name`. If a new project needs other selections, extend
   `src/autocfd/scripts/spaceclaim_prepare.py` with another rule.
3. **Fluent Meshing.** Mesh the .scdocx with the watertight workflow and save it
   (*File > Workflow > Save*, `.wft`). Sizes that should be study parameters are listed in `[mesh]`.
4. **Fluent.** Set up the physics on that mesh. Every input you want to vary must be a
   **named expression** used in the boundary conditions / sources; every result must be a
   **report definition**. Save the case (`.cas.h5`).
5. Run `autocfd inspect` and fill `[cad]`, `[fluent]`, `[solver] reports` and `[outputs]`.
6. Check one point: `STEPS = ["geometry", "mesh", "solve"]`, open the resulting case in
   Fluent and compare with the base case.

## How each stage works

| Stage | What it does | Cache / output |
|---|---|---|
| SolidWorks | copies the part to `geom_<key>.SLDPRT` (the original is never touched), sets the global variables, rebuilds, and **fails if the rebuild has errors** (invalid combination) | `output/cases/<key>/` |
| SpaceClaim | `SpaceClaim.exe /Headless /RunScript`: opens the part, checks the bodies, creates the named selections, saves `geometry.scdocx` | `spaceclaim_report.json` |
| Fluent Meshing | loads the `.wft`, replaces the geometry file, sizes and prism layers, runs every task up to the volume mesh and checks that all tasks are up to date | `mesh.msh.h5` |
| Fluent | reads the base case, **replaces the mesh** (zones matched by name, so all the setup is kept), sets the named expressions, iterates | `studies/<name>/points/dpNNN.cas.h5` |

The `<key>` is a hash of `[cad]` + `[mesh]`: changing only `[fluent]` values reuses
the geometry and the mesh; within a study all points that share a geometry run in
one Fluent session, each starting from the previous solution.

**Convergence.** The flow behind the cells keeps a small oscillation (vortex shedding
and some backflow at the outlet) that never dies out in a steady RANS: `p_in`
oscillates about ±1 % while `t_max` settles. The reports are therefore sampled every
`sample_every` iterations and the **block averages** are compared (`rel_tol`); the
results are averages over the last `average_window` iterations, and the relative
oscillation of each monitored report is saved as `<name>_osc`.

**Failures.** A point whose geometry fails (SolidWorks rebuild error, meshing error)
is written to `results.csv` with `status = geometry_failed` and the study goes on; a
Fluent failure gives `solve_failed` and the session is restarted. Only `status = ok`
and converged points are used by the analysis, the ML and the optimizer. To retry a
failed point, delete its row from `results.csv`.

## Output layout (`output/`, not in git)

```
output/
  cases/<key>/              one folder per distinct geometry + mesh
    geom_<key>.SLDPRT         rebuilt part
    geometry.scdocx           prepared geometry
    mesh.msh.h5
    params.json               [cad] + [mesh] that produced it
  single_runs.csv           results of `solve`
  studies/<name>/
    design.csv                LHS plan
    results.csv               one row per point, written as soon as it finishes
    points/dpNNN.cas.h5       case + data of each point
    sensitivity.csv, sensitivity_sobol.png, response_surfaces.png
    ml_model.joblib, ml_metrics.csv, ml_parity.png
    optimization.csv          each optimization iteration: proposal, prediction, CFD result
```

## Sensitivity, ML and optimization

- **Sensitivity** (`analyze`): quadratic response surface, CoP (1 − PRESS/SS, as
  optiSLang's MOP), Sobol S1/ST on the surrogate, Spearman on the raw data.
- **ML** (`train`): one Gaussian process per output (Matern 5/2, ARD), validated by
  leave-one-out. `autocfd.ml.predict(model, {...})` gives mean and standard deviation.
- **Optimization** (`optimize`): Bayesian optimization. Each iteration fits the GPs,
  maximizes EI(objective) × P(constraints satisfied) with differential evolution,
  runs that point through the whole chain and adds it to `results.csv`. The answer
  is always a CFD-verified point (`best`).

## Known limitations of the current case

- The SolidWorks global variable **`H` is not linked** to any dimension (both
  extrusions have a fixed 70.15 mm): link it in SolidWorks before using `cad.H`.
- The reports are in **SI units** (`t_*` in K, `p_*` in Pa), whatever the display units
  of the case.
- The heat source is a volumetric value (W/m³): if `cad.D` changes, the heat per cell
  changes with the cell volume. To keep the heat per cell constant, define the
  source in Fluent as an expression of the power and the zone volume.

## Tests

```powershell
pytest
```

They cover the configuration, the convergence logic, the workflow/case parsing,
the ML model and the optimizer, without needing Ansys or SolidWorks.
