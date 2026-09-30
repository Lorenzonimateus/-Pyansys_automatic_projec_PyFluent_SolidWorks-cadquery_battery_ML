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
autocfd images output/single/<case>.cas.h5   # (re)generate the images of saved cases
autocfd postprocess                    # redo images + ML fields of all saved study points
autocfd postprocess --single           # same for the points of `solve`
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

**Operating point (C-rate).** With `[operating] heat_source` set, the Fluent named
expression `parameter_heat_source_wperm3` is **computed** for every point, like in
the P50B project: heat per cell Q = (C·Ah)²·R_DC, and q = Q / V_cell in W/m³ with
V_cell = π(D/2)²·h, where D is the current value of `cad.D` (so the same C-rate
gives the same watts per cell whatever the diameter). Vary `operating.c_rate` in
`[study.parameters]`; the heat source itself cannot be a study parameter. Each
result stores `Q_cell_W` and `q_W_m3`; the C-rate does not change the mesh.

**Units.** PyFluent always returns the reports and field data in **SI** (temperature
in K, pressure in Pa), even when the case shows °C in the GUI. The `[outputs]`
convert the temperatures (`t_max - 273.15` → `T_max_C`) and the images convert
the field, so everything saved is in **°C**, independent of the case's display units.

**Images.** After each point (`solve`, `study`, `optimize`) three planes through
the **centre of the domain** are cut: `xy` (z = z_mid), `yz` (x = x_mid) and
`xz` (y = y_mid). Temperature (°C) is cut through the fluid and the cells,
static pressure (gauge Pa) and velocity magnitude (m/s) through the fluid only
(the cells stay white). Fluent-like style: `jet` colormap (blue = minimum, red =
maximum), vertical colorbar on the left with "Static Temperature [C]" (etc.)
above it, 11 equally spaced ticks from min to max in `%.1f`, white background,
no grid. One PNG per field and plane:
`output/studies/<name>/images/dpNNN/<field>_<plane>.png` (and
`output/single/images/<case>/`). By default each image is scaled to its own
min/max; set `[images] ranges` to use the same scale for every point.

**Fields for Machine Learning (CNN).** The jet images are for people; for a model
they are a poor format (one value spread over three non-linear RGB channels,
8-bit quantization, and a per-image scale that changes the meaning of each
color). So each point also writes, per mid-plane, a `.npz` with the raw fields on
a **fixed regular grid** (same domain and resolution for every case,
`[ml_fields]`; changing the geometry moves the solids inside the grid, never the
grid):

| key | content |
|---|---|
| `T`, `p`, `U` | float32 in °C, Pa (gauge), m/s; NaN where undefined (outside the domain; `p`, `U` also in the cells) |
| `T_norm`, `p_norm`, `U_norm` | normalized with **fixed limits for the whole dataset** (`[ml_fields.normalization]`), 0 where undefined; limits stored as `T_limits`, ... |
| `mask` | 1 = fluid, 0 = solid or outside the domain |
| `sdf` | signed distance in mm to the nearest solid, wall or grid edge (> 0 in the fluid) |
| `params`, `param_names` | the point's parameters (`cad.*`, `fluent.*`) |
| `h`, `v`, `axes`, `plane_pos_mm` | grid coordinates in mm and plane position |

Interpolation is barycentric on Fluent's own triangles of the cut (it never
fills the cells with fluid values, as a convex-hull `griddata` would). Files:
`output/studies/<name>/ml_fields/<plane>/dpNNN.npz` and `normalization.json`.
A warning is logged when a case exceeds the grid or the normalization limits.
For training:

```python
from autocfd.ml_fields import load_dataset
ds = load_dataset("output/studies/doe_v1/ml_fields", "xz", targets=("T_norm",))
ds["geometry"]  # (N, 2, 128, 256): mask, sdf    -> CNN input
ds["params"]    # (N, P): velocity, heat source... -> vector or constant channels
ds["targets"]   # (N, 1, 128, 256): T_norm        -> CNN output (U-Net)
```

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
  single/images/<case>/     images of each `solve`
  studies/<name>/
    design.csv                LHS plan
    results.csv               one row per point, written as soon as it finishes
    points/dpNNN.cas.h5       case + data of each point
    images/dpNNN/             temperature/pressure/velocity on the planes xy, yz, xz
    ml_fields/<plane>/dpNNN.npz   the same fields as float32 on a fixed grid (CNN dataset)
    ml_fields/normalization.json  grid and normalization limits of the dataset
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
- With the current layout, the mid-plane `yz` (x_mid) falls in the gap between the
  2nd and 3rd columns of cells: it shows the air between them, not a cell section.
- **Steady state.** At high C-rates the discharge lasts minutes, so the steady-state
  temperature is conservative (overestimated); a transient run is the next step there.

## Tests

```powershell
pytest
```

They cover the configuration, the convergence logic, the workflow/case parsing,
the ML model and the optimizer, without needing Ansys or SolidWorks.
