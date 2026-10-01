# autocfd

Parametric CFD pipeline for studies that are too complex to script the geometry
in Python: the geometry lives in **SolidWorks**, it is prepared in **SpaceClaim**,
meshed with a **Fluent Meshing** workflow and solved with a **Fluent** base case,
all set up and tuned by hand once. Python (PyAnsys + COM) then drives the whole
chain for every design point and feeds the results to sensitivity analysis,
Machine Learning, optimization and a transient model of the battery cells (PINN).

**Results and conclusions: [`reports/final_report.md`](reports/final_report.md)** (in Portuguese).

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
                          images, ml_fields/*.npz  ──► people / CNN training (U-Net)
                          cells/*.npz (h maps)      ──► transient cell temperature (PINN)
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
autocfd study --retry-failed           # compute again the points that failed
autocfd images output/single/<case>.cas.h5   # (re)generate the images of saved cases
autocfd postprocess                    # redo images, ML fields and cell data of the study points
autocfd postprocess --only cells       # only some outputs (images, ml_fields, cells)
autocfd postprocess --single           # same for the points of `solve`
autocfd pinn                           # transient temperature of a cell, one load ([pinn])
autocfd pinn --parametric              # one PINN for every C-rate in [pinn] c_rate_range
autocfd pinn --replot                  # redo the PINN figures from the saved model
autocfd analyze                        # sensitivity: Sobol indices, CoP, response surfaces
autocfd geometry-report                # which geometry is best: fair comparison + report.md
autocfd train                          # ML model, leave-one-out validation
autocfd optimize                       # Bayesian optimization with CFD verification
autocfd best                           # best CFD points so far
```

## Requirements

- Windows, Python ≥ 3.11
- SolidWorks (tested with 2025, API rev 33) — only for `geometry`
- Ansys 2025 R2 (v252): SpaceClaim + Fluent licenses — for `geometry`, `mesh`, `solve`,
  `study`, `optimize`, `postprocess`. `analyze`, `train`, `best`, `pinn` and `inspect` of
  the case run without Ansys.
- PyTorch with CUDA for `pinn` (tested: torch 2.14 + CUDA 12.6 on an RTX 6000 Ada; it
  also runs on the CPU, much slower).

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
pip install torch --index-url https://download.pytorch.org/whl/cu126   # for `pinn`
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

**Transient cell temperature (PINN).** See the section below.

**Failures.** Statuses in `results.csv`: `geometry_failed` (SolidWorks or SpaceClaim:
usually an invalid parameter combination), `mesh_failed` (Fluent Meshing, after
`[mesh] attempts` tries: the CAD import of the .scdocx can fail for reasons outside
the geometry, e.g. "attaching to assembly failed"), `solve_failed` (Fluent; the session
is restarted). The study goes on, but stops after `[study] max_consecutive_failures`
identical failures in a row (that is the environment, not the parameters).
`study --retry-failed` computes those points again (the failed rows are kept in
`results_failed_<time>.csv`). Only `status = ok` and converged points are used by
the analysis, the ML and the optimizer.

## Transient cell model (PINN)

The steady CFD tells how the air cools each cell; a discharge at high C-rate lasts
minutes, so the steady temperature is never reached (and overestimates it). The
cell-only model takes the cooling from the CFD and solves the transient conduction
in ONE cell:

    rho cp dT/dt = div(K grad T) + q(t)          q(t) = (C(t) Ah)^2 R_DC / V
    -K grad T . n = h(phi, y) (T - T_inlet)       on the side
    dT/dy = 0                                     at the ends (they touch the duct walls)

**Data from the CFD** (`cells/<point>.npz`, written after each solved point, or
`postprocess --only cells`): per cell, h(phi, y) = q''/(Ts - T_inlet) from the wall
heat flux and temperature of the cell-air interface, geometry (axis, radius, height),
steady CFD temperatures, and the material read from the case file (orthotropic K in
global axes, exactly as Fluent used it). Assumptions: the air responds quasi-steadily
and the h map holds for any C-rate (constant properties: the air side is linear in
the heat).

**Three solutions of the same model, each checking the other:**

| | what it is | used for |
|---|---|---|
| steady finite volumes | `validate_steady`: the reduced model at steady state with the CFD heat | checks the h map: must reproduce the CFD |
| transient finite volumes | `FVSolver`: implicit, polar grid, full anisotropic tensor | reference for the PINN |
| PINN | `CellPINN` / `ParametricCellPINN` (PyTorch, GPU) | fast, differentiable, continuous in time, space and C-rate |

The PINN is trained with NO simulation data, only with the residuals of the
equation, the Robin condition and the adiabatic ends; the initial condition is exact
by construction (T = T0 + dT tau N), the problem is made dimensionless (Fourier
numbers per direction), and the loss terms are balanced by gradient norms during
training.

**One load** (`autocfd pinn`): constant C-rate or a C(t) profile (`profile_csv`,
columns `time_s, c_rate`). **Parametric** (`autocfd pinn --parametric`): the C-rate
is an input of the network; every C-rate in `c_rate_range` is a full discharge
(3600/C s, or `duration_s`), each with its own temperature scale. One training
answers any C-rate in the range in milliseconds; it is validated against finite
volumes at `validate_c_rates`.

**Figures** (`output/studies/<name>/pinn/<point>/`, Fluent style for the maps):

| file | content |
|---|---|
| `<cell>_pinn_vs_fv.png` | T max and T mean over time, PINN vs finite volumes; mid-height section at the end: PINN, FV, difference |
| `<cell>_sections_time.png` | PINN mid-height section at 25, 50, 75 and 100 % of the discharge (one colour scale) |
| `<cell>_axial_section.png` | section through the cell axis (z = 0) at the end: PINN, FV, difference |
| `<cell>_surface_and_h.png` | unwrapped side surface: PINN temperature at the end, and the CFD h map that cools it |
| `<cell>_training.png` | loss terms and adaptive weights over the iterations |
| `<cell>_parametric_curves.png` | T max during the discharge for each validated C-rate (PINN vs FV), and T max at the end vs C-rate (PINN curve, FV points) |
| `<cell>_parametric_end_sections.png` | mid-height section at the end of the discharge for the lowest, middle and highest C-rate |
| `<cell>_parametric_training.png` | training of the parametric network |

Tables: `steady_vs_cfd.csv`, `*_pinn_vs_fv.csv`, `*_metrics.csv`,
`*_parametric_metrics.csv`, `*_parametric_sweep.csv`; models: `*.pt`.

**Results on doe_crate_kfix** (RTX 6000 Ada; full report: `reports/final_report.md`):

- Steady check (reduced model with the CFD h map vs the CFD itself), 9 cells: T max
  within −1.4..+1.0 K at up to 214 C (dp021), ±0.25 K (dp030).
- One load, 10C for 360 s (20 000 iterations, ~10 min): T max at the end within 0.6-0.7 K
  of finite volumes (dp021: 138.5 vs 139.1 C; dp030: 116.6 vs 117.2 C).
- Parametric, 1C to 10C (40 000 iterations, ~25 min), max error of T max against finite
  volumes at 1, 2.5, 5, 7.5, 10C: ≤ 0.71 K (dp021), ≤ 0.55 K (dp030). After training, 46
  C-rates are swept in 0.12 s (finite volumes: ~40 s per C-rate).
- Lessons: the adaptive loss weights must be capped (`max_weight_ratio`; without it one
  run ended 43 K off), and the parametric network is trained on a C-rate range 10 %
  wider than the one used (`c_margin`; at the edge of an exact range: 5.6 K error).
- At 10C the steady state would give ~250 C at the worst point; a full 360 s discharge
  reaches 139 C (117 C at the recommended geometry).
- At low C-rates the cell varies by less than 1 K, so with 11 colorbar ticks at one
  decimal place some tick labels repeat (e.g. 21.3, 21.3): the values are right, the
  format is the project's rule.

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
    results_failed_<time>.csv failed rows removed by `study --retry-failed`
    cells/dpNNN.npz           h maps, geometry, material of the 9 cells (PINN input)
    geometry/                 geometry report: report.md, figures, tables
    pinn/<point>/             PINN models, comparisons with finite volumes, figures
```

## Geometry choice (`geometry-report`)

The C-rate dominates the cell temperature (Sobol S_T 0.89 in doe_crate_kfix) and hides the
geometry. With constant properties the problem is linear in the heat, so the
**thermal resistance R_th = (T max − T inlet) / Q_cell** depends only on the geometry
and the air velocity. `autocfd geometry-report` fits, on the converged points,

    log R_th = a0 + b log v + sum c_i g_i        log dP = d0 + e log v + sum f_i g_i

(standard errors, exact leave-one-out) to read the effect of each parameter and
whether it is significant. The comparison itself uses **Gaussian processes** on the
same inputs (log v, `cad.*`): the cell's internal conduction resistance is in series
with the convection, so at the same fan power R_th can have a minimum inside a gap's
range, which a log-linear law cannot represent. The inlet area is fitted from the
measured mass flow (a gap that widens the duct gets more air at the same velocity),
and the `cad.*` parameters are compared in three ways relative to
`[geometry_report] reference`: the same inlet velocity, the same mass flow and the
same fan power (solved on the study's velocity range). The report opens with an
explicit **recommendation**: at the same fan power, values within `tie_tolerance_k`
or 2σ (model uncertainty) of the best are tied, and `indifferent_choice` picks among
them (default `"min"`: the smallest gap, the most compact module); parameters with no
significant effect take it directly. The size of the cell array is measured from
`cells/*.npz`.

Output in `output/studies/<name>/geometry/`: `report.md` (short report in
Portuguese, with tables and figures), `geometry_tradeoff.png`, `geometry_maps.png`
(two geometric parameters: T max at `c_rate` for the three criteria on one colour
scale, and dP), `geometry_comparison.csv`, `geometry_effects.csv`.

Result for doe_crate_kfix (L1 = gap across the flow, P1 = gap along the flow, both 2 to
10 mm): P1 has no significant effect (+0.5 % ± 4.2 % on R_th). At the same fan power L1
has an optimum near 6 mm (149.9 C at 10C); 3.4 to 10 mm tie within the uncertainty and
L1 = 2 mm is 11 K warmer. Recommendation: **L1 = 3.4 mm, P1 = 2 mm** (cell array 356 cm³,
against 425 cm³ at L1 = P1 = 6), provided the fan delivers ~150 Pa at 13 g/s.

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
- **Mesh vs CAD diameter.** The faceted CAD import makes the meshed cells ~0.32 mm smaller
  than the CAD (D = 22 → 21.68 mm; the SolidWorks cylinders are exactly D). The heat is
  corrected (`[operating] diameter_offset_mm`), but the real gaps are ~0.3 mm larger than
  L1 and P1; tightening the faceting in the `.wft` and remeshing would remove it.
- **Fixed in doe_crate_kfix** (doe_crate is kept for comparison): the cell material had
  the axial k = 25 W/m·K along z instead of y (the cells' axis), and the cells got 2.9 %
  less heat than the nominal C-rate. Together they lowered the temperature rise by
  11-64 % (median 33 %).
- **Steady state.** At high C-rates the discharge lasts minutes, so the steady-state
  temperature is conservative (overestimated): use the transient cell model (PINN).
- **PINN, parametric curve.** Between validated C-rates the parametric curve can show a
  small wiggle (dp021, near 7C); `autocfd pinn` always writes the finite-volume solution
  of the same model next to it for checking.

## Tests

```powershell
pytest
```

They cover the configuration, the convergence logic, the workflow/case parsing, the
retries, the ML fields, the ML model and the optimizer, the cell model (analytical
solutions, energy conservation, lumped limit) and the PINNs (small networks on the
CPU), without needing Ansys or SolidWorks.
