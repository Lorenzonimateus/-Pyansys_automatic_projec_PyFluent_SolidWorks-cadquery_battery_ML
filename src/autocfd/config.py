"""Project configuration, read from a TOML file (see configs/default.toml).

Free-form sections ([cad], [fluent], [outputs]) are plain dicts, because their
names come from the SolidWorks part and the Fluent base case, not from the code.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Sections whose values change the geometry/mesh (a new case directory per combination).
GEOMETRY_SECTIONS = ("cad", "mesh")
# Sections that accept "section.name" overrides and study parameters.
PARAMETER_SECTIONS = ("cad", "mesh", "fluent", "operating")


@dataclass(frozen=True)
class Paths:
    output_dir: str = "output"
    sldprt: str = ""
    mesh_workflow: str = ""
    base_case: str = ""


@dataclass(frozen=True)
class SpaceClaim:
    exe: str = "C:/Program Files/ANSYS Inc/v252/scdm/SpaceClaim.exe"
    timeout_s: int = 900
    fluid_body: str = "volume"
    flow_axis: str = "x"
    inlet_name: str = "inlet"
    outlet_name: str = "output"


@dataclass(frozen=True)
class Mesh:
    processors: int = 16
    min_size_mm: float = 0.3
    max_size_mm: float = 3.0
    hex_max_size_mm: float = 2.4
    bl_layers: int = 5
    attempts: int = 3            # Fluent Meshing tries per geometry (CAD import can fail transiently)
    retry_wait_s: float = 120.0  # wait between tries


@dataclass(frozen=True)
class Solver:
    processors: int = 16
    mpi: str = "intel"
    start_timeout_s: int = 300
    reports: tuple[str, ...] = ()
    monitor: tuple[str, ...] = ()
    min_iterations: int = 150
    max_iterations: int = 1500
    check_every: int = 50
    sample_every: int = 10
    rel_tol: float = 2e-3
    average_window: int = 200
    save_case_data: bool = True


@dataclass(frozen=True)
class Operating:
    """Battery operating point -> volumetric heat source of the Fluent case.

    Q per cell = (c_rate * capacity_ah)^2 * r_dc_ohm; the Fluent named expression
    `heat_source` receives Q / V_cell in W/m^3, with V_cell = pi (D/2)^2 h and D
    read from the parameter `diameter` (so it follows the geometry). Inactive
    when heat_source is empty."""
    c_rate: float = 1.0
    capacity_ah: float = 5.0
    r_dc_ohm: float = 0.0128
    cell_height_mm: float = 70.15
    diameter: str = "cad.D"
    heat_source: str = ""


@dataclass(frozen=True)
class Images:
    enabled: bool = True
    fields: tuple[str, ...] = ("temperature", "pressure", "velocity")
    planes: tuple[str, ...] = ("xy", "yz", "xz")
    dpi: int = 150
    ranges: dict[str, tuple[float, float]] = field(default_factory=dict)


@dataclass(frozen=True)
class MLFields:
    enabled: bool = True
    planes: tuple[str, ...] = ("xy", "yz", "xz")
    x_mm: tuple[float, float] = (0.0, 330.0)
    y_mm: tuple[float, float] = (0.0, 70.15)
    z_mm: tuple[float, float] = (-110.0, 0.0)
    resolution: tuple[int, int, int] = (256, 64, 128)
    normalization: dict[str, tuple[float, float]] = field(default_factory=lambda: {
        "temperature": (20.0, 60.0), "pressure": (-200.0, 300.0), "velocity": (0.0, 30.0)})


@dataclass(frozen=True)
class PINN:
    """Transient cell-only model (cell_thermal.py, pinn.py); h map from a solved point."""
    export_cells: bool = True          # write cells/<point>.npz after each solved point
    point: str = "dp010"               # point whose CFD gives the h map (cells/<point>.npz)
    cell: str = "hottest"              # a zone name, "hottest" (max CFD temperature) or "all"
    c_rate: float = 10.0               # constant-current discharge ...
    profile_csv: str = ""              # ... or a C-rate profile: CSV with time_s, c_rate
    duration_s: float = 0.0            # 0 = full discharge at c_rate: 3600 / c_rate
    initial_temperature_c: float = 20.0
    iterations: int = 20000
    width: int = 128
    depth: int = 6
    device: str = "cuda"
    # parametric in the C-rate: one network for every constant-current discharge in the range
    parametric: bool = False
    c_rate_range: tuple[float, float] = (1.0, 10.0)
    validate_c_rates: tuple[float, ...] = (1.0, 2.5, 5.0, 7.5, 10.0)
    parametric_iterations: int = 40000


@dataclass(frozen=True)
class GeometryReport:
    """Fair comparison of the geometric (cad.*) study parameters (geometry_report.py)."""
    velocity: str = "fluent.parameter_velocity_inlet"   # study parameter: inlet air velocity
    temperature: str = "T_max_C"                        # result columns
    pressure: str = "dP_Pa"
    heat: str = "Q_cell_W"
    mass_flow: str = "m_out"
    inlet_temperature_c: float = 20.0
    air_density: float = 1.177
    c_rate: float = 10.0                                # C-rate of the temperature tables
    # A parameter with no significant effect, or whose effect at the same fan power is below
    # tie_tolerance_k, is set by this rule: "min" = smallest value (most compact module),
    # "max" or "reference".
    indifferent_choice: str = "min"
    tie_tolerance_k: float = 2.0
    reference: dict[str, float] = field(default_factory=dict)   # empty: middle of each range
    labels: dict[str, str] = field(default_factory=dict)        # axis labels of the parameters


@dataclass(frozen=True)
class Study:
    name: str = "study"
    samples: int = 20
    seed: int = 42
    outputs: tuple[str, ...] = ()
    max_consecutive_failures: int = 3  # stop the study: failures in a row point to the environment
    parameters: dict[str, tuple[float, float]] = field(default_factory=dict)


@dataclass(frozen=True)
class Optimize:
    objective: str = ""
    sense: str = "min"
    constraints: tuple[str, ...] = ()
    iterations: int = 5
    seed: int = 0


@dataclass(frozen=True)
class Config:
    paths: Paths = field(default_factory=Paths)
    cad: dict[str, float] = field(default_factory=dict)
    spaceclaim: SpaceClaim = field(default_factory=SpaceClaim)
    mesh: Mesh = field(default_factory=Mesh)
    fluent: dict[str, float] = field(default_factory=dict)
    fluent_units: dict[str, str] = field(default_factory=dict)
    solver: Solver = field(default_factory=Solver)
    operating: Operating = field(default_factory=Operating)
    outputs: dict[str, str] = field(default_factory=dict)
    images: Images = field(default_factory=Images)
    ml_fields: MLFields = field(default_factory=MLFields)
    pinn: PINN = field(default_factory=PINN)
    geometry_report: GeometryReport = field(default_factory=GeometryReport)
    study: Study = field(default_factory=Study)
    optimize: Optimize = field(default_factory=Optimize)

    # ----------------------------------------------------------- derived
    @property
    def output_dir(self) -> Path:
        return Path(self.paths.output_dir)

    def geometry_params(self) -> dict[str, Any]:
        mesh = dataclasses.asdict(self.mesh)
        for run_setting in ("processors", "attempts", "retry_wait_s"):  # do not change the mesh
            mesh.pop(run_setting)
        return {"cad": dict(sorted(self.cad.items())), "mesh": mesh}

    def geometry_key(self) -> str:
        """Short hash of everything that defines geometry + mesh.
        Two configs with the same key share the same case directory."""
        blob = json.dumps(self.geometry_params(), sort_keys=True).encode()
        return hashlib.sha1(blob).hexdigest()[:10]

    def case_dir(self) -> Path:
        return self.output_dir / "cases" / self.geometry_key()

    # ----------------------------------------------------------- operating point
    def cell_volume_m3(self) -> float:
        d_mm = self.get(self.operating.diameter)
        return math.pi * (d_mm / 2000.0) ** 2 * (self.operating.cell_height_mm / 1000.0)

    def cell_heat_w(self) -> float:
        """Joule heat per cell, Q = (C * Ah)^2 * R_DC."""
        op = self.operating
        return (op.c_rate * op.capacity_ah) ** 2 * op.r_dc_ohm

    def heat_inputs(self) -> dict[str, float]:
        """Derived values saved with each result (empty when [operating] is inactive)."""
        if not self.operating.heat_source:
            return {}
        return {"Q_cell_W": self.cell_heat_w(), "q_W_m3": self.cell_heat_w() / self.cell_volume_m3()}

    def _with_derived(self) -> "Config":
        """Write the volumetric heat of the operating point into its Fluent named expression."""
        if not self.operating.heat_source:
            return self
        q = self.cell_heat_w() / self.cell_volume_m3()
        return dataclasses.replace(self, fluent={**self.fluent, self.operating.heat_source: q})

    def fluent_expression(self, name: str) -> str:
        """Definition string of a Fluent named expression, e.g. '3.0 [m/s]'."""
        unit = self.fluent_units.get(name, "")
        value = repr(float(self.fluent[name]))
        return f"{value} [{unit}]" if unit else value

    # ----------------------------------------------------------- overrides
    def get(self, dotted: str) -> float:
        section, _, name = dotted.partition(".")
        sub = getattr(self, section)
        return sub[name] if isinstance(sub, dict) else getattr(sub, name)

    def with_overrides(self, overrides: dict[str, Any]) -> "Config":
        """Return a copy with changed values, e.g. {"cad.D": 20.0, "fluent.parameter_velocity_inlet": 5}.

        Only names that already exist can be overridden (catches typos early)."""
        cfg = self
        for dotted, value in overrides.items():
            section, _, name = dotted.partition(".")
            if not name or not hasattr(cfg, section):
                raise KeyError(f"Unknown parameter: {dotted!r}")
            sub = getattr(cfg, section)
            if isinstance(sub, dict):
                if name not in sub:
                    raise KeyError(f"Unknown parameter: {dotted!r} (not in [{section}])")
                cfg = dataclasses.replace(cfg, **{section: {**sub, name: type(sub[name])(value)}})
            else:
                if name not in {f.name for f in dataclasses.fields(sub)}:
                    raise KeyError(f"Unknown parameter: {dotted!r}")
                ftype = type(getattr(sub, name))
                if ftype is int:
                    value = round(float(value))
                cfg = dataclasses.replace(cfg, **{section: dataclasses.replace(sub, **{name: ftype(value)})})
        # The heat source always follows the operating point (C-rate, diameter)
        return cfg._with_derived()


def is_geometric(parameter: str) -> bool:
    return parameter.split(".", 1)[0] in GEOMETRY_SECTIONS


def _build(cls, data: dict[str, Any]):
    known = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise KeyError(f"Unknown fields in [{cls.__name__.lower()}]: {sorted(unknown)}")
    data = {k: tuple(v) if isinstance(v, list) else v for k, v in data.items()}
    return cls(**data)


def load_config(path: str | Path) -> Config:
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    unknown = set(raw) - {f.name for f in dataclasses.fields(Config)}
    if unknown:
        raise KeyError(f"Unknown sections in {path}: {sorted(unknown)}")

    study_raw = dict(raw.get("study", {}))
    params, fixed = {}, {}
    for k, v in study_raw.pop("parameters", {}).items():
        values = v if isinstance(v, list) else [v]
        if len(values) == 2:        # [min, max]: sampled
            params[k] = (float(values[0]), float(values[1]))
        elif len(values) == 1:      # [value]: fixed for the whole study
            fixed[k] = float(values[0])
        else:
            raise ValueError(f"[study.parameters] {k!r} = {v}: use [min, max] for a range "
                             "or [value] for a fixed value")
    images_raw = dict(raw.get("images", {}))
    ranges = {k: (float(v[0]), float(v[1])) for k, v in images_raw.pop("ranges", {}).items()}
    mlf_raw = dict(raw.get("ml_fields", {}))
    if "normalization" in mlf_raw:
        mlf_raw["normalization"] = {k: (float(v[0]), float(v[1]))
                                    for k, v in mlf_raw["normalization"].items()}
    cfg = Config(
        paths=_build(Paths, raw.get("paths", {})),
        cad={k: float(v) for k, v in raw.get("cad", {}).items()},
        spaceclaim=_build(SpaceClaim, raw.get("spaceclaim", {})),
        mesh=_build(Mesh, raw.get("mesh", {})),
        fluent={k: float(v) for k, v in raw.get("fluent", {}).items()},
        fluent_units=dict(raw.get("fluent_units", {})),
        solver=_build(Solver, raw.get("solver", {})),
        operating=_build(Operating, raw.get("operating", {})),
        outputs=dict(raw.get("outputs", {})),
        images=_build(Images, {**images_raw, "ranges": ranges}),
        ml_fields=_build(MLFields, mlf_raw),
        pinn=_build(PINN, raw.get("pinn", {})),
        geometry_report=_build(GeometryReport, raw.get("geometry_report", {})),
        study=_build(Study, {**study_raw, "parameters": params}),
        optimize=_build(Optimize, raw.get("optimize", {})),
    )
    for k in fixed:
        if k.split(".", 1)[0] not in PARAMETER_SECTIONS:
            raise KeyError(f"Study parameter {k!r} must be in one of {PARAMETER_SECTIONS}")
    op = cfg.operating
    if op.heat_source:
        if op.heat_source not in cfg.fluent:
            raise KeyError(f"[operating] heat_source {op.heat_source!r} is not in [fluent]")
        driven = f"fluent.{op.heat_source}"
        if driven in params or driven in fixed:
            raise ValueError(f"{driven} is computed from [operating] (C-rate); vary "
                             "operating.c_rate in [study.parameters] instead")
        cfg.get(op.diameter)  # validates the diameter parameter
    cfg = cfg.with_overrides(fixed)  # fixed study values replace the nominal ones; derives the heat
    validate(cfg)
    return cfg


def validate(cfg: Config) -> None:
    for p, (lo, hi) in cfg.study.parameters.items():
        if p.split(".", 1)[0] not in PARAMETER_SECTIONS:
            raise KeyError(f"Study parameter {p!r} must be in one of {PARAMETER_SECTIONS}")
        cfg.with_overrides({p: lo})  # validates the name
        if not lo < hi:
            raise ValueError(f"Study parameter {p!r}: range [{lo}, {hi}] is empty")
    missing = set(cfg.solver.monitor) - set(cfg.solver.reports)
    if missing:
        raise ValueError(f"[solver] monitor has names not in reports: {sorted(missing)}")
    s = cfg.solver
    if s.check_every % s.sample_every or s.average_window < s.sample_every:
        raise ValueError("[solver] check_every must be a multiple of sample_every, "
                         "and average_window >= sample_every")
    im = cfg.images
    bad = (set(im.fields) - {"temperature", "pressure", "velocity"}) | (set(im.planes) - {"xy", "yz", "xz"})
    if bad or set(im.ranges) - set(im.fields):
        raise ValueError("[images] fields must be temperature/pressure/velocity, planes xy/yz/xz, "
                         "and ranges only for listed fields")
    ml = cfg.ml_fields
    if set(ml.planes) - {"xy", "yz", "xz"}:
        raise ValueError("[ml_fields] planes must be xy, yz or xz")
    if set(ml.normalization) != {"temperature", "pressure", "velocity"}:
        raise ValueError("[ml_fields.normalization] needs temperature, pressure and velocity")
    if len(ml.resolution) != 3 or min(ml.resolution) < 8:
        raise ValueError("[ml_fields] resolution = [nx, ny, nz], each >= 8")
    for name, (lo, hi) in [("x_mm", ml.x_mm), ("y_mm", ml.y_mm), ("z_mm", ml.z_mm),
                           *ml.normalization.items()]:
        if not lo < hi:
            raise ValueError(f"[ml_fields] {name}: [{lo}, {hi}] is empty")
    if cfg.geometry_report.indifferent_choice not in ("min", "max", "reference"):
        raise ValueError("[geometry_report] indifferent_choice must be min, max or reference")
    if cfg.spaceclaim.flow_axis not in ("x", "y", "z"):
        raise ValueError("spaceclaim.flow_axis must be 'x', 'y' or 'z'")
    if cfg.optimize.sense not in ("min", "max"):
        raise ValueError("optimize.sense must be 'min' or 'max'")
