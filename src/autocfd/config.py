"""Project configuration, read from a TOML file (see configs/default.toml).

Free-form sections ([cad], [fluent], [outputs]) are plain dicts, because their
names come from the SolidWorks part and the Fluent base case, not from the code.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Sections whose values change the geometry/mesh (a new case directory per combination).
GEOMETRY_SECTIONS = ("cad", "mesh")
# Sections that accept "section.name" overrides and study parameters.
PARAMETER_SECTIONS = ("cad", "mesh", "fluent")


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
class Study:
    name: str = "study"
    samples: int = 20
    seed: int = 42
    outputs: tuple[str, ...] = ()
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
    outputs: dict[str, str] = field(default_factory=dict)
    images: Images = field(default_factory=Images)
    ml_fields: MLFields = field(default_factory=MLFields)
    study: Study = field(default_factory=Study)
    optimize: Optimize = field(default_factory=Optimize)

    # ----------------------------------------------------------- derived
    @property
    def output_dir(self) -> Path:
        return Path(self.paths.output_dir)

    def geometry_params(self) -> dict[str, Any]:
        mesh = dataclasses.asdict(self.mesh)
        mesh.pop("processors")
        return {"cad": dict(sorted(self.cad.items())), "mesh": mesh}

    def geometry_key(self) -> str:
        """Short hash of everything that defines geometry + mesh.
        Two configs with the same key share the same case directory."""
        blob = json.dumps(self.geometry_params(), sort_keys=True).encode()
        return hashlib.sha1(blob).hexdigest()[:10]

    def case_dir(self) -> Path:
        return self.output_dir / "cases" / self.geometry_key()

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
        return cfg


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
        outputs=dict(raw.get("outputs", {})),
        images=_build(Images, {**images_raw, "ranges": ranges}),
        ml_fields=_build(MLFields, mlf_raw),
        study=_build(Study, {**study_raw, "parameters": params}),
        optimize=_build(Optimize, raw.get("optimize", {})),
    )
    for k in fixed:
        if k.split(".", 1)[0] not in PARAMETER_SECTIONS:
            raise KeyError(f"Study parameter {k!r} must be in one of {PARAMETER_SECTIONS}")
    cfg = cfg.with_overrides(fixed)  # fixed study values replace the nominal ones
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
    if cfg.spaceclaim.flow_axis not in ("x", "y", "z"):
        raise ValueError("spaceclaim.flow_axis must be 'x', 'y' or 'z'")
    if cfg.optimize.sense not in ("min", "max"):
        raise ValueError("optimize.sense must be 'min' or 'max'")
