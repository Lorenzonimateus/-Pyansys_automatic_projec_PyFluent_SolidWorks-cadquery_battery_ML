"""Fluent stage: base case + new mesh + input parameters -> output reports.

All the physics lives in the base case (set up and tuned by hand in the GUI):
models, materials, boundary conditions, named expressions and report
definitions. For each geometry the mesh is REPLACED (zones are matched by name,
so the setup is kept); for each point only the named expressions change. The
next point on the same mesh starts from the previous solution.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np

from .config import Config
from .meshing import launch

log = logging.getLogger(__name__)


def derived_outputs(cfg: Config, reports: dict[str, float]) -> dict[str, float]:
    """Evaluate [outputs] (numpy expressions over the report names)."""
    out = {}
    for name, expr in cfg.outputs.items():
        try:
            out[name] = float(eval(expr, {"__builtins__": {}, "np": np}, dict(reports)))  # noqa: S307
        except Exception as e:  # noqa: BLE001
            raise ValueError(f"[outputs] {name} = {expr!r} failed: {e}") from e
    return out


def block_converged(samples: list[dict[str, float]], monitor, per_block: int,
                    rel_tol: float) -> tuple[bool, dict[str, float]]:
    """Compare the average of the last block of samples with the previous block.

    Averages (not instantaneous values) are used because a steady RANS of flow
    around bluff bodies often keeps a small oscillation (vortex shedding) that
    never dies out; its mean, however, settles."""
    if len(samples) < 2 * per_block:
        return False, {k: math.inf for k in monitor}
    last, prev = samples[-per_block:], samples[-2 * per_block:-per_block]
    change = {}
    for k in monitor:
        a, b = np.mean([p[k] for p in last]), np.mean([p[k] for p in prev])
        change[k] = float(abs(a - b) / max(abs(a), 1e-12))
    return all(c < rel_tol for c in change.values()), change


def window_average(samples: list[dict[str, float]], n: int) -> tuple[dict[str, float], dict[str, float]]:
    """Mean of each report over the last n samples, and its relative oscillation (std/|mean|)."""
    window = samples[-n:]
    mean = {k: float(np.mean([p[k] for p in window])) for k in window[-1]}
    osc = {k: float(np.std([p[k] for p in window]) / max(abs(mean[k]), 1e-12)) for k in mean}
    return mean, osc


class FluentRunner:
    """Fluent session bound to one mesh. Use it as a context manager (releases the license):

        with FluentRunner(cfg, mesh_file, work_dir) as f:
            f.run(cfg_point)
    """

    def __init__(self, cfg: Config, mesh_file: Path | None, work_dir: Path):
        self.cfg = cfg
        self.mesh_file = Path(mesh_file) if mesh_file else None
        self.work_dir = Path(work_dir)
        self.session = None
        self._initialized = False

    # ------------------------------------------------------------ session
    def __enter__(self) -> "FluentRunner":
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.session = launch(self.cfg, "solver", self.cfg.solver.processors, self.work_dir)
        try:
            self._prepare()
        except BaseException:
            self.close()
            raise
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        if self.session is None:
            return
        try:
            self.session.exit()
            log.info("Fluent session closed, license released.")
        except Exception as e:  # noqa: BLE001
            log.warning("Failed to close the Fluent session: %s", e)
        self.session = None

    def _prepare(self) -> None:
        st = self.session.settings
        base = Path(self.cfg.paths.base_case).resolve()
        st.file.read_case(file_name=str(base))
        if self.mesh_file is not None:
            log.info("Replacing the mesh of %s with %s", base.name, self.mesh_file)
            st.file.replace_mesh(file_name=str(self.mesh_file.resolve()))
        st.mesh.check()
        self._check_case()
        self._disable_residual_stop()

    def _check_case(self) -> None:
        """Fail early if the base case lacks what the config refers to."""
        st = self.session.settings
        exprs = set(st.setup.named_expressions.get_object_names())
        missing = set(self.cfg.fluent) - exprs
        if missing:
            raise KeyError(f"Named expressions not in the base case: {sorted(missing)}; "
                           f"available: {sorted(exprs)}")
        rd = st.solution.report_definitions
        reports = set()
        for kind in ("surface", "volume", "force", "flux", "expression", "user_defined"):
            if hasattr(rd, kind):
                reports |= set(getattr(rd, kind).get_object_names())
        missing = set(self.cfg.solver.reports) - reports
        if missing:
            raise KeyError(f"Report definitions not in the base case: {sorted(missing)}; "
                           f"available: {sorted(reports)}")

    def _disable_residual_stop(self) -> None:
        """Stopping is decided by the monitored reports, not by the residuals."""
        eqs = self.session.settings.solution.monitor.residual.equations
        try:
            for name in eqs.get_object_names():
                eqs[name].check_convergence = False
        except Exception as e:  # noqa: BLE001
            log.warning("Could not disable the residual criterion: %s", e)

    # ------------------------------------------------------------ solve
    def compute_reports(self) -> dict[str, float]:
        names = list(self.cfg.solver.reports)
        out = self.session.settings.solution.report_definitions.compute(report_defs=names)
        values = {}
        for item in out:
            for k, v in item.items():
                values[k] = float(v[0] if isinstance(v, (list, tuple)) else v)
        return values

    def set_inputs(self, cfg: Config) -> None:
        ne = self.session.settings.setup.named_expressions
        for name in cfg.fluent:
            ne[name].definition = cfg.fluent_expression(name)

    def run(self, cfg: Config, save_as: Path | None = None, images_dir: Path | None = None,
            fields_dir: Path | None = None, name: str = "point") -> dict:
        """Solve one point: cfg carries the [fluent] values of this point.

        Afterwards: images (<images_dir>/<field>_<plane>.png) and ML fields
        (<fields_dir>/<plane>/<name>.npz), see postprocess.py."""
        s = cfg.solver
        sol = self.session.settings.solution
        self.set_inputs(cfg)
        if not self._initialized:
            sol.initialization.hybrid_initialize()
            self._initialized = True

        samples: list[dict[str, float]] = []
        iters, converged = 0, False
        while iters < s.max_iterations:
            for _ in range(s.check_every // s.sample_every):
                sol.run_calculation.iterate(iter_count=s.sample_every)
                iters += s.sample_every
                cur = self.compute_reports()
                if not all(math.isfinite(v) for v in cur.values()):
                    raise RuntimeError(f"Solution diverged at iteration {iters}: {cur}")
                samples.append(cur)
            if iters >= s.min_iterations:
                converged, change = block_converged(samples, s.monitor, s.check_every // s.sample_every,
                                                    s.rel_tol)
                log.info("it %d: %s", iters, "  ".join(f"{k}={cur[k]:.6g} (d_avg={change[k]:.1e})"
                                                     for k in s.monitor))
                if converged:
                    break
        if not converged:
            log.warning("Point %s did NOT converge in %d iterations", cfg.fluent, iters)

        values, oscillation = window_average(samples, max(1, s.average_window // s.sample_every))
        result = {**cfg.heat_inputs(), **values, **derived_outputs(cfg, values),
                  **{f"{k}_osc": oscillation[k] for k in s.monitor},
                  "iterations": iters, "converged": converged}
        if s.save_case_data and save_as is not None:
            save_as.parent.mkdir(parents=True, exist_ok=True)
            self.session.settings.file.write(file_type="case-data", file_name=str(save_as.resolve()))
            result["case_data"] = str(save_as)
        from .postprocess import postprocess
        result.update(postprocess(self.session, cfg, name, images_dir, fields_dir))
        log.info("Result: %s", result)
        return result
