"""SpaceClaim stage: .SLDPRT or .step -> .scdocx with the named selections Fluent Meshing needs.

SpaceClaim runs in batch (/Headless, /ExitAfterScript) with the IronPython script
scripts/spaceclaim_prepare.py. The body names come from SolidWorks or from the
CadQuery STEP (cell_1..N, volume); the inlet/outlet faces are found by position
(ends of the flow axis), so they survive any change of dimensions.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path

from .config import Config

log = logging.getLogger(__name__)

SCRIPT = Path(__file__).parent / "scripts" / "spaceclaim_prepare.py"


def prepare(cfg: Config, source: Path, scdocx: Path, required_bodies: list[str] | None = None,
            body_boxes: dict[str, list[float]] | None = None) -> Path:
    """body_boxes: {name: [xmin, ymin, zmin, xmax, ymax, zmax] in mm} to name the bodies
    of a STEP by position (SpaceClaim's STEP reader calls every body "Solid")."""
    sc = cfg.spaceclaim
    exe = Path(sc.exe)
    if not exe.exists():
        raise FileNotFoundError(f"SpaceClaim not found: {exe} (set spaceclaim.exe)")
    work = scdocx.parent
    args_file = work / "spaceclaim_args.json"
    report_file = work / "spaceclaim_report.json"
    args_file.write_text(json.dumps({
        "source": str(source.resolve()),
        "target": str(scdocx.resolve()),
        "report": str(report_file.resolve()),
        "fluid_body": sc.fluid_body,
        "flow_axis": sc.flow_axis,
        "inlet_name": sc.inlet_name,
        "outlet_name": sc.outlet_name,
        "required_bodies": required_bodies or [sc.fluid_body],
        "body_boxes": {n: [v / 1000.0 for v in box] for n, box in (body_boxes or {}).items()},
    }, indent=2))

    cmd = [str(exe), f"/RunScript={SCRIPT.resolve()}", "/ScriptAPI=252", "/Headless=True",
           "/Splash=False", "/Welcome=False", "/ExitAfterScript=True"]
    env = {**os.environ, "AUTOCFD_SC_ARGS": str(args_file.resolve())}
    # One retry: the first headless start after a while sometimes opens the part empty.
    for attempt in (1, 2):
        log.info("SpaceClaim: %s -> %s (attempt %d)", source.name, scdocx.name, attempt)
        report_file.unlink(missing_ok=True)
        try:
            subprocess.run(cmd, env=env, timeout=sc.timeout_s, check=False, cwd=work)
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"SpaceClaim did not finish in {sc.timeout_s} s") from e
        if not report_file.exists():
            raise RuntimeError("SpaceClaim finished without writing its report: the script did not run "
                               "(license? SpaceClaim already asking something on screen?)")
        report = json.loads(report_file.read_text())
        if report["ok"]:
            break
        log.warning("SpaceClaim script failed:\n%s", report["error"])
    else:
        raise RuntimeError(f"SpaceClaim script failed twice:\n{report['error']}")
    if not scdocx.exists():
        raise RuntimeError(f"SpaceClaim reported success but {scdocx} does not exist")
    log.info("SpaceClaim: bodies %s, named selections %s", report["bodies"], report["named_selections"])
    return scdocx
