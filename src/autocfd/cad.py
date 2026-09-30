"""SolidWorks stage: set the global variables of a COPY of the part and rebuild it.

The original .SLDPRT is never modified. Each geometry gets its own copy,
output/cases/<key>/geom_<key>.SLDPRT (unique name, because SolidWorks cannot open
two documents with the same file name).

Uses the SolidWorks API through COM (pywin32). If SolidWorks is already running,
the open session is reused (and left open); otherwise a hidden instance is
started and closed at the end.
"""

from __future__ import annotations

import logging
import re
import shutil
from contextlib import contextmanager
from pathlib import Path

from .config import Config

log = logging.getLogger(__name__)

SW_DOC_PART = 1
SW_OPEN_SILENT = 1
SW_SAVE_SILENT = 1
_GLOBAL_RE = re.compile(r'^\s*"([^"]+)"\s*=\s*(.+)$')


def _byref_int():
    import pythoncom
    import win32com.client

    return win32com.client.VARIANT(pythoncom.VT_BYREF | pythoncom.VT_I4, 0)


@contextmanager
def solidworks():
    """SolidWorks application (COM). Reuses a running instance when there is one."""
    import pythoncom
    import win32com.client

    pythoncom.CoInitialize()
    started = False
    try:
        app = win32com.client.Dispatch(win32com.client.GetActiveObject("SldWorks.Application"))
    except pythoncom.com_error:
        log.info("Starting SolidWorks (hidden)...")
        app = win32com.client.Dispatch("SldWorks.Application")
        app.Visible = False
        started = True
    try:
        yield app
    finally:
        if started:
            app.ExitApp
        pythoncom.CoUninitialize()


def _open(app, path: Path):
    err, warn = _byref_int(), _byref_int()
    doc = app.OpenDoc6(str(path.resolve()), SW_DOC_PART, SW_OPEN_SILENT, "", err, warn)
    if doc is None or err.value:
        raise RuntimeError(f"SolidWorks could not open {path} (error code {err.value})")
    return doc


def _set_equation(eqm, index: int, text: str) -> None:
    """IEquationMgr.Equation(index) = text. A parameterized property put is not
    reachable through late-bound attributes, so it is invoked directly
    (SetEquationAndConfigurationOption returns -1 for global variables)."""
    import pythoncom

    dispid = eqm._oleobj_.GetIDsOfNames("Equation")
    eqm._oleobj_.Invoke(dispid, 0, pythoncom.DISPATCH_PROPERTYPUT, False, index, text)
    if eqm.Equation(index).replace(" ", "") != text.replace(" ", ""):
        raise RuntimeError(f"SolidWorks rejected the equation {text!r}")


def global_variables(doc) -> dict[str, tuple[int, str, float]]:
    """{name: (index, equation text, current value)} of the part's global variables."""
    eqm = doc.GetEquationMgr
    out = {}
    for i in range(eqm.GetCount):
        if eqm.GlobalVariable(i):
            m = _GLOBAL_RE.match(eqm.Equation(i))
            if m:
                out[m.group(1)] = (i, eqm.Equation(i), float(eqm.Value(i)))
    return out


def list_parameters(sldprt: Path) -> dict[str, float]:
    """Global variables of the part (read only), to decide what to parametrize."""
    tmp = sldprt.with_name(f"_inspect_{sldprt.name}")
    shutil.copy2(sldprt, tmp)
    try:
        with solidworks() as app:
            doc = _open(app, tmp)
            try:
                return {k: v for k, (_, _, v) in global_variables(doc).items()}
            finally:
                app.CloseDoc(doc.GetTitle)
    finally:
        tmp.unlink(missing_ok=True)


def build(cfg: Config, case_dir: Path) -> Path:
    """Copy the part, apply [cad], rebuild and save. Returns the new .SLDPRT."""
    src = Path(cfg.paths.sldprt)
    if not src.exists():
        raise FileNotFoundError(f"paths.sldprt not found: {src}")
    case_dir.mkdir(parents=True, exist_ok=True)
    dst = case_dir / f"geom_{cfg.geometry_key()}.SLDPRT"
    shutil.copy2(src, dst)

    with solidworks() as app:
        doc = _open(app, dst)
        try:
            gv = global_variables(doc)
            missing = set(cfg.cad) - set(gv)
            if missing:
                raise KeyError(f"Global variables not found in the part: {sorted(missing)}; "
                               f"available: {sorted(gv)}")
            eqm = doc.GetEquationMgr
            for name, value in cfg.cad.items():
                _set_equation(eqm, gv[name][0], f'"{name}"= {value!r}')
            eqm.EvaluateAll  # no-argument methods are attributes in late-bound COM
            if not doc.ForceRebuild3(False):
                raise RuntimeError(f"SolidWorks rebuild failed for {cfg.cad}")

            errors = doc.Extension.GetWhatsWrongCount
            if errors:
                raise RuntimeError(f"SolidWorks rebuild has {errors} error(s) for {cfg.cad}: "
                                   "the combination of parameters is not valid")
            got = {k: v for k, (_, _, v) in global_variables(doc).items() if k in cfg.cad}
            bad = {k: (cfg.cad[k], v) for k, v in got.items() if abs(v - cfg.cad[k]) > 1e-6}
            if bad:
                raise RuntimeError(f"Values not applied (wanted, got): {bad}")

            err, warn = _byref_int(), _byref_int()
            if not doc.Save3(SW_SAVE_SILENT, err, warn):
                raise RuntimeError(f"SolidWorks could not save {dst} (error code {err.value})")
            log.info("SolidWorks: %s rebuilt with %s", dst.name, cfg.cad)
        finally:
            app.CloseDoc(doc.GetTitle)
    return dst
