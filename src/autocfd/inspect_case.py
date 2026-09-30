"""Read what a Fluent .cas.h5 exposes (named expressions, report definitions,
output parameters) straight from the HDF5 file, without starting Fluent."""

from __future__ import annotations

import re
from pathlib import Path

_EXPR_RE = re.compile(r'\(name \. "([^"]+)"\) \(definition \. "([^"]*)"\).*?\(input-parameter \. (#t|#f)\)')
_REPORT_RE = re.compile(r'name "([^"]+)" type "([a-z-]+)"')
_OUTPUT_RE = re.compile(r'\(name \(value \. "([^"]+)"\) \(type \. string-class\)\) \(settings \(report-defn-name \. "([^"]+)"\)')


def _settings_text(case: Path) -> str:
    import h5py

    with h5py.File(case, "r") as f:
        raw = f["settings"]["Rampant Variables"][()]
    if isinstance(raw, bytes):
        return raw.decode(errors="ignore")
    return b"".join(raw).decode(errors="ignore")


def case_info(case: str | Path) -> dict:
    text = _settings_text(Path(case))
    block = text[text.find("(named-expressions "):]
    block = block[:block.find("\n")] if "\n" in block else block
    exprs = [{"name": n, "definition": d, "input_parameter": p == "#t"}
             for n, d, p in _EXPR_RE.findall(block)]
    reports = dict(sorted(set(_REPORT_RE.findall(text))))
    outputs = dict(_OUTPUT_RE.findall(text))
    return {"named_expressions": exprs, "report_definitions": reports, "output_parameters": outputs}


def print_case_info(case: str | Path) -> None:
    info = case_info(case)
    print(f"\nBase case: {case}")
    print("  Named expressions ([fluent] section; input parameters marked *):")
    for e in info["named_expressions"]:
        print(f"    {'*' if e['input_parameter'] else ' '} {e['name']:<36} = {e['definition']}")
    print("  Report definitions ([solver] reports):")
    for name, kind in info["report_definitions"].items():
        print(f"      {name:<36} {kind}")
    if info["output_parameters"]:
        print("  Output parameters: " + ", ".join(info["output_parameters"]))
