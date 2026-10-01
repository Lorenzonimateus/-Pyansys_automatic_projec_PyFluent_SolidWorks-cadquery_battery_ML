"""Plot style shared by the project's figures.

Contour maps (Fluent style): jet colormap, vertical colorbar on the left with
"<Variable> [unit]" above it, 11 equally spaced ticks with one decimal place,
white background, no grid, titles with commas (no dashes).
Line charts: one hue per series in a fixed order, a legend, a single y axis.
"""

from __future__ import annotations

import numpy as np

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK_2 = "#000000", "#52514e"


def plt_agg():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def contour(fig, ax, X, Y, Z, label: str, title: str, vmin: float | None = None, vmax: float | None = None,
            xlabel: str = "x [mm]", ylabel: str = "z [mm]"):
    """Fluent-style map on `ax` (NaN = white)."""
    from matplotlib.ticker import FormatStrFormatter

    finite = Z[np.isfinite(Z)]
    vmin = float(finite.min()) if vmin is None else vmin
    vmax = float(finite.max()) if vmax is None else vmax
    if vmax - vmin < 1e-9:
        vmax = vmin + 1e-9
    m = ax.pcolormesh(X, Y, np.ma.masked_invalid(Z), cmap="jet", vmin=vmin, vmax=vmax, shading="auto")
    ax.set_facecolor("white")
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_title(title, fontsize=10, loc="left", color=INK, pad=20)  # above the colorbar label
    ax.set_xlabel(xlabel, fontsize=9, color=INK_2)
    ax.set_ylabel(ylabel, fontsize=9, color=INK_2)
    ax.tick_params(labelsize=8, colors=INK_2)
    # colorbar on the left, same height as the map, a fixed distance (inches) away so the
    # map's own y labels fit between them whatever the map's width
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    cax = make_axes_locatable(ax).append_axes("left", size=0.18, pad=1.3)
    cb = fig.colorbar(m, cax=cax)
    cb.set_ticks(np.linspace(vmin, vmax, 11))
    cb.ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    cb.ax.tick_params(labelsize=8)
    cb.ax.set_title(label, fontsize=8, loc="left")
    return m


def style_lines(ax, xlabel: str, ylabel: str, title: str):
    ax.set_xlabel(xlabel, fontsize=9, color=INK_2)
    ax.set_ylabel(ylabel, fontsize=9, color=INK_2)
    ax.set_title(title, fontsize=11, loc="left", color=INK)
    ax.grid(False)
    ax.tick_params(labelsize=8, colors=INK_2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
