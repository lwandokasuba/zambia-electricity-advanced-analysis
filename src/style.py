"""Shared figure style: one palette and one save helper for every notebook figure.

Colour rules (same validated palette as the Assignment 4 dashboard):
focus country = orange, up to three named comparisons = blue / aqua / violet, others = grey;
magnitudes use one-hue sequential ramps, signed values a red-grey-blue diverging scale.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"

FOCUS = "#eb6834"
BLUE = "#2a78d6"
AQUA = "#1baf7a"
VIOLET = "#4a3aa7"
GREY = "#c3c2b7"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

BLUES = LinearSegmentedColormap.from_list("blues", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
ORANGES = LinearSegmentedColormap.from_list("oranges", ["#fde3d6", "#f39a70", "#eb6834", "#9a3a15", "#6b280e"])
DIVERGING = LinearSegmentedColormap.from_list("div", ["#9e1f18", "#e34948", "#f0efec", "#3987e5", "#104281"])


def apply() -> None:
    mpl.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GREY,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titleweight": "bold",
        "axes.titlesize": 12,
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "font.size": 10,
        "font.family": "sans-serif",
        "legend.frameon": False,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })


def save(fig, name: str) -> Path:
    FIG_DIR.mkdir(exist_ok=True)
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path)
    return path
