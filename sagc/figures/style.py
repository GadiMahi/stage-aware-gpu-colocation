"""Shared figure styling.

Targets an IEEE two-column paper: small, serif, thin marks, recessive axes, and
a categorical palette validated for colour-vision deficiency.

PALETTE
-------
The five policy colours are assigned in a FIXED order and never cycled. The
order below passed the six-check validator (lightness band, chroma floor,
adjacent-pair CVD separation, normal-vision floor, contrast) with no failures;
the two contrast warnings are relieved by direct labels and a legend, both of
which every figure here carries. Do not reorder without re-validating.

Every figure also renders a provenance footer, so a figure lifted out of the
repository and pasted into a slide still says whether it came from hardware.
"""
from __future__ import annotations

from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# fixed categorical order, validated for CVD
PALETTE: List[str] = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7"]

POLICY_COLOR: Dict[str, str] = {
    "exclusive": PALETTE[0],
    "blind": PALETTE[1],
    "greedy": PALETTE[2],
    "utilisation": PALETTE[3],
    "oracle": PALETTE[4],
    "bandit": "#444444",
}

POLICY_LABEL: Dict[str, str] = {
    "exclusive": "Exclusive (status quo)",
    "blind": "Blind sharing",
    "greedy": "Stage-aware (ours)",
    "utilisation": "Utilisation-driven",
    "oracle": "Oracle",
    "bandit": "Bandit (stretch)",
}

# single-hue sequential ramp for magnitude (light to dark), never a rainbow
SEQUENTIAL = "Blues"

INK = "#1a1a1a"
MUTED = "#6b6b6b"
GRID = "#d8d8d8"
SURFACE = "#ffffff"

ONE_COL = (3.4, 2.5)
ONE_COL_TALL = (3.4, 3.1)
TWO_COL = (7.0, 2.8)
TWO_COL_TALL = (7.0, 4.2)


def apply() -> None:
    plt.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.facecolor": SURFACE,
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "axes.labelcolor": INK,
        "axes.edgecolor": MUTED,
        "axes.linewidth": 0.6,
        "axes.facecolor": SURFACE,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "legend.fontsize": 7,
        "legend.frameon": False,
        "lines.linewidth": 1.4,
        "lines.markersize": 4,
        "patch.linewidth": 0.6,
    })


def footer(fig, kind: str, extra: str = "") -> None:
    """Stamp provenance onto a figure. Called by every figure function."""
    from ..common.provenance import figure_footer

    fig.text(0.005, 0.005, figure_footer(kind, extra), fontsize=5.5,
             color=MUTED, ha="left", va="bottom", style="italic")


def label_policy(p: str) -> str:
    return POLICY_LABEL.get(p, p)


def color_policy(p: str) -> str:
    return POLICY_COLOR.get(p, "#888888")
