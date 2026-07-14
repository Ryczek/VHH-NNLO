"""Publication plot styling (aligned with smeft_scan_workbench)."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Tuple

import matplotlib.pyplot as plt

# Colour-blind friendly palette (same as plots.py)
CB_PDF_FILL = "#0066CC"
CB_SCALE_EDGE = "#FF0000"
CB_SCALE_HATCH = "\\\\\\"
CB_CURVE = "#222222"

PROCESS_TITLES = {
    "WplusHH": r"$W^+ HH$",
    "WminusHH": r"$W^- HH$",
    "ZHH": "Zhh",
}


@dataclass
class PlotStyle:
    """Bundle typography, geometry, legend, and inset layout for scan plots."""

    axis_label_fontsize: float = 18
    tick_label_fontsize: float = 16
    legend_fontsize: float = 14
    title_fontsize: float = 18

    figsize_single: Tuple[float, float] = (8.0, 4.8)
    figsize_two_panel: Tuple[float, float] = (8.0, 7.0)
    panel_height_ratios: Tuple[float, float] = (2.6, 1.0)
    line_width: float = 2.2
    save_dpi: int = 300
    legend_framealpha: float = 0.92

    title_y: float = 1.02
    title_layout_top: float = 0.97
    x_pad_frac: float = 0.04
    inset_x_fraction: float = 0.01
    inset_scale: float = 1.5
    inset_width_frac: float = 0.20
    inset_height_frac: float = 0.18
    inset_margin: float = 0.04

    legend_h: str = "right"
    legend_v: str = "lower"
    inset_h: Optional[str] = "left"
    inset_v: str = "upper"

    grid_alpha: float = 0.25

    def legend_loc(self) -> str:
        return f"{self.legend_v} {self.legend_h}"

    def set_axis_labels(self, ax, *, xlabel: Optional[str] = None, ylabel: Optional[str] = None) -> None:
        if xlabel is not None:
            ax.set_xlabel(xlabel, fontsize=self.axis_label_fontsize)
        if ylabel is not None:
            ax.set_ylabel(ylabel, fontsize=self.axis_label_fontsize)
        ax.tick_params(axis="both", which="major", labelsize=self.tick_label_fontsize)

    def apply_xlim(self, ax, xmin: float, xmax: float) -> None:
        span = max(float(xmax) - float(xmin), 1e-12)
        pad = self.x_pad_frac * span
        ax.set_xlim(float(xmin) - pad, float(xmax) + pad)

    def apply_suptitle(self, fig, title: str) -> None:
        fig.suptitle(title, fontsize=self.title_fontsize, y=self.title_y, fontweight="normal")
        fig.subplots_adjust(top=self.title_layout_top)

    def apply_ax_title(self, ax, title: str) -> None:
        ax.set_title(title, fontsize=self.title_fontsize, pad=10)

    def make_legend(self, ax, *, ncol: int = 1):
        return ax.legend(
            loc=self.legend_loc(),
            fontsize=self.legend_fontsize,
            framealpha=self.legend_framealpha,
            ncol=ncol,
        )

    def finalize_figure(self, fig, *, two_panel: bool = False) -> None:
        if two_panel:
            fig.subplots_adjust(left=0.11, right=0.97, bottom=0.10, top=self.title_layout_top, hspace=0.06)
        else:
            fig.tight_layout(rect=[0, 0, 1, self.title_layout_top])

    def save_figure(self, fig, path, *, save: bool) -> None:
        if path is not None and save:
            fig.savefig(path, dpi=self.save_dpi, bbox_inches="tight")

    def inset_bbox(self) -> Tuple[float, float, float, float]:
        """Return ``(x0, y0, width, height)`` in axes coordinates for the inset box."""
        w = min(self.inset_width_frac * self.inset_scale, 0.48)
        h = min(self.inset_height_frac * self.inset_scale, 0.48)
        m = self.inset_margin
        if self.inset_h == "left":
            x0 = m
        elif self.inset_h == "right":
            x0 = 1.0 - w - m
        else:
            raise ValueError(f"inset_h must be 'left' or 'right', got {self.inset_h!r}")
        if self.inset_v == "upper":
            y0 = 1.0 - h - m
        elif self.inset_v == "lower":
            y0 = m
        else:
            raise ValueError(f"inset_v must be 'upper' or 'lower', got {self.inset_v!r}")
        return (x0, y0, w, h)


# Publication defaults: legend lower-right, inset upper-left (corners kept apart).
DEFAULT_PLOT_STYLE = PlotStyle()


def parse_corner_loc(loc: str) -> tuple[str, str]:
    """Parse ``\"lower right\"`` -> ``(vertical, horizontal)`` for ``legend_v`` / ``legend_h``."""
    parts = loc.strip().lower().split()
    if len(parts) != 2 or parts[0] not in ("upper", "lower") or parts[1] not in ("left", "right"):
        raise ValueError(f"Corner loc must be 'upper/lower left/right', got {loc!r}")
    return parts[0], parts[1]


def plot_style_with_layout(
    *,
    legend_loc: str = "lower right",
    inset_loc: Optional[str] = "upper left",
    sigma_inset: bool = True,
    base: Optional[PlotStyle] = None,
) -> PlotStyle:
    """Build ``PlotStyle`` from corner strings (e.g. ``legend_loc='lower right'``)."""
    legend_v, legend_h = parse_corner_loc(legend_loc)
    kwargs: dict = dict(legend_v=legend_v, legend_h=legend_h)
    if sigma_inset and inset_loc:
        inset_v, inset_h = parse_corner_loc(inset_loc)
        kwargs.update(inset_v=inset_v, inset_h=inset_h)
    else:
        kwargs["inset_h"] = None
    return replace(base or DEFAULT_PLOT_STYLE, **kwargs)


def default_plot_title(process: str, energy_tev: float, nnlo_label: str = "NNLO") -> str:
    """Main title line, e.g. ``Zhh @ NNLO QCD, sqrt(s) = 14.0 TeV``."""
    proc = PROCESS_TITLES.get(process, process)
    return rf"{proc} @ {nnlo_label} QCD, $\sqrt{{s}}$ = {float(energy_tev):g} TeV"


def scan_range_suffix(x_label: str, vmin: float, vmax: float) -> str:
    """Short scan-window label for filenames or subtitles."""
    return f"{x_label}_{float(vmin):g}_to_{float(vmax):g}"


def scan_plot_filename_stem(
    process: str,
    energy_tev: float,
    scan_x_key: str,
    vmin: float,
    vmax: float,
) -> str:
    return f"{process}_{float(energy_tev):g}TeV_{scan_x_key}_{float(vmin):g}_to_{float(vmax):g}"
