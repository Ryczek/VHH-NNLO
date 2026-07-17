"""Publication-style scan plots."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np

from .plot_style import (
    CB_CURVE,
    CB_PDF_FILL,
    CB_SCALE_EDGE,
    CB_SCALE_HATCH,
    DEFAULT_PLOT_STYLE,
    PlotStyle,
)
from .smeft_operators import SMEFT_WC_LATEX

# HEFT κ + SMEFT Wilson-coefficient axes (shared plots for both notebooks)
X_LABELS = {
    "kappa_lambda": r"$\kappa_{\lambda}$",
    "kappa_w": r"$\kappa_{W}$",
    "kappa_z": r"$\kappa_{Z}$",
    "kappa_2w": r"$\kappa_{2W}$",
    "kappa_2z": r"$\kappa_{2Z}$",
    "kappa_t": r"$\kappa_{t}$",
    **{k: f"${v}$" for k, v in SMEFT_WC_LATEX.items()},
}

_SCAN_VALUE_KEYS = frozenset(
    {
        "sigma_lo",
        "sigma_nnlo",
        "k",
        "sigma_heft_over_sm_lo",
        "sigma_heft_over_sm_nnlo",
        "sigma_smeft_over_sm_lo",
        "sigma_smeft_over_sm_nnlo",
        "sigma_lo_pdfas",
        "sigma_lo_inf",
        "sigma_lo_sup",
        "sigma_nnlo_pdfas",
        "sigma_nnlo_inf",
        "sigma_nnlo_sup",
        "k_pdfas",
        "k_inf",
        "k_sup",
    }
)


def _style(style: Optional[PlotStyle]) -> PlotStyle:
    return style if style is not None else DEFAULT_PLOT_STYLE


def _uncertainty_band_zorders(pdf, inf, sup) -> tuple[int, int]:
    pdf_mag = float(np.mean(np.asarray(pdf, dtype=float)))
    scale_mag = float(
        np.mean(np.maximum(np.asarray(inf, dtype=float), np.asarray(sup, dtype=float)))
    )
    if scale_mag > pdf_mag:
        return 2, 3
    return 3, 2


def _fill_pdf(ax, x, y, delta, *, label: str, zorder: int = 2) -> None:
    ax.fill_between(
        x, y - delta, y + delta, color=CB_PDF_FILL, alpha=0.35, linewidth=0, label=label, zorder=zorder
    )


def _fill_scale(ax, x, y, lo, hi, *, label: str, zorder: int = 3) -> None:
    with plt.rc_context({"hatch.linewidth": 1.4}):
        ax.fill_between(
            x,
            y - lo,
            y + hi,
            facecolor=CB_SCALE_EDGE,
            alpha=0.30,
            edgecolor=CB_SCALE_EDGE,
            hatch=CB_SCALE_HATCH,
            linewidth=1.2,
            linestyle="--",
            label=label,
            zorder=zorder,
        )


def _fill_uncertainty_bands(
    ax,
    x,
    y,
    pdf,
    inf,
    sup,
    *,
    pdf_label: str = "PDF+$\\alpha_s$",
    scale_label: str = "scale envelope",
) -> None:
    z_pdf, z_scale = _uncertainty_band_zorders(pdf, inf, sup)
    _fill_pdf(ax, x, y, pdf, label=pdf_label, zorder=z_pdf)
    _fill_scale(ax, x, y, inf, sup, label=scale_label, zorder=z_scale)


def _k_ylim_from_scan(scan: Dict[str, np.ndarray], *, show_uncertainty: bool) -> tuple[float, float]:
    k = np.asarray(scan["k"], dtype=float)
    if show_uncertainty:
        ylo = float(np.min(k - np.asarray(scan["k_inf"], dtype=float) - np.asarray(scan["k_pdfas"], dtype=float)))
        yhi = float(np.max(k + np.asarray(scan["k_sup"], dtype=float) + np.asarray(scan["k_pdfas"], dtype=float)))
    else:
        ylo, yhi = float(np.min(k)), float(np.max(k))
    pad = 0.06 * max(yhi - ylo, 1e-6)
    return ylo - pad, yhi + pad


def _infer_x_key(scan: Dict[str, np.ndarray]) -> str:
    for key in X_LABELS:
        if key in scan:
            return key
    # Fallback: first non-value array column
    for key in scan:
        if key not in _SCAN_VALUE_KEYS and hasattr(scan[key], "__len__"):
            return key
    raise KeyError(f"Could not infer scan x-axis key from {sorted(scan)}")


def _scan_xlim(
    scan: Dict[str, np.ndarray],
    x_key: str,
    *,
    xmin: Optional[float],
    xmax: Optional[float],
) -> tuple[float, float]:
    x = np.asarray(scan[x_key], dtype=float)
    return (float(xmin) if xmin is not None else float(np.min(x)), float(xmax) if xmax is not None else float(np.max(x)))


def _apply_scan_xlim(ax, scan: Dict[str, np.ndarray], x_key: str, style: PlotStyle, *, xmin, xmax) -> None:
    lo, hi = _scan_xlim(scan, x_key, xmin=xmin, xmax=xmax)
    style.apply_xlim(ax, lo, hi)


def _overlay_simulation(ax, x: np.ndarray, y: np.ndarray, *, label: str = "simulation") -> None:
    mask = np.isfinite(x) & np.isfinite(y)
    if not np.any(mask):
        return
    ax.scatter(
        x[mask],
        y[mask],
        s=42,
        marker="o",
        facecolors="none",
        edgecolors="#CC6600",
        linewidths=1.6,
        label=label,
        zorder=5,
    )


def _inset_connector_ys(inset_v: str, ylo_z: float, yhi_z: float) -> tuple[float, float]:
    """Y on the zoom box (data) and on the inset axes (0–1) for connector lines.

    Upper inset → from top of zoom box to lower edge of inset.
    Lower inset → from bottom of zoom box to upper edge of inset.
    """
    if inset_v == "lower":
        return ylo_z, 1.0
    return yhi_z, 0.0


def _add_sigma_inset(
    ax,
    x,
    sigma,
    pdfas,
    inf,
    sup,
    *,
    style: PlotStyle,
    xlim: Optional[tuple[float, float]] = None,
) -> None:
    from matplotlib.patches import ConnectionPatch, Rectangle
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    if style.inset_h is None:
        return

    x = np.asarray(x, dtype=float)
    x_min, x_max = float(np.min(x)), float(np.max(x))
    span = max(x_max - x_min, 1e-12)
    if xlim is None:
        xlo, xhi = x_min, x_min + float(style.inset_x_fraction) * span
    else:
        xlo, xhi = xlim

    x_plot = np.linspace(xlo, xhi, 120)
    sm = np.interp(x_plot, x, sigma)
    pdf_plot = np.interp(x_plot, x, pdfas)
    inf_plot = np.interp(x_plot, x, inf)
    sup_plot = np.interp(x_plot, x, sup)

    bbox = style.inset_bbox()
    axins = inset_axes(
        ax,
        width="100%",
        height="100%",
        bbox_to_anchor=bbox,
        bbox_transform=ax.transAxes,
        loc="lower left",
        borderpad=0,
    )

    lw_inset = max(style.line_width * 0.65, 1.2)
    axins.plot(x_plot, sm, color=CB_CURVE, lw=lw_inset)
    _fill_uncertainty_bands(
        axins, x_plot, sm, pdf_plot, inf_plot, sup_plot, pdf_label="_inset", scale_label="_inset"
    )
    ylo = float(np.min(sm - pdf_plot - inf_plot))
    yhi = float(np.max(sm + pdf_plot + sup_plot))
    pad = 0.08 * max(yhi - ylo, 1e-6)
    ylo_z, yhi_z = ylo - pad, yhi + pad
    axins.set_xlim(xlo, xhi)
    axins.set_ylim(ylo_z, yhi_z)
    axins.set_xticks([])
    axins.set_yticks([])
    axins.tick_params(labelsize=max(style.tick_label_fontsize - 4, 8))
    axins.grid(alpha=style.grid_alpha * 0.8)

    line_kw = dict(linestyle="--", color="0.45", lw=0.8, clip_on=False)
    ax.add_patch(
        Rectangle((xlo, ylo_z), xhi - xlo, yhi_z - ylo_z, fill=False, transform=ax.transData, **line_kw)
    )
    fig = ax.figure
    y_data, y_inset = _inset_connector_ys(style.inset_v, ylo_z, yhi_z)
    # Two connectors: left→left and right→right on the facing edge of the inset
    for x_data, x_inset in ((xlo, 0.0), (xhi, 1.0)):
        fig.add_artist(
            ConnectionPatch(
                xyA=(x_data, y_data),
                coordsA=ax.transData,
                xyB=(x_inset, y_inset),
                coordsB=axins.transAxes,
                **line_kw,
            )
        )


def _make_two_panel_figure(style: PlotStyle):
    return plt.subplots(
        2,
        1,
        figsize=style.figsize_two_panel,
        sharex=True,
        gridspec_kw={"height_ratios": list(style.panel_height_ratios)},
    )


def _finalize_two_panel(fig, style: PlotStyle, *, title: Optional[str]) -> None:
    if title:
        style.apply_suptitle(fig, title)
    else:
        fig.subplots_adjust(top=0.98)
    style.finalize_figure(fig, two_panel=True)


def _clear_legend(ax) -> None:
    leg = ax.get_legend()
    if leg is not None:
        leg.remove()


def _plot_sigma_panel(
    ax,
    scan: Dict[str, np.ndarray],
    x,
    *,
    style: PlotStyle,
    nnlo_label: str,
    sigma_inset: bool,
    inset_xlim: Optional[tuple[float, float]],
    sim_scan: Optional[Dict[str, np.ndarray]],
    xmin: Optional[float],
    xmax: Optional[float],
    x_key: str,
) -> None:
    ax.plot(
        x,
        scan["sigma_nnlo"],
        color=CB_CURVE,
        lw=style.line_width,
        label=rf"$\sigma_{{\mathrm{{{nnlo_label}}}}}$",
        zorder=4,
    )
    if "sigma_nnlo_pdfas" in scan:
        _fill_uncertainty_bands(
            ax,
            x,
            scan["sigma_nnlo"],
            scan["sigma_nnlo_pdfas"],
            scan["sigma_nnlo_inf"],
            scan["sigma_nnlo_sup"],
        )
    if sim_scan is not None and len(sim_scan.get("x", [])):
        _overlay_simulation(ax, sim_scan["x"], sim_scan["sigma_nnlo"])
    style.set_axis_labels(ax, ylabel=rf"$\sigma_{{\mathrm{{{nnlo_label}}}}}$ [fb]")
    ax.grid(alpha=style.grid_alpha)
    style.make_legend(ax)
    _apply_scan_xlim(ax, scan, x_key, style, xmin=xmin, xmax=xmax)
    if sigma_inset and "sigma_nnlo_pdfas" in scan:
        _add_sigma_inset(
            ax,
            x,
            scan["sigma_nnlo"],
            scan["sigma_nnlo_pdfas"],
            scan["sigma_nnlo_inf"],
            scan["sigma_nnlo_sup"],
            style=style,
            xlim=inset_xlim,
        )


def plot_sigma_only(
    scan: Dict[str, np.ndarray],
    *,
    output: Optional[Path] = None,
    title: Optional[str] = None,
    x_key: Optional[str] = None,
    x_label: Optional[str] = None,
    nnlo_label: str = "NNLO",
    sigma_inset: bool = True,
    sigma_inset_xlim: Optional[tuple[float, float]] = None,
    sim_scan: Optional[Dict[str, np.ndarray]] = None,
    xmin: Optional[float] = None,
    xmax: Optional[float] = None,
    style: Optional[PlotStyle] = None,
    save: bool = True,
) -> plt.Figure:
    """Single-panel $\\sigma_{\\mathrm{NNLO}}$ scan with uncertainty bands."""
    st = _style(style)
    key = x_key or _infer_x_key(scan)
    x = scan[key]
    x_label = x_label or X_LABELS.get(key, key)

    fig, ax = plt.subplots(1, 1, figsize=st.figsize_single)
    if title:
        st.apply_ax_title(ax, title)

    _plot_sigma_panel(
        ax,
        scan,
        x,
        style=st,
        nnlo_label=nnlo_label,
        sigma_inset=sigma_inset,
        inset_xlim=sigma_inset_xlim,
        sim_scan=sim_scan,
        xmin=xmin,
        xmax=xmax,
        x_key=key,
    )
    st.set_axis_labels(ax, xlabel=x_label)
    fig.tight_layout()
    st.save_figure(fig, output, save=save)
    return fig


def plot_sigma_lo_only(
    scan: Dict[str, np.ndarray],
    *,
    output: Optional[Path] = None,
    title: Optional[str] = None,
    x_key: Optional[str] = None,
    x_label: Optional[str] = None,
    xmin: Optional[float] = None,
    xmax: Optional[float] = None,
    style: Optional[PlotStyle] = None,
    save: bool = True,
) -> plt.Figure:
    """Single-panel $\\sigma_{\\mathrm{LO}}$ scan with uncertainty bands."""
    st = _style(style)
    key = x_key or _infer_x_key(scan)
    x = scan[key]
    x_label = x_label or X_LABELS.get(key, key)

    fig, ax = plt.subplots(1, 1, figsize=st.figsize_single)
    if title:
        st.apply_ax_title(ax, title)

    ax.plot(x, scan["sigma_lo"], color=CB_CURVE, lw=st.line_width, label=r"$\sigma_{\mathrm{LO}}$", zorder=4)
    if "sigma_lo_pdfas" in scan:
        _fill_uncertainty_bands(
            ax,
            x,
            scan["sigma_lo"],
            scan["sigma_lo_pdfas"],
            scan["sigma_lo_inf"],
            scan["sigma_lo_sup"],
        )
    st.set_axis_labels(ax, xlabel=x_label, ylabel=r"$\sigma_{\mathrm{LO}}$ [fb]")
    st.make_legend(ax)
    ax.grid(alpha=st.grid_alpha)
    _apply_scan_xlim(ax, scan, key, st, xmin=xmin, xmax=xmax)
    fig.tight_layout()
    st.save_figure(fig, output, save=save)
    return fig


def plot_sigma_nnlo_and_enhancement_nnlo(
    scan: Dict[str, np.ndarray],
    *,
    output: Optional[Path] = None,
    title: Optional[str] = None,
    x_key: Optional[str] = None,
    x_label: Optional[str] = None,
    nnlo_label: str = "NNLO",
    sigma_inset: bool = True,
    sigma_inset_xlim: Optional[tuple[float, float]] = None,
    show_enhancement_uncertainty: bool = False,
    xmin: Optional[float] = None,
    xmax: Optional[float] = None,
    style: Optional[PlotStyle] = None,
    save: bool = True,
) -> plt.Figure:
    """Two panels: $\\sigma_{\\mathrm{NNLO}}$ and $\\sigma_{\\mathrm{HEFT}}/\\sigma_{\\mathrm{SM}}$ at NNLO."""
    st = _style(style)
    key = x_key or _infer_x_key(scan)
    x = scan[key]
    x_label = x_label or X_LABELS.get(key, key)
    ykey, pdfkey, infkey, supkey, order_label = _enhancement_series(scan, nnlo_label, nnlo_label)

    fig, (ax0, ax1) = _make_two_panel_figure(st)
    _plot_sigma_panel(
        ax0,
        scan,
        x,
        style=st,
        nnlo_label=nnlo_label,
        sigma_inset=sigma_inset,
        inset_xlim=sigma_inset_xlim,
        sim_scan=None,
        xmin=xmin,
        xmax=xmax,
        x_key=key,
    )

    y = scan[ykey]
    ax1.axhline(1.0, color="0.55", ls=":", lw=1.0, zorder=1)
    ax1.plot(x, y, color=CB_CURVE, lw=st.line_width, zorder=4)
    if show_enhancement_uncertainty and pdfkey in scan:
        _fill_uncertainty_bands(
            ax1,
            x,
            y,
            scan[pdfkey],
            scan[infkey],
            scan[supkey],
            pdf_label="_nolegend_",
            scale_label="_nolegend_",
        )
    st.set_axis_labels(
        ax1,
        xlabel=x_label,
        ylabel=rf"$\sigma_{{\mathrm{{{order_label}}}}}/\sigma_{{\mathrm{{{order_label}}}}}^{{\mathrm{{SM}}}}$",
    )
    if show_enhancement_uncertainty and pdfkey in scan:
        ax1.set_ylim(_ylim_from_series(y, scan[pdfkey], scan[infkey], scan[supkey]))
    else:
        pad = 0.06 * max(float(np.max(y) - np.min(y)), 1e-6)
        ax1.set_ylim(float(np.min(y)) - pad, float(np.max(y)) + pad)
    ax1.grid(alpha=st.grid_alpha)
    _clear_legend(ax1)
    _apply_scan_xlim(ax1, scan, key, st, xmin=xmin, xmax=xmax)

    _finalize_two_panel(fig, st, title=title)
    st.save_figure(fig, output, save=save)
    return fig


def _enhancement_series(scan: Dict[str, np.ndarray], order: str, nnlo_label: str) -> tuple[str, str, str, str, str]:
    order_u = order.upper()
    if order_u == "LO":
        ykey = "enhancement_lo" if "enhancement_lo" in scan else "sigma_heft_over_sm_lo"
        return (
            ykey,
            "enhancement_lo_pdfas",
            "enhancement_lo_inf",
            "enhancement_lo_sup",
            "LO",
        )
    if order_u in ("NNLO", "HHZ"):
        ykey = "enhancement_nnlo" if "enhancement_nnlo" in scan else "sigma_heft_over_sm_nnlo"
        return (
            ykey,
            "enhancement_nnlo_pdfas",
            "enhancement_nnlo_inf",
            "enhancement_nnlo_sup",
            nnlo_label,
        )
    raise ValueError("order must be LO or NNLO/HHZ")


def plot_enhancement_only(
    scan: Dict[str, np.ndarray],
    *,
    order: str = "NNLO",
    output: Optional[Path] = None,
    title: Optional[str] = None,
    x_key: Optional[str] = None,
    x_label: Optional[str] = None,
    nnlo_label: str = "NNLO",
    show_uncertainty: bool = True,
    xmin: Optional[float] = None,
    xmax: Optional[float] = None,
    style: Optional[PlotStyle] = None,
    save: bool = True,
) -> plt.Figure:
    """Single-panel $\\sigma_{\\mathrm{HEFT}}/\\sigma_{\\mathrm{SM}}$ scan at LO or NNLO."""
    st = _style(style)
    key = x_key or _infer_x_key(scan)
    x = scan[key]
    x_label = x_label or X_LABELS.get(key, key)
    ykey, pdfkey, infkey, supkey, order_label = _enhancement_series(scan, order, nnlo_label)

    fig, ax = plt.subplots(1, 1, figsize=st.figsize_single)
    if title:
        st.apply_ax_title(ax, title)

    y = scan[ykey]
    ax.axhline(1.0, color="0.55", ls=":", lw=1.0, zorder=1)
    ax.plot(
        x,
        y,
        color=CB_CURVE,
        lw=st.line_width,
        label=rf"$\sigma_{{\mathrm{{{order_label}}}}}/\sigma_{{\mathrm{{{order_label}}}}}^{{\mathrm{{SM}}}}$",
        zorder=4,
    )
    if show_uncertainty and pdfkey in scan:
        _fill_uncertainty_bands(ax, x, y, scan[pdfkey], scan[infkey], scan[supkey])
    st.set_axis_labels(
        ax,
        xlabel=x_label,
        ylabel=rf"$\sigma_{{\mathrm{{{order_label}}}}}/\sigma_{{\mathrm{{{order_label}}}}}^{{\mathrm{{SM}}}}$",
    )
    if show_uncertainty and pdfkey in scan:
        ax.set_ylim(_ylim_from_series(y, scan[pdfkey], scan[infkey], scan[supkey]))
    else:
        pad = 0.06 * max(float(np.max(y) - np.min(y)), 1e-6)
        ax.set_ylim(float(np.min(y)) - pad, float(np.max(y)) + pad)
    st.make_legend(ax)
    ax.grid(alpha=st.grid_alpha)
    _apply_scan_xlim(ax, scan, key, st, xmin=xmin, xmax=xmax)
    fig.tight_layout()
    st.save_figure(fig, output, save=save)
    return fig


def plot_sigma_nnlo_and_kfactor(
    scan: Dict[str, np.ndarray],
    *,
    output: Optional[Path] = None,
    title: Optional[str] = None,
    x_key: Optional[str] = None,
    x_label: Optional[str] = None,
    nnlo_label: str = "NNLO",
    show_k_uncertainty: bool = False,
    sigma_inset: bool = True,
    sigma_inset_xlim: Optional[tuple[float, float]] = None,
    sim_scan: Optional[Dict[str, np.ndarray]] = None,
    k_ylim: Optional[tuple[float, float]] = None,
    xmin: Optional[float] = None,
    xmax: Optional[float] = None,
    style: Optional[PlotStyle] = None,
    save: bool = True,
) -> plt.Figure:
    st = _style(style)
    key = x_key or _infer_x_key(scan)
    x = scan[key]
    x_label = x_label or X_LABELS.get(key, key)

    fig, (ax0, ax1) = _make_two_panel_figure(st)
    _plot_sigma_panel(
        ax0,
        scan,
        x,
        style=st,
        nnlo_label=nnlo_label,
        sigma_inset=sigma_inset,
        inset_xlim=sigma_inset_xlim,
        sim_scan=sim_scan,
        xmin=xmin,
        xmax=xmax,
        x_key=key,
    )

    ax1.plot(x, scan["k"], color=CB_CURVE, lw=st.line_width, zorder=4)
    if show_k_uncertainty:
        _fill_uncertainty_bands(ax1, x, scan["k"], scan["k_pdfas"], scan["k_inf"], scan["k_sup"])
    if sim_scan is not None and len(sim_scan.get("x", [])):
        sim_k = sim_scan["sigma_nnlo"] / sim_scan["sigma_lo"]
        _overlay_simulation(ax1, sim_scan["x"], sim_k, label="simulation $K$")
    st.set_axis_labels(ax1, xlabel=x_label, ylabel=rf"$K_{{\mathrm{{{nnlo_label}}}}}$")
    ax1.set_ylim(*(k_ylim or _k_ylim_from_scan(scan, show_uncertainty=show_k_uncertainty)))
    ax1.grid(alpha=st.grid_alpha)
    _clear_legend(ax1)
    _apply_scan_xlim(ax1, scan, key, st, xmin=xmin, xmax=xmax)

    _finalize_two_panel(fig, st, title=title)
    st.save_figure(fig, output, save=save)
    return fig


def _ylim_from_series(y, pdf, inf, sup) -> tuple[float, float]:
    y = np.asarray(y, dtype=float)
    ylo = float(np.min(y - np.maximum(np.asarray(inf, dtype=float), np.asarray(pdf, dtype=float))))
    yhi = float(np.max(y + np.maximum(np.asarray(sup, dtype=float), np.asarray(pdf, dtype=float))))
    pad = 0.06 * max(yhi - ylo, 1e-6)
    return ylo - pad, yhi + pad


def plot_sm_enhancement(
    scan: Dict[str, np.ndarray],
    *,
    output: Optional[Path] = None,
    title: Optional[str] = None,
    x_key: Optional[str] = None,
    x_label: Optional[str] = None,
    nnlo_label: str = "NNLO",
    show_uncertainty: bool = True,
    xmin: Optional[float] = None,
    xmax: Optional[float] = None,
    style: Optional[PlotStyle] = None,
    save: bool = True,
) -> plt.Figure:
    """Two-panel $\\sigma_{\\mathrm{HEFT}}/\\sigma_{\\mathrm{SM}}$ for LO and NNLO."""
    st = _style(style)
    key = x_key or _infer_x_key(scan)
    x = scan[key]
    x_label = x_label or X_LABELS.get(key, key)

    fig, (ax0, ax1) = plt.subplots(
        2, 1, figsize=st.figsize_two_panel, sharex=True, gridspec_kw={"height_ratios": [1.0, 1.0]}
    )
    if title:
        st.apply_suptitle(fig, title)

    for ax, order in ((ax0, "LO"), (ax1, nnlo_label)):
        ykey, pdfkey, infkey, supkey, order_label = _enhancement_series(scan, order, nnlo_label)
        y = scan[ykey]
        ax.axhline(1.0, color="0.55", ls=":", lw=1.0, zorder=1)
        ax.plot(
            x,
            y,
            color=CB_CURVE,
            lw=st.line_width,
            label=rf"$\sigma_{{\mathrm{{{order_label}}}}}/\sigma_{{\mathrm{{{order_label}}}}}^{{\mathrm{{SM}}}}$",
            zorder=4,
        )
        if show_uncertainty and pdfkey in scan:
            _fill_uncertainty_bands(ax, x, y, scan[pdfkey], scan[infkey], scan[supkey])
        st.set_axis_labels(
            ax, ylabel=rf"$\sigma_{{\mathrm{{{order_label}}}}}/\sigma_{{\mathrm{{{order_label}}}}}^{{\mathrm{{SM}}}}$"
        )
        if show_uncertainty and pdfkey in scan:
            ax.set_ylim(_ylim_from_series(y, scan[pdfkey], scan[infkey], scan[supkey]))
        else:
            pad = 0.06 * max(float(np.max(y) - np.min(y)), 1e-6)
            ax.set_ylim(float(np.min(y)) - pad, float(np.max(y)) + pad)
        st.make_legend(ax)
        ax.grid(alpha=st.grid_alpha)

    st.set_axis_labels(ax1, xlabel=x_label)
    _apply_scan_xlim(ax1, scan, key, st, xmin=xmin, xmax=xmax)
    st.finalize_figure(fig)
    st.save_figure(fig, output, save=save)
    return fig


def plot_kfactor_only(
    scan: Dict[str, np.ndarray],
    *,
    output: Optional[Path] = None,
    title: Optional[str] = None,
    x_key: Optional[str] = None,
    x_label: Optional[str] = None,
    nnlo_label: str = "NNLO",
    show_uncertainty: bool = False,
    sim_scan: Optional[Dict[str, np.ndarray]] = None,
    k_ylim: Optional[tuple[float, float]] = None,
    xmin: Optional[float] = None,
    xmax: Optional[float] = None,
    style: Optional[PlotStyle] = None,
    save: bool = True,
) -> plt.Figure:
    st = _style(style)
    key = x_key or _infer_x_key(scan)
    x = np.asarray(scan[key], dtype=float)
    x_label = x_label or X_LABELS.get(key, key)

    fig, ax = plt.subplots(1, 1, figsize=st.figsize_single)
    if title:
        st.apply_ax_title(ax, title)
    ax.plot(x, scan["k"], color=CB_CURVE, lw=st.line_width, label=rf"$K_{{\mathrm{{{nnlo_label}}}}}$")
    if show_uncertainty:
        _fill_uncertainty_bands(ax, x, scan["k"], scan["k_pdfas"], scan["k_inf"], scan["k_sup"])
    if sim_scan is not None and len(sim_scan.get("x", [])):
        sim_k = sim_scan["sigma_nnlo"] / sim_scan["sigma_lo"]
        _overlay_simulation(ax, sim_scan["x"], sim_k, label="simulation $K$")
    st.make_legend(ax)
    st.set_axis_labels(ax, xlabel=x_label, ylabel=rf"$K_{{\mathrm{{{nnlo_label}}}}}$")
    ax.set_ylim(*(k_ylim or _k_ylim_from_scan(scan, show_uncertainty=show_uncertainty)))
    ax.grid(alpha=st.grid_alpha)
    _apply_scan_xlim(ax, scan, key, st, xmin=xmin, xmax=xmax)
    fig.tight_layout()
    st.save_figure(fig, output, save=save)
    return fig
