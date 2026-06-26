"""Publication-style scan plots (adapted from sigma_kfactor_workbench)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

CB_PDF_FILL = "#0066CC"
CB_SCALE_EDGE = "#FF0000"
CB_SCALE_HATCH = "\\\\\\"
CB_CURVE = "#222222"
INSET_LEGEND_SCALE = 1.5
AXIS_LABEL_FONTSIZE = 13
TICK_LABEL_FONTSIZE = 11

X_LABELS = {
    "kappa_lambda": r"$\kappa_{\lambda}$",
    "kappa_w": r"$\kappa_{W}$",
    "kappa_z": r"$\kappa_{Z}$",
    "kappa_2w": r"$\kappa_{2W}$",
    "kappa_2z": r"$\kappa_{2Z}$",
    "kappa_t": r"$\kappa_{t}$",
}


def _set_axis_labels(ax, *, xlabel: Optional[str] = None, ylabel: Optional[str] = None) -> None:
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=AXIS_LABEL_FONTSIZE)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="both", which="major", labelsize=TICK_LABEL_FONTSIZE)


def _uncertainty_band_zorders(pdf, inf, sup) -> tuple[int, int]:
    """Return (pdf_zorder, scale_zorder); draw the wider band on top."""
    pdf_mag = float(np.mean(np.asarray(pdf, dtype=float)))
    scale_mag = float(
        np.mean(np.maximum(np.asarray(inf, dtype=float), np.asarray(sup, dtype=float)))
    )
    if scale_mag > pdf_mag:
        return 2, 3
    return 3, 2


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


def _fill_pdf(ax, x, y, delta, *, label: str, zorder: int = 2) -> None:
    ax.fill_between(x, y - delta, y + delta, color=CB_PDF_FILL, alpha=0.35, linewidth=0, label=label, zorder=zorder)


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
    for key in (
        "kappa_lambda",
        "kappa_w",
        "kappa_z",
        "kappa_2w",
        "kappa_2z",
        "kappa_t",
    ):
        if key in scan:
            return key
    raise KeyError("Could not infer scan x-axis key")


def _overlay_simulation(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    *,
    label: str = "simulation",
) -> None:
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


def _add_sigma_inset(
    ax,
    x,
    sigma,
    pdfas,
    inf,
    sup,
    *,
    legend=None,
    xlim: Optional[tuple[float, float]] = None,
    x_fraction: float = 0.01,
) -> None:
    from matplotlib.patches import ConnectionPatch, Rectangle
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    x = np.asarray(x, dtype=float)
    x_min, x_max = float(np.min(x)), float(np.max(x))
    span = max(x_max - x_min, 1e-12)
    if xlim is None:
        xlo, xhi = x_min, x_min + float(x_fraction) * span
    else:
        xlo, xhi = xlim
    x_plot = np.linspace(xlo, xhi, 120)
    sm = np.interp(x_plot, x, sigma)
    pdf_plot = np.interp(x_plot, x, pdfas)
    inf_plot = np.interp(x_plot, x, inf)
    sup_plot = np.interp(x_plot, x, sup)

    fig = ax.figure
    if legend is not None:
        fig.canvas.draw()
        leg_bbox = legend.get_window_extent().transformed(ax.transAxes.inverted())
        gap = 0.015
        w = float(leg_bbox.width) * INSET_LEGEND_SCALE
        h = float(leg_bbox.height) * INSET_LEGEND_SCALE
        x_anchor = float(np.clip(leg_bbox.x0, 0.0, 1.0 - w))
        y_anchor = float(np.clip(leg_bbox.y0 - h - gap, 0.02, 1.0 - h))
        axins = inset_axes(
            ax,
            width="100%",
            height="100%",
            bbox_to_anchor=(x_anchor, y_anchor, w, h),
            bbox_transform=ax.transAxes,
            loc="lower left",
            borderpad=0,
        )
    else:
        axins = inset_axes(ax, width="28%", height="22%", loc="upper right", borderpad=0.8)

    axins.plot(x_plot, sm, color=CB_CURVE, lw=1.4)
    _fill_uncertainty_bands(axins, x_plot, sm, pdf_plot, inf_plot, sup_plot, pdf_label="_inset", scale_label="_inset")
    ylo = float(np.min(sm - pdf_plot - inf_plot))
    yhi = float(np.max(sm + pdf_plot + sup_plot))
    pad = 0.08 * max(yhi - ylo, 1e-6)
    ylo_z, yhi_z = ylo - pad, yhi + pad
    axins.set_xlim(xlo, xhi)
    axins.set_ylim(ylo_z, yhi_z)
    axins.set_xticks([])
    axins.set_yticks([])
    axins.grid(alpha=0.2)

    line_kw = dict(linestyle="--", color="0.45", lw=0.8, clip_on=False)
    ax.add_patch(Rectangle((xlo, ylo_z), xhi - xlo, yhi_z - ylo_z, fill=False, transform=ax.transData, **line_kw))
    for x_corner, inset_x in ((xlo, 0.0), (xhi, 1.0)):
        fig.add_artist(
            ConnectionPatch(
                xyA=(x_corner, yhi_z),
                coordsA=ax.transData,
                xyB=(inset_x, 0.0),
                coordsB=axins.transAxes,
                **line_kw,
            )
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
    save: bool = True,
) -> plt.Figure:
    """Single-panel $\\sigma_{\\mathrm{NNLO}}$ scan with uncertainty bands."""
    key = x_key or _infer_x_key(scan)
    x = scan[key]
    x_label = x_label or X_LABELS.get(key, key)

    fig, ax = plt.subplots(1, 1, figsize=(8, 4.8), constrained_layout=True)
    if title:
        ax.set_title(title)

    ax.plot(x, scan["sigma_nnlo"], color=CB_CURVE, lw=2, label=rf"$\sigma_{{\mathrm{{{nnlo_label}}}}}$", zorder=4)
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
    _set_axis_labels(ax, xlabel=x_label, ylabel=rf"$\sigma_{{\mathrm{{{nnlo_label}}}}}$ [fb]")
    ax.grid(alpha=0.25)
    legend = ax.legend(loc="best")
    if sigma_inset:
        _add_sigma_inset(
            ax,
            x,
            scan["sigma_nnlo"],
            scan["sigma_nnlo_pdfas"],
            scan["sigma_nnlo_inf"],
            scan["sigma_nnlo_sup"],
            legend=legend,
            xlim=sigma_inset_xlim,
        )

    if output and save:
        fig.savefig(output, dpi=200, bbox_inches="tight")
    return fig


def plot_sigma_lo_only(
    scan: Dict[str, np.ndarray],
    *,
    output: Optional[Path] = None,
    title: Optional[str] = None,
    x_key: Optional[str] = None,
    x_label: Optional[str] = None,
    save: bool = True,
) -> plt.Figure:
    """Single-panel $\\sigma_{\\mathrm{LO}}$ scan with uncertainty bands."""
    key = x_key or _infer_x_key(scan)
    x = scan[key]
    x_label = x_label or X_LABELS.get(key, key)

    fig, ax = plt.subplots(1, 1, figsize=(8, 4.8), constrained_layout=True)
    if title:
        ax.set_title(title)

    ax.plot(x, scan["sigma_lo"], color=CB_CURVE, lw=2, label=r"$\sigma_{\mathrm{LO}}$", zorder=4)
    if "sigma_lo_pdfas" in scan:
        _fill_uncertainty_bands(
            ax,
            x,
            scan["sigma_lo"],
            scan["sigma_lo_pdfas"],
            scan["sigma_lo_inf"],
            scan["sigma_lo_sup"],
        )
    _set_axis_labels(ax, xlabel=x_label, ylabel=r"$\sigma_{\mathrm{LO}}$ [fb]")
    ax.legend(loc="best")
    ax.grid(alpha=0.25)

    if output and save:
        fig.savefig(output, dpi=200, bbox_inches="tight")
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
    save: bool = True,
) -> plt.Figure:
    """Two panels: $\\sigma_{\\mathrm{NNLO}}$ and $\\sigma_{\\mathrm{HEFT}}/\\sigma_{\\mathrm{SM}}$ at NNLO."""
    key = x_key or _infer_x_key(scan)
    x = scan[key]
    x_label = x_label or X_LABELS.get(key, key)
    ykey, pdfkey, infkey, supkey, order_label = _enhancement_series(scan, nnlo_label, nnlo_label)

    fig, (ax0, ax1) = plt.subplots(
        2, 1, figsize=(8, 7), sharex=True, constrained_layout=True, gridspec_kw={"height_ratios": [2.6, 1.0]}
    )
    if title:
        fig.suptitle(title)

    ax0.plot(x, scan["sigma_nnlo"], color=CB_CURVE, lw=2, label=rf"$\sigma_{{\mathrm{{{nnlo_label}}}}}$", zorder=4)
    if "sigma_nnlo_pdfas" in scan:
        _fill_uncertainty_bands(
            ax0,
            x,
            scan["sigma_nnlo"],
            scan["sigma_nnlo_pdfas"],
            scan["sigma_nnlo_inf"],
            scan["sigma_nnlo_sup"],
        )
    _set_axis_labels(ax0, ylabel=rf"$\sigma_{{\mathrm{{{nnlo_label}}}}}$ [fb]")
    ax0.grid(alpha=0.25)
    legend = ax0.legend(loc="best")
    if sigma_inset and "sigma_nnlo_pdfas" in scan:
        _add_sigma_inset(
            ax0,
            x,
            scan["sigma_nnlo"],
            scan["sigma_nnlo_pdfas"],
            scan["sigma_nnlo_inf"],
            scan["sigma_nnlo_sup"],
            legend=legend,
            xlim=sigma_inset_xlim,
        )

    y = scan[ykey]
    ax1.axhline(1.0, color="0.55", ls=":", lw=1.0, zorder=1)
    ax1.plot(
        x,
        y,
        color=CB_CURVE,
        lw=2,
        label=rf"$\sigma_{{\mathrm{{{order_label}}}}}/\sigma_{{\mathrm{{{order_label}}}}}^{{\mathrm{{SM}}}}$",
        zorder=4,
    )
    if show_enhancement_uncertainty and pdfkey in scan:
        _fill_uncertainty_bands(ax1, x, y, scan[pdfkey], scan[infkey], scan[supkey])
    _set_axis_labels(
        ax1,
        xlabel=x_label,
        ylabel=rf"$\sigma_{{\mathrm{{{order_label}}}}}/\sigma_{{\mathrm{{{order_label}}}}}^{{\mathrm{{SM}}}}$",
    )
    if show_enhancement_uncertainty and pdfkey in scan:
        ax1.set_ylim(_ylim_from_series(y, scan[pdfkey], scan[infkey], scan[supkey]))
    else:
        pad = 0.06 * max(float(np.max(y) - np.min(y)), 1e-6)
        ax1.set_ylim(float(np.min(y)) - pad, float(np.max(y)) + pad)
    ax1.legend(loc="best")
    ax1.grid(alpha=0.25)

    if output and save:
        fig.savefig(output, dpi=200, bbox_inches="tight")
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
    save: bool = True,
) -> plt.Figure:
    """Single-panel $\\sigma_{\\mathrm{HEFT}}/\\sigma_{\\mathrm{SM}}$ scan at LO or NNLO."""
    key = x_key or _infer_x_key(scan)
    x = scan[key]
    x_label = x_label or X_LABELS.get(key, key)
    ykey, pdfkey, infkey, supkey, order_label = _enhancement_series(scan, order, nnlo_label)

    fig, ax = plt.subplots(1, 1, figsize=(8, 4.2), constrained_layout=True)
    if title:
        ax.set_title(title)

    y = scan[ykey]
    ax.axhline(1.0, color="0.55", ls=":", lw=1.0, zorder=1)
    ax.plot(
        x,
        y,
        color=CB_CURVE,
        lw=2,
        label=rf"$\sigma_{{\mathrm{{{order_label}}}}}/\sigma_{{\mathrm{{{order_label}}}}}^{{\mathrm{{SM}}}}$",
        zorder=4,
    )
    if show_uncertainty and pdfkey in scan:
        _fill_uncertainty_bands(ax, x, y, scan[pdfkey], scan[infkey], scan[supkey])
    _set_axis_labels(
        ax,
        xlabel=x_label,
        ylabel=rf"$\sigma_{{\mathrm{{{order_label}}}}}/\sigma_{{\mathrm{{{order_label}}}}}^{{\mathrm{{SM}}}}$",
    )
    if show_uncertainty and pdfkey in scan:
        ax.set_ylim(_ylim_from_series(y, scan[pdfkey], scan[infkey], scan[supkey]))
    else:
        pad = 0.06 * max(float(np.max(y) - np.min(y)), 1e-6)
        ax.set_ylim(float(np.min(y)) - pad, float(np.max(y)) + pad)
    ax.legend(loc="best")
    ax.grid(alpha=0.25)

    if output and save:
        fig.savefig(output, dpi=200, bbox_inches="tight")
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
    save: bool = True,
) -> plt.Figure:
    key = x_key or _infer_x_key(scan)
    x = scan[key]
    x_label = x_label or X_LABELS.get(key, key)

    fig, (ax0, ax1) = plt.subplots(
        2, 1, figsize=(8, 7), sharex=True, constrained_layout=True, gridspec_kw={"height_ratios": [2.6, 1.0]}
    )
    if title:
        fig.suptitle(title)

    ax0.plot(x, scan["sigma_nnlo"], color=CB_CURVE, lw=2, label=rf"$\sigma_{{\mathrm{{{nnlo_label}}}}}$", zorder=4)
    _fill_uncertainty_bands(
        ax0,
        x,
        scan["sigma_nnlo"],
        scan["sigma_nnlo_pdfas"],
        scan["sigma_nnlo_inf"],
        scan["sigma_nnlo_sup"],
    )
    if sim_scan is not None and len(sim_scan.get("x", [])):
        _overlay_simulation(ax0, sim_scan["x"], sim_scan["sigma_nnlo"])
    _set_axis_labels(ax0, ylabel=rf"$\sigma_{{\mathrm{{{nnlo_label}}}}}$ [fb]")
    ax0.grid(alpha=0.25)
    legend = ax0.legend(loc="best")
    if sigma_inset:
        _add_sigma_inset(
            ax0,
            x,
            scan["sigma_nnlo"],
            scan["sigma_nnlo_pdfas"],
            scan["sigma_nnlo_inf"],
            scan["sigma_nnlo_sup"],
            legend=legend,
            xlim=sigma_inset_xlim,
        )

    ax1.plot(x, scan["k"], color=CB_CURVE, lw=2, label=rf"$K=\sigma_{{\mathrm{{{nnlo_label}}}}}/\sigma_{{\mathrm{{LO}}}}$")
    if show_k_uncertainty:
        _fill_uncertainty_bands(ax1, x, scan["k"], scan["k_pdfas"], scan["k_inf"], scan["k_sup"])
    if sim_scan is not None and len(sim_scan.get("x", [])):
        sim_k = sim_scan["sigma_nnlo"] / sim_scan["sigma_lo"]
        _overlay_simulation(ax1, sim_scan["x"], sim_k, label="simulation $K$")
    _set_axis_labels(ax1, xlabel=x_label, ylabel=rf"$K_{{\mathrm{{{nnlo_label}}}}}$")
    ax1.set_ylim(*(k_ylim or _k_ylim_from_scan(scan, show_uncertainty=show_k_uncertainty)))
    ax1.grid(alpha=0.25)

    if output and save:
        fig.savefig(output, dpi=200, bbox_inches="tight")
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
    save: bool = True,
) -> plt.Figure:
    """Two-panel $\\sigma_{\\mathrm{HEFT}}/\\sigma_{\\mathrm{SM}}$ for LO and NNLO."""
    key = x_key or _infer_x_key(scan)
    x = scan[key]
    x_label = x_label or X_LABELS.get(key, key)

    fig, (ax0, ax1) = plt.subplots(
        2, 1, figsize=(8, 7), sharex=True, constrained_layout=True, gridspec_kw={"height_ratios": [1.0, 1.0]}
    )
    if title:
        fig.suptitle(title)

    for ax, order in ((ax0, "LO"), (ax1, nnlo_label)):
        ykey, pdfkey, infkey, supkey, order_label = _enhancement_series(scan, order, nnlo_label)
        y = scan[ykey]
        ax.axhline(1.0, color="0.55", ls=":", lw=1.0, zorder=1)
        ax.plot(
            x,
            y,
            color=CB_CURVE,
            lw=2,
            label=rf"$\sigma_{{\mathrm{{{order_label}}}}}/\sigma_{{\mathrm{{{order_label}}}}}^{{\mathrm{{SM}}}}$",
            zorder=4,
        )
        if show_uncertainty and pdfkey in scan:
            _fill_uncertainty_bands(ax, x, y, scan[pdfkey], scan[infkey], scan[supkey])
        _set_axis_labels(ax, ylabel=rf"$\sigma_{{\mathrm{{{order_label}}}}}/\sigma_{{\mathrm{{{order_label}}}}}^{{\mathrm{{SM}}}}$")
        if show_uncertainty and pdfkey in scan:
            ax.set_ylim(_ylim_from_series(y, scan[pdfkey], scan[infkey], scan[supkey]))
        else:
            pad = 0.06 * max(float(np.max(y) - np.min(y)), 1e-6)
            ax.set_ylim(float(np.min(y)) - pad, float(np.max(y)) + pad)
        ax.legend(loc="best")
        ax.grid(alpha=0.25)

    _set_axis_labels(ax1, xlabel=x_label)
    if output and save:
        fig.savefig(output, dpi=200, bbox_inches="tight")
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
    save: bool = True,
) -> plt.Figure:
    key = x_key or _infer_x_key(scan)
    x = np.asarray(scan[key], dtype=float)
    x_label = x_label or X_LABELS.get(key, key)

    fig, ax = plt.subplots(1, 1, figsize=(8, 4.2), constrained_layout=True)
    if title:
        ax.set_title(title)
    ax.plot(x, scan["k"], color=CB_CURVE, lw=2, label=rf"$K_{{\mathrm{{{nnlo_label}}}}}$")
    if show_uncertainty:
        _fill_uncertainty_bands(ax, x, scan["k"], scan["k_pdfas"], scan["k_inf"], scan["k_sup"])
    if sim_scan is not None and len(sim_scan.get("x", [])):
        sim_k = sim_scan["sigma_nnlo"] / sim_scan["sigma_lo"]
        _overlay_simulation(ax, sim_scan["x"], sim_k, label="simulation $K$")
    ax.legend(loc="best")
    _set_axis_labels(ax, xlabel=x_label, ylabel=rf"$K_{{\mathrm{{{nnlo_label}}}}}$")
    ax.set_ylim(*(k_ylim or _k_ylim_from_scan(scan, show_uncertainty=show_uncertainty)))
    ax.grid(alpha=0.25)
    if output and save:
        fig.savefig(output, dpi=200, bbox_inches="tight")
    return fig
