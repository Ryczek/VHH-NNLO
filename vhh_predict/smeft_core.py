"""Closed-form SMEFT σ predictions from linear B coefficients."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from itertools import product

from .smeft_analysis import SMEFTAnalysis
from .smeft_operators import (
    SMEFT_WC_INTERVALS,
    SMEFT_WC_PLAIN,
    normalize_wc_dict,
    scan_axes,
    sm_wc_values,
    wc_keys_for_process,
)


@dataclass(frozen=True)
class SMEFTPrediction:
    sigma_lo: float
    sigma_nnlo: float
    k_factor: float
    sigma_lo_pdfas: float
    sigma_lo_scale_up: float
    sigma_lo_scale_down: float
    sigma_nnlo_pdfas: float
    sigma_nnlo_scale_up: float
    sigma_nnlo_scale_down: float
    k_pdfas: float
    k_scale_up: float
    k_scale_down: float
    sigma_sm_lo: float
    sigma_sm_nnlo: float


def _wc_vector(keys: Tuple[str, ...], wcs: Dict[str, float]) -> np.ndarray:
    return np.asarray([wcs.get(k, 0.0) for k in keys], dtype=float)


def _prop_std(m: np.ndarray, cov: np.ndarray) -> float:
    return float(math.sqrt(max(float(m @ cov @ m), 0.0)))


def _sigma_pdf_unc(
    m: np.ndarray,
    *,
    sm_pdf: float,
    cov_B: np.ndarray,
    cov_sm_B: np.ndarray,
) -> float:
    """PDF on σ = σ_SM + C·B including Cov(σ_SM, B) from replicas.

    Var = δσ_SM² + mᵀ C_B m + 2 m · Cov(σ_SM, B).
    """
    sm_u = sm_pdf if math.isfinite(sm_pdf) else 0.0
    var = sm_u * sm_u + float(m @ cov_B @ m) + 2.0 * float(m @ cov_sm_B)
    return float(math.sqrt(max(var, 0.0)))


def _sigma_alpha_s_unc(
    m: np.ndarray,
    *,
    sm_alpha: float,
    delta_B_alpha: np.ndarray,
) -> float:
    """α_s shift on σ = σ_SM + C·B: δσ_α = δσ_SM_α + C · δB_α (2-point)."""
    sm_a = sm_alpha if math.isfinite(sm_alpha) else 0.0
    return float(abs(sm_a + float(m @ delta_B_alpha)))


def _sigma_scale_envelope(
    wcs: Dict[str, float],
    wc_keys: Tuple[str, ...],
    sigma_sm: float,
    b_central: np.ndarray,
    central: float,
    by_scale: Optional[Dict],
    sm_by_scale: Optional[Dict],
) -> Tuple[float, float]:
    if not by_scale:
        return float("nan"), float("nan")
    m = _wc_vector(wc_keys, wcs)
    b0 = np.asarray(b_central, dtype=float)
    sup, inf = 0.0, 0.0
    for key, b_vec in by_scale.items():
        b = np.asarray(b_vec, dtype=float)
        if b.shape != b0.shape:
            raise ValueError(
                f"Scale B vector shape {b.shape} != central shape {b0.shape} at {key}"
            )
        # Missing/NaN scale-refit entries: fall back to the central B_i
        # (important: 0*nan is nan, which would otherwise zero the envelope).
        bad = ~np.isfinite(b)
        if np.any(bad):
            b = b.copy()
            b[bad] = b0[bad]
        sm_s = sm_by_scale.get(key, sigma_sm) if sm_by_scale else sigma_sm
        if not math.isfinite(sm_s):
            sm_s = sigma_sm
        val = float(sm_s + m @ b)
        if not math.isfinite(val):
            continue
        sup = max(sup, val - central)
        inf = max(inf, central - val)
    return sup, inf


def sigma(
    analysis: SMEFTAnalysis,
    wcs: Dict[str, float],
    order: str = "NNLO",
) -> float:
    wcs = normalize_wc_dict(analysis.process, wcs)
    order_u = order.upper()
    if order_u == "LO":
        m = _wc_vector(analysis.wc_keys_lo, wcs)
        return float(analysis.sigma_sm_lo + m @ analysis.B_LO)
    if order_u in ("NNLO", "HHZ"):
        m = _wc_vector(analysis.wc_keys_nnlo, wcs)
        return float(analysis.sigma_sm_nnlo + m @ analysis.B_NNLO)
    raise ValueError("order must be LO or NNLO/HHZ")


def sigma_uncertainties(
    analysis: SMEFTAnalysis,
    wcs: Dict[str, float],
    order: str,
) -> Dict[str, float]:
    wcs = normalize_wc_dict(analysis.process, wcs)
    order_u = order.upper()
    if order_u == "LO":
        wc_keys = analysis.wc_keys_lo
        m = _wc_vector(wc_keys, wcs)
        cov_pdf = analysis.C_LO_pdf
        sigma_sm = analysis.sigma_sm_lo
        sm_pdf = analysis.sigma_sm_lo_pdf
        sm_alpha = analysis.sigma_sm_lo_alpha_s
        delta_B_alpha = analysis.delta_B_LO_alpha_s
        cov_sm_B = analysis.cov_sm_B_LO_pdf
        b_central = analysis.B_LO
        by_scale = analysis.B_LO_by_scale
        sm_by_scale = analysis.sigma_sm_lo_by_scale
    elif order_u in ("NNLO", "HHZ"):
        wc_keys = analysis.wc_keys_nnlo
        m = _wc_vector(wc_keys, wcs)
        cov_pdf = analysis.C_NNLO_pdf
        sigma_sm = analysis.sigma_sm_nnlo
        sm_pdf = analysis.sigma_sm_nnlo_pdf
        sm_alpha = analysis.sigma_sm_nnlo_alpha_s
        delta_B_alpha = analysis.delta_B_NNLO_alpha_s
        cov_sm_B = analysis.cov_sm_B_NNLO_pdf
        b_central = analysis.B_NNLO
        by_scale = analysis.B_NNLO_by_scale
        sm_by_scale = analysis.sigma_sm_nnlo_by_scale
    else:
        raise ValueError("order must be LO or NNLO/HHZ")

    central = float(sigma_sm + m @ b_central)
    scale_up, scale_down = _sigma_scale_envelope(
        wcs, wc_keys, sigma_sm, b_central, central, by_scale, sm_by_scale
    )
    pdf = _sigma_pdf_unc(m, sm_pdf=sm_pdf, cov_B=cov_pdf, cov_sm_B=cov_sm_B)
    alpha = _sigma_alpha_s_unc(m, sm_alpha=sm_alpha, delta_B_alpha=delta_B_alpha)
    pdfas = float(math.sqrt(max(pdf * pdf + alpha * alpha, 0.0)))

    return {
        "central": central,
        "pdf": pdf,
        "alpha_s": alpha,
        "pdf_alpha_s": pdfas,
        "scale_up": scale_up,
        "scale_down": scale_down,
        "sigma_sm": sigma_sm,
    }


def sm_enhancement(
    analysis: SMEFTAnalysis,
    wcs: Dict[str, float],
    order: str = "NNLO",
) -> float:
    sm = sm_wc_values(analysis.process)
    u = sigma_uncertainties(analysis, wcs, order)
    u_sm = sigma_uncertainties(analysis, sm, order)
    den = u_sm["central"]
    return u["central"] / den if den else float("nan")


def _k_unc(s_lo: float, s_nn: float, d_lo: float, d_nn: float) -> float:
    if s_lo == 0.0:
        return float("nan")
    var = (d_nn * d_nn) / (s_lo * s_lo) + (s_nn * s_nn) * (d_lo * d_lo) / (s_lo**4)
    return math.sqrt(max(var, 0.0))


def predict(analysis: SMEFTAnalysis, wcs: Dict[str, float]) -> SMEFTPrediction:
    wcs = normalize_wc_dict(analysis.process, wcs)
    u_lo = sigma_uncertainties(analysis, wcs, "LO")
    u_nn = sigma_uncertainties(analysis, wcs, "NNLO")
    slo, snn = u_lo["central"], u_nn["central"]
    k0 = snn / slo if slo else float("nan")

    if analysis.has_scale_envelope and math.isfinite(k0):
        sm = sm_wc_values(analysis.process)
        m_lo = _wc_vector(analysis.wc_keys_lo, wcs)
        m_nn = _wc_vector(analysis.wc_keys_nnlo, wcs)
        m_lo_sm = _wc_vector(analysis.wc_keys_lo, sm)
        m_nn_sm = _wc_vector(analysis.wc_keys_nnlo, sm)
        k_vals: List[float] = []
        b_lo0 = np.asarray(analysis.B_LO, dtype=float)
        b_nn0 = np.asarray(analysis.B_NNLO, dtype=float)
        for key, b_lo in analysis.B_LO_by_scale.items():  # type: ignore[union-attr]
            b_nn = analysis.B_NNLO_by_scale.get(key)  # type: ignore[union-attr]
            if b_nn is None:
                continue
            b_lo_a = np.asarray(b_lo, dtype=float)
            b_nn_a = np.asarray(b_nn, dtype=float)
            bad_lo = ~np.isfinite(b_lo_a)
            bad_nn = ~np.isfinite(b_nn_a)
            if np.any(bad_lo):
                b_lo_a = b_lo_a.copy()
                b_lo_a[bad_lo] = b_lo0[bad_lo]
            if np.any(bad_nn):
                b_nn_a = b_nn_a.copy()
                b_nn_a[bad_nn] = b_nn0[bad_nn]
            sm_lo_s = (analysis.sigma_sm_lo_by_scale or {}).get(key, analysis.sigma_sm_lo)
            sm_nn_s = (analysis.sigma_sm_nnlo_by_scale or {}).get(key, analysis.sigma_sm_nnlo)
            s_lo_s = float(sm_lo_s + m_lo @ b_lo_a)
            s_nn_s = float(sm_nn_s + m_nn @ b_nn_a)
            if s_lo_s != 0.0 and math.isfinite(s_lo_s) and math.isfinite(s_nn_s):
                k_vals.append(s_nn_s / s_lo_s)
        k_scale_up = max((kv - k0) for kv in k_vals) if k_vals else float("nan")
        k_scale_down = max((k0 - kv) for kv in k_vals) if k_vals else float("nan")
    else:
        k_scale_up = _k_unc(slo, snn, u_lo["scale_up"], u_nn["scale_up"])
        k_scale_down = _k_unc(slo, snn, u_lo["scale_down"], u_nn["scale_down"])

    return SMEFTPrediction(
        sigma_lo=slo,
        sigma_nnlo=snn,
        k_factor=k0,
        sigma_lo_pdfas=u_lo["pdf_alpha_s"],
        sigma_lo_scale_up=u_lo["scale_up"],
        sigma_lo_scale_down=u_lo["scale_down"],
        sigma_nnlo_pdfas=u_nn["pdf_alpha_s"],
        sigma_nnlo_scale_up=u_nn["scale_up"],
        sigma_nnlo_scale_down=u_nn["scale_down"],
        k_pdfas=_k_unc(slo, snn, u_lo["pdf_alpha_s"], u_nn["pdf_alpha_s"]),
        k_scale_up=k_scale_up,
        k_scale_down=k_scale_down,
        sigma_sm_lo=u_lo["sigma_sm"],
        sigma_sm_nnlo=u_nn["sigma_sm"],
    )


def _pct(delta: float, central: float) -> float:
    if central == 0.0 or not math.isfinite(central):
        return float("nan")
    return 100.0 * delta / central


def spot_check_caption(analysis: SMEFTAnalysis, wcs: Dict[str, float]) -> str:
    wcs = normalize_wc_dict(analysis.process, wcs)
    parts = [f"{SMEFT_WC_PLAIN.get(k, k)}={v:g}" for k, v in wcs.items() if v != 0.0]
    wc_str = ", ".join(parts) if parts else "SM (all C=0)"
    return f"{analysis.process} @ {analysis.energy_tev:g} TeV — {wc_str}"


def spot_check_table(
    analysis: SMEFTAnalysis,
    wcs: Dict[str, float],
    *,
    as_percent: bool = False,
    compare_simulation: bool = False,
    results_root=None,
):
    import pandas as pd

    from .smeft_simulation import load_smeft_simulation_central

    wcs = normalize_wc_dict(analysis.process, wcs)
    p = predict(analysis, wcs)
    nn_label = "NNLO"  # display name; ZHH uses HHZ internally
    sim = (
        load_smeft_simulation_central(
            analysis.process,
            analysis.energy_tev,
            wcs,
            root=results_root,
        )
        if compare_simulation
        else None
    )
    sim_nnlo = (
        sim.sigma_hhz if sim and analysis.is_zhh else sim.sigma_nnlo if sim else None
    )

    def _scale_str(central: float, up: float, down: float) -> str:
        if not math.isfinite(central) or central == 0.0:
            return "—"
        if as_percent:
            return f"+{_pct(up, central):.2f}% / −{_pct(down, central):.2f}%"
        return f"+{up:.4g} / −{down:.4g} fb"

    def _pdf_str(central: float, pdf: float) -> str:
        if not math.isfinite(central) or central == 0.0:
            return "—"
        if as_percent:
            return f"±{_pct(pdf, central):.2f}%"
        return f"±{pdf:.4g} fb"

    def _val_str(value: float) -> str:
        return f"{value:.6g}" if math.isfinite(value) else "—"

    def _sim_str(value: Optional[float]) -> str:
        if value is None or not math.isfinite(value):
            return "—"
        return f"{value:.6g}"

    def _diff_str(pred: float, measured: Optional[float]) -> str:
        if measured is None or not math.isfinite(measured) or measured == 0.0:
            return "—"
        if not math.isfinite(pred):
            return "—"
        return f"{100.0 * (pred - measured) / measured:+.2f}%"

    rows: List[Dict[str, str]] = []

    def _add_row(
        quantity: str,
        pred: float,
        scale_up: float,
        scale_down: float,
        pdf: float,
        sim_val: Optional[float],
        *,
        with_uncertainty: bool = True,
    ) -> None:
        row = {
            "Quantity": quantity,
            "SMEFT": _val_str(pred),
            "Scale (+/−)": _scale_str(pred, scale_up, scale_down) if with_uncertainty else "—",
            "PDF+αs": _pdf_str(pred, pdf) if with_uncertainty else "—",
        }
        if compare_simulation:
            row["Simulation"] = _sim_str(sim_val)
            row["Δ vs simulation"] = _diff_str(pred, sim_val)
        rows.append(row)

    _add_row(
        "σ_LO [fb]",
        p.sigma_lo,
        p.sigma_lo_scale_up,
        p.sigma_lo_scale_down,
        p.sigma_lo_pdfas,
        sim.sigma_lo if sim else None,
    )
    _add_row(
        f"σ_{nn_label} [fb]",
        p.sigma_nnlo,
        p.sigma_nnlo_scale_up,
        p.sigma_nnlo_scale_down,
        p.sigma_nnlo_pdfas,
        sim_nnlo,
    )
    _add_row(
        "K",
        p.k_factor,
        float("nan"),
        float("nan"),
        float("nan"),
        sim.k_measured if sim else None,
        with_uncertainty=False,
    )
    return pd.DataFrame(rows)


def resolve_scan_axis(process: str, axis: str) -> Tuple[int, str]:
    axes = scan_axes(process)
    if axis not in axes:
        raise ValueError(f"Unknown scan axis {axis!r} for {process}; use one of: {axes}")
    return axes.index(axis), axis


def _scan_base_wcs(analysis: SMEFTAnalysis, fixed_wcs: Optional[Dict[str, float]]) -> Dict[str, float]:
    base = normalize_wc_dict(analysis.process, fixed_wcs or sm_wc_values(analysis.process))
    return base


def scan(
    analysis: SMEFTAnalysis,
    axis: str,
    *,
    vmin: float = -1.0,
    vmax: float = 6.0,
    n_points: int = 400,
    fixed_wcs: Optional[Dict[str, float]] = None,
    uncertainties: bool = False,
) -> Dict[str, np.ndarray]:
    base = _scan_base_wcs(analysis, fixed_wcs)
    _, x_key = resolve_scan_axis(analysis.process, axis)

    xs = np.linspace(vmin, vmax, n_points)
    out: Dict[str, List[float]] = {
        x_key: [],
        "sigma_lo": [],
        "sigma_nnlo": [],
        "sigma_smeft_over_sm_lo": [],
        "sigma_smeft_over_sm_nnlo": [],
        "k": [],
    }
    if uncertainties:
        out.update(
            {
                "sigma_lo_pdfas": [],
                "sigma_lo_sup": [],
                "sigma_lo_inf": [],
                "sigma_nnlo_pdfas": [],
                "sigma_nnlo_sup": [],
                "sigma_nnlo_inf": [],
                "k_pdfas": [],
                "k_sup": [],
                "k_inf": [],
            }
        )

    for x in xs:
        wcs = dict(base)
        wcs[x_key] = float(x)
        p = predict(analysis, wcs)
        out[x_key].append(float(x))
        out["sigma_lo"].append(p.sigma_lo)
        out["sigma_nnlo"].append(p.sigma_nnlo)
        out["sigma_smeft_over_sm_lo"].append(sm_enhancement(analysis, wcs, "LO"))
        out["sigma_smeft_over_sm_nnlo"].append(sm_enhancement(analysis, wcs, "NNLO"))
        out["k"].append(p.k_factor)
        if uncertainties:
            out["sigma_lo_pdfas"].append(p.sigma_lo_pdfas)
            out["sigma_lo_sup"].append(p.sigma_lo_scale_up)
            out["sigma_lo_inf"].append(p.sigma_lo_scale_down)
            out["sigma_nnlo_pdfas"].append(p.sigma_nnlo_pdfas)
            out["sigma_nnlo_sup"].append(p.sigma_nnlo_scale_up)
            out["sigma_nnlo_inf"].append(p.sigma_nnlo_scale_down)
            out["k_pdfas"].append(p.k_pdfas)
            out["k_sup"].append(p.k_scale_up)
            out["k_inf"].append(p.k_scale_down)
    # Aliases for shared plot helpers (HEFT naming)
    out["sigma_heft_over_sm_lo"] = list(out["sigma_smeft_over_sm_lo"])
    out["sigma_heft_over_sm_nnlo"] = list(out["sigma_smeft_over_sm_nnlo"])
    return {k: np.asarray(v) for k, v in out.items()}


def scan_grid(
    analysis: SMEFTAnalysis,
    axes: Sequence[str],
    *,
    windows: Optional[Dict[str, Tuple[float, float]]] = None,
    n_points: int = 40,
    fixed_wcs: Optional[Dict[str, float]] = None,
    uncertainties: bool = False,
) -> Dict[str, np.ndarray]:
    """Scan several WC axes **simultaneously** on a Cartesian product grid.

    Non-scanned $C_i$ stay at *fixed_wcs* (defaults SM). Default *n_points* is
    40 per axis (e.g. 2 axes → 1600 evaluations).
    """
    if not axes:
        raise ValueError("scan_grid requires at least one axis")

    base = _scan_base_wcs(analysis, fixed_wcs)
    resolved: List[Tuple[str, float, float]] = []
    seen: set[str] = set()
    for axis in axes:
        _, x_key = resolve_scan_axis(analysis.process, axis)
        if x_key in seen:
            raise ValueError(f"Duplicate scan axis {x_key!r}")
        seen.add(x_key)
        if windows and axis in windows:
            vmin, vmax = windows[axis]
        elif x_key in SMEFT_WC_INTERVALS:
            vmin, vmax = SMEFT_WC_INTERVALS[x_key]
        else:
            raise KeyError(
                f"No scan window for {axis!r}; pass windows={{'{axis}': (vmin, vmax)}}"
            )
        resolved.append((x_key, float(vmin), float(vmax)))

    grids = [np.linspace(vmin, vmax, n_points) for _, vmin, vmax in resolved]
    x_keys = [x_key for x_key, _, _ in resolved]

    out: Dict[str, List[float]] = {k: [] for k in x_keys}
    out.update(
        {
            "sigma_lo": [],
            "sigma_nnlo": [],
            "sigma_smeft_over_sm_lo": [],
            "sigma_smeft_over_sm_nnlo": [],
            "k": [],
        }
    )
    if uncertainties:
        out.update(
            {
                "sigma_lo_pdfas": [],
                "sigma_lo_sup": [],
                "sigma_lo_inf": [],
                "sigma_nnlo_pdfas": [],
                "sigma_nnlo_sup": [],
                "sigma_nnlo_inf": [],
                "k_pdfas": [],
                "k_sup": [],
                "k_inf": [],
            }
        )

    for vals in product(*grids):
        wcs = dict(base)
        for x_key, val in zip(x_keys, vals):
            wcs[x_key] = float(val)
            out[x_key].append(float(val))
        p = predict(analysis, wcs)
        out["sigma_lo"].append(p.sigma_lo)
        out["sigma_nnlo"].append(p.sigma_nnlo)
        out["sigma_smeft_over_sm_lo"].append(sm_enhancement(analysis, wcs, "LO"))
        out["sigma_smeft_over_sm_nnlo"].append(sm_enhancement(analysis, wcs, "NNLO"))
        out["k"].append(p.k_factor)
        if uncertainties:
            out["sigma_lo_pdfas"].append(p.sigma_lo_pdfas)
            out["sigma_lo_sup"].append(p.sigma_lo_scale_up)
            out["sigma_lo_inf"].append(p.sigma_lo_scale_down)
            out["sigma_nnlo_pdfas"].append(p.sigma_nnlo_pdfas)
            out["sigma_nnlo_sup"].append(p.sigma_nnlo_scale_up)
            out["sigma_nnlo_inf"].append(p.sigma_nnlo_scale_down)
            out["k_pdfas"].append(p.k_pdfas)
            out["k_sup"].append(p.k_scale_up)
            out["k_inf"].append(p.k_scale_down)

    out["sigma_heft_over_sm_lo"] = list(out["sigma_smeft_over_sm_lo"])
    out["sigma_heft_over_sm_nnlo"] = list(out["sigma_smeft_over_sm_nnlo"])
    return {k: np.asarray(v) for k, v in out.items()}
