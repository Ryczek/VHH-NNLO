"""Closed-form σ and K predictions from HEFT coefficients and covariances."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .analysis import VHHAnalysis
from .monomials import monomial_vector


@dataclass(frozen=True)
class Prediction:
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


def _prop_std(m: np.ndarray, cov: np.ndarray) -> float:
    return float(math.sqrt(max(float(m @ cov @ m), 0.0)))


def _sigma_scale_envelope(
    m: np.ndarray,
    central: float,
    by_scale: Optional[Dict],
) -> Tuple[float, float]:
    """7-point σ envelope: refit A_i at each (μ_R, μ_F), then evaluate m·A."""
    if not by_scale:
        return float("nan"), float("nan")
    sup, inf = 0.0, 0.0
    for a_vec in by_scale.values():
        val = float(m @ a_vec)
        sup = max(sup, val - central)
        inf = max(inf, central - val)
    return sup, inf


def _normalize_kappa(process: str, kappa: Tuple[float, ...]) -> Tuple[float, ...]:
    if process == "ZHH":
        if len(kappa) == 3:
            return (kappa[0], kappa[1], kappa[2], 1.0)
        if len(kappa) != 4:
            raise ValueError("ZHH needs 3 or 4 kappa values (kl, kZ, k2Z [, kt])")
        return tuple(kappa)
    if len(kappa) != 3:
        raise ValueError(f"{process} needs 3 kappa values (kl, kV, k2V)")
    return tuple(kappa[:3])


def sigma(
    analysis: VHHAnalysis,
    kappa: Tuple[float, ...],
    order: str = "NNLO",
) -> float:
    kappa = _normalize_kappa(analysis.process, kappa)
    order_u = order.upper()
    if order_u == "LO":
        m = monomial_vector(analysis.process, kappa, "LO")
        return float(m @ analysis.A_LO)
    if order_u in ("NNLO", "HHZ"):
        m = monomial_vector(analysis.process, kappa, "NNLO")
        return float(m @ analysis.A_NNLO)
    raise ValueError("order must be LO or NNLO/HHZ")


def sigma_uncertainties(
    analysis: VHHAnalysis,
    kappa: Tuple[float, ...],
    order: str,
) -> Dict[str, float]:
    kappa = _normalize_kappa(analysis.process, kappa)
    order_u = order.upper()
    if order_u == "LO":
        m = monomial_vector(analysis.process, kappa, "LO")
        cov_pdf = analysis.C_LO_pdf
        cov_alpha_s = analysis.C_LO_alphaS
        a_central = analysis.A_LO
        by_scale = analysis.A_LO_by_scale
    elif order_u in ("NNLO", "HHZ"):
        m = monomial_vector(analysis.process, kappa, "NNLO")
        cov_pdf = analysis.C_NNLO_pdf
        cov_alpha_s = analysis.C_NNLO_alphaS
        a_central = analysis.A_NNLO
        by_scale = analysis.A_NNLO_by_scale
    else:
        raise ValueError("order must be LO or NNLO/HHZ")

    central = float(m @ a_central)
    scale_up, scale_down = _sigma_scale_envelope(m, central, by_scale)

    pdf_unc = _prop_std(m, cov_pdf)
    alpha_s_unc = _prop_std(m, cov_alpha_s)
    pdf_alpha_s_unc = _prop_std(m, cov_pdf + cov_alpha_s)

    return {
        "central": central,
        "pdf": pdf_unc,
        "alpha_s": alpha_s_unc,
        "pdf_alpha_s": pdf_alpha_s_unc,
        "scale_up": scale_up,
        "scale_down": scale_down,
        "scale_method": "envelope" if by_scale else "linear_cov",
    }


def _ratio_pdfas_unc(
    num: float,
    den: float,
    m_num: np.ndarray,
    m_den: np.ndarray,
    cov: np.ndarray,
) -> float:
    if den == 0.0 or not math.isfinite(den):
        return float("nan")
    var = (
        float(m_num @ cov @ m_num) / den**2
        + (num / den**2) ** 2 * float(m_den @ cov @ m_den)
        - 2.0 * num / den**3 * float(m_num @ cov @ m_den)
    )
    return math.sqrt(max(var, 0.0))


def _ratio_scale_envelope(
    m_num: np.ndarray,
    m_den: np.ndarray,
    num: float,
    den: float,
    by_scale: Optional[Dict],
) -> Tuple[float, float]:
    if not by_scale or den == 0.0 or not math.isfinite(den):
        return float("nan"), float("nan")
    central = num / den
    sup, inf = 0.0, 0.0
    for a_vec in by_scale.values():
        den_s = float(m_den @ a_vec)
        if den_s == 0.0:
            continue
        ratio = float(m_num @ a_vec) / den_s
        sup = max(sup, ratio - central)
        inf = max(inf, central - ratio)
    return sup, inf


def sm_kappa(process: str) -> Tuple[float, ...]:
    if process == "ZHH":
        return (1.0, 1.0, 1.0, 1.0)
    return (1.0, 1.0, 1.0)


def sm_enhancement(
    analysis: VHHAnalysis,
    kappa: Tuple[float, ...],
    order: str = "NNLO",
) -> float:
    """σ_HEFT(κ)/σ_HEFT(SM) at LO or NNLO (HHZ for ZHH)."""
    return enhancement_uncertainties(analysis, kappa, order)["central"]


def enhancement_uncertainties(
    analysis: VHHAnalysis,
    kappa: Tuple[float, ...],
    order: str,
) -> Dict[str, float]:
    """σ(kappa)/σ(SM) with PDF+αs and scale envelope uncertainties."""
    kappa = _normalize_kappa(analysis.process, kappa)
    sm = sm_kappa(analysis.process)
    order_u = order.upper()
    if order_u == "LO":
        m = monomial_vector(analysis.process, kappa, "LO")
        m_sm = monomial_vector(analysis.process, sm, "LO")
        cov = analysis.C_LO_pdfas
        by_scale = analysis.A_LO_by_scale
    elif order_u in ("NNLO", "HHZ"):
        m = monomial_vector(analysis.process, kappa, "NNLO")
        m_sm = monomial_vector(analysis.process, sm, "NNLO")
        cov = analysis.C_NNLO_pdfas
        by_scale = analysis.A_NNLO_by_scale
    else:
        raise ValueError("order must be LO or NNLO")

    num = float(m @ (analysis.A_LO if order_u == "LO" else analysis.A_NNLO))
    den = float(m_sm @ (analysis.A_LO if order_u == "LO" else analysis.A_NNLO))
    central = num / den if den else float("nan")
    scale_up, scale_down = _ratio_scale_envelope(m, m_sm, num, den, by_scale)
    return {
        "central": central,
        "pdf_alpha_s": _ratio_pdfas_unc(num, den, m, m_sm, cov),
        "scale_up": scale_up,
        "scale_down": scale_down,
        "sigma_sm": den,
    }


def _k_unc(s_lo: float, s_nn: float, d_lo: float, d_nn: float) -> float:
    if s_lo == 0.0:
        return float("nan")
    var = (d_nn * d_nn) / (s_lo * s_lo) + (s_nn * s_nn) * (d_lo * d_lo) / (s_lo**4)
    return math.sqrt(max(var, 0.0))


def predict(analysis: VHHAnalysis, kappa: Tuple[float, ...]) -> Prediction:
    kappa = _normalize_kappa(analysis.process, kappa)
    u_lo = sigma_uncertainties(analysis, kappa, "LO")
    u_nn = sigma_uncertainties(analysis, kappa, "NNLO")
    slo, snn = u_lo["central"], u_nn["central"]
    k0 = snn / slo if slo else float("nan")

    if analysis.has_scale_envelope and not math.isnan(k0):
        m_lo = monomial_vector(analysis.process, kappa, "LO")
        m_nn = monomial_vector(analysis.process, kappa, "NNLO")
        k_vals: List[float] = []
        for key, a_lo in analysis.A_LO_by_scale.items():
            a_nn = analysis.A_NNLO_by_scale.get(key)  # type: ignore[union-attr]
            if a_nn is None:
                continue
            s_lo_s = float(m_lo @ a_lo)
            s_nn_s = float(m_nn @ a_nn)
            if s_lo_s != 0.0:
                k_vals.append(s_nn_s / s_lo_s)
        k_scale_up = max((kv - k0) for kv in k_vals) if k_vals else float("nan")
        k_scale_down = max((k0 - kv) for kv in k_vals) if k_vals else float("nan")
    else:
        k_scale_up = _k_unc(slo, snn, u_lo["scale_up"], u_nn["scale_up"])
        k_scale_down = _k_unc(slo, snn, u_lo["scale_down"], u_nn["scale_down"])

    return Prediction(
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
    )


def _pct(delta: float, central: float) -> float:
    if central == 0.0 or not math.isfinite(central):
        return float("nan")
    return 100.0 * delta / central


def format_prediction(
    analysis: VHHAnalysis,
    kappa: Tuple[float, ...],
    *,
    as_percent: bool = False,
    compare_simulation: bool = False,
    include_enhancement: bool = True,
    results_root: Optional[Path] = None,
) -> str:
    from .simulation import compare_suffix, load_simulation_central

    kappa = _normalize_kappa(analysis.process, kappa)
    p = predict(analysis, kappa)
    nn_label = analysis.nnlo_label
    sim = (
        load_simulation_central(
            analysis.process,
            analysis.energy_tev,
            kappa,
            root=results_root,
        )
        if compare_simulation
        else None
    )
    sim_nnlo = (
        sim.sigma_hhz if sim and analysis.is_zhh else sim.sigma_nnlo if sim else None
    )

    def _unc(central: float, up: float, down: float, pdf: float, unit: str = "fb") -> str:
        if as_percent:
            return (
                f"+{_pct(up, central):.2f}%/-{_pct(down, central):.2f}% (scale)  "
                f"±{_pct(pdf, central):.2f}% (PDF+αs)"
            )
        if unit == "fb":
            return f"+{up:.4g}/-{down:.4g} (scale)  ±{pdf:.4g} (PDF+αs) fb"
        return f"+{up:.4g}/-{down:.4g} (scale)  ±{pdf:.4g} (PDF+αs)"

    if analysis.is_zhh:
        kl, kv, k2v, kt = kappa
        header = f"{analysis.process} @ {analysis.energy_tev} TeV  κ=({kl:g}, {kv:g}, {k2v:g}, {kt:g})"
    else:
        kl, kv, k2v = kappa[:3]
        header = (
            f"{analysis.process} @ {analysis.energy_tev} TeV  "
            f"κ_λ={kl:g}, κ_W={kv:g}, κ_2W={k2v:g}"
        )

    lines = [
        header,
        f"  σ_LO   HEFT {p.sigma_lo:.6f}  "
        f"{_unc(p.sigma_lo, p.sigma_lo_scale_up, p.sigma_lo_scale_down, p.sigma_lo_pdfas)}"
        f"{compare_suffix(p.sigma_lo, sim.sigma_lo if sim else None) if compare_simulation else ''}",
        f"  σ_{nn_label}  HEFT {p.sigma_nnlo:.6f}  "
        f"{_unc(p.sigma_nnlo, p.sigma_nnlo_scale_up, p.sigma_nnlo_scale_down, p.sigma_nnlo_pdfas)}"
        f"{compare_suffix(p.sigma_nnlo, sim_nnlo) if compare_simulation else ''}",
    ]
    if include_enhancement:
        u_lo_enh = enhancement_uncertainties(analysis, kappa, "LO")
        u_nn_enh = enhancement_uncertainties(analysis, kappa, "NNLO")
        lines.extend(
            [
                f"  σ/σ_SM (LO)   HEFT {u_lo_enh['central']:.6f}  "
                f"{_unc(u_lo_enh['central'], u_lo_enh['scale_up'], u_lo_enh['scale_down'], u_lo_enh['pdf_alpha_s'], unit='')}",
                f"  σ/σ_SM ({nn_label})  HEFT {u_nn_enh['central']:.6f}  "
                f"{_unc(u_nn_enh['central'], u_nn_enh['scale_up'], u_nn_enh['scale_down'], u_nn_enh['pdf_alpha_s'], unit='')}",
            ]
        )
    lines.append(
        f"  K      HEFT {p.k_factor:.6f}"
        f"{compare_suffix(p.k_factor, sim.k_measured if sim else None) if compare_simulation else ''}"
    )
    return "\n".join(lines)


SCAN_AXES_W = ("kappa_lambda", "kappa_w", "kappa_2w")
SCAN_AXES_Z = ("kappa_lambda", "kappa_z", "kappa_2z", "kappa_t")


def scan_axes(process: str) -> Tuple[str, ...]:
    if process == "ZHH":
        return SCAN_AXES_Z
    if process in ("WplusHH", "WminusHH"):
        return SCAN_AXES_W
    raise KeyError(f"Unknown process: {process}")


def resolve_scan_axis(process: str, axis: str) -> Tuple[int, str]:
    """Return (κ index, canonical scan output key) for *process*."""
    if process == "ZHH":
        axis_map = {
            "kappa_lambda": (0, "kappa_lambda"),
            "kappa_z": (1, "kappa_z"),
            "kappa_2z": (2, "kappa_2z"),
            "kappa_t": (3, "kappa_t"),
            "kappa_kt": (3, "kappa_t"),
        }
    elif process in ("WplusHH", "WminusHH"):
        axis_map = {
            "kappa_lambda": (0, "kappa_lambda"),
            "kappa_w": (1, "kappa_w"),
            "kappa_2w": (2, "kappa_2w"),
        }
    else:
        raise KeyError(f"Unknown process: {process}")

    if axis not in axis_map:
        valid = ", ".join(scan_axes(process))
        raise ValueError(f"Unknown scan axis {axis!r} for {process}; use one of: {valid}")
    return axis_map[axis]


def _scan_base_kappa(analysis: VHHAnalysis, fixed_kappa: Optional[Tuple[float, ...]]) -> List[float]:
    if analysis.is_zhh:
        base = list(fixed_kappa) if fixed_kappa else [1.0, 1.0, 1.0, 1.0]
        if len(base) == 3:
            base.append(1.0)
        return base
    base = list(fixed_kappa) if fixed_kappa else [1.0, 1.0, 1.0]
    if len(base) != 3:
        raise ValueError(f"{analysis.process} needs 3 kappa values (kl, kW, k2W)")
    return base


def scan(
    analysis: VHHAnalysis,
    axis: str,
    *,
    vmin: float = -1.0,
    vmax: float = 6.0,
    n_points: int = 400,
    fixed_kappa: Optional[Tuple[float, ...]] = None,
) -> Dict[str, np.ndarray]:
    """Scan one κ component; others fixed (defaults SM-like)."""
    base = _scan_base_kappa(analysis, fixed_kappa)
    idx, x_key = resolve_scan_axis(analysis.process, axis)

    xs = np.linspace(vmin, vmax, n_points)
    out: Dict[str, List[float]] = {
        x_key: [],
        "sigma_lo": [],
        "sigma_nnlo": [],
        "k": [],
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
    for x in xs:
        kappa = list(base)
        kappa[idx] = float(x)
        p = predict(analysis, tuple(kappa))
        out[x_key].append(float(x))
        out["sigma_lo"].append(p.sigma_lo)
        out["sigma_nnlo"].append(p.sigma_nnlo)
        out["k"].append(p.k_factor)
        out["sigma_lo_pdfas"].append(p.sigma_lo_pdfas)
        out["sigma_lo_sup"].append(p.sigma_lo_scale_up)
        out["sigma_lo_inf"].append(p.sigma_lo_scale_down)
        out["sigma_nnlo_pdfas"].append(p.sigma_nnlo_pdfas)
        out["sigma_nnlo_sup"].append(p.sigma_nnlo_scale_up)
        out["sigma_nnlo_inf"].append(p.sigma_nnlo_scale_down)
        out["k_pdfas"].append(p.k_pdfas)
        out["k_sup"].append(p.k_scale_up)
        out["k_inf"].append(p.k_scale_down)
    return {k: np.asarray(v) for k, v in out.items()}


def scan_sm_enhancement(
    analysis: VHHAnalysis,
    axis: str,
    *,
    vmin: float = -1.0,
    vmax: float = 6.0,
    n_points: int = 400,
    fixed_kappa: Optional[Tuple[float, ...]] = None,
) -> Dict[str, np.ndarray]:
    """Scan σ_HEFT/σ_SM for LO and NNLO along one κ component (SM at κ=(1,1,1) or (1,1,1,1))."""
    sm = sm_kappa(analysis.process)
    sigma_lo_sm = sigma(analysis, sm, "LO")
    sigma_nnlo_sm = sigma(analysis, sm, "NNLO")

    base = _scan_base_kappa(analysis, fixed_kappa)
    idx, x_key = resolve_scan_axis(analysis.process, axis)

    xs = np.linspace(vmin, vmax, n_points)
    out: Dict[str, List[float]] = {
        x_key: [],
        "enhancement_lo": [],
        "enhancement_nnlo": [],
        "enhancement_lo_pdfas": [],
        "enhancement_lo_sup": [],
        "enhancement_lo_inf": [],
        "enhancement_nnlo_pdfas": [],
        "enhancement_nnlo_sup": [],
        "enhancement_nnlo_inf": [],
        "sigma_lo_sm": [],
        "sigma_nnlo_sm": [],
    }
    for x in xs:
        kappa = list(base)
        kappa[idx] = float(x)
        kappa_t = tuple(kappa)
        u_lo = enhancement_uncertainties(analysis, kappa_t, "LO")
        u_nn = enhancement_uncertainties(analysis, kappa_t, "NNLO")
        out[x_key].append(float(x))
        out["enhancement_lo"].append(u_lo["central"])
        out["enhancement_nnlo"].append(u_nn["central"])
        out["enhancement_lo_pdfas"].append(u_lo["pdf_alpha_s"])
        out["enhancement_lo_sup"].append(u_lo["scale_up"])
        out["enhancement_lo_inf"].append(u_lo["scale_down"])
        out["enhancement_nnlo_pdfas"].append(u_nn["pdf_alpha_s"])
        out["enhancement_nnlo_sup"].append(u_nn["scale_up"])
        out["enhancement_nnlo_inf"].append(u_nn["scale_down"])
        out["sigma_lo_sm"].append(sigma_lo_sm)
        out["sigma_nnlo_sm"].append(sigma_nnlo_sm)
    return {k: np.asarray(v) for k, v in out.items()}
