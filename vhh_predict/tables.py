"""Wilson-coefficient benchmark tables (paper-style σ with uncertainties)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd

from .analysis import VHHAnalysis, load_analysis
from .core import Prediction, predict, scan_axes, sm_kappa

# 95% CL exclusion intervals for every HEFT κ (κ_λ, κ_W, κ_Z, κ_2W, κ_2Z, κ_t).
# Used as default scan windows and as the min/max points in §5 benchmark tables.
WILSON_INTERVALS: Dict[str, Tuple[float, float]] = {
    "kappa_lambda": (-0.70, 6.10),
    "kappa_w": (0.85, 1.20),
    "kappa_z": (0.90, 1.20),
    "kappa_2w": (0.70, 1.30),
    "kappa_2z": (0.70, 1.30),
    "kappa_t": (0.80, 1.20),
}

SM_VALUE = 1.0
TABLE_ENERGIES_TEV = (13.6, 14.0)

KAPPA_LATEX = {
    "kappa_lambda": r"\kappa_\lambda",
    "kappa_w": r"\kappa_W",
    "kappa_z": r"\kappa_Z",
    "kappa_2w": r"\kappa_{2W}",
    "kappa_2z": r"\kappa_{2Z}",
    "kappa_t": r"\kappa_t",
}

KAPPA_PLAIN = {
    "kappa_lambda": "κ_λ",
    "kappa_w": "κ_W",
    "kappa_z": "κ_Z",
    "kappa_2w": "κ_2W",
    "kappa_2z": "κ_2Z",
    "kappa_t": "κ_t",
}

BOUNDARY_VARIANTS = ("min", "max")
CHANNELS = ("WplusHH", "WminusHH", "ZHH")

ZHH_TABLE_GROUPS: Tuple[Tuple[str, ...], ...] = (
    ("kappa_lambda", "kappa_t"),
    ("kappa_z", "kappa_2z"),
)

ZHH_GROUP_TITLES = {
    ("kappa_lambda", "kappa_t"): r"$\kappa_\lambda$, $\kappa_t$",
    ("kappa_z", "kappa_2z"): r"$\kappa_Z$, $\kappa_{2Z}$",
}

ZHH_GROUP_LABELS = {
    ("kappa_lambda", "kappa_t"): "tab:zhh_HEFT_kl_kt",
    ("kappa_z", "kappa_2z"): "tab:zhh_HEFT_kz_k2z",
}


@dataclass(frozen=True)
class LatexTableStyle:
    position: str
    process_name: str
    label: str


LATEX_STYLES: Dict[str, LatexTableStyle] = {
    "WplusHH": LatexTableStyle(
        position="htbp!",
        process_name=r"$W^+hh$",
        label="tab:wplushh_HEFT",
    ),
    "WminusHH": LatexTableStyle(
        position="htbp",
        process_name=r"$W^-hh$",
        label="tab:wminushh_HEFT",
    ),
    "ZHH": LatexTableStyle(
        position="htbp",
        process_name=r"$Zhh$",
        label="tab:zhh_HEFT",
    ),
}


@dataclass(frozen=True)
class SigmaCell:
    sigma_nnlo: float
    scale_up_pct: float
    scale_down_pct: float
    pdf_pct: float

    @classmethod
    def from_prediction(cls, p: Prediction) -> "SigmaCell":
        s = p.sigma_nnlo
        return cls(
            sigma_nnlo=s,
            scale_up_pct=_pct(p.sigma_nnlo_scale_up, s),
            scale_down_pct=_pct(p.sigma_nnlo_scale_down, s),
            pdf_pct=_pct(p.sigma_nnlo_pdfas, s),
        )


def _pct(delta: float, central: float) -> float:
    if central == 0.0 or not central:
        return float("nan")
    return 100.0 * delta / central


def wilson_bounds(axis: str) -> Tuple[float, float]:
    """Return interval endpoints for tables (two-decimal manuscript precision)."""
    lo, hi = WILSON_INTERVALS[axis]
    return (round(float(lo), 2), round(float(hi), 2))


def kappa_with_varied_axis(
    process: str,
    axis: str,
    value: float,
) -> Tuple[float, ...]:
    from .core import resolve_scan_axis

    kappa = list(sm_kappa(process))
    idx, _ = resolve_scan_axis(process, axis)
    kappa[idx] = value
    return tuple(kappa)


def format_sigma_plain(cell: SigmaCell, *, digits: int = 3) -> str:
    return (
        f"{cell.sigma_nnlo:.{digits}f}"
        f" +{cell.scale_up_pct:.2f}%/-{cell.scale_down_pct:.2f}%"
        f" ±{cell.pdf_pct:.1f}%"
    )


def _sigma_latex_digits(sigma_nnlo: float) -> int:
    """Manuscript-style precision: two decimals in $1\\le\\sigma<10$ fb, else three."""
    val = abs(float(sigma_nnlo))
    if 1.0 <= val < 10.0:
        return 2
    return 3


def format_sigma_latex(cell: SigmaCell, *, digits: Optional[int] = None) -> str:
    nd = digits if digits is not None else _sigma_latex_digits(cell.sigma_nnlo)
    return (
        rf"${cell.sigma_nnlo:.{nd}f}"
        rf"^{{+{cell.scale_up_pct:.2f}\%}}_{{-{cell.scale_down_pct:.2f}\%}}"
        rf" \pm {cell.pdf_pct:.1f}\%$"
    )


def _format_energy_latex(energy_tev: float) -> str:
    return rf"${float(energy_tev):.1f}$"


def _format_kappa_bound(val: float) -> str:
    return f"{val:.2f}"


def _column_id(axis: str, variant: str) -> str:
    return f"{axis}_{variant}"


def table_column_keys_for_axes(axes: Sequence[str]) -> List[str]:
    keys = ["SM"]
    for axis in axes:
        for variant in BOUNDARY_VARIANTS:
            keys.append(_column_id(axis, variant))
    return keys


def table_column_keys(process: str) -> List[str]:
    return table_column_keys_for_axes(scan_axes(process))


def table_column_headers_for_axes(axes: Sequence[str]) -> List[str]:
    headers = ["sqrt s [TeV]", "SM"]
    for axis in axes:
        lo, hi = wilson_bounds(axis)
        kplain = KAPPA_PLAIN[axis]
        headers.append(f"{kplain}={lo:.2f}")
        headers.append(f"{kplain}={hi:.2f}")
    return headers


def table_column_headers(process: str) -> List[str]:
    return table_column_headers_for_axes(scan_axes(process))


def _predict_row_cells(
    process: str,
    energy_tev: float,
    *,
    analysis: Optional[VHHAnalysis] = None,
) -> Dict[str, SigmaCell]:
    analysis = analysis or load_analysis(process, energy_tev)
    cells: Dict[str, SigmaCell] = {
        "SM": SigmaCell.from_prediction(predict(analysis, sm_kappa(process))),
    }
    for axis in scan_axes(process):
        lo, hi = wilson_bounds(axis)
        for variant, value in zip(BOUNDARY_VARIANTS, (lo, hi)):
            kappa = kappa_with_varied_axis(process, axis, value)
            cells[_column_id(axis, variant)] = SigmaCell.from_prediction(
                predict(analysis, kappa)
            )
    return cells


def build_wilson_table(
    process: str,
    energy_tev: float,
    *,
    axes: Optional[Sequence[str]] = None,
    analysis: Optional[VHHAnalysis] = None,
) -> Tuple[pd.DataFrame, Dict[str, SigmaCell]]:
    axes = tuple(axes or scan_axes(process))
    all_cells = _predict_row_cells(process, energy_tev, analysis=analysis)
    keys = table_column_keys_for_axes(axes)
    row: Dict[str, object] = {"sqrt_s_TeV": energy_tev}
    for key in keys:
        row[key] = format_sigma_plain(all_cells[key])
    df = pd.DataFrame([row], columns=["sqrt_s_TeV", *keys])
    df.columns = table_column_headers_for_axes(axes)
    return df, {k: all_cells[k] for k in keys}


def build_channel_tables(
    process: str,
    *,
    energies_tev: Sequence[float] = TABLE_ENERGIES_TEV,
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Wide table per channel. ZHH returns two sub-tables keyed by κ group."""
    if process == "ZHH":
        return build_channel_table_groups(process, energies_tev=energies_tev)
    analyses = {e: load_analysis(process, e) for e in energies_tev}
    frames: List[pd.DataFrame] = []
    for energy in energies_tev:
        df, _ = build_wilson_table(process, energy, analysis=analyses[energy])
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def build_channel_table_groups(
    process: str,
    *,
    energies_tev: Sequence[float] = TABLE_ENERGIES_TEV,
) -> Dict[str, pd.DataFrame]:
    analyses = {e: load_analysis(process, e) for e in energies_tev}
    groups: Dict[str, pd.DataFrame] = {}
    axis_groups = ZHH_TABLE_GROUPS if process == "ZHH" else (tuple(scan_axes(process)),)
    for axes in axis_groups:
        frames: List[pd.DataFrame] = []
        for energy in energies_tev:
            df, _ = build_wilson_table(
                process, energy, axes=axes, analysis=analyses[energy]
            )
            frames.append(df)
        key = "_".join(axes)
        groups[key] = pd.concat(frames, ignore_index=True)
    return groups


def _col_spec(n_sigma: int) -> str:
    """Column spec for $\\sqrt{s}$ + *n_sigma* value columns (manuscript layout)."""
    if n_sigma <= 0:
        return "c"
    # e.g. n_sigma=5 → c @{\hspace{10pt}}c@{\hspace{10pt}}…c@{\hspace{10pt}}c  (6 cols)
    return r"c @{\hspace{10pt}}" + r"c@{\hspace{10pt}}" * (n_sigma - 1) + "c"


def _w_channel_caption(process_name: str) -> str:
    return (
        rf"Cross-section values $\sigma$ [fb] at NNLO QCD in HEFT for {process_name} "
        r"at $\sqrt{s} = 13.6$, $14.0$ TeV, for the SM and various $\kappa$ values at their "
        r"exclusion boundaries. All other $\kappa$ parameters are set to their SM values."
    )


def _zhh_caption(axes: Sequence[str]) -> str:
    if tuple(axes) == ("kappa_lambda", "kappa_t"):
        kappa_note = (
            r"the SM and $\kappa_\lambda$, $\kappa_{t}$ at their exclusion boundaries. "
            r"All other $\kappa$ parameters are set to their SM values."
        )
    else:
        kappa_note = (
            r"the SM and $\kappa_Z$, $\kappa_{2Z}$ at their exclusion boundaries. "
            r"All other $\kappa$ coefficients are set to their SM values."
        )
    return (
        r"Cross-section values $\sigma$ [fb] at NNLO QCD in HEFT for $Zhh$ "
        r"at $\sqrt{s} = 13.6$, $14.0$ TeV, for " + kappa_note
    )


def latex_wilson_table(
    process: str,
    *,
    axes: Optional[Sequence[str]] = None,
    energies_tev: Sequence[float] = TABLE_ENERGIES_TEV,
    caption: Optional[str] = None,
    label: Optional[str] = None,
) -> str:
    """LaTeX table matching the manuscript layout (requires multirow package)."""
    axes = tuple(axes or scan_axes(process))
    keys = table_column_keys_for_axes(axes)
    n_sigma = len(keys)
    style = LATEX_STYLES[process]
    last_col = 1 + n_sigma

    if caption is None:
        if process == "ZHH":
            cap = _zhh_caption(axes)
        else:
            cap = _w_channel_caption(style.process_name)
    else:
        cap = caption

    if label is None:
        if process == "ZHH":
            lbl = ZHH_GROUP_LABELS[tuple(axes)]
        else:
            lbl = style.label
    else:
        lbl = label

    row1 = [
        r"\multirow{2}{*}{$\sqrt{s}$}",
        rf"\multicolumn{{{n_sigma}}}{{c}}{{$\sigma_{{NNLO}}$}}",
    ]
    row2 = ["", "SM"]
    for axis in axes:
        lo, hi = wilson_bounds(axis)
        ktex = KAPPA_LATEX[axis]
        row2.append(rf"${ktex}={_format_kappa_bound(lo)}$")
        row2.append(rf"${ktex}={_format_kappa_bound(hi)}$")

    lines = [
        rf"\begin{{table}}[{style.position}]",
        r"\centering",
        r"\resizebox{\textwidth}{!}{%",
        r"  \renewcommand{\arraystretch}{1.45}%",
        r"  \setlength{\tabcolsep}{7pt}%",
        r"  \begin{tabular}{" + _col_spec(n_sigma) + "}",
        r"    \hline",
        "    " + " & ".join(row1) + r" \\",
        rf"    \cline{{{2}-{last_col}}}",
        "     & " + " & ".join(row2[1:]) + r" \\",
        r"    \hline",
    ]

    analyses = {e: load_analysis(process, e) for e in energies_tev}
    for energy in energies_tev:
        _, cells = build_wilson_table(
            process, energy, axes=axes, analysis=analyses[energy]
        )
        row = [_format_energy_latex(energy)] + [
            format_sigma_latex(cells[k]) for k in keys
        ]
        lines.append("    " + " & ".join(row) + r" \\")

    lines.extend(
        [
            r"    \hline",
            r"  \end{tabular}%",
            r"}",
            rf"\caption{{{cap}}}",
            rf"\label{{{lbl}}}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def latex_wilson_tables_for_process(
    process: str,
    *,
    energies_tev: Sequence[float] = TABLE_ENERGIES_TEV,
) -> str:
    if process == "ZHH":
        return "\n\n".join(
            latex_wilson_table(process, axes=axes, energies_tev=energies_tev)
            for axes in ZHH_TABLE_GROUPS
        )
    return latex_wilson_table(process, energies_tev=energies_tev)


def all_channels_latex(
    *,
    energies_tev: Sequence[float] = TABLE_ENERGIES_TEV,
) -> str:
    return "\n\n".join(
        latex_wilson_tables_for_process(proc, energies_tev=energies_tev)
        for proc in CHANNELS
    )
