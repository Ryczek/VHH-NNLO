"""Wilson-coefficient benchmark tables (paper-style σ with uncertainties)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd

from .analysis import VHHAnalysis, load_analysis
from .core import Prediction, predict, scan_axes, sm_kappa

# Table 2 in the HEFT Wilson-coefficient uncertainty note (95% CL intervals).
WILSON_INTERVALS: Dict[str, Tuple[float, float]] = {
    "kappa_lambda": (-0.7, 6.1),
    "kappa_w": (0.8, 1.2),
    "kappa_z": (0.9, 1.2),
    "kappa_2w": (0.7, 1.3),
    "kappa_2z": (0.7, 1.3),
    "kappa_t": (0.8, 1.2),
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
    sqrt_col: str
    pipe_after_sqrt: bool
    sqrt_header: str
    sigma_header: str  # format with {n}
    process_name: str
    label: str


LATEX_STYLES: Dict[str, LatexTableStyle] = {
    "WplusHH": LatexTableStyle(
        position="htbp",
        sqrt_col="c",
        pipe_after_sqrt=True,
        sqrt_header=r"\multirow{2}{*}{$\sqrt{s}$}",
        sigma_header=r"\multicolumn{{{n}}}{{c}}{{$\sigma$}}",
        process_name=r"$W^+HH$",
        label="tab:wplushh_HEFT",
    ),
    "WminusHH": LatexTableStyle(
        position="ht",
        sqrt_col="l",
        pipe_after_sqrt=False,
        sqrt_header=r"\multirow{2}{*}{$\sqrt{s}$ [TeV]}",
        sigma_header=r"\multicolumn{{{n}}}{{c|}}{{$\sigma$ [fb]}}",
        process_name=r"$W^-HH$",
        label="tab:wminushh_HEFT",
    ),
    "ZHH": LatexTableStyle(
        position="htbp",
        sqrt_col="l",
        pipe_after_sqrt=False,
        sqrt_header=r"\multirow{2}{*}{$\sqrt{s}$ [TeV]}",
        sigma_header=r"\multicolumn{{{n}}}{{c}}{{$\sigma$ [fb]}}",
        process_name=r"$ZHH$",
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
    lo, hi = WILSON_INTERVALS[axis]
    return (math.floor(lo * 10 + 0.5) / 10, math.floor(hi * 10 + 0.5) / 10)


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


def format_sigma_latex(cell: SigmaCell, *, digits: int = 3) -> str:
    return (
        rf"${cell.sigma_nnlo:.{digits}f}"
        rf"_{{+{cell.scale_up_pct:.2f}\%}}^{{-{cell.scale_down_pct:.2f}\%}}"
        rf" \pm {cell.pdf_pct:.1f}\%$"
    )


def _format_energy_latex(energy_tev: float) -> str:
    return rf"${energy_tev:g}$"


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
        headers.append(f"{kplain}={lo:.1f}")
        headers.append(f"{kplain}={hi:.1f}")
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


def _col_spec(style: LatexTableStyle, n_sigma: int) -> str:
    pipe = " | " if style.pipe_after_sqrt else ""
    return style.sqrt_col + pipe + r"@{\hspace{10pt}}c" * n_sigma


def _default_caption(style: LatexTableStyle, kappa_note: str) -> str:
    return (
        rf"Cross-section values $\sigma$ [fb] for {style.process_name} process "
        rf"for $\sqrt{{s}}=13.6,\, 14$ TeV, with {kappa_note}"
        r"each $\kappa$ at its interval boundary (others taken at their SM values)"
    )


def _zhh_caption(axes: Sequence[str]) -> str:
    title = ZHH_GROUP_TITLES[tuple(axes)]
    style = LATEX_STYLES["ZHH"]
    return (
        rf"Cross-section values $\sigma$ [fb] for {style.process_name} process "
        rf"for $\sqrt{{s}}=13.6,\, 14$ TeV, with {title} at their interval "
        r"boundaries (other $\kappa$ taken at their SM values)"
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
            cap = _default_caption(style, kappa_note="")
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
        style.sqrt_header,
        style.sigma_header.format(n=n_sigma),
    ]
    row2 = ["", "SM"]
    for axis in axes:
        lo, hi = wilson_bounds(axis)
        ktex = KAPPA_LATEX[axis]
        row2.append(rf"${ktex}={lo:.1f}$")
        row2.append(rf"${ktex}={hi:.1f}$")

    lines = [
        rf"\begin{{table}}[{style.position}]",
        r"\centering",
        r"\resizebox{\textwidth}{!}{%",
        r"  \renewcommand{\arraystretch}{1.45}%",
        r"  \setlength{\tabcolsep}{7pt}%",
        r"  \begin{tabular}{" + _col_spec(style, n_sigma) + "}",
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
            r"}%",
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
