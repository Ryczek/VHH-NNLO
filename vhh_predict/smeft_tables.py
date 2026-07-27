"""SMEFT Wilson-coefficient benchmark tables."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from .smeft_analysis import load_smeft_analysis
from .smeft_core import SMEFTPrediction, predict
from .smeft_operators import (
    SMEFT_WC_INTERVALS,
    SMEFT_WC_LATEX,
    SMEFT_WC_PLAIN,
    scan_axes,
    sm_wc_values,
)

TABLE_ENERGIES_TEV = (13.6, 14.0)
# Manuscript table order (distinct from PROCESSES, which starts with ZHH).
CHANNELS = ("WplusHH", "WminusHH", "ZHH")

# Publication column order (matches manuscript layout).
W_TABLE_AXES: Tuple[str, ...] = ("phi", "phiW", "phiD", "phiBox", "phiq3st")

# Optional display grouping for ZHH (§5). Together these cover every scan axis.
ZHH_TABLE_GROUPS: Tuple[Tuple[str, ...], ...] = (
    ("phi", "phiW", "phiD", "phiBox", "phiq3st"),
    ("phiq1st", "phiu", "phid", "phiQ3rd"),
    ("tphi", "phiB", "phiWB"),
)

ZHH_GROUP_LABELS = {
    ("phi", "phiW", "phiD", "phiBox", "phiq3st"): "tab:zhh_SMEFT_group1",
    ("phiq1st", "phiu", "phid", "phiQ3rd"): "tab:zhh_SMEFT_group2",
    ("tphi", "phiB", "phiWB"): "tab:zhh_SMEFT_group3",
}

# Manuscript uses \\Box rather than \\square for C_{φ□}.
_SMEFT_HEADER_LATEX = {
    **SMEFT_WC_LATEX,
    "phiBox": r"C_{\varphi \Box}",
    "phiQ3rd": r"C_{\varphi t} + C_{\varphi Q}^{(3)} - C_{\varphi Q}^{(1)}",
}


@dataclass(frozen=True)
class LatexTableStyle:
    position: str
    process_name: str
    label: str
    resize_width: str = r"\textwidth"


LATEX_STYLES: Dict[str, LatexTableStyle] = {
    "WplusHH": LatexTableStyle(
        position="htbp!",
        process_name=r"$W^+hh$",
        label="tab:wplushh_SMEFT",
    ),
    "WminusHH": LatexTableStyle(
        position="htbp!",
        process_name=r"$W^-hh$",
        label="tab:wminushh_SMEFT",
    ),
    "ZHH": LatexTableStyle(
        position="htbp!",
        process_name=r"$Zhh$",
        label="tab:zhh_SMEFT",
    ),
}


@dataclass(frozen=True)
class SigmaCell:
    sigma_nnlo: float
    scale_up_pct: float
    scale_down_pct: float
    pdf_pct: float

    @classmethod
    def from_prediction(cls, p: SMEFTPrediction) -> "SigmaCell":
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
    return 100.0 * delta / abs(central)


def _boundary_wcs(process: str, axis: str, value: float) -> Dict[str, float]:
    wcs = sm_wc_values(process)
    wcs[axis] = float(value)
    return wcs


def _table_groups(process: str) -> Tuple[Tuple[str, ...], ...]:
    """Partition of all scan axes used for benchmark σ tables."""
    axes = scan_axes(process)
    if process != "ZHH":
        ordered = tuple(k for k in W_TABLE_AXES if k in axes)
        missing = tuple(k for k in axes if k not in ordered)
        return (ordered + missing,)
    covered = {k for g in ZHH_TABLE_GROUPS for k in g}
    missing = tuple(k for k in axes if k not in covered)
    groups = tuple(tuple(k for k in g if k in axes) for g in ZHH_TABLE_GROUPS)
    groups = tuple(g for g in groups if g)
    if missing:
        groups = groups + (missing,)
    return groups


def _max_sigma_wc(
    process: str,
    axis: str,
    analysis,
) -> tuple[float, str]:
    """Return interval endpoint that maximises σ_NNLO (others at SM)."""
    lo, hi = SMEFT_WC_INTERVALS[axis]
    wcs_lo = _boundary_wcs(process, axis, lo)
    wcs_hi = _boundary_wcs(process, axis, hi)
    if predict(analysis, wcs_hi).sigma_nnlo >= predict(analysis, wcs_lo).sigma_nnlo:
        return hi, "max"
    return lo, "min"


def build_channel_tables(
    process: str,
    *,
    energies_tev: Sequence[float] = TABLE_ENERGIES_TEV,
) -> Dict[str, pd.DataFrame]:
    """σ benchmark tables: SM plus one max-σ WC endpoint per axis (no Point column)."""
    tables: Dict[str, pd.DataFrame] = {}
    for axes in _table_groups(process):
        key = "_".join(axes) if len(axes) < len(scan_axes(process)) else "all"
        if process == "ZHH":
            key = "_".join(axes)
        tables[key] = _build_boundary_table(process, axes, energies_tev=energies_tev)
    return tables


def _build_boundary_table(
    process: str,
    axes: Sequence[str],
    *,
    energies_tev: Sequence[float],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for energy in energies_tev:
        analysis = load_smeft_analysis(process, energy)
        nn_label = analysis.nnlo_label
        sm = sm_wc_values(process)
        p_sm = predict(analysis, sm)
        rows.append(
            {
                "√s [TeV]": energy,
                "WC": "SM",
                "σ_LO [fb]": p_sm.sigma_lo,
                f"σ_{nn_label} [fb]": p_sm.sigma_nnlo,
                "K": p_sm.k_factor,
            }
        )
        for axis in axes:
            if axis not in SMEFT_WC_INTERVALS:
                continue
            val, _ = _max_sigma_wc(process, axis, analysis)
            wcs = _boundary_wcs(process, axis, val)
            p = predict(analysis, wcs)
            rows.append(
                {
                    "√s [TeV]": energy,
                    "WC": f"{SMEFT_WC_PLAIN.get(axis, axis)}={val:g}",
                    "σ_LO [fb]": p.sigma_lo,
                    f"σ_{nn_label} [fb]": p.sigma_nnlo,
                    "K": p.k_factor,
                }
            )
    return pd.DataFrame(rows)


def latex_wc_interval_table() -> str:
    """LaTeX table of allowed SMEFT WC intervals (bosonic + fermionic)."""
    bosonic = ("phi", "phiW", "phiB", "phiWB", "phiD", "phiBox")
    fermionic = (
        "phiq3st",
        "phit",
        "phiQ3",
        "phiQ1rd",
        "phiQ3rd",
        "phiq1st",
        "phiu",
        "phid",
        "tphi",
    )
    lines = []
    for title, keys in (("Bosonic", bosonic), ("Fermionic", fermionic)):
        valid_keys = [k for k in keys if k in SMEFT_WC_INTERVALS]
        hdr = " & ".join(
            f"${_SMEFT_HEADER_LATEX.get(k, SMEFT_WC_LATEX.get(k, k))}$" for k in valid_keys
        )
        row = " & ".join(
            f"$[{SMEFT_WC_INTERVALS[k][0]:g},\\ {SMEFT_WC_INTERVALS[k][1]:g}]$"
            for k in valid_keys
        )
        ncol = len(valid_keys)
        lines.append(f"% {title} SMEFT intervals")
        lines.append(r"\begin{tabular}{|" + "c|" * (ncol + 1) + "}")
        lines.append(r"\hline")
        lines.append("Coefficient $[1/\\mathrm{TeV}^2]$ & " + hdr + r" \\ \hline")
        lines.append("Interval & " + row + r" \\ \hline")
        lines.append(r"\end{tabular}")
        lines.append("")
    return "\n".join(lines)


def format_sigma_latex(cell: SigmaCell, *, digits: int = 3) -> str:
    """Manuscript SMEFT form: $x_{+..%}^{-..%} \\pm ..%$."""
    return (
        rf"${cell.sigma_nnlo:.{digits}f}"
        rf"_{{+{cell.scale_up_pct:.2f}\%}}^{{-{cell.scale_down_pct:.2f}\%}}"
        rf" \pm {cell.pdf_pct:.1f}\%$"
    )


def _format_energy_latex(energy_tev: float) -> str:
    return rf"${float(energy_tev):.1f}$"


def _format_wc_bound(val: float) -> str:
    return f"{val:g}"


def _wc_header(axis: str, value: float) -> str:
    tex = _SMEFT_HEADER_LATEX.get(axis, SMEFT_WC_LATEX.get(axis, axis))
    # Long combo header matches manuscript spacing around '='.
    if axis == "phiQ3rd":
        return rf"${tex} = {_format_wc_bound(value)}$"
    return rf"${tex}={_format_wc_bound(value)}$"


def _col_spec(n_sigma: int) -> str:
    """Column spec for $\\sqrt{s}$ + *n_sigma* value columns."""
    if n_sigma <= 0:
        return "c"
    if n_sigma == 1:
        return "c c"
    return "c c@{\\hspace{10pt}}" + r"c@{\hspace{10pt}}" * (n_sigma - 2) + "c"


def _caption(process_name: str) -> str:
    return (
        rf"Cross-section values $\sigma$ [fb] at NNLO QCD in SMEFT for {process_name} "
        r"at $\sqrt{s}=13.6,\, 14.0$ TeV, with each Wilson coefficient set to the value "
        r"maximising $\sigma$ within its scan interval. All other coefficients are set to "
        r"their SM values."
    )


def _predict_cells(
    process: str,
    energy_tev: float,
    axes: Sequence[str],
    *,
    analysis=None,
    wc_values: Optional[Dict[str, float]] = None,
) -> Dict[str, SigmaCell]:
    analysis = analysis or load_smeft_analysis(process, energy_tev)
    cells: Dict[str, SigmaCell] = {
        "SM": SigmaCell.from_prediction(predict(analysis, sm_wc_values(process))),
    }
    for axis in axes:
        if axis not in SMEFT_WC_INTERVALS:
            continue
        val = (
            wc_values[axis]
            if wc_values is not None and axis in wc_values
            else _max_sigma_wc(process, axis, analysis)[0]
        )
        cells[axis] = SigmaCell.from_prediction(
            predict(analysis, _boundary_wcs(process, axis, val))
        )
    return cells


def latex_wilson_table(
    process: str,
    *,
    axes: Optional[Sequence[str]] = None,
    energies_tev: Sequence[float] = TABLE_ENERGIES_TEV,
    caption: Optional[str] = None,
    label: Optional[str] = None,
    resize_width: Optional[str] = None,
) -> str:
    """LaTeX table: SM + one max-σ WC column per axis (manuscript layout)."""
    groups = _table_groups(process)
    axes = tuple(axes or groups[0])
    style = LATEX_STYLES[process]
    keys = ["SM", *axes]
    n_sigma = len(keys)
    last_col = 1 + n_sigma

    # Max-σ endpoints from the first energy (interval endpoints are energy-independent).
    ref = load_smeft_analysis(process, float(energies_tev[0]))
    wc_values = {
        axis: _max_sigma_wc(process, axis, ref)[0]
        for axis in axes
        if axis in SMEFT_WC_INTERVALS
    }

    if caption is None:
        cap = _caption(style.process_name)
    else:
        cap = caption

    if label is None:
        if process == "ZHH":
            lbl = ZHH_GROUP_LABELS.get(tuple(axes), f"{style.label}_{'_'.join(axes)}")
        else:
            lbl = style.label
    else:
        lbl = label

    width = resize_width
    if width is None:
        if process == "ZHH" and tuple(axes) == ("tphi", "phiB", "phiWB"):
            width = r"0.8\textwidth"
        else:
            width = style.resize_width

    row1 = [
        r"\multirow{2}{*}{$\sqrt{s}$}",
        rf"\multicolumn{{{n_sigma}}}{{c}}{{$\sigma_{{NNLO}}$}}",
    ]
    row2 = ["", "SM"] + [_wc_header(axis, wc_values[axis]) for axis in axes]

    lines = [
        rf"\begin{{table}}[{style.position}]",
        r"\centering",
        rf"\resizebox{{{width}}}{{!}}{{%",
        r"  \renewcommand{\arraystretch}{1.45}%",
        r"  \setlength{\tabcolsep}{7pt}%",
        r"  \begin{tabular}{" + _col_spec(n_sigma) + "}",
        r"    \hline",
        "    " + " & ".join(row1) + r" \\",
        rf"    \cline{{{2}-{last_col}}}",
        "     & " + " & ".join(row2[1:]) + r" \\",
        r"    \hline",
    ]

    analyses = {e: load_smeft_analysis(process, e) for e in energies_tev}
    for energy in energies_tev:
        cells = _predict_cells(
            process,
            energy,
            axes,
            analysis=analyses[energy],
            wc_values=wc_values,
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


def latex_publication_tables(
    energy_tev: Optional[float] = None,
    *,
    energies_tev: Optional[Sequence[float]] = None,
    processes: Sequence[str] = CHANNELS,
) -> str:
    """Publication LaTeX for SMEFT σ tables (both energies by default)."""
    del energy_tev
    energies = tuple(energies_tev or TABLE_ENERGIES_TEV)
    pieces = [
        latex_wilson_tables_for_process(process, energies_tev=energies)
        for process in processes
    ]
    return "\n\n".join(pieces)
