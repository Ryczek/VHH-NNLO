"""SMEFT Wilson-coefficient benchmark tables."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import pandas as pd

from .analysis import PROCESSES
from .smeft_analysis import load_smeft_analysis
from .smeft_core import predict
from .smeft_operators import (
    SMEFT_WC_INTERVALS,
    SMEFT_WC_LATEX,
    SMEFT_WC_PLAIN,
    scan_axes,
    sm_wc_values,
)

TABLE_ENERGIES_TEV = (13.6, 14.0)
CHANNELS = PROCESSES

# Optional display grouping for ZHH (§5). Together these cover every scan axis.
ZHH_TABLE_GROUPS: Tuple[Tuple[str, ...], ...] = (
    ("phi", "phiBox", "phiD", "phiW", "phiB", "phiWB"),
    ("phiq3st", "phiq1st", "phiu", "phid"),
    ("tphi", "phiQ3rd"),
)


def _boundary_wcs(process: str, axis: str, value: float) -> Dict[str, float]:
    wcs = sm_wc_values(process)
    wcs[axis] = float(value)
    return wcs


def _table_groups(process: str) -> Tuple[Tuple[str, ...], ...]:
    """Partition of all scan axes used for benchmark σ tables."""
    axes = scan_axes(process)
    if process != "ZHH":
        return (axes,)
    covered = {k for g in ZHH_TABLE_GROUPS for k in g}
    missing = tuple(k for k in axes if k not in covered)
    groups = tuple(tuple(k for k in g if k in axes) for g in ZHH_TABLE_GROUPS)
    groups = tuple(g for g in groups if g)
    if missing:
        groups = groups + (missing,)
    return groups


def build_channel_tables(
    process: str,
    *,
    energies_tev: Sequence[float] = TABLE_ENERGIES_TEV,
) -> Dict[str, pd.DataFrame]:
    """Boundary σ tables for every scan-axis WC of ``process``."""
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
                "Point": "SM",
                "WC": "—",
                "σ_LO [fb]": p_sm.sigma_lo,
                f"σ_{nn_label} [fb]": p_sm.sigma_nnlo,
                "K": p_sm.k_factor,
            }
        )
        for axis in axes:
            if axis not in SMEFT_WC_INTERVALS:
                continue
            lo, hi = SMEFT_WC_INTERVALS[axis]
            for label, val in (("min", lo), ("max", hi)):
                wcs = _boundary_wcs(process, axis, val)
                p = predict(analysis, wcs)
                rows.append(
                    {
                        "√s [TeV]": energy,
                        "Point": label,
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
        hdr = " & ".join(f"${SMEFT_WC_LATEX.get(k, k)}$" for k in valid_keys)
        row = " & ".join(
            f"$[{SMEFT_WC_INTERVALS[k][0]:g},\\ {SMEFT_WC_INTERVALS[k][1]:g}]$" for k in valid_keys
        )
        ncol = len(valid_keys)
        lines.append(f"% {title} SMEFT intervals")
        lines.append(r"\begin{tabular}{|" + "c|" * (ncol + 1) + "}")
        lines.append(r"\hline")
        lines.append(
            "Coefficient $[1/\\mathrm{TeV}^2]$ & " + hdr + r" \\ \hline"
        )
        lines.append("Interval & " + row + r" \\ \hline")
        lines.append(r"\end{tabular}")
        lines.append("")
    return "\n".join(lines)
