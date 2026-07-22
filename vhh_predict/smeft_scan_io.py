"""WC-scan I/O for SMEFT predictions."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .analysis import points_dir
from .smeft_analysis import SMEFTAnalysis
from .smeft_core import scan
from .smeft_operators import SMEFT_WC_INTERVALS, SMEFT_WC_PLAIN

ArrayDict = Dict[str, np.ndarray]

SCAN_VALUE_KEYS = (
    "sigma_lo",
    "sigma_nnlo",
    "k",
    "sigma_smeft_over_sm_lo",
    "sigma_smeft_over_sm_nnlo",
)


def smeft_scan_points_path(
    process: str,
    energy_tev: float,
    scan_x_key: str,
    *,
    root: Optional[Path] = None,
) -> Path:
    name = f"{process}_{float(energy_tev):g}TeV_{scan_x_key}.txt"
    return (root or points_dir()) / "SMEFT" / name


def scan_and_save(
    analysis: SMEFTAnalysis,
    axis: str,
    *,
    vmin: float,
    vmax: float,
    fixed_wcs: Optional[Dict[str, float]] = None,
    n_points: int = 400,
    path: Optional[Path] = None,
    save: bool = True,
    uncertainties: bool = False,
) -> Tuple[ArrayDict, Path]:
    data = scan(
        analysis,
        axis,
        vmin=vmin,
        vmax=vmax,
        n_points=n_points,
        fixed_wcs=fixed_wcs,
        uncertainties=uncertainties,
    )
    from .smeft_core import resolve_scan_axis

    _, x_key = resolve_scan_axis(analysis.process, axis)
    out_path = path or smeft_scan_points_path(analysis.process, analysis.energy_tev, x_key)
    if save:
        from .smeft_operators import normalize_wc_dict, sm_wc_values

        wcs_used = normalize_wc_dict(analysis.process, fixed_wcs or sm_wc_values(analysis.process))
        _write_scan_file(
            out_path,
            analysis=analysis,
            scan_axis=axis,
            scan_x_key=x_key,
            vmin=vmin,
            vmax=vmax,
            fixed_wcs=wcs_used,
            data=data,
        )
    return data, out_path


def scan_axes_and_save(
    analysis: SMEFTAnalysis,
    axes: Sequence[str],
    *,
    windows: Optional[Dict[str, Tuple[float, float]]] = None,
    n_points: int = 400,
    fixed_wcs: Optional[Dict[str, float]] = None,
    save: bool = True,
    uncertainties: bool = False,
) -> Dict[str, Tuple[ArrayDict, Path]]:
    results: Dict[str, Tuple[ArrayDict, Path]] = {}
    for axis in axes:
        lo, hi = (windows or {}).get(axis) or SMEFT_WC_INTERVALS.get(axis, (-1.0, 1.0))
        data, path = scan_and_save(
            analysis,
            axis,
            vmin=lo,
            vmax=hi,
            fixed_wcs=fixed_wcs,
            n_points=n_points,
            save=save,
            uncertainties=uncertainties,
        )
        results[axis] = (data, path)
    return results


def _write_scan_file(
    path: Path,
    *,
    analysis: SMEFTAnalysis,
    scan_axis: str,
    scan_x_key: str,
    vmin: float,
    vmax: float,
    fixed_wcs: Dict[str, float],
    data: ArrayDict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fixed_str = ",".join(f"{k}={fixed_wcs[k]:g}" for k in sorted(fixed_wcs))
    header = [
        "# VHH-NNLO SMEFT scan points",
        f"# process: {analysis.process}",
        f"# energy_tev: {analysis.energy_tev}",
        f"# scan_axis: {scan_axis}",
        f"# fixed_wcs: {fixed_str}",
        "#",
        "\t".join(
            [scan_x_key]
            + list(SCAN_VALUE_KEYS)
        ),
    ]
    lines = ["\n".join(header)]
    n = len(data[scan_x_key])
    for i in range(n):
        row = [f"{data[scan_x_key][i]:.10g}"]
        for key in SCAN_VALUE_KEYS:
            row.append(f"{data[key][i]:.10g}")
        lines.append("\t".join(row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
