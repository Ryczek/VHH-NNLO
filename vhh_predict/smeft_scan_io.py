"""WC-scan I/O for SMEFT predictions."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

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
    # PDF+αs (symmetric) and scale envelope (up / down)
    "sigma_lo_pdfas",
    "sigma_lo_sup",
    "sigma_lo_inf",
    "sigma_nnlo_pdfas",
    "sigma_nnlo_sup",
    "sigma_nnlo_inf",
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
    """Scan one WC; when ``save=True``, also write σ PDF+αs / scale columns."""
    need_unc = bool(uncertainties or save)
    data = scan(
        analysis,
        axis,
        vmin=vmin,
        vmax=vmax,
        n_points=n_points,
        fixed_wcs=fixed_wcs,
        uncertainties=need_unc,
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
    """Independent **1D** scans (one file per axis). For a joint grid use ``scan_grid_and_save``."""
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


def scan_grid_points_path(
    process: str,
    energy_tev: float,
    axes: Sequence[str],
    *,
    root: Optional[Path] = None,
) -> Path:
    from .smeft_core import resolve_scan_axis

    keys = [resolve_scan_axis(process, a)[1] for a in axes]
    name = f"{process}_{float(energy_tev):g}TeV_{'_x_'.join(keys)}.txt"
    return (root or points_dir()) / "SMEFT" / name


def scan_grid_and_save(
    analysis: SMEFTAnalysis,
    axes: Sequence[str],
    *,
    windows: Optional[Dict[str, Tuple[float, float]]] = None,
    n_points: Union[int, Dict[str, int]] = 40,
    fixed_wcs: Optional[Dict[str, float]] = None,
    path: Optional[Path] = None,
    save: bool = True,
    uncertainties: bool = False,
) -> Tuple[ArrayDict, Path]:
    """Scan *axes* **simultaneously** on a Cartesian grid; one ``.txt`` under ``Results/Points/SMEFT/``.

    When ``save=True``, σ PDF+αs / scale columns are always computed and written.
    """
    from .smeft_core import resolve_scan_axis, scan_grid
    from .smeft_operators import normalize_wc_dict, sm_wc_values

    if not axes:
        raise ValueError("scan_grid_and_save requires at least one axis")

    need_unc = bool(uncertainties or save)
    data = scan_grid(
        analysis,
        axes,
        windows=windows,
        n_points=n_points,
        fixed_wcs=fixed_wcs,
        uncertainties=need_unc,
    )
    out_path = path or scan_grid_points_path(analysis.process, analysis.energy_tev, axes)
    if save:
        axis_keys = [resolve_scan_axis(analysis.process, a)[1] for a in axes]
        wins: Dict[str, Tuple[float, float]] = {}
        n_map: Dict[str, int] = {}
        for axis, key in zip(axes, axis_keys):
            wins[key] = (windows or {}).get(axis) or (windows or {}).get(key) or SMEFT_WC_INTERVALS.get(
                key, SMEFT_WC_INTERVALS.get(axis, (-1.0, 1.0))
            )
            if isinstance(n_points, int):
                n_map[key] = n_points
            else:
                n_map[key] = int(n_points.get(axis, n_points.get(key, 40)))
        wcs_used = normalize_wc_dict(analysis.process, fixed_wcs or sm_wc_values(analysis.process))
        _write_grid_scan_file(
            out_path,
            analysis=analysis,
            axis_keys=axis_keys,
            windows=wins,
            n_points=n_map,
            fixed_wcs=wcs_used,
            data=data,
        )
    return data, out_path


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


def _write_grid_scan_file(
    path: Path,
    *,
    analysis: SMEFTAnalysis,
    axis_keys: Sequence[str],
    windows: Dict[str, Tuple[float, float]],
    n_points: Dict[str, int],
    fixed_wcs: Dict[str, float],
    data: ArrayDict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [*axis_keys, *SCAN_VALUE_KEYS]
    fixed_str = ",".join(f"{k}={fixed_wcs[k]:g}" for k in sorted(fixed_wcs))
    win_str = "; ".join(f"{k}=[{windows[k][0]:g},{windows[k][1]:g}]" for k in axis_keys)
    n_str = ",".join(f"{k}:{n_points[k]}" for k in axis_keys)
    header = [
        "# VHH-NNLO SMEFT multi-axis scan points",
        f"# process: {analysis.process}",
        f"# energy_tev: {analysis.energy_tev}",
        f"# scan_axes: {','.join(axis_keys)}",
        f"# windows: {win_str}",
        f"# n_points: {n_str}",
        f"# fixed_wcs: {fixed_str}",
        "#",
        "\t".join(columns),
    ]
    lines = ["\n".join(header)]
    n = len(data[axis_keys[0]])
    for i in range(n):
        row = [f"{data[k][i]:.10g}" for k in axis_keys]
        for key in SCAN_VALUE_KEYS:
            row.append(f"{data[key][i]:.10g}")
        lines.append("\t".join(row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
