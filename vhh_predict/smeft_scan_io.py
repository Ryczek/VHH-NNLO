"""WC-scan I/O for SMEFT predictions."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .analysis import points_dir
from .smeft_analysis import SMEFTAnalysis
from .smeft_core import scan
from .smeft_operators import SMEFT_WC_INTERVALS

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
    return (root or points_dir("SMEFT", process, energy_tev)) / name


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
    return (root or points_dir("SMEFT", process, energy_tev)) / name


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
    """Independent 1D scans (one file per axis). Prefer ``scan_grid_and_save`` for joint scans."""
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


def scan_grid_and_save(
    analysis: SMEFTAnalysis,
    axes: Sequence[str],
    *,
    windows: Optional[Dict[str, Tuple[float, float]]] = None,
    n_points: int = 40,
    fixed_wcs: Optional[Dict[str, float]] = None,
    path: Optional[Path] = None,
    save: bool = True,
    uncertainties: bool = False,
) -> Tuple[ArrayDict, Path]:
    """Scan *axes* **simultaneously** on a Cartesian grid; one ``.txt`` under ``results/points/smeft/{Process}/{energy}/``."""
    from .smeft_core import resolve_scan_axis, scan_grid
    from .smeft_operators import normalize_wc_dict, scan_axes, sm_wc_values

    if not axes:
        raise ValueError("scan_grid_and_save requires at least one axis")

    allowed = set(scan_axes(analysis.process))
    for axis in axes:
        if axis not in allowed:
            raise ValueError(
                f"Invalid scan axis {axis!r} for {analysis.process}; "
                f"allowed: {', '.join(sorted(allowed))}"
            )

    data = scan_grid(
        analysis,
        axes,
        windows=windows,
        n_points=n_points,
        fixed_wcs=fixed_wcs,
        uncertainties=uncertainties,
    )
    x_keys = [resolve_scan_axis(analysis.process, a)[1] for a in axes]
    out_path = path or scan_grid_points_path(analysis.process, analysis.energy_tev, axes)
    if save:
        wcs_used = normalize_wc_dict(analysis.process, fixed_wcs or sm_wc_values(analysis.process))
        resolved_windows = {
            a: ((windows or {}).get(a) or SMEFT_WC_INTERVALS[a]) for a in axes
        }
        _write_grid_scan_file(
            out_path,
            analysis=analysis,
            axes=list(axes),
            x_keys=x_keys,
            windows={a: (float(lo), float(hi)) for a, (lo, hi) in resolved_windows.items()},
            n_points=n_points,
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
        "\t".join([scan_x_key] + list(SCAN_VALUE_KEYS)),
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
    axes: Sequence[str],
    x_keys: Sequence[str],
    windows: Dict[str, Tuple[float, float]],
    n_points: int,
    fixed_wcs: Dict[str, float],
    data: ArrayDict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [*x_keys, *SCAN_VALUE_KEYS]
    fixed_str = ",".join(f"{k}={fixed_wcs[k]:g}" for k in sorted(fixed_wcs))
    win_str = "; ".join(f"{a}=[{windows[a][0]:g},{windows[a][1]:g}]" for a in axes)
    n_total = len(data[x_keys[0]])
    header = [
        "# VHH-NNLO SMEFT grid scan points",
        f"# process: {analysis.process}",
        f"# energy_tev: {analysis.energy_tev}",
        f"# scan_axes: {','.join(axes)}",
        f"# windows: {win_str}",
        f"# n_points_per_axis: {int(n_points)}",
        f"# n_points_total: {int(n_total)}",
        f"# fixed_wcs: {fixed_str}",
        "#",
        "\t".join(columns),
    ]
    lines = ["\n".join(header)]
    for i in range(n_total):
        row = [f"{data[col][i]:.10g}" for col in columns]
        lines.append("\t".join(row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
