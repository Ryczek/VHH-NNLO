"""κ-scan: one function to evaluate and save point tables under ``Results/Points/``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .analysis import VHHAnalysis, points_dir

ArrayDict = Dict[str, np.ndarray]

SCAN_VALUE_KEYS = (
    "sigma_lo",
    "sigma_nnlo",
    "k",
    "sigma_heft_over_sm_lo",
    "sigma_heft_over_sm_nnlo",
)


def scan_points_path(
    process: str,
    energy_tev: float,
    scan_x_key: str,
    *,
    root: Optional[Path] = None,
) -> Path:
    """``Results/Points/{Process}_{energy}TeV_{axis}.json``."""
    name = f"{process}_{float(energy_tev):g}TeV_{scan_x_key}.json"
    return (root or points_dir()) / name


def scan_and_save(
    analysis: VHHAnalysis,
    axis: str,
    *,
    vmin: float,
    vmax: float,
    fixed_kappa: Optional[Tuple[float, ...]] = None,
    n_points: int = 400,
    path: Optional[Path] = None,
    save: bool = True,
    uncertainties: bool = False,
) -> Tuple[ArrayDict, Path]:
    """Scan one κ component and optionally save σ_LO, σ_NNLO, K, σ_HEFT/σ_SM.

  Returns ``(scan_data, path)``. The saved JSON contains only the scanned κ grid
  and the five physics columns above (no uncertainty bands).

  Set ``uncertainties=True`` if you need PDF/scale bands for plotting (extra
  columns in memory only; the file stays slim).
    """
    from .core import resolve_scan_axis, scan

    data = scan(
        analysis,
        axis,
        vmin=vmin,
        vmax=vmax,
        n_points=n_points,
        fixed_kappa=fixed_kappa,
        uncertainties=uncertainties,
    )
    _, x_key = resolve_scan_axis(analysis.process, axis)
    out_path = path or scan_points_path(analysis.process, analysis.energy_tev, x_key)
    if save:
        from .core import _scan_base_kappa

        kappa_used = tuple(_scan_base_kappa(analysis, fixed_kappa))
        _write_scan_file(
            out_path,
            analysis=analysis,
            scan_axis=axis,
            scan_x_key=x_key,
            vmin=vmin,
            vmax=vmax,
            n_points=n_points,
            fixed_kappa=kappa_used,
            data=data,
        )
    return data, out_path


def load_scan_results(path: Path) -> Dict[str, Any]:
    """Load a scan JSON; array values are numpy ndarrays."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    version = payload.get("version", 1)

    if version == 2:
        meta_keys = {
            "version",
            "process",
            "energy_tev",
            "scan_axis",
            "scan_x_key",
            "vmin",
            "vmax",
            "n_points",
            "fixed_kappa",
        }
        out: Dict[str, Any] = {k: payload[k] for k in meta_keys if k in payload}
        for key, values in payload.items():
            if key not in meta_keys:
                out[key] = np.asarray(values, dtype=float)
        return out

    if version != 1:
        raise RuntimeError(f"Unsupported scan file version: {version}")

    out = {k: v for k, v in payload.items() if k != "scan"}
    nested = payload.get("scan", {})
    for key, values in nested.items():
        out[key] = np.asarray(values, dtype=float)
    if "enhancement" in payload:
        for key, values in payload["enhancement"].items():
            if key.startswith("enhancement_"):
                out[key.replace("enhancement_", "sigma_heft_over_sm_")] = np.asarray(values, dtype=float)
    return out


def _arrays_to_lists(data: Mapping[str, np.ndarray]) -> Dict[str, list]:
    return {key: np.asarray(values, dtype=float).tolist() for key, values in data.items()}


def _slim_scan_payload(
    data: ArrayDict,
    scan_x_key: str,
) -> Dict[str, np.ndarray]:
    slim: Dict[str, np.ndarray] = {scan_x_key: data[scan_x_key]}
    for key in SCAN_VALUE_KEYS:
        slim[key] = data[key]
    return slim


def _write_scan_file(
    path: Path,
    *,
    analysis: VHHAnalysis,
    scan_axis: str,
    scan_x_key: str,
    vmin: float,
    vmax: float,
    n_points: int,
    fixed_kappa: Tuple[float, ...],
    data: ArrayDict,
) -> Path:
    slim = _slim_scan_payload(data, scan_x_key)
    payload: Dict[str, Any] = {
        "version": 2,
        "process": analysis.process,
        "energy_tev": float(analysis.energy_tev),
        "scan_axis": scan_axis,
        "scan_x_key": scan_x_key,
        "vmin": float(vmin),
        "vmax": float(vmax),
        "n_points": int(n_points),
        "fixed_kappa": [float(k) for k in fixed_kappa],
        **_arrays_to_lists(slim),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path
