"""κ-scan: evaluate and save point tables under ``Results/Points/``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

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
    """``Results/Points/{Process}_{energy}TeV_{axis}.txt``."""
    name = f"{process}_{float(energy_tev):g}TeV_{scan_x_key}.txt"
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

    Returns ``(scan_data, path)``. The saved ``.txt`` is a comment header plus a
    tab-separated table (scanned κ and five physics columns). Uncertainty bands
    are not written to disk.

    Set ``uncertainties=True`` if you need PDF/scale bands for plotting (in
    memory only).
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
    """Load a scan ``.txt`` (or legacy ``.json``); array values are numpy ndarrays."""
    path = Path(path)
    if path.suffix.lower() == ".json":
        return _load_scan_json(path)
    return _load_scan_txt(path)


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
    columns = [scan_x_key, *SCAN_VALUE_KEYS]
    rows = np.column_stack([slim[col] for col in columns])

    header_lines = [
        "# VHH-NNLO scan points",
        f"# process: {analysis.process}",
        f"# energy_tev: {float(analysis.energy_tev):g}",
        f"# scan_axis: {scan_axis}",
        f"# scan_x_key: {scan_x_key}",
        f"# vmin: {float(vmin):g}",
        f"# vmax: {float(vmax):g}",
        f"# n_points: {int(n_points)}",
        f"# fixed_kappa: {','.join(f'{float(k):g}' for k in fixed_kappa)}",
        "#",
        "\t".join(columns),
    ]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body_lines = ["\t".join(f"{float(x):.10g}" for x in row) for row in rows]
    path.write_text("\n".join(header_lines) + "\n" + "\n".join(body_lines) + "\n", encoding="utf-8")
    return path


def _load_scan_txt(path: Path) -> Dict[str, Any]:
    comments: Dict[str, str] = {}
    table_lines: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            payload = line[1:].strip()
            if ":" in payload:
                key, _, value = payload.partition(":")
                comments[key.strip()] = value.strip()
            continue
        table_lines.append(line)

    if len(table_lines) < 2:
        raise ValueError(f"No data table in {path}")

    header = table_lines[0].split("\t")
    if len(header) == 1:
        header = table_lines[0].split()
    rows = np.asarray(
        [[float(x) for x in (ln.split("\t") if "\t" in ln else ln.split())] for ln in table_lines[1:]],
        dtype=float,
    )

    out: Dict[str, Any] = {}
    for key in (
        "process",
        "scan_axis",
        "scan_x_key",
    ):
        if key in comments:
            out[key] = comments[key]
    for key in ("energy_tev", "vmin", "vmax"):
        if key in comments:
            out[key] = float(comments[key])
    if "n_points" in comments:
        out["n_points"] = int(comments["n_points"])
    if "fixed_kappa" in comments:
        out["fixed_kappa"] = [float(x) for x in comments["fixed_kappa"].split(",")]

    for j, col in enumerate(header):
        out[col] = rows[:, j]
    return out


def _load_scan_json(path: Path) -> Dict[str, Any]:
    """Legacy JSON scan files (v1 / v2)."""
    payload = json.loads(path.read_text(encoding="utf-8"))
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
