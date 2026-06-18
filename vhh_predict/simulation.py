"""Compare HEFT predictions to bundled MadGraph .out files under data/.../Simulation/."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .analysis import data_root, process_simulation_dir
from .core import resolve_scan_axis
from .out_parser import (
    kappa_from_filename,
    nnlo_sigma_for_process,
    parse_out_central,
)

NO_SIM = "No value to compare with"


@dataclass(frozen=True)
class SimulationCentral:
    path: Path
    sigma_lo: Optional[float]
    sigma_nnlo: Optional[float]
    sigma_hhz: Optional[float]

    @property
    def k_measured(self) -> Optional[float]:
        if self.sigma_lo and self.sigma_lo > 0:
            if self.sigma_hhz is not None:
                return self.sigma_hhz / self.sigma_lo
            if self.sigma_nnlo is not None:
                return self.sigma_nnlo / self.sigma_lo
        return None


@dataclass(frozen=True)
class SimulationScanPoint:
    kappa: Tuple[float, ...]
    x: float
    sigma_lo: Optional[float]
    sigma_nnlo: Optional[float]
    path: Path


def default_results_root() -> Path:
    """Return bundled ``data/`` root (legacy name kept for notebooks)."""
    return data_root()


def _simulation_search_dirs(
    process: str,
    energy_tev: float,
    *,
    data_root_override: Optional[Path] = None,
) -> List[Path]:
    sim_dir = process_simulation_dir(process, energy_tev, data_root_override=data_root_override)
    if not sim_dir.is_dir():
        return []
    return [sim_dir]


def _iter_simulation_out_paths(search: Path):
    for path in sorted(search.glob("*.out")):
        if "KappaLambda_Value" not in path.name or path.stat().st_size < 50:
            continue
        yield path


def _kappa_close(a: Sequence[float], b: Sequence[float], tol: float = 1e-4) -> bool:
    return len(a) == len(b) and all(abs(x - y) < tol for x, y in zip(a, b))


def _scan_axis_index(process: str, axis: str) -> int:
    idx, _ = resolve_scan_axis(process, axis)
    return idx


def find_simulation_out(
    process: str,
    energy_tev: float,
    kappa: Tuple[float, ...],
    *,
    root: Optional[Path] = None,
) -> Optional[Path]:
    target = tuple(kappa)
    for search in _simulation_search_dirs(process, energy_tev, data_root_override=root):
        for path in _iter_simulation_out_paths(search):
            try:
                k_out = kappa_from_filename(path, process=process)
            except ValueError:
                continue
            if _kappa_close(k_out, target):
                return path
    return None


def load_simulation_central(
    process: str,
    energy_tev: float,
    kappa: Tuple[float, ...],
    *,
    root: Optional[Path] = None,
) -> Optional[SimulationCentral]:
    path = find_simulation_out(process, energy_tev, kappa, root=root)
    if path is None:
        return None
    central = parse_out_central(path, process=process)
    return SimulationCentral(
        path=path,
        sigma_lo=central.sigma_lo,
        sigma_nnlo=central.sigma_nnlo,
        sigma_hhz=central.sigma_hhz,
    )


def collect_simulation_scan_points(
    process: str,
    energy_tev: float,
    axis: str,
    *,
    fixed_kappa: Tuple[float, ...],
    vmin: float,
    vmax: float,
    root: Optional[Path] = None,
) -> List[SimulationScanPoint]:
    """Collect simulation central values along one κ axis with the rest fixed."""
    idx = _scan_axis_index(process, axis)
    fixed = list(fixed_kappa)
    if process == "ZHH" and len(fixed) == 3:
        fixed.append(1.0)

    points: List[SimulationScanPoint] = []
    seen_x: set[float] = set()
    for search in _simulation_search_dirs(process, energy_tev, data_root_override=root):
        for path in _iter_simulation_out_paths(search):
            try:
                k_out = tuple(kappa_from_filename(path, process=process))
            except ValueError:
                continue
            if len(k_out) != len(fixed):
                continue
            other = [k_out[i] for i in range(len(fixed)) if i != idx]
            other_fixed = [fixed[i] for i in range(len(fixed)) if i != idx]
            if not _kappa_close(other, other_fixed):
                continue
            x = float(k_out[idx])
            if x < vmin - 1e-9 or x > vmax + 1e-9:
                continue
            key = round(x, 8)
            if key in seen_x:
                continue
            central = parse_out_central(path, process=process)
            nnlo = nnlo_sigma_for_process(central, process=process)
            if nnlo is None or not math.isfinite(nnlo):
                continue
            seen_x.add(key)
            points.append(
                SimulationScanPoint(
                    kappa=k_out,
                    x=x,
                    sigma_lo=central.sigma_lo,
                    sigma_nnlo=nnlo,
                    path=path,
                )
            )
    points.sort(key=lambda p: p.x)
    return points


def simulation_scan_arrays(
    points: Sequence[SimulationScanPoint],
) -> Dict[str, np.ndarray]:
    if not points:
        return {"x": np.array([]), "sigma_lo": np.array([]), "sigma_nnlo": np.array([])}
    return {
        "x": np.asarray([p.x for p in points], dtype=float),
        "sigma_lo": np.asarray([p.sigma_lo if p.sigma_lo is not None else np.nan for p in points], dtype=float),
        "sigma_nnlo": np.asarray([p.sigma_nnlo for p in points], dtype=float),
    }


def compare_line(heft: float, measured: Optional[float], *, label: str = "sim") -> str:
    if measured is None or not math.isfinite(measured):
        return NO_SIM
    diff = (heft - measured) / measured * 100.0
    return f"{label} {measured:.6g}  (diff {diff:+.3f}% vs HEFT)"
