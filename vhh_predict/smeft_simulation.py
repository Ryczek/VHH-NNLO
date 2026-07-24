"""Compare SMEFT predictions to bundled simulation ``.out`` files."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .analysis import process_simulation_dir, smeft_data_root
from .out_parser import nnlo_sigma_for_process, parse_out_central
from .smeft_operators import Z_OPERATORS, W_OPERATORS, normalize_wc_dict, sm_wc_values

# Value token is digits/underscores/signs only; optional trailing `_Corrected` is ignored.
WC_VALUE_RE = re.compile(
    r"(?i)(cH\w*|cth|chust|chdst|churd)_Value_([-0-9_]+?)(?:_Corrected)?(?=\.|$)"
)


@dataclass(frozen=True)
class SMEFTSimulationCentral:
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


def _norm_wc_token(token: str) -> float:
    return float(token.replace("_", "."))


def _fortran_to_key(fortran: str) -> Optional[str]:
    fl = fortran.lower()
    for op in list(W_OPERATORS) + list(Z_OPERATORS.values()):
        if op.fortran.lower() == fl:
            return op.key
    return None


def wc_values_from_filename(path: Path) -> Dict[str, float]:
    """Parse nonzero Wilson coefficients encoded in a SMEFT .out filename."""
    out: Dict[str, float] = {}
    for m in WC_VALUE_RE.finditer(path.name):
        key = _fortran_to_key(m.group(1))
        if key is None:
            continue
        out[key] = _norm_wc_token(m.group(2))
    return out


def _wc_close(a: Dict[str, float], b: Dict[str, float], tol: float = 1e-5) -> bool:
    keys = set(a) | set(b)
    return all(abs(a.get(k, 0.0) - b.get(k, 0.0)) < tol for k in keys)


def _simulation_search_dirs(
    process: str,
    energy_tev: float,
    *,
    data_root_override: Optional[Path] = None,
) -> List[Path]:
    sim_dir = process_simulation_dir(
        process,
        energy_tev,
        data_root_override=data_root_override or smeft_data_root(),
        framework="SMEFT",
    )
    if not sim_dir.is_dir():
        return []
    return [sim_dir]


def find_smeft_simulation_out(
    process: str,
    energy_tev: float,
    wcs: Dict[str, float],
    *,
    root: Optional[Path] = None,
) -> Optional[Path]:
    target = normalize_wc_dict(process, wcs)
    for search in _simulation_search_dirs(process, energy_tev, data_root_override=root):
        for path in sorted(search.glob("*.out")):
            if path.stat().st_size < 50:
                continue
            try:
                parsed = wc_values_from_filename(path)
                full = sm_wc_values(process)
                full.update(parsed)
                if _wc_close(full, target):
                    return path
            except ValueError:
                continue
    return None


def load_smeft_simulation_central(
    process: str,
    energy_tev: float,
    wcs: Dict[str, float],
    *,
    root: Optional[Path] = None,
) -> Optional[SMEFTSimulationCentral]:
    path = find_smeft_simulation_out(process, energy_tev, wcs, root=root)
    if path is None:
        return None
    central = parse_out_central(path, process=process)
    return SMEFTSimulationCentral(
        path=path,
        sigma_lo=central.sigma_lo,
        sigma_nnlo=central.sigma_nnlo,
        sigma_hhz=central.sigma_hhz,
    )


def compare_suffix(pred: float, measured: Optional[float], *, label: str = "simulation") -> str:
    if measured is None or not math.isfinite(measured):
        return ""
    diff = (pred - measured) / measured * 100.0
    return f"    {label} {measured:.6g}  (diff {diff:+.3f}%)"
