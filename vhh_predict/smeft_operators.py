"""SMEFT Wilson-coefficient naming and allowed intervals for the release package."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

# Global-fit intervals from J. ter Hoeve, L. Mantani, A. N. Rossia, J. Rojo,
# E. Vryonidou, JHEP 06 (2025) 125 [arXiv:2502.20453]
# ("Connecting scales: RGE effects in the SMEFT at the LHC and future colliders"),
# linear EFT fit with RGE, Table E.1 — units TeV^{-2}.
SMEFT_WC_INTERVALS: Dict[str, Tuple[float, float]] = {
    # Bosonic
    "phi": (-15.0, 5.0),
    "phiW": (-1.0, 1.0),
    "phiB": (-0.5, 0.5),
    "phiWB": (-1.5, 1.5),
    "phiD": (-2.0, 2.0),
    "phiBox": (-1.5, 1.5),
    # Fermionic (Table E.1, linear EFT w/ RGE)
    "phiq3st": (-0.2, 0.05),
    "phiq1st": (-3.0, 1.0),
    "phiu": (-3.5, 1.0),
    "phid": (-4.0, 4.0),
    "tphi": (-15.0, 5.0),  # C_tφ (Fortran cth); distinct from C_φt
    "phit": (-24.4, 33.9),  # C_φt (enters B₁₂; not scanned individually in this release)
    "phiQ3": (-7.7, 2.0),  # C_φQ^(3)
    "phiQ1rd": (-6.5, 30.5),  # C_φQ^(1)
    # B₁₂ scan axis only: C_φt + C_φQ^(3) − C_φQ^(1) (Fortran cHq3rd)
    "phiQ3rd": (-8.0, 2.0),
}

SMEFT_WC_LATEX = {
    "phi": r"C_\varphi",
    "phiBox": r"C_{\varphi\square}",
    "phiD": r"C_{\varphi D}",
    "phiq3st": r"C_{\varphi q}^{(3)}",
    "phiW": r"C_{\varphi W}",
    "phiq1st": r"C_{\varphi q}^{(1)}",
    "phiu": r"C_{\varphi u}",
    "phid": r"C_{\varphi d}",
    "phiB": r"C_{\varphi B}",
    "phiWB": r"C_{\varphi WB}",
    "tphi": r"C_{t\varphi}",
    "phit": r"C_{\varphi t}",
    "phiQ3": r"C_{\varphi Q}^{(3)}",
    "phiQ3rd": r"C_{\varphi t}+C_{\varphi Q}^{(3)}-C_{\varphi Q}^{(1)}",
    "phiQ1rd": r"C_{\varphi Q}^{(1)}",
}

SMEFT_WC_PLAIN = {
    "phi": "C_φ",
    "phiBox": "C_φ□",
    "phiD": "C_φD",
    "phiq3st": "C_φq^(3)",
    "phiW": "C_φW",
    "phiq1st": "C_φq^(1)",
    "phiu": "C_φu",
    "phid": "C_φd",
    "phiB": "C_φB",
    "phiWB": "C_φWB",
    "tphi": "C_tφ",
    "phit": "C_φt",
    "phiQ3": "C_φQ^(3)",
    "phiQ1rd": "C_φQ^(1)",
    "phiQ3rd": "C_φt+C_φQ^(3)−C_φQ^(1)",
}

W_SCAN_WC_KEYS: Tuple[str, ...] = ("phi", "phiBox", "phiD", "phiq3st", "phiW")

Z_LO_WC_KEYS: Tuple[str, ...] = (
    "phi",
    "phiBox",
    "phiD",
    "phiq3st",
    "phiW",
    "phiq1st",
    "phiu",
    "phid",
    "phiB",
    "phiWB",
)

Z_NNLO_EXTRA_WC_KEYS: Tuple[str, ...] = ("tphi", "phiQ3rd")


@dataclass(frozen=True)
class WCOperator:
    key: str
    fortran: str


W_OPERATORS: Tuple[WCOperator, ...] = tuple(
    WCOperator(k, f)
    for k, f in (
        ("phi", "cH"),
        ("phiBox", "cHsq"),
        ("phiD", "cHdup"),
        ("phiq3st", "cHq3st"),
        ("phiW", "cHw"),
    )
)

Z_OPERATORS: Dict[str, WCOperator] = {
    **{op.key: op for op in W_OPERATORS},
    "phiq1st": WCOperator("phiq1st", "cHq1st"),
    "phiu": WCOperator("phiu", "chust"),
    "phid": WCOperator("phid", "chdst"),
    "phiB": WCOperator("phiB", "cHb"),
    "phiWB": WCOperator("phiWB", "cHwb"),
    "tphi": WCOperator("tphi", "cth"),
    "phiQ3rd": WCOperator("phiQ3rd", "cHq3rd"),
}


def wc_keys_for_process(process: str, order: str = "NNLO") -> Tuple[str, ...]:
    if process in ("WplusHH", "WminusHH"):
        return W_SCAN_WC_KEYS
    if process == "ZHH":
        if order.upper() in ("LO",):
            return Z_LO_WC_KEYS
        return Z_LO_WC_KEYS + Z_NNLO_EXTRA_WC_KEYS
    raise KeyError(f"Unknown process: {process}")


def scan_axes(process: str) -> Tuple[str, ...]:
    return wc_keys_for_process(process, "LO")


def sm_wc_values(process: str) -> Dict[str, float]:
    return {k: 0.0 for k in wc_keys_for_process(process, "NNLO")}


def normalize_wc_dict(process: str, wcs: Dict[str, float]) -> Dict[str, float]:
    allowed = set(wc_keys_for_process(process, "NNLO"))
    out = sm_wc_values(process)
    for key, val in wcs.items():
        if key not in allowed:
            raise ValueError(f"Unknown WC {key!r} for {process}; allowed: {sorted(allowed)}")
        out[key] = float(val)
    return out
