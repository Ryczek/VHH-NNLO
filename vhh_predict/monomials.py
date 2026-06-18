"""HEFT monomial vectors for W±HH (6 terms) and ZHH (6 LO / 18 NNLO terms)."""

from __future__ import annotations

import numpy as np

N_COEFF_W = 6
N_COEFF_Z_LO = 6
N_COEFF_Z_NNLO = 18


def monomial_vector_w(kl: float, kv: float, k2v: float) -> np.ndarray:
    return np.array(
        [
            kv**4,
            kv**3 * kl,
            kv**2 * k2v,
            kv**2 * kl**2,
            kv * kl * k2v,
            k2v**2,
        ],
        dtype=float,
    )


def monomial_vector_z_nnlo(kl: float, kv: float, k2v: float, kt: float) -> np.ndarray:
    """18 monomials — equation (39), σ_HHZ / NNLO total."""
    return np.array(
        [
            kv**4,
            kv**3 * kl,
            kv**2 * k2v,
            kv**2 * kl**2,
            kv * kl * k2v,
            k2v**2,
            k2v * kt * kl,
            k2v * kv * kt,
            k2v * kt**2,
            kv**3 * kt,
            kv**2 * kt**2,
            kv**2 * kl * kt,
            kv * kl**2 * kt,
            kv * kl * kt**2,
            kt**2 * kl**2,
            kl * kt**3,
            kv * kt**3,
            kt**4,
        ],
        dtype=float,
    )


def monomial_vector_z_lo(kl: float, kv: float, k2v: float, kt: float = 0.0) -> np.ndarray:
    """LO ZHH: first six terms only (no k_t dependence)."""
    return monomial_vector_z_nnlo(kl, kv, k2v, kt)[:N_COEFF_Z_LO]


def monomial_vector(process: str, kappa: tuple[float, ...], order: str) -> np.ndarray:
    order_u = order.upper()
    if process == "ZHH":
        kl, kv, k2v, kt = kappa
        if order_u == "LO":
            return monomial_vector_z_lo(kl, kv, k2v, kt)
        return monomial_vector_z_nnlo(kl, kv, k2v, kt)
    if process in ("WplusHH", "WminusHH"):
        kl, kv, k2v = kappa[:3]
        return monomial_vector_w(kl, kv, k2v)
    raise ValueError(f"Unknown process: {process}")
