"""Minimal parser for central .out cross sections."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

ZHH_SIG_HHZ_MAX_FB = 5.0


@dataclass(frozen=True)
class OutCentral:
    sigma_lo: Optional[float]
    sigma_nnlo: Optional[float]
    sigma_hhz: Optional[float]
    sigma_lo_stat: Optional[float] = None
    sigma_nnlo_stat: Optional[float] = None
    sigma_hhz_stat: Optional[float] = None


def _parse_float(text: str) -> float:
    return float(text.replace("D", "E"))


def parse_kappa_token(token: str) -> float:
    token = token.strip()
    sign = -1.0 if token.startswith("-") else 1.0
    body = token[1:] if token.startswith("-") else token
    if "_" not in body:
        return sign * float(body)
    left, right = body.split("_", 1)
    return sign * float(f"{left}.{right}")


def kappa_from_filename(path: Path, *, process: str) -> Tuple[float, ...]:
    name = path.name
    for ext in (".out", ".ou", ".o"):
        if name.endswith(ext):
            name = name[: -len(ext)]
            break
    else:
        name = name.rstrip(".")

    if process == "ZHH":
        pat = re.compile(
            r"KappaLambda_Value_(?P<kl>[-0-9_]+)_"
            r"KappaV_Value_(?P<kv>[-0-9_]+)_"
            r"Kappa2V_Value_(?P<k2v>[-0-9_]+)"
            r"(?:_(?:Kappa_t|Kappat)_Value_(?P<kt>[-0-9_]+))?$"
        )
        m = pat.search(name)
        if not m:
            raise ValueError(f"Cannot parse ZHH kappas from {path.name}")
        kt = parse_kappa_token(m.group("kt")) if m.group("kt") else 1.0
        return (
            parse_kappa_token(m.group("kl")),
            parse_kappa_token(m.group("kv")),
            parse_kappa_token(m.group("k2v")),
            kt,
        )

    pat = re.compile(
        r"KappaLambda_Value_(?P<kl>[-0-9_]+)_"
        r"KappaV_Value_(?P<kv>[-0-9_]+)_"
        r"Kappa2V_Value_(?P<k2v>[-0-9_]+)"
    )
    m = pat.search(name)
    if not m:
        raise ValueError(f"Cannot parse kappas from {path.name}")
    return (
        parse_kappa_token(m.group("kl")),
        parse_kappa_token(m.group("kv")),
        parse_kappa_token(m.group("k2v")),
    )


def _is_valid_hhz(value: Optional[float]) -> bool:
    if value is None or not math.isfinite(value):
        return False
    return 0.0 < value <= ZHH_SIG_HHZ_MAX_FB


def parse_out_central(path: Path, *, process: str) -> OutCentral:
    """Read central σ at NSET=0, fact_scale=ren_scale=1.0."""
    sig_re = re.compile(
        r"sig_(LO|NNLO|HHZ)\s*=\s*\(\s*([0-9EeDd+\-.]+)\s*\+\-\s*([0-9EeDd+\-.]+)\s*\)"
    )
    nset: Optional[int] = None
    fact_scale: Optional[float] = None
    ren_scale: Optional[float] = None
    sig_lo = sig_lo_stat = None
    sig_nnlo = sig_nnlo_stat = None
    sig_hhz = sig_hhz_stat = None

    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue
            if m := re.search(r"NSET\s*=\s*(-?\d+)", s):
                nset = int(m.group(1))
            elif m := re.search(r"fact_scale\s+([0-9EeDd+\-.]+)", s):
                fact_scale = _parse_float(m.group(1))
            elif m := re.search(r"ren_scale\s+([0-9EeDd+\-.]+)", s):
                ren_scale = _parse_float(m.group(1))
            elif m := sig_re.search(s):
                val = _parse_float(m.group(2))
                err = _parse_float(m.group(3))
                if m.group(1) == "LO":
                    sig_lo, sig_lo_stat = val, err
                elif m.group(1) == "NNLO":
                    sig_nnlo, sig_nnlo_stat = val, err
                else:
                    sig_hhz, sig_hhz_stat = val, err

            if nset == 0 and fact_scale == 1.0 and ren_scale == 1.0:
                if process == "ZHH" and sig_hhz is not None:
                    break
                if process != "ZHH" and sig_nnlo is not None:
                    break

    if process == "ZHH" and not _is_valid_hhz(sig_hhz):
        sig_hhz = None
        sig_hhz_stat = None

    return OutCentral(
        sigma_lo=sig_lo,
        sigma_nnlo=sig_nnlo,
        sigma_hhz=sig_hhz,
        sigma_lo_stat=sig_lo_stat,
        sigma_nnlo_stat=sig_nnlo_stat,
        sigma_hhz_stat=sig_hhz_stat,
    )


def nnlo_sigma_for_process(central: OutCentral, *, process: str) -> Optional[float]:
    if process == "ZHH":
        return central.sigma_hhz
    return central.sigma_nnlo


def _replica_sigmas_at_central_scale(path: Path) -> Dict[int, Dict[str, float]]:
    """Collect ``sig_*`` at ``fact_scale=ren_scale=1`` keyed by NSET."""
    sig_re = re.compile(
        r"sig_(LO|NNLO|HHZ)\s*=\s*\(\s*([0-9EeDd+\-.]+)\s*\+\-\s*([0-9EeDd+\-.]+)\s*\)"
    )
    nset: Optional[int] = None
    fact_scale: Optional[float] = None
    ren_scale: Optional[float] = None
    cur: Dict[str, float] = {}
    out: Dict[int, Dict[str, float]] = {}

    def _flush() -> None:
        nonlocal cur
        if nset is not None and fact_scale == 1.0 and ren_scale == 1.0 and cur:
            out[nset] = dict(cur)
        cur = {}

    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue
            if m := re.search(r"NSET\s*=\s*(-?\d+)", s):
                _flush()
                nset = int(m.group(1))
                fact_scale = ren_scale = None
            elif m := re.search(r"fact_scale\s+([0-9EeDd+\-.]+)", s):
                fact_scale = _parse_float(m.group(1))
            elif m := re.search(r"ren_scale\s+([0-9EeDd+\-.]+)", s):
                ren_scale = _parse_float(m.group(1))
            elif m := sig_re.search(s):
                cur[m.group(1)] = _parse_float(m.group(2))
        _flush()
    return out


def sm_pdf_uncertainties_from_out(
    path: Path,
    *,
    process: str,
    sigma_sm_lo: float,
    sigma_sm_nnlo: float,
) -> Dict[str, float]:
    """PDF / α_s / PDF+α_s on σ_SM from replica NSET=1..40 and α_s NSET=41/42.

    Central values are taken from ``sigma_sm_*`` (NSET=0 at (1,1) is sometimes
    absent from bundled SM files that only store a subset of scale points).
    """
    by_nset = _replica_sigmas_at_central_scale(path)
    nn_key = "HHZ" if process == "ZHH" else "NNLO"

    def _one(order_key: str, central: float) -> Tuple[float, float, float]:
        reps = [by_nset[n][order_key] for n in range(1, 41) if n in by_nset and order_key in by_nset[n]]
        if len(reps) < 10 or not math.isfinite(central):
            return float("nan"), float("nan"), float("nan")
        pdf = math.sqrt(sum((r - central) ** 2 for r in reps))
        a41 = by_nset.get(41, {}).get(order_key)
        a42 = by_nset.get(42, {}).get(order_key)
        if a41 is None or a42 is None:
            alpha = 0.0
        else:
            alpha = 0.5 * (a42 - a41)
        pdfas = math.sqrt(pdf * pdf + alpha * alpha)
        return pdf, alpha, pdfas

    lo_pdf, lo_a, lo_pdfas = _one("LO", sigma_sm_lo)
    nn_pdf, nn_a, nn_pdfas = _one(nn_key, sigma_sm_nnlo)
    return {
        "sigma_sm_lo_pdf": lo_pdf,
        "sigma_sm_lo_alpha_s": lo_a,
        "sigma_sm_lo_pdfas": lo_pdfas,
        "sigma_sm_nnlo_pdf": nn_pdf,
        "sigma_sm_nnlo_alpha_s": nn_a,
        "sigma_sm_nnlo_pdfas": nn_pdfas,
    }


def nnlo_sigma_stat_for_process(central: OutCentral, *, process: str) -> Optional[float]:
    if process == "ZHH":
        return central.sigma_hhz_stat
    return central.sigma_nnlo_stat
