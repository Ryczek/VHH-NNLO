"""Load SMEFT B coefficients and uncertainties from bundled JSON data."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from .analysis import PROCESSES, process_data_dir, smeft_data_root
from .scale_coefficients import ScaleDict, SEVEN_POINT_SCALES

PDF_ALPHA_S_FILENAME = "pdf_alpha_s_covariance.json"
SCALE_COEFFICIENTS_FILENAME = "scale_coefficients.json"
SIGMA_SM_FILENAME = "sigma_sm.json"


@dataclass(frozen=True)
class OrderBCovariance:
    labels: Tuple[str, ...]
    wc_keys: Tuple[str, ...]
    B_central: np.ndarray
    delta_pdf: np.ndarray
    delta_alpha_s: np.ndarray
    delta_pdf_alpha_s: np.ndarray
    C_pdf: np.ndarray
    C_alphaS: np.ndarray
    C_pdf_alphaS: np.ndarray


@dataclass
class SMEFTAnalysis:
    process: str
    energy_tev: float
    data_dir: Path
    sigma_sm_lo: float
    sigma_sm_nnlo: float
    B_LO: np.ndarray
    B_NNLO: np.ndarray
    wc_keys_lo: Tuple[str, ...]
    wc_keys_nnlo: Tuple[str, ...]
    C_LO_pdf: np.ndarray
    C_LO_alphaS: np.ndarray
    C_LO_pdfas: np.ndarray
    C_NNLO_pdf: np.ndarray
    C_NNLO_alphaS: np.ndarray
    C_NNLO_pdfas: np.ndarray
    B_LO_by_scale: Optional[ScaleDict] = None
    B_NNLO_by_scale: Optional[ScaleDict] = None
    sigma_sm_lo_by_scale: Optional[Dict[Tuple[float, float], float]] = None
    sigma_sm_nnlo_by_scale: Optional[Dict[Tuple[float, float], float]] = None

    @property
    def has_scale_envelope(self) -> bool:
        return bool(self.B_LO_by_scale and self.B_NNLO_by_scale)

    @property
    def is_zhh(self) -> bool:
        return self.process == "ZHH"

    @property
    def nnlo_label(self) -> str:
        return "HHZ" if self.is_zhh else "NNLO"


def _order_from_dict(raw: dict) -> OrderBCovariance:
    labels = tuple(raw["labels"])
    n = len(labels)
    delta_pdf = np.asarray(raw["delta_pdf"], dtype=float)
    delta_alpha_s = np.asarray(raw["delta_alpha_s"], dtype=float)
    if "delta_pdf_alpha_s" in raw:
        delta_pdf_alpha_s = np.asarray(raw["delta_pdf_alpha_s"], dtype=float)
    else:
        delta_pdf_alpha_s = np.sqrt(delta_pdf * delta_pdf + delta_alpha_s * delta_alpha_s)
    return OrderBCovariance(
        labels=labels,
        wc_keys=tuple(raw.get("wc_keys", labels)),
        B_central=np.asarray(raw["B_central"], dtype=float),
        delta_pdf=delta_pdf,
        delta_alpha_s=delta_alpha_s,
        delta_pdf_alpha_s=delta_pdf_alpha_s,
        C_pdf=np.asarray(raw["C_pdf"], dtype=float).reshape(n, n),
        C_alphaS=np.asarray(raw["C_alphaS"], dtype=float).reshape(n, n),
        C_pdf_alphaS=np.asarray(raw["C_pdf_alphaS"], dtype=float).reshape(n, n),
    )


def _scale_key_from_str(text: str) -> Tuple[float, float]:
    fs, rs = text.split(",", 1)
    return (float(fs), float(rs))


def _load_scale_block(path: Path) -> tuple:
    if not path.is_file():
        return None, None, None, None
    payload = json.loads(path.read_text(encoding="utf-8"))
    lo_raw = payload.get("B_LO_by_scale", {})
    nn_raw = payload.get("B_NNLO_by_scale", {})
    sm_lo_raw = payload.get("sigma_sm_lo_by_scale", {})
    sm_nn_raw = payload.get("sigma_sm_nnlo_by_scale", {})
    if not lo_raw or not nn_raw:
        return None, None, None, None
    lo = {_scale_key_from_str(k): np.asarray(v, dtype=float) for k, v in lo_raw.items()}
    nn = {_scale_key_from_str(k): np.asarray(v, dtype=float) for k, v in nn_raw.items()}
    sm_lo = {_scale_key_from_str(k): float(v) for k, v in sm_lo_raw.items()}
    sm_nn = {_scale_key_from_str(k): float(v) for k, v in sm_nn_raw.items()}
    return lo, nn, sm_lo, sm_nn


def load_smeft_analysis(
    process: str,
    energy_tev: float,
    *,
    data_dir: Optional[Path] = None,
) -> SMEFTAnalysis:
    if process not in PROCESSES:
        raise KeyError(f"Unknown process: {process}")
    dir_obj = Path(data_dir) if data_dir else process_data_dir(
        process, energy_tev, root=smeft_data_root(), framework="SMEFT"
    )
    cov_path = dir_obj / PDF_ALPHA_S_FILENAME
    if not cov_path.is_file():
        raise FileNotFoundError(f"Missing {cov_path}")

    payload = json.loads(cov_path.read_text(encoding="utf-8"))
    if payload.get("version") != 1:
        raise RuntimeError(f"Unsupported {cov_path.name} version: {payload.get('version')}")

    lo = _order_from_dict(payload["LO"])
    nn = _order_from_dict(payload["NNLO"])
    sigma_sm = payload.get("sigma_sm", {})
    sigma_sm_lo = float(sigma_sm.get("LO", payload.get("sigma_sm_lo", 0.0)))
    sigma_sm_nn = float(sigma_sm.get("NNLO", payload.get("sigma_sm_nnlo", 0.0)))

    sigma_path = dir_obj / SIGMA_SM_FILENAME
    if sigma_path.is_file():
        sm_payload = json.loads(sigma_path.read_text(encoding="utf-8"))
        sigma_sm_lo = float(sm_payload.get("sigma_sm_lo", sigma_sm_lo))
        sigma_sm_nn = float(sm_payload.get("sigma_sm_nnlo", sigma_sm_nn))

    lo_by_scale, nn_by_scale, sm_lo_by_scale, sm_nn_by_scale = _load_scale_block(
        dir_obj / SCALE_COEFFICIENTS_FILENAME
    )

    return SMEFTAnalysis(
        process=process,
        energy_tev=float(energy_tev),
        data_dir=dir_obj.resolve(),
        sigma_sm_lo=sigma_sm_lo,
        sigma_sm_nnlo=sigma_sm_nn,
        B_LO=lo.B_central,
        B_NNLO=nn.B_central,
        wc_keys_lo=lo.wc_keys,
        wc_keys_nnlo=nn.wc_keys,
        C_LO_pdf=lo.C_pdf,
        C_LO_alphaS=lo.C_alphaS,
        C_LO_pdfas=lo.C_pdf_alphaS,
        C_NNLO_pdf=nn.C_pdf,
        C_NNLO_alphaS=nn.C_alphaS,
        C_NNLO_pdfas=nn.C_pdf_alphaS,
        B_LO_by_scale=lo_by_scale,
        B_NNLO_by_scale=nn_by_scale,
        sigma_sm_lo_by_scale=sm_lo_by_scale,
        sigma_sm_nnlo_by_scale=sm_nn_by_scale,
    )
