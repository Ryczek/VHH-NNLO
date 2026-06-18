"""Load PDF and alpha_s covariance matrices from pdf_alpha_s_covariance.json."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

PDF_ALPHA_S_FILENAME = "pdf_alpha_s_covariance.json"


@dataclass(frozen=True)
class OrderCovariance:
    labels: Tuple[str, ...]
    A_central: np.ndarray
    delta_pdf: np.ndarray
    delta_alpha_s: np.ndarray
    delta_pdf_alpha_s: np.ndarray
    C_pdf: np.ndarray
    C_alphaS: np.ndarray
    C_pdf_alphaS: np.ndarray


@dataclass(frozen=True)
class CovarianceData:
    process: str
    energy_tev: float
    method: str
    LO: OrderCovariance
    NNLO: OrderCovariance


def pdf_alpha_s_covariance_path(data_dir: Path) -> Path:
    return data_dir / PDF_ALPHA_S_FILENAME


def _order_from_dict(raw: dict) -> OrderCovariance:
    labels = tuple(raw["labels"])
    n = len(labels)
    delta_pdf = np.asarray(raw["delta_pdf"], dtype=float)
    delta_alpha_s = np.asarray(raw["delta_alpha_s"], dtype=float)
    if "delta_pdf_alpha_s" in raw:
        delta_pdf_alpha_s = np.asarray(raw["delta_pdf_alpha_s"], dtype=float)
    else:
        delta_pdf_alpha_s = np.sqrt(delta_pdf * delta_pdf + delta_alpha_s * delta_alpha_s)
    return OrderCovariance(
        labels=labels,
        A_central=np.asarray(raw["A_central"], dtype=float),
        delta_pdf=delta_pdf,
        delta_alpha_s=delta_alpha_s,
        delta_pdf_alpha_s=delta_pdf_alpha_s,
        C_pdf=np.asarray(raw["C_pdf"], dtype=float).reshape(n, n),
        C_alphaS=np.asarray(raw["C_alphaS"], dtype=float).reshape(n, n),
        C_pdf_alphaS=np.asarray(raw["C_pdf_alphaS"], dtype=float).reshape(n, n),
    )


def load_covariance_matrices(path: Path) -> Optional[CovarianceData]:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") != 1:
        raise RuntimeError(f"Unsupported {path.name} version: {payload.get('version')}")
    return CovarianceData(
        process=str(payload["process"]),
        energy_tev=float(payload["energy_tev"]),
        method=str(payload.get("method", "")),
        LO=_order_from_dict(payload["LO"]),
        NNLO=_order_from_dict(payload["NNLO"]),
    )
