"""Load HEFT coefficients and uncertainties from bundled JSON data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .covariance_matrices import load_covariance_matrices, pdf_alpha_s_covariance_path
from .scale_coefficients import ScaleDict, load_scale_coefficients, scale_coefficients_path

PROCESSES = ("ZHH", "WplusHH", "WminusHH")


@dataclass
class VHHAnalysis:
    process: str
    energy_tev: float
    data_dir: Path
    A_LO: np.ndarray
    A_NNLO: np.ndarray
    C_LO_pdf: np.ndarray
    C_LO_alphaS: np.ndarray
    C_LO_pdfas: np.ndarray
    C_NNLO_pdf: np.ndarray
    C_NNLO_alphaS: np.ndarray
    C_NNLO_pdfas: np.ndarray
    A_LO_by_scale: Optional[ScaleDict] = None
    A_NNLO_by_scale: Optional[ScaleDict] = None

    @property
    def has_scale_envelope(self) -> bool:
        return bool(self.A_LO_by_scale and self.A_NNLO_by_scale)

    @property
    def is_zhh(self) -> bool:
        return self.process == "ZHH"

    @property
    def nnlo_label(self) -> str:
        return "NNLO"


def package_root() -> Path:
    return Path(__file__).resolve().parents[1]


def data_root() -> Path:
    return package_root() / "data"


def results_dir() -> Path:
    return package_root() / "Results"


def plots_dir() -> Path:
    return results_dir() / "Plots"


def tables_dir() -> Path:
    return results_dir() / "Tables"


def process_data_dir(process: str, energy_tev: float, root: Optional[Path] = None) -> Path:
    root = root or data_root()
    e = float(energy_tev)
    if abs(e - 13.6) < 0.05:
        tag = "13_6TeV"
    elif abs(e - 14.0) < 0.05:
        tag = "14_0TeV"
    else:
        raise KeyError(f"Unsupported energy: {energy_tev} TeV (use 13.6 or 14.0)")
    if process not in PROCESSES:
        raise KeyError(f"Unknown process: {process}")
    return root / process / tag


def process_simulation_dir(
    process: str,
    energy_tev: float,
    *,
    data_root_override: Optional[Path] = None,
) -> Path:
    """Bundled MadGraph central values: ``data/{Process}/{energy}/Simulation/``."""
    return process_data_dir(process, energy_tev, root=data_root_override) / "Simulation"


def load_analysis(
    process: str,
    energy_tev: float,
    *,
    data_dir: Optional[Path] = None,
) -> VHHAnalysis:
    dir_obj = Path(data_dir) if data_dir else process_data_dir(process, energy_tev)

    cov_data = load_covariance_matrices(pdf_alpha_s_covariance_path(dir_obj))
    if cov_data is None:
        raise FileNotFoundError(f"Missing {pdf_alpha_s_covariance_path(dir_obj)}")

    lo_by_scale, nn_by_scale = load_scale_coefficients(scale_coefficients_path(dir_obj))
    if not lo_by_scale or not nn_by_scale:
        raise FileNotFoundError(f"Missing {scale_coefficients_path(dir_obj)}")

    return VHHAnalysis(
        process=process,
        energy_tev=float(energy_tev),
        data_dir=dir_obj.resolve(),
        A_LO=cov_data.LO.A_central,
        A_NNLO=cov_data.NNLO.A_central,
        C_LO_pdf=cov_data.LO.C_pdf,
        C_LO_alphaS=cov_data.LO.C_alphaS,
        C_LO_pdfas=cov_data.LO.C_pdf_alphaS,
        C_NNLO_pdf=cov_data.NNLO.C_pdf,
        C_NNLO_alphaS=cov_data.NNLO.C_alphaS,
        C_NNLO_pdfas=cov_data.NNLO.C_pdf_alphaS,
        A_LO_by_scale=lo_by_scale,
        A_NNLO_by_scale=nn_by_scale,
    )
