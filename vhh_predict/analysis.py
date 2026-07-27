"""Load HEFT coefficients and uncertainties from bundled JSON data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .covariance_matrices import load_covariance_matrices, pdf_alpha_s_covariance_path
from .scale_coefficients import ScaleDict, load_scale_coefficients, scale_coefficients_path

PROCESSES = ("ZHH", "WplusHH", "WminusHH")
FRAMEWORKS = ("HEFT", "SMEFT")


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


def data_root(framework: str = "HEFT") -> Path:
    fw = framework.upper()
    if fw not in FRAMEWORKS:
        raise KeyError(f"Unknown framework: {framework} (use HEFT or SMEFT)")
    return package_root() / "data" / fw


def heft_data_root() -> Path:
    return data_root("HEFT")


def smeft_data_root() -> Path:
    return data_root("SMEFT")


def results_dir() -> Path:
    return package_root() / "results"


def _framework_subdir(framework: str) -> str:
    fw = framework.upper()
    if fw not in FRAMEWORKS:
        raise KeyError(f"Unknown framework: {framework} (use HEFT or SMEFT)")
    return fw.lower()


def _results_leaf(kind: str, framework: str, process: Optional[str], energy_tev: Optional[float]) -> Path:
    root = results_dir() / kind / _framework_subdir(framework)
    if process is None and energy_tev is None:
        return root
    if process is None or energy_tev is None:
        raise ValueError("process and energy_tev must be provided together")
    return root / process / process_data_dir(process, float(energy_tev), framework=framework).name


def plots_dir(
    framework: str = "HEFT",
    process: Optional[str] = None,
    energy_tev: Optional[float] = None,
) -> Path:
    return _results_leaf("plots", framework, process, energy_tev)


def tables_dir(
    framework: str = "HEFT",
    process: Optional[str] = None,
    energy_tev: Optional[float] = None,
) -> Path:
    return _results_leaf("tables", framework, process, energy_tev)


def points_dir(
    framework: str = "HEFT",
    process: Optional[str] = None,
    energy_tev: Optional[float] = None,
) -> Path:
    return _results_leaf("points", framework, process, energy_tev)


def _ensure_results_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_path(
    framework: str,
    process: str,
    energy_tev: float,
    filename: str,
) -> Path:
    """``results/plots/{framework}/{Process}/{energy}/{filename}``."""
    return _ensure_results_dir(plots_dir(framework, process, energy_tev)) / filename


def table_path(
    framework: str,
    process: str,
    energy_tev: float,
    filename: str,
) -> Path:
    """``results/tables/{framework}/{Process}/{energy}/{filename}``."""
    return _ensure_results_dir(tables_dir(framework, process, energy_tev)) / filename


def heft_wilson_tables_path() -> Path:
    """``results/tables/heft/heft_publication_tables.tex``."""
    return _ensure_results_dir(tables_dir("HEFT")) / "heft_publication_tables.tex"


def smeft_wc_intervals_path() -> Path:
    """``results/tables/smeft/smeft_publication_tables.tex``."""
    return _ensure_results_dir(tables_dir("SMEFT")) / "smeft_publication_tables.tex"


def process_data_dir(
    process: str,
    energy_tev: float,
    root: Optional[Path] = None,
    *,
    framework: str = "HEFT",
) -> Path:
    root = root or data_root(framework)
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
    framework: str = "HEFT",
) -> Path:
    """Bundled simulation central values: ``data/{framework}/{Process}/{energy}/Simulation/``."""
    return process_data_dir(
        process, energy_tev, root=data_root_override, framework=framework
    ) / "Simulation"


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
