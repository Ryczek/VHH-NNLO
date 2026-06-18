"""
VHH HEFT closed-form predictions at NNLO.

Predict σ_LO, σ_NNLO, and K from bundled ``data/`` coefficients.
"""

from .analysis import VHHAnalysis, data_root, load_analysis, plots_dir, process_data_dir
from .core import (
    Prediction,
    SCAN_AXES_W,
    SCAN_AXES_Z,
    format_prediction,
    predict,
    resolve_scan_axis,
    scan,
    scan_axes,
    scan_sm_enhancement,
    sigma,
    sigma_uncertainties,
    sm_kappa,
)
from .simulation import (
    collect_simulation_scan_points,
    default_results_root,
    load_simulation_central,
    simulation_scan_arrays,
)

__all__ = [
    "VHHAnalysis",
    "Prediction",
    "data_root",
    "plots_dir",
    "process_data_dir",
    "load_analysis",
    "predict",
    "sigma",
    "sigma_uncertainties",
    "sm_kappa",
    "format_prediction",
    "scan",
    "scan_axes",
    "resolve_scan_axis",
    "SCAN_AXES_W",
    "SCAN_AXES_Z",
    "scan_sm_enhancement",
    "default_results_root",
    "load_simulation_central",
    "collect_simulation_scan_points",
    "simulation_scan_arrays",
    "plot_sigma_nnlo_and_kfactor",
    "plot_kfactor_only",
    "plot_sm_enhancement",
]


def __getattr__(name: str):
    if name in ("plot_sigma_nnlo_and_kfactor", "plot_kfactor_only", "plot_sm_enhancement"):
        from .plots import plot_kfactor_only, plot_sigma_nnlo_and_kfactor, plot_sm_enhancement

        return {
            "plot_sigma_nnlo_and_kfactor": plot_sigma_nnlo_and_kfactor,
            "plot_kfactor_only": plot_kfactor_only,
            "plot_sm_enhancement": plot_sm_enhancement,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__version__ = "0.1.0"
