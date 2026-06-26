"""
VHH HEFT closed-form predictions at NNLO.

Predict σ_LO, σ_NNLO, and K from bundled ``data/`` coefficients.
"""

from .analysis import (
    VHHAnalysis,
    data_root,
    load_analysis,
    plots_dir,
    process_data_dir,
    process_simulation_dir,
    results_dir,
    tables_dir,
)
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
    sm_enhancement,
    sm_kappa,
)
from .tables import (
    CHANNELS,
    TABLE_ENERGIES_TEV,
    WILSON_INTERVALS,
    all_channels_latex,
    build_channel_tables,
    build_channel_table_groups,
    latex_wilson_table,
    latex_wilson_tables_for_process,
)

__all__ = [
    "VHHAnalysis",
    "Prediction",
    "data_root",
    "results_dir",
    "plots_dir",
    "tables_dir",
    "process_data_dir",
    "process_simulation_dir",
    "load_analysis",
    "predict",
    "sigma",
    "sigma_uncertainties",
    "sm_kappa",
    "sm_enhancement",
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
    "WILSON_INTERVALS",
    "TABLE_ENERGIES_TEV",
    "CHANNELS",
    "build_channel_tables",
    "build_channel_table_groups",
    "latex_wilson_table",
    "latex_wilson_tables_for_process",
    "all_channels_latex",
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
    if name in (
        "default_results_root",
        "load_simulation_central",
        "collect_simulation_scan_points",
        "simulation_scan_arrays",
    ):
        from .simulation import (
            collect_simulation_scan_points,
            default_results_root,
            load_simulation_central,
            simulation_scan_arrays,
        )

        return {
            "default_results_root": default_results_root,
            "load_simulation_central": load_simulation_central,
            "collect_simulation_scan_points": collect_simulation_scan_points,
            "simulation_scan_arrays": simulation_scan_arrays,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__version__ = "0.1.0"
