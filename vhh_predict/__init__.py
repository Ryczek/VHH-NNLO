"""
VHH HEFT and SMEFT closed-form predictions at NNLO.

HEFT: predict σ from bundled ``data/HEFT/`` A coefficients.
SMEFT: predict σ from bundled ``data/SMEFT/`` B coefficients (see ``smeft_*`` modules).
"""

from .analysis import (
    VHHAnalysis,
    data_root,
    heft_data_root,
    heft_wilson_tables_path,
    smeft_data_root,
    smeft_wc_intervals_path,
    load_analysis,
    plot_path,
    plots_dir,
    points_dir,
    process_data_dir,
    process_simulation_dir,
    results_dir,
    table_path,
    tables_dir,
)
from .core import (
    Prediction,
    SCAN_AXES_W,
    SCAN_AXES_Z,
    format_prediction,
    spot_check_caption,
    spot_check_table,
    predict,
    resolve_scan_axis,
    scan,
    scan_axes,
    scan_grid,
    scan_sm_enhancement,
    sigma,
    sigma_uncertainties,
    sm_enhancement,
    sm_kappa,
)
from .scan_io import (
    load_scan_results,
    scan_and_save,
    scan_axes_and_save,
    scan_grid_and_save,
    scan_grid_points_path,
    scan_points_path,
)
from .plot_style import (
    DEFAULT_PLOT_STYLE,
    PlotStyle,
    default_plot_title,
    plot_style_with_layout,
    scan_plot_filename_stem,
    scan_plot_path,
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
    "heft_data_root",
    "smeft_data_root",
    "results_dir",
    "plots_dir",
    "points_dir",
    "tables_dir",
    "plot_path",
    "table_path",
    "heft_wilson_tables_path",
    "smeft_wc_intervals_path",
    "process_data_dir",
    "process_simulation_dir",
    "load_analysis",
    "predict",
    "sigma",
    "sigma_uncertainties",
    "sm_kappa",
    "sm_enhancement",
    "format_prediction",
    "spot_check_caption",
    "spot_check_table",
    "scan",
    "scan_axes",
    "scan_grid",
    "resolve_scan_axis",
    "SCAN_AXES_W",
    "SCAN_AXES_Z",
    "scan_sm_enhancement",
    "scan_and_save",
    "scan_axes_and_save",
    "scan_grid_and_save",
    "load_scan_results",
    "scan_points_path",
    "scan_grid_points_path",
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
    "plot_sigma_nnlo_and_enhancement_nnlo",
    "plot_sigma_only",
    "plot_sigma_lo_only",
    "plot_kfactor_only",
    "plot_enhancement_only",
    "plot_sm_enhancement",
    "PlotStyle",
    "DEFAULT_PLOT_STYLE",
    "default_plot_title",
    "plot_style_with_layout",
    "scan_plot_filename_stem",
    "scan_plot_path",
]


def __getattr__(name: str):
    if name in (
        "plot_sigma_nnlo_and_kfactor",
        "plot_sigma_nnlo_and_enhancement_nnlo",
        "plot_sigma_only",
        "plot_sigma_lo_only",
        "plot_kfactor_only",
        "plot_enhancement_only",
        "plot_sm_enhancement",
    ):
        from .plots import (
            plot_enhancement_only,
            plot_kfactor_only,
            plot_sigma_lo_only,
            plot_sigma_nnlo_and_enhancement_nnlo,
            plot_sigma_nnlo_and_kfactor,
            plot_sigma_only,
            plot_sm_enhancement,
        )

        return {
            "plot_sigma_nnlo_and_kfactor": plot_sigma_nnlo_and_kfactor,
            "plot_sigma_nnlo_and_enhancement_nnlo": plot_sigma_nnlo_and_enhancement_nnlo,
            "plot_sigma_only": plot_sigma_only,
            "plot_sigma_lo_only": plot_sigma_lo_only,
            "plot_kfactor_only": plot_kfactor_only,
            "plot_enhancement_only": plot_enhancement_only,
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
