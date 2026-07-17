"""Bootstrap helpers for the prediction notebooks."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict


def _ensure_repo(root: Path, data_subdir: str, notebook: str) -> None:
    if (root / "data" / data_subdir).is_dir() and (root / "vhh_predict").is_dir():
        return
    raise RuntimeError(
        f"Expected 'data/{data_subdir}/' and 'vhh_predict/' under {root}, but didn't find them.\n"
        f"Start Jupyter from the VHH-NNLO repository root, e.g.:\n"
        f"    cd VHH-NNLO && jupyter notebook {notebook}"
    )


def _purge_vhh_predict() -> None:
    for name in list(sys.modules):
        if name == "vhh_predict" or name.startswith("vhh_predict."):
            del sys.modules[name]


def _prepare_repo(data_subdir: str, notebook: str, root: Path | None = None) -> Path:
    repo = (root or Path(".").resolve()).resolve()
    _ensure_repo(repo, data_subdir, notebook)
    repo_str = str(repo)
    if repo_str in sys.path:
        sys.path.remove(repo_str)
    sys.path.insert(0, repo_str)
    _purge_vhh_predict()
    return repo


def setup_heft(root: Path | None = None) -> Dict[str, Any]:
    """Validate repo, reload package, import HEFT notebook symbols."""
    import matplotlib.pyplot as plt
    from IPython.display import Markdown, display

    repo = _prepare_repo("HEFT", "vhh_prediction_HEFT.ipynb", root)
    from vhh_predict import (
        CHANNELS,
        TABLE_ENERGIES_TEV,
        WILSON_INTERVALS,
        build_channel_tables,
        default_plot_title,
        latex_wilson_tables_for_process,
        load_analysis,
        plot_sigma_nnlo_and_enhancement_nnlo,
        plot_sigma_nnlo_and_kfactor,
        plot_style_with_layout,
        plots_dir,
        resolve_scan_axis,
        scan_and_save,
        scan_axes,
        scan_axes_and_save,
        scan_grid_and_save,
        scan_plot_filename_stem,
        spot_check_caption,
        spot_check_table,
        tables_dir,
    )
    from vhh_predict.tables import KAPPA_PLAIN, ZHH_TABLE_GROUPS

    plots = plots_dir("HEFT")
    tables = tables_dir("HEFT")
    plots.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    return {
        "REPO_ROOT": repo,
        "PLOTS_DIR": plots,
        "TABLES_DIR": tables,
        "CHANNELS": CHANNELS,
        "TABLE_ENERGIES_TEV": TABLE_ENERGIES_TEV,
        "WILSON_INTERVALS": WILSON_INTERVALS,
        "build_channel_tables": build_channel_tables,
        "latex_wilson_tables_for_process": latex_wilson_tables_for_process,
        "load_analysis": load_analysis,
        "plots_dir": plots_dir,
        "resolve_scan_axis": resolve_scan_axis,
        "scan_and_save": scan_and_save,
        "scan_axes": scan_axes,
        "scan_axes_and_save": scan_axes_and_save,
        "scan_grid_and_save": scan_grid_and_save,
        "tables_dir": tables_dir,
        "spot_check_caption": spot_check_caption,
        "spot_check_table": spot_check_table,
        "default_plot_title": default_plot_title,
        "plot_style_with_layout": plot_style_with_layout,
        "scan_plot_filename_stem": scan_plot_filename_stem,
        "plot_sigma_nnlo_and_enhancement_nnlo": plot_sigma_nnlo_and_enhancement_nnlo,
        "plot_sigma_nnlo_and_kfactor": plot_sigma_nnlo_and_kfactor,
        "KAPPA_PLAIN": KAPPA_PLAIN,
        "ZHH_TABLE_GROUPS": ZHH_TABLE_GROUPS,
        "plt": plt,
        "Markdown": Markdown,
        "display": display,
    }


def setup_smeft(root: Path | None = None) -> Dict[str, Any]:
    """Validate repo, reload package, import SMEFT notebook symbols."""
    import matplotlib.pyplot as plt
    from IPython.display import Markdown, display

    repo = _prepare_repo("SMEFT", "vhh_prediction_SMEFT.ipynb", root)
    from vhh_predict.analysis import plots_dir, tables_dir
    from vhh_predict.plot_style import (
        default_plot_title,
        plot_style_with_layout,
        scan_plot_filename_stem,
    )
    from vhh_predict.plots import (
        plot_sigma_nnlo_and_enhancement_nnlo,
        plot_sigma_nnlo_and_kfactor,
    )
    from vhh_predict.smeft_analysis import load_smeft_analysis
    from vhh_predict.smeft_core import spot_check_caption, spot_check_table
    from vhh_predict.smeft_operators import (
        SMEFT_WC_INTERVALS,
        SMEFT_WC_PLAIN,
        scan_axes,
        sm_wc_values,
    )
    from vhh_predict.smeft_scan_io import scan_and_save, scan_axes_and_save, scan_grid_and_save
    from vhh_predict.smeft_tables import (
        CHANNELS,
        TABLE_ENERGIES_TEV,
        build_channel_tables,
        latex_wc_interval_table,
    )

    plots = plots_dir("SMEFT")
    tables = tables_dir("SMEFT")
    plots.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    return {
        "REPO_ROOT": repo,
        "PLOTS_DIR": plots,
        "TABLES_DIR": tables,
        "plt": plt,
        "Markdown": Markdown,
        "display": display,
        "default_plot_title": default_plot_title,
        "plot_style_with_layout": plot_style_with_layout,
        "scan_plot_filename_stem": scan_plot_filename_stem,
        "plot_sigma_nnlo_and_enhancement_nnlo": plot_sigma_nnlo_and_enhancement_nnlo,
        "plot_sigma_nnlo_and_kfactor": plot_sigma_nnlo_and_kfactor,
        "load_smeft_analysis": load_smeft_analysis,
        "spot_check_caption": spot_check_caption,
        "spot_check_table": spot_check_table,
        "SMEFT_WC_INTERVALS": SMEFT_WC_INTERVALS,
        "SMEFT_WC_PLAIN": SMEFT_WC_PLAIN,
        "scan_axes": scan_axes,
        "sm_wc_values": sm_wc_values,
        "scan_and_save": scan_and_save,
        "scan_axes_and_save": scan_axes_and_save,
        "scan_grid_and_save": scan_grid_and_save,
        "CHANNELS": CHANNELS,
        "TABLE_ENERGIES_TEV": TABLE_ENERGIES_TEV,
        "build_channel_tables": build_channel_tables,
        "latex_wc_interval_table": latex_wc_interval_table,
    }
