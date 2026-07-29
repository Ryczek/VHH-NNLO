# AGENTS.md — VHH-NNLO

Context for AI agents working in this repository.

## Purpose

Closed-form predictions for vector-boson-associated Higgs-pair production in two frameworks:

| Framework | Formula | Data root | Main loader |
|-----------|---------|-----------|-------------|
| **HEFT** | $\sigma = \mathbf{m}(\kappa)^\top \mathbf{A}$ | `data/HEFT/` | `load_analysis()` |
| **SMEFT** | $\sigma = \sigma_{\mathrm{SM}} + \sum_i B_i C_i$ | `data/SMEFT/` | `load_smeft_analysis()` |

- Processes: `WplusHH`, `WminusHH`, `ZHH`
- Energies: **13.6 TeV** and **14.0 TeV**
- Orders: **LO** and **NNLO** (for ZHH, NNLO is the HHZ contribution; always labelled NNLO)

Predictions use bundled JSON under `data/{HEFT|SMEFT}/`, not external Monte Carlo.

## Install (package only)

```bash
cd VHH-NNLO
pip install -e ".[notebook]"
```

- Sole install path: `pyproject.toml` (no `requirements.txt`, no notebook `sys.path` hacks).
- After editing `vhh_predict/`, restart the Jupyter kernel.
- Run notebooks from the **repo root**.

## Repository layout

```
VHH-NNLO/
├── vhh_prediction_HEFT.ipynb   # HEFT entry point
├── vhh_prediction_SMEFT.ipynb  # SMEFT entry point
├── pyproject.toml
├── scripts/build_smeft_package_data.py
├── vhh_predict/
│   ├── analysis.py             # load_analysis(), path helpers
│   ├── core.py                 # HEFT predict(), scan(), scan_grid()
│   ├── scan_io.py              # HEFT scan_and_save(), scan_grid_and_save()
│   ├── smeft_analysis.py       # load_smeft_analysis()
│   ├── smeft_core.py           # SMEFT predict(), scan(), scan_grid()
│   ├── smeft_operators.py      # WC keys, intervals, scan axes
│   ├── smeft_simulation.py     # SMEFT .out comparison
│   ├── smeft_scan_io.py        # SMEFT scan_and_save(), scan_grid_and_save()
│   ├── smeft_tables.py         # SMEFT benchmark tables
│   ├── tables.py, simulation.py, plots.py, …
├── data/
│   ├── HEFT/{Process}/{13_6TeV|14_0TeV}/
│   │   ├── pdf_alpha_s_covariance.json
│   │   ├── scale_coefficients.json
│   │   ├── {Process}_{energy}_analysis_A.txt   # human reference only
│   │   └── Simulation/*.out
│   └── SMEFT/{Process}/{13_6TeV|14_0TeV}/
│       ├── pdf_alpha_s_covariance.json
│       ├── scale_coefficients.json
│       ├── sigma_sm.json
│       ├── {Process}_{energy}_analysis_B.txt
│       └── Simulation/*.out
└── results/                         # shipped notebook outputs
    ├── points/{heft,smeft}/{Process}/{13_6TeV|14_0TeV}/
    ├── plots/{heft,smeft}/{Process}/{13_6TeV|14_0TeV}/
    └── tables/{heft,smeft}/         # heft_publication_tables.tex / smeft_publication_tables.tex
```

Path resolution:

- `package_root()` = parent of `vhh_predict/` (repo root in editable install)
- `heft_data_root()` = `package_root() / "data" / "HEFT"`
- `smeft_data_root()` = `package_root() / "data" / "SMEFT"`
- `data_root(framework)` accepts `"HEFT"` or `"SMEFT"`
- `results_dir()` → `results/`
- `points_dir(framework, process, energy_tev)` / `plots_dir(framework, process, energy_tev)` → `results/{points,plots}/{heft|smeft}/{Process}/{13_6TeV|14_0TeV}/`; `tables_dir(framework)` → `results/tables/{heft|smeft}/`
- `scan_plot_path(framework, process, energy_tev, scan_x_key, vmin, vmax, variant)` → default scan PNG under `results/plots/…`
- `heft_wilson_tables_path()` / `smeft_wc_intervals_path()` → publication LaTeX files under `results/tables/{heft|smeft}/`, named `heft_publication_tables.tex` and `smeft_publication_tables.tex`
- Scan point tables use `scan_points_path` / `smeft_scan_points_path` (no notebook config needed)

## HEFT physics / API

### Prediction

σ(κ) = **m**(κ)ᵀ **A**, with PDF+αs uncertainty √(mᵀ C m).

- Scale uncertainty: 7-point envelope from refitted **A** in `scale_coefficients.json`
- K-factor: σ_NNLO / σ_LO

### κ tuple layout

| Process | κ tuple | Scan axes |
|---------|---------|-----------|
| `WplusHH`, `WminusHH` | `(κ_λ, κ_W, κ_{2W})` | `kappa_lambda`, `kappa_w`, `kappa_2w` |
| `ZHH` | `(κ_λ, κ_Z, κ_{2Z}, κ_t)` | `kappa_lambda`, `kappa_z`, `kappa_2z`, `kappa_t` |

SM: all κ = 1 (`sm_kappa(process)`).

### Wilson intervals (HEFT)

`WILSON_INTERVALS` in `vhh_predict/tables.py` holds the **95% CL exclusion intervals for every HEFT $\kappa$** ($\kappa_\lambda$, $\kappa_W$, $\kappa_Z$, $\kappa_{2W}$, $\kappa_{2Z}$, $\kappa_t$). They set default scan windows and the min/max columns in the publication benchmark tables.

### Simulation (HEFT)

`data/HEFT/{Process}/{energy}/Simulation/*.out` — κ-encoded filenames.

`COMPARE_SIMULATION` in HEFT notebook: **§2 spot check only**.

## SMEFT physics / API

### Prediction

σ = σ_SM + **C**ᵀ **B** (LO and NNLO separately; ZHH NNLO = HHZ), with $C_i$ in TeV⁻², $B_i$ in fb·TeV².

- PDF+αs on \(\sigma=\sigma_{\mathrm{SM}}+\mathbf{C}\cdot\mathbf{B}\):
  \(\sqrt{\delta\sigma_{\mathrm{PDF}}^2+\delta\sigma_{\alpha_s}^2}\) with
  \(\mathrm{Var}_{\mathrm{PDF}}=\delta\sigma_{\mathrm{SM}}^2+\mathbf{C}^\top C_B\mathbf{C}+2\,\mathbf{C}\cdot\mathrm{Cov}(\sigma_{\mathrm{SM}},B)\)
  (diagonal \(C_B\) from single-operator scans; \(\mathrm{Cov}\) from the same replicas) and
  \(\delta\sigma_{\alpha_s}=\delta\sigma_{\mathrm{SM},\alpha_s}+\mathbf{C}\cdot\delta B_{\alpha_s}\)
- Scale: 7-point envelope on σ using refitted B_i and σ_SM at each (μ_R, μ_F)

### WC keys (logical names)

| Channel | LO keys | NNLO extras (ZHH only) |
|---------|---------|------------------------|
| W± | `phi`, `phiBox`, `phiD`, `phiq3st`, `phiW` | — |
| ZHH | above + `phiq1st`, `phiu`, `phid`, `phiB`, `phiWB` | `tphi` ($C_{t\varphi}$), `phiQ3rd` (B₁₂ combo) |

`scan_axes(process)` returns the full NNLO set (so ZHH scans may use `tphi` / `phiQ3rd`). SM: all C_i = 0 (`sm_wc_values(process)`).

Benchmark σ tables (§5) cover **all** `scan_axes` WCs. ZHH is split into display groups `ZHH_TABLE_GROUPS` (bosonic / fermionic LO / NNLO extras) that together include every axis.

Doc-only interval keys (not scan axes): `phit` ($C_{\varphi t}$), `phiQ3` ($C_{\varphi Q}^{(3)}$), `phiQ1rd` ($C_{\varphi Q}^{(1)}$).

### WC intervals (SMEFT)

`SMEFT_WC_INTERVALS` in `vhh_predict/smeft_operators.py` — bosonic and fermionic tables match ter Hoeve et al., JHEP 06 (2025) 125 ([arXiv:2502.20453](https://arxiv.org/abs/2502.20453)), Table E.1 (linear+RGE, rounded).

### Simulation (SMEFT)

`data/SMEFT/{Process}/{energy}/Simulation/*.out` — Fortran coefficient encoded in filename (`cHw_Value_-0_2`, etc.).

`load_smeft_simulation_central()`, `find_smeft_simulation_out()` in `smeft_simulation.py`.

## Key functions

### HEFT

```python
from vhh_predict import load_analysis, predict, scan_and_save, scan_grid_and_save

analysis = load_analysis("ZHH", 14.0)
p = predict(analysis, (1.0, 1.0, 1.0, 1.0))   # ZHH: 4-tuple κ
scan_data, path = scan_grid_and_save(
    analysis, ("kappa_lambda", "kappa_z"), n_points=40, save=True
)
```

### SMEFT

```python
from vhh_predict.smeft_analysis import load_smeft_analysis
from vhh_predict.smeft_core import predict
from vhh_predict.smeft_scan_io import scan_and_save, scan_grid_and_save

analysis = load_smeft_analysis("ZHH", 14.0)
p = predict(analysis, {"phiW": -0.2})
scan_data, path = scan_and_save(analysis, "phiW", vmin=-1, vmax=1, save=True)
```

## Notebooks

### `vhh_prediction_HEFT.ipynb`

Independent HEFT workflow. Setup verifies editable install + `data/HEFT/`. Sections §1–§5. Writes points/plots under `results/{points,plots}/heft/<Process>/<energy>/` and tables to `results/tables/heft/heft_publication_tables.tex`.

### `vhh_prediction_SMEFT.ipynb`

Independent SMEFT workflow. Setup verifies editable install + `data/SMEFT/`. Uses `WCS` dict instead of `KAPPA` tuple. Writes points/plots under `results/{points,plots}/smeft/<Process>/<energy>/` and tables to `results/tables/smeft/smeft_publication_tables.tex`.

Neither notebook depends on the other. Both import shared plot helpers from `vhh_predict.plots`.

## Development notes

- Install: `pip install -e ".[notebook]"`.
- Run notebooks from **repo root**; Setup cells verify expected `data/{HEFT|SMEFT}/` exists.
- Restart the Jupyter kernel after changing `vhh_predict/` code.
- Regenerate SMEFT bundle: `python scripts/build_smeft_package_data.py --repo-root <main-repo>`.
- `*_analysis_A.txt` / `*_analysis_B.txt` are human reference only.
- HEFT simulation: κ filenames. SMEFT simulation: `cH*` / `cth` / `chust` filenames.
- Everything under `results/{points,plots,tables}/` is **meant to be committed**. Local junk (`.venv`, `__pycache__`, …) stays in `.gitignore`.
- `.gitkeep` files only keep empty result subdirs in git until real outputs are added.

## Packaging (`pyproject.toml`)

- Package `vhh-predict`, Python ≥3.10.
- Editable install (`pip install -e ".[notebook]"`) is the **only** supported install path. `data/` is located relative to the repo root via `package_root()` (parent of `vhh_predict/`), which only resolves correctly for an editable checkout — a non-editable install or built wheel would not find the bundled data, so no `data-files`/`package_data` packaging is attempted.
