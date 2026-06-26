# AGENTS.md — VHH-NNLO

Context for AI agents working in this repository.

## Purpose

Closed-form **HEFT** predictions for vector-boson-associated Higgs-pair production:

- Processes: `WplusHH`, `WminusHH`, `ZHH`
- Energies: **13.6 TeV** and **14.0 TeV**
- Orders: **LO** and **NNLO** (ZHH NNLO uses the HHZ component label in outputs)

Predictions use bundled JSON under `data/`, not external Monte Carlo.

## Repository layout

```
VHH-NNLO/
├── vhh_prediction.ipynb      # Main entry point (run from repo root)
├── pyproject.toml            # pip packaging config
├── vhh_predict/
│   ├── analysis.py           # load_analysis(), path helpers (data_root, plots_dir, points_dir, …)
│   ├── core.py               # predict(), scan(), format_prediction(), sm_enhancement()
│   ├── scan_io.py            # scan_and_save(), load_scan_results(), scan_points_path()
│   ├── tables.py             # Wilson-interval benchmark tables + LaTeX
│   ├── simulation.py         # MadGraph .out comparison helpers
│   ├── out_parser.py
│   ├── monomials.py
│   ├── covariance_matrices.py
│   ├── scale_coefficients.py
│   └── plots.py
├── data/
│   └── {Process}/{13_6TeV|14_0TeV}/
│       ├── pdf_alpha_s_covariance.json
│       ├── scale_coefficients.json
│       ├── {Process}_{energy}_analysis_A.txt   # human reference only
│       └── Simulation/*.out
└── Results/
    ├── Points/               # scan JSON from scan_and_save()
    ├── Plots/                # notebook figures
    └── Tables/               # LaTeX tables
```

Path resolution: `package_root()` = parent of `vhh_predict/` (repo root in dev). `data_root()` = `package_root() / "data"`.

## Physics / API

### Prediction formula

σ(κ) = **m**(κ)ᵀ **A**, with PDF+αs uncertainty √(mᵀ C m).

- Scale uncertainty: 7-point envelope from refitted **A** in `scale_coefficients.json`
- K-factor: σ_NNLO / σ_LO (central value printed; no K uncertainty in `format_prediction()`)

### κ tuple layout (order matters — no auto-padding)

| Process | κ tuple | Scan axes |
|---------|---------|-----------|
| `WplusHH`, `WminusHH` | `(κ_λ, κ_W, κ_2W)` — **3 values** | `kappa_lambda`, `kappa_w`, `kappa_2w` |
| `ZHH` | `(κ_λ, κ_Z, κ_2Z, κ_t)` — **4 values** | `kappa_lambda`, `kappa_z`, `kappa_2z`, `kappa_t` |

SM: all κ = 1 (`sm_kappa(process)`).

### Wilson-coefficient intervals (benchmark tables, §6)

| Coefficient | Min | SM | Max |
|-------------|-----|-----|-----|
| κ_λ | −1.7 | 1 | 6.6 |
| κ_W, κ_Z | 0.9 | 1 | 1.2 |
| κ_{2W}, κ_{2Z} | 0.4 | 1 | 1.6 |
| κ_t | 0.9 | 1 | 1.2 |

ZHH tables: **two groups** — `(kappa_lambda, kappa_t)` and `(kappa_z, kappa_2z)`.

### Simulation comparison

`data/{Process}/{energy}/Simulation/*.out` — MadGraph central values only.

- `COMPARE_SIMULATION` in notebook: **§2 spot check only** (not on scan plots)
- `load_simulation_central()`, `collect_simulation_scan_points()` in `simulation.py`

## Key functions

```python
from vhh_predict import (
    load_analysis, predict, format_prediction, scan, scan_and_save,
    load_scan_results, build_channel_tables, all_channels_latex,
    plots_dir, points_dir, tables_dir, results_dir,
)

analysis = load_analysis("ZHH", 14.0)
p = predict(analysis, (1.0, 1.0, 1.0, 1.0))   # ZHH needs 4-tuple

scan_data, path = scan_and_save(
    analysis, "kappa_t", vmin=0.85, vmax=1.15,
    fixed_kappa=(3.0, 1.0, 1.0, 1.0),
    n_points=400, uncertainties=True, save=True,
)
```

### Scan I/O

- `scan_and_save()` — primary API: one scan + optional save to `Results/Points/{Process}_{energy}TeV_{axis}.json`
- Saved JSON (v2): κ grid + `sigma_lo`, `sigma_nnlo`, `k`, `sigma_heft_over_sm_lo`, `sigma_heft_over_sm_nnlo` (no uncertainty columns)
- `scan(..., uncertainties=True)` adds PDF/scale bands **in memory** for plotting
- `load_scan_results()` supports v1 and v2 formats

### Plot helpers (`vhh_predict/plots.py`)

| Function | Use |
|----------|-----|
| `plot_sigma_only` | Single panel σ_NNLO |
| `plot_sigma_lo_only` | Single panel σ_LO |
| `plot_kfactor_only` | Single panel K |
| `plot_enhancement_only` | Single panel σ_HEFT/σ_SM (LO or NNLO) |
| `plot_sigma_nnlo_and_kfactor` | Two panel: σ_NNLO + K |
| `plot_sigma_nnlo_and_enhancement_nnlo` | Two panel: σ_NNLO + σ_HEFT/σ_SM (NNLO) |
| `plot_sm_enhancement` | Legacy two-panel LO+NNLO enhancement (not used in notebook) |

Enhancement plots read `sigma_heft_over_sm_*` from scan data.

## Notebook sections (`vhh_prediction.ipynb`)

1. **Configuration** — `PROCESS`, `ENERGY_TEV`, `KAPPA`, `SAVE_SCAN_POINTS`, `SAVE_PLOTS`, `SIGMA_INSET`, `COMPARE_SIMULATION`
2. **Spot check** — `format_prediction()` (σ, K, σ_HEFT/σ_SM; sim on σ lines if enabled)
3. **Scan** — single `scan_and_save(..., uncertainties=True)` → `scan_data` (+ optional Points JSON)
4. **Single-panel plots** — σ_NNLO, σ_LO, K, σ_HEFT/σ_SM LO, σ_HEFT/σ_SM NNLO
5. **Two-panel plots** — σ_NNLO+K; σ_NNLO+σ_HEFT/σ_SM NNLO
6. **Wilson tables** — all channels + LaTeX → `Results/Tables/wilson_tables.tex`

Plot cells reuse `scan_data` from §3 (no second scan, no sim overlay on scans).

## Development notes

- Run notebook from **repo root**; setup cell adds `.` to `sys.path`.
- Install: `pip install -e ".[notebook]"` (see `pyproject.toml`).
- `requirements.txt` is a flat alternative to the pyproject extras.
- Do not commit generated `Results/Plots/*.png`, `Results/Tables/*.tex` unless intentional.
- `Results/Points/` may be committed for published scan grids.
- HEFT coefficients do not depend on m_h in code; tables label m_h = 125 GeV only.
- New simulation points: `.out` files under `data/.../Simulation/` with existing filename convention.

## Packaging (`pyproject.toml`)

- Defines package `vhh-predict`, Python ≥3.10, dependencies (numpy, matplotlib, pandas).
- Optional `[notebook]` extra: jupyter, ipykernel.
- `pip install -e ".[notebook]"` installs the package in editable mode from the repo.
- `[tool.setuptools.data-files]` bundles `data/**/*` for non-editable installs.
