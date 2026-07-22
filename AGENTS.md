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
- Orders: **LO** and **NNLO** (ZHH NNLO uses HHZ label in outputs)

Predictions use bundled JSON under `data/{HEFT|SMEFT}/`, not external Monte Carlo.

## Repository layout

```
VHH-NNLO/
├── vhh_prediction_HEFT.ipynb   # HEFT entry point
├── vhh_prediction_SMEFT.ipynb  # SMEFT entry point
├── scripts/build_smeft_package_data.py
├── vhh_predict/
│   ├── analysis.py             # load_analysis(), path helpers
│   ├── core.py                 # HEFT predict(), scan(), …
│   ├── smeft_analysis.py       # load_smeft_analysis()
│   ├── smeft_core.py           # SMEFT predict(), scan(), spot_check_table()
│   ├── smeft_operators.py      # WC keys, intervals, scan axes
│   ├── smeft_simulation.py     # SMEFT .out comparison
│   ├── smeft_scan_io.py        # SMEFT scan_and_save()
│   ├── smeft_tables.py         # SMEFT benchmark tables
│   ├── scan_io.py, tables.py, simulation.py, plots.py, …
├── data/
│   ├── HEFT/{Process}/{13_6TeV|14_0TeV}/
│   │   ├── pdf_alpha_s_covariance.json
│   │   ├── scale_coefficients.json
│   │   ├── {Process}_{energy}_analysis_A.txt   # human reference only
│   │   └── Simulation/*.out
│   └── SMEFT/{Process}/{13_6TeV|14_0TeV}/
│       ├── pdf_alpha_s_covariance.json   # B_i + diagonal covariances
│       ├── scale_coefficients.json       # B_i refit at 7 scale points
│       ├── sigma_sm.json
│       ├── {Process}_{energy}_analysis_B.txt
│       └── Simulation/*.out
└── Results/
    ├── Points/ (+ SMEFT/)
    ├── Plots/ (+ SMEFT/)
    └── Tables/ (+ SMEFT/)
```

Path resolution:

- `package_root()` = parent of `vhh_predict/` (repo root in dev)
- `heft_data_root()` = `package_root() / "data" / "HEFT"`
- `smeft_data_root()` = `package_root() / "data" / "SMEFT"`
- `data_root(framework)` accepts `"HEFT"` or `"SMEFT"`

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

See `WILSON_INTERVALS` in `vhh_predict/tables.py`.

### Simulation (HEFT)

`data/HEFT/{Process}/{energy}/Simulation/*.out` — κ-encoded filenames.

`COMPARE_SIMULATION` in HEFT notebook: **§2 spot check only**.

## SMEFT physics / API

### Prediction

σ = σ_SM + **C**ᵀ **B** (LO and NNLO/HHZ separately), with $C_i$ in TeV⁻², $B_i$ in fb·TeV².

- PDF+αs: √(C_wcᵀ C C_wc) with diagonal C from single-operator B extraction
- Scale: 7-point envelope on σ using refitted B_i and σ_SM at each (μ_R, μ_F)

### WC keys (logical names)

| Channel | LO keys | NNLO extras (ZHH only) |
|---------|---------|------------------------|
| W± | `phi`, `phiBox`, `phiD`, `phiq3st`, `phiW` | — |
| ZHH | above + `phiq1st`, `phiu`, `phid`, `phiB`, `phiWB` | `tphi` ($C_{t\varphi}$), `phiQ3rd` (B₁₂ combo) |

SM: all C_i = 0 (`sm_wc_values(process)`).

### WC intervals (SMEFT)

`SMEFT_WC_INTERVALS` in `vhh_predict/smeft_operators.py` — bosonic and fermionic tables match the paper (terHoeve:2025gey global fit).

### Simulation (SMEFT)

`data/SMEFT/{Process}/{energy}/Simulation/*.out` — Fortran coefficient encoded in filename (`cHw_Value_-0_2`, etc.).

`load_smeft_simulation_central()`, `find_smeft_simulation_out()` in `smeft_simulation.py`.

## Key functions

### HEFT

```python
from vhh_predict import load_analysis, predict, scan_and_save

analysis = load_analysis("ZHH", 14.0)
p = predict(analysis, (1.0, 1.0, 1.0, 1.0))   # ZHH: 4-tuple κ
```

### SMEFT

```python
from vhh_predict.smeft_analysis import load_smeft_analysis
from vhh_predict.smeft_core import predict, scan
from vhh_predict.smeft_scan_io import scan_and_save

analysis = load_smeft_analysis("ZHH", 14.0)
p = predict(analysis, {"phiW": -0.2})
scan_data, path = scan_and_save(analysis, "phiW", vmin=-1, vmax=1, save=True)
```

## Notebooks

### `vhh_prediction_HEFT.ipynb`

Independent HEFT workflow. Setup checks `data/HEFT/`. Sections §1–§5 as before.

### `vhh_prediction_SMEFT.ipynb`

Independent SMEFT workflow. Setup checks `data/SMEFT/`. Uses `WCS` dict instead of `KAPPA` tuple. Scan outputs under `Results/Points/SMEFT/`, plots under `Results/Plots/SMEFT/`.

Neither notebook depends on the other. Both import shared plot helpers from `vhh_predict.plots`.

## Development notes

- Run notebooks from **repo root**; Setup cells verify expected `data/{HEFT|SMEFT}/` exists.
- Install: `pip install -e ".[notebook]"`.
- Regenerate SMEFT bundle: `python scripts/build_smeft_package_data.py --repo-root <main-repo>`.
- `*_analysis_A.txt` / `*_analysis_B.txt` are human reference only.
- HEFT simulation: κ filenames. SMEFT simulation: `cH*` / `cth` / `chust` filenames.
- Do not commit generated `Results/Plots/*.png` unless intentional.

## Packaging (`pyproject.toml`)

- Package `vhh-predict`, Python ≥3.10.
- `[tool.setuptools.data-files]` bundles `data/**/*` for non-editable installs.
