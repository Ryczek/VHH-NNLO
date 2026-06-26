# VHH HEFT prediction package

Minimal package to evaluate **vector-boson-fusion Higgs pair** production (`W±HH`, `ZHH`) from bundled JSON inputs via `vhh_prediction.ipynb`.

$$
\sigma(\kappa) = \mathbf{m}(\kappa)^{\mathsf T}\mathbf{A},
\qquad
\Delta\sigma = \sqrt{\mathbf{m}^{\mathsf T}\mathbf{C}\,\mathbf{m}}
$$

Supported processes: `WplusHH`, `WminusHH`, `ZHH` at **13.6 TeV** and **14 TeV**.

---

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH="$(pwd):$PYTHONPATH"
```

Or: `pip install -e ".[notebook]"`

---

## What is included

```
VHH-NNLO/
├── vhh_prediction.ipynb          # main entry point
├── vhh_predict/                  # prediction code
│   ├── analysis.py               # load JSON data
│   ├── core.py                   # sigma, K, uncertainties, scans
│   ├── covariance_matrices.py    # PDF + alpha_s loader
│   ├── scale_coefficients.py     # scale envelope loader
│   ├── monomials.py              # HEFT monomial vectors
│   └── plots.py                  # scan plots
├── data/
│   └── {Process}/{energy}/
│       ├── {Process}_{energy}_analysis_A.txt   # human-readable summary (reference)
│       ├── pdf_alpha_s_covariance.json
│       ├── scale_coefficients.json
│       └── Simulation/           # MadGraph .out central values (comparison)
├── Results/
│   ├── Plots/                    # notebook figures (gitignored *.png)
│   └── Tables/                   # LaTeX tables (gitignored *.tex)
```

### Data files (required)

| File | Contents |
|------|----------|
| `{Process}_{energy}_analysis_A.txt` | Human-readable reference (methods, tables, covariances); not read at runtime |
| `pdf_alpha_s_covariance.json` | Central $A_i$, PDF and $\alpha_s$ deltas, `C_pdf`, `C_alphaS`, `C_pdf_alphaS` |
| `scale_coefficients.json` | 7-point refitted $A_i$ at each $(\mu_R, \mu_F)$ |
| `Simulation/*.out` | MadGraph central $\sigma$ values for comparison with HEFT extrapolation |

### Uncertainties

- **PDF + $\alpha_s$**: $\sqrt{\mathbf{m}^{\mathsf T} C_{\mathrm{pdf}+\alpha_s}\,\mathbf{m}}$ from `pdf_alpha_s_covariance.json`
- **Scale**: 7-point $\sigma$ envelope from `scale_coefficients.json` (refit $A_i$, evaluate $\mathbf{m}\cdot\mathbf{A}$ at each scale)

---

## Usage

Open `vhh_prediction.ipynb` from the repo root. Plots go to `Results/Plots/`, tables to `Results/Tables/`.

```python
from vhh_predict import load_analysis, predict, scan, plots_dir, tables_dir
from vhh_predict.plots import plot_sigma_nnlo_and_kfactor

analysis = load_analysis("WplusHH", 14.0)
predict(analysis, (1.0, 1.0, 1.0))
```
