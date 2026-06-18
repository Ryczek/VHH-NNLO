# VHH HEFT prediction package

Minimal package to evaluate **vector-boson-fusion Higgs pair** production (`W^±HH`, `ZHH`) from bundled JSON inputs via `vhh_prediction.ipynb`.

\[
\sigma(\kappa) = \mathbf{m}(\kappa)^{\mathsf T}\mathbf{A},
\qquad
\Delta\sigma = \sqrt{\mathbf{m}^{\mathsf T}\mathbf{C}\,\mathbf{m}}
\]

Supported: `WplusHH`, `WminusHH`, `ZHH` at **13.6 TeV** and **14 TeV**.

---

## Installation

```bash
cd Package
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH="$(pwd):$PYTHONPATH"
```

Or: `pip install -e ".[notebook]"`

---

## What is included

```
Package/
├── vhh_prediction.ipynb          # main entry point
├── vhh_predict/                  # prediction code
│   ├── analysis.py               # load JSON data
│   ├── core.py                   # σ, K, uncertainties, scans
│   ├── covariance_matrices.py    # PDF + α_s loader
│   ├── scale_coefficients.py     # scale envelope loader
│   ├── monomials.py              # HEFT monomial vectors
│   └── plots.py                  # scan plots
├── data/
│   └── {Process}/{energy}/
│       ├── {Process}_{energy}_analysis_A.txt   # human-readable summary (reference)
│       ├── pdf_alpha_s_covariance.json
│       └── scale_coefficients.json
└── plots/                        # notebook output (gitignored *.png)
```

### Data files (required)

| File | Contents |
|------|----------|
| `{Process}_{energy}_analysis_A.txt` | Human-readable reference (methods, tables, covariances); not read at runtime |
| `pdf_alpha_s_covariance.json` | Central \(A_i\), PDF/αₛ deltas, `C_pdf`, `C_alphaS`, `C_pdf_alphaS` |
| `scale_coefficients.json` | 7-point refitted \(A_i\) at each \((\mu_R,\mu_F)\) |

### Uncertainties

- **PDF / αₛ**: \(\sqrt{\mathbf{m}^{\mathsf T} C_{\mathrm{pdf/}\alpha_s}\,\mathbf{m}}\) from `pdf_alpha_s_covariance.json`
- **Scale**: 7-point σ envelope from `scale_coefficients.json` (refit \(A_i\), evaluate \(m\cdot A\) at each scale)

---

## Usage

Open `vhh_prediction.ipynb` from the `Package/` directory. Plots are written to `plots/`.

```python
from vhh_predict import load_analysis, predict, scan, plots_dir
from vhh_predict.plots import plot_sigma_nnlo_and_kfactor

analysis = load_analysis("WplusHH", 14.0)
predict(analysis, (1.0, 1.0, 1.0))
```
