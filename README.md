# VHH-NNLO — HEFT predictions for double Higgs production in association with a vector boson #

Bundled predictions for **$W^\pm HH$** and **$ZHH$** production in the Higgs Effective Field Theory (HEFT) at **LO** and **NNLO QCD**, with PDF + $\alpha_s$ and scale uncertainties.

This repository accompanies an **in-preparation publication** "Precise predictions for double Higgs production in association with a vector boson in Effective Field Theory". A formal citation (arXiv link and BibTeX) will be added here once the paper is public.

---

## What you can do

| Task | How |
|------|-----|
| **Spot check** $\sigma_{\mathrm{LO}}$, $\sigma_{\mathrm{NNLO}}$, $K$ at any $\kappa$ | Notebook §2 or `predict()` / `format_prediction()` |
| **Print** $\sigma_{\mathrm{HEFT}}/\sigma_{\mathrm{SM}}$ (LO and NNLO) at a point | `format_prediction(..., include_enhancement=True)` or `sm_enhancement(analysis, kappa)` |
| **Scan** one Wilson coefficient with uncertainty bands | Notebook §4–5 or `scan()` + `plot_sigma_nnlo_and_kfactor()` |
| **Compare** HEFT vs bundled MadGraph simulation | `format_prediction(..., compare_simulation=True)` — sim $\sigma$ on the same line as HEFT |
| **SM enhancement scan** $\sigma/\sigma_{\mathrm{SM}}$ vs $\kappa$ | Notebook §6 or `scan_sm_enhancement()` |
| **Paper tables** at Wilson-interval boundaries | Notebook §7 → `Results/Tables/wilson_tables.tex` |

All coefficients and simulation benchmarks ship in `data/` — no external Monte Carlo or fitting step is required to run predictions.

---

## Wilson coefficients ($\kappa$)

The package varies the HEFT couplings below. In the SM, all $\kappa = 1$.

| Symbol | Meaning | $W^\pm HH$ | $ZHH$ |
|--------|---------|:----------:|:-----:|
| $\kappa_\lambda$ | Higgs self-coupling | ✓ | ✓ |
| $\kappa_W$ | $WWH$ coupling | ✓ | — |
| $\kappa_Z$ | $ZZH$ coupling | — | ✓ |
| $\kappa_{2W}$ | $W$-boson operator | ✓ | — |
| $\kappa_{2Z}$ | $Z$-boson operator | — | ✓ |
| $\kappa_t$ | Top Yukawa / $\overline{\mathrm{MT}}$ operator | — | ✓ |

**Tuple order** passed to `predict()` (must match):

| Process | `kappa` tuple |
|---------|----------------|
| `WplusHH`, `WminusHH` | `(kappa_lambda, kappa_w, kappa_2w)` |
| `ZHH` | `(kappa_lambda, kappa_z, kappa_2z)` or add `kappa_t` as a fourth entry (defaults to `1`) |

**Scan axes** (one coefficient varied, others fixed): `kappa_lambda`, `kappa_w` / `kappa_z`, `kappa_2w` / `kappa_2z`, and `kappa_t` (ZHH only).

**Wilson intervals** used in the benchmark tables (§7); boundaries are rounded to $0.1$ in table headers:

| $\kappa$ | Interval |
|----------|----------|
| $\kappa_\lambda$ | $[-1.7,\, 6.6]$ |
| $\kappa_W$, $\kappa_Z$ | $[0.9,\, 1.2]$ |
| $\kappa_{2W}$, $\kappa_{2Z}$ | $[0.4,\, 1.6]$ |
| $\kappa_t$ | $[0.9,\, 1.2]$ |

---

## Quick start

### Install

```bash
git clone <repo-url>
cd VHH-NNLO
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[notebook]"
```

Or, without editable install:

```bash
pip install -r requirements.txt
export PYTHONPATH="$(pwd):$PYTHONPATH"
```

### Notebook (recommended)

Open [`vhh_prediction.ipynb`](vhh_prediction.ipynb) from the **repo root** and run all cells.

| Section | Content |
|---------|---------|
| 1 | Choose process, energy, $\kappa$, flags |
| 2 | Print $\sigma$, $\sigma/\sigma_{\mathrm{SM}}$, and $K$ ($K$ without uncertainties; sim on $\sigma$ lines when enabled) |
| 3 | Structured `predict()` output |
| 4–5 | $\kappa$ scans → `Results/Plots/` |
| 6 | SM enhancement (optional) |
| 7 | Wilson benchmark tables → `Results/Tables/wilson_tables.tex` |

### Python API

```python
from vhh_predict import load_analysis, predict, format_prediction, sm_enhancement, scan
from vhh_predict.plots import plot_sigma_nnlo_and_kfactor

analysis = load_analysis("WplusHH", 14.0)
kappa = (1.0, 1.0, 1.0)   # SM: (κ_λ, κ_W, κ_2W)

p = predict(analysis, kappa)
print(p.sigma_lo, p.sigma_nnlo, p.k_factor)
print(sm_enhancement(analysis, kappa, "NNLO"))  # σ_HEFT/σ_SM at NNLO

print(format_prediction(analysis, kappa, compare_simulation=True, include_enhancement=True))

scan_data = scan(analysis, axis="kappa_lambda", vmin=-1.0, vmax=2.0, fixed_kappa=kappa)
```

Paper-ready LaTeX tables:

```python
from vhh_predict import all_channels_latex, tables_dir

(tables_dir() / "wilson_tables.tex").write_text(all_channels_latex())
```

---

## Repository layout

```
VHH-NNLO/
├── vhh_prediction.ipynb       # main entry point
├── vhh_predict/               # Python package
├── data/
│   └── {Process}/{13_6TeV|14_0TeV}/
│       ├── pdf_alpha_s_covariance.json
│       ├── scale_coefficients.json
│       ├── {Process}_{energy}_analysis_A.txt   # human reference (not read at runtime)
│       └── Simulation/*.out                    # MadGraph central values (comparison)
└── Results/
    ├── Plots/                 # figures from the notebook
    └── Tables/                # LaTeX tables (e.g. wilson_tables.tex)
```

### Bundled data

| File | Role |
|------|------|
| `pdf_alpha_s_covariance.json` | Central $A_i$, PDF and $\alpha_s$ deltas, covariance blocks |
| `scale_coefficients.json` | Refitted $A_i$ at seven $(\mu_R, \mu_F)$ points for scale envelopes |
| `Simulation/*.out` | MadGraph central $\sigma$ for validation overlays |
| `*_analysis_A.txt` | Methods and tables for human inspection only |

---

## Citation

Publication **in preparation**. If you use this code or the bundled numbers before the paper appears, please contact the authors for a preprint reference. A BibTeX entry will be added to this README upon release.

---

## License

See repository license file (if present). For questions about the physics or bundled inputs, refer to the in-preparation paper or open an issue.
