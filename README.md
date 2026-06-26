# VHH-NNLO

Closed-form **HEFT** predictions for **$W^\pm HH$** and **$ZHH$** at **LO** and **NNLO QCD**, with PDF + $\alpha_s$ and scale uncertainties.

Bundled coefficient files live in `data/` — no external Monte Carlo or fitting step is required.

This repository accompanies an **in-preparation publication** (*Precise predictions for double Higgs production in association with a vector boson in Effective Field Theory*). A formal citation will be added once the paper is public.

---

## What you can do

1. **One point** — print $\sigma_{\mathrm{LO}}$, $\sigma_{\mathrm{NNLO}}$, $K = \sigma_{\mathrm{NNLO}}/\sigma_{\mathrm{LO}}$, and $\sigma_{\mathrm{HEFT}}/\sigma_{\mathrm{SM}}$ at any $\kappa$.
2. **Scan** — vary one Wilson coefficient; get uncertainty bands and optional plots.
3. **Compare** — optional check against bundled MadGraph central values (if present in `data/`).

---

## Install and run (recommended)

From the **repository root**:

```bash
git clone <repo-url>
cd VHH-NNLO
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[notebook]"
jupyter notebook vhh_prediction.ipynb
```

Then run all cells top to bottom.

**Alternative** (no editable install):

```bash
pip install -r requirements.txt
export PYTHONPATH="$(pwd):$PYTHONPATH"
jupyter notebook vhh_prediction.ipynb
```

> **Always run the notebook from the repo root.** Paths to `data/` and `Results/` are resolved relative to the project directory.

---

## Notebook guide

Open [`vhh_prediction.ipynb`](vhh_prediction.ipynb). Edit **§1**, then run the rest.

| § | What it does | Main output |
|---|--------------|-------------|
| **1** | Choose process, energy, $\kappa$, flags | — |
| **2** | Spot check at one $\kappa$ | printed $\sigma$, $K$, $\sigma_{\mathrm{HEFT}}/\sigma_{\mathrm{SM}}$ |
| **3** | Scan one $\kappa$ axis | `scan_data` in memory; optional JSON in `Results/Points/` |
| **4** | Single-panel plots | PNGs in `Results/Plots/` |
| **5** | Two-panel plots | PNGs in `Results/Plots/` |
| **6** | Wilson benchmark tables | `Results/Tables/wilson_tables.tex` |

### §1 flags (most common)

| Variable | Meaning |
|----------|---------|
| `PROCESS` | `WplusHH`, `WminusHH`, or `ZHH` |
| `ENERGY_TEV` | `13.6` or `14.0` |
| `KAPPA` | Wilson coefficients at the fixed point (see below) |
| `COMPARE_SIMULATION` | Compare to MadGraph in §2 only |
| `SAVE_SCAN_POINTS` | Write scan JSON to `Results/Points/` |
| `SAVE_PLOTS` | Write plot PNGs to `Results/Plots/` |
| `SIGMA_INSET` | Zoom inset on $\sigma_{\mathrm{NNLO}}$ panels |

### §4 single-panel plots (in order)

| Plot | File suffix |
|------|-------------|
| $\sigma_{\mathrm{NNLO}}$ | `_sigma_nnlo.png` |
| $\sigma_{\mathrm{LO}}$ | `_sigma_lo.png` |
| $K$ | `_K.png` |
| $\sigma_{\mathrm{HEFT}}/\sigma_{\mathrm{SM}}$ (LO) | `_sigmaSM_LO.png` |
| $\sigma_{\mathrm{HEFT}}/\sigma_{\mathrm{SM}}$ (NNLO) | `_sigmaSM_NNLO.png` |

### §5 two-panel plots

| Plot | File suffix |
|------|-------------|
| $\sigma_{\mathrm{NNLO}}$ + $K$ | `_sigma_nnlo_K.png` |
| $\sigma_{\mathrm{NNLO}}$ + $\sigma_{\mathrm{HEFT}}/\sigma_{\mathrm{SM}}$ (NNLO) | `_sigma_nnlo_sigmaSM_NNLO.png` |

Run **§3 once** before §4–5. Plot cells reuse `scan_data` from §3 (no second scan).

---

## Wilson coefficients ($\kappa$)

In the SM, all $\kappa = 1$.

| Symbol | Meaning | $W^\pm HH$ | $ZHH$ |
|--------|---------|:----------:|:-----:|
| $\kappa_\lambda$ | Higgs self-coupling | ✓ | ✓ |
| $\kappa_W$ | $WWH$ coupling | ✓ | — |
| $\kappa_Z$ | $ZZH$ coupling | — | ✓ |
| $\kappa_{2W}$ | $W$-boson operator | ✓ | — |
| $\kappa_{2Z}$ | $Z$-boson operator | — | ✓ |
| $\kappa_t$ | Top Yukawa / $\overline{\mathrm{MT}}$ operator | — | ✓ |

**Tuple order** (must match exactly):

| Process | `kappa` argument |
|---------|------------------|
| `WplusHH`, `WminusHH` | `(kappa_lambda, kappa_w, kappa_2w)` — **3 numbers** |
| `ZHH` | `(kappa_lambda, kappa_z, kappa_2z, kappa_t)` — **4 numbers** |

**Scan axes** (one coefficient varied, others held fixed):

- $W^\pm HH$: `kappa_lambda`, `kappa_w`, `kappa_2w`
- $ZHH$: `kappa_lambda`, `kappa_z`, `kappa_2z`, `kappa_t`

---

## Python API (minimal)

```python
from vhh_predict import load_analysis, predict, format_prediction, scan_and_save

# W± example (3-tuple)
analysis = load_analysis("WplusHH", 14.0)
kappa = (1.0, 1.0, 1.0)   # SM

print(format_prediction(analysis, kappa, compare_simulation=True))

scan_data, path = scan_and_save(
    analysis,
    "kappa_lambda",
    vmin=-1.0,
    vmax=2.0,
    fixed_kappa=kappa,
    n_points=400,
    uncertainties=True,   # bands for plotting (stays in memory)
    save=True,            # write Results/Points/...json
)
```

```python
# ZHH example (4-tuple — include kappa_t)
analysis = load_analysis("ZHH", 14.0)
kappa = (3.0, 1.0, 1.0, 1.0)
```

Plots (after a scan with `uncertainties=True`):

```python
from vhh_predict.plots import plot_sigma_nnlo_and_kfactor, plot_sigma_lo_only

plot_sigma_lo_only(scan_data, output="Results/Plots/example_sigma_lo.png")
plot_sigma_nnlo_and_kfactor(scan_data, output="Results/Plots/example_sigma_K.png")
```

LaTeX Wilson tables:

```python
from vhh_predict import all_channels_latex, tables_dir

(tables_dir() / "wilson_tables.tex").write_text(all_channels_latex())
```

---

## Output files

```
Results/
├── Points/    # κ-scan JSON from scan_and_save()
├── Plots/     # figures from the notebook
└── Tables/    # LaTeX tables (wilson_tables.tex)
```

### Scan JSON (`Results/Points/`)

Filename: `{Process}_{energy}TeV_{axis}.json`  
Example: `ZHH_14.0TeV_kappa_t.json`

| Field | Meaning |
|-------|---------|
| `kappa_*` | scanned Wilson coefficient grid |
| `sigma_lo`, `sigma_nnlo` | HEFT cross sections [fb] |
| `k` | $\sigma_{\mathrm{NNLO}}/\sigma_{\mathrm{LO}}$ |
| `sigma_heft_over_sm_lo`, `sigma_heft_over_sm_nnlo` | $\sigma_{\mathrm{HEFT}}/\sigma_{\mathrm{SM}}$ |

Reload: `load_scan_results(path)`. Uncertainty bands are **not** stored in the JSON; pass `uncertainties=True` to `scan()` / `scan_and_save()` when you need them for plots.

---

## Repository layout

```
VHH-NNLO/
├── vhh_prediction.ipynb       # start here
├── pyproject.toml             # package metadata for pip install
├── requirements.txt           # flat dependency list (alternative install)
├── vhh_predict/               # Python package
├── data/
│   └── {Process}/{13_6TeV|14_0TeV}/
│       ├── pdf_alpha_s_covariance.json
│       ├── scale_coefficients.json
│       └── Simulation/*.out   # MadGraph central values (optional comparison)
└── Results/                   # generated outputs (see above)
```

| File in `data/` | Role |
|-----------------|------|
| `pdf_alpha_s_covariance.json` | Central $A_i$, PDF/$\alpha_s$ deltas, covariance |
| `scale_coefficients.json` | Refitted $A_i$ at seven scale points |
| `Simulation/*.out` | MadGraph central $\sigma$ for validation |
| `*_analysis_A.txt` | Human-readable reference only (not read at runtime) |

---

## Citation

Publication **in preparation**. Contact the authors for a preprint reference before the paper is public. BibTeX will be added here upon release.

---

## License

See the repository license file (if present).
