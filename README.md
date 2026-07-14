# VHH-NNLO

Closed-form **HEFT** predictions for **$W^\pm HH$** and **$ZHH$** at **LO** and **NNLO QCD**, with PDF + $\alpha_s$ and scale uncertainties.

Bundled coefficient files live in `data/` — no external Monte Carlo or fitting step is required.

This repository accompanies an **in-preparation publication** (*Precise predictions for double Higgs production in association with a vector boson in Effective Field Theory*). A formal citation will be added once the paper is public.

---

## What is this, in plain terms?

$W^\pm HH$ and $ZHH$ are LHC processes where a vector boson ($W$ or $Z$) is produced
alongside **two** Higgs bosons. Their rate is sensitive to the Higgs self-coupling
and other couplings that are hard to probe otherwise. This code gives you the
predicted cross section $\sigma$ for these processes at a chosen collider energy,
either in the **Standard Model (SM)** or at any point in a small set of new-physics
parameters (Wilson coefficients $\kappa$), computed at **LO** and **NNLO QCD**
accuracy using **HEFT** (Higgs Effective Field Theory).

Under the hood, each cross section is a fixed polynomial in $\kappa$ with
coefficients ($A_i$) fitted once and bundled as JSON in `data/`. Evaluating a
point is therefore just a dot product — fast, closed-form, no simulation needed.

**First time here?** Open [`vhh_prediction.ipynb`](vhh_prediction.ipynb), edit
the one configuration cell (§1), and run all cells. The notebook has its own
glossary and walkthrough, so you don't need to read this whole file first.

### Glossary

| Term | Meaning |
|---|---|
| **HEFT** | Higgs Effective Field Theory — parametrizes possible deviations from SM Higgs couplings without assuming a specific new-physics model. |
| $\kappa$ (**kappa**, Wilson coefficient) | Multiplicative rescaling of an SM coupling ($\kappa_\lambda$ = Higgs self-coupling, $\kappa_W/\kappa_Z$ = $VVH$ coupling, $\kappa_{2W}/\kappa_{2Z}$ = contact operator, $\kappa_t$ = top Yukawa, ZHH only). SM = all $\kappa = 1$. |
| $\sigma_{\mathrm{LO}}$, $\sigma_{\mathrm{NNLO}}$ | Predicted cross section in **fb**, at leading order and next-to-next-to-leading order QCD. |
| $K$-factor | $\sigma_{\mathrm{NNLO}}/\sigma_{\mathrm{LO}}$ — the size of higher-order QCD corrections. |
| $\sigma_{\mathrm{HEFT}}/\sigma_{\mathrm{SM}}$ | Enhancement relative to the SM point ($\kappa=1$), at the same order. |
| PDF+$\alpha_s$ uncertainty | Symmetric uncertainty from parton distribution functions and the strong coupling constant. |
| Scale uncertainty | Asymmetric uncertainty from varying renormalization/factorization scales (7-point envelope). |

---

## What you can do

1. **One point** — print $\sigma_{\mathrm{LO}}$, $\sigma_{\mathrm{NNLO}}$, $K = \sigma_{\mathrm{NNLO}}/\sigma_{\mathrm{LO}}$, and $\sigma_{\mathrm{HEFT}}/\sigma_{\mathrm{SM}}$ at any $\kappa$.
2. **Scan** — vary one Wilson coefficient; get uncertainty bands and optional plots.
3. **Compare** — optional check against bundled simulation central values (if present in `data/`).

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

Open [`vhh_prediction.ipynb`](vhh_prediction.ipynb). Edit **§1** (process, flags), **§2** ($\kappa$), **§3** (scan + plots), then run the rest.

| § | What it does | Main output |
|---|--------------|-------------|
| **1** | Process, energy, output flags, `SIGMA_INSET` | — |
| **2** | Set `KAPPA`; spot-check table | HEFT + uncertainties (+ simulation if enabled) |
| **3** | Single-axis scan and two-panel plots | `scan_data` + PNGs (+ optional `.txt`) |
| **4** | Batch scan any set of axes | one `.txt` per axis in `Results/Points/` |
| **5** | Wilson-coefficient benchmark tables | `Results/Tables/wilson_tables.tex` |

### §3 two-panel plots

| Plot | File suffix |
|------|-------------|
| $\sigma_{\mathrm{NNLO}}$ + $K$ | `_sigma_nnlo_K.png` |
| $\sigma_{\mathrm{NNLO}}$ + $\sigma_{\mathrm{HEFT}}/\sigma_{\mathrm{SM}}$ (NNLO) | `_sigma_nnlo_sigmaSM_NNLO.png` |

§4 batch scans are independent (save-only, no plots).

### §4 batch scan (API)

```python
from vhh_predict import scan_axes_and_save, WILSON_INTERVALS

batch = scan_axes_and_save(
    analysis,
    ("kappa_lambda", "kappa_t"),
    windows={"kappa_t": (0.9, 1.1)},  # others → WILSON_INTERVALS
    fixed_kappa=kappa,
    n_points=400,
    save=True,
)
# batch["kappa_lambda"] → (scan_data, path)
```

### §1 flags (most common)

| Variable | Meaning |
|----------|---------|
| `PROCESS` | `WplusHH`, `WminusHH`, or `ZHH` |
| `ENERGY_TEV` | `13.6` or `14.0` |
| `COMPARE_SIMULATION` | Compare to bundled simulation in §2 only |
| `UNCERTAINTIES_AS_PERCENT` | Print §2 uncertainties as % (`False` → fb) |
| `SAVE_SCAN_POINTS` | Write scan `.txt` table to `Results/Points/` |
| `SAVE_PLOTS` | Write plot PNGs to `Results/Plots/` |
| `SIGMA_INSET` | Zoom inset on $\sigma_{\mathrm{NNLO}}$ panels (`True` / `False`) |
| `LEGEND_LOC` | Top-panel legend: `upper left`, `upper right`, `lower left`, `lower right` |
| `INSET_LOC` | Inset corner (same choices); ignored when `SIGMA_INSET=False` |

Lower panels ($K$, $\sigma_{\mathrm{HEFT}}/\sigma_{\mathrm{SM}}$) have **no legend** — the y-axis label is enough.

Spot-check table columns **Simulation** and **Δ vs simulation** appear when `COMPARE_SIMULATION=True`.

Plot typography uses `DEFAULT_PLOT_STYLE`; corners via `plot_style_with_layout(...)`.

**Scan limits** — set `scan_vmin`, `scan_vmax` in §3.

### §2 — Wilson coefficients

Set `KAPPA` in the spot-check cell (§2). §3 and §4 scans vary one axis at a time;
the other components stay at the matching entries in that tuple.
See **Wilson coefficients ($\kappa$)** below for tuple order.

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

**95% CL intervals** (Table 2, HEFT Wilson-coefficient uncertainty note; defined as `WILSON_INTERVALS` in `vhh_predict/tables.py`):

| Coefficient | Min | SM | Max |
|-------------|-----|----|-----|
| $\kappa_\lambda$ | −1.7 | 1 | 6.6 |
| $\kappa_W$ | 0.86 | 1 | 1.18 |
| $\kappa_Z$ | 0.90 | 1 | 1.17 |
| $\kappa_{2W}$ | 0.4 | 1 | 1.6 |
| $\kappa_{2Z}$ | 0.4 | 1 | 1.6 |
| $\kappa_t$ | 0.85 | 1 | 1.15 |

Used as default scan windows in §4 batch scans, as boundary points in §5 Wilson tables, and printed as a reference in §3 (override plot windows with `scan_vmin` / `scan_vmax`).

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
    save=True,            # write Results/Points/...txt
)
```

```python
# ZHH example (4-tuple — include kappa_t)
analysis = load_analysis("ZHH", 14.0)
kappa = (3.0, 1.0, 1.0, 1.0)
```

Plots (after a scan with `uncertainties=True`):

```python
from vhh_predict import PlotStyle, default_plot_title, scan_plot_filename_stem
from vhh_predict.plots import plot_sigma_nnlo_and_kfactor

style = PlotStyle(legend_h="right", legend_v="lower", inset_h="left", inset_v="upper")
title = default_plot_title("ZHH", 14.0)  # e.g. "Zhh @ NNLO QCD, sqrt(s) = 14 TeV"
plot_sigma_nnlo_and_kfactor(
    scan_data,
    title=title,
    xmin=scan_vmin,
    xmax=scan_vmax,
    style=style,
    sigma_inset=True,
    output="Results/Plots/example.png",
)
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
├── Points/    # κ-scan tables from scan_and_save() (.txt)
├── Plots/     # figures from the notebook
└── Tables/    # LaTeX tables (wilson_tables.tex)
```

### Scan files (`Results/Points/`)

Filename: `{Process}_{energy}TeV_{axis}.txt`  
Example: `ZHH_14.0TeV_kappa_t.txt`

Plain text: a short `#` comment header (process, energy, scan range, fixed κ) followed by a tab-separated table:

```
# process: ZHH
# energy_tev: 14
# scan_axis: kappa_t
...
kappa_t	sigma_lo	sigma_nnlo	k	sigma_heft_over_sm_lo	sigma_heft_over_sm_nnlo
0.85	...
```

| Column | Meaning |
|--------|---------|
| `kappa_*` | scanned Wilson coefficient |
| `sigma_lo`, `sigma_nnlo` | HEFT cross sections [fb] |
| `k` | $\sigma_{\mathrm{NNLO}}/\sigma_{\mathrm{LO}}$ |
| `sigma_heft_over_sm_lo`, `sigma_heft_over_sm_nnlo` | $\sigma_{\mathrm{HEFT}}/\sigma_{\mathrm{SM}}$ |

Reload in Python: `load_scan_results(path)`. Legacy `.json` scans still load. Uncertainty bands are not saved; pass `uncertainties=True` to `scan_and_save()` when plotting.

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
│       └── Simulation/*.out   # simulation central values (optional comparison)
└── Results/                   # generated outputs (see above)
```

| File in `data/` | Role |
|-----------------|------|
| `pdf_alpha_s_covariance.json` | Central $A_i$, PDF/$\alpha_s$ deltas, covariance |
| `scale_coefficients.json` | Refitted $A_i$ at seven scale points |
| `Simulation/*.out` | Fortran central $\sigma$ for validation |
| `*_analysis_A.txt` | Human-readable reference only (not read at runtime) |

---

## Citation

Publication **in preparation**. Contact the authors for a preprint reference before the paper is public. BibTeX will be added here upon release.

---

## License

See the repository license file (if present).
