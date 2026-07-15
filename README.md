# VHH-NNLO

Closed-form **HEFT** predictions for **$W^\pm HH$** and **$ZHH$** at **LO HEFT** ($\kappa$ framework) and **NNLO QCD**, with PDF + $\alpha_s$ and scale uncertainties. Coefficients are bundled in `data/` — no Monte Carlo or fitting step required.

Accompanies an **in-preparation publication** (*Precise predictions for double Higgs production in association with a vector boson in Effective Field Theory*). Citation to be added when the paper is public.

---

## What it does

- **Spot check** — $\sigma_{\mathrm{LO}}$, $\sigma_{\mathrm{NNLO}}$, $K$-factor, and $\sigma_{\mathrm{HEFT}}/\sigma_{\mathrm{SM}}$ at any Wilson-coefficient point ($\kappa$).
- **Scan** — vary one $\kappa$ (§3) or batch-scan **any set of** $\kappa$ axes (§4); optional PDF/scale bands and `.txt` tables in `Results/Points/`.
- **Plot** — two-panel figures from a §3 scan: $\sigma_{\mathrm{NNLO}}+K$ and $\sigma_{\mathrm{NNLO}}+\sigma_{\mathrm{HEFT}}/\sigma_{\mathrm{SM}}$ → `Results/Plots/`.
- **Compare** — optional check against bundled simulation central values (§2 only).
- **Wilson tables** — benchmark tables at 95% CL interval boundaries → LaTeX.

SM: all $\kappa = 1$. Processes: `WplusHH`, `WminusHH`, `ZHH` at **13.6** or **14.0** TeV.

---

## Quick start

From the **repository root**:

```bash
git clone <repo-url>
cd VHH-NNLO
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[notebook]"
jupyter notebook vhh_prediction.ipynb
```

Run all cells top to bottom. Edit **§1** (process, flags), **§2** ($\kappa$), **§3** (scan + plots), then the rest.

> Always start Jupyter from the repo root — paths to `data/` and `Results/` are relative to it.

---

## Notebook layout

[`vhh_prediction.ipynb`](vhh_prediction.ipynb) is the main entry point:

**Setup** — Adds the repo to `sys.path`, imports `vhh_predict`, and checks that `data/` exists. Run once; nothing to edit.

**§1 Configuration** — Choose the physics setup: process (`WplusHH`, `WminusHH`, or `ZHH`), energy (13.6 or 14 TeV), and output switches (save scan tables, save plots, compare to simulation). Plot styling (legend/inset corners, σ inset zoom) is set here too. Loads the analysis object used everywhere below. The Wilson 95% CL interval table is listed here for reference when picking scan ranges.

**§2 Spot check** — Set `KAPPA`, a single Wilson-coefficient point, and print a table of LO/NNLO cross sections, K-factor, enhancement over SM, and scale + PDF+αs uncertainties. Optional simulation columns compare to bundled `.out` files. This point is **only** for the table — scans use `FIXED_KAPPA` in §3–§4 instead.

**§3 Scan and plots** — Pick one κ axis to vary (`scan_axis`) and a window (`scan_vmin` … `scan_vmax`). Set `FIXED_KAPPA` for all **other** κ components. The cell runs the 1D grid, keeps uncertainty bands in memory, optionally writes a `.txt` to `Results/Points/`, then produces two figures: σ_NNLO + K, and σ_NNLO + σ_HEFT/σ_SM (NNLO).

**§4 Batch scan** — Same idea as §3, but loop over **many** axes at once (`BATCH_SCAN_AXES`), one output file per axis. Save-only (no plots); windows default to the Wilson intervals unless you override them in `BATCH_SCAN_WINDOWS`. Set `BATCH_SCAN_AXES = ()` to skip.

**§5 Wilson tables** — Independent of §1–§4: builds paper-style benchmark tables for all three channels at both energies, evaluating σ_NNLO at SM and at each κ interval boundary. Displays tables in the notebook and writes combined LaTeX to `Results/Tables/wilson_tables.tex`.

| Section | Main knobs | Output |
|---------|------------|--------|
| Setup | — | imports |
| §1 | `PROCESS`, `ENERGY_TEV`, flags, plot layout | `analysis` |
| §2 | `KAPPA` | spot-check table |
| §3 | `FIXED_KAPPA`, `scan_axis`, scan window | PNGs + optional `.txt` |
| §4 | `FIXED_KAPPA`, `BATCH_SCAN_AXES` | one `.txt` per axis |
| §5 | — (run as-is) | LaTeX tables |

---

## Main options

Key variables (details in the notebook sections above):

| Variable | Section | Role |
|----------|---------|------|
| `PROCESS`, `ENERGY_TEV` | §1 | Channel and collider energy |
| `COMPARE_SIMULATION`, `UNCERTAINTIES_AS_PERCENT` | §1 / §2 | Spot-check table format |
| `SAVE_SCAN_POINTS`, `SAVE_PLOTS` | §1 | Write `.txt` scans and PNG figures |
| `SIGMA_INSET`, `LEGEND_LOC`, `INSET_LOC` | §1 | Plot layout (§3) |
| `KAPPA` | §2 | Spot-check point only (table in §2) |
| `FIXED_KAPPA` | §3, §4 | Full κ tuple; non-scanned components held at these values |
| `scan_axis`, `scan_vmin`, `scan_vmax` | §3 | Which κ to sweep and over what range |
| `BATCH_SCAN_AXES`, `BATCH_SCAN_WINDOWS` | §4 | Multi-axis batch scan |

**Outputs:** `Results/Points/` (scan `.txt`), `Results/Plots/` (figures), `Results/Tables/` (`wilson_tables.tex`).

---

## Wilson coefficients ($\kappa$)

SM: all $\kappa = 1$.

| Symbol | Meaning | $W^\pm HH$ | $ZHH$ |
|--------|---------|:----------:|:-----:|
| $\kappa_\lambda$ | Higgs self-coupling | ✓ | ✓ |
| $\kappa_W$ | $WWH$ coupling | ✓ | — |
| $\kappa_Z$ | $ZZH$ coupling | — | ✓ |
| $\kappa_{2W}$ | $WHHH$ coupling | ✓ | — |
| $\kappa_{2Z}$ | $ZZHH$ coupling | — | ✓ |
| $\kappa_t$ | $tth$ coupling | — | ✓ |

**Tuple order** (must match exactly):

| Process | `KAPPA` (§2) / `FIXED_KAPPA` (§3–§4) |
|---------|-------------------------------------|
| `WplusHH`, `WminusHH` | `(κ_λ, κ_W, κ_2W)` — 3 numbers |
| `ZHH` | `(κ_λ, κ_Z, κ_2Z, κ_t)` — 4 numbers |

**Scan axes** (one coefficient varied per scan; others fixed at `FIXED_KAPPA`):

- $W^\pm HH$: `kappa_lambda`, `kappa_w`, `kappa_2w`
- $ZHH$: `kappa_lambda`, `kappa_z`, `kappa_2z`, `kappa_t`

**95% CL intervals** (Table 2, HEFT Wilson-coefficient uncertainty note; `WILSON_INTERVALS` in `vhh_predict/tables.py`):

| Coefficient | Min | SM | Max |
|-------------|-----|----|-----|
| $\kappa_\lambda$ | −0.7 | 1 | 6.1 |
| $\kappa_W$ | 0.8 | 1 | 1.2 |
| $\kappa_Z$ | 0.9 | 1 | 1.2 |
| $\kappa_{2W}$ | 0.7 | 1 | 1.3 |
| $\kappa_{2Z}$ | 0.7 | 1 | 1.3 |
| $\kappa_t$ | 0.8 | 1 | 1.2 |

Default batch-scan windows (§4) and Wilson benchmark tables (§5) use these bounds; override plot windows with `scan_vmin` / `scan_vmax` in §3.

---

## Repository layout

```
VHH-NNLO/
├── vhh_prediction.ipynb       # start here
├── README.md
├── pyproject.toml             # Python packaging (see below)
├── vhh_predict/               # importable package (predict, scan, plots, tables, …)
├── data/                      # bundled HEFT coefficients (read at runtime)
│   └── {Process}/{13_6TeV|14_0TeV}/
│       ├── pdf_alpha_s_covariance.json
│       ├── scale_coefficients.json
│       ├── {Process}_{energy}_analysis_A.txt   # human reference only
│       └── Simulation/*.out                    # optional simulation compare
└── Results/                   # notebook output (created on first run)
    ├── Points/                # scan tables (.txt)
    ├── Plots/                 # figures (.png)
    └── Tables/                # wilson_tables.tex
```

`Process` is `WplusHH`, `WminusHH`, or `ZHH`. Energies use folder names `13_6TeV` and `14_0TeV`.

---

## Citation & license

Publication **in preparation** — contact the authors for a preprint reference. BibTeX will be added upon release.

See the repository license file (if present).
