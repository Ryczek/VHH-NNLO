# VHH-NNLO

Closed-form predictions for **vector-boson-associated double-Higgs production** ($W^\pm HH$, $ZHH$) at **LO** and **NNLO QCD**, with PDF + $\alpha_s$ and scale uncertainties. The package supports two EFT frameworks:

| Framework | Expansion | Data | Notebook |
|-----------|-----------|------|----------|
| **HEFT** | $\sigma = \mathbf{m}(\kappa)^\top \mathbf{A}$ | `data/HEFT/` | [`vhh_prediction_HEFT.ipynb`](vhh_prediction_HEFT.ipynb) |
| **SMEFT** | $\sigma = \sigma_{\mathrm{SM}} + \sum_i B_i C_i$ | `data/SMEFT/` | [`vhh_prediction_SMEFT.ipynb`](vhh_prediction_SMEFT.ipynb) |

Coefficients and optional simulation reference points are bundled under `data/` — no Monte Carlo or fitting step required.

Accompanies an **in-preparation publication** (*Precise predictions for double Higgs production in association with a vector boson in Effective Field Theory*). Citation to be added when the paper is public.

---

## What it does

Both notebooks share the same workflow:

- **Spot check** — $\sigma_{\mathrm{LO}}$, $\sigma_{\mathrm{NNLO}}$ (HHZ for $ZHH$), $K$-factor, and enhancement over SM at any EFT point.
- **Scan** — vary one Wilson parameter; optional PDF/scale bands and `.txt` tables.
- **Plot** — two-panel figures: $\sigma_{\mathrm{NNLO}}+K$ and $\sigma/\sigma_{\mathrm{SM}}$.
- **Compare** — optional check against bundled simulation `.out` files (spot check only).
- **Benchmark tables** — interval-boundary tables (HEFT $\kappa$ or SMEFT $C_i$).

Processes: `WplusHH`, `WminusHH`, `ZHH` at **13.6** or **14.0** TeV.

---

## Quick start

From the **repository root**:

```bash
git clone <repo-url>
cd VHH-NNLO
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[notebook]"
jupyter notebook vhh_prediction_HEFT.ipynb    # HEFT (κ)
# or
jupyter notebook vhh_prediction_SMEFT.ipynb   # SMEFT (C_i)
```

Run all cells top to bottom. Edit **§1** (process, flags), **§2** (EFT point), **§3** (scan + plots).

> Always start Jupyter from the repo root — paths to `data/HEFT/`, `data/SMEFT/`, and `Results/` are relative to it.

---

## HEFT ($\kappa$ framework)

**Notebook:** [`vhh_prediction_HEFT.ipynb`](vhh_prediction_HEFT.ipynb)

SM: all $\kappa = 1$.

| Symbol | Meaning | $W^\pm HH$ | $ZHH$ |
|--------|---------|:----------:|:-----:|
| $\kappa_\lambda$ | Higgs self-coupling | ✓ | ✓ |
| $\kappa_W$ | $WWH$ coupling | ✓ | — |
| $\kappa_Z$ | $ZZH$ coupling | — | ✓ |
| $\kappa_{2W}$ | $WHHH$ coupling | ✓ | — |
| $\kappa_{2Z}$ | $ZZHH$ coupling | — | ✓ |
| $\kappa_t$ | $tth$ coupling | — | ✓ |

**Tuple order:**

| Process | `KAPPA` / `FIXED_KAPPA` |
|---------|-------------------------|
| `WplusHH`, `WminusHH` | `(κ_λ, κ_W, κ_{2W})` |
| `ZHH` | `(κ_λ, κ_Z, κ_{2Z}, κ_t)` |

**95% CL intervals** (`WILSON_INTERVALS` in `vhh_predict/tables.py`):

| Coefficient | Min | SM | Max |
|-------------|-----|----|-----|
| $\kappa_\lambda$ | −0.7 | 1 | 6.1 |
| $\kappa_W$ | 0.8 | 1 | 1.2 |
| $\kappa_Z$ | 0.9 | 1 | 1.2 |
| $\kappa_{2W}$ | 0.7 | 1 | 1.3 |
| $\kappa_{2Z}$ | 0.7 | 1 | 1.3 |
| $\kappa_t$ | 0.8 | 1 | 1.2 |

---

## SMEFT ($C_i$ framework)

**Notebook:** [`vhh_prediction_SMEFT.ipynb`](vhh_prediction_SMEFT.ipynb)

Linear expansion with Wilson coefficients $C_i$ in $\mathrm{TeV}^{-2}$ and $B_i$ in $\mathrm{fb\,TeV}^2$:

- **$W^\pm HH$:** $B_1$–$B_5$ ↔ $C_\varphi$, $C_{\varphi\square}$, $C_{\varphi D}$, $C_{\varphi q}^{(3)}$, $C_{\varphi W}$
- **$ZHH$ LO:** $B_1$–$B_{10}$ (eq. 80 in the paper)
- **$ZHH$ NNLO:** adds $B_{11}$ ($C_{t\varphi}$) and $B_{12}$ ($C_{\varphi t}+C_{\varphi Q}^{(3)}-C_{\varphi Q}^{(1)}$)

SM: all $C_i = 0$. Set **`WCS`** as a dictionary, e.g. `{"phiW": -0.2}`.

**Allowed Wilson-coefficient intervals** (global fit, Ref. terHoeve:2025gey; `SMEFT_WC_INTERVALS` in `vhh_predict/smeft_operators.py`):

| Bosonic $C_i$ [TeV$^{-2}$] | Interval |
|----------------------------|----------|
| $C_\varphi$ | [−15, 5] |
| $C_{\varphi W}$ | [−1, 1] |
| $C_{\varphi B}$ | [−0.5, 0.5] |
| $C_{\varphi WB}$ | [−1.5, 1.5] |
| $C_{\varphi D}$ | [−2, 2] |
| $C_{\varphi\square}$ | [−1.5, 1.5] |

| Fermionic $C_i$ [TeV$^{-2}$] | Interval |
|------------------------------|----------|
| $C_{\varphi q}^{(3)}$ | [−0.2, 0.05] |
| $C_{\varphi t}+C_{\varphi Q}^{(3)}-C_{\varphi Q}^{(1)}$ | [−8, 2] |
| $C_{\varphi q}^{(1)}$ | [−3, 1] |
| $C_{\varphi Q}^{(1)}$ | [−6.1, 30.5] |
| $C_{\varphi u}$ | [−3.5, 1] |
| $C_{\varphi d}$ | [−4, 4] |
| $C_{t\varphi}$ | [−15, 5] |

> **$C_{t\varphi}$ ≠ $C_{\varphi t}$:** $C_{t\varphi}$ ($\mathcal{O}_{u\varphi}$, Fortran `cth`) enters $B_{11}$; $C_{\varphi t}$ enters the $B_{12}$ combination.

---

## Repository layout

```
VHH-NNLO/
├── vhh_prediction_HEFT.ipynb   # HEFT entry point
├── vhh_prediction_SMEFT.ipynb  # SMEFT entry point
├── README.md
├── AGENTS.md
├── pyproject.toml
├── scripts/
│   └── build_smeft_package_data.py   # regenerate data/SMEFT from SMEFT_Results
├── vhh_predict/                # importable package
│   ├── analysis.py             # path helpers, load_analysis() [HEFT]
│   ├── core.py                 # HEFT predict/scan
│   ├── smeft_*.py              # SMEFT load/predict/scan/simulation
│   └── ...
├── data/
│   ├── HEFT/{Process}/{13_6TeV|14_0TeV}/
│   │   ├── pdf_alpha_s_covariance.json
│   │   ├── scale_coefficients.json
│   │   ├── {Process}_{energy}_analysis_A.txt
│   │   └── Simulation/*.out
│   └── SMEFT/{Process}/{13_6TeV|14_0TeV}/
│       ├── pdf_alpha_s_covariance.json
│       ├── scale_coefficients.json
│       ├── sigma_sm.json
│       ├── {Process}_{energy}_analysis_B.txt
│       └── Simulation/*.out
└── Results/                    # notebook output (created on first run)
    ├── Points/                 # HEFT scan tables
    │   └── SMEFT/              # SMEFT scan tables
    ├── Plots/
    │   └── SMEFT/
    └── Tables/
        └── SMEFT/
```

`Process` is `WplusHH`, `WminusHH`, or `ZHH`. Energies use folder names `13_6TeV` and `14_0TeV`.

Human-readable `*_analysis_A.txt` / `*_analysis_B.txt` files are reference only; the code reads JSON at runtime.

---

## Regenerating SMEFT data

From the main analysis repository (with `SMEFT_Results/` present):

```bash
cd VHH-NNLO
python scripts/build_smeft_package_data.py --repo-root /path/to/pphhV
```

This rebuilds `data/SMEFT/` including B coefficients, uncertainties, and simulation `.out` files from `full/` scans.

---

## Citation & license

Publication **in preparation** — contact the authors for a preprint reference. BibTeX will be added upon release.

See the repository license file (if present).
