# VHH-NNLO

Closed-form predictions for **vector-boson-associated double-Higgs production** ($W^\pm HH$, $ZHH$) at **NNLO QCD**, with PDF + $\alpha_s$ and scale uncertainties. The package supports two EFT frameworks:

| Framework | Expansion | Data | Notebook |
|-----------|-----------|------|----------|
| **HEFT** | $\sigma = \sum_k A_k \kappa_i^n \kappa_j^m$ | `data/HEFT/` | [`vhh_prediction_HEFT.ipynb`](vhh_prediction_HEFT.ipynb) |
| **SMEFT** | $\sigma = \sigma_{\mathrm{SM}} + \sum_i B_i C_i$ | `data/SMEFT/` | [`vhh_prediction_SMEFT.ipynb`](vhh_prediction_SMEFT.ipynb) |

Coefficients and optional simulation reference points are bundled under `data/` — no Monte Carlo or fitting step required. If you only need the fitted **$A_k$** (HEFT) or **$B_i$** (SMEFT) arrays, load the JSON files described under [Contents of `data/`](#contents-of-data) — the Python package is optional.

Accompanies an **in-preparation publication** (*Precise predictions for double Higgs production in association with a vector boson in Effective Field Theory*).

---

## What it does

Both notebooks share the same workflow:

- **Spot check** — $\sigma_{\mathrm{LO}}$, $\sigma_{\mathrm{NNLO}}$, $K$-factor, and enhancement over SM at any EFT point.
- **Single-axis scan** — vary one EFT parameter; optional PDF/scale bands and `.txt` output table.
- **Plot** — two-panel figures: $\sigma_{\mathrm{NNLO}}+K$-factor and $\sigma_{\mathrm{NNLO}}+\sigma^{\mathrm{EFT}}_{\mathrm{NNLO}}/\sigma^{\mathrm{SM}}_{\mathrm{NNLO}}$.
- **Joint multi-axis scan** — scan over several coefficients axes at once; `.txt` output table.
- **Benchmark tables** — interval-boundary tables in `.tex` (HEFT $\kappa$ or SMEFT $C_i$).

Processes: `pp>W+HH`, `pp>W-HH`, `pp>ZHH` at **13.6** or **14.0** TeV.

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

Run all cells top to bottom. Edit **§1** (process, flags), **§2** (EFT point), **§3** (scan + plots), **§4** (joint grid scan), **§5** (benchmark tables).

---

## HEFT ($\kappa$ framework)

**Notebook:** [`vhh_prediction_HEFT.ipynb`](vhh_prediction_HEFT.ipynb)

Outputs under `results/points/heft/<Process>/<energy>/`, `results/plots/heft/<Process>/<energy>/`, and `results/tables/heft/` (single publication `.tex`).

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

**95% CL exclusion intervals for all HEFT $\kappa$** (`WILSON_INTERVALS` in `vhh_predict/tables.py`; also the default scan / §5 table bounds):

| Coefficient | Min | SM | Max |
|-------------|-----|----|-----|
| $\kappa_\lambda$ | −0.70 | 1 | 6.10 |
| $\kappa_W$ | 0.85 | 1 | 1.20 |
| $\kappa_Z$ | 0.90 | 1 | 1.20 |
| $\kappa_{2W}$ | 0.70 | 1 | 1.30 |
| $\kappa_{2Z}$ | 0.70 | 1 | 1.30 |
| $\kappa_t$ | 0.80 | 1 | 1.20 |

---

## SMEFT ($C_i$ framework)

**Notebook:** [`vhh_prediction_SMEFT.ipynb`](vhh_prediction_SMEFT.ipynb)

Outputs under `results/points/smeft/<Process>/<energy>/`, `results/plots/smeft/<Process>/<energy>/`, and `results/tables/smeft/` (single publication `.tex`).

- **$W^\pm HH$:** $B_1$–$B_5$ ↔ $C_\varphi$, $C_{\varphi\square}$, $C_{\varphi D}$, $C_{\varphi q}^{(3)}$, $C_{\varphi W}$
- **$ZHH$ LO:** $B_1$–$B_{10}$
- **$ZHH$ NNLO:** adds $B_{11}$ ($C_{t\varphi}$) and $B_{12}$ ($C_{\varphi t}+C_{\varphi Q}^{(3)}-C_{\varphi Q}^{(1)}$)

SM: all $C_i = 0$. Set **`WCS`** as a dictionary, e.g. `{"phiW": -0.2}`.

**Allowed Wilson-coefficient intervals** (global fit, Ref. [ter Hoeve et al., JHEP 06 (2025) 125](https://arxiv.org/abs/2502.20453), Table E.1; `SMEFT_WC_INTERVALS` in `vhh_predict/smeft_operators.py`):

| Bosonic $C_i$ [TeV$^{-2}$] | Interval |
|----------------------------|----------|
| $C_\varphi$ | [−15, 5] |
| $C_{\varphi W}$ | [−1, 1] |
| $C_{\varphi B}$ | [−0.5, 0.5] |
| $C_{\varphi WB}$ | [−1.5, 1.5] |
| $C_{\varphi D}$ | [−1.5, 1.5] |
| $C_{\varphi\square}$ | [−2, 2] |

| Fermionic $C_i$ [TeV$^{-2}$] | Interval |
|------------------------------|----------|
| $C_{\varphi q}^{(3)}$ | [−0.2, 0.05] |
| $C_{\varphi t}$ | [−25, 34] |
| $C_{\varphi Q}^{(3)}$ | [−8, 2] |
| $C_{\varphi Q}^{(1)}$ | [−6.5, 30.5] |
| $C_{\varphi q}^{(1)}$ | [−3, 1] |
| $C_{\varphi u}$ | [−3.5, 1] |
| $C_{\varphi d}$ | [−4, 4] |
| $C_{t\varphi}$ | [−15, 5] |

---

## Repository layout

```
VHH-NNLO/
├── vhh_prediction_HEFT.ipynb   # HEFT entry point
├── vhh_prediction_SMEFT.ipynb  # SMEFT entry point
├── README.md
├── AGENTS.md
├── pyproject.toml              # pip install -e ".[notebook]"
├── scripts/
│   └── build_smeft_package_data.py   # regenerate data/SMEFT from SMEFT_Results
├── vhh_predict/                # importable package
│   ├── analysis.py             # path helpers, load_analysis() [HEFT]
│   ├── core.py                 # HEFT predict / scan / scan_grid
│   ├── scan_io.py              # HEFT scan_and_save / scan_grid_and_save
│   ├── smeft_*.py              # SMEFT load / predict / scan / simulation
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
└── results/                         # shipped notebook outputs (see below)
    ├── points/{heft,smeft}/{Process}/{13_6TeV|14_0TeV}/
    ├── plots/{heft,smeft}/{Process}/{13_6TeV|14_0TeV}/
    └── tables/{heft,smeft}/         # one publication .tex per framework
```

`Process` is `WplusHH`, `WminusHH`, or `ZHH`. Energies use folder names `13_6TeV` and `14_0TeV`.

### Contents of `data/`

Each process/energy folder holds the closed-form coefficients used by the predictors. You can use these JSON files on their own (any language) without installing `vhh-predict`.

| File | Framework | Role |
|------|-----------|------|
| `pdf_alpha_s_covariance.json` | HEFT & SMEFT | Central coefficients + PDF/$\alpha_s$ uncertainties and covariances |
| `scale_coefficients.json` | HEFT & SMEFT | Same coefficients refitted at each of the 7 $(\mu_R,\mu_F)$ scale points |
| `sigma_sm.json` | SMEFT only | SM cross sections $\sigma_{\mathrm{SM}}$ (also duplicated under `sigma_sm` in the covariance JSON) |
| `*_analysis_A.txt` / `*_analysis_B.txt` | HEFT / SMEFT | Human-readable dump of the same numbers (reference only; runtime code reads JSON) |
| `Simulation/*.out` | optional | Monte Carlo spot-check points |

**HEFT** — $\sigma = \mathbf{m}(\kappa)^\top \mathbf{A}$ (fb). Central vectors are

- `LO.A_central` and `NNLO.A_central` in `pdf_alpha_s_covariance.json`
- length **6** for $W^\pm HH$ (all orders) and for $ZHH$ at LO; **18** for $ZHH$ at NNLO (HHZ)
- companion keys: `delta_pdf`, `delta_alpha_s`, `C_pdf`, `C_pdf_alphaS`, …
- scale variations: `A_LO_by_scale` / `A_NNLO_by_scale` in `scale_coefficients.json`, keyed by `"μR,μF"` strings such as `"1,1"`, `"0.5,0.5"`, …

**SMEFT** — $\sigma = \sigma_{\mathrm{SM}} + \mathbf{C}^\top \mathbf{B}$ with $C_i$ in TeV$^{-2}$ and $B_i$ in fb·TeV$^{2}$. Central vectors are

- `LO.B_central` and `NNLO.B_central` in `pdf_alpha_s_covariance.json`
- `wc_keys` lists the logical WC names in the same order as $\mathbf{B}$ (e.g. `phi`, `phiBox`, …)
- $\sigma_{\mathrm{SM}}$: `sigma_sm.json` (`sigma_sm_lo`, `sigma_sm_nnlo`) or the `sigma_sm` block in the covariance JSON
- scale variations: `B_LO_by_scale` / `B_NNLO_by_scale` and `sigma_sm_*_by_scale` in `scale_coefficients.json`

Minimal example (central $A$ / $B$ only):

```python
import json
from pathlib import Path

# HEFT A at NNLO
heft = json.loads(Path("data/HEFT/WplusHH/13_6TeV/pdf_alpha_s_covariance.json").read_text())
A_nnlo = heft["NNLO"]["A_central"]  # list[float]

# SMEFT B and σ_SM at NNLO
smeft = json.loads(Path("data/SMEFT/WplusHH/13_6TeV/pdf_alpha_s_covariance.json").read_text())
B_nnlo = smeft["NNLO"]["B_central"]
wc_keys = smeft["NNLO"]["wc_keys"]
sigma_sm_nnlo = smeft["sigma_sm"]["NNLO"]
```

For full predictions with uncertainties, prefer `load_analysis()` / `load_smeft_analysis()` and `predict()` from the package.

### Contents of `results/`

 Scan points, plots, and tables used for publication **in preparation** produced by the notebooks are part of the repository (empty channel/energy folders may still use `.gitkeep` until filled).

| Subtree | Path | Contents |
|---------|------|----------|
| **`points/`** | `results/points/{heft\|smeft}/<Process>/<energy>/` | Scan `.txt` tables from §3–§4 |
| **`plots/`** | `results/plots/{heft\|smeft}/<Process>/<energy>/` | Scan PNGs (e.g. `…_sigma_nnlo_and_K_nnlo.png`, `…_sigma_nnlo_and_EFT_enhancement.png`) |
| **`tables/`** | `results/tables/{heft\|smeft}/` | One publication `.tex` per framework |

**HEFT** (`results/tables/heft/heft_publication_tables.tex`): single file with all HEFT benchmark tables. Each table evaluates $\sigma$ at **both interval boundaries** (min and max) for every scan-axis $\kappa$, with other $\kappa$ fixed to SM ($=1$), at 13.6 and 14.0 TeV.

**SMEFT** (`results/tables/smeft/smeft_publication_tables.tex`): single file with all SMEFT benchmark tables. For each WC, **only the interval endpoint that maximises** $\sigma_{\mathrm{NNLO}}$ is shown (other $C_i=0$), at 13.6 and 14.0 TeV.

Re-running the notebooks overwrites the corresponding files under these paths.

---

## Citation & license

Publication **in preparation** — contact the authors for a preprint reference.
