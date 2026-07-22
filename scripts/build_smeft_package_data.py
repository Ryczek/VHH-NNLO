#!/usr/bin/env python3
"""Build SMEFT data/ bundle from SMEFT_Results in the main repository.

Run from the VHH-NNLO package root (or pass --repo-root):

    python scripts/build_smeft_package_data.py --repo-root ../../..

Writes under data/SMEFT/{Process}/{13_6TeV|14_0TeV}/:
  - pdf_alpha_s_covariance.json
  - scale_coefficients.json
  - sigma_sm.json
  - {Process}_{energy}_analysis_B.txt
  - Simulation/*.out  (from SMEFT_Results/.../full/)
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Main-repo analysis tools (not shipped with the release package).
_SCRIPT = Path(__file__).resolve()
_PKG_ROOT = _SCRIPT.parents[1]


def _add_analysis_tools(repo_root: Path) -> None:
    tools = repo_root / "analysis_tools"
    if not tools.is_dir():
        raise FileNotFoundError(f"analysis_tools not found under {repo_root}")
    if str(tools) not in sys.path:
        sys.path.insert(0, str(tools))


def _energy_tag(energy_tev: float) -> str:
    if abs(energy_tev - 13.6) < 0.05:
        return "13_6TeV"
    if abs(energy_tev - 14.0) < 0.05:
        return "14_0TeV"
    raise ValueError(f"Unsupported energy {energy_tev}")


def _scale_key(fs: float, rs: float) -> str:
    return f"{fs:g},{rs:g}"


def _diag_covariance(deltas: Sequence[float]) -> List[List[float]]:
    n = len(deltas)
    out = [[0.0] * n for _ in range(n)]
    for i, d in enumerate(deltas):
        out[i][i] = float(d) ** 2
    return out


def _extract_b_at_scale(analyze_mod, sm_data, wc_data, wc, channel, quantity, fs, rs):
    from analyze_out_uncertainties import get_entry

    try:
        sm = get_entry(sm_data, 0, fs, rs)
        pt = get_entry(wc_data, 0, fs, rs)
    except KeyError:
        return None
    return analyze_mod._safe_extract_b(sm, pt, wc, channel, quantity)


def _sigma_sm_at_scale(analyze_mod, sm_data, channel, quantity, fs, rs):
    from analyze_out_uncertainties import get_entry

    try:
        sm = get_entry(sm_data, 0, fs, rs)
    except KeyError:
        return None
    try:
        return analyze_mod.get_sigma(sm, channel, quantity)
    except KeyError:
        return None


def export_channel_energy(
    *,
    smeft_src: Path,
    dest: Path,
    analyze_mod,
    naming_mod,
) -> None:
    from analyze_B_coefficients import analyze_channel_energy, find_sm_file, find_uncert_root
    from analyze_out_uncertainties import get_entry, parse_out_file

    channel = smeft_src.parent.name
    energy_tag = smeft_src.name

    report, analysis_txt = analyze_channel_energy(smeft_src)
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(analysis_txt, dest / analysis_txt.name)

    uncert_root = find_uncert_root(smeft_src)
    specs = naming_mod.b_specs_for_channel(channel)
    sm_path = find_sm_file(uncert_root)
    if sm_path is None:
        raise FileNotFoundError(f"No SM .out under {uncert_root}")
    sm_data = parse_out_file(sm_path)
    sm0 = get_entry(sm_data, 0, 1.0, 1.0)
    sigma_sm_lo = analyze_mod.get_sigma(sm0, channel, "LO")
    if channel == "ZHH":
        sigma_sm_nn = analyze_mod.get_sigma(sm0, channel, "NNLO")
    else:
        sigma_sm_nn = analyze_mod.get_sigma(sm0, channel, "NNLO")

    # Re-run per-spec extraction to collect structured results
    lo_results = []
    nn_results = []
    loop_root = smeft_src / "loop"
    for spec in specs:
        if spec.lo:
            r = analyze_mod.analyze_one_b(spec, uncert_root, channel, "LO")
            if r is None and loop_root.is_dir():
                r = analyze_mod.analyze_one_b(spec, loop_root, channel, "LO")
            if r is not None:
                lo_results.append(r)
        if spec.nnlo:
            r = analyze_mod.analyze_one_b(spec, uncert_root, channel, "NNLO")
            if r is None and loop_root.is_dir():
                r = analyze_mod.analyze_one_b(spec, loop_root, channel, "NNLO")
            if r is not None:
                nn_results.append(r)

    def _dedupe(rows):
        seen = {}
        for r in rows:
            seen.setdefault(r.name, r)
        order = [s.name for s in specs]
        return [seen[n] for n in order if n in seen]

    lo_results = _dedupe(lo_results)
    nn_results = _dedupe(nn_results)

    wc_keys = [r.op.key for r in lo_results] if lo_results else [r.op.key for r in nn_results]
    energy_tev = float(energy_tag.removesuffix("TeV").replace("_", "."))

    def _order_payload(results: List, order_label: str) -> dict:
        labels = [r.name for r in results]
        n = len(labels)
        central = [r.central for r in results]
        delta_pdf = [r.pdf for r in results]
        delta_alpha = [r.alpha for r in results]
        delta_pdfas = [r.pdf_alpha for r in results]
        return {
            "labels": labels,
            "wc_keys": [r.op.key for r in results],
            "fortran_names": [r.op.fortran for r in results],
            "B_central": central,
            "delta_pdf": delta_pdf,
            "delta_alpha_s": delta_alpha,
            "delta_pdf_alpha_s": delta_pdfas,
            "C_pdf": _diag_covariance(delta_pdf),
            "C_alphaS": _diag_covariance(delta_alpha),
            "C_pdf_alphaS": _diag_covariance(delta_pdfas),
        }

    cov_payload = {
        "version": 1,
        "framework": "SMEFT",
        "process": channel,
        "energy_tev": energy_tev,
        "method": (
            "B_i = (σ(C_i)−σ_SM)/C_i from single-operator scans; "
            "PDF: replica envelope; alpha_s: NSET 41/42; "
            "C matrices diagonal (one scan per B_i)"
        ),
        "sigma_sm": {"LO": sigma_sm_lo, "NNLO": sigma_sm_nn},
        "LO": _order_payload(lo_results, "LO"),
        "NNLO": _order_payload(nn_results, "NNLO" if channel != "ZHH" else "HHZ"),
    }
    (dest / "pdf_alpha_s_covariance.json").write_text(
        json.dumps(cov_payload, indent=2), encoding="utf-8"
    )

    # Scale-refitted B_i and σ_SM at each (μ_R, μ_F)
    lo_by_scale: Dict[str, List[float]] = {}
    nn_by_scale: Dict[str, List[float]] = {}
    sm_lo_by_scale: Dict[str, float] = {}
    sm_nn_by_scale: Dict[str, float] = {}

    for fs, rs in analyze_mod.SEVEN_POINT_SCALES:
        sk = _scale_key(fs, rs)
        sm_lo = _sigma_sm_at_scale(analyze_mod, sm_data, channel, "LO", fs, rs)
        sm_nn = _sigma_sm_at_scale(analyze_mod, sm_data, channel, "NNLO", fs, rs)
        if sm_lo is not None:
            sm_lo_by_scale[sk] = sm_lo
        if sm_nn is not None:
            sm_nn_by_scale[sk] = sm_nn

        lo_vec: List[float] = []
        for r in lo_results:
            wc_path = uncert_root
            for cand in naming_mod.disk_subdir_candidates(r.op.key):
                p = uncert_root / cand
                if p.is_dir():
                    wc_path = p
                    break
            files = analyze_mod.collect_scan_points(wc_path, r.op.fortran)
            wc_file = files.get(r.ref_wc)
            if wc_file is None:
                lo_vec.append(float("nan"))
                continue
            wc_data = parse_out_file(wc_file)
            b = _extract_b_at_scale(
                analyze_mod, sm_data, wc_data, r.ref_wc, channel, "LO", fs, rs
            )
            lo_vec.append(b if b is not None else float("nan"))
        lo_by_scale[sk] = lo_vec

        nn_vec: List[float] = []
        for r in nn_results:
            wc_path = uncert_root
            for cand in naming_mod.disk_subdir_candidates(r.op.key):
                p = uncert_root / cand
                if p.is_dir():
                    wc_path = p
                    break
            files = analyze_mod.collect_scan_points(wc_path, r.op.fortran)
            wc_file = files.get(r.ref_wc)
            if wc_file is None:
                nn_vec.append(float("nan"))
                continue
            wc_data = parse_out_file(wc_file)
            b = _extract_b_at_scale(
                analyze_mod, sm_data, wc_data, r.ref_wc, channel, "NNLO", fs, rs
            )
            nn_vec.append(b if b is not None else float("nan"))
        nn_by_scale[sk] = nn_vec

    scale_payload = {
        "version": 1,
        "framework": "SMEFT",
        "method": "7-point B_i refit at NSET=0",
        "wc_keys_lo": [r.op.key for r in lo_results],
        "wc_keys_nnlo": [r.op.key for r in nn_results],
        "sigma_sm_lo_by_scale": sm_lo_by_scale,
        "sigma_sm_nnlo_by_scale": sm_nn_by_scale,
        "B_LO_by_scale": lo_by_scale,
        "B_NNLO_by_scale": nn_by_scale,
    }
    (dest / "scale_coefficients.json").write_text(
        json.dumps(scale_payload, indent=2), encoding="utf-8"
    )

    sigma_payload = {
        "version": 1,
        "process": channel,
        "energy_tev": energy_tev,
        "sigma_sm_lo": sigma_sm_lo,
        "sigma_sm_nnlo": sigma_sm_nn,
        "sm_reference_file": sm_path.name,
    }
    (dest / "sigma_sm.json").write_text(json.dumps(sigma_payload, indent=2), encoding="utf-8")

    # Simulation: copy all .out from full/ then loop/ (skip .parallel_jobs).
    # Prefer *_Corrected.out over a plain sibling; write without "_Corrected".
    sim_dest = dest / "Simulation"
    if sim_dest.exists():
        shutil.rmtree(sim_dest)
    sim_dest.mkdir(parents=True)
    by_dest_name: Dict[str, Path] = {}
    for sub in ("full", "loop"):
        src_root = smeft_src / sub
        if not src_root.is_dir():
            continue
        for out in sorted(src_root.rglob("*.out")):
            if ".parallel_jobs" in out.parts:
                continue
            dest_name = out.name.replace("_Corrected", "")
            prev = by_dest_name.get(dest_name)
            if prev is None:
                by_dest_name[dest_name] = out
                continue
            # Prefer Corrected source when both map to the same dest name.
            if "_Corrected" in out.name and "_Corrected" not in prev.name:
                by_dest_name[dest_name] = out
    copied = 0
    for dest_name, src in sorted(by_dest_name.items()):
        shutil.copy2(src, sim_dest / dest_name)
        copied += 1
    print(f"  {dest.relative_to(_PKG_ROOT)}: {copied} simulation .out files")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=_PKG_ROOT.parents[1],
        help="Main pphhV repository root (contains SMEFT_Results/)",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=_PKG_ROOT / "data" / "SMEFT",
        help="Output SMEFT data root",
    )
    args = parser.parse_args(argv)

    repo_root = args.repo_root.resolve()
    smeft_root = repo_root / "SMEFT_Results"
    if not smeft_root.is_dir():
        print(f"SMEFT_Results not found at {smeft_root}", file=sys.stderr)
        return 1

    _add_analysis_tools(repo_root)
    import analyze_B_coefficients as analyze_mod  # noqa: E402
    import smeft_wc_naming as naming_mod  # noqa: E402

    targets = []
    for ch in ("WplusHH", "WminusHH", "ZHH"):
        for en in ("13_6TeV", "14_0TeV"):
            p = smeft_root / ch / en
            if p.is_dir():
                targets.append((p, args.dest / ch / en))

    if not targets:
        print("No SMEFT_Results channel/energy directories found.", file=sys.stderr)
        return 1

    for src, dest in targets:
        print(f"Building {dest.relative_to(_PKG_ROOT)} from {src} ...")
        export_channel_energy(
            smeft_src=src,
            dest=dest,
            analyze_mod=analyze_mod,
            naming_mod=naming_mod,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
