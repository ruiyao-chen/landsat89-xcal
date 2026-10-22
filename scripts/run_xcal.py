#!/usr/bin/env python3
"""Run the Landsat 8/9 cross-calibration end to end and write figures + a report.

Examples
--------
# Offline, synthetic data (no Earth Engine needed) - great first run:
python scripts/run_xcal.py --backend synthetic --site Libya4

# Real data from Google Earth Engine (after `earthengine authenticate`):
python scripts/run_xcal.py --backend gee --site Libya4 --project my-ee-project
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# make src/ importable without installing
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from landsat_xcal import config, data, radiometric, uncertainty, geometric  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", choices=["synthetic", "gee"], default="synthetic")
    p.add_argument("--site", default=config.DEFAULT_SITE, choices=list(config.SITES))
    p.add_argument("--n-pairs", type=int, default=6, help="synthetic only")
    p.add_argument("--project", default=None, help="Earth Engine project id (gee backend)")
    p.add_argument("--seed", type=int, default=0, help="synthetic RNG seed")
    p.add_argument("--no-figures", action="store_true")
    p.add_argument("--outdir", default=str(Path(__file__).resolve().parents[1] / "figures"))
    args = p.parse_args()

    site = config.SITES[args.site]
    print(f"Site: {site.name}  ({site.lon:.2f}, {site.lat:.2f})  ROI +/-{site.roi_km} km")
    print(f"Backend: {args.backend}")

    pairs = data.get_scene_pairs(args.site, backend=args.backend,
                                 n_pairs=args.n_pairs, project=args.project)
    print(f"Scene pairs: {len(pairs)}")
    for pr in pairs:
        print(f"  L8 {pr.l8_date}  /  L9 {pr.l9_date}  (gap {pr.gap_days} d)")

    samples = [
        data.extract_roi_samples(pr, backend=args.backend, project=args.project,
                                 seed=(args.seed + i) if args.backend == "synthetic" else None)
        for i, pr in enumerate(pairs)
    ]

    # --- radiometric + uncertainty ---------------------------------------- #
    results = radiometric.cross_calibrate(samples)
    budget = uncertainty.budget(results)

    print("\nRadiometric cross-calibration (L9 - L8):")
    print(f"  {'band':10s} {'bias':>12s} {'k=2 unc':>12s}  significant?")
    for u in budget:
        if u.kind == "reflective":
            print(f"  {u.label:10s} {u.bias*100:>+10.3f}% {u.u_expanded_k2*100:>10.3f}%"
                  f"   {'YES' if u.significant else 'no'}")
        else:
            print(f"  {u.label:10s} {u.bias:>+10.3f}K {u.u_expanded_k2:>10.3f}K"
                  f"   {'YES' if u.significant else 'no'}")

    # --- geometric -------------------------------------------------------- #
    regs = geometric.register_pairs(samples)
    if regs:
        import numpy as np
        mag = np.array([r.magnitude_px for r in regs])
        print("\nGeometric registration (L9 -> L8):")
        print(f"  mean offset: {mag.mean():.3f} px  ({mag.mean()*geometric.PIXEL_SIZE_M:.2f} m)")
        print(f"  max  offset: {mag.max():.3f} px  ({mag.max()*geometric.PIXEL_SIZE_M:.2f} m)")

    # --- figures ---------------------------------------------------------- #
    if not args.no_figures:
        from landsat_xcal import plotting
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        plotting.plot_reflective_bias(budget).savefig(outdir / "reflective_bias.png", dpi=130)
        plotting.plot_thermal_bias(budget).savefig(outdir / "thermal_bias.png", dpi=130)
        ref0 = next(u for u in budget if u.kind == "reflective")
        plotting.plot_uncertainty_components(ref0).savefig(outdir / "uncertainty_components.png", dpi=130)
        if regs:
            plotting.plot_geometric(regs).savefig(outdir / "geometric_offset.png", dpi=130)
        print(f"\nFigures written to {outdir}")

    # --- synthetic validation against injected ground truth --------------- #
    if args.backend == "synthetic":
        print("\nSynthetic validation (recovered vs injected):")
        truth = samples[0].truth_bias
        for u in budget:
            t = truth.get(u.band_key)
            if t is None:
                continue
            if u.kind == "reflective":
                print(f"  {u.label:10s} recovered {u.bias*100:>+7.3f}%  injected {t*100:>+7.3f}%")
            else:
                print(f"  {u.label:10s} recovered {u.bias:>+7.3f}K  injected {t:>+7.3f}K")
        if regs:
            gdy, gdx = truth["_geo_shift_px"]
            print(f"  geometry   recovered (dy,dx)=({regs[0].dy_px:+.3f},{regs[0].dx_px:+.3f}) px"
                  f"  injected ({gdy:+.3f},{gdx:+.3f}) px")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
