#!/usr/bin/env python3
"""Tie-point network + RANSAC geometric registration demo (synthetic).

Builds an L8/L9 image pair related by a known affine displacement field, with
injected cloud/smoke outlier patches, then recovers the field with a tie-point
network + RANSAC and validates against ground truth.

    python scripts/run_tiepoints.py
    python scripts/run_tiepoints.py --grid 8 --chip 48 --outlier-frac 0.2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from landsat_xcal import data, tiepoints, geometric  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=256, help="image size (px)")
    p.add_argument("--grid", type=int, default=8, help="NxN grid of chips")
    p.add_argument("--chip", type=int, default=64, help="chip size (px)")
    p.add_argument("--outlier-frac", type=float, default=0.2)
    p.add_argument("--threshold-px", type=float, default=0.4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-figures", action="store_true")
    p.add_argument("--outdir", default=str(Path(__file__).resolve().parents[1] / "figures"))
    args = p.parse_args()

    ref, mov, field = data.synthetic_warped_pair(
        n=args.n, outlier_frac=args.outlier_frac, seed=args.seed)
    print(f"Image: {args.n}x{args.n}px | injected affine field: "
          f"tx={field['tx']}, ty={field['ty']}, gx={field['gx']}, gy={field['gy']}")

    result = tiepoints.register_tiepoints(
        ref, mov, grid=(args.grid, args.grid), chip=args.chip,
        threshold_px=args.threshold_px, seed=args.seed)

    print(f"\nTie points: {result.n_tiepoints} | inliers: {result.n_inliers} "
          f"| outliers rejected: {result.n_tiepoints - result.n_inliers}")
    print(f"Inlier residual RMSE: {result.rmse_px:.3f} px "
          f"({result.rmse_m:.2f} m at {geometric.PIXEL_SIZE_M:.0f} m/px)")

    # --- validate the fitted field against ground truth at sample points ---- #
    print("\nValidation (recovered vs injected displacement):")
    for (c, r, label) in [(args.n // 2, args.n // 2, "centre"),
                          (args.n // 5, args.n // 5, "corner")]:
        tdy, tdx = data._true_displacement(c, r, (args.n, args.n), field)
        pdy, pdx = result.model.predict(c, r)
        print(f"  {label:7s} recovered (dy,dx)=({float(pdy):+.3f},{float(pdx):+.3f})  "
              f"injected ({float(tdy):+.3f},{float(tdx):+.3f}) px")

    if not args.no_figures:
        from landsat_xcal import plotting
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        fig = plotting.plot_tiepoint_field(result, image_shape=(args.n, args.n))
        fig.savefig(outdir / "tiepoint_field.png", dpi=130)
        print(f"\nFigure written to {outdir / 'tiepoint_field.png'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
