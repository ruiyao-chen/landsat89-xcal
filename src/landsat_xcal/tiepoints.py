"""Tie-point network + RANSAC geometric registration.

A step up from the single global shift in ``geometric.py``: instead of one
rigid translation for the whole scene, we build a *network* of local tie points
and fit a spatial transform to them, rejecting mismatches (cloud, smoke, shadow)
with RANSAC. This is the area-based analog of a GCP/tie-point workflow:

    1. Lay a grid of chips over the overlap.
    2. For each chip, estimate the local sub-pixel displacement of the moving
       (L9) image relative to the reference (L8) by phase cross-correlation
       (reusing geometric._phase_cross_correlation).
    3. Fit a first-order (affine) model of the displacement field by RANSAC,
       which discards tie points whose local match disagrees with the global
       geometry (the outliers you'd get over clouds or smoke plumes).
    4. Report the fitted field, inlier residual RMSE (pixels and metres), and the
       inlier/outlier labelling.

The affine displacement field captures a translation plus the small
scale/rotation/keystone differences between two sensors' line-of-sight models -
the terms a rigid single-shift cannot. Full orthorectification (DEM + orbit /
attitude + RPC) and local thin-plate-spline warping are the next tier and are
out of scope here; this is the tie-point/warp-estimation and accuracy-assessment
layer that such a pipeline is validated against.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometric import _phase_cross_correlation, PIXEL_SIZE_M


@dataclass
class TiePoint:
    row: float          # chip-centre row in the reference image (px)
    col: float          # chip-centre col in the reference image (px)
    dy: float           # local displacement of L9 relative to L8 (row, px)
    dx: float           # local displacement (col, px)
    response: float     # phase-correlation peak quality


@dataclass
class AffineDisplacement:
    """First-order (affine) model of the displacement field.

        dx(col,row) = ax0 + ax1*col + ax2*row
        dy(col,row) = ay0 + ay1*col + ay2*row
    """
    ax: np.ndarray      # (3,) coefficients for dx
    ay: np.ndarray      # (3,) coefficients for dy

    def predict(self, col, row):
        col = np.asarray(col, dtype=float)
        row = np.asarray(row, dtype=float)
        dx = self.ax[0] + self.ax[1] * col + self.ax[2] * row
        dy = self.ay[0] + self.ay[1] * col + self.ay[2] * row
        return dy, dx


@dataclass
class TiePointResult:
    tiepoints: list[TiePoint]
    model: AffineDisplacement
    inliers: np.ndarray             # bool mask over tiepoints
    rmse_px: float                  # inlier residual RMSE (px)
    rmse_m: float                   # inlier residual RMSE (m)

    @property
    def n_tiepoints(self) -> int:
        return len(self.tiepoints)

    @property
    def n_inliers(self) -> int:
        return int(self.inliers.sum())


# --------------------------------------------------------------------------- #
# 1-2. Tie-point extraction
# --------------------------------------------------------------------------- #
def extract_tiepoints(
    ref_img: np.ndarray,
    mov_img: np.ndarray,
    grid: tuple[int, int] = (6, 6),
    chip: int = 48,
    min_response: float = 0.0,
) -> list[TiePoint]:
    """Lay a grid of chips and estimate a local displacement at each.

    ``grid`` is (n_rows, n_cols) of chips; ``chip`` is the square chip size in
    pixels. Each chip is phase-correlated to get a sub-pixel (dy, dx).
    """
    H, W = ref_img.shape
    half = chip // 2
    rows = np.linspace(half, H - half, grid[0]).astype(int)
    cols = np.linspace(half, W - half, grid[1]).astype(int)

    tps: list[TiePoint] = []
    for r in rows:
        for c in cols:
            ref_chip = ref_img[r - half:r + half, c - half:c + half]
            mov_chip = mov_img[r - half:r + half, c - half:c + half]
            if ref_chip.shape != (chip, chip):
                continue
            dy, dx, resp = _phase_cross_correlation(ref_chip, mov_chip)
            if resp >= min_response:
                tps.append(TiePoint(row=float(r), col=float(c),
                                    dy=dy, dx=dx, response=resp))
    return tps


# --------------------------------------------------------------------------- #
# 3. RANSAC affine fit of the displacement field
# --------------------------------------------------------------------------- #
def _solve_affine(cols, rows, vals):
    """Least-squares plane fit: vals ~ a0 + a1*col + a2*row."""
    A = np.column_stack([np.ones_like(cols), cols, rows])
    coef, *_ = np.linalg.lstsq(A, vals, rcond=None)
    return coef


def fit_affine_ransac(
    tiepoints: list[TiePoint],
    threshold_px: float = 0.5,
    iters: int = 500,
    seed: int = 0,
) -> tuple[AffineDisplacement, np.ndarray]:
    """Fit an affine displacement field, rejecting outliers with RANSAC.

    Returns the refit model (on inliers) and a boolean inlier mask.
    """
    rng = np.random.default_rng(seed)
    cols = np.array([t.col for t in tiepoints], dtype=float)
    rows = np.array([t.row for t in tiepoints], dtype=float)
    dxs = np.array([t.dx for t in tiepoints], dtype=float)
    dys = np.array([t.dy for t in tiepoints], dtype=float)
    n = len(tiepoints)

    if n < 3:
        ax = _solve_affine(cols, rows, dxs) if n else np.zeros(3)
        ay = _solve_affine(cols, rows, dys) if n else np.zeros(3)
        return AffineDisplacement(ax, ay), np.ones(n, dtype=bool)

    best_inliers = np.zeros(n, dtype=bool)
    best_count = -1
    idx_all = np.arange(n)
    for _ in range(iters):
        sample = rng.choice(idx_all, size=3, replace=False)
        cs, rs = cols[sample], rows[sample]
        # need non-collinear sample for a well-posed plane
        if np.linalg.matrix_rank(np.column_stack([np.ones(3), cs, rs])) < 3:
            continue
        ax = _solve_affine(cs, rs, dxs[sample])
        ay = _solve_affine(cs, rs, dys[sample])
        model = AffineDisplacement(ax, ay)
        pdy, pdx = model.predict(cols, rows)
        resid = np.hypot(dxs - pdx, dys - pdy)
        inliers = resid < threshold_px
        if inliers.sum() > best_count:
            best_count = int(inliers.sum())
            best_inliers = inliers

    # refit on the best inlier set
    if best_inliers.sum() >= 3:
        ax = _solve_affine(cols[best_inliers], rows[best_inliers], dxs[best_inliers])
        ay = _solve_affine(cols[best_inliers], rows[best_inliers], dys[best_inliers])
    else:
        ax = _solve_affine(cols, rows, dxs)
        ay = _solve_affine(cols, rows, dys)
        best_inliers = np.ones(n, dtype=bool)
    return AffineDisplacement(ax, ay), best_inliers


def register_tiepoints(
    ref_img: np.ndarray,
    mov_img: np.ndarray,
    grid: tuple[int, int] = (8, 8),
    chip: int = 64,
    threshold_px: float = 0.4,
    pixel_size_m: float = PIXEL_SIZE_M,
    seed: int = 0,
) -> TiePointResult:
    """Full tie-point + RANSAC registration of ``mov_img`` (L9) onto ``ref_img`` (L8)."""
    tps = extract_tiepoints(ref_img, mov_img, grid=grid, chip=chip)
    model, inliers = fit_affine_ransac(tps, threshold_px=threshold_px, seed=seed)

    cols = np.array([t.col for t in tps], dtype=float)
    rows = np.array([t.row for t in tps], dtype=float)
    dxs = np.array([t.dx for t in tps], dtype=float)
    dys = np.array([t.dy for t in tps], dtype=float)
    pdy, pdx = model.predict(cols, rows)
    resid = np.hypot(dxs - pdx, dys - pdy)

    if inliers.any():
        rmse = float(np.sqrt(np.mean(resid[inliers] ** 2)))
    else:
        rmse = float("nan")
    return TiePointResult(
        tiepoints=tps, model=model, inliers=inliers,
        rmse_px=rmse, rmse_m=rmse * pixel_size_m,
    )
