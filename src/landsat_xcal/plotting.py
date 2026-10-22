"""Figures for the cross-calibration report.

Kept dependency-light (matplotlib only) and theme-neutral. Each function
returns the Matplotlib Figure so callers can save or embed it.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from .uncertainty import UncertaintyResult


def plot_reflective_bias(results: list[UncertaintyResult]):
    """Per-band reflective cross-calibration bias (%) with k=2 error bars."""
    refl = [r for r in results if r.kind == "reflective"]
    labels = [r.label for r in refl]
    bias = np.array([r.bias for r in refl]) * 100.0
    err = np.array([r.u_expanded_k2 for r in refl]) * 100.0

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(labels))
    ax.bar(x, bias, yerr=err, capsize=4, color="#4C78A8", alpha=0.9)
    ax.axhline(0, color="#333", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("L9 - L8 bias (%)")
    ax.set_title("OLI reflective cross-calibration bias (k=2 error bars)")
    fig.tight_layout()
    return fig


def plot_thermal_bias(results: list[UncertaintyResult]):
    """Per-band thermal cross-calibration bias (K) with k=2 error bars."""
    th = [r for r in results if r.kind == "thermal"]
    labels = [r.label for r in th]
    bias = np.array([r.bias for r in th])
    err = np.array([r.u_expanded_k2 for r in th])

    fig, ax = plt.subplots(figsize=(5, 4.5))
    x = np.arange(len(labels))
    ax.bar(x, bias, yerr=err, capsize=4, color="#E45756", alpha=0.9)
    ax.axhline(0, color="#333", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("L9 - L8 brightness temperature (K)")
    ax.set_title("TIRS thermal cross-calibration bias (k=2 error bars)")
    fig.tight_layout()
    return fig


def plot_uncertainty_components(result: UncertaintyResult):
    """Breakdown of 1-sigma uncertainty components for one band."""
    comps = result.components
    labels = list(comps.keys())
    vals = np.array([comps[k] for k in labels])
    if result.kind == "reflective":
        vals = vals * 100.0
        unit = "%"
    else:
        unit = "K"

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(labels))
    ax.bar(x, vals, color="#72B7B2", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel(f"1-sigma contribution ({unit})")
    ax.set_title(f"Uncertainty budget - {result.label}")
    fig.tight_layout()
    return fig


def plot_tiepoint_field(result, image_shape=None):
    """Quiver of tie-point displacement vectors; inliers green, outliers red.

    ``result`` is a tiepoints.TiePointResult. Vectors are exaggerated for
    visibility (sub-pixel shifts would otherwise be invisible).
    """
    tps = result.tiepoints
    cols = np.array([t.col for t in tps])
    rows = np.array([t.row for t in tps])
    dx = np.array([t.dx for t in tps])
    dy = np.array([t.dy for t in tps])
    inl = result.inliers

    fig, ax = plt.subplots(figsize=(6.5, 6))
    scale = 1.0  # quiver autoscale; sub-pixel vectors shown in data units
    ax.quiver(cols[inl], rows[inl], dx[inl], dy[inl], color="#54A24B",
              angles="xy", scale_units="xy", scale=0.02, width=0.004,
              label=f"inliers (n={int(inl.sum())})")
    if (~inl).any():
        ax.quiver(cols[~inl], rows[~inl], dx[~inl], dy[~inl], color="#E45756",
                  angles="xy", scale_units="xy", scale=0.02, width=0.004,
                  label=f"outliers (n={int((~inl).sum())})")
    if image_shape is not None:
        ax.set_xlim(0, image_shape[1])
        ax.set_ylim(image_shape[0], 0)
    ax.set_aspect("equal")
    ax.set_xlabel("column (px)")
    ax.set_ylabel("row (px)")
    ax.set_title(f"Tie-point displacement field (RANSAC)\n"
                 f"inlier RMSE = {result.rmse_px:.3f} px ({result.rmse_m:.2f} m)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def plot_geometric(reg_results):
    """Scatter of inter-sensor registration offsets (pixels)."""
    dy = np.array([r.dy_px for r in reg_results])
    dx = np.array([r.dx_px for r in reg_results])

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.axhline(0, color="#999", lw=0.8)
    ax.axvline(0, color="#999", lw=0.8)
    ax.scatter(dx, dy, color="#54A24B", s=50, alpha=0.9, edgecolor="k")
    lim = max(0.6, float(np.max(np.abs(np.concatenate([dx, dy]))) * 1.3) if len(dx) else 0.6)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("dx offset (pixels)")
    ax.set_ylabel("dy offset (pixels)")
    ax.set_title("L9-to-L8 geometric registration offset")
    fig.tight_layout()
    return fig
