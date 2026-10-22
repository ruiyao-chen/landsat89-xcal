"""Radiometric cross-calibration.

Reflective (OLI) bands are compared as a multiplicative ratio of ROI-mean TOA
reflectance; thermal (TIRS) bands as an additive difference of ROI-mean
brightness temperature (K). L9 is SBAF-corrected onto the L8 spectral footing
first.

Convention: a positive reflective bias means L9 reads high relative to L8
(ratio > 1 -> bias = ratio - 1); a positive thermal bias means L9 warmer than
L8 (L9 - L8).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config, sbaf as sbaf_mod


@dataclass
class BandResult:
    band_key: str
    label: str
    kind: str                 # "reflective" | "thermal"
    n_pairs: int
    n_pixels: int
    sbaf: float
    # reflective: ratio (L8/L9 means); thermal: n/a
    ratio: float | None
    # reflective: fractional bias (ratio-1); thermal: kelvin bias (L9-L8)
    bias: float
    # 1-sigma statistical spread of the per-pair estimates (same units as bias)
    scatter: float


def _roi_means(pixels: np.ndarray) -> float:
    """Robust ROI central value: median is resistant to residual outliers."""
    return float(np.median(pixels))


def reflective_band_bias(
    samples_list,
    band_key: str,
    sbaf: float | None = None,
) -> BandResult:
    """Cross-calibration bias for one reflective band across all scene pairs.

    ratio = (SBAF * L9_mean) / L8_mean,   bias = ratio - 1
    A positive bias means L9 reads high relative to L8.
    """
    band = next(b for b in config.REFLECTIVE_BANDS if b.key == band_key)
    sbaf = sbaf_mod.default_sbaf(band_key) if sbaf is None else sbaf

    ratios = []
    n_pix = 0
    for s in samples_list:
        l8, l9 = s.reflective[band_key]
        n_pix = max(n_pix, min(len(l8), len(l9)))
        m8 = _roi_means(l8)
        m9 = _roi_means(sbaf_mod.apply_sbaf(l9, sbaf))
        if m8 > 0:
            ratios.append(m9 / m8)
    ratios = np.asarray(ratios)
    ratio = float(np.mean(ratios))
    scatter = float(np.std(ratios, ddof=1)) if len(ratios) > 1 else 0.0
    return BandResult(
        band_key=band_key, label=band.label, kind="reflective",
        n_pairs=len(ratios), n_pixels=n_pix, sbaf=sbaf,
        ratio=ratio, bias=ratio - 1.0, scatter=scatter,
    )


def thermal_band_bias(
    samples_list,
    band_key: str,
    sbaf_K: float = 0.0,
) -> BandResult:
    """Cross-calibration bias for one thermal band across all scene pairs.

    bias = mean_pairs( (L9_mean + sbaf_K) - L8_mean )   [kelvin]
    """
    band = next(b for b in config.THERMAL_BANDS if b.key == band_key)
    diffs = []
    n_pix = 0
    for s in samples_list:
        l8, l9 = s.thermal[band_key]
        n_pix = max(n_pix, min(len(l8), len(l9)))
        m8 = _roi_means(l8)
        m9 = _roi_means(l9) + sbaf_K
        diffs.append(m9 - m8)
    diffs = np.asarray(diffs)
    bias = float(np.mean(diffs))
    scatter = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
    return BandResult(
        band_key=band_key, label=band.label, kind="thermal",
        n_pairs=len(diffs), n_pixels=n_pix, sbaf=sbaf_K,
        ratio=None, bias=bias, scatter=scatter,
    )


def cross_calibrate(samples_list, sbaf_table: dict[str, float] | None = None) -> list[BandResult]:
    """Run all reflective and thermal bands. ``sbaf_table`` maps band_key->SBAF."""
    sbaf_table = sbaf_table or {}
    results: list[BandResult] = []
    for b in config.REFLECTIVE_BANDS:
        results.append(reflective_band_bias(samples_list, b.key, sbaf_table.get(b.key)))
    for b in config.THERMAL_BANDS:
        results.append(thermal_band_bias(samples_list, b.key, sbaf_table.get(b.key, 0.0)))
    return results
