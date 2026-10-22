"""Tests for the calibration science, using the synthetic backend with a known
injected bias so recovered values can be checked against ground truth.

Run:  pytest -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from landsat_xcal import config, data, radiometric, uncertainty, geometric, sbaf, tiepoints


def _samples(n_pairs=8, seed=0, inject_bias=True):
    pairs = data.get_scene_pairs(config.DEFAULT_SITE, backend="synthetic", n_pairs=n_pairs)
    return [
        data.extract_roi_samples(p, backend="synthetic", seed=seed + i, inject_bias=inject_bias)
        for i, p in enumerate(pairs)
    ]


def test_zero_bias_recovered_when_none_injected():
    s = _samples(inject_bias=False)
    results = radiometric.cross_calibrate(s)
    for r in results:
        if r.kind == "reflective":
            assert abs(r.bias) < 0.003, f"{r.label}: {r.bias}"
        else:
            assert abs(r.bias) < 0.1, f"{r.label}: {r.bias}"


def test_reflective_bias_sign_and_scale():
    # A single fixed, known multiplicative bias on one pair -> exact recovery.
    pairs = data.get_scene_pairs(config.DEFAULT_SITE, backend="synthetic", n_pairs=1)
    s = data.extract_roi_samples(pairs[0], backend="synthetic", seed=1, inject_bias=True)
    r = radiometric.reflective_band_bias([s], "red")
    injected = s.truth_bias["red"]
    # recovered bias should match injected to within the ROI noise level
    assert r.bias == pytest.approx(injected, abs=2e-3)


def test_uncertainty_is_positive_and_expands():
    s = _samples()
    results = radiometric.cross_calibrate(s)
    budget = uncertainty.budget(results)
    for u in budget:
        assert u.u_combined_k1 > 0
        assert u.u_expanded_k2 == pytest.approx(2 * u.u_combined_k1)
        # combined >= largest single component (quadrature property)
        assert u.u_combined_k1 >= max(u.components.values()) - 1e-12


def test_sbaf_unity_is_identity():
    x = np.array([0.1, 0.2, 0.3])
    assert np.allclose(sbaf.apply_sbaf(x, 1.0), x)


def test_sbaf_from_rsr_recovers_known_ratio():
    # Flat spectrum -> SBAF must be exactly 1 regardless of RSR shape.
    wl = np.linspace(0.4, 0.5, 100)
    r8 = np.exp(-((wl - 0.45) ** 2) / 0.001)
    r9 = np.exp(-((wl - 0.452) ** 2) / 0.001)
    flat = np.ones_like(wl)
    assert sbaf.sbaf_from_rsr(wl, r8, r9, flat) == pytest.approx(1.0, abs=1e-9)


def test_geometric_recovers_injected_shift():
    pairs = data.get_scene_pairs(config.DEFAULT_SITE, backend="synthetic", n_pairs=1)
    s = data.extract_roi_samples(pairs[0], backend="synthetic", seed=3, inject_bias=True)
    reg = geometric.register(s.l8_image, s.l9_image)
    gdy, gdx = s.truth_bias["_geo_shift_px"]
    # sub-pixel recovery within ~0.15 px of truth
    assert reg.dy_px == pytest.approx(gdy, abs=0.15)
    assert reg.dx_px == pytest.approx(gdx, abs=0.15)


def test_tiepoints_recover_affine_field():
    # no outliers: the fitted field should match the injected one near centre
    ref, mov, field = data.synthetic_warped_pair(n=256, outlier_frac=0.0, seed=0)
    res = tiepoints.register_tiepoints(ref, mov, grid=(8, 8), chip=64)
    cy, cx = 128, 128
    tdy, tdx = data._true_displacement(cx, cy, (256, 256), field)
    pdy, pdx = res.model.predict(cx, cy)
    assert float(pdy) == pytest.approx(float(tdy), abs=0.1)
    assert float(pdx) == pytest.approx(float(tdx), abs=0.1)
    assert res.rmse_px < 0.2
    assert res.n_inliers == res.n_tiepoints  # nothing to reject


def test_tiepoints_reject_outliers():
    # with injected cloud/smoke blocks, RANSAC must flag some tie points
    ref, mov, field = data.synthetic_warped_pair(n=256, outlier_frac=0.25, seed=0)
    res = tiepoints.register_tiepoints(ref, mov, grid=(8, 8), chip=64, threshold_px=0.4)
    assert res.n_inliers < res.n_tiepoints        # some rejected
    assert res.n_inliers >= 0.6 * res.n_tiepoints  # but majority kept
    assert res.rmse_px < 0.4                        # inlier fit stays sub-pixel
