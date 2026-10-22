"""Data access.

Two backends with the same output contract so the calibration science never
needs to know where the pixels came from:

    get_scene_pairs(...)        -> list[ScenePair]
    extract_roi_samples(pair)   -> RoiSamples

``backend="gee"``  pulls real Landsat 8/9 Collection-2 TOA data from Google
Earth Engine (requires ``earthengine-api`` and an authenticated project).

``backend="synthetic"`` generates physically plausible paired ROIs with a
known injected inter-sensor bias, so the whole pipeline (radiometry,
uncertainty, geometry, plots) runs offline and the recovered bias can be
checked against ground truth. Use it to develop and to let others run the repo
without a GEE account.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np

from . import config


# --------------------------------------------------------------------------- #
# Output contract
# --------------------------------------------------------------------------- #
@dataclass
class ScenePair:
    site: str
    l8_id: str
    l9_id: str
    l8_date: str
    l9_date: str
    gap_days: int


@dataclass
class RoiSamples:
    """Per-band ROI pixel samples for one scene pair.

    reflective[band_key] -> (l8_pixels, l9_pixels) in TOA reflectance (unitless)
    thermal[band_key]    -> (l8_pixels, l9_pixels) in brightness temperature (K)
    l8_image / l9_image  -> optional 2-D arrays of a single band, for the
                            geometric registration step (None if unavailable).
    truth_bias           -> dict of injected bias per band (synthetic only).
    """
    pair: ScenePair
    reflective: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    thermal: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    l8_image: np.ndarray | None = None
    l9_image: np.ndarray | None = None
    truth_bias: dict[str, float] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Synthetic backend
# --------------------------------------------------------------------------- #
def _synthetic_pairs(site_key: str, n_pairs: int) -> list[ScenePair]:
    d0 = date.fromisoformat(config.DATE_START)
    pairs = []
    for i in range(n_pairs):
        l8 = d0 + timedelta(days=32 * i)
        l9 = l8 + timedelta(days=8)  # L9 trails L8 by 8 days on the same track
        pairs.append(ScenePair(
            site=site_key,
            l8_id=f"SYN/LC08/{site_key}/{i:03d}",
            l9_id=f"SYN/LC09/{site_key}/{i:03d}",
            l8_date=l8.isoformat(),
            l9_date=l9.isoformat(),
            gap_days=(l9 - l8).days,
        ))
    return pairs


# Nominal bright-desert TOA reflectance per reflective band (rough Libya-4).
_NOMINAL_REFL = {
    "coastal": 0.34, "blue": 0.37, "green": 0.43, "red": 0.50,
    "nir": 0.56, "swir1": 0.62, "swir2": 0.58,
}
# Nominal daytime desert at-sensor brightness temperature (K).
_NOMINAL_BT = {"tirs1": 315.0, "tirs2": 313.0}


def synthetic_truth(truth_seed: int = 12345) -> dict:
    """The *fixed* inter-sensor bias injected into the synthetic L9 data.

    A real inter-sensor calibration bias is a property of the sensor pair, not
    of an individual scene, so it is constant across all pairs. Reflective
    biases are fractional; thermal biases are in kelvin; the geometric shift is
    a fixed (dy, dx) in pixels.
    """
    r = np.random.default_rng(truth_seed)
    truth = {}
    for b in config.REFLECTIVE_BANDS:
        truth[b.key] = float(r.uniform(-0.01, 0.01))     # +/-1 %
    for b in config.THERMAL_BANDS:
        truth[b.key] = float(r.uniform(-0.3, 0.3))        # +/-0.3 K
    truth["_geo_shift_px"] = (0.35, -0.20)
    return truth


def _synthetic_samples(
    pair: ScenePair,
    n_pix: int,
    rng: np.random.Generator,
    inject_bias: bool,
    truth_seed: int = 12345,
) -> RoiSamples:
    """Create paired ROIs over a bright, uniform PICS-like surface.

    A fixed, band-dependent multiplicative bias (reflective) or additive bias
    (thermal, K) - constant across scene pairs - is injected into L9 relative to
    L8, so the mean recovered cross-calibration bias validates against ground
    truth. Per-pair, per-pixel speckle is added on top.
    """
    s = RoiSamples(pair=pair)
    truth = synthetic_truth(truth_seed) if inject_bias else {}

    for b in config.REFLECTIVE_BANDS:
        mu = _NOMINAL_REFL[b.key]
        base = mu * (1.0 + 0.01 * rng.standard_normal(n_pix))   # site texture
        bias = 1.0 + truth.get(b.key, 0.0)
        l8 = base + mu * 0.002 * rng.standard_normal(n_pix)
        l9 = base * bias + mu * 0.002 * rng.standard_normal(n_pix)
        s.reflective[b.key] = (l8, l9)
        s.truth_bias[b.key] = truth.get(b.key, 0.0)

    for b in config.THERMAL_BANDS:
        mu = _NOMINAL_BT[b.key]
        base = mu + 0.3 * rng.standard_normal(n_pix)
        dK = truth.get(b.key, 0.0)
        l8 = base + 0.1 * rng.standard_normal(n_pix)
        l9 = base + dK + 0.1 * rng.standard_normal(n_pix)
        s.thermal[b.key] = (l8, l9)
        s.truth_bias[b.key] = dK

    # Single-band ROI image pair with a fixed sub-pixel shift for geometry.
    img = _synthetic_image(128, rng)
    dy, dx = truth.get("_geo_shift_px", (0.0, 0.0))
    s.l8_image = img
    s.l9_image = _shift_subpixel(img, dy, dx) + 0.002 * rng.standard_normal(img.shape)
    s.truth_bias["_geo_shift_px"] = (dy, dx)
    return s


def _synthetic_image(n: int, rng: np.random.Generator) -> np.ndarray:
    """A broadband textured field standing in for a single-band ROI image.

    Rich high-frequency content (a spectrum of sinusoids plus fine speckle) gives
    phase correlation sharp features to lock onto, so sub-pixel registration is
    well conditioned - as a real textured scene would be.
    """
    y, x = np.mgrid[0:n, 0:n].astype(float)
    field_ = np.full((n, n), 0.6)
    for (px, py, amp) in [(23, 31, 0.20), (11, 7, 0.12), (5, 13, 0.08),
                          (37, 17, 0.06), (3, 4, 0.05)]:
        field_ += amp * np.sin(2 * np.pi * x / px) * np.cos(2 * np.pi * y / py)
    return field_ + 0.02 * rng.standard_normal((n, n))


def _shift_subpixel(img: np.ndarray, dy: float, dx: float) -> np.ndarray:
    """Shift an image by sub-pixel (dy, dx) with cubic-spline interpolation.

    Cubic (order=3) preserves the injected shift far better than bilinear, so a
    correct registration recovers (dy, dx) to a few hundredths of a pixel.
    """
    from scipy.ndimage import shift
    return shift(img, shift=(dy, dx), order=3, mode="nearest")


# Ground-truth affine displacement field for the tie-point demo:
#   dx(col,row) = tx + gx*(col - cx)
#   dy(col,row) = ty + gy*(row - cy)
# A translation (tx, ty) plus small gradients (gx, gy) standing in for the
# scale/rotation/keystone difference between two sensors' line-of-sight models.
SYNTH_FIELD = dict(tx=0.35, ty=-0.20, gx=0.0020, gy=0.0015)


def _true_displacement(col, row, shape, field=SYNTH_FIELD):
    H, W = shape
    cx, cy = W / 2.0, H / 2.0
    dx = field["tx"] + field["gx"] * (np.asarray(col) - cx)
    dy = field["ty"] + field["gy"] * (np.asarray(row) - cy)
    return dy, dx


def synthetic_warped_pair(
    n: int = 256,
    outlier_frac: float = 0.15,
    seed: int = 0,
    field: dict = SYNTH_FIELD,
):
    """Build an (L8, L9) image pair related by a known affine displacement field.

    L9 is L8 resampled through the field ``SYNTH_FIELD`` (translation + gradients).
    A fraction of the scene is then corrupted with decorrelated noise patches to
    mimic clouds / smoke, so RANSAC has genuine outliers to reject. Returns
    (l8_img, l9_img, field).
    """
    from scipy.ndimage import map_coordinates
    rng = np.random.default_rng(seed)
    ref = _synthetic_image(n, rng)

    rows, cols = np.mgrid[0:n, 0:n].astype(float)
    dy, dx = _true_displacement(cols, rows, (n, n), field)
    # mov[o] = ref[o - d(o)]  (feature moves by +d), matching register() sign
    mov = map_coordinates(ref, [rows - dy, cols - dx], order=3, mode="nearest")
    mov = mov + 0.001 * rng.standard_normal((n, n))

    # inject outlier patches (clouds/smoke): decorrelated blocks big enough to
    # corrupt whole chips so RANSAC has genuine outliers to reject.
    n_blocks = 0 if outlier_frac <= 0 else max(3, int(round(outlier_frac * 30)))
    for _ in range(n_blocks):
        bs = int(rng.integers(55, 85))
        r0 = int(rng.integers(0, n - bs))
        c0 = int(rng.integers(0, n - bs))
        mov[r0:r0 + bs, c0:c0 + bs] = 0.6 + 0.25 * rng.standard_normal((bs, bs))
    return ref, mov, field


# --------------------------------------------------------------------------- #
# GEE backend
# --------------------------------------------------------------------------- #
def _require_ee():
    try:
        import ee  # noqa: F401
    except ImportError as exc:  # pragma: no cover - env dependent
        raise ImportError(
            "The 'gee' backend needs the Earth Engine API.\n"
            "    pip install earthengine-api\n"
            "    earthengine authenticate\n"
            "Then initialise with your project: ee.Initialize(project='your-project')."
        ) from exc
    return __import__("ee")


def _gee_pairs(site_key: str, project: str | None) -> list[ScenePair]:  # pragma: no cover
    ee = _require_ee()
    ee.Initialize(project=project)
    site = config.SITES[site_key]
    roi = ee.Geometry.Point([site.lon, site.lat]).buffer(site.roi_km * 1000).bounds()

    def prep(coll_id):
        return (
            ee.ImageCollection(coll_id)
            .filterBounds(roi)
            .filterDate(config.DATE_START, config.DATE_END)
            .filter(ee.Filter.lt("CLOUD_COVER", config.MAX_CLOUD_PCT))
        )

    l8 = prep(config.EE_COLLECTIONS["L8"])
    l9 = prep(config.EE_COLLECTIONS["L9"])

    # Join L8 to the nearest L9 scene within MAX_PAIR_GAP_DAYS.
    millis = config.MAX_PAIR_GAP_DAYS * 86400 * 1000
    time_filter = ee.Filter.maxDifference(
        difference=millis, leftField="system:time_start", rightField="system:time_start"
    )
    joined = ee.Join.saveBest(matchKey="match", measureKey="dt").apply(
        l8, l9, time_filter
    )

    pairs: list[ScenePair] = []
    for feat in joined.getInfo()["features"]:
        props = feat["properties"]
        l8_id = props["system:index"]
        match = props.get("match")
        if not match:
            continue
        l9_props = match["properties"]
        t8 = props["system:time_start"] / 86400000.0
        t9 = l9_props["system:time_start"] / 86400000.0
        pairs.append(ScenePair(
            site=site_key,
            l8_id=l8_id,
            l9_id=l9_props["system:index"],
            l8_date=date.fromtimestamp(props["system:time_start"] / 1000).isoformat(),
            l9_date=date.fromtimestamp(l9_props["system:time_start"] / 1000).isoformat(),
            gap_days=int(round(abs(t9 - t8))),
        ))
    return pairs


def _gee_samples(pair: ScenePair, project: str | None, max_pixels: int) -> RoiSamples:  # pragma: no cover
    ee = _require_ee()
    site = config.SITES[pair.site]
    roi = ee.Geometry.Point([site.lon, site.lat]).buffer(site.roi_km * 1000).bounds()

    def sample(coll_id, img_id, band):
        img = ee.Image(f"{coll_id}/{img_id}")
        arr = img.select(band).sampleRectangle(region=roi, defaultValue=0)
        return np.array(arr.get(band).getInfo(), dtype=float).ravel()

    s = RoiSamples(pair=pair)
    for b in config.REFLECTIVE_BANDS:
        l8 = sample(config.EE_COLLECTIONS["L8"], pair.l8_id, b.ee_band)
        l9 = sample(config.EE_COLLECTIONS["L9"], pair.l9_id, b.ee_band)
        n = min(len(l8), len(l9), max_pixels)
        s.reflective[b.key] = (l8[:n], l9[:n])
    for b in config.THERMAL_BANDS:
        l8 = sample(config.EE_COLLECTIONS["L8"], pair.l8_id, b.ee_band)
        l9 = sample(config.EE_COLLECTIONS["L9"], pair.l9_id, b.ee_band)
        n = min(len(l8), len(l9), max_pixels)
        s.thermal[b.key] = (l8[:n], l9[:n])
    # For geometry, return the red-band ROI image for each sensor.
    red = next(b for b in config.REFLECTIVE_BANDS if b.key == "red")
    s.l8_image = _gee_image(ee, config.EE_COLLECTIONS["L8"], pair.l8_id, red.ee_band, roi)
    s.l9_image = _gee_image(ee, config.EE_COLLECTIONS["L9"], pair.l9_id, red.ee_band, roi)
    return s


def _gee_image(ee, coll_id, img_id, band, roi):  # pragma: no cover
    arr = ee.Image(f"{coll_id}/{img_id}").select(band).sampleRectangle(region=roi, defaultValue=0)
    return np.array(arr.get(band).getInfo(), dtype=float)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def get_scene_pairs(
    site_key: str = config.DEFAULT_SITE,
    backend: str = "synthetic",
    n_pairs: int = 6,
    project: str | None = None,
) -> list[ScenePair]:
    if backend == "synthetic":
        return _synthetic_pairs(site_key, n_pairs)
    if backend == "gee":
        return _gee_pairs(site_key, project)
    raise ValueError(f"unknown backend {backend!r}")


def extract_roi_samples(
    pair: ScenePair,
    backend: str = "synthetic",
    n_pix: int = 4000,
    inject_bias: bool = True,
    project: str | None = None,
    seed: int | None = None,
) -> RoiSamples:
    if backend == "synthetic":
        rng = np.random.default_rng(seed)
        return _synthetic_samples(pair, n_pix, rng, inject_bias)
    if backend == "gee":
        return _gee_samples(pair, project, n_pix)
    raise ValueError(f"unknown backend {backend!r}")
