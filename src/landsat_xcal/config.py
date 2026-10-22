"""Configuration: calibration sites, acquisition windows, and sensor definitions.

Landsat 8 (L8) and Landsat 9 (L9) share the same WRS-2 orbit, 8 days out of
phase, so for a given path/row they acquire near-coincident, near-identical-
geometry scenes. Over a spatially uniform, radiometrically stable
pseudo-invariant calibration site (PICS) this makes a clean inter-sensor
cross-calibration possible with minimal confounds.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# --------------------------------------------------------------------------- #
# Pseudo-invariant calibration sites (PICS)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Site:
    name: str
    lon: float          # center longitude (deg)
    lat: float          # center latitude (deg)
    roi_km: float       # half-width of the square ROI (km)
    note: str = ""


# Well-characterised radiometric-stability sites from the cal/val literature.
SITES: dict[str, Site] = {
    "Libya4": Site("Libya-4", 23.39, 28.55, 10.0,
                   "CEOS reference PICS; bright, stable Saharan sand."),
    "RVUS": Site("Railroad Valley (RVUS)", -115.69, 38.50, 7.5,
                 "CEOS RadCalNet site; playa surface, US."),
    "Algeria3": Site("Algeria-3", 7.66, 30.32, 10.0,
                     "CEOS reference PICS; Saharan sand."),
}

DEFAULT_SITE = "Libya4"


# --------------------------------------------------------------------------- #
# Acquisition window
# --------------------------------------------------------------------------- #
# L8 and L9 are 8 days apart on the same track. Pairing scenes within a short
# window keeps surface/illumination drift small. We search a season and keep
# the closest-in-time cloud-free pairs.
DATE_START = "2023-01-01"
DATE_END = "2023-12-31"
MAX_PAIR_GAP_DAYS = 8          # keep L8/L9 scenes no more than this far apart
MAX_CLOUD_PCT = 5.0            # per-scene scene-level cloud cover ceiling


# --------------------------------------------------------------------------- #
# Sensors and bands
# --------------------------------------------------------------------------- #
# OLI reflective bands are compared in TOA reflectance; TIRS thermal bands are
# compared in at-sensor brightness temperature (K).
#
# GEE Collection 2 Tier 1 TOA ids:
#   L8  LANDSAT/LC08/C02/T1_TOA
#   L9  LANDSAT/LC09/C02/T1_TOA
EE_COLLECTIONS = {
    "L8": "LANDSAT/LC08/C02/T1_TOA",
    "L9": "LANDSAT/LC09/C02/T1_TOA",
}


@dataclass(frozen=True)
class Band:
    key: str            # internal name
    ee_band: str        # GEE band id (same for L8/L9 OLI/TIRS)
    kind: str           # "reflective" or "thermal"
    wavelength_um: float
    label: str


# OLI reflective + TIRS thermal. Center wavelengths are nominal (um).
BANDS: list[Band] = [
    Band("coastal", "B1", "reflective", 0.443, "Coastal/Aerosol"),
    Band("blue", "B2", "reflective", 0.482, "Blue"),
    Band("green", "B3", "reflective", 0.561, "Green"),
    Band("red", "B4", "reflective", 0.655, "Red"),
    Band("nir", "B5", "reflective", 0.865, "NIR"),
    Band("swir1", "B6", "reflective", 1.609, "SWIR-1"),
    Band("swir2", "B7", "reflective", 2.201, "SWIR-2"),
    Band("tirs1", "B10", "thermal", 10.90, "TIRS-1"),
    Band("tirs2", "B11", "thermal", 12.00, "TIRS-2"),
]

REFLECTIVE_BANDS = [b for b in BANDS if b.kind == "reflective"]
THERMAL_BANDS = [b for b in BANDS if b.kind == "thermal"]


# --------------------------------------------------------------------------- #
# Uncertainty-budget default 1-sigma components (fractional for reflective,
# kelvin for thermal). These are starting estimates; refine from the data and
# from published L8/L9 cal reports. See uncertainty.py.
# --------------------------------------------------------------------------- #
@dataclass
class UncertaintyPriors:
    # reflective (fractional, 1-sigma)
    refl_sbaf: float = 0.003          # residual spectral-band-adjustment error
    refl_speckle_floor: float = 0.001  # irreducible ROI noise floor
    # thermal (kelvin, 1-sigma)
    thermal_sbaf_K: float = 0.05
    thermal_noise_floor_K: float = 0.05
    # shared
    temporal_brdf_frac: float = 0.0005  # per-day surface/illumination drift (frac)
    atmos_frac: float = 0.002           # residual atmospheric contribution (frac)
    atmos_K: float = 0.10               # residual atmospheric contribution (K)


PRIORS = UncertaintyPriors()
