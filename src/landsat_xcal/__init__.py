"""landsat_xcal: inter-sensor radiometric (and geometric) cross-calibration of
Landsat 8 and Landsat 9 over a pseudo-invariant calibration site.

The package separates concerns so the calibration science is independent of the
data source:

    config       sites, date windows, sensor/band definitions
    data         Google Earth Engine access + a synthetic fallback generator
    sbaf         spectral band adjustment factors (RSR mismatch correction)
    radiometric  per-band cross-calibration bias (OLI reflectance, TIRS BT)
    uncertainty  end-to-end calibration uncertainty budget
    geometric    inter-sensor geolocation / band-to-band registration offsets
    plotting     figures for the report
"""

__version__ = "0.1.0"
