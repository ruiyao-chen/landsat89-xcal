"""Spectral Band Adjustment Factor (SBAF).

Even for nominally matched bands, L8 OLI and L9 OLI (and TIRS-1/2) have slightly
different relative spectral responses (RSRs). Over a site with a non-flat
surface reflectance spectrum this produces an apparent inter-sensor difference
that is *spectral*, not radiometric. The SBAF corrects for it so the residual
difference is attributable to calibration.

For band b, with reference surface+atmosphere spectrum L(lambda) (TOA radiance
or reflectance) and sensor RSRs R8_b, R9_b:

    Lhat_8 = integral(L * R8_b dlambda) / integral(R8_b dlambda)
    Lhat_9 = integral(L * R9_b dlambda) / integral(R9_b dlambda)
    SBAF_b = Lhat_8 / Lhat_9

The L9 measurement is multiplied by SBAF_b to place it on the L8 spectral
footing before the radiometric comparison.

L8 and L9 RSRs are very close, so SBAFs are typically within a few tenths of a
percent of unity; the residual RSR-knowledge error is carried in the
uncertainty budget (config.PRIORS.refl_sbaf / thermal_sbaf_K), not here.
"""

from __future__ import annotations

import numpy as np

# numpy>=2 renamed trapz -> trapezoid; support both.
_trapz = getattr(np, "trapezoid", None) or np.trapz


def sbaf_from_rsr(
    wavelength: np.ndarray,
    rsr_l8: np.ndarray,
    rsr_l9: np.ndarray,
    ref_spectrum: np.ndarray,
) -> float:
    """Compute a single band's SBAF = Lhat_8 / Lhat_9 by RSR-weighted integration.

    All arrays share the same wavelength grid. ``ref_spectrum`` is the TOA
    radiance or reflectance spectrum of the site (e.g. a Hyperion/PRISMA or
    modelled desert spectrum). Returns the multiplicative factor applied to L9.
    """
    rsr_l8 = np.clip(rsr_l8, 0, None)
    rsr_l9 = np.clip(rsr_l9, 0, None)
    lhat8 = _trapz(ref_spectrum * rsr_l8, wavelength) / _trapz(rsr_l8, wavelength)
    lhat9 = _trapz(ref_spectrum * rsr_l9, wavelength) / _trapz(rsr_l9, wavelength)
    if lhat9 == 0:
        return 1.0
    return float(lhat8 / lhat9)


def default_sbaf(band_key: str) -> float:
    """Placeholder SBAFs (~unity) for use before real RSRs are loaded.

    Replace by calling ``sbaf_from_rsr`` with the published L8/L9 OLI and TIRS
    RSRs (USGS/CalVal) and a site reference spectrum. Returning ~1.0 here means
    the pipeline runs without RSR files while making the approximation explicit.
    """
    return 1.0


def apply_sbaf(l9_values: np.ndarray, sbaf: float) -> np.ndarray:
    """Place L9 on the L8 spectral footing (multiplicative, reflective bands)."""
    return l9_values * sbaf
