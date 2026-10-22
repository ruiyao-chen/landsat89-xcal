"""End-to-end calibration uncertainty budget.

The reported cross-calibration bias for each band is combined with a 1-sigma
uncertainty built by adding independent components in quadrature:

  * statistical  - spread of the per-pair estimates / sqrt(N_pairs)
                   (standard error of the mean bias)
  * sbaf         - residual spectral-band-adjustment knowledge error
  * temporal/BRDF- surface + illumination drift over the L8/L9 acquisition gap
  * atmospheric  - residual atmospheric contribution over the site

Reflective components are fractional (of reflectance); thermal components are
in kelvin. Results are given at k=1 (1-sigma) and k=2 (~95%).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config
from .radiometric import BandResult


@dataclass
class UncertaintyResult:
    band_key: str
    label: str
    kind: str
    bias: float               # fractional (reflective) or K (thermal)
    components: dict[str, float]   # 1-sigma, same units as bias
    u_combined_k1: float      # combined standard uncertainty (k=1)
    u_expanded_k2: float      # expanded uncertainty (k=2)

    @property
    def significant(self) -> bool:
        """True if the bias exceeds its own k=2 uncertainty."""
        return abs(self.bias) > self.u_expanded_k2


def _quadrature(components: dict[str, float]) -> float:
    return float(np.sqrt(sum(v * v for v in components.values())))


def budget_for_band(
    result: BandResult,
    priors: config.UncertaintyPriors = config.PRIORS,
    gap_days: float = config.MAX_PAIR_GAP_DAYS,
) -> UncertaintyResult:
    n = max(result.n_pairs, 1)
    stat = result.scatter / np.sqrt(n)   # standard error of the mean

    if result.kind == "reflective":
        components = {
            "statistical": stat,
            "sbaf": priors.refl_sbaf,
            "temporal_brdf": priors.temporal_brdf_frac * gap_days,
            "atmospheric": priors.atmos_frac,
            "noise_floor": priors.refl_speckle_floor,
        }
    else:  # thermal (kelvin)
        components = {
            "statistical": stat,
            "sbaf": priors.thermal_sbaf_K,
            "temporal_brdf": priors.temporal_brdf_frac * gap_days * 10.0,  # ~frac->K scaling near 300K/... kept explicit
            "atmospheric": priors.atmos_K,
            "noise_floor": priors.thermal_noise_floor_K,
        }

    u1 = _quadrature(components)
    return UncertaintyResult(
        band_key=result.band_key, label=result.label, kind=result.kind,
        bias=result.bias, components=components,
        u_combined_k1=u1, u_expanded_k2=2.0 * u1,
    )


def budget(results: list[BandResult], **kw) -> list[UncertaintyResult]:
    return [budget_for_band(r, **kw) for r in results]
