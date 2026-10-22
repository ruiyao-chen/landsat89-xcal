"""Geometric cross-calibration: inter-sensor registration offset.

Complements the radiometric comparison by quantifying how well L8 and L9 agree
*geometrically* over the site. We estimate the sub-pixel translation that best
aligns the L9 ROI image to L8 using phase cross-correlation with upsampled-DFT
sub-pixel refinement (Guizar-Sicairos, Thurman & Fienup, Opt. Lett. 2008 - the
same algorithm behind scikit-image's phase_cross_correlation), then convert to
ground distance via the pixel size.

This is the geolocation / band-to-band-style registration accuracy check: for
near-coincident same-orbit scenes over a stable site, the residual shift should
be a small fraction of a pixel.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# Landsat OLI reflective pixel size (m). TIRS is 100 m (resampled to 30 m).
PIXEL_SIZE_M = 30.0


@dataclass
class RegistrationResult:
    dy_px: float              # row shift (L9 relative to L8), pixels
    dx_px: float              # column shift, pixels
    magnitude_px: float       # total shift magnitude, pixels
    magnitude_m: float        # total shift magnitude, metres
    response: float           # normalised correlation peak (0..1)


def _upsampled_dft(data: np.ndarray, region: int, upsample: int, offsets) -> np.ndarray:
    """Upsampled inverse DFT over a small region, via matrix multiplies.

    Faithful to the Guizar-Sicairos matrix-multiply DFT (as in scikit-image):
    evaluates the IDFT of ``data`` on a ``region`` x ``region`` grid with
    1/upsample-pixel spacing, offset by ``offsets`` = (row0, col0), without
    zero-padding the full FFT.
    """
    im2pi = 1j * 2 * np.pi
    nr, nc = data.shape
    # column (last axis) kernel
    kernel_c = np.exp(
        -im2pi
        * (np.arange(region)[:, None] - offsets[1])
        * np.fft.fftfreq(nc, upsample)[None, :]
    )
    data = kernel_c @ data.T          # -> (region, nr)
    # row kernel
    kernel_r = np.exp(
        -im2pi
        * (np.arange(region)[:, None] - offsets[0])
        * np.fft.fftfreq(nr, upsample)[None, :]
    )
    data = kernel_r @ data.T          # -> (region, region)
    return data


def _phase_cross_correlation(ref: np.ndarray, mov: np.ndarray, upsample: int = 50):
    """Return (dy, dx, response): displacement of ``mov`` relative to ``ref``."""
    ref = ref - ref.mean()
    mov = mov - mov.mean()
    F = np.fft.fft2(ref)
    G = np.fft.fft2(mov)
    R = F * np.conj(G)

    corr = np.fft.ifft2(R)
    shape = np.array(corr.shape)
    midpoints = shape // 2
    peak = np.array(np.unravel_index(np.argmax(np.abs(corr)), corr.shape), dtype=float)

    # map integer peak to [-n/2, n/2)
    shifts = peak.copy()
    shifts[shifts > midpoints] -= shape[shifts > midpoints]

    # --- sub-pixel refinement via upsampled DFT around the integer peak ---- #
    region = int(np.ceil(upsample * 1.5))
    dftshift = region // 2
    sample_offsets = dftshift - shifts * upsample
    cc = _upsampled_dft(R.conj(), region, upsample, sample_offsets).conj()
    sub = np.array(np.unravel_index(np.argmax(np.abs(cc)), cc.shape), dtype=float) - dftshift
    shifts = shifts + sub / upsample

    peak_val = np.abs(cc).max()
    response = float(peak_val / (np.abs(F).sum() ** 0.5 * np.abs(G).sum() ** 0.5 + 1e-12))

    # 'shifts' (skimage convention) aligns mov onto ref; the physical
    # displacement of mov relative to ref is the negative of that.
    dy, dx = -shifts[0], -shifts[1]
    return float(dy), float(dx), response


def register(ref_image: np.ndarray, mov_image: np.ndarray, pixel_size_m: float = PIXEL_SIZE_M) -> RegistrationResult:
    """Estimate the (dy, dx) shift that aligns ``mov_image`` (L9) to ``ref_image`` (L8)."""
    if ref_image is None or mov_image is None:
        raise ValueError("geometric registration needs single-band ROI images for both sensors")
    if ref_image.shape != mov_image.shape:
        ny = min(ref_image.shape[0], mov_image.shape[0])
        nx = min(ref_image.shape[1], mov_image.shape[1])
        ref_image = ref_image[:ny, :nx]
        mov_image = mov_image[:ny, :nx]

    dy, dx, response = _phase_cross_correlation(ref_image, mov_image)
    mag_px = float(np.hypot(dy, dx))
    return RegistrationResult(
        dy_px=dy, dx_px=dx, magnitude_px=mag_px,
        magnitude_m=mag_px * pixel_size_m, response=response,
    )


def register_pairs(samples_list, pixel_size_m: float = PIXEL_SIZE_M) -> list[RegistrationResult]:
    """Run registration for every scene pair that carries ROI images."""
    out = []
    for s in samples_list:
        if s.l8_image is not None and s.l9_image is not None:
            out.append(register(s.l8_image, s.l9_image, pixel_size_m))
    return out
