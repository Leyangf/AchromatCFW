"""Achromatic Color Fringe Width (CFW) core routines.

This module exposes high-level helpers that compute colour-fringe related
metrics for an optical system.  Point spread function kernels and the
low-level edge response are implemented in :mod:`achromatcfw.core.psf`.

Main public entry points
------------------------
- :func:Edge            – edge profile for a single colour channel.
- :func:Farbsaum        – *binary* colour‑fringe flag at one pixel.
- :func:Farbsaumbreite  – fringe width (in pixels) across an x‑range.
- :func:ColorFringe     – *actual* RGB edge responses at one pixel.
"""
from __future__ import annotations

from typing import Tuple, Literal

import numpy as np

# Import the spectral response curves of the sensor.  ``channel_products``
# returns wavelength dependent energy terms (S·D) for each colour channel.
from achromatcfw.io.spectrum_loader import channel_products
from .psf import ALLOWED_PSF_MODES, DEFAULT_PSF_MODE, compute_edge_jit

# ------------------------------ Global constants ------------------------------
# Nominal f-number of the optical system.  Used when converting the chromatic
# focal shift into an equivalent blur radius.
K: float = 1.4

# Parameters controlling the non-linear exposure curve and gamma used when
# computing edge responses.
F_VALUE: float = 8.0
GAMMA_VALUE: float = 1.0

TOL: float = 0.15         # colour difference tolerance for the binary detector
XRANGE_VAL: int = 400     # half-width of the evaluation window in pixels

defocusrange: int = 650  # defocus range of CHL data in microns

# ------------------------------ Sensor data ------------------------------------
# Pre-compute the wavelength weighted sensor responses for each channel.  Only
# the second column (S·D) is needed for the kernel.
prods = channel_products()
sensor_map = {
    "R": prods["red"][:, 1],
    "G": prods["green"][:, 1],
    "B": prods["blue"][:, 1],
}

# ------------------------------------------------------------------------------
# High‑level Python wrappers (type safety, defaults, validation)
# ------------------------------------------------------------------------------

def _resolve_param(value: float | None, default: float) -> float:  # noqa: D401 – helper
    """Return *default* if *value* is None, else *value*."""
    return default if value is None else value


def Edge(
    color: Literal["R", "G", "B"],
    x: float,
    z: float,
    F: float | None = None,
    gamma: float | None = None,
    CHLdata: np.ndarray | None = None,
    K_param: float = K,
    psf_mode: Literal["disk", "gauss", "gauss_sphe"] = DEFAULT_PSF_MODE,
) -> float:
    """Compute *edge response* for a single colour channel.

    Parameters
    ----------
    color
        'R', 'G' or 'B'. Case‑insensitive.
    x, z
        Pixel offset and defocus (same units as *CHLdata*).
    F, gamma
        Exposure curve factor and display gamma. If *None*, fall back to
        :data:F_VALUE and :data:GAMMA_VALUE.
    CHLdata
        1‑D array with chromatic focal shift curve (µm). Required.
    K_param
        Effective f‑number (default from global constant).
    psf_mode
        Point‑spread‑function model: 'disk', 'gauss' or 'gauss_sphe'.

    Returns
    -------
    float
        Normalised edge response in [0, 1].
    """
    # Validate PSF mode early to provide a clear error for the user.
    if psf_mode not in ALLOWED_PSF_MODES:
        raise ValueError(
            f"psf_mode must be one of {ALLOWED_PSF_MODES}, got {psf_mode!r}")

    if CHLdata is None:
        raise ValueError("CHLdata array is required (got None)")

    # Use default exposure and gamma values if the caller did not specify them.
    F_val: float = _resolve_param(F, F_VALUE)
    gamma_val: float = _resolve_param(gamma, GAMMA_VALUE)

    # Map the channel name to its spectral sensitivity vector.
    sensor_data = sensor_map[color.upper()]
    return compute_edge_jit(
        float(x), float(z), F_val, gamma_val, sensor_data, CHLdata, K_param, psf_mode
    )


# --------------------------------------------------------------------------
# Colour‑fringe utilities (built atop Edge)
# --------------------------------------------------------------------------

def Farbsaum(
    x: float,
    z: float,
    F: float | None = None,
    gamma: float | None = None,
    CHLdata: np.ndarray | None = None,
    psf_mode: Literal["disk", "gauss", "gauss_sphe"] = DEFAULT_PSF_MODE,
) -> int:
    """Binary colour‑fringe detector (1 if fringe, 0 if not)."""
    if CHLdata is None:
        raise ValueError("CHLdata array is required (got None)")

    # Evaluate the edge response for all three channels and compare their
    # differences against ``TOL`` to decide if a colour fringe is present.
    r = Edge("R", x, z, F, gamma, CHLdata, psf_mode=psf_mode)
    g = Edge("G", x, z, F, gamma, CHLdata, psf_mode=psf_mode)
    b = Edge("B", x, z, F, gamma, CHLdata, psf_mode=psf_mode)
    return 1 if (abs(r - b) > TOL or abs(r - g) > TOL or abs(g - b) > TOL) else 0


def Farbsaumbreite(
    z: float,
    F: float | None = None,
    gamma: float | None = None,
    CHLdata: np.ndarray | None = None,
    psf_mode: Literal["disk", "gauss", "gauss_sphe"] = DEFAULT_PSF_MODE,
) -> int:
    """Return *width* (in pixels) of the colour fringe at defocus *z*."""
    if CHLdata is None:
        raise ValueError("CHLdata array is required (got None)")

    # Scan across the specified range of pixel offsets and count the number of
    # locations where a fringe is detected.
    xs = np.arange(-XRANGE_VAL, XRANGE_VAL + 1, dtype=np.int32)
    width = 0
    for x in xs:
        width += Farbsaum(float(x), z, F, gamma, CHLdata, psf_mode)
    return width


def ColorFringe(
    x: float,
    z: float,
    F: float | None = None,
    gamma: float | None = None,
    CHLdata: np.ndarray | None = None,
    psf_mode: Literal["disk", "gauss", "gauss_sphe"] = DEFAULT_PSF_MODE,
) -> Tuple[float, float, float]:
    """RGB edge responses *without* binarisation (diagnostics helper)."""
    # ``CHLdata`` defines the chromatic focal shift curve.  Without it no
    # meaningful prediction can be made.
    if CHLdata is None:
        raise ValueError("CHLdata array is required (got None)")

    # Simply return the raw edge responses for inspection.
    return (
        Edge("R", x, z, F, gamma, CHLdata, psf_mode=psf_mode),
        Edge("G", x, z, F, gamma, CHLdata, psf_mode=psf_mode),
        Edge("B", x, z, F, gamma, CHLdata, psf_mode=psf_mode),
    )
