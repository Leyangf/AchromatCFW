"""Point spread function (PSF) models and edge computation kernels.

This module provides Numba-accelerated helpers describing various point
spread functions and a low-level edge response kernel.  Higher level colour
fringe utilities build upon these primitives.
"""
from __future__ import annotations

from math import erf as math_erf, tanh, sqrt, pi, exp, fabs
from typing import Literal

import numpy as np
from numba import njit

# Valid PSF modes -------------------------------------------------------------
ALLOWED_PSF_MODES: tuple[str, ...] = ("disk", "gauss", "gauss_sphe")
DEFAULT_PSF_MODE: Literal["gauss"] = "gauss"


@njit(cache=True)
def Exposure_jit(x: float, F: float) -> float:
    """Normalised exposure curve using hyperbolic tangent."""
    return tanh(F * x) / tanh(F)


@njit(cache=True)
def disk_ESF_jit(x: float, ratio: float) -> float:
    """Disk PSF edge-spread function (geometric blur)."""
    if ratio < 1e-6:
        return 1.0 if x >= 0.0 else 0.0
    if x >= ratio:
        return 1.0
    if x <= -ratio:
        return 0.0
    return 0.5 * (1.0 + x / ratio)


@njit(cache=True)
def gauss_ESF_jit(x: float, ratio: float) -> float:
    """Gaussian PSF edge-spread function."""
    if ratio < 1e-6:
        return 1.0 if x >= 0.0 else 0.0
    return 0.5 * (1.0 + math_erf(x / (sqrt(2.0) * 0.5 * ratio)))


@njit(cache=True)
def gauss_ESF_sphe_jit(x: float, ratio: float) -> float:
    """Gaussian PSF with first-order spherical aberration (approx.)."""
    zernike_coef = 0.1  # waves – parameterise if needed
    phi_sigma = 2.0 * pi * zernike_coef
    strehl = exp(-(phi_sigma ** 2.0))
    if ratio < 1e-6:
        return strehl if x >= 0.0 else 0.0
    return 0.5 * (1.0 + math_erf(x / ratio * sqrt(strehl * 0.5)))


@njit(cache=True)
def compute_edge_jit(
    x: float,
    z: float,
    F: float,
    gamma: float,
    sensor_data: np.ndarray,
    CHLdata: np.ndarray,
    K_param: float,
    psf_mode: Literal["disk", "gauss", "gauss_sphe"],
) -> float:
    """Edge response for a single pixel location (low-level kernel)."""
    denom_factor = sqrt(4.0 * K_param ** 2.0 - 1.0)

    acc = 0.0
    for n in range(CHLdata.size):
        ratio = fabs((z - CHLdata[n]) / denom_factor)
        if psf_mode == "disk":
            weight = disk_ESF_jit(x, ratio)
        elif psf_mode == "gauss":
            weight = gauss_ESF_jit(x, ratio)
        else:  # "gauss_sphe"
            weight = gauss_ESF_sphe_jit(x, ratio)
        acc += sensor_data[n] * weight

    denom = np.sum(sensor_data)
    if denom == 0.0:
        return 0.0

    return Exposure_jit(acc / denom, F) ** gamma
