"""Spectral Hamiltonian and initial states, in pure Array API."""

from __future__ import annotations

import math

from array_api_compat import array_namespace

from toolbox_talk.data import SAMPLE_IMAGES, SAMPLE_SHIFTED_IMAGES, sample_image
from toolbox_talk.utils import Array


def apply_hamiltonian(psi: Array, V: Array, K: Array) -> Array:
    """
    Spectral Hamiltonian operator. Uses Fast Fourier Transforms to compute exact
    continuous derivatives, eliminating grid dispersion entirely.
    """
    xp = array_namespace(psi, V)

    # Momentum space -> apply K -> Real space
    psi_k = xp.fft.fft2(psi)
    T_psi = xp.fft.ifft2(K * psi_k)

    return T_psi + V * psi


def get_propagators(N: int, L: float, xp, device=None, dtype=None):
    """
    Creates the 2D grid, Harmonic Oscillator potential, and initial state.
    Calculates the exact spectral momentum operator (K) using pure Array API.
    """
    # 1. Spatial Grid and Potential (V)
    x = xp.linspace(-L, L, N, endpoint=False, device=device, dtype=dtype)
    X, Y = xp.meshgrid(x, x, indexing="xy")

    V = -0.5 * (X**2 + Y**2)

    # 2. Momentum Grid and Kinetic Operator (K)
    dx_val = 2 * L / N
    f = xp.fft.fftfreq(N, d=dx_val, device=device)

    # fftfreq returns default float; cast to match spatial grid if dtype is specified
    if dtype is not None:
        f = xp.astype(f, dtype)

    kx = 2 * math.pi * f
    Kx, Ky = xp.meshgrid(kx, kx, indexing="xy")

    K = -0.5 * (Kx**2 + Ky**2)

    return V, K


def get_initial_state(
    N: int,
    L: float,
    xp,
    state_image: str = "cat_state",
    blur: float = 0.35,
    device=None,
    dtype=None,
) -> Array:
    """
    Builds a normalised complex initial state on the N x N grid.
    """
    x = xp.linspace(-L, L, N, endpoint=False, device=device)
    X, Y = xp.meshgrid(x, x, indexing="xy")

    # Resolve target dtype so we don't accidentally fall back to defaults that conflict with the device
    target_dtype = dtype if dtype is not None else xp.complex128

    if state_image == "cat_state":
        width = L / 8.0
        offset = L / 2.0
        packet_left = xp.exp(-((X + offset) ** 2 + Y**2) / (2 * width**2))
        packet_right = xp.exp(-((X - offset) ** 2 + Y**2) / (2 * width**2))
        amplitude = packet_left + packet_right

    elif state_image == "cosine":
        # A broad Gaussian envelope to prevent FFT boundary artifacts
        envelope = xp.exp(-(X**2 + Y**2) / (2 * (L / 3) ** 2))
        frequency = 3.0
        amplitude = xp.cos(frequency * X) * envelope

    elif state_image == "double_slit":
        slit_distance = L / 1.5
        slit_width = L / 30.0
        slit_1 = xp.exp(-(Y**2 + (X - slit_distance / 2) ** 2) / (2 * slit_width**2))
        slit_2 = xp.exp(-(Y**2 + (X + slit_distance / 2) ** 2) / (2 * slit_width**2))
        amplitude = slit_1 + slit_2

    elif state_image == "single_shifted_slit":
        slit_distance = L / 1.5
        slit_width = L / 30.0
        amplitude = xp.exp(-(Y**2 + (X - slit_distance / 2) ** 2) / (2 * slit_width**2))

    elif state_image == "orbit":
        width = L / 8.0
        x_offset = -L / 3.0
        k_y = 6.0
        envelope = xp.exp(-((X - x_offset) ** 2 + Y**2) / (2 * width**2))
        phase_kick = xp.exp(1j * k_y * Y)
        amplitude = envelope * phase_kick

    elif state_image == "vortex":
        width = L / 4.0
        envelope = xp.exp(-(X**2 + Y**2) / (2 * width**2))
        # The (X + iY) term creates the spinning rainbow phase
        amplitude = (X + 1j * Y) * envelope

    elif state_image == "lattice":
        amplitude = xp.zeros((N, N), dtype=target_dtype, device=device)
        width = L / 20.0
        spacing = L / 2.5
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                x0 = i * spacing
                y0 = j * spacing
                spot = xp.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2 * width**2))
                amplitude = amplitude + xp.astype(spot, target_dtype)

    elif state_image in SAMPLE_IMAGES or state_image in SAMPLE_SHIFTED_IMAGES:
        host_array = sample_image(name=state_image, N=N)

        # Ingest directly into the target Array API namespace
        amplitude = xp.asarray(host_array, device=device)

    else:
        raise ValueError(
            f"unknown state_image {state_image!r}; expected one of "
            "'cat_state', 'double_slit', 'orbit', 'vortex', 'lattice', or"
            f"one of the sample images: {', '.join(SAMPLE_IMAGES)} or their"
            " shifted variants (e.g. 'horse_shifted')."
        )

    # Cast to the final complex target type
    psi = xp.asarray(amplitude, dtype=target_dtype, device=device)

    # The vortex carries phase winding, applied after the real envelope.
    if state_image == "vortex":
        psi = psi * xp.exp(1j * xp.atan2(Y, X))

    if blur > 0:
        psi = _gaussian_blur(psi, xp, blur)

    norm = xp.linalg.vector_norm(psi)
    return psi / norm


def _gaussian_blur(psi: Array, xp, sigma_pixels: float) -> Array:
    """
    Applies a spatial Gaussian blur using the Convolution Theorem (via FFT).

    Parameters
    ----------
    psi : Array
        The quantum state to be blurred.
    xp : module
        The Array API namespace.
    sigma_pixels : float
        The standard deviation of the Gaussian blur in spatial pixels.
    """
    n = psi.shape[-1]

    # Frequencies in cycles per pixel
    f = xp.fft.fftfreq(n, d=1.0, device=getattr(psi, "device", None))
    Fx, Fy = xp.meshgrid(f, f, indexing="xy")

    # The Fourier transform of a spatial Gaussian with std dev `sigma`
    # is a frequency-domain Gaussian: F{ exp(-x^2 / 2*sigma^2) } = exp(-2 * pi^2 * sigma^2 * f^2)
    window = xp.exp(-2 * math.pi**2 * sigma_pixels**2 * (Fx**2 + Fy**2))

    # Apply the blur in momentum space and return to real space
    return xp.fft.ifft2(xp.fft.fft2(psi) * xp.astype(window, psi.dtype))
