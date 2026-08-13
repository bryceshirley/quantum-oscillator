"""Spectral Hamiltonian and initial states, in pure Array API."""

from __future__ import annotations

import math

from array_api_compat import array_namespace

from toolbox_talk.utils import Array, resolve_backend


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


def get_propagators(N: int, L: float, backend: str = "numpy") -> tuple[Array, Array]:
    """
    Creates the 2D grid, Harmonic Oscillator potential, and initial state.
    Calculates the exact spectral momentum operator (K) using pure Array API.
    """
    backend_info = resolve_backend(backend)
    xp = backend_info.xp
    device = backend_info.device
    dtype = backend_info.complex

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
