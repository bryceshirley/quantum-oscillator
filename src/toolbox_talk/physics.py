import numpy as np
import scipy.linalg
import scipy.ndimage
import torch
from array_api_compat import get_namespace
from skimage import data

from toolbox_talk.utils import Array

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
if DEVICE == "mps":
    DTYPE_COMPLEX = np.complex64
    DTYPE_REAL = np.float32
else:
    DTYPE_COMPLEX = np.complex128
    DTYPE_REAL = np.float64


def apply_hamiltonian(psi: Array, V: Array, K2: Array) -> Array:
    """
    SPECTRAL HAMILTONIAN
    Uses Fast Fourier Transforms to compute exact continuous derivatives,
    eliminating grid dispersion entirely.
    """
    xp = get_namespace(psi, V)

    # Momentum space -> apply K^2 / 2 -> Real space
    psi_k = xp.fft.fft2(psi)
    T_psi = xp.fft.ifft2(0.5 * K2 * psi_k)

    return T_psi + V * psi


def get_initial_state(
    N: int, sigma: float = 10.0, state_image: str = "horse", backend: str = "numpy"
) -> Array:
    """
    Creates the 2D grid, Harmonic Oscillator potential, and initial state.
    Calculates the exact spectral momentum operator (K2) for perfect derivatives.

    Parameters
    ----------
    N : int
        The number of grid points in each dimension.
    sigma : float
        The standard deviation for Gaussian smoothing of the horse image.
    state_image : str
        The type of initial state to create (currently only "horse" is supported).
    backend : str
        The backend to use for array computations ('numpy' or 'torch').

    Returns
    -------
    initial_state : Array
        The initial quantum state represented as a 2D array.
    """

    # 3. Load and prepare the horse
    if state_image == "horse":
        raw_img = data.horse().astype(np.float64)
    else:
        raise ValueError(
            f"Unsupported state_image: {state_image}. Only 'horse' is supported."
        )
    raw_img = np.max(raw_img) - raw_img
    raw_img = scipy.ndimage.gaussian_filter(raw_img, sigma=sigma)

    target_size = N // 2
    zoom_factor = target_size / raw_img.shape[0]
    resized_img = scipy.ndimage.zoom(raw_img, zoom_factor)

    horse = np.zeros((N, N), dtype=np.complex128)
    start_idx = (N - target_size) // 2
    actual_h, actual_w = resized_img.shape
    horse[start_idx : start_idx + actual_h, start_idx : start_idx + actual_w] = (
        resized_img
    )

    horse = horse / np.linalg.norm(horse)

    if backend == "torch":
        torch_comp = torch.complex64 if DEVICE == "mps" else torch.complex128

        horse = torch.tensor(horse, dtype=torch_comp, device=DEVICE)
        return horse
    else:
        return horse


def get_propagators(N: int, L: float, backend: str = "numpy") -> tuple[Array, Array]:
    """
    Creates the 2D grid, Harmonic Oscillator potential, and initial state.
    Calculates the exact spectral momentum operator (K2) for perfect derivatives.

    Parameters
    ----------
    N : int
        The number of grid points in each dimension.
    L : float
        The spatial extent of the simulation grid.
    backend : str
        The backend to use for array computations ('numpy' or 'torch').

    Returns
    -------
    V : Array
        The potential energy operator.
    K2 : Array
        The kinetic energy operator.
    """
    x = np.linspace(-L, L, N, endpoint=False)
    X, Y = np.meshgrid(x, x)

    # 1. Harmonic Oscillator Potential
    V = 0.5 * (X**2 + Y**2)

    # 2. Spectral Kinetic Energy Operator (K^2)
    dx_val = 2 * L / N
    kx = 2 * np.pi * np.fft.fftfreq(N, d=dx_val)
    Kx, Ky = np.meshgrid(kx, kx)
    K2 = Kx**2 + Ky**2  # Momentum squared

    if backend == "torch":
        torch_real = torch.float32 if DEVICE == "mps" else torch.float64
        xp_V = torch.tensor(V, dtype=torch_real, device=DEVICE)
        xp_K2 = torch.tensor(K2, dtype=torch_real, device=DEVICE)
        return xp_V, xp_K2
    else:
        return V, K2
