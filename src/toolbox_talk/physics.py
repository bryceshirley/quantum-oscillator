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


def apply_hamiltonian(psi: Array, V: Array, K: Array) -> Array:
    """
    SPECTRAL HAMILTONIAN
    Uses Fast Fourier Transforms to compute exact continuous derivatives,
    eliminating grid dispersion entirely.
    """
    xp = get_namespace(psi, V)

    # Momentum space -> apply K -> Real space
    psi_k = xp.fft.fft2(psi)
    T_psi = xp.fft.ifft2(K * psi_k)

    return T_psi + V * psi


def get_initial_state(
    N: int, L: float, state_image: str = "double_slit", backend: str = "numpy"
):
    """
    Creates the 2D grid, Harmonic Oscillator potential, and initial state.
    Calculates the exact spectral momentum operator (K) for perfect derivatives.

    Parameters
    ----------
    N : int
        The number of grid points in each dimension.
    L : float
        The spatial extent of the simulation grid.
    state_image : str
        The type of initial state to create ("horse", "shifted_horse", "cosine", "double_slit").
    backend : str
        The backend to use for array computations ('numpy' or 'torch').

    Returns
    -------
    initial_state : Array
        The initial quantum state represented as a 2D array.
    """
    x = np.linspace(-L, L, N, endpoint=False)
    X, Y = np.meshgrid(x, x)

    if state_image in ["horse", "shifted_horse", "no_blur_horse"]:
        # Sigma for Gaussian smoothing of the horse image
        if state_image == "no_blur_horse":
            sigma = 0.0  # No Gaussian smoothing
        else:
            sigma = 10.0  # Standard deviation for Gaussian smoothing

        # 1. Load and prepare the horse
        raw_img = data.horse().astype(np.float64)
        raw_img = np.max(raw_img) - raw_img
        raw_img = scipy.ndimage.gaussian_filter(raw_img, sigma=sigma)

        target_size = N // 2
        zoom_factor = target_size / raw_img.shape[0]
        resized_img = scipy.ndimage.zoom(raw_img, zoom_factor)

        base_state = np.zeros((N, N), dtype=np.complex128)
        actual_h, actual_w = resized_img.shape

        start_y = (N - actual_h) // 2
        start_x = (N - actual_w) // 2

        # Shift the horse to the right
        if state_image == "shifted_horse":
            start_x += N // 6

        base_state[start_y : start_y + actual_h, start_x : start_x + actual_w] = (
            resized_img
        )

    elif state_image == "cosine":
        # A broad Gaussian envelope to prevent FFT boundary artifacts
        envelope = np.exp(-(X**2 + Y**2) / (2 * (L / 3) ** 2))

        # Cosine wave oscillating along the x-axis
        frequency = 3.0  # Adjust frequency as needed
        base_state = (np.cos(frequency * X) * envelope).astype(np.complex128)

    elif state_image == "double_slit":
        # Distance between the two slits and their width
        slit_distance = L / 1.5
        slit_width = L / 30.0

        # Create two identical Gaussian wave packets shifted along the X-axis
        slit_1 = np.exp(-(Y**2 + (X - slit_distance / 2) ** 2) / (2 * slit_width**2))
        slit_2 = np.exp(-(Y**2 + (X + slit_distance / 2) ** 2) / (2 * slit_width**2))

        # The state is a quantum superposition of being in both slits simultaneously
        base_state = (slit_1 + slit_2).astype(np.complex128)

    elif state_image == "single_shifted_slit":
        # Distance between the a slits and their width
        slit_distance = L / 1.5
        slit_width = L / 30.0

        # Create a Gaussian wave packets shifted along the X-axis
        slit = np.exp(-(Y**2 + (X - slit_distance / 2) ** 2) / (2 * slit_width**2))

        base_state = (slit).astype(np.complex128)

    elif state_image == "cat_state":
        # Two Gaussian wave packets placed at opposite edges
        width = L / 8.0
        offset = L / 2.0
        packet_left = np.exp(-((X + offset) ** 2 + Y**2) / (2 * width**2))
        packet_right = np.exp(-((X - offset) ** 2 + Y**2) / (2 * width**2))
        base_state = (packet_left + packet_right).astype(np.complex128)

    elif state_image == "vortex":
        # A Laguerre-Gaussian beam (Optical Vortex)
        # The (X + iY) term creates the spinning rainbow phase
        width = L / 4.0
        envelope = np.exp(-(X**2 + Y**2) / (2 * width**2))
        base_state = ((X + 1j * Y) * envelope).astype(np.complex128)

    elif state_image == "orbit":
        # A single Gaussian packet given a strong initial momentum kick
        width = L / 8.0
        x_offset = -L / 3.0
        k_y = 6.0  # Momentum kick in the Y direction

        # The physical envelope (Position)
        envelope = np.exp(-((X - x_offset) ** 2 + Y**2) / (2 * width**2))

        # The momentum kick (Phase Gradient)
        phase_kick = np.exp(1j * k_y * Y)

        base_state = (envelope * phase_kick).astype(np.complex128)

    elif state_image == "lattice":
        # A 3x3 crystal of soft Gaussian wave packets
        base_state = np.zeros((N, N), dtype=np.complex128)
        width = L / 20.0
        spacing = L / 2.5

        # Build the grid
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                x0 = i * spacing
                y0 = j * spacing
                spot = np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2 * width**2))
                base_state += spot.astype(np.complex128)

    else:
        raise ValueError(
            f"Unsupported state_image: '{state_image}'. Supported options are: "
            "'horse', 'shifted_horse', 'cosine','vortex' or 'double_slit'."
        )

    # Normalize the total probability to 1.0
    base_state = base_state / np.linalg.norm(base_state)

    if backend == "torch":
        torch_comp = torch.complex64 if DEVICE == "mps" else torch.complex128
        base_state = torch.tensor(base_state, dtype=torch_comp, device=DEVICE)

    return base_state


def get_propagators(N: int, L: float, backend: str = "numpy") -> tuple[Array, Array]:
    """
    Creates the 2D grid, Harmonic Oscillator potential, and initial state.
    Calculates the exact spectral momentum operator (K) for perfect derivatives.

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
    K : Array
        The diagonalized kinetic energy operator.
    """
    x = np.linspace(-L, L, N, endpoint=False)
    X, Y = np.meshgrid(x, x)

    # 1. Harmonic Oscillator Potential
    V = -0.5 * (X**2 + Y**2)

    # 2. Spectral Kinetic Energy Operator (K)
    dx_val = 2 * L / N
    kx = 2 * np.pi * np.fft.fftfreq(N, d=dx_val)
    Kx, Ky = np.meshgrid(kx, kx)
    K = -0.5 * (Kx**2 + Ky**2)  # Momentum space kinetic energy operator

    if backend == "torch":
        torch_real = torch.float32 if DEVICE == "mps" else torch.float64
        xp_V = torch.tensor(V, dtype=torch_real, device=DEVICE)
        xp_K = torch.tensor(K, dtype=torch_real, device=DEVICE)
        return xp_V, xp_K
    else:
        return V, K
