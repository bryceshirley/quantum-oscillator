from array_api_compat import get_namespace

from toolbox_talk.physics import apply_hamiltonian
from toolbox_talk.utils import Array


def lie_trotter_step(psi: Array, V: Array, K: Array, dt: float) -> Array:
    """
    First-order Lie-Trotter Operator Splitting.

    Parameters
    ----------
    psi : Array
        The current quantum state.
    V : Array
        The potential energy operator.
    K : Array
        The diagonalized kinetic energy operator.
    dt : float
        The time step for evolution.

    Returns
    -------
    psi_new : Array
        The evolved quantum state after applying the split operator step.
    """
    # Get the appropriate array namespace (NumPy, PyTorch, etc.)
    xp = get_namespace(psi, V)

    # 1. Kinetic wave phase shift in momentum space
    # Momentum space -> apply exp(1j * dt * K) -> Real space
    psi_k = xp.fft.fft2(psi)
    psi_k = psi_k * xp.exp(1j * dt * K)
    psi = xp.fft.ifft2(psi_k)

    # 2. Potential wave phase shift in real space
    return psi * xp.exp(1j * dt * V)


def forward_euler_step(psi: Array, V: Array, K: Array, dt: float) -> Array:
    """
    First-order Taylor expansion. Fast, but physically unstable (not unitary).

    Parameters
    ----------
    psi : Array
        The current quantum state.
    V : Array
        The potential energy operator.
    K : Array
        The diagonalized kinetic energy operator.
    dt : float
        The time step for evolution.

    Returns
    -------
    psi_new : Array
        The evolved quantum state after applying the forward Euler step.
    """
    H_psi = apply_hamiltonian(psi, V, K)
    return psi + 1j * dt * H_psi
