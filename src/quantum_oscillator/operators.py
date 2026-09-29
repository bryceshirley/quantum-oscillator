from typing import Literal, overload

from array_api_compat import array_namespace

from quantum_oscillator.physics import apply_hamiltonian
from quantum_oscillator.utils import Array, PropagatorFunc


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
    xp = array_namespace(psi, V)

    # 1. Kinetic wave phase shift in momentum space
    # Momentum space -> apply exp(1j * dt * K) -> Real space
    psi_k = xp.fft.fft2(psi)
    psi_k = psi_k * xp.exp(1j * dt * K)
    psi = xp.fft.ifft2(psi_k)

    # 2. Potential wave phase shift in real space
    return psi * xp.exp(1j * dt * V)


def strang_step(psi: Array, V: Array, K: Array, dt: float) -> Array:
    """
    Second-order Strang (symmetric) Operator Splitting.

    Applies a half potential kick, a full kinetic drift, and a second half
    potential kick. The symmetry cancels the leading error term of
    Lie-Trotter, giving O(dt^2) accuracy for the same number of FFTs.

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
    xp = array_namespace(psi, V)

    # 1. Half potential wave phase shift in real space
    psi = psi * xp.exp(0.5j * dt * V)

    # 2. Full kinetic wave phase shift in momentum space
    psi_k = xp.fft.fft2(psi)
    psi_k = psi_k * xp.exp(1j * dt * K)
    psi = xp.fft.ifft2(psi_k)

    # 3. Half potential wave phase shift in real space
    return psi * xp.exp(0.5j * dt * V)


def conjugate_strang_step(psi: Array, V: Array, K: Array, dt: float) -> Array:
    """
    Conjugate Strang splitting: the kinetic-first ordering.

    Applies a half kinetic drift, a full potential kick, and a second half
    kinetic drift -- the roles of V and K swapped relative to strang_step.
    Also second order, but its leading error term differs, so contrasting
    the two orderings makes the splitting error itself visible. Costs two
    FFT pairs per step instead of one.

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
    xp = array_namespace(psi, V)

    # 1. Half kinetic wave phase shift in momentum space
    psi_k = xp.fft.fft2(psi)
    psi = xp.fft.ifft2(psi_k * xp.exp(0.5j * dt * K))

    # 2. Full potential wave phase shift in real space
    psi = psi * xp.exp(1j * dt * V)

    # 3. Half kinetic wave phase shift in momentum space
    psi_k = xp.fft.fft2(psi)
    return xp.fft.ifft2(psi_k * xp.exp(0.5j * dt * K))


def suzuki_trotter_step(
    psi: Array, V: Array, K: Array, dt: float, order: int = 8
) -> Array:
    """
    High-order Suzuki-Trotter Operator Splitting.

    Builds an order-2k integrator by Suzuki's fractal composition of five
    order-(2k-2) steps with weights (p, p, 1 - 4p, p, p), where
    p = 1 / (4 - 4^(1/(2k-1))). The base case is the Strang step, so
    order=4 is the classic fourth-order Suzuki scheme (25 phase factors,
    10 FFT pairs per step).

    The middle sub-step runs backwards in time (1 - 4p < 0). For the
    real-time Schrodinger equation this is harmless: every factor remains a
    pure phase, so the scheme stays exactly unitary at any order.

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
    order : int, optional
        The order of accuracy; must be an even integer >= 2.

    Returns
    -------
    psi_new : Array
        The evolved quantum state after applying the split operator step.
    """
    if order < 2 or order % 2 != 0:
        raise ValueError(f"order must be an even integer >= 2, got {order}")

    if order == 2:
        return strang_step(psi, V, K, dt)

    p = 1.0 / (4.0 - 4.0 ** (1.0 / (order - 1)))
    for weight in (p, p, 1.0 - 4.0 * p, p, p):
        psi = suzuki_trotter_step(psi, V, K, weight * dt, order=order - 2)
    return psi


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


@overload
def evolve(
    psi: Array,
    V: Array,
    K: Array,
    T: float,
    n_steps: int,
    step_fn: PropagatorFunc = ...,
    track_norm: Literal[False] = ...,
) -> Array: ...


@overload
def evolve(
    psi: Array,
    V: Array,
    K: Array,
    T: float,
    n_steps: int,
    step_fn: PropagatorFunc = ...,
    track_norm: Literal[True] = ...,
) -> tuple[Array, list[float]]: ...


def evolve(
    psi: Array,
    V: Array,
    K: Array,
    T: float,
    n_steps: int,
    step_fn: PropagatorFunc = lie_trotter_step,
    track_norm: bool = False,
) -> Array | tuple[Array, list[float]]:
    """
    March a state forward by a total time T in n_steps equal steps.

    Parameters
    ----------
    psi : Array
        The initial quantum state.
    V : Array
        The potential energy operator.
    K : Array
        The diagonalized kinetic energy operator.
    T : float
        The total evolution time.
    n_steps : int
        The number of equal steps to take.
    step_fn : PropagatorFunc, optional
        The stepping operator to apply; defaults to ``lie_trotter_step``.
    track_norm : bool, optional
        Also record ``||psi||`` along the way. The recorded list starts with
        the initial norm, so it has ``n_steps + 1`` entries.

    Returns
    -------
    psi_new : Array
        The evolved quantum state at time T. If ``track_norm`` is set, the
        tuple ``(psi_new, norms)`` instead.

    Notes
    -----
    The steppers are purely functional (the input state is never mutated),
    so this loop is safe to differentiate through with torch autograd.
    """
    dt = T / n_steps
    if not track_norm:
        for _ in range(n_steps):
            psi = step_fn(psi, V, K, dt)
        return psi

    xp = array_namespace(psi, V)
    norms = [float(xp.linalg.vector_norm(psi))]
    for _ in range(n_steps):
        psi = step_fn(psi, V, K, dt)
        norms.append(float(xp.linalg.vector_norm(psi)))
    return psi, norms
