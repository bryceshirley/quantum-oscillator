from array_api_compat import get_namespace

from toolbox_talk.operators_noncompat import agnostic_expm
from toolbox_talk.physics import apply_hamiltonian
from toolbox_talk.utils import Array


def split_operator_step(psi: Array, V: Array, K: Array, dt: float) -> Array:
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


def arnoldi_step(
    psi: Array, V: Array, K: Array, dt: float, tol: float = 1e-1, n_krylov: int = 500
) -> Array:
    """
    Projects a massive operator into a tiny Krylov subspace, evaluating
    at each step to dynamically check for convergence, and returns the propagated state.

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
    tol : float, optional
        The tolerance for convergence in the Krylov subspace.
    n_krylov : int, optional
        The maximum number of Krylov iterations to perform.

    Returns
    -------
    psi_propagated : Array
        The evolved quantum state after applying the Arnoldi step.
    """
    if n_krylov < 1:
        raise ValueError("n_krylov must be at least 1")

    def operator(psi):
        """Wrapper to apply the Hamiltonian operator."""
        return apply_hamiltonian(psi, V, K)

    xp = get_namespace(psi, V)
    device = getattr(psi, "device", None)
    dtype = psi.dtype

    norm_psi = xp.linalg.vector_norm(psi)
    q = psi / norm_psi

    Q_cols = [q]
    h_cols = []

    # Declared optional because they are only assigned inside the loop; the
    # guard after the loop is what lets the type checker (and the reader) know
    # they are populated by the time they are used.
    H_k_final: Array | None = None
    F_k: Array | None = None

    for k in range(n_krylov):
        v = operator(Q_cols[k])

        h_col = []
        for j in range(k + 1):
            # Compute conjugate on the fly to save memory
            h_jk = xp.sum(xp.conj(Q_cols[j]) * v)
            h_col.append(h_jk)
            v = v - h_jk * Q_cols[j]

        h_next = xp.linalg.vector_norm(v)

        h_col.append(xp.asarray(h_next, dtype=dtype, device=device))
        h_cols.append(h_col)

        # A vanishing subdiagonal means the Krylov space is exhausted: the
        # subspace is already invariant, so the answer is exact here. Dividing
        # by it would produce NaNs.
        breakdown = float(h_next) < 1e-12

        # ---------------------------------------------------------
        # BUILD CURRENT (k+1) x (k+1) HESSENBERG MATRIX
        # ---------------------------------------------------------
        H_k_cols = []
        for c in range(k + 1):
            col_data = h_cols[c][: k + 1]
            pad_len = (k + 1) - len(col_data)

            if pad_len > 0:
                pad = xp.zeros((pad_len,), dtype=dtype, device=device)
                col_padded = xp.concat([xp.stack(col_data), pad])
            else:
                col_padded = xp.stack(col_data)

            H_k_cols.append(col_padded)

        H_k = xp.stack(H_k_cols, axis=1)
        H_k_final = H_k

        # ---------------------------------------------------------
        # CONVERGENCE CHECK (every 20 iterations, after the 10th)
        # ---------------------------------------------------------
        if k > 10 and k % 20 == 0:
            F_k = agnostic_expm(dt, H_k)
            F_k_elem = F_k[k, 0]

            err = float(norm_psi) * float(h_next) * float(xp.abs(F_k_elem))

            if err < tol or breakdown:
                if not breakdown:
                    Q_cols.append(v / h_next)
                break

        if breakdown:
            break

        q_next = v / h_next
        Q_cols.append(q_next)

    actual_k = len(h_cols)

    # ---------------------------------------------------------
    # RECONSTRUCT THE PROPAGATED WAVEFUNCTION
    # ---------------------------------------------------------

    if H_k_final is None:  # unreachable given n_krylov >= 1, but proves it
        raise RuntimeError("Arnoldi produced no Hessenberg matrix")

    if F_k is None or F_k.shape[0] != actual_k:
        F_k = agnostic_expm(dt, H_k_final)

    psi_propagated = xp.zeros_like(psi)

    # \psi_new = ||\psi|| * \sum (Q_c * F_k[c, 0])
    for c in range(actual_k):
        psi_propagated = psi_propagated + Q_cols[c] * F_k[c, 0]

    psi_propagated = psi_propagated * norm_psi

    # Python's garbage collector will now automatically destroy Q_cols and H_k_final
    return psi_propagated
