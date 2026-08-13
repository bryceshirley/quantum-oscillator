from typing import cast

import numpy as np
import scipy
from array_api_compat import get_namespace

from toolbox_talk.physics import apply_hamiltonian
from toolbox_talk.utils import Array, from_numpy, to_numpy


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
            F_k = _agnostic_expm(dt, H_k)
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
        F_k = _agnostic_expm(dt, H_k_final)

    psi_propagated = xp.zeros_like(psi)

    # \psi_new = ||\psi|| * \sum (Q_c * F_k[c, 0])
    for c in range(actual_k):
        psi_propagated = psi_propagated + Q_cols[c] * F_k[c, 0]

    psi_propagated = psi_propagated * norm_psi

    # Python's garbage collector will now automatically destroy Q_cols and H_k_final
    return psi_propagated


def crank_nicolson_step(psi: Array, V: Array, K: Array, dt: float) -> Array:
    """
    Crank-Nicolson, [1,1] Padé approximant of the exponential of the Hamiltonian.

    Unitary and stable.
    Uses SciPy's GMRES on the CPU for the solver algorithm, but safely
    dispatches the heavy FFT Hamiltonian operations to PyTorch MPS / GPU.

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
        The evolved quantum state after applying the Crank-Nicolson step.
    """
    xp = get_namespace(psi, V)
    device = getattr(psi, "device", None)
    shape = psi.shape
    total_elements = int(np.prod(shape))

    def lhs_operator(vec_np):
        # 1. Bring flat CPU numpy vector to device (e.g., MPS) and reshape
        psi_vec = xp.reshape(
            from_numpy(vec_np, xp, dtype=psi.dtype, device=device), shape
        )

        # 2. Perform the heavy FFT operations on the GPU
        H_psi = apply_hamiltonian(psi_vec, V, K)
        res = psi_vec + 0.5j * dt * H_psi

        # 3. Send the flat result back to the CPU for SciPy
        return to_numpy(xp.reshape(res, (-1,)), xp)

    # SciPy requires standard complex Python types
    linear_op = scipy.sparse.linalg.LinearOperator(
        (total_elements, total_elements), matvec=lhs_operator, dtype=complex
    )

    # Compute the Right-Hand Side entirely on the device, then move to CPU
    H_psi_old = apply_hamiltonian(psi, V, K)
    rhs_device = psi - 0.5j * dt * H_psi_old
    rhs_np = to_numpy(xp.reshape(rhs_device, (-1,)), xp)

    # Solve the massive linear system using SciPy (bouncing to device inside matvec)
    psi_new_flat_np, info = scipy.sparse.linalg.gmres(linear_op, rhs_np, rtol=1e-4)
    if info != 0:
        raise RuntimeError(
            f"GMRES did not converge in crank_nicolson_step (info={info}); "
            "the step is not trustworthy."
        )

    # Bring the final converged solution back to the device and reshape
    psi_new = xp.reshape(
        from_numpy(psi_new_flat_np, xp, dtype=psi.dtype, device=device), shape
    )

    return psi_new


def _agnostic_expm(dt: float, matrix: Array) -> Array:
    """
    Computes the matrix exponential of a given matrix scaled by -1j * dt.
    This function is agnostic to the array namespace (NumPy, PyTorch, etc.)

    Parameters
    ----------
    dt : float
        The time step for evolution.
    matrix : Array
        The matrix to exponentiate.

    Returns
    -------
    exp_matrix : Array
        The matrix exponential of the scaled matrix.
    """
    xp = get_namespace(matrix)

    # Scale the Hessenberg matrix by the time step and imaginary unit!
    # This prevents the exponential from blowing up to infinity.
    scaled_matrix = -1j * dt * matrix

    if "torch" in xp.__name__:
        import torch

        # PyTorch MPS doesn't support matrix_exp yet, so the small k x k
        # exponential is computed on the CPU and sent straight back.
        # The casts are erased at runtime; they only tell the type checker that
        # a torch namespace implies torch tensors, which it cannot infer from
        # the Array protocol.
        cpu_mat = cast("torch.Tensor", scaled_matrix).cpu()
        exp_mat = torch.matrix_exp(cpu_mat)
        return cast(Array, exp_mat.to(cast("torch.Tensor", matrix).device))
    else:
        std_matrix = np.asarray(scaled_matrix)
        # Ensure it returns an array on the correct device/namespace
        device = getattr(matrix, "device", None)
        return xp.asarray(
            scipy.linalg.expm(std_matrix), dtype=matrix.dtype, device=device
        )
