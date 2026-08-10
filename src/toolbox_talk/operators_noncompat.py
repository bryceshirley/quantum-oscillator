import numpy as np
import scipy
from array_api_compat import get_namespace

from toolbox_talk.physics import apply_hamiltonian
from toolbox_talk.utils import from_numpy, to_numpy


def crank_nicolson_step(psi, V, K, dt):
    """
    Padé approximant. Unitary and stable.
    Uses SciPy's GMRES on the CPU for the solver algorithm, but safely
    dispatches the heavy FFT Hamiltonian operations to PyTorch MPS / GPU.
    """
    xp = get_namespace(psi, V)
    device = getattr(psi, "device", None)
    shape = psi.shape
    total_elements = np.prod(shape)

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
    psi_new_flat_np, _ = scipy.sparse.linalg.gmres(linear_op, rhs_np, rtol=1e-4)

    # Bring the final converged solution back to the device and reshape
    psi_new = xp.reshape(
        from_numpy(psi_new_flat_np, xp, dtype=psi.dtype, device=device), shape
    )

    return psi_new


def agnostic_expm(dt, matrix):
    xp = get_namespace(matrix)

    # Scale the Hessenberg matrix by the time step and imaginary unit!
    # This prevents the exponential from blowing up to infinity.
    scaled_matrix = -1j * dt * matrix

    if "torch" in xp.__name__:
        # PyTorch MPS doesn't support matrix_exp yet.
        cpu_mat = scaled_matrix.cpu()
        import torch

        exp_mat = torch.matrix_exp(cpu_mat)
        return exp_mat.to(matrix.device)
    else:
        std_matrix = np.asarray(scaled_matrix)
        # Ensure it returns an array on the correct device/namespace
        device = getattr(matrix, "device", None)
        return xp.asarray(
            scipy.linalg.expm(std_matrix), dtype=matrix.dtype, device=device
        )
