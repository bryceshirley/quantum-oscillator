"""Property-based unit tests for the time-evolution operators.

Reduced to focus on backend compatibility and basic physics properties.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from quantum_oscillator.operators import (
    conjugate_strang_step,
    evolve,
    forward_euler_step,
    lie_trotter_step,
    strang_step,
    suzuki_trotter_step,
)
from quantum_oscillator.physics import get_propagators
from quantum_oscillator.utils import resolve_backend, to_host

N = 32
L = 8.0

# --------------------------------------------------------------------------
# backends
# --------------------------------------------------------------------------


def _available_backends():
    """Provide string names for the backends to test."""
    backends = ["numpy", "torch"]
    return backends


@pytest.fixture(params=_available_backends())
def backend_name(request):
    """The string name of the backend (e.g., 'numpy' or 'torch')."""
    return request.param


@pytest.fixture
def backend_cfg(backend_name):
    """The resolved Backend namedtuple (xp, device, real, complex) from utils."""
    return resolve_backend(backend_name)


@pytest.fixture
def xp(backend_cfg):
    return backend_cfg.xp


@pytest.fixture
def device(backend_cfg):
    return backend_cfg.device


def single_precision(device) -> bool:
    """Metal has no FP64 at the hardware level, so MPS is float32 only."""
    return device in ("mps", "cuda")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def grid(backend_cfg):
    x = backend_cfg.xp.linspace(
        -L, L, N, endpoint=False, device=backend_cfg.device, dtype=backend_cfg.real
    )
    return backend_cfg.xp.meshgrid(x, x, indexing="xy")


def gaussian(backend_cfg, x0=1.5, sigma=1.0):
    """A smooth, well-resolved packet that has decayed to ~0 at the boundary."""
    X, Y = grid(backend_cfg)
    xp = backend_cfg.xp
    cdt = backend_cfg.complex

    psi = xp.astype(xp.exp(-((X - x0) ** 2 + Y**2) / (2 * sigma**2)), cdt)
    return psi / xp.astype(xp.linalg.vector_norm(psi), cdt)


def fidelity(a, b) -> float:
    """|<a|b>| for normalised states; 1.0 means identical up to global phase."""
    a, b = to_host(a).ravel(), to_host(b).ravel()
    return float(abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b)))


@pytest.fixture
def propagators(backend_name):
    """V and K at an EXPLICIT dtype. Passes the backend string for internal resolution."""
    return get_propagators(N, L, backend_name)


ALL_STEPS = [
    lie_trotter_step,
    strang_step,
    conjugate_strang_step,
    suzuki_trotter_step,
    forward_euler_step,
]

# Every factor in a splitting method is a pure phase, so these are unitary
# to machine precision at any dt.
SPLITTING_STEPS = [
    lie_trotter_step,
    strang_step,
    conjugate_strang_step,
    suzuki_trotter_step,
]

# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------


@pytest.mark.parametrize("step", ALL_STEPS, ids=lambda f: f.__name__)
def test_step_preserves_shape_and_dtype(step, backend_cfg, propagators):
    """Ensure the operator returns a state of the correct shape and type."""
    V, K = propagators
    psi = gaussian(backend_cfg)
    out = step(psi, V, K, 0.01)

    assert out.shape == psi.shape
    assert out.dtype == psi.dtype


@pytest.mark.parametrize("step", SPLITTING_STEPS, ids=lambda f: f.__name__)
def test_splitting_step_is_exactly_unitary(step, device, backend_cfg, propagators):
    """Each factor is a pure phase, so the norm is conserved."""
    V, K = propagators
    psi = gaussian(backend_cfg)
    out = step(psi, V, K, 0.5)

    tol = 1e-4 if single_precision(device) else 1e-9
    assert float(backend_cfg.xp.linalg.vector_norm(out)) == pytest.approx(1.0, abs=tol)


def test_suzuki_trotter_is_fourth_order(device, backend_cfg, propagators):
    """Halving dt should shrink the error by ~2^4; require a safe margin."""
    if single_precision(device):
        pytest.skip("convergence-order measurement needs double precision")

    V, K = propagators
    psi0 = gaussian(backend_cfg)
    T = 1.0

    def evolve(n_steps):
        psi = psi0
        for _ in range(n_steps):
            psi = suzuki_trotter_step(psi, V, K, T / n_steps)
        return to_host(psi).ravel()

    ref = evolve(128)
    err_coarse = np.linalg.norm(evolve(4) - ref)
    err_fine = np.linalg.norm(evolve(8) - ref)

    assert err_coarse / err_fine > 10


def test_suzuki_trotter_rejects_bad_order(backend_cfg, propagators):
    V, K = propagators
    psi = gaussian(backend_cfg)

    for bad_order in (0, 3, -2):
        with pytest.raises(ValueError):
            suzuki_trotter_step(psi, V, K, 0.01, order=bad_order)


def test_evolve_matches_manual_stepping(backend_cfg, propagators):
    """evolve(T, n_steps) is exactly n_steps equal applications of step_fn."""
    V, K = propagators
    psi0 = gaussian(backend_cfg)
    T, n_steps = 0.4, 5

    manual = psi0
    for _ in range(n_steps):
        manual = strang_step(manual, V, K, T / n_steps)
    driven = evolve(psi0, V, K, T, n_steps, step_fn=strang_step)

    np.testing.assert_array_equal(to_host(driven), to_host(manual))


def test_evolve_defaults_to_lie_trotter(backend_cfg, propagators):
    V, K = propagators
    psi0 = gaussian(backend_cfg)

    driven = evolve(psi0, V, K, 0.1, 1)
    manual = lie_trotter_step(psi0, V, K, 0.1)

    np.testing.assert_array_equal(to_host(driven), to_host(manual))


def test_evolve_track_norm(backend_cfg, propagators):
    """track_norm returns (psi, norms) with the initial norm first."""
    V, K = propagators
    psi0 = gaussian(backend_cfg)
    n_steps = 5

    plain = evolve(psi0, V, K, 0.4, n_steps, step_fn=strang_step)
    tracked, norms = evolve(
        psi0, V, K, 0.4, n_steps, step_fn=strang_step, track_norm=True
    )

    np.testing.assert_array_equal(to_host(tracked), to_host(plain))
    assert len(norms) == n_steps + 1
    # a splitting method conserves the norm at every recorded point
    np.testing.assert_allclose(norms, norms[0], rtol=1e-5)

    # forward Euler must grow the norm on every step
    _, euler_norms = evolve(
        psi0, V, K, 0.4, n_steps, step_fn=forward_euler_step, track_norm=True
    )
    assert all(b > a for a, b in itertools.pairwise(euler_norms))


def test_backends_agree(propagators, device, backend_cfg):
    """The same algorithm on a different backend must give the same physics."""
    V, K = propagators
    psi = gaussian(backend_cfg)

    # Evolve with the parameterized backend (Torch CPU, MPS, etc.)
    out = lie_trotter_step(psi, V, K, 0.02)

    # Evolve with NumPy as the baseline to check against
    numpy_cfg = resolve_backend("numpy")
    Vn, Kn = get_propagators(N, L, "numpy")
    psin = gaussian(numpy_cfg)
    out_n = lie_trotter_step(psin, Vn, Kn, 0.02)

    threshold = 0.9999 if single_precision(device) else 0.999999
    assert fidelity(out, out_n) > threshold
