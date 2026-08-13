"""Property-based unit tests for the time-evolution operators.

Reduced to focus on backend compatibility and basic physics properties.
"""

from __future__ import annotations

import numpy as np
import pytest

from toolbox_talk.operators import (
    forward_euler_step,
    lie_trotter_step,
)
from toolbox_talk.operators_extra import (
    arnoldi_step,
)
from toolbox_talk.physics import get_propagators
from toolbox_talk.utils import resolve_backend, to_host

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


ALL_STEPS = [lie_trotter_step, forward_euler_step, arnoldi_step]

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


def test_lie_trotter_step_is_exactly_unitary(device, backend_cfg, propagators):
    """Each factor is a pure phase, so the norm is conserved."""
    V, K = propagators
    psi = gaussian(backend_cfg)
    out = lie_trotter_step(psi, V, K, 0.5)

    tol = 1e-4 if single_precision(device) else 1e-9
    assert float(backend_cfg.xp.linalg.vector_norm(out)) == pytest.approx(1.0, abs=tol)


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
