"""Property-based unit tests for the time-evolution operators.

The tests are written against the *physics*, not against a saved output: a
propagator for H = T + V must be unitary, must reproduce exp(iHt) on an
eigenstate, and every method must agree with every other in the small-dt
limit. That last one is what catches sign errors, which norm-conservation
alone cannot see (a backwards step is just as unitary as a forwards one).

Precision is chosen per device rather than hardcoded: Metal has no FP64, so
every dtype and tolerance here is derived from `single_precision(device)`.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from toolbox_talk.operators_compat import (
    arnoldi_step,
    forward_euler_step,
    split_operator_step,
)
from toolbox_talk.operators_noncompat import agnostic_expm
from toolbox_talk.physics import get_propagators

N = 32
L = 8.0

# --------------------------------------------------------------------------
# backends
# --------------------------------------------------------------------------


def _backends():
    """Every installed backend, so the same physics runs on CPU and GPU."""
    import numpy as _np

    out = [pytest.param((_np, None), id="numpy")]
    try:
        import array_api_compat.torch as _torch_xp
        import torch

        out.append(pytest.param((_torch_xp, "cpu"), id="torch-cpu"))
        if torch.backends.mps.is_available():
            out.append(pytest.param((_torch_xp, "mps"), id="torch-mps"))
        if torch.cuda.is_available():
            out.append(pytest.param((_torch_xp, "cuda"), id="torch-cuda"))
    except ImportError:
        pass
    return out


@pytest.fixture(params=_backends())
def backend(request):
    return request.param


@pytest.fixture
def xp(backend):
    return backend[0]


@pytest.fixture
def device(backend):
    return backend[1]


# --------------------------------------------------------------------------
# precision
# --------------------------------------------------------------------------


def single_precision(device) -> bool:
    """Metal has no FP64 at the hardware level, so MPS is float32 only."""
    return device == "mps"


def real_dtype(xp, device):
    return xp.float32 if single_precision(device) else xp.float64


def complex_dtype(xp, device):
    return xp.complex64 if single_precision(device) else xp.complex128


@pytest.fixture
def tol(device):
    """Round-off floor for the active precision: ~1e-7 for f32, ~1e-16 for f64."""
    return 1e-4 if single_precision(device) else 1e-9


@pytest.fixture
def atol(device):
    """Elementwise comparison tolerance for the active precision."""
    return 1e-5 if single_precision(device) else 1e-9


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def host(a) -> np.ndarray:
    """Bring any backend array back to NumPy for assertions."""
    if type(a).__module__.split(".")[0] == "torch":
        return a.detach().cpu().numpy()
    return np.asarray(a)


def to_backend(m: np.ndarray, xp, device):
    """Put a host matrix on the backend at the precision that device supports."""
    np_dtype = np.complex64 if single_precision(device) else np.complex128
    return xp.asarray(np.asarray(m, dtype=np_dtype), device=device)


def grid(xp, device=None):
    x = xp.linspace(
        -L, L, N, endpoint=False, device=device, dtype=real_dtype(xp, device)
    )
    return xp.meshgrid(x, x, indexing="xy")


def gaussian(xp, device=None, x0=1.5, sigma=1.0, kick=0.0):
    """A smooth, well-resolved packet that has decayed to ~0 at the boundary."""
    cdt = complex_dtype(xp, device)
    X, Y = grid(xp, device)
    psi = xp.astype(xp.exp(-((X - x0) ** 2 + Y**2) / (2 * sigma**2)), cdt)
    if kick:
        psi = psi * xp.exp(1j * kick * xp.astype(X, cdt))
    return psi / xp.astype(xp.linalg.vector_norm(psi), cdt)


def ground_state(xp, device=None):
    """Ground state of the 2D oscillator: an eigenstate, so only its phase moves."""
    cdt = complex_dtype(xp, device)
    X, Y = grid(xp, device)
    psi = xp.astype(xp.exp(-(X**2 + Y**2) / 2), cdt)
    return psi / xp.astype(xp.linalg.vector_norm(psi), cdt)


def evolve(step, psi, V, K, dt, n):
    for _ in range(n):
        psi = step(psi, V, K, dt)
    return psi


def reference(psi, V, K, T, n=2000):
    """High-accuracy propagator: many tiny split-operator steps."""
    return evolve(split_operator_step, psi, V, K, T / n, n)


def fidelity(a, b) -> float:
    """|<a|b>| for normalised states; 1.0 means identical up to global phase."""
    a, b = host(a).ravel(), host(b).ravel()
    return float(abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b)))


@pytest.fixture
def propagators(xp, device):
    """V and K at an EXPLICIT dtype.

    Passing dtype is not optional here. torch.linspace defaults to
    torch.get_default_dtype() (float32) while numpy.linspace defaults to
    float64, so leaving dtype=None silently runs torch-cpu at half the
    precision of numpy -- identical code, different answers.
    """
    return get_propagators(N, L, xp, device=device, dtype=real_dtype(xp, device))


ALL_STEPS = [split_operator_step, forward_euler_step, arnoldi_step]


# --------------------------------------------------------------------------
# shape / dtype contract
# --------------------------------------------------------------------------


@pytest.mark.parametrize("step", ALL_STEPS, ids=lambda f: f.__name__)
def test_step_preserves_shape_and_dtype(step, xp, device, propagators):
    V, K = propagators
    psi = gaussian(xp, device)
    out = step(psi, V, K, 0.01)
    assert out.shape == psi.shape
    assert out.dtype == psi.dtype


@pytest.mark.parametrize("step", ALL_STEPS, ids=lambda f: f.__name__)
def test_step_is_linear(step, xp, device, propagators, atol):
    """The Schrodinger equation is linear: stepping a sum == summing the steps."""
    V, K = propagators
    a = gaussian(xp, device, x0=1.5)
    b = gaussian(xp, device, x0=-2.0, sigma=0.8)
    dt = 0.01
    lhs = step(a + 2.0 * b, V, K, dt)
    rhs = step(a, V, K, dt) + 2.0 * step(b, V, K, dt)
    assert np.allclose(host(lhs), host(rhs), atol=max(atol, 1e-6))


# --------------------------------------------------------------------------
# unitarity
# --------------------------------------------------------------------------


@pytest.mark.parametrize("dt", [0.001, 0.05, 0.5, 2.0])
def test_split_operator_is_exactly_unitary(dt, xp, device, propagators, tol):
    """Each factor is a pure phase, so the norm is conserved at ANY step size."""
    V, K = propagators
    psi = gaussian(xp, device)
    out = evolve(split_operator_step, psi, V, K, dt, 10)
    assert float(xp.linalg.vector_norm(out)) == pytest.approx(1.0, abs=tol)


def test_arnoldi_is_unitary(xp, device, propagators, tol):
    V, K = propagators
    psi = gaussian(xp, device)
    out = evolve(arnoldi_step, psi, V, K, 0.05, 5)
    assert float(xp.linalg.vector_norm(out)) == pytest.approx(1.0, abs=max(tol, 1e-8))


def test_forward_euler_grows_the_norm(xp, device, propagators):
    """|1 + iE dt| = sqrt(1 + E^2 dt^2) > 1, so Euler gains norm every step.

    This is the defining defect of the method, not a tolerance issue: it is
    monotone and it never goes away, however small dt is.
    """
    V, K = propagators
    psi = gaussian(xp, device)
    norms = []
    for _ in range(20):
        psi = forward_euler_step(psi, V, K, 0.01)
        norms.append(float(xp.linalg.vector_norm(psi)))
    assert norms[0] > 1.0
    assert all(b > a for a, b in itertools.pairwise(norms))


def test_forward_euler_growth_shrinks_with_dt(xp, device, propagators):
    """Halving dt delays the blow-up but never prevents it."""
    V, K = propagators
    growth = []
    for dt in (0.02, 0.01, 0.005):
        psi = evolve(forward_euler_step, gaussian(xp, device), V, K, dt, int(0.2 / dt))
        growth.append(float(xp.linalg.vector_norm(psi)) - 1.0)
    assert all(g > 0 for g in growth)
    assert growth[0] > growth[1] > growth[2]


# --------------------------------------------------------------------------
# correctness against known solutions
# --------------------------------------------------------------------------


@pytest.mark.parametrize("step", ALL_STEPS, ids=lambda f: f.__name__)
def test_eigenstate_only_acquires_phase(step, xp, device, propagators):
    """The ground state is an eigenstate of H, so |psi| must not change."""
    V, K = propagators
    psi = ground_state(xp, device)
    out = step(psi, V, K, 0.01)
    assert np.allclose(np.abs(host(out)), np.abs(host(psi)), atol=2e-4)


@pytest.mark.parametrize("step", ALL_STEPS, ids=lambda f: f.__name__)
def test_agrees_with_reference_propagator(step, xp, device, propagators):
    """Every method must approximate the same exp(iHt) over a single step.

    Norm conservation cannot catch a wrong sign -- exp(-iHt) is exactly as
    unitary as exp(+iHt) -- so this comparison against an independently
    computed forward solution is the test that pins the direction of time.
    """
    V, K = propagators
    psi = gaussian(xp, device)
    dt = 0.05
    expected = reference(psi, V, K, dt)
    assert fidelity(step(psi, V, K, dt), expected) == pytest.approx(1.0, abs=1e-2)


def test_split_operator_converges_at_first_order(xp, device, propagators):
    """Lie-Trotter is O(dt): refining the step must reduce the error."""
    V, K = propagators
    psi = gaussian(xp, device)
    T = 0.1
    expected = reference(psi, V, K, T)
    errors = []
    for n in (25, 50, 100):
        out = evolve(split_operator_step, psi, V, K, T / n, n)
        errors.append(1.0 - fidelity(out, expected))
    assert errors[0] > errors[1] > errors[2]
    assert errors[2] < 1e-4


# --------------------------------------------------------------------------
# the physics the talk is actually about
# --------------------------------------------------------------------------


def test_half_period_inverts_the_picture(xp, device, propagators):
    """At t = pi the oscillator maps psi(r) -> psi(-r): the horse turns over."""
    V, K = propagators
    psi = gaussian(xp, device, x0=2.0, sigma=0.9)
    out = evolve(split_operator_step, psi, V, K, np.pi / 2000, 2000)
    flipped = host(psi)[::-1, ::-1]
    # the grid excludes +L, so parity about x=0 is an index roll of one cell
    flipped = np.roll(flipped, shift=(1, 1), axis=(0, 1))
    assert fidelity(out, flipped) > 0.99


def test_full_period_returns_the_state(xp, device, propagators):
    """At t = 2pi the state comes back -- this is the whole point of the talk."""
    V, K = propagators
    psi = gaussian(xp, device, x0=2.0, sigma=0.9)
    out = evolve(split_operator_step, psi, V, K, 2 * np.pi / 4000, 4000)
    assert fidelity(out, psi) > 0.999


# --------------------------------------------------------------------------
# agnostic_expm
#
# The codebase convention is Psi(t) = exp(+iHt) Psi_0: split_operator_step
# applies exp(+1j*dt*V) and forward_euler_step computes psi + 1j*dt*H*psi.
# agnostic_expm must therefore compute exp(+i dt M) too.
# --------------------------------------------------------------------------


def test_agnostic_expm_of_zero_is_identity(xp, device):
    z = xp.zeros((4, 4), dtype=complex_dtype(xp, device), device=device)
    assert np.allclose(host(agnostic_expm(0.5, z)), np.eye(4), atol=1e-6)


def test_agnostic_expm_is_unitary_for_hermitian_input(xp, device):
    """exp(i dt H) is unitary when H is Hermitian -- the Hessenberg case."""
    rng = np.random.default_rng(1)
    a = rng.normal(size=(5, 5)) + 1j * rng.normal(size=(5, 5))
    h = a + a.conj().T
    u = host(agnostic_expm(0.4, to_backend(h, xp, device)))
    tolerance = 1e-4 if single_precision(device) else 1e-9
    assert np.allclose(u @ u.conj().T, np.eye(5), atol=tolerance)


# --------------------------------------------------------------------------
# arnoldi specifics
# --------------------------------------------------------------------------


def test_arnoldi_rejects_zero_krylov_dimension(xp, device, propagators):
    V, K = propagators
    with pytest.raises(ValueError):
        arnoldi_step(gaussian(xp, device), V, K, 0.01, n_krylov=0)


def test_arnoldi_handles_immediate_breakdown(xp, device, propagators):
    """An eigenstate exhausts the Krylov space at k=0; this must not divide by 0."""
    V, K = propagators
    out = arnoldi_step(ground_state(xp, device), V, K, 0.01)
    assert np.all(np.isfinite(host(out)))


# --------------------------------------------------------------------------
# cross-backend agreement
# --------------------------------------------------------------------------


def test_backends_agree(propagators, xp, device):
    """The same algorithm on a different backend must give the same physics.

    Single precision only buys ~7 digits, so the bar is set at the precision
    the device can actually deliver rather than at float64 agreement.
    """
    import numpy as _np

    V, K = propagators
    psi = gaussian(xp, device)
    out = evolve(split_operator_step, psi, V, K, 0.02, 10)

    Vn, Kn = get_propagators(N, L, _np, dtype=_np.float64)
    psin = host(gaussian(_np, None))
    out_n = evolve(split_operator_step, psin, Vn, Kn, 0.02, 10)

    threshold = 0.9999 if single_precision(device) else 0.999999
    assert fidelity(out, out_n) > threshold
