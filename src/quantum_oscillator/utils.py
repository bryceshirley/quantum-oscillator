"""Shared types and host/device helpers for the quantum oscillator.

The Array API standard deliberately does not ship a runtime Protocol for arrays
(see data-apis.org/array-api/latest/design_topics/static_typing.html), so a
hand-rolled structural type is the sanctioned approach. The rule that keeps it
honest: this Protocol may only contain members defined by *every* backend we
support. Anything torch-specific belongs behind a narrowing check, not in here.
"""

from __future__ import annotations

import logging
from typing import Any, NamedTuple, Protocol, runtime_checkable

import numpy as np
from array_api_compat import is_torch_array

logger = logging.getLogger(__name__)

# torch is imported lazily so the NumPy path does not hard-depend on it --
# which is the whole claim the talk makes about array-agnostic code.
try:
    import torch

    _HAS_TORCH = True
except ImportError:  # pragma: no cover - depends on the installed extras
    _HAS_TORCH = False


def _select_device() -> str:
    if _HAS_TORCH and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = _select_device()

if DEVICE == "mps":
    logger.info("Using Apple Silicon GPU (MPS); Metal has no FP64, so float32.")
    DTYPE_COMPLEX = np.complex64
    DTYPE_REAL = np.float32
else:
    logger.info("Using CPU (MPS not available).")
    DTYPE_COMPLEX = np.complex128
    DTYPE_REAL = np.float64


class Backend(NamedTuple):
    """A resolved backend: namespace plus the device and dtypes to build on."""

    xp: Any
    device: Any
    real: Any
    complex: Any


@runtime_checkable
class Array(Protocol):
    """A structural type for the array operations we actually use.

    Every member here is part of the Array API specification, so NumPy arrays,
    torch tensors and CuPy arrays all satisfy it.

    Deliberately absent:

    * ``cpu()`` / ``clone()`` -- torch-only. Requiring them would mean a NumPy
      array does not satisfy this Protocol, which is exactly backwards.
    * ``__array_namespace__`` -- it is in the spec, but ``torch.Tensor`` does
      not implement it (that is precisely why array-api-compat exists), so
      requiring it would exclude our GPU backend.
    """

    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def ndim(self) -> int: ...

    @property
    def dtype(self) -> Any: ...

    @property
    def device(self) -> Any: ...

    def __getitem__(self, key: Any, /) -> Array: ...

    # float() is called on 0-d arrays such as norms
    def __float__(self) -> float: ...
    def __complex__(self) -> complex: ...

    def __abs__(self) -> Array: ...
    def __neg__(self) -> Array: ...
    def __add__(self, other: Any, /) -> Array: ...
    def __radd__(self, other: Any, /) -> Array: ...
    def __sub__(self, other: Any, /) -> Array: ...
    def __rsub__(self, other: Any, /) -> Array: ...
    def __mul__(self, other: Any, /) -> Array: ...
    def __rmul__(self, other: Any, /) -> Array: ...
    def __truediv__(self, other: Any, /) -> Array: ...
    def __rtruediv__(self, other: Any, /) -> Array: ...
    def __pow__(self, other: Any, /) -> Array: ...


class PropagatorFunc(Protocol):
    """A callable operator that explicitly has a ``__name__`` attribute."""

    __name__: str

    def __call__(self, psi: Array, V: Array, K: Array, dt: float) -> Array: ...


def to_host(array: Any) -> np.ndarray:
    """Pull an array of any backend back to a CPU NumPy array."""
    if is_torch_array(array):
        # is_torch_array is a TypeIs guard, so the checker now knows this is a
        # torch.Tensor and .cpu()/.numpy() resolve without a cast.
        return array.detach().cpu().numpy()
    return np.asarray(array)


def state_distance(psi: Array, target: Array) -> float:
    """How far apart two states are: the L2 norm of their difference.

    Both states are pulled to the host first, so the backends may differ.
    """
    return float(np.linalg.norm(to_host(psi) - to_host(target)))


def to_device(array: Any, device: Any) -> Any:
    """Move an array to a device, for backends that have the concept."""
    if is_torch_array(array):
        return array.to(device)
    return array


def resolve_backend(backend: str = "numpy", precision: str = "double") -> Backend:
    """Map a backend name onto its namespace, device and dtypes.

    This is the single place where a string turns into a namespace. Everything
    downstream takes ``xp`` and never asks which library it is.
    Parameters
    ----------
    backend : str
        The backend to use for array computations ('numpy', 'torch', or 'cupy').
    precision : str
        'double' for float64/complex128 or 'single' for float32/complex64.
        The torch backend on MPS is always single: Metal has no FP64.
    Returns
    -------
    Backend
        A named tuple containing the namespace, device, and dtypes for the backend.
    """
    if precision not in ("single", "double"):
        raise ValueError(
            f"unknown precision {precision!r}; expected 'single' or 'double'"
        )
    single = precision == "single"

    if backend == "torch":
        if not _HAS_TORCH:
            raise ImportError('backend="torch" requires torch to be installed')
        import array_api_compat.torch as xp

        if DEVICE == "mps" or single:
            # Metal has no FP64, so the GPU path is single precision.
            return Backend(xp, DEVICE, torch.float32, torch.complex64)
        return Backend(xp, DEVICE, torch.float64, torch.complex128)

    if backend == "numpy":
        import array_api_compat.numpy as xp

        if single:
            return Backend(xp, None, xp.float32, xp.complex64)  # type: ignore
        return Backend(xp, None, xp.float64, xp.complex128)  # type: ignore

    if backend == "cupy":
        import array_api_compat.cupy as xp

        if single:
            return Backend(xp, None, xp.float32, xp.complex64)  # type: ignore
        return Backend(xp, None, xp.float64, xp.complex128)  # type: ignore

    raise ValueError(
        f"unknown backend {backend!r}; expected 'numpy', 'torch' or 'cupy'"
    )
