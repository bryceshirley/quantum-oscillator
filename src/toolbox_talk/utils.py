from typing import Any, Protocol

import numpy as np
import torch

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
if DEVICE == "mps":
    print("Using Apple Silicon GPU (MPS) for PyTorch, limited to float32 precision.")
    DTYPE_COMPLEX = np.complex64
    DTYPE_REAL = np.float32
else:
    print("Using CPU for PyTorch (MPS not available).")
    DTYPE_COMPLEX = np.complex128
    DTYPE_REAL = np.float64


class Array(Protocol):
    """A structural type defining the array operations we actually use."""

    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def dtype(self) -> Any: ...

    @property
    def device(self) -> Any: ...

    def cpu(self) -> "Array": ...
    def clone(self) -> "Array": ...
    def __abs__(self) -> "Array": ...
    def __add__(self, other: Any) -> "Array": ...
    def __sub__(self, other: Any) -> "Array": ...
    def __mul__(self, other: Any) -> "Array": ...
    def __rmul__(self, other: Any) -> "Array": ...
    def __truediv__(self, other: Any) -> "Array": ...


def to_host(array) -> np.ndarray:
    """Helper to safely pull PyTorch MPS arrays to CPU for NumPy/W&B processing."""
    if hasattr(array, "cpu"):
        return array.cpu().numpy()
    return np.asarray(array)


def to_device(array, device):
    """Helper to safely push NumPy arrays to PyTorch MPS device."""
    if hasattr(array, "to"):
        return array.to(device)
    return array


#  Array API / PyTorch tensor to standard CPU NumPy array
def to_numpy(arr, xp):
    if "torch" in xp.__name__:
        return arr.cpu().numpy()
    return np.asarray(arr)


#  Send CPU NumPy array to Array API / PyTorch tensor on target device
def from_numpy(arr_np, xp, dtype, device):
    if "torch" in xp.__name__:
        import torch

        return torch.tensor(arr_np, dtype=dtype, device=device)
    return xp.asarray(arr_np, dtype=dtype, device=device)


class PropagatorFunc(Protocol):
    """Protocol defining a callable operator that explicitly has a __name__ attribute."""

    __name__: str

    def __call__(self, psi: Array, V: Array, K2: Array, dt: float) -> Array: ...
