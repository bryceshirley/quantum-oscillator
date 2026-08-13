import numpy as np
import pytest

# Adjust this import to match your actual module structure
from toolbox_talk.utils import (
    _HAS_TORCH,
    Backend,
    from_numpy,
    resolve_backend,
    to_device,
    to_host,
    to_numpy,
)

# --------------------------------------------------------------------------
# Fixtures & Skips
# --------------------------------------------------------------------------

requires_torch = pytest.mark.skipif(not _HAS_TORCH, reason="PyTorch is not installed")


@pytest.fixture
def numpy_array():
    return np.array([[1.0, 2.0], [3.0, 4.0]])


# --------------------------------------------------------------------------
# Tests for Backend Resolution
# --------------------------------------------------------------------------


def test_resolve_backend_numpy():
    backend = resolve_backend("numpy")
    assert isinstance(backend, Backend)
    assert backend.device is None
    assert backend.real == np.float64
    assert backend.complex == np.complex128


@requires_torch
def test_resolve_backend_torch():
    import torch

    backend = resolve_backend("torch")
    assert isinstance(backend, Backend)
    assert backend.device in ("cpu", "mps")

    if backend.device == "mps":
        assert backend.real == torch.float32
        assert backend.complex == torch.complex64
    else:
        assert backend.real == torch.float64
        assert backend.complex == torch.complex128


def test_resolve_backend_invalid():
    with pytest.raises(ValueError, match="unknown backend"):
        resolve_backend("tensorflow")


# --------------------------------------------------------------------------
# Tests for Array Conversion (Host/Numpy)
# --------------------------------------------------------------------------


def test_to_host_numpy(numpy_array):
    result = to_host(numpy_array)
    assert isinstance(result, np.ndarray)
    assert np.array_equal(result, numpy_array)


def test_to_numpy_numpy(numpy_array):
    result = to_numpy(numpy_array)
    assert isinstance(result, np.ndarray)
    assert np.array_equal(result, numpy_array)


@requires_torch
def test_to_host_torch(numpy_array):
    import torch

    tensor = torch.tensor(numpy_array, device="cpu")
    result = to_host(tensor)
    assert isinstance(result, np.ndarray)
    assert np.array_equal(result, numpy_array)


@requires_torch
def test_to_numpy_torch(numpy_array):
    import torch

    tensor = torch.tensor(numpy_array, device="cpu", requires_grad=True)
    result = to_numpy(tensor)
    assert isinstance(result, np.ndarray)
    assert np.array_equal(result, numpy_array)


# --------------------------------------------------------------------------
# Tests for Device Placement
# --------------------------------------------------------------------------


def test_to_device_numpy(numpy_array):
    # NumPy doesn't have a device concept, so it should just return the array
    result = to_device(numpy_array, "cpu")
    assert result is numpy_array


@requires_torch
def test_to_device_torch():
    import torch

    tensor = torch.tensor([1.0, 2.0])
    # CPU to CPU is a valid no-op check
    result = to_device(tensor, "cpu")
    assert isinstance(result, torch.Tensor)
    assert str(result.device) == "cpu"


# --------------------------------------------------------------------------
# Tests for from_numpy
# --------------------------------------------------------------------------


def test_from_numpy_to_numpy_backend(numpy_array):
    import array_api_compat.numpy as xp

    result = from_numpy(numpy_array, xp, dtype=np.float32, device=None)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32


@requires_torch
def test_from_numpy_to_torch_backend(numpy_array):
    import array_api_compat.torch as xp
    import torch

    result = from_numpy(numpy_array, xp, dtype=torch.float32, device="cpu")
    assert isinstance(result, torch.Tensor)
    assert result.dtype == torch.float32
    assert str(result.device) == "cpu"
