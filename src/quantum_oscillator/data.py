"""Initial-state images sourced from ``skimage.data``.

Every loader returns a float32 ``(N, N)`` array normalised to [0, 1], with the
subject bright on a black background, centred with margin from the periodic
boundary. That is the contract the propagators expect: a real, non-negative
amplitude which ``get_initial_state`` then casts to complex and normalises.
"""

from __future__ import annotations

import math

import numpy as np
from skimage import data
from skimage.transform import resize

from quantum_oscillator.utils import DTYPE_REAL, Array, resolve_backend

# Sample images from skimage.data that make legible initial states.
#   invert : subject is dark on a light background, so flip it
#   note   : what it is / why it is interesting to propagate
SAMPLE_IMAGES: dict[str, dict] = {
    # --- binary silhouettes: highest contrast, cleanest revivals -------------
    "horse": {"invert": True, "note": "the classic silhouette (328x400 bool)"},
    "binary_blobs": {"invert": False, "note": "random blobs, different every call"},
    "checkerboard": {"invert": False, "note": "pure high-frequency content"},
    # --- smooth greyscale: gentle spectra, least aliasing --------------------
    "shepp_logan_phantom": {"invert": False, "note": "medical phantom, already [0,1]"},
    "cell": {"invert": False, "note": "microscopy, soft edges"},
    "moon": {"invert": False, "note": "low contrast, broad spectrum"},
    "clock": {"invert": False, "note": "one bright object on dark"},
    # --- photographic greyscale --------------------------------------------
    "camera": {"invert": True, "note": "the cameraman"},
    "coins": {"invert": False, "note": "several compact bright objects"},
    "page": {"invert": True, "note": "printed text on paper"},
    "text": {"invert": True, "note": "handwriting, very wide (172x448)"},
    "microaneurysms": {"invert": True, "note": "small, faint features"},
    # --- textures: broadband, revivals look busy ----------------------------
    "brick": {"invert": False, "note": "regular texture"},
    "grass": {"invert": False, "note": "fine broadband texture"},
    "gravel": {"invert": True, "note": "fine broadband texture"},
    # --- colour, converted to luminance -------------------------------------
    "astronaut": {"invert": False, "note": "portrait (512x512x3)"},
    "chelsea": {"invert": False, "note": "the cat (300x451x3)"},
    "coffee": {"invert": False, "note": "cup of coffee (400x600x3)"},
    "rocket": {"invert": False, "note": "rocket on a launchpad, wide"},
    "colorwheel": {"invert": False, "note": "smooth radial gradient"},
    "logo": {"invert": False, "note": "scikit-image logo, has alpha"},
    "immunohistochemistry": {"invert": False, "note": "stained tissue"},
    "hubble_deep_field": {"invert": False, "note": "sparse points on black"},
    "retina": {"invert": False, "note": "fundus photograph, large"},
}
SAMPLE_SHIFTED_IMAGES = {
    name + "_shifted": info for name, info in SAMPLE_IMAGES.items()
}
ANALYTIC_STATES = (
    "cat_state",
    "cosine",
    "double_slit",
    "triple_slit",
    "single_shifted_slit",
    "orbit",
    "vortex",
    "lattice",
)


def get_initial_state(
    N: int,
    L: float,
    state_image: str = "horse",
    blur: float = 5.0,
    backend: str = "numpy",
    precision: str = "double",
) -> Array:
    """
    Builds a normalised complex initial state on the N x N grid.
    """
    backend_info = resolve_backend(backend, precision)

    xp = backend_info.xp
    device = backend_info.device
    dtype = backend_info.complex
    x = xp.linspace(-L, L, N, endpoint=False, device=device)
    X, Y = xp.meshgrid(x, x, indexing="xy")

    # Resolve target dtype so we don't accidentally fall back to defaults that conflict with the device
    target_dtype = dtype if dtype is not None else xp.complex128

    if state_image == "cat_state":
        width = L / 8.0
        offset = L / 2.0
        packet_left = xp.exp(-((X + offset) ** 2 + Y**2) / (2 * width**2))
        packet_right = xp.exp(-((X - offset) ** 2 + Y**2) / (2 * width**2))
        amplitude = packet_left + packet_right

    elif state_image == "cosine":
        # A broad Gaussian envelope to prevent FFT boundary artifacts
        envelope = xp.exp(-(X**2 + Y**2) / (2 * (L / 3) ** 2))
        frequency = 3.0
        amplitude = xp.cos(frequency * X) * envelope

    elif state_image == "double_slit":
        slit_distance = L / 1.5
        slit_width = L / 30.0
        slit_1 = xp.exp(-(Y**2 + (X - slit_distance / 2) ** 2) / (2 * slit_width**2))
        slit_2 = xp.exp(-(Y**2 + (X + slit_distance / 2) ** 2) / (2 * slit_width**2))
        amplitude = slit_1 + slit_2

    elif state_image == "triple_slit":
        # Three slits at the vertices of an equilateral triangle, point-up.
        # slit_distance is the side length, i.e. the centre-to-centre spacing
        # of any two slits, matching double_slit's separation.
        slit_distance = L / 1.5
        slit_width = L / 30.0
        radius = slit_distance / math.sqrt(3.0)  # circumradius of the triangle
        amplitude = xp.zeros((N, N), dtype=target_dtype, device=device)
        for angle_deg in (90.0, 210.0, 330.0):
            theta = math.radians(angle_deg)
            x0 = radius * math.cos(theta)
            y0 = radius * math.sin(theta)
            spot = xp.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2 * slit_width**2))
            amplitude = amplitude + xp.astype(spot, target_dtype)

    elif state_image == "single_shifted_slit":
        slit_distance = L / 1.5
        slit_width = L / 30.0
        amplitude = xp.exp(-(Y**2 + (X - slit_distance / 2) ** 2) / (2 * slit_width**2))

    elif state_image == "orbit":
        width = L / 8.0
        x_offset = -L / 3.0
        k_y = 6.0
        envelope = xp.exp(-((X - x_offset) ** 2 + Y**2) / (2 * width**2))
        phase_kick = xp.exp(1j * k_y * Y)
        amplitude = envelope * phase_kick

    elif state_image == "vortex":
        width = L / 4.0
        envelope = xp.exp(-(X**2 + Y**2) / (2 * width**2))
        # The (X + iY) term creates the spinning rainbow phase
        amplitude = (X + 1j * Y) * envelope

    elif state_image == "lattice":
        amplitude = xp.zeros((N, N), dtype=target_dtype, device=device)
        width = L / 20.0
        spacing = L / 2.5
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                x0 = i * spacing
                y0 = j * spacing
                spot = xp.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2 * width**2))
                amplitude = amplitude + xp.astype(spot, target_dtype)

    elif state_image in SAMPLE_IMAGES or state_image in SAMPLE_SHIFTED_IMAGES:
        host_array = sample_image(name=state_image, N=N)

        # Ingest directly into the target Array API namespace
        amplitude = xp.asarray(host_array, device=device)

    else:
        raise ValueError(
            f"unknown state_image {state_image!r}; expected one of "
            f"{', '.join(ANALYTIC_STATES)}, or one of the sample images: "
            f"{', '.join(SAMPLE_IMAGES)}, or their shifted variants "
            "(e.g. 'horse_shifted')."
        )

    # Cast to the final complex target type
    psi = xp.asarray(amplitude, dtype=target_dtype, device=device)

    # The vortex carries phase winding, applied after the real envelope.
    if state_image == "vortex":
        psi = psi * xp.exp(1j * xp.atan2(Y, X))

    if blur > 0:
        psi = _gaussian_blur(psi, xp, blur)

    norm = xp.linalg.vector_norm(psi)
    return psi / norm


def _gaussian_blur(psi: Array, xp, sigma_pixels: float) -> Array:
    """
    Applies a spatial Gaussian blur using the Convolution Theorem (via FFT).

    Parameters
    ----------
    psi : Array
        The quantum state to be blurred.
    xp : module
        The Array API namespace.
    sigma_pixels : float
        The standard deviation of the Gaussian blur in spatial pixels.
    """
    n = psi.shape[-1]

    # Frequencies in cycles per pixel
    f = xp.fft.fftfreq(n, d=1.0, device=getattr(psi, "device", None))
    Fx, Fy = xp.meshgrid(f, f, indexing="xy")

    # The Fourier transform of a spatial Gaussian with std dev `sigma`
    # is a frequency-domain Gaussian: F{ exp(-x^2 / 2*sigma^2) } = exp(-2 * pi^2 * sigma^2 * f^2)
    window = xp.exp(-2 * math.pi**2 * sigma_pixels**2 * (Fx**2 + Fy**2))

    # Apply the blur in momentum space and return to real space
    return xp.fft.ifft2(xp.fft.fft2(psi) * xp.astype(window, psi.dtype))


def _to_grey(img: np.ndarray) -> np.ndarray:
    """Collapse an image of any skimage.data flavour to a 2D float array."""
    img = np.asarray(img)
    if img.ndim == 3:
        if img.shape[-1] == 4:  # RGBA: composite onto black using alpha
            rgb = img[..., :3].astype(DTYPE_REAL)
            alpha = img[..., 3:4].astype(DTYPE_REAL)
            alpha = alpha / 255.0 if alpha.max() > 1.0 else alpha
            img = rgb * alpha
        else:
            img = img[..., :3].astype(DTYPE_REAL)
        # Rec. 601 luminance
        img = img @ np.array([0.299, 0.587, 0.114])
    return img.astype(DTYPE_REAL)


def _normalise(img: np.ndarray) -> np.ndarray:
    lo, hi = float(img.min()), float(img.max())
    span = hi - lo
    return (img - lo) / span if span > 0 else np.zeros_like(img)


def sample_image(
    name: str = "horse",
    N: int = 128,
    fill: float = 0.5,
    invert: bool | None = None,
) -> np.ndarray:
    """
    Build an initial-state amplitude from a ``skimage.data`` sample.

    Parameters
    ----------
    name : str
        A key of ``SAMPLE_IMAGES`` (e.g. "horse", "camera", "astronaut").
    N : int
        The number of grid points in each dimension.
    fill : float
        Fraction of the grid the subject spans along its LONGEST axis. Keeping
        this below 1 leaves the margin the periodic boundary needs. Note this
        differs from the original horse loader, which scaled on height alone;
        ``fill=0.61`` reproduces its exact horse size at any N.
    invert : bool, optional
        Override the registry's guess about whether the subject is dark on a
        light background.

    Returns
    -------
    initial_state : np.ndarray
        The initial amplitude as an ``(N, N)`` {dtype} array in [0, 1].
    """
    # Check if name has "_shifted" suffix to determine if the image should be shifted
    if name.endswith("_shifted"):
        shifted = True
        name = name[:-8]  # Remove the "_shifted" suffix for lookup
    else:
        shifted = False

    if name not in SAMPLE_IMAGES:
        raise ValueError(
            f"unknown sample image {name!r}; available: "
            + ", ".join(sorted(SAMPLE_IMAGES))
        )
    if not 0 < fill <= 1:
        raise ValueError(f"fill must be in (0, 1], got {fill}")

    raw = _normalise(_to_grey(getattr(data, name)()))

    if invert is None:
        invert = SAMPLE_IMAGES[name]["invert"]
    if invert:
        raw = 1.0 - raw

    # Scale on the LONGEST axis, so wide images (text, rocket, coffee) cannot
    # overflow the grid. Scaling on height alone puts "text" 167px wide into a
    # 128px box and raises a broadcast error.
    target = max(1, round(fill * N))
    scale = target / max(raw.shape)
    new_shape = (
        max(1, round(raw.shape[0] * scale)),
        max(1, round(raw.shape[1] * scale)),
    )

    # resize() anti-aliases when downsampling; ndimage.zoom does not, and its
    # cubic spline also undershoots into negatives around hard edges.
    small = resize(
        raw,
        new_shape,
        order=1,
        anti_aliasing=True,
        preserve_range=True,
        mode="constant",
        cval=0.0,
    )

    canvas = np.zeros((N, N), dtype=DTYPE_REAL)
    h, w = small.shape
    start_y = (N - h) // 2
    start_x = (N - w) // 2
    if shifted:
        start_x += N // 6
    # Clamp so an aggressive fill or shift crops rather than raising.
    start_y = int(np.clip(start_y, 0, max(0, N - h)))
    start_x = int(np.clip(start_x, 0, max(0, N - w)))
    canvas[start_y : start_y + h, start_x : start_x + w] = small[
        : N - start_y, : N - start_x
    ]

    # Keep the background at exactly zero: any interpolation undershoot would
    # otherwise show up as faint negative amplitude.
    canvas = np.clip(canvas, 0.0, None)

    return canvas.astype(DTYPE_REAL)
