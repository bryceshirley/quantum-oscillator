"""Initial-state images sourced from ``skimage.data``.

Every loader returns a float32 ``(N, N)`` array normalised to [0, 1], with the
subject bright on a black background, centred with margin from the periodic
boundary. That is the contract the propagators expect: a real, non-negative
amplitude which ``get_initial_state`` then casts to complex and normalises.
"""

from __future__ import annotations

import numpy as np
from skimage import data
from skimage.transform import resize

from toolbox_talk.utils import DTYPE_REAL

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
