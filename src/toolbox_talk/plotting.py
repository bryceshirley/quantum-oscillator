import pathlib

import matplotlib.colors as mcolors
import numpy as np
from matplotlib import pyplot as plt

from toolbox_talk.utils import Array, to_host


# ==============================================================================
# VIDEO RENDERER
# ==============================================================================
def complex_to_rgb(z_array: np.ndarray) -> np.ndarray:
    """
    Maps a complex numpy array to an RGB image.
    Phase -> Hue (Color)
    Magnitude -> Value (Brightness)
    Imaginary Component -> Saturation (White in real-space, rainbow in momentum-space)
    """
    # Extract phase and map from [-pi, pi] to [0, 1] for the Hue channel
    phase = np.angle(z_array)
    h = (phase + np.pi) / (2 * np.pi)

    # Extract magnitude and normalize it for the Value (brightness) channel
    mag = np.abs(z_array)
    # Avoid division by zero by adding a tiny epsilon
    v = mag / (np.max(mag) + 1e-12)

    # np.abs(np.sin(phase)) is 0 when purely real (phase = 0 or pi), making it white.
    s = np.sqrt(np.abs(np.sin(phase)))

    # Stack channels and convert HSV to RGB
    hsv = np.dstack((h, s, v))
    rgb = mcolors.hsv_to_rgb(hsv)

    return rgb


def plot_state(
    psi: Array,
    title: str = "Quantum State Magnitude |psi|",
    L: float = 10.0,
    save_path: str | pathlib.Path | None = None,
    show: bool = True,
    close: bool = True,
):
    """
    Plots the magnitude of the quantum state |psi| as a 2D image.
    """
    psi_host = to_host(psi)
    final_rgb = complex_to_rgb(psi_host)

    plt.figure(figsize=(6, 6))
    plt.imshow(
        final_rgb,
        extent=(-L, L, -L, L),
        cmap="viridis",
        aspect="equal",
    )
    plt.colorbar(label="|psi|")
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    if save_path:
        plt.savefig(save_path, dpi=300)
    if show:
        plt.show()
    if close:
        plt.close()
