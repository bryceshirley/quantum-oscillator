import pathlib

import matplotlib.colors as mcolors
import numpy as np
from matplotlib import pyplot as plt

from quantum_oscillator.utils import Array, to_host


# ==============================================================================
# VIDEO RENDERER
# ==============================================================================
def complex_to_rgb(z_array: np.ndarray, cmap: str | None = None) -> np.ndarray:
    """
    Maps a complex numpy array to an RGB image.
    Phase -> colour, magnitude -> brightness.

    With ``cmap`` None (the default), phase drives the full HSV hue wheel and
    the imaginary component drives saturation (purely real values are white).
    Pass the name of a cyclic colormap (e.g. ``"twilight_shifted"``) for a
    muted palette with the same rule: purely real values (phase 0 or pi)
    render white, and the colormap tints only the imaginary component.
    """
    phase = np.angle(z_array)
    h = (phase + np.pi) / (2 * np.pi)

    mag = np.abs(z_array)
    v = mag / (np.max(mag) + 1e-12)

    if cmap is not None:
        base = plt.get_cmap(cmap)(h)[..., :3]
        s = np.abs(np.sin(phase))[..., None]  # 0 when purely real -> white
        return ((1.0 - s) + s * base) * v[..., None]

    # np.abs(np.sin(phase)) is 0 when purely real (phase = 0 or pi), making it white.
    s = np.sqrt(np.abs(np.sin(phase)))
    return mcolors.hsv_to_rgb(np.dstack((h, s, v)))


def tidy(ax) -> None:
    """Strip the top/right spines and add a light grid, for cleaner plots."""
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, alpha=0.25)


def show_states(
    states: list[Array],
    titles: list[str],
    size: float = 3.0,
    gamma: float | list[float] = 1.0,
):
    """
    Plots a row of quantum states as a domain-coloring gallery:
    phase -> hue, magnitude -> brightness.

    Parameters
    ----------
    states : list of Array
        The complex quantum states to display (any backend; pulled to host).
    titles : list of str
        One title per state.
    size : float
        The size of each panel in inches.
    gamma : float or list of float
        Magnitude exponent per state (scalar broadcasts). gamma < 1 brightens
        faint amplitude, e.g. to reveal a Fourier-transformed frame's structure.
    """
    gammas = [gamma] * len(states) if isinstance(gamma, (int, float)) else gamma
    _, axes = plt.subplots(1, len(states), figsize=(size * len(states), size))
    for ax, z, title, g in zip(np.atleast_1d(axes), states, titles, gammas):
        z = to_host(z)
        if g != 1.0:  # rescale |z| -> |z|^g, phase untouched
            mag = np.abs(z)
            # floor the ratio: 0 ** (g - 1) is inf for g < 1, and 0 * inf = NaN
            ratio = np.maximum(mag, 1e-12 * mag.max()) / (mag.max() + 1e-30)
            z = z * ratio ** (g - 1.0)
        ax.imshow(complex_to_rgb(z))
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


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
