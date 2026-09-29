"""Quantum-revival animation of an image-based state in a harmonic well.

Evolves the state for one full period (t = 2 pi) with the split-operator
method and renders |psi(x, y)> as a flat domain-colouring GIF (hue = phase,
brightness = magnitude). The harmonic well acts as a quantum lens: the run
passes through the Fourier transform of the initial image at pi/2, its
spatial inversion at pi, the inverse Fourier transform at 3 pi/2, and the
exact revival of the image at 2 pi. The GIF pauses where the image is
recognisable — upright at the loop seam, upside down at t = pi — and is
rendered borderless with no annotations, so it loops as an endless
oscillation.
"""

import pathlib
import time

import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt

from quantum_oscillator.data import get_initial_state
from quantum_oscillator.operators import lie_trotter_step
from quantum_oscillator.physics import get_propagators
from quantum_oscillator.plotting import complex_to_rgb
from quantum_oscillator.utils import to_host

SAVE_DIR = pathlib.Path(__file__).resolve().parent.parent / "animation_output"


def generate_animation(
    N: int = 512,
    L: float = 10.0,
    num_steps: int = 300,
    state_image: str = "horse",
    blur: float = 0.35,
    backend: str = "torch",
    steps_per_frame: int = 4,  # GIFs are uncompressed-ish; don't keep every step
    fps: int = 15,
    dpi: int = 150,
    pause_seconds: float = 1.0,  # hold at the upright/inverted milestones
    cmap: str | None = "twilight_shifted",  # muted cyclic phase palette
):
    """Render the revival GIF for ``state_image`` to animation_output/.

    ``num_steps`` sets the integrator steps per quarter period, so
    dt = (pi/2) / num_steps; one frame is written every ``steps_per_frame``
    steps.
    """
    dt = (np.pi / 2) / num_steps
    target_time = 2 * np.pi  # one full period = quantum revival
    num_steps_total = int(target_time / dt)

    print("\n" + "=" * 50)
    print(" GENERATING QUANTUM REVIVAL GIF ")
    print("=" * 50)

    psi = get_initial_state(N, L, state_image, blur, backend=backend)
    V, K = get_propagators(N, L, backend=backend)

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    output_filename = SAVE_DIR / f"quantum_revival_{state_image}.gif"

    # Borderless full-bleed frame: no axes, ticks or title
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.axis("off")
    im = ax.imshow(complex_to_rgb(to_host(psi), cmap))

    writer = animation.PillowWriter(fps=fps)
    n_frames = num_steps_total // steps_per_frame + 1
    print(
        f"Simulating {num_steps_total} steps -> {n_frames} frames, {output_filename}..."
    )
    start_time = time.time()

    # Hold where the image is recognisable, snapped to sampled frames. The
    # endpoints get half a hold each: on loop the two halves meet, so every
    # recognisable moment pauses equally and the GIF oscillates seamlessly.
    half = steps_per_frame * round(np.pi / dt / steps_per_frame)
    holds = {
        0: round(fps * pause_seconds / 2),
        half: round(fps * pause_seconds),
        num_steps_total - num_steps_total % steps_per_frame: round(
            fps * pause_seconds / 2
        ),
    }

    with writer.saving(fig, output_filename, dpi=dpi):
        for step in range(num_steps_total + 1):
            if step > 0:
                psi = lie_trotter_step(psi, V, K, dt)

            if step % steps_per_frame == 0:
                im.set_data(complex_to_rgb(to_host(psi), cmap))
                for _ in range(1 + holds.get(step, 0)):
                    writer.grab_frame()

            if step % 200 == 0 and step > 0:
                print(f"   -> step {step}/{num_steps_total} (t={step * dt:.2f})")

    elapsed_time = time.time() - start_time
    print(f"\n >>> Rendered in {elapsed_time:.2f}s to '{output_filename}'")
    return str(output_filename)


def _parse_args():
    import argparse

    p = argparse.ArgumentParser(
        description="Quantum-revival GIF of an image-based state in a harmonic well."
    )
    p.add_argument("--N", type=int, default=512, help="grid points per axis")
    p.add_argument("--L", type=float, default=10.0, help="half-width of the box")
    p.add_argument(
        "--num-steps", type=int, default=300, help="integrator steps per quarter period"
    )
    p.add_argument("--state-image", default="horse", help="initial state image")
    p.add_argument(
        "--blur",
        type=float,
        default=5.0,  # soften the binary silhouette: sharp edges alias/ring
        help="Gaussian blur (pixels)",
    )
    p.add_argument("--backend", default="torch", choices=("numpy", "torch"))
    p.add_argument("--steps-per-frame", type=int, default=4)
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument(
        "--cmap",
        default="twilight_shifted",
        help="cyclic colormap for the phase; 'hsv-legacy' for the rainbow look",
    )
    p.add_argument(
        "--pause-seconds",
        type=float,
        default=1.0,
        help="hold at the upright/inverted milestones",
    )
    args = vars(p.parse_args())
    if args["cmap"] == "hsv-legacy":
        args["cmap"] = None
    return args


if __name__ == "__main__":
    generate_animation(**_parse_args())
