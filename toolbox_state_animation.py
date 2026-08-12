import time

import matplotlib.colors as mcolors
import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt

from toolbox_talk.operators_compat import split_operator_step
from toolbox_talk.physics import get_initial_state, get_propagators
from toolbox_talk.utils import resolve_backend, to_host


# ==============================================================================
# VIDEO RENDERER
# ==============================================================================
def complex_to_rgb(z_array):
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


def generate_animation(
    N: int = 512,
    L: float = 10.0,
    num_steps: int = 300,
    state_image: str = "horse",
    blur: float = 0.35,
    backend: str = "torch",
):

    dt = (np.pi / 2) / num_steps  # Time step for evolution
    target_time = 2 * np.pi  # Extended to 2*pi for full Quantum Revival

    print("\n" + "=" * 50)
    print(" GENERATING QUANTUM REVIVAL MP4 ")
    print("=" * 50)

    backend_info = resolve_backend(backend)
    psi = get_initial_state(
        N,
        L,
        backend_info.xp,
        state_image,
        blur,
        device=backend_info.device,
        dtype=backend_info.complex,
    )
    V, K = get_propagators(
        N, L, backend_info.xp, device=backend_info.device, dtype=backend_info.real
    )

    # Calculate frames based on dt to reach 2*pi
    num_steps = int(target_time / dt)
    output_filename = f"quantum_revival_{state_image}.mp4"

    # Define exact frame steps for our milestones
    milestones = {
        0: "Initial State",
        int((np.pi / 2) / dt): "Fourier Transform (Quantum Lens)",
        int(np.pi / dt): "Spatial Inversion",
        int((3 * np.pi / 2) / dt): "Inverse Fourier Transform",
        int((2 * np.pi) / dt): "Quantum Revival",
    }

    fps = 40
    pause_duration = 0.5  # Pause for 1.0 seconds at each milestone
    pause_frames = int(fps * pause_duration)

    # Set up the Matplotlib figure
    fig, ax = plt.subplots(figsize=(6, 5))

    # Render the initial state for the first frame using Domain Coloring
    psi_host = to_host(psi)
    initial_rgb = complex_to_rgb(psi_host)

    # Notice we removed cmap="inferno" because we are providing pure RGB data
    im = ax.imshow(initial_rgb, extent=(-L, L, -L, L))
    title = ax.set_title("Initial State | t = 0.00")

    # Configure the FFMpeg Video Writer
    writer = animation.FFMpegWriter(fps=fps, metadata={"title": "Quantum Revival"})

    print(f"Simulating {num_steps} frames and encoding to {output_filename}...")
    start_time = time.time()

    # Start capturing frames
    with writer.saving(fig, output_filename, dpi=200):
        for step in range(num_steps + 1):
            # Step physics forward (skip step 0 to preserve the starting frame)
            if step > 0:
                psi = split_operator_step(psi, V, K, dt)

            # Extract complex state, convert to RGB, and update plot
            psi_host = to_host(psi)
            current_rgb = complex_to_rgb(psi_host)
            im.set_data(current_rgb)

            current_t = step * dt

            # Check if current step is a major milestone
            if step in milestones:
                stage_name = milestones[step]
                title.set_text(f"{stage_name} | t = {current_t:.2f}")
                print(
                    f"\n   [!] Milestone Reached: {stage_name}. Pausing video for {pause_duration}s..."
                )

                # Write the same frame multiple times to create a pause in the video
                for _ in range(pause_frames):
                    writer.grab_frame()
            else:
                title.set_text(f"Evolution | t = {current_t:.2f}")
                writer.grab_frame()

            # Log progress
            if step % 50 == 0 and step not in milestones:
                print(f"   -> Rendered frame {step}/{num_steps} (t={current_t:.2f})")

    elapsed_time = time.time() - start_time
    print(f"\n >>> Video rendering complete in {elapsed_time:.2f} seconds!")
    print(f" >>> Check your directory for '{output_filename}'")


if __name__ == "__main__":
    N = 512
    L = 10.0
    num_steps = 300
    state_image = "vortex"  # Options: "horse", "shifted_horse", "cosine", "double_slit", "single_shifted_slit"
    blur = 5.0  # Blur factor for the initial state image
    backend = "torch"  # Options: "numpy" or "torch"
    generate_animation(
        N=N,
        L=L,
        num_steps=num_steps,
        state_image=state_image,
        blur=blur,
        backend=backend,
    )
