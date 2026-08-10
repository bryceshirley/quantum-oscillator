import pathlib
import time

import matplotlib.colors as mcolors
import numpy as np
from array_api_compat import get_namespace
from matplotlib import pyplot as plt

from toolbox_talk.operators_compat import (
    arnoldi_step,
    forward_euler_step,
    split_operator_step,
)
from toolbox_talk.operators_noncompat import crank_nicolson_step
from toolbox_talk.physics import get_initial_state, get_propagators
from toolbox_talk.utils import Array, PropagatorFunc, to_host


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


def run_propagation_experiment(
    psi0: Array,
    V: Array,
    K: Array,
    num_steps: int,
    L: float,
    milestones: list,
    operator: PropagatorFunc,
    results_dir: pathlib.Path,
) -> Array:
    """
    Runs a propagation experiment for a given operator and saves images at milestones.
    """

    xp = get_namespace(psi0)
    psi = psi0.clone() if hasattr(psi0, "clone") else xp.copy(psi0)

    start_time = time.time()
    for milestone_name, total_dt_jump in milestones:
        # Calculate how many microscopic steps are needed to reach the milestone
        dt = total_dt_jump / num_steps

        for step in range(num_steps):
            psi = operator(psi, V, K, dt)

        prob = float(xp.linalg.vector_norm(psi))
        print(f"   -> Probability Norm: {prob:.4f}")

        # Check if probability is NaN
        if np.isnan(prob):
            print(
                f"   -> Probability Norm is NaN! The {operator.__name__} has become unstable at milestone: {milestone_name}."
            )
            break

        print(f"Saving image for milestone: {milestone_name} | {operator.__name__}")

        # Convert the complex array to RGB domain coloring
        psi_host = to_host(psi)
        final_rgb = complex_to_rgb(psi_host)

        plt.figure(figsize=(6, 5))
        plt.imshow(final_rgb, extent=(-L, L, -L, L))
        plt.title(f"{milestone_name} | {operator.__name__}")

        safe_name = (
            milestone_name.replace(" ", "_").replace("(", "").replace(")", "").lower()
        )
        plt.savefig(
            results_dir / f"{operator.__name__.lower()}_{safe_name}.png", dpi=300
        )
        plt.close()

    end_time = time.time() - start_time
    print(f" >>> {operator.__name__} finished in {end_time:.2f} seconds.")
    return psi


def run_experiment(N=128, L=10.0, state_image="horse", num_steps=1, backend="torch"):

    # Create results directory if it doesn't exist
    results_dir = pathlib.Path("results")
    results_dir.mkdir(exist_ok=True)

    print("\n" + "=" * 50)
    print(" Quantum Revival Simulation ")
    print("=" * 50)
    milestones = [
        ("Quantum Lens (Fourier)", np.pi / 2),
        ("Inverted Image", np.pi / 2),
        ("Quantum Revival", np.pi),
    ]
    operators = [
        ("Forward Euler", forward_euler_step),
        ("Split-Operator", split_operator_step),
        ("Crank-Nicolson", crank_nicolson_step),
        ("Arnoldi Krylov", arnoldi_step),
    ]
    psi0 = get_initial_state(N, L, state_image=state_image, backend=backend)
    V, K = get_propagators(N, L, backend=backend)

    # Save the initial state image for reference using Domain Coloring
    psi0_host = to_host(psi0)
    initial_rgb = complex_to_rgb(psi0_host)

    plt.figure(figsize=(6, 5))
    plt.imshow(initial_rgb, extent=(-L, L, -L, L))
    plt.title("Initial State | t = 0")
    plt.savefig(results_dir / "initial.png", dpi=300)
    plt.close()

    for operator_name, operator_func in operators:
        print(f"\nRunning {operator_name}...")
        run_propagation_experiment(
            psi0, V, K, num_steps, L, milestones, operator_func, results_dir
        )


if __name__ == "__main__":
    N = 128
    L = 10.0
    state_image = "horse"
    backend = "torch"
    num_steps = 100
    run_experiment(
        N=N, L=L, state_image=state_image, num_steps=num_steps, backend=backend
    )
