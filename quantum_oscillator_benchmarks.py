import pathlib
import time

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


def run_propagation_experiment(
    psi0: Array,
    V: Array,
    K2: Array,
    num_steps: int,
    L: float,
    milestones: list,
    operator: PropagatorFunc,
    results_dir: pathlib.Path,
) -> Array:
    """
    Runs a propagation experiment for a given operator and saves images at milestones.

    Parameters
    ----------
    psi0 : array-like
        The initial quantum state.
    V : array-like
        The potential energy operator.
    K2 : array-like
        The kinetic energy operator.
    num_steps : int
        The number of time steps to simulate.
    L : float
        The spatial extent of the simulation grid.
    milestones : list of tuples
        A list of (milestone_name, total_dt_jump) tuples indicating when to save images.
    operator : function
        The propagation operator function to use (e.g., split_operator_step).
    results_dir : pathlib.Path
        The directory where results will be saved.
    """

    xp = get_namespace(psi0)
    psi = psi0.clone() if hasattr(psi0, "clone") else xp.copy(psi0)

    start_time = time.time()
    for milestone_name, total_dt_jump in milestones:
        # Calculate how many microscopic steps are needed to reach the milestone
        dt = total_dt_jump / num_steps

        for step in range(num_steps):
            psi = operator(psi, V, K2, dt)

        prob = float(xp.linalg.vector_norm(psi))
        print(f"   -> Probability Norm: {prob:.4f}")

        # Check if probability is NaN
        if np.isnan(prob):
            print(
                f"   -> Probability Norm is NaN! The {operator.__name__} has become unstable at milestone: {milestone_name}."
            )
            break

        final_mag = to_host(xp.abs(psi))
        final_mag = final_mag / np.max(final_mag)
        print(f"Saving image for milestone: {milestone_name} | {operator.__name__}")

        plt.figure(figsize=(6, 5))
        plt.imshow(final_mag, cmap="inferno", extent=(-L, L, -L, L))
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


def run_experiment(N=128, L=10.0, sigma=10.0, num_steps=1, backend="torch"):

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
    psi0 = get_initial_state(N, sigma=sigma, backend=backend)
    V, K2 = get_propagators(N, L, backend=backend)

    xp = get_namespace(psi0)

    # Save the intial state image for reference
    plt.figure(figsize=(6, 5))
    plt.imshow(to_host(xp.abs(psi0)), cmap="inferno", extent=(-L, L, -L, L))
    plt.title("Initial State | t = 0")
    plt.savefig(results_dir / "initial.png", dpi=300)
    plt.close()

    for operator_name, operator_func in operators:
        print(f"\nRunning {operator_name}...")
        run_propagation_experiment(
            psi0, V, K2, num_steps, L, milestones, operator_func, results_dir
        )


if __name__ == "__main__":
    N = 128
    L = 10.0
    backend = "torch"
    num_steps = 100
    run_experiment(N=N, L=L, sigma=10.0, num_steps=num_steps, backend=backend)
