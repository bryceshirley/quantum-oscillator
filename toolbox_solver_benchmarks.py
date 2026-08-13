import pathlib
import time

import numpy as np
from array_api_compat import get_namespace

from toolbox_talk.data import get_initial_state
from toolbox_talk.operators import (
    forward_euler_step,
    lie_trotter_step,
)
from toolbox_talk.operators_extra import (
    arnoldi_step,
    crank_nicolson_step,
)
from toolbox_talk.physics import get_propagators
from toolbox_talk.plotting import plot_state
from toolbox_talk.utils import Array, PropagatorFunc


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
    psi = xp.asarray(psi0, copy=True)

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
        safe_name = (
            milestone_name.replace(" ", "_").replace("(", "").replace(")", "").lower()
        )
        save_path = results_dir / f"{operator.__name__.lower()}_{safe_name}.png"
        plot_state(
            psi,
            title=f"{operator.__name__} | {milestone_name}",
            L=L,
            save_path=save_path,
            show=False,
            close=True,
        )

    end_time = time.time() - start_time
    print(f" >>> {operator.__name__} finished in {end_time:.2f} seconds.")
    return psi


def run_experiment(
    N=128, L=10.0, state_image="horse", blur=0.35, num_steps=1, backend="torch"
):

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
        ("Lie-Trotter", lie_trotter_step),
        ("Crank-Nicolson", crank_nicolson_step),
        ("Arnoldi Krylov", arnoldi_step),
    ]

    psi0 = get_initial_state(
        N,
        L,
        state_image,
        blur,
        backend=backend,
    )
    V, K = get_propagators(N, L, backend=backend)

    # Save the initial state image for reference using Domain Coloring
    plot_state(
        psi0,
        title="Initial State | t = 0",
        L=L,
        save_path=results_dir / "initial.png",
        show=False,
        close=True,
    )

    for operator_name, operator_func in operators:
        print(f"\nRunning {operator_name}...")
        run_propagation_experiment(
            psi0, V, K, num_steps, L, milestones, operator_func, results_dir
        )


if __name__ == "__main__":
    N = 256
    L = 10.0  # math.sqrt(math.pi * N / 2)  # Spatial extent of the simulation grid
    state_image = "camera"
    backend = "torch"
    num_steps = 100
    blur = 3.0  # Gaussian blur factor for the initial state image
    run_experiment(
        N=N,
        L=L,
        state_image=state_image,
        blur=blur,
        num_steps=num_steps,
        backend=backend,
    )
