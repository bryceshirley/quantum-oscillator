"""3D harmonic-well quantum animation.

Renders |psi|^2 riding on the potential surface returned by
``toolbox_talk.physics.get_propagators``, with tracer dots and trails painted
onto the well so the descent is visible, plus a modulus projection on the floor.

The two lobes of a cat state are tracked as *independent components*: the
initial state is split at x = 0 and each half is evolved with the same
split-operator. Because the Schrodinger equation is linear this is exact
(psi_L + psi_R == psi to machine precision, checked at the end of the run),
and the component centroids pass straight through each other at the origin
instead of appearing to bounce.

Runs to t = 2*pi (one full period of the omega = 1 oscillator) by default.
"""

from __future__ import annotations

import time
from typing import cast

import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.axis3d import Axis as Axis3D

# Using your exact toolbox imports
from toolbox_talk.operators_compat import split_operator_step
from toolbox_talk.physics import get_initial_state, get_propagators
from toolbox_talk.utils import to_host

MAGMA = plt.cm.magma


def _remove(artist) -> None:
    """Remove a Matplotlib artist, tolerating the ContourSet API change in 3.8."""
    if artist is None:
        return
    try:
        artist.remove()
        return
    except (AttributeError, NotImplementedError, ValueError):
        pass
    for coll in getattr(artist, "collections", []):
        try:
            coll.remove()
        except (AttributeError, ValueError):
            pass


def _as_backend(host_array: np.ndarray, like):
    """Put a host numpy array onto the same backend/device/dtype as ``like``."""
    if type(like).__module__.split(".")[0] == "torch":
        import torch

        return torch.as_tensor(host_array, dtype=like.dtype, device=like.device)
    return np.asarray(host_array, dtype=like.dtype)


def generate_3d_animation_on_bowl(
    N: int = 128,  # Keep at 128 for smooth 3D rendering
    L: float = 10.0,
    num_steps: int = 300,  # integrator steps per quarter period (pi/2)
    target_time: float = 2 * np.pi,  # full period of the omega = 1 oscillator
    state_image: str = "horse",
    backend: str = "torch",
    well_clip: float = 15.0,  # visual ceiling on the potential walls
    wave_height: float = 4.0,  # physical thickness of the probability blob
    flip_well_view: bool = True,  # draw a -0.5*r^2 potential as an upright bowl
    fixed_scale: bool = True,  # scale by the t=0 peak, not per-frame
    track_components: bool = True,  # evolve each lobe separately for the tracers
    seam_tol: float = 1e-3,  # max density on the x=0 seam, relative to peak
    show_tracers: bool = True,
    show_shadow: bool = True,
    shadow_quantity: str = "modulus",  # "modulus" (|psi|) or "intensity" (|psi|^2)
    trail_length: int = 200,  # frames of trail kept on the well surface
    surf_stride: int = 1,  # >1 decimates the surface mesh for speed
    steps_per_frame: int = 3,
    fps: int = 30,
    dpi: int = 150,
    azim_sweep: float = 120.0,  # total camera pan in degrees over the run
    output_filename: str | None = None,
) -> str:
    dt = (np.pi / 2) / num_steps
    num_steps_total = round(target_time / dt)
    if output_filename is None:
        output_filename = f"3d_well_sim_{state_image}.mp4"
    if shadow_quantity not in ("modulus", "intensity"):
        raise ValueError("shadow_quantity must be 'modulus' or 'intensity'")

    print("\n" + "=" * 50)
    print(f" GENERATING 3D ANIMATION: {state_image.upper()} ")
    print("=" * 50)

    # 1. Initialize Physics Engine via Toolbox
    psi = get_initial_state(N, L, state_image=state_image, backend=backend)
    V, K = get_propagators(N, L, backend=backend)

    # 2. Setup 3D Spatial Grid
    x = np.linspace(-L, L, N, endpoint=False)
    dx = 2 * L / N
    X, Y = np.meshgrid(x, x)

    # Extract the potential to the host for the wireframe geometry.
    V_host = np.real(to_host(V))

    # get_propagators returns V = -0.5*(x^2 + y^2), i.e. an inverted paraboloid
    # (and K is negated to match, so the dynamics are just the standard
    # oscillator running backwards in time). Clipping that to [0, 15] as the
    # old script did would flatten the well to a plane, so flip it for display.
    inverted = float(V_host.mean()) < 0.0
    if inverted:
        print(
            " note: V from get_propagators is negative (inverted paraboloid);"
            f" {'flipping it upright for display' if flip_well_view else 'drawing it as-is'}."
        )
    V_plot = -V_host if (inverted and flip_well_view) else V_host

    # Clip the potential so the walls don't shoot to infinity and hide the camera.
    V_visual = np.clip(V_plot, 0.0, well_clip)

    z_floor = -0.15 * well_clip
    z_top = well_clip + wave_height + 1.0

    # 3. Reference scales + lobe decomposition ---------------------------
    psi_host = to_host(psi)
    prob = np.abs(psi_host) ** 2
    prob_ref = float(prob.max()) + 1e-15
    mod_ref = float(np.sqrt(prob_ref))

    # Split the initial state into independent lobes. Only do it if the seam at
    # x = 0 is empty, otherwise we would be slicing a single packet in half.
    components: list = []
    if track_components:
        seam = float(prob[:, N // 2].max()) / prob_ref
        frac_left = float(prob[X < 0].sum()) / (float(prob.sum()) + 1e-30)
        if seam < seam_tol and 0.05 < frac_left < 0.95:
            components = [_as_backend(psi_host * m, psi) for m in (X < 0, X >= 0)]
            print(f" tracking 2 independent lobes (seam density {seam:.1e} of peak)")
        else:
            print(
                f" single-lobe tracking: state not cleanly separable at x=0 "
                f"(seam {seam:.1e}, left mass {frac_left:.2f})"
            )
    n_tracers = len(components) if components else 1

    # 4. Setup Matplotlib 3D Figure (Cinematic Dark Theme)
    fig = plt.figure(figsize=(10, 8))
    fig.patch.set_facecolor("#111111")
    try:
        # computed_zorder=False lets the tracers/trails draw on top of the
        # surface instead of being swallowed by matplotlib's naive 3D sorting.
        ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
    except (AttributeError, TypeError):  # matplotlib < 3.5
        ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#111111")

    # Draw the Potential Bowl wireframe (brighter than a plain grey so the
    # funnel between the floor and the clip plateau actually reads on screen)
    ax.plot_wireframe(
        X,
        Y,
        V_visual,
        color="#8fa6b8",
        alpha=0.45,
        rstride=4,
        cstride=4,
        linewidth=0.6,
        zorder=1,
    )

    # 5. Surface / projection helpers ------------------------------------
    stride = max(1, int(surf_stride))
    Xs, Ys = X[::stride, ::stride], Y[::stride, ::stride]

    def make_surface(prob_now: np.ndarray):
        """Lift the normalised probability onto the well geometry and colour it."""
        ref = prob_ref if fixed_scale else float(prob_now.max()) + 1e-15
        t = np.clip(prob_now / ref, 0.0, 1.0)

        # THE MAGIC TRICK: add the potential height to the probability height,
        # so the wave packet physically rides on the wireframe walls.
        Z_surface = V_visual + t * wave_height

        rgba = MAGMA(t)
        # Fade empty regions out so the wireframe shows through them.
        rgba[..., 3] = np.clip(3.0 * t, 0.0, 1.0) ** 0.6

        return ax.plot_surface(
            Xs,
            Ys,
            Z_surface[::stride, ::stride],
            facecolors=rgba[::stride, ::stride],
            shade=False,
            rstride=1,
            cstride=1,
            linewidth=0,
            antialiased=False,
            zorder=2,
        )

    # x-y projection on the floor. Modulus by default: sqrt pulls the tails up,
    # so the shadow shows the full spatial footprint instead of just the core.
    use_modulus = shadow_quantity == "modulus"
    shadow_ref = mod_ref if use_modulus else prob_ref
    shadow_levels = np.linspace(0.03, 1.0, 10) * shadow_ref

    def shadow_field(prob_now: np.ndarray) -> np.ndarray:
        return np.sqrt(prob_now) if use_modulus else prob_now

    def draw_shadow(prob_now: np.ndarray):
        try:
            return ax.contourf(
                X,
                Y,
                shadow_field(prob_now),
                levels=shadow_levels,
                zdir="z",
                offset=z_floor,
                cmap="magma",
                alpha=0.55,
                zorder=0,
            )
        except (ValueError, TypeError):
            return None

    def _idx(val: float) -> int:
        return int(np.clip(round((val + L) / dx), 0, N - 1))

    def centroid_on_well(prob_now: np.ndarray):
        """Centre of mass of one component, projected onto the well surface."""
        m = float(prob_now.sum()) + 1e-30
        xc = float((X * prob_now).sum() / m)
        yc = float((Y * prob_now).sum() / m)
        return xc, yc, float(V_visual[_idx(yc), _idx(xc)])

    def component_probs():
        if components:
            return [np.abs(to_host(c)) ** 2 for c in components]
        return [np.abs(to_host(psi)) ** 2]

    surface = make_surface(prob)
    shadow = draw_shadow(prob) if show_shadow else None

    # Tracer dots + trails painted onto the well surface
    dot_colors = ("#00e5ff", "#ffd166", "#8cff66", "#ff7ab6")
    tracers, trails, history = [], [], [[] for _ in range(n_tracers)]
    if show_tracers:
        for i in range(n_tracers):
            c = dot_colors[i % len(dot_colors)]
            tracers.append(
                ax.plot(
                    [],
                    [],
                    [],
                    "o",
                    color=c,
                    markersize=7,
                    markeredgecolor="white",
                    markeredgewidth=0.5,
                    zorder=6,
                )[0]
            )
            trails.append(
                ax.plot([], [], [], "-", color=c, linewidth=2.0, alpha=0.9, zorder=5)[0]
            )

    # 6. 3D Aesthetics
    title = ax.set_title(
        "Evolution | t = 0.00", color="white", fontsize=16, fontweight="bold", pad=20
    )
    ax.set_xlim(-L, L)
    ax.set_ylim(-L, L)
    ax.set_zlim(z_floor, z_top)
    try:  # zoom fills more of the frame; kwarg exists on matplotlib >= 3.6
        ax.set_box_aspect((1, 1, 0.65), zoom=1.25)
    except (AttributeError, TypeError):
        try:
            ax.set_box_aspect((1, 1, 0.65))
        except AttributeError:  # matplotlib < 3.3
            pass

    ax.tick_params(colors="white", labelsize=8)
    for axis in cast("tuple[Axis3D, Axis3D, Axis3D]", (ax.xaxis, ax.yaxis, ax.zaxis)):
        axis.pane.fill = False
        axis.pane.set_edgecolor("white")
    ax.set_xlabel("X Position", color="white", labelpad=10)
    ax.set_ylabel("Y Position", color="white", labelpad=10)
    ax.set_zlabel("Energy / Probability", color="white", labelpad=10)

    # 7. Animation Loop
    writer = animation.FFMpegWriter(fps=fps, metadata={"title": "3D Quantum Evolution"})
    n_frames = num_steps_total // steps_per_frame + 1
    print(
        f"Simulating {num_steps_total} steps -> {n_frames} frames "
        f"({n_frames / fps:.1f}s of video), encoding to {output_filename}..."
    )
    start_time = time.time()

    with writer.saving(fig, output_filename, dpi=dpi):
        for step in range(num_steps_total + 1):
            # Step physics forward using your unified Split-Operator.
            # Each lobe is propagated with the identical operator, so the sum
            # of the components stays equal to the full state.
            if step > 0:
                psi = split_operator_step(psi, V, K, dt)
                components = [split_operator_step(c, V, K, dt) for c in components]

            # Render frame
            if step % steps_per_frame == 0:
                prob = np.abs(to_host(psi)) ** 2

                _remove(surface)
                surface = make_surface(prob)

                if show_shadow:
                    _remove(shadow)
                    shadow = draw_shadow(prob)

                if show_tracers:
                    for i, p_i in enumerate(component_probs()):
                        xc, yc, zc = centroid_on_well(p_i)
                        history[i].append((xc, yc, zc))
                        if trail_length > 0:
                            history[i] = history[i][-trail_length:]
                        hx, hy, hz = (np.asarray(a) for a in zip(*history[i]))
                        trails[i].set_data_3d(hx, hy, hz + 0.3)
                        tracers[i].set_data_3d(
                            np.array([xc]), np.array([yc]), np.array([zc + 0.25])
                        )

                current_t = step * dt
                title.set_text(
                    f"Evolution | t = {current_t:.2f}  ({current_t / np.pi:.2f}\u03c0)"
                )

                # Slowly pan the camera around the bowl
                ax.view_init(elev=25, azim=-45 + (step / num_steps_total) * azim_sweep)

                writer.grab_frame()

            if step % 100 == 0 and step > 0:
                print(
                    f"   -> step {step}/{num_steps_total} (t={step * dt:.2f}, "
                    f"{time.time() - start_time:.1f}s elapsed)"
                )

    # Linearity check: the components must still add up to the full state.
    if components:
        recon = sum(to_host(c) for c in components)
        resid = float(np.max(np.abs(recon - to_host(psi))))
        print(
            f" component/full-state max residual: {resid:.2e} "
            f"(should be at machine precision)"
        )

    elapsed_time = time.time() - start_time
    print(f"\n >>> Video rendering complete in {elapsed_time:.2f} seconds!")
    print(f" >>> Check your directory for '{output_filename}'")
    plt.close(fig)
    return output_filename


if __name__ == "__main__":
    # You can plug in "cat_state", "double_slit", "vortex", or "orbit" here!
    generate_3d_animation_on_bowl(
        N=128,
        L=10.0,
        num_steps=300,
        target_time=2 * np.pi,
        state_image="double_slit",
        backend="torch",
    )
