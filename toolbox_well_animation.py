"""3D harmonic-well animation.

WHAT IS PHYSICAL HERE
---------------------
* The bowl is the potential from ``get_propagators``, drawn in textbook signs
  and in energy units. Nothing else is drawn on it.
* The state lies flat on the floor beneath the bowl, as an image of |psi|. It
  is never added to V: a probability density and an energy have different
  units, and summing them makes the packet look as though it gains and loses
  energy as it slides.
* The marker points are the centres of mass of independent pieces of the
  state. Each piece is split off at t = 0 and evolved with the same operator,
  which is exact because the Schrodinger equation is linear. Each marker sits
  at V(x_c, y_c), so its height is the potential energy of that piece, and its
  x-y position is where that piece of the picture has got to.

The readout shows ||psi||. The split-operator method is unitary in exact
arithmetic, so this drifts only by floating-point round-off; the energy is a
different story and is reported in the console at the end.
"""

from __future__ import annotations

import pathlib
import time

import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt

from toolbox_talk.data import get_initial_state
from toolbox_talk.operators import lie_trotter_step
from toolbox_talk.physics import get_propagators
from toolbox_talk.utils import to_host

MAGMA = plt.cm.magma
DOT_COLORS = ("#00e5ff", "#ffd166", "#8cff66", "#ff7ab6")


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


def _partition_by_blob(prob: np.ndarray, n_track: int, seam_tol: float):
    """Split the grid into one region per blob of the initial state.

    Blobs are found by labelling the connected regions where the density is
    above a small threshold, then every remaining pixel is assigned to its
    nearest blob. That second step matters: the pieces must tile the whole
    grid so that summing them reproduces psi exactly, which is what makes
    evolving them independently legitimate.

    Returns (masks, seam, n_found). ``masks`` is None if the state does not
    actually contain n_track separable blobs.
    """
    if n_track == 1:
        return [np.ones_like(prob, dtype=bool)], 0.0, 1

    from scipy import ndimage

    seeds, n_found = ndimage.label(prob > 1e-3 * prob.max())
    if n_found < n_track:
        return None, 0.0, n_found

    if n_found > n_track:
        # keep the heaviest blobs; the rest get absorbed by the nearest one
        mass = ndimage.sum(prob, seeds, index=range(1, n_found + 1))
        keep = np.argsort(mass)[::-1][:n_track] + 1
        seeds, _ = ndimage.label(np.isin(seeds, keep))

    # nearest-seed (Voronoi) assignment of every pixel
    _, idx = ndimage.distance_transform_edt(seeds == 0, return_indices=True)
    region = seeds[tuple(idx)]

    masks = [region == k for k in range(1, n_track + 1)]
    # order left-to-right, then bottom-to-top, so marker colours are stable
    order = sorted(
        range(n_track),
        key=lambda i: ndimage.center_of_mass(prob, region, i + 1)[::-1],
    )
    masks = [masks[i] for i in order]

    # density on the cuts between regions: high means we sliced through a blob
    edge = np.zeros_like(prob, dtype=bool)
    edge[:, :-1] |= region[:, :-1] != region[:, 1:]
    edge[:-1, :] |= region[:-1, :] != region[1:, :]
    seam = float(prob[edge].max()) / float(prob.max()) if edge.any() else 0.0
    return masks, seam, n_found


def generate_well_animation(
    N: int = 128,
    L: float = 10.0,
    num_steps: int = 300,  # integrator steps per quarter period (pi/2)
    target_time: float = 2 * np.pi,
    state_image: str = "triple_slit",
    blur: float = 0.35,
    backend: str = "torch",
    well_clip: float = 20.0,  # visual ceiling on the potential walls
    n_track: int = 3,  # marker points: one per blob of the initial state
    seam_tol: float = 1e-3,  # max density on a split line, relative to peak
    show_trails: bool = True,  # the path each marker traces on the bowl
    trail_length: int = 0,  # frames of path kept; 0 = the whole run
    show_energy_plane: bool = False,  # optional <E> disc; off by default
    surf_stride: int = 1,
    steps_per_frame: int = 3,
    fps: int = 30,
    dpi: int = 150,
    azim_sweep: float = 120.0,
    output_filename: str | None = None,
) -> str:
    dt = (np.pi / 2) / num_steps
    num_steps_total = round(target_time / dt)

    save_dir = pathlib.Path("animation_output")
    save_dir.mkdir(parents=True, exist_ok=True)
    if output_filename is None:
        output_filename = str(save_dir / f"3d_well_sim_{state_image}.mp4")
    if n_track < 1:
        raise ValueError("n_track must be at least 1")

    print("\n" + "=" * 60)
    print(f" GENERATING 3D WELL ANIMATION: {state_image.upper()} ")
    print("=" * 60)

    # 1. Physics engine ----------------------------------------------------

    psi = get_initial_state(N, L, state_image, blur, backend=backend)
    V, K = get_propagators(N, L, backend=backend)

    x = np.linspace(-L, L, N, endpoint=False)
    dx = 2 * L / N
    X, Y = np.meshgrid(x, x, indexing="xy")

    V_host = np.real(to_host(V))
    K_host = np.real(to_host(K))

    # get_propagators returns V = -0.5 r^2 and K = -0.5 k^2: the whole
    # Hamiltonian is negated, so the dynamics are the standard oscillator run
    # backwards. For display we use textbook signs, so the well is a bowl.
    inverted = float(V_host.mean()) < 0.0
    V_std = -V_host if inverted else V_host
    K_std = -K_host if inverted else K_host
    if inverted:
        print(" note: V from get_propagators is negative; displaying textbook signs.")

    V_visual = np.clip(V_std, 0.0, well_clip)

    def observables(psi_dev):
        """(<V>, <T>, <E>, ||psi||) in display sign convention."""
        p = to_host(psi_dev)
        prob = np.abs(p) ** 2
        total = prob.sum()
        v_exp = float((V_std * prob).sum() / total)
        pk = np.abs(np.fft.fft2(p)) ** 2
        t_exp = float((K_std * pk).sum() / pk.sum())
        return v_exp, t_exp, v_exp + t_exp, float(np.sqrt(total))

    V0, T0, E0, norm0 = observables(psi)
    print(f" initial:  <V>={V0:.4f}  <T>={T0:.4f}  <E>={E0:.4f}  ||psi||={norm0:.8f}")

    if show_energy_plane and E0 >= well_clip:
        well_clip = E0 * 1.3
        V_visual = np.clip(V_std, 0.0, well_clip)

    # The state sits in its own band below the well minimum. It rises from
    # z_floor up to at most z = 0, so it never touches or overlaps the bowl and
    # is never added to V -- the two live in separate parts of the axis.
    blob_band = 0.20 * well_clip
    z_floor = -blob_band
    z_top = well_clip + 0.5

    # 2. One fixed brightness scale, found by a physics-only pre-pass -------
    print(" pre-pass: finding the global peak so one fixed scale fits the run...")
    scan = psi
    peak = 0.0
    for step in range(num_steps_total + 1):
        if step % steps_per_frame == 0:
            peak = max(peak, float(np.abs(to_host(scan)).max()))
        scan = lie_trotter_step(scan, V, K, dt)
    del scan
    peak_ref = peak + 1e-30

    psi_host = to_host(psi)
    prob = np.abs(psi_host) ** 2

    # 3. Split the state into the pieces whose centres we will follow -------
    def _as_backend(host_array):
        if type(psi).__module__.split(".")[0] == "torch":
            import torch

            return torch.as_tensor(host_array, dtype=psi.dtype, device=psi.device)
        return np.asarray(host_array, dtype=psi.dtype)

    masks, worst_seam, n_found = _partition_by_blob(prob, n_track, seam_tol)

    if masks is None:
        print(
            f" found only {n_found} separable blob(s) but n_track={n_track}; "
            "following the whole state with one marker."
        )
        masks = [np.ones_like(X, dtype=bool)]
    else:
        masses = [float(prob[m].sum()) / float(prob.sum()) for m in masks]
        # Slicing through a blob would give each piece a hard edge, which a
        # spectral method turns into ringing, and the centre of mass of a
        # ringing piece means little.
        if worst_seam >= seam_tol or min(masses) < 0.02:
            print(
                f" not cleanly separable (seam {worst_seam:.1e}, smallest piece "
                f"{min(masses):.3f}); following the whole state with one marker."
            )
            masks = [np.ones_like(X, dtype=bool)]
        else:
            print(
                f" found {n_found} blob(s); tracking {len(masks)} "
                f"(seam {worst_seam:.1e}, masses "
                f"{', '.join(f'{m:.3f}' for m in masses)})"
            )
    components = [_as_backend(psi_host * m) for m in masks]
    n_markers = len(components)
    print(f" following {n_markers} point(s) on the state")

    # 4. Figure -------------------------------------------------------------
    fig = plt.figure(figsize=(10, 8))
    fig.patch.set_facecolor("#111111")
    try:
        ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
    except (AttributeError, TypeError):  # matplotlib < 3.5
        ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#111111")

    # The bowl. Nothing is drawn on it except the markers.
    ax.plot_wireframe(
        X,
        Y,
        V_visual,
        color="#9db4c6",
        alpha=0.55,
        rstride=4,
        cstride=4,
        linewidth=0.6,
        zorder=1,
    )

    if show_energy_plane:
        plane = np.full_like(V_std, E0)
        plane[V_std > E0] = np.nan
        ax.plot_surface(
            X,
            Y,
            plane,
            color="#FCFDBF",
            alpha=0.18,
            shade=False,
            rstride=2,
            cstride=2,
            linewidth=0,
            antialiased=False,
            zorder=2,
        )

    # 5. The state, lying flat on the floor ---------------------------------
    stride = max(1, int(surf_stride))
    Xs, Ys = X[::stride, ::stride], Y[::stride, ::stride]

    def make_blobs(prob_now):
        """The state as blobs standing on the floor: |psi|, one fixed scale.

        Height here is |psi| on its own scale, not an energy, which is why the
        blobs live in a band below the well and are never added to V.

        The gamma is a fixed display transfer curve applied identically to
        every frame: it lifts the dim end so the spread-out state stays
        visible, but it does not renormalise per frame, so both height and
        brightness stay comparable across the whole run.
        """
        t = np.clip(np.sqrt(prob_now) / peak_ref, 0.0, 1.0)
        Z = z_floor + t * blob_band
        rgba = MAGMA(t**0.6)
        rgba[..., 3] = np.clip(3.5 * t, 0.0, 1.0) ** 0.6
        return ax.plot_surface(
            Xs,
            Ys,
            Z[::stride, ::stride],
            facecolors=rgba[::stride, ::stride],
            shade=False,
            rstride=1,
            cstride=1,
            linewidth=0,
            antialiased=False,
            zorder=0,
        )

    def _idx(val):
        return int(np.clip(round((val + L) / dx), 0, N - 1))

    def centroid(prob_now):
        """Centre of mass of one piece, lifted onto the bowl."""
        m = float(prob_now.sum()) + 1e-30
        xc = float((X * prob_now).sum() / m)
        yc = float((Y * prob_now).sum() / m)
        return xc, yc, float(V_visual[_idx(yc), _idx(xc)])

    def component_probs():
        return [np.abs(to_host(c)) ** 2 for c in components]

    blobs = make_blobs(prob)

    markers, trails, history = [], [], [[] for _ in range(n_markers)]
    for i in range(n_markers):
        c = DOT_COLORS[i % len(DOT_COLORS)]
        markers.append(
            ax.plot(
                [],
                [],
                [],
                "o",
                color=c,
                markersize=9,
                markeredgecolor="white",
                markeredgewidth=0.8,
                zorder=8,
                label=f"CoM {i + 1}",  # <-- ADDED LABEL HERE
            )[0]
        )
        if show_trails:
            trails.append(
                ax.plot([], [], [], color=c, linewidth=2.0, alpha=0.85, zorder=7)[0]
            )

    # 6. Axes ---------------------------------------------------------------
    title = ax.set_title(
        "t = 0.00", color="white", fontsize=15, fontweight="bold", pad=18
    )
    ax.set_xlim(-L, L)
    ax.set_ylim(-L, L)
    ax.set_zlim(z_floor, z_top)
    try:
        ax.set_box_aspect((1, 1, 0.68), zoom=1.2)
    except (AttributeError, TypeError):
        try:
            ax.set_box_aspect((1, 1, 0.68))
        except AttributeError:
            pass

    ax.tick_params(colors="white", labelsize=8)
    ax.set_xlabel("x", color="white", labelpad=10)
    ax.set_ylabel("y", color="white", labelpad=10)
    ax.set_zlabel("Energy", color="white", labelpad=10)

    ax.legend(
        loc="upper right",
        framealpha=0.1,
        edgecolor="white",
        labelcolor="white",
        fontsize=9,
        bbox_to_anchor=(0.95, 0.95),
    )

    # 7. Loop ---------------------------------------------------------------
    writer = animation.FFMpegWriter(fps=fps, metadata={"title": "Quantum well"})
    n_frames = num_steps_total // steps_per_frame + 1
    print(
        f" simulating {num_steps_total} steps -> {n_frames} frames "
        f"({n_frames / fps:.1f}s of video)"
    )
    start_time = time.time()
    e_drift = 0.0

    with writer.saving(fig, output_filename, dpi=dpi):
        for step in range(num_steps_total + 1):
            if step > 0:
                psi = lie_trotter_step(psi, V, K, dt)
                components = [lie_trotter_step(c, V, K, dt) for c in components]

            if step % steps_per_frame == 0:
                prob = np.abs(to_host(psi)) ** 2
                _, _, e_exp, nrm = observables(psi)
                e_drift = max(e_drift, abs(e_exp - E0))

                _remove(blobs)
                blobs = make_blobs(prob)

                for i, p_i in enumerate(component_probs()):
                    xc, yc, zc = centroid(p_i)
                    markers[i].set_data_3d(
                        np.array([xc]), np.array([yc]), np.array([zc])
                    )
                    if show_trails:
                        history[i].append((xc, yc, zc))
                        if trail_length > 0:
                            history[i] = history[i][-trail_length:]
                        hx, hy, hz = (np.asarray(a) for a in zip(*history[i]))
                        # nudged off the surface so the path is not z-fighting
                        trails[i].set_data_3d(hx, hy, hz + 0.06 * blob_band)

                t_now = step * dt
                title.set_text(
                    f"t = {t_now:.2f}   ({t_now / np.pi:.2f}\u03c0), ||psi|| = {nrm:.6f}"
                )

                ax.view_init(elev=24, azim=-45 + (step / num_steps_total) * azim_sweep)
                writer.grab_frame()

            if step % 200 == 0 and step > 0:
                print(
                    f"   -> step {step}/{num_steps_total} (t={step * dt:.2f}, "
                    f"{time.time() - start_time:.1f}s)"
                )

    _, _, e_end, n_end = observables(psi)
    print(f"\n ||psi||: {norm0:.10f} -> {n_end:.10f}  (unitary up to round-off)")
    print(
        f" <E>: {E0:.6f} -> {e_end:.6f}, max drift {e_drift:.3e} "
        f"({100 * e_drift / abs(E0):.3f}%) -- Trotter splitting error, "
        f"falls with dt"
    )
    recon = sum(to_host(c) for c in components)
    print(f" piece/full-state residual: {np.max(np.abs(recon - to_host(psi))):.2e}")
    print(f"\n >>> rendered in {time.time() - start_time:.1f}s to '{output_filename}'")
    plt.close(fig)
    return output_filename


if __name__ == "__main__":
    generate_well_animation(
        N=128,
        L=10.0,
        num_steps=300,
        target_time=2 * np.pi,
        n_track=3,
        state_image="triple_slit",
        backend="torch",
        azim_sweep=0.0,
    )
