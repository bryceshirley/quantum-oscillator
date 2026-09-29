"""3D harmonic-well animation in two panels: position and momentum space.

Left panel: the potential bowl 0.5 r^2 with |psi(x,y)|^2 on the floor.
Right panel: its Fourier mirror -- the kinetic bowl 0.5 k^2 (the identical
paraboloid for omega = 1) with the momentum density |psi~(k)|^2 on the floor.

Harmonic evolution rotates phase space, so the panels show the same movie a
quarter period apart, and the CoM markers counter-rotate: when the position
markers reach the bottom of the well, the momentum markers are at maximum
displacement, and vice versa.

The markers follow independent pieces of the state, split off at t = 0 and
evolved with the same operator (exact, by linearity). The floor images are
probability densities on their own scale, kept in a band below the bowls and
never added to V or K -- densities and energies have different units.

The readout shows the total probability; the split-operator method is
unitary, so it drifts only by round-off. Energy drift (Trotter splitting
error) is reported in the console at the end.
"""

from __future__ import annotations

import pathlib
import time

import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt

from quantum_oscillator.data import get_initial_state
from quantum_oscillator.operators import lie_trotter_step
from quantum_oscillator.physics import get_propagators
from quantum_oscillator.utils import to_host

SAVE_DIR = pathlib.Path(__file__).resolve().parent.parent / "animation_output"
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

    Connected regions above a small density threshold seed a Voronoi
    assignment of every pixel, so the pieces tile the whole grid and sum
    back to psi exactly -- which is what makes evolving them independently
    legitimate.

    Returns (masks, seam, n_found); ``masks`` is None if fewer than n_track
    separable blobs exist.
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


def _momentum_density(psi_host: np.ndarray, M: int) -> np.ndarray:
    """|psi~(k)|^2 on the fftshifted k grid, M points per axis.

    M beyond the grid size zero-pads psi, sampling the same spectrum on a
    finer k grid. Display-only: the dynamics never see it.
    """
    return np.abs(np.fft.fftshift(np.fft.fft2(psi_host, s=(M, M)))) ** 2


def generate_well_animation(
    N: int = 128,
    L: float = 10.0,
    num_steps: int = 300,  # integrator steps per quarter period (pi/2)
    target_time: float = 2 * np.pi,
    state_image: str = "triple_slit",
    blur: float = 0.35,
    backend: str = "torch",
    well_clip: float = 20.0,  # visual ceiling on both bowls, in energy units
    n_track: int = 3,  # marker points: one per blob of the initial state
    seam_tol: float = 1e-3,  # max density on a split line, relative to peak
    show_trails: bool = True,
    trail_length: int = 0,  # frames of path kept; 0 = the whole run
    show_energy_plane: bool = False,  # optional <E> disc
    k_upsample: int = 2,  # display-only zero-padding factor for the k panel
    surf_stride: int = 1,
    steps_per_frame: int = 3,
    fps: int = 30,
    dpi: int = 150,
    azim_sweep: float = 120.0,
    output_filename: str | None = None,
) -> str:
    dt = (np.pi / 2) / num_steps
    num_steps_total = round(target_time / dt)

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    if output_filename is None:
        output_filename = str(SAVE_DIR / f"3d_well_sim_{state_image}.mp4")
    if n_track < 1:
        raise ValueError("n_track must be at least 1")

    print("\n" + "=" * 60)
    print(f" GENERATING 3D WELL ANIMATION: {state_image.upper()} ")
    print("=" * 60)

    # 1. Physics engine ------------------------------------------------------
    psi = get_initial_state(N, L, state_image, blur, backend=backend)
    V, K = get_propagators(N, L, backend=backend)

    x = np.linspace(-L, L, N, endpoint=False)
    dx = 2 * L / N
    X, Y = np.meshgrid(x, x, indexing="xy")

    # fftshifted k axis. The physical spacing dk = pi/L is coarser than dx,
    # so the display transform zero-pads to M points per axis (k_upsample)
    # to make both panels resolve comparably.
    M = N * max(1, int(k_upsample))
    k_ax = np.fft.fftshift(2 * np.pi * np.fft.fftfreq(M, d=dx))
    KX, KY = np.meshgrid(k_ax, k_ax, indexing="xy")

    V_host = np.real(to_host(V))
    K_host = np.real(to_host(K))

    # get_propagators negates the whole Hamiltonian (V = -0.5 r^2,
    # K = -0.5 k^2), i.e. the oscillator run backwards; display textbook signs
    inverted = float(V_host.mean()) < 0.0
    V_std = -V_host if inverted else V_host
    K_std = -K_host if inverted else K_host
    if inverted:
        print(" note: V from get_propagators is negative; displaying textbook signs.")

    V_visual = np.clip(V_std, 0.0, well_clip)
    # kinetic bowl, same energy units and ceiling so the panels share a z axis
    K_disp = 0.5 * (KX**2 + KY**2)
    K_visual = np.clip(K_disp, 0.0, well_clip)

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
        K_visual = np.clip(K_disp, 0.0, well_clip)

    # the state's display band sits below the well minimum (z <= 0), so it
    # never overlaps the bowls
    blob_band = 0.20 * well_clip
    z_floor = -blob_band
    z_top = well_clip + 0.5

    # 2. Physics-only pre-pass: fixed brightness scales --------------------
    print(" pre-pass: finding the global peaks so one fixed scale fits the run...")
    scan = psi
    peak = 0.0
    peak_k_sq = 0.0
    for step in range(num_steps_total + 1):
        if step % steps_per_frame == 0:
            scan_host = to_host(scan)
            peak = max(peak, float(np.abs(scan_host).max()))
            peak_k_sq = max(peak_k_sq, float(_momentum_density(scan_host, M).max()))
        scan = lie_trotter_step(scan, V, K, dt)
    del scan
    peak_ref = peak + 1e-30
    peak_k_ref = np.sqrt(peak_k_sq) + 1e-30

    # For omega = 1 the quarter-period rotation maps x <-> k one-to-one, so
    # the momentum panel spans the same +/-L as the position panel: the state
    # appears at the same scale in both. The k grid's excess range (out to
    # pi/dx) holds only faint spectral tails, cropped from display.
    k_lim = L
    ksl = np.flatnonzero(np.abs(k_ax) <= k_lim)
    ksl = slice(int(ksl[0]), int(ksl[-1]) + 1)

    psi_host = to_host(psi)
    prob = np.abs(psi_host) ** 2

    # 3. Split the state into the pieces whose centres we will follow --------
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
        # a cut through a blob gives the piece a hard edge, which a spectral
        # method turns into ringing; its centre of mass then means little
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
    print(f" following {n_markers} point(s) on the state, in both domains")

    # 4. Figure: position panel left, momentum panel right -------------------
    fig = plt.figure(figsize=(16, 7.2))
    fig.patch.set_facecolor("#111111")

    def _add_3d(pos):
        try:
            axi = fig.add_subplot(pos, projection="3d", computed_zorder=False)
        except (AttributeError, TypeError):  # matplotlib < 3.5
            axi = fig.add_subplot(pos, projection="3d")
        axi.set_facecolor("#111111")
        return axi

    ax_x = _add_3d(121)
    ax_k = _add_3d(122)

    wire_kw = {"color": "#9db4c6", "alpha": 0.55, "linewidth": 0.6, "zorder": 1}
    ax_x.plot_wireframe(X, Y, V_visual, rstride=4, cstride=4, **wire_kw)
    k_wire_stride = 4 * max(1, int(k_upsample))  # same line density as the left
    ax_k.plot_wireframe(
        KX[ksl, ksl],
        KY[ksl, ksl],
        K_visual[ksl, ksl],
        rstride=k_wire_stride,
        cstride=k_wire_stride,
        **wire_kw,
    )

    if show_energy_plane:
        for axi, bowl, XX, YY, sl in (
            (ax_x, V_std, X, Y, slice(None)),
            (ax_k, K_disp, KX, KY, ksl),
        ):
            plane = np.full_like(bowl, E0)
            plane[bowl > E0] = np.nan
            axi.plot_surface(
                XX[sl, sl],
                YY[sl, sl],
                plane[sl, sl],
                color="#FCFDBF",
                alpha=0.18,
                shade=False,
                rstride=2,
                cstride=2,
                linewidth=0,
                antialiased=False,
                zorder=2,
            )

    # 5. The state, lying flat on the floor of each panel --------------------
    stride = max(1, int(surf_stride))

    def make_blobs(axi, XX, YY, prob_now, ref, sl):
        """The density as blobs on the floor, on one fixed |psi| scale.

        The gamma is a fixed transfer curve, identical every frame: it lifts
        the dim end so the spread-out state stays visible without per-frame
        renormalisation, so height and brightness stay comparable across the
        whole run.
        """
        t = np.clip(np.sqrt(prob_now) / ref, 0.0, 1.0)
        Z = z_floor + t * blob_band
        rgba = MAGMA(t**0.6)
        rgba[..., 3] = np.clip(3.5 * t, 0.0, 1.0) ** 0.6
        return axi.plot_surface(
            XX[sl, sl][::stride, ::stride],
            YY[sl, sl][::stride, ::stride],
            Z[sl, sl][::stride, ::stride],
            facecolors=rgba[sl, sl][::stride, ::stride],
            shade=False,
            rstride=1,
            cstride=1,
            linewidth=0,
            antialiased=False,
            zorder=0,
        )

    def _bowl_height(a, b):
        # analytic, not a grid lookup: nearest-cell sampling quantises the
        # height and makes the marker paths stair-step
        return min(0.5 * (a * a + b * b), well_clip)

    def _centroid(AA, BB, density):
        """Centre of mass of one piece, lifted onto its bowl."""
        m = float(density.sum()) + 1e-30
        ac = float((AA * density).sum() / m)
        bc = float((BB * density).sum() / m)
        return ac, bc, _bowl_height(ac, bc)

    def component_probs():
        hosts = [to_host(c) for c in components]
        return (
            [np.abs(h) ** 2 for h in hosts],
            [_momentum_density(h, M) for h in hosts],
        )

    blobs_x = make_blobs(ax_x, X, Y, prob, peak_ref, slice(None))
    blobs_k = make_blobs(ax_k, KX, KY, _momentum_density(psi_host, M), peak_k_ref, ksl)

    markers_x, markers_k = [], []
    trails_x, trails_k = [], []
    history_x = [[] for _ in range(n_markers)]
    history_k = [[] for _ in range(n_markers)]
    for i in range(n_markers):
        c = DOT_COLORS[i % len(DOT_COLORS)]
        for axi, markers, label in (
            (ax_x, markers_x, f"CoM {i + 1}"),
            (ax_k, markers_k, None),
        ):
            markers.append(
                axi.plot(
                    [],
                    [],
                    [],
                    "o",
                    color=c,
                    markersize=9,
                    markeredgecolor="white",
                    markeredgewidth=0.8,
                    zorder=8,
                    label=label,
                )[0]
            )
        if show_trails:
            for axi, trails in ((ax_x, trails_x), (ax_k, trails_k)):
                trails.append(
                    axi.plot([], [], [], color=c, linewidth=2.0, alpha=0.85, zorder=7)[
                        0
                    ]
                )

    # 6. Axes -----------------------------------------------------------------
    title = fig.suptitle("t = 0.00", color="white", fontsize=15, fontweight="bold")
    ax_x.set_title("Position domain", color="white", fontsize=12, pad=12)
    ax_k.set_title("Momentum (Fourier) domain", color="white", fontsize=12, pad=12)

    ax_x.set_xlim(-L, L)
    ax_x.set_ylim(-L, L)
    ax_k.set_xlim(-k_lim, k_lim)
    ax_k.set_ylim(-k_lim, k_lim)
    for axi in (ax_x, ax_k):
        axi.set_zlim(z_floor, z_top)
        try:
            axi.set_box_aspect((1, 1, 0.68), zoom=1.2)
        except (AttributeError, TypeError):
            try:
                axi.set_box_aspect((1, 1, 0.68))
            except AttributeError:
                pass
        axi.tick_params(colors="white", labelsize=8)
        axi.set_zlabel("Energy", color="white", labelpad=10)
        # transparent panes so the box reads as black, with light grid lines
        for axis in (axi.xaxis, axi.yaxis, axi.zaxis):
            axis.set_pane_color((0.0, 0.0, 0.0, 0.0))
            try:
                axis._axinfo["grid"]["color"] = (1.0, 1.0, 1.0, 0.30)
                axis._axinfo["grid"]["linewidth"] = 0.6
            except (AttributeError, KeyError):
                pass

    ax_x.set_xlabel("x", color="white", labelpad=10)
    ax_x.set_ylabel("y", color="white", labelpad=10)
    ax_k.set_xlabel("$k_x$", color="white", labelpad=10)
    ax_k.set_ylabel("$k_y$", color="white", labelpad=10)

    ax_x.legend(
        loc="upper right",
        framealpha=0.1,
        edgecolor="white",
        labelcolor="white",
        fontsize=9,
        bbox_to_anchor=(0.95, 0.95),
    )

    # 7. Loop -------------------------------------------------------------------
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
                psi_now = to_host(psi)
                prob = np.abs(psi_now) ** 2
                _, _, e_exp, nrm = observables(psi)
                e_drift = max(e_drift, abs(e_exp - E0))

                _remove(blobs_x)
                _remove(blobs_k)
                blobs_x = make_blobs(ax_x, X, Y, prob, peak_ref, slice(None))
                blobs_k = make_blobs(
                    ax_k, KX, KY, _momentum_density(psi_now, M), peak_k_ref, ksl
                )

                probs_x, probs_k = component_probs()
                for i in range(n_markers):
                    for AA, BB, p_i, markers, trails, history in (
                        (X, Y, probs_x[i], markers_x, trails_x, history_x),
                        (KX, KY, probs_k[i], markers_k, trails_k, history_k),
                    ):
                        xc, yc, zc = _centroid(AA, BB, p_i)
                        markers[i].set_data_3d(
                            np.array([xc]), np.array([yc]), np.array([zc])
                        )
                        if show_trails:
                            history[i].append((xc, yc, zc))
                            if trail_length > 0:
                                history[i] = history[i][-trail_length:]
                            hx, hy, hz = (np.asarray(a) for a in zip(*history[i]))
                            # nudged off the surface to avoid z-fighting
                            trails[i].set_data_3d(hx, hy, hz + 0.06 * blob_band)

                t_now = step * dt
                title.set_text(
                    f"t = {t_now:.2f}   ({t_now / np.pi:.2f}π), "
                    f"∫|ψ|² = {nrm * nrm:#.4g}"
                )

                azim = -45 + (step / num_steps_total) * azim_sweep
                for axi in (ax_x, ax_k):
                    axi.view_init(elev=24, azim=azim)
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


def _parse_args():
    import argparse

    p = argparse.ArgumentParser(
        description="3D harmonic-well animation: position and momentum panels."
    )
    p.add_argument("--N", type=int, default=128, help="grid points per axis")
    p.add_argument("--L", type=float, default=10.0, help="half-width of the box")
    p.add_argument(
        "--num-steps", type=int, default=300, help="integrator steps per quarter period"
    )
    p.add_argument(
        "--target-time", type=float, default=2 * np.pi, help="total evolution time"
    )
    p.add_argument("--state-image", default="triple_slit", help="initial state image")
    p.add_argument("--blur", type=float, default=0.35, help="Gaussian blur (pixels)")
    p.add_argument("--backend", default="torch", choices=("numpy", "torch"))
    p.add_argument(
        "--well-clip", type=float, default=20.0, help="visual ceiling on the bowls"
    )
    p.add_argument("--n-track", type=int, default=3, help="CoM markers: one per blob")
    p.add_argument(
        "--seam-tol", type=float, default=1e-3, help="max relative density on a cut"
    )
    p.add_argument("--show-trails", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--trail-length", type=int, default=0, help="trail frames kept; 0 = all"
    )
    p.add_argument(
        "--show-energy-plane", action=argparse.BooleanOptionalAction, default=False
    )
    p.add_argument(
        "--k-upsample", type=int, default=2, help="display zero-padding for the k panel"
    )
    p.add_argument("--surf-stride", type=int, default=1)
    p.add_argument("--steps-per-frame", type=int, default=3)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument(
        "--azim-sweep", type=float, default=0.0, help="camera azimuth sweep (degrees)"
    )
    p.add_argument("--output-filename", default=None)
    return vars(p.parse_args())


if __name__ == "__main__":
    generate_well_animation(**_parse_args())
