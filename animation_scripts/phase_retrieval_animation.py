"""Watch phase retrieval happen: the horse assembling itself out of noise.

Runs the blind reconstruction from the demo notebook's section 7 -- recover a
never-seen image from nothing but the noisy, phaseless brightness a camera
records at the Fourier plane, a positivity constraint, a loose support box and
a total-variation penalty against noise-fitting speckle -- and renders the
optimisation as a three-panel animation:

  1. the current guess for the initial image (t = 0), starting as random
     speckle and converging to the horse;
  2. that guess evolved half a period to t = pi, so the upside-down horse
     appears the moment the reconstruction becomes right;
  3. the loss falling as Adam walks downhill through the physics.

Torch autodiff differentiates through every FFT of the split-operator
integrator; the same `evolve` that runs the forward physics is the forward
model of the inverse problem.

By default the run is written to animation_output/phase_retrieval_<image>.mp4
(requires ffmpeg); pass --live to watch it in a matplotlib window instead.
"""

import pathlib
import time

import matplotlib
import numpy as np
import torch

from quantum_oscillator.data import get_initial_state
from quantum_oscillator.operators import evolve
from quantum_oscillator.physics import get_propagators
from quantum_oscillator.plotting import complex_to_rgb, tidy
from quantum_oscillator.utils import to_host

SAVE_DIR = pathlib.Path(__file__).resolve().parent.parent / "animation_output"


def total_variation(image):
    """The summed jumps between neighbouring pixels."""
    return (image[1:, :] - image[:-1, :]).abs().mean() + (
        image[:, 1:] - image[:, :-1]
    ).abs().mean()


def generate_retrieval_animation(
    N: int = 256,  # matches the notebook's N_PR: no TV bloat, same GPU cost
    state_image: str = "horse",
    blur: float = 1.0,
    iters: int = 5000,
    lr: float = 0.1,
    seed: int = 0,
    noise: float = 1.0,  # measurement noise, relative to the mean signal
    tv_lambda: float = 1e-3,  # TV weight in the loss
    noise_seed: int = 42,
    steps_per_quarter: int = 32,
    frame_every: int = 25,  # optimiser iterations per rendered frame
    fps: int = 15,
    dpi: int = 130,
    hold_seconds: float = 2.0,  # freeze on the final reconstruction
    live: bool = False,
) -> str | None:
    if not live:
        # Agg renders at exactly figsize * dpi; the macOS HiDPI backend hands
        # ffmpeg a double-size buffer and the video comes out sheared.
        matplotlib.use("Agg", force=True)
    from matplotlib import animation
    from matplotlib import pyplot as plt

    L = float(np.sqrt(np.pi * N / 2))  # self-dual box: t = pi/2 is exactly an FFT
    quarter = np.pi / 2

    print("\n" + "=" * 60)
    print(f" PHASE RETRIEVAL: {state_image.upper()} FROM BRIGHTNESS ALONE ")
    print(f" noise = {noise:g} x mean signal, TV lambda = {tv_lambda:g}")
    print("=" * 60)

    # 1. The ground-truth experiment the optimiser never sees ---------------
    psi_true = get_initial_state(N, L, state_image, blur, backend="torch")
    V, K = get_propagators(N, L, backend="torch")
    with torch.no_grad():
        # as_tensor is a no-op on tensors; it narrows the static type to torch
        measured = torch.as_tensor(
            abs(evolve(psi_true, V, K, quarter, steps_per_quarter))
        )
        if noise > 0:
            torch.manual_seed(noise_seed)
            sigma = noise * measured.mean()
            measured = (measured + sigma * torch.randn_like(measured)).clamp(min=0)
        true_img = abs(psi_true)

    # 2. The unknown image: non-negative by construction, inside a loose box
    torch.manual_seed(seed)
    a = (0.1 * torch.randn(N, N, device=measured.device)).requires_grad_(True)
    optimiser = torch.optim.Adam([a], lr=lr)

    margin = N // 4 - 4
    support = torch.zeros(N, N, device=measured.device)
    support[margin : N - margin, margin : N - margin] = 1.0

    def image_guess():
        x = (a**2) * support
        return x / torch.linalg.vector_norm(x)

    # 3. The three panels ----------------------------------------------------
    fig, (ax_now, ax_pi, ax_loss) = plt.subplots(1, 3, figsize=(13.5, 4.6))
    fig.subplots_adjust(top=0.82, wspace=0.25)

    def rgb(z):
        return complex_to_rgb(to_host(z))

    with torch.no_grad():
        guess0 = image_guess().to(psi_true.dtype)
        im_now = ax_now.imshow(rgb(guess0))
        im_pi = ax_pi.imshow(
            rgb(evolve(guess0, V, K, 2 * quarter, 2 * steps_per_quarter))
        )
    ax_now.set_title("current guess  |  t = 0", fontsize=11)
    ax_pi.set_title("the guess at t = $\\pi$", fontsize=11)
    for axi in (ax_now, ax_pi):
        axi.axis("off")

    (loss_line,) = ax_loss.semilogy([], [], lw=2)
    ax_loss.set_xlim(0, iters)
    ax_loss.set_xlabel("Adam iteration")
    ax_loss.set_ylabel("loss")
    ax_loss.set_title("reconstruction loss", fontsize=11)
    tidy(ax_loss)

    title = fig.suptitle("iteration 0", fontsize=13)

    writer = None
    output = None
    if not live:
        SAVE_DIR.mkdir(parents=True, exist_ok=True)
        output = SAVE_DIR / f"phase_retrieval_{state_image}.mp4"
        writer = animation.FFMpegWriter(
            fps=fps,
            metadata={"title": "Phase retrieval"},
            # libx264 needs even frame dimensions; pad by a pixel if necessary
            extra_args=["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"],
        )
        writer.setup(fig, str(output), dpi=dpi)
        print(f" writing {iters // frame_every + 1} frames to {output}")
    else:
        plt.ion()
        plt.show()

    def draw_frame(iteration, losses):
        with torch.no_grad():
            guess = image_guess().to(psi_true.dtype)
            im_now.set_data(rgb(guess))
            im_pi.set_data(rgb(evolve(guess, V, K, 2 * quarter, 2 * steps_per_quarter)))
        loss_line.set_data(np.arange(len(losses)), losses)
        ax_loss.relim()
        ax_loss.autoscale_view(scalex=False)
        title.set_text(f"iteration {iteration}   |   loss {losses[-1]:.2e}")
        if live:
            fig.canvas.draw_idle()
            plt.pause(0.001)
        else:
            assert writer is not None  # live is False, so setup() ran above
            writer.grab_frame()

    # 4. Optimise, drawing as we go ------------------------------------------
    start = time.time()
    losses = []
    for iteration in range(iters + 1):
        optimiser.zero_grad()
        x = image_guess()
        psi_out = torch.as_tensor(
            evolve(x.to(psi_true.dtype), V, K, quarter, steps_per_quarter)
        )
        loss = ((abs(psi_out) - measured) ** 2).mean() + tv_lambda * total_variation(x)
        loss.backward()
        optimiser.step()
        losses.append(loss.item())

        if iteration % frame_every == 0:
            draw_frame(iteration, losses)
        if iteration % 1000 == 0:
            print(f"   iteration {iteration:5d}   loss {losses[-1]:.3e}")

    with torch.no_grad():
        overlap = float((image_guess() * true_img).sum())
    print(f"\n final overlap with the true image: {overlap:.3f}")
    print(f" optimised and rendered in {time.time() - start:.1f}s")

    if writer is not None:
        for _ in range(int(hold_seconds * fps)):  # freeze on the final frame
            writer.grab_frame()
        writer.finish()
        print(f" >>> wrote {output}")
        plt.close(fig)
        return str(output)

    plt.ioff()
    plt.show()  # keep the final window up
    return None


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Animate blind phase retrieval: recover the initial image "
        "from the noisy, phaseless brightness a camera records at the Fourier "
        "plane, regularised with total variation."
    )
    p.add_argument("--N", type=int, default=256, help="grid points per axis")
    p.add_argument("--state-image", default="horse", help="initial state image")
    p.add_argument("--blur", type=float, default=1.0, help="gaussian blur, pixels")
    p.add_argument("--iters", type=int, default=5000, help="Adam iterations")
    p.add_argument("--lr", type=float, default=0.1, help="Adam learning rate")
    p.add_argument("--seed", type=int, default=0, help="random init seed")
    p.add_argument(
        "--noise",
        type=float,
        default=1.0,
        help="measurement noise relative to the mean signal (0 for a clean camera)",
    )
    p.add_argument("--tv", type=float, default=1e-3, help="TV weight in the loss")
    p.add_argument("--frame-every", type=int, default=25, help="iterations per frame")
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--dpi", type=int, default=130)
    p.add_argument(
        "--live",
        action="store_true",
        help="show a live matplotlib window instead of writing an mp4",
    )
    args = p.parse_args()

    generate_retrieval_animation(
        N=args.N,
        state_image=args.state_image,
        blur=args.blur,
        iters=args.iters,
        lr=args.lr,
        seed=args.seed,
        noise=args.noise,
        tv_lambda=args.tv,
        frame_every=args.frame_every,
        fps=args.fps,
        dpi=args.dpi,
        live=args.live,
    )
