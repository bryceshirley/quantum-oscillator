#!/usr/bin/env python3
"""arrows_demo.py — renders animation_output/splitting-methods.mp4.

Non-commuting operators, told with arrows: A translates a vector, B
rotates it, and e^{(A+B)z} != e^{Az}e^{Bz}. The schemes are the matrix
twins of the operators in quantum_oscillator.operators (a 3x3 matrix
exponential stands in for each FFT phase factor; translate ~ the kinetic
drift, rotate ~ the potential kick).

Staged in two panels on one white 16:9 slide, letterboxed to 1280x720:

  1. the opening            a single centred panel tells the one-step
                            story: the exact flow travels first, then
                            both Lie-Trotter orderings, then the navy
                            double arrow measures the gap
  2. the transformation     the panel glides left and BECOMES the
                            "chain it" panel while the "splitting error"
                            log-log panel slides in from the right
  3. the acts               one method at a time re-runs the journey in
                            n = 1, 2, 4, 8 sub-steps and visibly
                            converges to the dashed exact flow: Lie,
                            then Strang (with the conjugated Lie chain,
                            which telescopes to exactly Strang), then
                            4th-order Suzuki — error points appearing in
                            sync over faint full-curve ghosts

Run:
  uv run python arrows_demo.py                 # ~32 s mp4
  uv run python arrows_demo.py --seconds 24
  uv run python arrows_demo.py --still 300     # one frame, to check

Writes animation_output/splitting-methods.mp4.
"""

import argparse
import subprocess
import tempfile
from pathlib import Path

import matplotlib
import numpy as np
from scipy.linalg import expm

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Polygon

OUTDIR = Path(__file__).resolve().parent.parent / "animation_output"
OUTDIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------- the maths
OMEGA = 2.0
VEL = np.array([1.5, 0.0])
Z = 1.0

A = np.zeros((3, 3))
A[0, 2], A[1, 2] = VEL  # translate ~ drift (K)
B = np.array(
    [
        [0.0, -OMEGA, 0.0],
        [OMEGA, 0.0, 0.0],  # rotate ~ kick (V)
        [0.0, 0.0, 0.0],
    ]
)

# The chains, leg by leg (generator, fraction of the total time t).
# Products act right to left, so [e^{Ah}e^{Bh}]^n rotates first.
# Five-fold 4th-order Suzuki fractal, the same composition as
# quantum_oscillator.operators.suzuki_trotter_step(order=4).
W1 = 1.0 / (4.0 - 4.0 ** (1.0 / 3.0))
W0 = 1.0 - 4.0 * W1  # the backward middle step


def lie_legs(n):
    return [(B, 1 / n), (A, 1 / n)] * n


def strang_legs(n):
    return [(A, 0.5 / n), (B, 1 / n), (A, 0.5 / n)] * n


def conj_lie_legs(n):
    """e^{Ah/2} [e^{Bh}e^{Ah}]^n e^{-Ah/2} — telescopes to Strang^n."""
    return [(A, -0.5 / n)] + [(A, 1 / n), (B, 1 / n)] * n + [(A, 0.5 / n)]


def suzuki_legs(n):
    def s(w):
        return [(A, w / 2), (B, w), (A, w / 2)]

    legs = []
    for _ in range(n):
        for w in (W1, W1, W0, W1, W1):
            legs += s(w / n)
    return legs


def split_M(legs, s):
    """Transform at progress s of a chain, walked leg by leg."""
    total = sum(abs(w) for _, w in legs)
    M, acc = np.eye(3), 0.0
    for G, w in legs:
        frac = abs(w) / total
        if s < acc + frac:
            return expm(G * Z * w * (s - acc) / frac) @ M
        M = expm(G * Z * w) @ M
        acc += frac
    return M


P0 = np.array([1.15, 0.0])
P0H = np.array([1.15, 0.0, 1.0])


def move(M, pts):
    q = np.column_stack([pts, np.ones(len(pts))]) @ M.T
    return q[:, :2]


def path(legs, npts=200):
    return np.array([(split_M(legs, s) @ P0H)[:2] for s in np.linspace(0, 1, npts)])


M_EX = expm((A + B) * Z)
P_EX = (M_EX @ P0H)[:2]
EX_PATH = np.array([(expm((A + B) * Z * t) @ P0H)[:2] for t in np.linspace(0, 1, 200)])

TR_LEGS = [(A, 1.0), (B, 1.0)]  # translate -> rotate (panel 1 only)
P_RT = (split_M(lie_legs(1), 1.0) @ P0H)[:2]
P_TR = (split_M(TR_LEGS, 1.0) @ P0H)[:2]
GAP = float(np.linalg.norm(P_RT - P_TR))
RT_PATH, TR_PATH = path(lie_legs(1)), path(TR_LEGS)

SUB_NS = (1, 2, 4, 8)  # revealed in the acts
NS_FULL = tuple(2**k for k in range(8))
CHAINS = {
    "lie": lie_legs,
    "strang": strang_legs,
    "conj_lie": conj_lie_legs,
    "suzuki": suzuki_legs,
}
PATHS = {name: {n: path(f(n)) for n in SUB_NS} for name, f in CHAINS.items()}
ERRS = {
    name: [
        float(np.linalg.norm((split_M(f(n), 1.0) @ P0H)[:2] - P_EX)) for n in NS_FULL
    ]
    for name, f in CHAINS.items()
}

# ---------------------------------------------------------------- style
NAVY, ORANGE, TEAL = "#16305C", "#D9820F", "#12766F"
BLUE, LIGHTBLUE, CRIMSON = "#2E6DB4", "#7FA8DC", "#B03A66"
SOFT, RULE, GREY, GHOST = "#5A6B85", "#C3CEDB", "#9AA7B8", "#CBD5E0"
START_GREY, PALE = "#8794A6", "#C3CCD9"

# the block-arrow glyph, tail at the origin: the vector being evolved
SHAPE = 1.25 * np.array(
    [
        [0.0, 0.05],
        [0.62, 0.05],
        [0.62, 0.22],
        [0.95, 0.0],
        [0.62, -0.22],
        [0.62, -0.05],
        [0.0, -0.05],
    ]
)

# each act: (key, panel-2 formula rows, panel-2 travellers, panel-3 lines)
ACTS = [
    (
        "lie",
        [("Lie", ORANGE, r"$[e^{Ah}e^{Bh}]^n$", 0.115)],
        [("lie", ORANGE, "-")],
        [("lie", ORANGE, "-o", 1.6, 4.0, "Lie   O(1/n)")],
    ),
    (
        "strang",
        [
            ("Strang", BLUE, r"$[e^{Ah/2}e^{Bh}e^{Ah/2}]^n$", 0.185),
            ("Conj. Lie", LIGHTBLUE, r"$e^{Ah/2}[e^{Bh}e^{Ah}]^n\,e^{-Ah/2}$", 0.235),
        ],
        [("strang", BLUE, "-"), ("conj_lie", LIGHTBLUE, "--")],
        [
            ("strang", BLUE, "-", 2.2, 0.0, "Strang   O(1/n²)"),
            ("conj_lie", BLUE, "--x", 1.0, 5.0, "Conj. Lie"),
        ],
    ),
    (
        "suzuki",
        [
            (
                "Suzuki",
                CRIMSON,
                (
                    r"$[S(w_1h)^2\,S(w_0h)\,S(w_1h)^2]^n$"
                    r"   $(S(x)\equiv e^{Ax/2}e^{Bx}e^{Ax/2})$"
                ),
                0.175,
            )
        ],
        [("suzuki", CRIMSON, "-")],
        [("suzuki", CRIMSON, "-d", 1.6, 4.0, "Suzuki   O(1/n⁴)")],
    ),
]

# ---------------------------------------------------------------- timeline
FPS = 25
# after the one-step story the lone panel holds for a beat, then the
# layout opens up (solo_hold, slide) and the acts begin
PH = {
    "intro_exact": 1.6,
    "intro_travel": 2.2,
    "intro_gap": 1.0,
    "solo_hold": 0.7,
    "slide": 1.2,
}
for _key, _n1, _n2, _n4, _n8 in (
    ("lie", 1.7, 1.2, 1.2, 1.5),
    ("strang", 2.1, 1.4, 1.4, 1.7),
    ("suzuki", 2.1, 1.4, 1.4, 1.7),
):
    # the pause after n = 1 lets the single-step endpoint register
    PH.update(
        {
            f"{_key}_1": _n1,
            f"{_key}_pause1": 0.6,
            f"{_key}_2": _n2,
            f"{_key}_4": _n4,
            f"{_key}_8": _n8,
            f"{_key}_hold": 0.6,
        }
    )
PH["end_hold"] = 2.7


def schedule(total):
    scale = total / sum(PH.values())
    t, out = 0.0, {}
    for k, v in PH.items():
        out[k] = (t * scale, (t + v) * scale)
        t += v
    return out


def _frac(t, ph, phase):
    a, b = ph[phase]
    return float(np.clip((t - a) / (b - a), 0.0, 1.0))


def ease(u):
    return u * u * (3.0 - 2.0 * u)


# ---------------------------------------------------------------- drawing
# both geometry panels share one frame so the diagrams line up exactly;
# the y range is chosen so the data aspect fills the identical boxes
GEO_XLIM = (-2.40, 3.20)
GEO_YLIM = (-1.50, 3.08)


# one geometry panel carries the whole story: it opens centre stage with
# the one-step scene, then glides left and becomes the chain panel while
# the error panel slides in from beyond the right edge. Two panels remain.
R_CENTER, R_GEO = (0.3495, 0.185, 0.301, 0.590), (0.115, 0.185, 0.301, 0.590)
R_ERR_OFF, R_ERR = (1.430, 0.185, 0.270, 0.590), (0.615, 0.185, 0.270, 0.590)


def lerp(a, b, s):
    return tuple(ai + (bi - ai) * s for ai, bi in zip(a, b))


def build_figure():
    fig = plt.figure(figsize=(12.80, 5.34), dpi=100)
    fig.patch.set_facecolor("white")
    ax_geo = fig.add_axes(R_CENTER)
    ax_err = fig.add_axes(R_ERR_OFF)
    fig.text(
        0.035,
        0.951,
        "Non-commuting operators:",
        color=NAVY,
        fontsize=13.5,
        fontweight="bold",
        va="center",
    )
    fig.text(
        0.268,
        0.951,
        r"$e^{(A+B)t} \neq e^{At}e^{Bt}$",
        color=NAVY,
        fontsize=13.5,
        va="center",
    )
    fig.text(
        0.035,
        0.892,
        r"$A$ translates,   $B$ rotates,   $[A,B] \neq 0$",
        color=SOFT,
        fontsize=10.5,
        va="center",
    )
    return fig, (ax_geo, ax_err)


def _geo_axis(ax, title, xlim, ylim):
    ax.set_facecolor("white")
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(RULE)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_title(title, color=NAVY, fontsize=11.5, fontweight="bold", pad=7)
    ax.plot(0, 0, "+", color=SOFT, ms=9, mew=1.4, zorder=2)


def _glyph(ax, M, c, alpha=1.0, z=6):
    ax.add_patch(
        Polygon(
            move(M, SHAPE),
            closed=True,
            fc=c,
            ec=c,
            alpha=alpha,
            lw=1.2,
            zorder=z,
            joinstyle="round",
        )
    )


def _start(ax, label_dy=-0.42):
    _glyph(ax, np.eye(3), START_GREY, z=4)
    ax.text(
        P0[0] + 0.25,
        label_dy,
        "start",
        color=SOFT,
        fontsize=10.5,
        fontweight="bold",
        ha="center",
    )


def _current_act(t, ph):
    """The latest act whose first sub-round has begun, and its state."""
    current = None
    for key, rows, travellers, lines in ACTS:
        if _frac(t, ph, f"{key}_1") > 0:
            current = (key, rows, travellers, lines)
    return current


def draw_panel1(ax, t, ph):
    _geo_axis(ax, "one step:  the order matters", GEO_XLIM, GEO_YLIM)
    _start(ax)
    ue = ease(_frac(t, ph, "intro_exact"))
    if ue > 0:
        k = max(2, int(ue * len(EX_PATH)))
        ax.plot(
            EX_PATH[:k, 0],
            EX_PATH[:k, 1],
            "--",
            color=GREY,
            lw=1.7,
            dashes=(4, 3),
            zorder=3,
        )
        _glyph(ax, expm((A + B) * Z * ue), PALE, z=5)
    if ue >= 1.0:
        ax.text(
            P_EX[0] + 0.35,
            P_EX[1] + 0.05,
            "exact",
            color=GREY,
            fontsize=10,
            fontweight="bold",
            ha="left",
            va="bottom",
        )
    u = ease(_frac(t, ph, "intro_travel"))
    if u > 0:
        m = max(2, int(u * len(RT_PATH)))
        ax.plot(
            RT_PATH[:m, 0],
            RT_PATH[:m, 1],
            "-",
            color=ORANGE,
            lw=1.5,
            alpha=0.85,
            zorder=3,
        )
        ax.plot(
            TR_PATH[:m, 0],
            TR_PATH[:m, 1],
            "-",
            color=TEAL,
            lw=1.5,
            alpha=0.85,
            zorder=3,
        )
        _glyph(ax, split_M(lie_legs(1), u), ORANGE, z=7)
        _glyph(ax, split_M(TR_LEGS, u), TEAL, z=7)
    ax.text(
        0.0,
        -0.055,
        "rotate + translate",
        transform=ax.transAxes,
        color=GREY,
        fontsize=10.5,
        fontweight="bold",
        va="top",
    )
    ax.text(
        0.0,
        -0.135,
        "rotate → translate",
        transform=ax.transAxes,
        color=ORANGE,
        fontsize=10.5,
        fontweight="bold",
        va="top",
    )
    ax.text(
        0.0,
        -0.215,
        "translate → rotate",
        transform=ax.transAxes,
        color=TEAL,
        fontsize=10.5,
        fontweight="bold",
        va="top",
    )
    v = _frac(t, ph, "intro_gap")
    if v > 0:
        ax.annotate(
            "",
            xy=tuple(P_TR),
            xytext=tuple(P_RT),
            arrowprops={
                "arrowstyle": "<->",
                "color": NAVY,
                "lw": 1.6,
                "alpha": v,
                "shrinkA": 2,
                "shrinkB": 2,
            },
            zorder=9,
        )
        ax.text(
            0.08,
            1.60,
            "gap",
            color=NAVY,
            fontsize=10.5,
            fontweight="bold",
            alpha=v,
            ha="right",
            va="top",
            zorder=11,
        )


def draw_panel2(ax, t, ph):
    act = _current_act(t, ph)
    key, rows, travellers = (
        (act[0], act[1], act[2]) if act else (ACTS[0][0], ACTS[0][1], ACTS[0][2])
    )

    # which sub-round of the act is running, and how far along it is
    n_now, u = SUB_NS[0], 0.0
    if act:
        for n in SUB_NS:
            f = _frac(t, ph, f"{key}_{n}")
            if f > 0:
                n_now, u = n, f
        if _frac(t, ph, f"{key}_hold") > 0:
            n_now, u = SUB_NS[-1], 1.0

    _geo_axis(ax, f"chain it:  n = {n_now}", GEO_XLIM, GEO_YLIM)
    ax.plot(
        EX_PATH[:, 0], EX_PATH[:, 1], "--", color=GREY, lw=1.7, dashes=(4, 3), zorder=3
    )
    _glyph(ax, M_EX, PALE, z=5)
    _start(ax, label_dy=-0.45)

    if act:
        e = ease(u)
        for name, color, style in travellers:
            legs = CHAINS[name](n_now)
            p = PATHS[name][n_now]
            m = max(2, int(e * len(p)))
            ax.plot(p[:m, 0], p[:m, 1], style, color=color, lw=1.5, alpha=0.9, zorder=4)
            _glyph(ax, split_M(legs, e), color, z=8)

    # the formula legend, underneath the box
    y = -0.055
    for name, color, formula, fx in rows:
        ax.text(
            0.0,
            y,
            name,
            transform=ax.transAxes,
            color=color,
            fontsize=9,
            fontweight="bold",
            va="top",
        )
        ax.text(
            fx, y, formula, transform=ax.transAxes, color=color, fontsize=8.5, va="top"
        )
        y -= 0.08
    ax.text(
        0.0,
        y,
        "Exact",
        transform=ax.transAxes,
        color=GREY,
        fontsize=9,
        fontweight="bold",
        va="top",
    )
    ax.text(
        0.13,
        y,
        r"$e^{(A+B)t}$",
        transform=ax.transAxes,
        color=GREY,
        fontsize=8.5,
        va="top",
    )


def draw_panel3(ax, t, ph):
    ax.set_facecolor("white")
    for sp in ax.spines.values():
        sp.set_color(RULE)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.75, 170)
    ax.set_ylim(2e-11, 6)
    ax.set_yticks([1e0, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10])
    ax.tick_params(colors=SOFT, labelsize=8)
    ax.grid(True, which="both", color=RULE, alpha=0.35, lw=0.5)
    ax.set_xlabel("number of steps  n", color=SOFT, fontsize=9.5)
    ax.set_ylabel(r"$\|$split $-$ exact$\|$", color=SOFT, fontsize=9.5)
    ax.set_title("splitting error", color=NAVY, fontsize=11.5, fontweight="bold", pad=7)

    for name in ("lie", "strang", "suzuki"):  # full-curve ghosts
        ax.plot(NS_FULL, ERRS[name], "-", color=GHOST, lw=1.1, zorder=2)

    handles, labels = [], []
    for key, rows, travellers, lines in ACTS:
        if _frac(t, ph, f"{key}_1") <= 0:
            break
        # a point appears the moment its sub-round begins
        k = sum(1 for n in SUB_NS if _frac(t, ph, f"{key}_{n}") > 0)
        for name, color, style, lw, ms, label in lines:
            (line,) = ax.plot(
                NS_FULL[:k], ERRS[name][:k], style, color=color, lw=lw, ms=ms, zorder=4
            )
            handles.append(line)
            labels.append(label)
    if handles:
        ax.legend(
            handles,
            labels,
            loc="lower left",
            frameon=True,
            facecolor="white",
            edgecolor="none",
            framealpha=0.85,
            fontsize=8,
            handlelength=2.0,
            labelcolor=SOFT,
            borderaxespad=0.4,
        )


def draw(axes, t, ph):
    ax_geo, ax_err = axes
    s = ease(_frac(t, ph, "slide"))
    ax_geo.clear()
    ax_err.clear()
    ax_geo.set_position(lerp(R_CENTER, R_GEO, s))
    ax_err.set_position(lerp(R_ERR_OFF, R_ERR, s))
    if t < ph["slide"][0]:
        draw_panel1(ax_geo, t, ph)  # the one-step story, centre stage
    else:
        draw_panel2(ax_geo, t, ph)  # the same panel, now "chain it"
    draw_panel3(ax_err, t, ph)


# ---------------------------------------------------------------- render
def _ffmpeg(args):
    r = subprocess.run(
        ["ffmpeg", "-y", *args], capture_output=True, text=True, check=False
    )
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed ({r.returncode}):\n" + r.stderr[-2000:])


def render(seconds, out_mp4):
    ph = schedule(seconds)
    frames = int(seconds * FPS)
    fig, axes = build_figure()

    def draw_frame(i):
        draw(axes, i / FPS, ph)
        return []

    anim = animation.FuncAnimation(fig, draw_frame, frames=frames)
    with tempfile.TemporaryDirectory() as td:
        raw = Path(td) / "raw.mp4"
        writer = animation.FFMpegWriter(
            fps=FPS, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-crf", "18"]
        )
        anim.save(str(raw), writer=writer, savefig_kwargs={"facecolor": "white"})
        # letterbox the 1280x534 slide onto 16:9, like the original export
        _ffmpeg(
            [
                "-i",
                str(raw),
                "-vf",
                "pad=1280:720:0:93:black",
                "-c:v",
                "libx264",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                str(out_mp4),
            ]
        )
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=32.0)
    ap.add_argument(
        "--still",
        type=int,
        default=None,
        help="render frame N to /tmp/sm_frame.png and stop",
    )
    args = ap.parse_args()

    if args.still is not None:
        ph = schedule(args.seconds)
        fig, axes = build_figure()
        draw(axes, args.still / FPS, ph)
        fig.savefig("/tmp/sm_frame.png", facecolor="white")
        print("wrote /tmp/sm_frame.png")
    else:
        out = OUTDIR / "splitting-methods.mp4"
        render(args.seconds, out)
        print(f"[ok] {out}")
