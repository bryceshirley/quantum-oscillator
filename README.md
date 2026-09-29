
# Quantum Oscillator to turn an image upside down and back up again

<p align="center">
  <img src="quantum_revival_horse.gif" width="420"
       alt="A horse silhouette as a quantum state, oscillating forever between upright and upside down in a harmonic well" />
</p>
<p align="center"><em>A horse, evolved as a wavefunction in a harmonic well: it transforms into its
Fourier transform, reappears upside down at half a period and revives – forever.</em></p>

Code and slides for a talk on building a quantum oscillator to rotate an image with
[array-api-compat](https://data-apis.org/array-api-compat/).

The integrators in
[`src/quantum_oscillator/operators.py`](src/quantum_oscillator/operators.py) never
import NumPy or PyTorch: they call `array_namespace(psi)` and do the maths in whatever
namespace the array belongs to. NumPy arrays run on the CPU; the same source lines
run torch tensors on a GPU.

**Write the maths once; let the data decide where it runs.**

## Repository layout

| Path | What it is |
| --- | --- |
| `src/quantum_oscillator/` | The package: the physics and the array-agnostic integrators |
| `integrators_demo.ipynb` | The demo notebook (see below) |
| `animation_scripts/` | Scripts that render the talk's animations (see below) |
| `test/` | The pytest suite |
| `Presentation.pptx` | The slides |

### The package, `src/quantum_oscillator/`

| Module | What it is |
| --- | --- |
| `operators.py` | Time steppers (forward Euler, Lie–Trotter, Strang, Suzuki–Trotter) and the `evolve` driver |
| `physics.py` | The spectral Hamiltonian and the `V`, `K` propagator grids |
| `data.py` | Initial states — analytic patterns and scikit-image samples (see below) |
| `plotting.py` | Domain-colouring renderers and plot styling |
| `utils.py` | Backend resolution, host/device transfer, `state_distance`, the `Array` protocol |

## Installation

The project is managed with [uv](https://docs.astral.sh/uv/); `uv.lock` pins the
exact environment.
[Install uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
git clone https://github.com/bryceshirley/quantum-oscillator.git
cd quantum-oscillator
uv sync
```

No activation needed: prefix commands with `uv run` and they execute in the project
environment (the right Python is downloaded if missing).

## The demo notebook

`integrators_demo.ipynb` is the talk in notebook form: why naive time stepping
blows up, how operator splitting fixes it and converges at its theoretical order,
the quarter-period milestones as a domain-coloured gallery, a CPU-vs-GPU benchmark
of the very same array-agnostic source (15–80× speed-ups on Apple silicon), and a
finale that recovers a never-seen image from a noisy, phaseless measurement by
differentiating through the physics.

```bash
uv run --with jupyterlab jupyter lab integrators_demo.ipynb
```

or open it in VS Code with the project `.venv` as the kernel. A full run takes about
two minutes; without Apple silicon the benchmark runs torch on the CPU.

## Animation scripts

The four scripts in `animation_scripts/` render into `animation_output/`. Each is
self-documenting — the full story is in its docstring, and every parameter is a
`--help` flag. The `.mp4` scripts need **ffmpeg** (`brew install ffmpeg` /
`sudo apt install ffmpeg`); the GIF needs nothing extra.

| Script | What it renders |
| --- | --- |
| `state_animation.py` | The quantum-revival loop as a domain-colouring GIF |
| `well_animation.py` | Position and momentum space as side-by-side 3D wells, a quarter period apart |
| `phase_retrieval_animation.py` | The notebook's blind phase retrieval converging (or `--live` to watch) |
| `arrows_demo.py` | Operator splitting told with arrows — no quantum mechanics required |

```bash
uv run python animation_scripts/well_animation.py --state-image horse
```

## Initial-state options (`data.py`)

`get_initial_state(N, L, state_image=..., blur=..., backend=...)` builds a
normalised complex state on the `N × N` grid; `blur` is a Gaussian blur in pixels,
`backend` is `"numpy"`, `"torch"` or `"cupy"`. `state_image` accepts:

### Analytic states

| Name | What it is |
| --- | --- |
| `cat_state` | Two Gaussian packets side by side — a Schrödinger-cat superposition |
| `cosine` | A cosine grating under a broad Gaussian envelope |
| `double_slit` | Two narrow Gaussian slits; evolves into an interference pattern |
| `triple_slit` | Three slits on the vertices of a triangle — ideal for the well animation's blob tracking |
| `single_shifted_slit` | One off-centre slit; sloshes back and forth in the well |
| `orbit` | An off-centre packet with a momentum kick, so it orbits the well |
| `vortex` | A ring with a phase winding — the spinning rainbow |
| `lattice` | A 3 × 3 grid of Gaussian spots |

### Sample images from `skimage.data`

Any of these [`skimage.data`](https://scikit-image.org/docs/stable/api/skimage.data.html)
images, under its scikit-image name — each converted to grayscale, normalised to
[0, 1], inverted where needed so the subject is bright on black, and centred with
margin from the periodic boundary:

`horse`, `astronaut`, `binary_blobs`, `brick`, `camera`,
`cell`, `checkerboard`, `chelsea`, `clock`, `coffee`, `coins`, `colorwheel`,
`grass`, `gravel`, `hubble_deep_field`, `immunohistochemistry`, `logo`,
`microaneurysms`, `moon`, `page`, `retina`, `rocket`, `shepp_logan_phantom`,
`text`.

### Shifted variants

Append `_shifted` to any sample-image name (e.g. `"horse_shifted"`) to place the
subject off-centre, so it oscillates in the well instead of sitting at the bottom.
Anything else raises a `ValueError` listing every available option.

## Development

```bash
uv run pre-commit install          # once per clone: ruff, ty and lockfile checks on commit
uv run pre-commit run --all-files  # check everything up front
uv run pytest                      # the test suite
```
