
# Quantum Oscillator Toolbox Talk

This repository contains the code and slides for a talk on building a quantum oscillator to rotate an image with useful python tools such as uv, ruff, ty, pre-commit, and array-api-compat.

## Getting started

This project is managed with [uv](https://docs.astral.sh/uv/), an extremely fast
Python package and project manager from Astral, written in Rust.

Everything needed to reproduce the environment is committed here — `pyproject.toml`
declares the dependencies and `uv.lock` pins the exact resolution, so
`uv sync` gives you the same versions this was developed against.


### 1. Set up the project

First, [Install uv](https://docs.astral.sh/uv/getting-started/installation/), then clone the repository and sync the environment:

```bash
git clone https://github.com/bryceshirley/quantum-oscillator.git
cd toolbox-talk
uv sync
```

`uv sync` reads `uv.lock`, creates `.venv/`, and installs the exact pinned
versions — including the correct Python, downloading it if you don't have it.
You never need to activate the environment yourself.

### 3. Run things

Prefix commands with `uv run` and they execute inside the project environment:

```bash
uv run python toolbox_solver_benchmarks.py   # solver comparison
```

Benchmark output lands in `results/`.

Video encoding requires **ffmpeg**, which is not a Python package and so is not
installed by `uv sync`:

```bash
brew install ffmpeg          # macOS
sudo apt install ffmpeg      # Debian/Ubuntu
```

Rendered videos are written to the repository root as `.mp4`.

```bash
uv run python toolbox_well_animation.py      # 3D well animation
uv run python toolbox_state_animation.py     # state evolution
```

### 4. Enable the pre-commit hooks

If you intend to commit, install the git hooks once per clone:

```bash
uv run pre-commit install
```

Commits are then checked with `ruff` (lint and format) and `ty` (types), and
`uv.lock` is kept in step with `pyproject.toml`. To check everything up front:

```bash
uv run pre-commit run --all-files
```

### Common commands

| Command | What it does |
| --- | --- |
| `uv sync` | Install the locked environment |
| `uv add <package>` | Add a dependency and update the lockfile |
| `uv add --dev <package>` | Add a development-only dependency |
| `uv run <command>` | Run a command inside the project environment |
| `uv lock --upgrade` | Re-resolve dependencies to their latest allowed versions |