# Revo 2

A CUDA-first cellular evolutionary search engine, demonstrated by evolving small programs that generate images.

Rust runs the experiment and the local playground. CUDA initializes genomes, selects local parents, breeds and mutates programs, evaluates image error, replaces individuals, reduces statistics, and renders previews. Genomes and fitness stay in device memory between generations. A parallel Rust CPU reference implements the same rules.

This is the first working image-program implementation. The execution interface separates the UI from the backends, but its genome and objective are currently specialized to this demo; a general third-party problem/plugin API is future work.

## Start here

With Rust and the CUDA runtime compiler installed (see installation below):

```powershell
cargo run --release
```

On Windows, the launcher also supports optional project-local Rust and NVRTC installations under the ignored `.tools/` directory:

```powershell
.\scripts\dev.ps1
```

Open **http://127.0.0.1:8787**. The initial population is paused; click **Run evolution**. Select a preset or upload a PNG/JPEG, inspect cells in the population map, and export the best image, program JSON, or GLSL. The map supports arrow keys. Applying settings or uploading an image resets the experiment.

For ordinary Cargo commands using the local tools:

```powershell
. .\scripts\env.ps1
cargo run --release
cargo run --release -- --backend cpu
```

The server binds to loopback only. All HTML, CSS, scripts, and images are served locally; the playground has no external assets or services. Ctrl+C in the server terminal stops it.

## Installation on another machine

- Install stable Rust and a native linker. On Windows, use Visual Studio C++ Build Tools with a Windows SDK.
- For the default CUDA backend, install an NVIDIA driver and CUDA **12.8** (or a compatible 12.x NVRTC runtime). Put the NVRTC shared-library directory on `PATH` on Windows, or `LD_LIBRARY_PATH` on Linux. Kernels are compiled by NVRTC at startup; `nvcc` and the CUDA toolkit are not required at Rust build time.
- `cargo run --release -- --backend cpu` explicitly selects the CPU reference. `cargo run --release --no-default-features -- --backend cpu` also removes the CUDA dependency. There is no silent fallback from CUDA to CPU.

Only Windows with an RTX 4070 Laptop GPU has been exercised so far. Other platforms are unverified.

## Reproducible headless experiments

```powershell
cargo run --release -- run --preset rings --generations 10000 --output runs/rings
cargo run --release -- run --target path/to/image.png --population 64 --nodes 32 --seed 42 --output runs/custom
cargo run --release -- run --config examples/default.json --preset orbit --generations 2000 --output runs/orbit
```

`--population N` creates an N×N grid. The JSON configuration supports rectangular grids and the complete set of evolution parameters. `--config` conflicts with explicit population/node/seed/sample overrides. `--checkpoint` controls progress observation frequency; it is not a resumable population checkpoint.

Outputs are `target.png`, `best.png`, `program.json`, `program.glsl`, `config.json`, `run.json`, and `history.csv`. Existing nonempty output directories are rejected. `run.json` records backend, configuration, generation, timings, active nodes, and a separate 128×128 validation MSE. To repeat a run, use its `config.json` and `target.png` with the same backend. CPU/CUDA floating-point differences can eventually change the search trajectory.

Render an exported program without initializing CUDA:

```powershell
cargo run --release -- render docs/examples/rings-program.json --size 1024 --output rings.png
```

## What evolves

Each population cell contains a complete RGB image program, not one image pixel. A genome is an acyclic graph of 3–64 instructions. Each instruction has an opcode, two connections to inputs or earlier instructions, and a bounded floating-point parameter. The last three instructions produce red, green, and blue.

Inputs are x, y, radius, sin(πx), sin(πy), and 1. Coordinates use pixel centers in [-1, 1], with y increasing downward. Operations are constants, arithmetic, protected division, sine, cosine, absolute value, min, max, tanh, and a threshold. Intermediate values clamp to [-8, 8]; RGB clamps to [0, 1]. Protected division returns its numerator for denominators smaller than 0.01 in magnitude. The threshold uses `a > b + 0.0001` to reduce sensitivity to near-equal CPU/CUDA transcendental results.

Evolution is synchronous on a toroidal grid:

1. Independently choose two parents using binary tournaments within center/left/right/up/down.
2. Optionally perform positional one-point crossover; backward connections remain valid.
3. Independently at each node, mutate one of opcode, connection A, connection B, or parameter with the configured mutation probability.
4. Evaluate the offspring on the same fixed training sample grid as its incumbent.
5. Keep the offspring if its score is no worse; allow neutral drift on ties. Otherwise retain the incumbent.

Score is mean squared RGB error plus `complexity_weight × active_nodes / total_nodes`. Lower is better. Inactive instructions do not contribute to complexity. Each cell's penalized training score is nonincreasing, although raw validation error need not be. The deterministic integer RNG is keyed by seed, cell, and generation and is independent of thread scheduling. Operations are specialized by genome length once per experiment, not compiled separately for each candidate.

CUDA evaluation uses one block of 64 threads per candidate. Threads share a program and evaluate different samples, then reduce error. Four kernels handle breeding, evaluation, replacement, and occasional statistics; only observations transfer fitness maps, selected genomes, and preview pixels back to the host. Evolution uses preallocated double buffers. Kernel launches remain host-driven.

## Reading the playground

- Training error includes the small complexity penalty. The footer reports unpenalized validation MSE at 64×64, using the best program evaluated by the CPU reference.
- The population map uses a relative current-generation error scale. Uniform color means scores have converged; it does not imply the target has been solved. White marks the inspected cell.
- The chart shows best and population-mean penalized training error on a logarithmic scale. Long histories are decimated.
- Search speed counts time inside evolution batches, including synchronization. UI rendering, setup, image encoding, and polling are excluded.
- Inactive graph instructions are muted. Exports always contain the best program, even when inspecting another cell. GLSL exports a `vec3 revo(vec2 p)` function, with p in [-1,1].

Uploads are center-cropped and resized to 512×512, with transparency composited onto a dark background. The default training grid is only 24×24: high-frequency programs can fit those samples while producing artifacts between them. The separate validation score and larger exported render make this visible. Increase `sample_resolution` in a JSON configuration for more demanding targets. This is a small-program approximation demo, not a photorealistic image model.

## Validation

```powershell
cargo test
cargo test --no-default-features
cargo test --test evolution -- --ignored
cargo clippy --all-targets -- -D warnings
cargo fmt --all -- --check
```

GPU tests require the NVIDIA driver and NVRTC. They compare seeded CPU/CUDA genomes, fitness, and rendered pixels, exercise 3- and 64-node programs and small rectangular grids, and check reproducibility and elitist replacement. CPU tests also cover wrapping, valid connections, bounded numeric behavior, image normalization, reset failure handling, and program serialization.

An initial single-run timing on the RTX 4070 Laptop GPU, 32×32 population, 24 nodes, 24×24 samples, seed 7, and the Orbit target:

| Backend | 200 generations | Final best score |
| --- | ---: | ---: |
| Rayon CPU reference | 3.398 s | 0.01672918 |
| CUDA | 0.107 s | 0.01672919 |

These are exploratory evolution-loop measurements, including progress observations and excluding setup, JIT compilation, rendering, and export. The CPU reference allocates offspring on the host and is not a tuned CPU competitor. They are not a general speedup guarantee. Raw run records are available locally in ignored `runs/benchmark-cpu` and `runs/benchmark-cuda`.

The checked-in [rings example](docs/examples/rings-program.json) came from a live CUDA run using default settings and 144,200 generations. Its images show the target and approximation below; some high-frequency artifacts remain.

| Target | Evolved program |
| --- | --- |
| ![Rings target](docs/examples/rings-target.png) | ![Evolved rings](docs/examples/rings-best.png) |

## Source map and next steps

- `src/program.rs`: genome, CPU interpreter, active-node analysis, GLSL export.
- `src/evolution/`: configuration, common execution interface, CPU and CUDA implementations.
- `kernels/image.cu`: GPU implementation and matching RNG/ABI.
- `src/target.rs`: targets, sample grids, image loading, CPU rendering.
- `src/server.rs` and `web/`: worker-owned experiment and local browser UI.
- `src/export.rs`: reproducible output bundle.

The next useful experiments are maintaining diversity across neighborhoods, comparing cellular/global parent selection, multi-resolution evaluation with incumbent re-evaluation whenever the objective changes, and extracting a reusable CUDA problem interface after adding a second genome/objective. Full-population save/resume and CUDA/graphics interop are not implemented yet.
