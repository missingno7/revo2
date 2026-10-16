# Adding a problem to Revo

`revo-core` is an independent crate. It depends on `anyhow`, `serde`, `rayon`, and optionally `cudarc`; it has no image, packing, HTTP, or GUI dependencies. The root `revo2` package supplies the demos, rendering, UI, and file formats.

```toml
[dependencies]
revo-core = { path = "path/to/revo2/crates/revo-core" }
```

Disable default features for CPU-only applications. The complete [standalone sphere test](../crates/revo-core/tests/problem.rs) demonstrates a third problem implemented outside the engine, with both CPU and CUDA callbacks.

## Host contract

Implement `revo_core::Problem`. Your fixed-length genome is a slice of `u32` words; encode floats with `to_bits`/`from_bits`, or use words for integer, permutation, or packed representations. Return a length of 1–4096 words. This is a storage interface, not a prescribed genome representation.

| Hook | Responsibility |
| --- | --- |
| `name` | Diagnostic name |
| `genome_len` | Fixed number of words per individual for the experiment |
| `validate` | Validate immutable problem data before any allocation or launch |
| `initialize` | Write a valid genome into every word of the supplied output |
| `recombine` | Write a complete valid child from selected parents; respect the crossover flag |
| `mutate` | Apply problem-specific variation while preserving validity |
| `score` | Return the objective value; lower is better |
| `cuda` | Optionally supply device source, immutable data words, and evaluation lane count |

The engine owns toroidal L5 neighborhoods, binary tournaments, crossover decisions, RNG seeding, synchronous generations, preallocated current/next buffers, fitness evaluation scheduling, elitist replacement, neutral drift, and statistics. No engine switch selects a problem type.

Construct a problem, place it in an `Arc`, then run:

```rust,ignore
let mut engine = revo_core::create("cuda", &config, std::sync::Arc::new(problem))?;
engine.step(100)?;
let stats = engine.stats()?;
let best_words = engine.genome(stats.best_index)?;
```

CPU and CUDA callbacks should consume random numbers in the same order if cross-backend reproducibility matters. `rng::Rng` exposes `next_u32`, `unit` (uniform [0,1)), and `index`. On-device `next(rng)` and `unit(rng)` use the same integer generator. Streams are keyed by evolution seed, cell, and generation. The core supports negative and greater-than-two scores; nonfinite scores are replaced by `f32::MAX` rather than accidentally winning selection.

## CUDA contract

Return a `CudaProblem` constructed with `unsafe { CudaProblem::new(source, data, block_size) }`. Its constructor is unsafe because the engine launches your trusted device code. You must ensure bounds, genome validity, uniform barrier usage, and race-free behavior. Runtime validation of sizes does not make arbitrary device source safe.

The shared compiler assembles:

1. `GENOME_WORDS` and `EVALUATION_LANES` definitions.
2. The core RNG helpers (`next`, `unit`, `bound`).
3. Your source and its problem-specific constants/types/functions.
4. The common initialization, breeding, evaluation, replacement, and statistics kernels.

Your source must define these device functions:

```cpp
__device__ void problem_initialize(
    unsigned *genome, const unsigned *data, unsigned &rng);

__device__ void problem_recombine(
    const unsigned *first, const unsigned *second, unsigned *child,
    const unsigned *data, unsigned &rng, bool crossover);

__device__ void problem_mutate(
    unsigned *genome, const unsigned *data, unsigned &rng,
    float mutation_rate, float mutation_scale);

__device__ float problem_score(
    const unsigned *genome, const unsigned *data,
    unsigned lane, unsigned lanes);
```

Initialization and variation run once per cell. They cannot use block-wide barriers, and they must write only their own genome. Inputs are read-only and may be shared between cells. Every child word must be initialized before scoring, and variation must preserve all indexing constraints used by evaluation.

Fitness uses one block per candidate. The core first copies that candidate into shared memory and synchronizes. Every lane then calls `problem_score`. The core sums the returned contributions across lanes, so normalization and regularization belong to your callback. Return an additive penalty from only one lane. Block size must be a power of two from 1 through 256. A sequential decoder can choose one lane; a sample-wise objective can use many lanes. Cooperative evaluators may use barriers if all lanes reach them uniformly. Source/destination buffers are never updated in place across generations.

The data buffer is immutable and uploaded once. Empty data gets one zero word so the device pointer is valid, but callbacks should access only their declared logical data. Values remain on the GPU between generations. There is currently no variable-length per-individual allocation, custom replacement policy, dynamic objective update, or asynchronous evolution mode.

## Two different encodings

- **Image programs:** four words per graph instruction (opcode, two backward connections, float bits). Shared data contains sample count, complexity weight, and pixel samples. The scorer uses 64 lanes per candidate to divide sample work. Graph-compatible positional crossover preserves backward connections.
- **Rectangle packing:** a permutation, rotation bits indexed by rectangle ID, and an evolving placement-width gene. Immutable data contains width bounds, area, and rectangle dimensions. Order-preserving crossover and swap/rotation/width mutations preserve feasibility. A one-lane skyline decoder scores the enclosing area.

Rendering is outside `Engine`. The app decodes observed genomes for inspectors and exports. The image demo has its own optional CUDA preview renderer; packing previews use the decoded geometry on the host. Neither adds rendering methods or problem-specific types to the core API.

To add a demo to the existing playground, extend the root application's `Demo` adapter and UI presentation. Applications using only `revo-core` do not need that adapter or its `ProblemKind` enum.

## Validation expectations

Test initialization and variation for genome validity, not just score improvement. Compare CPU/CUDA starting genomes and fitness. Use a deterministic small instance with a known answer, then check generation stepping, replacement, and repeatability. Floating-point objectives can diverge over long runs even with identical random streams. For packing, independently check permutation completeness, dimensions/rotations, bounds, non-overlap, and area conservation; a good objective value alone is insufficient.
