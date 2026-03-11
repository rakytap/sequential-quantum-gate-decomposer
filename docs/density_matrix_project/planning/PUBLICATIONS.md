# Publication Strategy

This document maps the density-matrix and partitioning work into a realistic,
publishable PhD output plan.

Primary thesis theme:

> Scalable methods for training quantum circuits under realistic noise models,
> combining advanced emulation frameworks with noise-aware optimization
> strategies.

That theme should remain visible in every major publication, even when a paper
is primarily about simulation or performance.

## Executive View

The codebase is already strong enough to support publishable work, but the
highest-confidence publication path is not "one equally strong paper per
implementation phase." The more realistic structure is:

- a smaller Phase 1 engineering / validation paper or workshop contribution,
- a first major paper at Phase 2,
- a strong methods / systems paper at Phase 3,
- an applications / optimizer paper at Phase 4,
- and the main trainability paper at Phase 5.

## Publication Principles

To maximize scientific credibility, every paper should satisfy as many of the
following as possible:

- compare against at least one strong external baseline, preferably Qiskit Aer
  and one additional simulator or framework when feasible,
- use realistic local noise models rather than only whole-register toy channels,
- separate exactness claims from approximation claims,
- include reproducible benchmark definitions and configuration logging,
- evaluate on workloads that matter for training, not only on synthetic
  simulation kernels,
- and open-source the implementation in a way that reviewers can inspect.

## Summary Table

| Phase | Publication Type | Confidence | Core Scientific Value |
|---|---|---|---|
| Phase 1 | Software / validation note, workshop, or methods short paper | Medium | Exact mixed-state backend and reference validation |
| Phase 2 | Major methods paper | High | Exact noisy training backend integrated into SQUANDER workflows |
| Phase 3 | Major methods / systems paper | High | Density-aware partitioning, fusion, and performance acceleration |
| Phase 4 | Applications / optimization paper | High | Optimizer behavior and noisy VQA workflows under exact noise |
| Phase 5 | Main thesis science paper | Very high | Trainability, entropy, and barren plateaus under realistic noise |

## Recommended Publication Ladder

### Paper 0: Phase 1 Foundation Note

### Recommended Positioning

Optional but useful.

This is most suitable as:

- a workshop paper,
- a software note,
- an engineering short paper,
- or an arXiv preprint that establishes the exact mixed-state baseline.

### Tentative Titles

- `An Exact Mixed-State Backend for SQUANDER`
- `Validation and Benchmarking of a Density-Matrix Quantum Circuit Module in SQUANDER`

### Core Question

Can SQUANDER reproduce exact noisy mixed-state dynamics with a validated density
matrix backend and competitive reference-level behavior on small-to-medium noisy
circuits?

### Contribution

- exact density-matrix backend,
- gate and noise integration into one execution path,
- correctness validation against Qiskit Aer,
- and benchmark tooling for mixed-state simulation.

### Why It Is Publishable

- The implementation is already real and testable.
- Exact mixed-state support is a meaningful extension of a state-vector-first
  platform.
- The scientific claim is modest and therefore easy to defend.

### Main Risk

By itself, this is more of an enabling contribution than a high-impact methods
paper unless the validation and benchmarking are extended substantially.

### Best Venues

- IEEE QCE workshops or poster track,
- software-oriented venues,
- arXiv preprint,
- or a short systems / implementation paper.

### Paper 1: Phase 2 Integration Paper

### Recommended Positioning

First major standalone paper.

### Tentative Titles

- `Exact Noisy Variational Quantum Circuit Emulation in SQUANDER`
- `Integrating Density-Matrix Simulation into a Large-Scale Quantum Training Framework`

### Core Question

Can an exact noisy density-matrix backend be integrated into a practical VQA /
training workflow without sacrificing correctness or reproducibility?

### Contribution

- backend selection between state-vector and density-matrix execution,
- expectation-value path `Tr(H*rho)`,
- bridge from existing circuit / gate sequences into the density-matrix backend,
- validation of noisy training workflows on representative circuits.

### Why It Is Strong

- It turns the density-matrix work from an isolated module into an actual
  scientific instrument.
- It is directly aligned with the PhD goal of training under realistic noise.
- It provides a complete story: exact noisy backend, observable evaluation, and
  training-loop usability.

### Required Evidence

- numerical agreement against Qiskit Aer for energies / observables,
- noisy VQE or analogous training workflow running end-to-end,
- representative qubit counts around the exact regime the backend can support,
- runtime and memory characterization,
- and clear statement of supported gate/noise scope.

### Best Venues

- `Quantum`,
- `Physical Review Research`,
- `Quantum Science and Technology`,
- or a strong quantum software / methods venue.

### Paper 2: Phase 3 Methods / Systems Paper

### Recommended Positioning

Second major paper and the most natural place for the partitioning/fusion work.

### Tentative Titles

- `Density-Aware Partitioning and Unitary-Island Fusion for Exact Noisy Quantum Circuit Simulation`
- `Extending Gate Fusion from State Vectors to Density Matrices in SQUANDER`

### Core Question

Can partitioning and fusion techniques materially accelerate exact
density-matrix simulation on training-relevant noisy circuits?

### Contribution

- density-aware partitioning objective,
- adapter-based unitary-island fusion with noise barriers,
- density-matrix-specific kernel and cost-model analysis,
- benchmark comparison between sequential density execution and partitioned/fused
  density execution,
- and, ideally, comparison against at least Qiskit Aer plus one additional
  simulator when feasible.

### Why It Is Strong

- This is where the project contributes a truly new methods result beyond the
  existing state-vector partitioning literature.
- The work directly connects the current `partitioning` subsystem to the new
  density-matrix backend.
- It creates a clean methods paper without needing to also claim new optimizer
  science.

### Required Evidence

- exact agreement with the unfused sequential density baseline,
- runtime and memory improvements on representative noisy circuits,
- sensitivity studies over partition size, noise density, and gate locality,
- and a clear explanation of where the speedups do and do not occur.

### Best Venues

- `Quantum`,
- `Journal of Computational Physics`,
- `IEEE QCE`,
- `IEEE CLUSTER`,
- or another systems / computational physics venue.

### Paper 3: Phase 4 Noisy Optimization Paper

### Recommended Positioning

Primary applications paper.

### Tentative Titles

- `Optimizer Behavior in Exact Noisy Variational Quantum Training`
- `Noise-Aware Optimization Strategies for Variational Quantum Circuits Under Exact Mixed-State Emulation`

### Core Question

How do different optimizers behave under realistic local noise when training
variational quantum circuits with exact noisy emulation rather than shot-based or
approximate simulation?

### Contribution

- optimizer comparisons under exact noisy conditions,
- integration of SQUANDER's optimizer strengths, especially BLS-style methods,
- entropy- and gradient-aware diagnostics during optimization,
- and reproducible noisy VQA workflows on application-relevant tasks.

### Why It Is Strong

- The exact backend removes approximation or shot noise as a confound.
- This paper connects infrastructure directly to training science.
- It aligns tightly with the PhD theme and can produce practical optimization
  guidance.

### Required Evidence

- at least one representative VQA application family,
- optimizer comparison under matched noise conditions,
- metrics such as convergence quality, runtime, gradient norms, and entropy,
- and ideally several noise regimes and ansatz families.

### Best Venues

- `Quantum`,
- `PRX Quantum`,
- `Physical Review A`,
- or `Physical Review Research`.

### Paper 4: Phase 5 Trainability Paper

### Recommended Positioning

Main thesis science paper and likely the highest-impact publication.

### Tentative Titles

- `Trainability of Variational Quantum Circuits Under Realistic Local Noise: An Exact Density-Matrix Study`
- `Entropy, Gradient Collapse, and Barren Plateaus in Noisy Variational Quantum Circuits`

### Core Question

How do realistic noise models affect trainability, entropy growth, expressivity,
and barren-plateau behavior in variational quantum circuits?

### Contribution

- systematic trainability analysis under exact noisy simulation,
- unital versus non-unital noise comparisons,
- depth/locality/noise phase diagrams,
- gradient-variance and entropy datasets,
- and practical design guidelines for noise-aware training.

### Why It Is Strong

- This is the most scientifically central question in the PhD plan.
- It depends on earlier implementation work but yields an independent
  science-first result.
- Exact noisy simulation makes the conclusions much cleaner than studies relying
  only on approximate or sampled backends.

### Required Evidence

- reproducible large experiment matrix across ansatze, depths, and noise types,
- careful statistical treatment,
- comparison to existing barren-plateau and trainability theory where possible,
- and publication-grade figures plus interpretation.

### Best Venues

- `Quantum`,
- `PRX Quantum`,
- `Nature Communications`,
- or another high-quality theory/empirical quantum venue.

## Optional Side Papers

These are useful, but they should not displace the primary thesis narrative.

### Side Paper A: Superoperator Or Channel-Native Fusion Decision Study

Question:

- When does barrier-based unitary-island fusion stop being enough, and when would
  channel-native fusion become justified?

This is only worth doing after Phase 3 benchmark evidence suggests the barrier
model is the dominant remaining limitation.

### Side Paper B: Exact Density Matrix Versus Trajectories / MPDO Cross-Over

Question:

- For training-relevant noisy workloads, where is the scientific and practical
  crossover between exact dense density matrices and approximate scaling methods?

This is valuable later, but only after the exact backend is a trustworthy
reference point.

### Side Paper C: Noise Calibration / Reference Simulator Benchmarking

Question:

- How stable and trustworthy are different reference simulators and noise-model
  implementations for small-to-medium exact noisy workloads?

Useful as a benchmark note, but not a substitute for the main results.

## If Only Three Strong Papers Are Realistic

If the project must optimize for a smaller number of stronger publications, the
recommended priority is:

1. `Phase 2` integration paper,
2. `Phase 3` partitioning/fusion paper,
3. `Phase 5` trainability paper.

In that compressed strategy:

- Phase 1 becomes a preprint or workshop contribution,
- Phase 4 results can be folded into the Phase 5 paper if needed.

## Common Evidence Package Across Papers

The following evidence package should be treated as the default standard for the
project:

- external baseline comparison,
- exact agreement checks where applicable,
- representative noisy circuit families,
- realistic local noise models,
- reproducible run configuration and logging,
- open-source code or branch availability,
- and clear statements of supported scale and limitations.

## What Should Not Be The Main Early Paper

The following directions are scientifically interesting, but poor choices for the
first major paper:

- noisy circuit re-synthesis,
- highly invasive channel-native fusion without benchmark evidence,
- approximate large-scale methods before the exact backend is fully integrated,
- or papers based mainly on global depolarizing toy models.

These directions can appear later or as secondary branches, but they should not
replace the exact-noisy-training narrative that best matches the PhD theme.

## Bottom Line

Yes, publishable results are realistic throughout the project.

The best publication strategy is:

- keep the exact density-matrix backend as the scientific anchor,
- use Phase 2 for the first major integration paper,
- use Phase 3 for the major partitioning/fusion methods paper,
- use Phase 4 for optimizer and workflow studies,
- and make Phase 5 the main trainability and thesis-result paper.
