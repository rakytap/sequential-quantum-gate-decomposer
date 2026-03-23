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

Phases 1-3 implementation and evidence delivery are already ahead of final
venue submission. The ladder below is therefore about publication packaging and
submission order, not about whether the corresponding code and evidence surfaces
exist.

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
| Phase 2 | Major methods paper | High | Exact noisy backend integrated into one canonical SQUANDER workflow |
| Phase 3 | Major methods / systems paper | High | Noise-aware partitioning and fusion for mixed-state circuits |
| Phase 4 | Applications / optimization paper | High | Broader noisy VQE/VQA workflows and optimizer behavior under exact noise |
| Phase 5 | Main thesis science paper | Very high | Trainability, entropy, and barren plateaus under realistic noise |

## Current Readiness Snapshot

- Phase 1 implementation is complete; a paper remains optional.
- Phase 2 implementation and evidence packaging are delivered; the remaining
  work is primarily manuscript narrative, figures, and venue shaping.
- Phase 3 backend and evidence surfaces are delivered; the remaining work is
  manuscript tightening, figure selection, and reviewer-facing positioning
  around bounded planner calibration and diagnosis-grounded performance
  closure.

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
- exact real-valued Hermitian energy evaluation via `Re Tr(H*rho)`,
- generated-`HEA` bridge from the current VQE circuit path into the
  density-matrix backend, with broader circuit-source support deferred,
- validation of one canonical noisy XXZ VQE workflow with explicit supported,
  optional, deferred, and unsupported boundaries.

### Why It Is Strong

- It turns the density-matrix work from an isolated module into an actual
  scientific instrument.
- It is directly aligned with the PhD goal of training under realistic noise.
- It provides a complete story: exact noisy backend, observable evaluation, and
  training-loop usability.

### Required Evidence

- numerical agreement against Qiskit Aer for exact Hermitian energies on both:
  - the mandatory 1 to 3 qubit micro-validation matrix,
  - and the mandatory 4 / 6 / 8 / 10 qubit workflow matrix with 10 fixed
    parameter vectors per required size,
- canonical noisy XXZ VQE workflow running end-to-end at 4 and 6 qubits,
  including at least one bounded optimization trace with runtime and peak-memory
  capture,
- structured support-surface evidence that distinguishes required, optional,
  deferred, and unsupported cases,
- machine-checkable reproducibility manifests linking every required artifact and
  status, semantic-closure rules, and canonical workflow identity, including the
  canonical workflow contract, end-to-end trace bundle, workflow matrix bundle,
  unsupported-boundary bundle, and interpretation-guardrail bundle, while
  preserving traceability to the underlying validation-evidence layers,
- only mandatory, complete, supported evidence may close the main Phase 2
  publication claim,
- and a clear statement of supported gate/noise scope plus explicit deferred
  boundaries.

### Current evidence maturity (after Phase 2 integration deliverables)

- implemented artifacts now include a complete top-level canonical-workflow
  manifest at
  `benchmarks/density_matrix/artifacts/workflow_evidence/workflow_publication_bundle.json`,
- mandatory artifact presence, expected-status checks, workflow-identity checks,
  and semantic-closure checks now pass across all five required workflow
  publication bundle components:
  `workflow_contract_bundle.json`,
  `end_to_end_trace_bundle.json`,
  `matrix_baseline_bundle.json`,
  `unsupported_workflow_bundle.json`, and
  `workflow_interpretation_bundle.json`,
- the workflow publication bundle packages one stable workflow ID and contract
  version, two passed end-to-end required cases at 4 and 6 qubits, 40 passed
  fixed-parameter required cases across 4 / 6 / 8 / 10 qubits, and explicit
  unsupported-workflow boundary evidence,
- the canonical workflow contract and workflow publication bundle reuse the
  validation-evidence layers—local correctness, workflow baseline, trace anchor,
  metric completeness, and interpretation—as referenced underlying evidence
  rather than replacing them,
- the canonical workflow contract now carries explicit threshold metadata,
  deterministic parameter/trace policy metadata, and required unsupported-case
  field inventory that downstream workflow-evidence bundles consume directly
  instead of re-declaring independently,
- required-local-noise mandatory baseline cases currently pass at `100%` for the
  frozen integrated-backend scope,
- documented 10-qubit anchor evidence remains present and is now incorporated
  into the workflow matrix baseline bundle and top-level publication surface,
- claim-closure semantics are now machine-checkable: only mandatory, complete,
  supported evidence closes the main claim, optional whole-register
  depolarizing remains supplemental, and deferred or unsupported evidence
  remains boundary-only,
- remaining paper-preparation work is primarily narrative packaging (figures,
  framing, venue shaping), not missing core correctness evidence for the frozen
  canonical Phase 2 workflow.

### Best Venues

- `Quantum`,
- `Physical Review Research`,
- `Quantum Science and Technology`,
- or a strong quantum software / methods venue.

### Paper 2: Phase 3 Methods / Systems Paper

### Recommended Positioning

Second major paper and the most natural place for the partitioning/fusion work.

### Tentative Titles

- `Noise-Aware Partitioning and Gate Fusion for Exact Mixed-State Quantum Circuit Simulation`
- `Making Circuit Partitioning Native to Noisy Density-Matrix Workflows in SQUANDER`

### Core Question

Can partitioning and fusion techniques be extended so noisy mixed-state circuits
are first-class objects while still yielding a useful, auditable performance
story for exact density-matrix simulation on training-relevant workloads?

### Contribution

- noise-aware partitioning representation for mixed-state circuits,
- fusion/runtime contracts that preserve explicit noise-channel placement and
  ordering,
- an executable partitioned runtime with at least one real fused execution path
  on eligible substructures inside noisy mixed-state circuits,
- bounded benchmark-calibrated planner and cost-model analysis under noisy
  workloads,
- optional AVX-level kernel tuning only when profiler and benchmark evidence
  show that it materially contributes to the Phase 3 runtime result,
- benchmark comparison between sequential density execution and partitioned/fused
  density execution across noise placements and noise densities,
- and, ideally, comparison against at least Qiskit Aer plus one additional
  simulator when feasible.

### Why It Is Strong

- This is where the project contributes a truly new methods result beyond the
  existing state-vector partitioning literature.
- The work directly connects the current `partitioning` subsystem to the
  density-matrix backend at the circuit-model level rather than only at
  partition boundaries.
- It creates a clean methods paper without needing to also claim new optimizer
  science, which stays in Phase 4.

### Required Evidence

- a result stronger than planner-only representation: Paper 2 needs an
  executable partitioned noisy-circuit runtime plus at least one real fused
  execution mode on representative workloads,
- exact agreement with the unfused sequential density baseline,
- runtime and memory measurements on representative noisy circuits, with either
  positive-threshold gains or diagnosis-grounded evidence explaining why the
  current baseline plateaus,
- sensitivity studies over partition size, noise placement/density, and gate
  locality,
- a clear explanation of where the speedups do and do not occur, including
  where the native noisy-circuit partition model still falls short,
- and an explicit statement that fully channel-native fused noisy blocks are a
  possible follow-on branch rather than the minimum publishable Phase 3 claim.

### Claim Boundary For Paper 2

Paper 2 should claim more than a noise-aware planner/runtime representation, but
less than a fully general noisy-block fusion architecture.

Minimum publishable scope:

- noisy mixed-state circuits are first-class partitioning inputs,
- the runtime executes partitioned noisy circuits end to end,
- and at least one real fused execution path is benchmarked on eligible
  substructures.

Not required for the baseline Paper 2 claim:

- fully channel-native fused noisy blocks,
- arbitrary CPTP or superoperator fusion of mixed gate+noise regions,
- or proof that every useful noisy partition already maps to a single fused
  noisy block.

### Prior-Art Pointer For Paper 2

Full literature references are maintained only in `REFERENCES.md`, which is the
planning single source of truth for citations.

For the Phase 3 paper, use the curated Phase 3 shortlist in
`REFERENCES.md` together with the detailed entries in:

- `SQUANDER Foundations`,
- `Quantum Circuit Partitioning And Gate Fusion`,
- `Density-Matrix And Open-System Simulation On Classical Hardware`,
- and `Quantum Software Framework References` when software-positioning context
  is needed.

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
- broader noisy VQE/VQA workflow support beyond the frozen Phase 2 canonical
  contract,
- density-backend gradient and optimizer routing for the supported Phase 4
  surface,
- integration of SQUANDER's optimizer strengths, especially BLS-style methods,
- entropy- and gradient-aware diagnostics during optimization,
- and reproducible noisy VQA workflows on application-relevant tasks.

### Why It Is Strong

- The exact backend removes approximation or shot noise as a confound.
- This paper connects infrastructure directly to training science.
- It aligns tightly with the PhD theme and can produce practical optimization
  guidance.

### Required Evidence

- at least one representative VQE/VQA application family beyond the frozen
  Phase 2 canonical workflow,
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

- When does the Phase 3 noise-aware partitioning baseline stop being enough, and
  when would more invasive channel-native fusion become justified?

The delivered Phase 3 benchmark evidence now makes this decision study
concrete, but it is only worth pursuing as a separate paper if the current
diagnosis package is developed into a benchmark-justified follow-on branch.

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
- explicit classification of mandatory, optional, deferred, and unsupported
  evidence,
- a rule that only mandatory, complete, supported evidence closes the main
  paper claim,
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
- use Phase 3 for the major noise-aware partitioning/fusion methods paper,
- use Phase 4 for broader optimizer and workflow studies,
- and make Phase 5 the main trainability and thesis-result paper.
