# Abstract for Phase 2

Variational quantum algorithms are among the main candidates for near-term
quantum applications, but their practical study is limited by two coupled
problems: realistic noise significantly changes training behavior, and exact
classical emulation of noisy mixed-state circuits is expensive and often poorly
integrated into existing training frameworks. SQUANDER already provides a
high-performance state-vector workflow and a newly introduced standalone
density-matrix module, and the current Phase 2 implementation has now delivered
the first narrow integration slice into the main variational workflow of the
framework.

This Phase 2 work targets that integration gap. The goal is to transform the
existing density-matrix functionality from a standalone exact noisy simulator
into a usable backend for noisy variational workflows. The currently implemented
slice already provides explicit backend selection, a generated-`HEA`
density-execution bridge with ordered fixed local noise, and exact real-valued
Hermitian energy evaluation for the XXZ anchor family. The full phase remains
intentionally limited to exact noisy backend integration and validation.
Density-aware partitioning, gate fusion, gradients, and approximate scaling
methods are explicitly deferred to later phases in order to keep the scientific
claim clear and defensible.

The evaluation remains evidence-driven and publication-oriented from the start.
Current implementation-backed evidence already includes exact agreement with
Qiskit Aer on 4- and 6-qubit fixed-parameter XXZ anchor cases plus one bounded
4-qubit optimization trace. Benchmarking continues to emphasize training-
relevant circuits and realistic local noise models, including local
depolarizing, phase damping, and amplitude damping, rather than relying mainly
on whole-register toy noise. The remaining Phase 2 work is now the broader
benchmark floor, 8/10-qubit evidence, runtime and peak-memory characterization,
and the full reproducibility/provenance bundle.

Backend-explicit machine-readable artifacts now exist for the current completed
and unsupported validation slice, but this still represents only the first
reproducibility/provenance layer rather than the full final Phase 2 package.

The expected contribution of Phase 2 is a research-grade exact noisy backend for
SQUANDER that enables reproducible noisy variational experiments and forms the
basis of the first major publication in the density-matrix track. By providing a
validated mixed-state workflow inside an established quantum training framework,
this phase establishes the foundation for later work on density-aware
acceleration, optimizer studies under noise, and trainability analysis of noisy
quantum circuits.
