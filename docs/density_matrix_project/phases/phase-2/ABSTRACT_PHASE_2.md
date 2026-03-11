# Abstract for Phase 2

Variational quantum algorithms are among the main candidates for near-term
quantum applications, but their practical study is limited by two coupled
problems: realistic noise significantly changes training behavior, and exact
classical emulation of noisy mixed-state circuits is expensive and often poorly
integrated into existing training frameworks. SQUANDER already provides a
high-performance state-vector workflow and a newly introduced standalone
density-matrix module, but the current mixed-state path is not yet integrated
into the main variational workflow of the framework.

This Phase 2 work targets that integration gap. The goal is to transform the
existing density-matrix functionality from a standalone exact noisy simulator
into a usable backend for noisy variational workflows. The phase focuses on
three core deliverables: backend selection between state-vector and
density-matrix execution, a validated expectation-value path based on
`Tr(H*rho)`, and a documented bridge from existing circuit and gate
representations into the density-matrix backend. The scope is intentionally
limited to exact noisy backend integration and validation. Density-aware
partitioning, gate fusion, gradients, and approximate scaling methods are
explicitly deferred to later phases in order to keep the scientific claim clear
and defensible.

The planned evaluation is evidence-driven and publication-oriented from the
start. Validation will be centered on exact agreement with trusted mixed-state
references, primarily Qiskit Aer, for noisy observables and end-to-end workflow
behavior. Benchmarking will emphasize training-relevant circuits and realistic
local noise models, including local depolarizing, phase damping, and amplitude
damping, rather than relying mainly on whole-register toy noise. Success will be
defined not only by integration completeness but also by the ability to execute
at least one representative exact noisy variational workflow in the documented
exact regime.

The expected contribution of Phase 2 is a research-grade exact noisy backend for
SQUANDER that enables reproducible noisy variational experiments and forms the
basis of the first major publication in the density-matrix track. By providing a
validated mixed-state workflow inside an established quantum training framework,
this phase establishes the foundation for later work on density-aware
acceleration, optimizer studies under noise, and trainability analysis of noisy
quantum circuits.
