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

This Phase 2 work closes that integration gap for a frozen, workflow-first
scope. The delivered integrated-backend slice provides explicit backend
selection, a generated-`HEA` density-execution bridge with ordered fixed local
noise, and exact real-valued Hermitian energy evaluation for the XXZ anchor
family. The scope remains intentionally limited to exact noisy backend
integration and validation. Density-aware partitioning, gate fusion, gradients,
and approximate scaling methods are explicitly deferred to later phases in
order to keep the scientific claim clear and defensible.

The evaluation remains evidence-driven and publication-oriented from the start.
The current implementation-backed evidence now includes exact agreement with
Qiskit Aer on the mandatory 1 to 3 qubit micro-validation matrix, the mandatory
4/6/8/10-qubit workflow-scale exact-regime matrix with 10 fixed parameter
vectors per required size, one bounded 4-qubit optimization trace, and explicit
unsupported-case evidence. Benchmarking continues to emphasize training-relevant
circuits and realistic local noise models, including local depolarizing, phase
damping, and amplitude damping, rather than relying mainly on whole-register toy
noise.

The complete integrated-backend validation bundle is now archived in
`benchmarks/density_matrix/artifacts/phase2_task5/`, with
`task5_story6_publication_bundle.json` as the top-level manifest. That manifest
links the phase-level local-correctness bundle
`story1_local_correctness_bundle.json`, the workflow-scale exact-regime bundle
`story2_workflow_baseline_bundle.json`, the trace-and-anchor bundle
`story3_trace_anchor_bundle.json` together with the canonical raw trace
`story2_trace_4q.json`, the metric-completeness bundle
`story4_metric_completeness_bundle.json`, and the interpretation-guardrail
bundle `story5_interpretation_bundle.json` in one backend-explicit
reproducibility package. An implementation learning that now shapes publication
claims is explicit support-tier classification (`required`, `optional`,
`deferred`, `unsupported`) together with hard-error boundary handling,
metric-completeness checks, and interpretation guardrails that prevent partial,
optional, or unsupported evidence from inflating the main claim.

The Phase 2 contribution is a research-grade exact noisy backend for SQUANDER
that enables reproducible noisy variational experiments and forms the basis of
the first major publication in the density-matrix track. By providing a
validated mixed-state workflow inside an established quantum training framework,
this phase establishes the foundation for later work on density-aware
acceleration, optimizer studies under noise, and trainability analysis of noisy
quantum circuits.
