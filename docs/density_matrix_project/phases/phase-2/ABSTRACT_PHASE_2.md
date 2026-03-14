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
Qiskit Aer on the mandatory 1 to 3 qubit micro-validation matrix, required
end-to-end workflow cases at 4 and 6 qubits, the mandatory 4/6/8/10-qubit
workflow-scale exact-regime matrix with 10 fixed parameter vectors per required
size, one bounded 4-qubit optimization trace driven by a deterministic
parameter policy, and explicit unsupported-case evidence. Benchmarking
continues to emphasize training-relevant circuits and realistic local noise
models, including local depolarizing, phase damping, and amplitude damping,
rather than relying mainly on whole-register toy noise.

The complete workflow-facing publication bundle is now archived in
`benchmarks/density_matrix/artifacts/phase2_task6/`, with
`task6_story6_publication_bundle.json` as the top-level manifest. That manifest
links the canonical workflow contract
`story1_canonical_workflow_contract.json`, the end-to-end plus trace bundle
`story2_end_to_end_trace_bundle.json`, the workflow matrix bundle
`story3_matrix_baseline_bundle.json`, the unsupported-workflow bundle
`story4_unsupported_workflow_bundle.json`, and the interpretation-guardrail
bundle `story5_interpretation_bundle.json` in one backend-explicit
reproducibility package while preserving traceability to the underlying Task 5
validation layers. An implementation learning that now shapes publication
claims is explicit support-tier classification (`required`, `optional`,
`deferred`, `unsupported`) together with hard-error boundary handling,
metric-completeness checks, and interpretation guardrails that prevent partial,
optional, deferred, or unsupported evidence from inflating the main claim. Only
mandatory, complete, supported evidence closes the main Phase 2 result, and the
top-level publication bundle now verifies semantic closure across those lower
evidence layers rather than only artifact presence and shared workflow
identity.

The Phase 2 contribution is a research-grade exact noisy backend slice for
SQUANDER that enables one reproducible canonical noisy XXZ VQE workflow and
forms the basis of the first major publication in the density-matrix track. By
providing a validated mixed-state workflow inside an established quantum
training framework, this phase establishes the foundation for later work on
density-aware acceleration, optimizer studies under noise, and trainability
analysis of noisy quantum circuits.

## Publication Surface Role

This document is the compact conference-abstract surface for the Phase 2 Paper 1
package.

## Paper 1 Claim Boundary

Main claim:
SQUANDER's exact density-matrix backend is integrated into one canonical noisy
XXZ VQE workflow through explicit backend selection, exact Hermitian-energy
evaluation via `Re Tr(H*rho)`, a generated-`HEA` bridge, realistic local-noise
support, and a publication-grade validation package.

Explicit non-claims:
- density-aware partitioning and fusion are future work for Phase 3, not current
  Paper 1 results
- density-matrix gradients and approximate scaling are future work beyond the
  current Paper 1 claim
- broad noisy-VQA workflow generality beyond the canonical supported path is not
  a current Paper 1 claim
- broad manual circuit parity or full `qgd_Circuit` parity is not a current
  Paper 1 claim
- optimizer-comparison studies and trainability analysis belong to later phases
  rather than to the delivered Phase 2 result

Supported-path boundary:
The guaranteed Paper 1 path is the generated-`HEA` VQE-facing density route
rather than broad standalone `NoisyCircuit` capability or full `qgd_Circuit`
parity.

No-fallback rule:
No implicit `auto` mode or silent fallback is part of the Phase 2 contract.

Exact-regime boundary:
Full end-to-end workflow execution is required at 4 and 6 qubits,
benchmark-ready fixed-parameter evaluation is required at 8 and 10 qubits, and
the documented 10-qubit case is the acceptance anchor for the current exact
regime.

Evidence-closure rule:
Only mandatory, complete, supported evidence closes the main Paper 1 claim.

Phase positioning:
Paper 1 is the Phase 2 exact noisy backend integration milestone in the
density-matrix publication ladder.
