# Abstract for Phase 3

## Draft Status

This is now an implementation-backed draft abstract for the Phase 3 Paper 2
package. The wording should track the delivered fused-runtime, planner-
calibration, correctness-evidence, and performance-evidence surfaces directly.
Final Phase 3 publication packaging still requires further tightening before
submission-ready wording is frozen.

Exact density-matrix simulation is a scientifically strong foundation for noisy
variational quantum research, but its computational cost quickly limits the
scale of exact studies. SQUANDER already provides two important ingredients for
addressing this problem: a mature state-vector partitioning and gate-fusion
stack, and a validated exact mixed-state backend integrated into one canonical
noisy workflow in Phase 2. The Phase 3 objective is to connect these two assets
without weakening the exact noisy semantics that make the density-matrix path a
trustworthy reference in the first place.

The current Phase 3 contribution is a noise-aware partitioning and limited-fusion
methods layer for exact mixed-state circuits. The delivered result is not only a
planner representation, but a native noisy mixed-state partitioning workflow in
which ordered gate and noise operations are first-class planner inputs,
partition descriptors preserve exact gate/noise order and parameter routing, and
an executable partitioned density runtime includes at least one real fused
execution mode on eligible substructures. In the current planner-calibration
implementation,
the benchmark-facing planning result is a benchmark-calibrated policy over a
bounded family of auditable `max_partition_qubits` span-budget settings on the
existing noisy planner surface rather than a broad family of separately
implemented noisy planner variants. The intended claim remains deliberately
bounded: fully channel-native fused noisy blocks, broader Phase 4 workflow
growth, density-matrix gradients, and approximate scaling methods remain outside
the baseline Paper 2 scope.

The planned evaluation is centered on two complementary evidence layers. First,
the partitioned runtime must preserve the sequential `NoisyCircuit` baseline
within strict exactness thresholds, with Qiskit Aer retained as the required
external reference on the mandatory 2 to 4 qubit microcases plus the current
4-qubit continuity anchor slice. Second, the
benchmark package must characterize representative noisy mixed-state workloads,
including the frozen Phase 2 noisy XXZ `HEA` continuity cases and structured
noisy `U3` / `CNOT` partitioning families across sparse, periodic, and dense
local-noise placement patterns. Success is defined by exact semantic
preservation plus either measurable runtime or memory benefit on representative
cases or a benchmark-grounded diagnosis of the dominant remaining bottleneck.

Current fused-runtime, planner-calibration, correctness-evidence, and
performance-evidence findings now make that interpretation more
concrete. The baseline implementation exercises a real descriptor-local
unitary-island fused path on representative 8- and 10-qubit structured
workloads and preserves exact semantics, while the planning layer calibrates a
bounded noisy-planner candidate family against the frozen benchmark inventory.
The emitted performance-evidence benchmark package records `34` counted
supported cases total,
including `4` continuity anchors and `30` structured cases, with a
representative review set of `6` primary-seed sparse structured cases. The
current performance interpretation still closes the Phase 3 rule through the
diagnosis branch rather than through a positive speedup or memory-reduction
threshold: all `6` representative review cases remain slower than the
sequential reference, do not reduce peak memory, and point primarily to
supported islands left unfused plus Python-level fused-path overhead.

The correctness-evidence surface now closes the exactness side of that package
on the currently selected `span_budget_q2` planning surface. The emitted
correctness-evidence bundles record
`25` counted supported cases, Qiskit Aer agreement on all `3` mandatory
microcases plus the 4-qubit continuity anchor, and `17` explicit unsupported-
boundary cases kept visible across planner-entry, descriptor-generation, and
runtime-stage layers. The performance-evidence summary-consistency bundle then
carries that boundary evidence forward while closing the benchmark layer through
diagnosis-grounded rather than speedup-grounded completion.

The expected Phase 3 result is a defensible methods milestone between exact
workflow integration and later noisy optimizer and trainability studies. By
making noisy mixed-state circuits native objects of partition planning and
runtime execution while preserving exact semantics, Phase 3 is intended to
provide the main Paper 2 contribution in the density-matrix publication ladder
and the stabilized backend needed for broader Phase 4 workflow science.

## Publication Surface Role

This document is the compact conference-abstract surface for the Phase 3 Paper 2
package.

## Paper 2 Claim Boundary

Main claim:
SQUANDER extends partitioning and limited fusion to exact noisy mixed-state
circuits by making noisy operations first-class planner inputs, preserving exact
gate/noise semantics across partition descriptors and runtime execution, and
validating the resulting partitioned density path on representative noisy
workloads.

Explicit non-claims:
- fully channel-native or superoperator-native fused noisy blocks are follow-on
  work beyond the baseline Paper 2 claim
- broader noisy VQE/VQA workflow growth, density-backend gradients, and
  optimizer-comparison studies remain Phase 4+ work
- approximate scaling methods such as trajectories or MPDO-style approaches are
  outside the current Paper 2 claim
- full `qgd_Circuit` parity is not part of the baseline Paper 2 claim
- universal speedup across all noisy workloads is not required for Paper 2;
  benchmark-grounded limitation reporting is part of the honest claim surface

Supported-path boundary:
The guaranteed Paper 2 path is the canonical noisy mixed-state planner surface
plus the documented exact lowering needed for the frozen Phase 2 continuity
workflow and the structured Phase 3 benchmark families.

No-fallback rule:
No silent substitution of sequential execution is part of the Phase 3 contract
for a benchmark that claims `partitioned_density` behavior.

Exact-regime boundary:
Mandatory internal correctness coverage spans 4, 6, 8, and 10 qubits, with
external micro-validation at 2 to 4 qubits and required performance recording
on representative 8- and 10-qubit structured families.

Evidence-closure rule:
Only mandatory, complete, supported correctness and reproducibility evidence,
plus either measured benefit or benchmark-grounded limitation reporting, closes
the main Paper 2 claim.

Phase positioning:
Paper 2 is the Phase 3 noise-aware partitioning and fusion milestone in the
density-matrix publication ladder.
