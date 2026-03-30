# Task 3 Mini-Spec: Correctness Evidence Package

**Phase planning traceability:** `DETAILED_PLANNING_PHASE_3_1.md` §8, Task 3.

**Layer 4 implementation plans (second vertical slice, Story P31-S06)**

- [`ENGINEERING_TASK_P31_S06_E01_IMPLEMENTATION_PLAN.md`](ENGINEERING_TASK_P31_S06_E01_IMPLEMENTATION_PLAN.md)
- [`ENGINEERING_TASK_P31_S06_E02_IMPLEMENTATION_PLAN.md`](ENGINEERING_TASK_P31_S06_E02_IMPLEMENTATION_PLAN.md)

**Pre-implementation checklist traceability**

| P31-C row | Role in this task |
|-----------|-------------------|
| **P31-C-03** | **Primary closure target:** frozen tolerances and CPTP / matrix checks for “match sequential reference” and optional regression vs Phase 3 path. |
| **P31-C-04** | **Primary closure target:** mandatory correctness case set (IDs, qubit bands, families, counts)—reuse or extend Phase 3 with explicit list. |
| **P31-C-05** | **Closed by** `P31-ADR-011`: external micro-validation is required on all four `phase31_microcase_*` cases plus `phase2_xxz_hea_q4_continuity`, and not required on `q6` continuity or counted 8/10-qubit performance cases. |
| **P31-C-08** | **Closed by** `P31-ADR-013`: existing `correctness_evidence` family is extended with Phase 3.1 fields and the required `channel_invariants` slice. |

Upstream: Tasks 1–2, `P31-ADR-003`, Phase 3 correctness culture.

---

## Scientific outcome

The program can claim **exact agreement** on a **frozen, machine-checkable** slice: internal sequential reference, optional Phase 3 partitioned comparison where both apply, and external reference where agreed. Without this, Phase 3.1 is opinion, not evidence.

---

## Given / When / Then

- **Given** frozen **P31-C-03**, **P31-C-04**, **P31-C-05**, and **P31-C-08**, and a build with Task 2 channel-native mode available for eligible cases.
- **When** the correctness harness runs the mandatory Phase 3.1 correctness suite.
- **Then** every mandatory case passes with documented metrics, and failures are treated as release-blocking for any claim of supported channel-native fusion on that slice.

---

## Assumptions and dependencies

- Sequential `NoisyCircuit` is the internal oracle; Phase 3 partitioned+fused path is a **regression** comparator only where applicable (same semantics family).
- External reference scope is **bounded** by `P31-ADR-011`—not every gate/noise
  combo need appear in Aer if explicitly `unsupported` or out of the frozen
  external slice.
- The mandatory slice must include motif-style cases on the frozen v1 class
  from `P31-ADR-007`, not only primitive gate/noise microcases.
- The counted correctness IDs are frozen by `P31-ADR-009`:
  - `phase31_microcase_1q_u3_local_noise_chain`,
  - `phase31_microcase_2q_cnot_local_noise_pair`,
  - `phase31_microcase_2q_multi_noise_entangler_chain`,
  - `phase31_microcase_2q_dense_same_support_motif`,
  - `phase2_xxz_hea_q4_continuity`,
  - `phase2_xxz_hea_q6_continuity`,
  - plus every counted performance case from `P31-ADR-010`.
- The required Aer subset is the same list **without** `phase2_xxz_hea_q6_continuity`
  and without the counted 8/10-qubit performance cases.

---

## Required behavior

- Emit **correctness_evidence** (or successor per **P31-C-08**) with stable **case IDs** aligned to **P31-C-04**.
- Emit the Phase 3.1 required counted-case metadata from `P31-ADR-013`:
  `claim_surface_id`, `runtime_class`, `representation_primary`,
  `fused_block_support_qbits`, `contains_noise`, and `counted_phase31_case`.
- Report matrix distance / trace / positivity checks per **P31-C-03**.
- Report **representation-level CPTP invariants** appropriate to the chosen
  primary form on the fused-block microcases: e.g. Kraus completeness, Choi
  positivity plus trace-preservation, or PTM structural constraints.
- Include Aer (or agreed successor) rows per **P31-C-05** with version metadata in artifact.
- Emit the required `channel_invariants` slice under `correctness_evidence`.
- Boundary cases visible: unsupported requests, max support edge, empty fusion
  region if relevant, and at least one richer mixed-motif case with multiple
  local noise insertions on the same support.

---

## Unsupported behavior

- Claiming “validated” without a frozen case list (**P31-C-04**).
- Omitting external slice when **P31-C-05** requires it for the claimed surface.
- Silent dropping of failed cases from bundles.

---

## Acceptance evidence

- Regeneratable JSON (or project-standard) manifests committed or produced by CI script; schema version field per **P31-C-08**.
- Table in checklist or planning: Case ID → internal pass → Aer pass → notes.
- Explicit invariant-check rows or metadata fields for the chosen representation
  on the fused-block microcases.
- Version-bumped successor schemas must be used when Phase 3.1 mandatory fields
  extend the old Phase 3 case/package shapes.
- Link from `PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md` closure map to this mini-spec section when C-03–C-05 and C-08 correctness parts are closed.

---

## Affected interfaces

- Benchmark / validation scripts under project conventions (paths TBD).
- Optional: pytest markers or case registries keyed by **P31-C-04** IDs.

---

## Publication relevance

- Short paper / full paper **validation** section cites the same case IDs and thresholds as the bundles (`P31-ADR-005` honesty extends to correctness, not only performance).
