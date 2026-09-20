# Task 1 Mini-Spec: Fusion Representation and IR Contract

**Phase planning traceability:** `DETAILED_PLANNING_PHASE_3_1.md` §8, Task 1.

**Pre-implementation checklist traceability**

| P31-C row | Role in this task |
|-----------|-------------------|
| **P31-C-01** | **Closed by** `P31-ADR-004`: primary exact representation is `kraus_bundle`; composition, completeness, and Choi-derived positivity checks define the counted claim, with Liouville form allowed only as a non-primary internal optimization. |
| **P31-C-02** | **Input:** support matrix must be consistent with what the representation can express on small supports; narrow eligibility patterns here if the math forces it. |
| **P31-C-08** | **Informs:** any serialized block or descriptor payload shape for fused channels must fit the agreed evidence / runtime handoff story. |

Upstream: `P31-ADR-001`, `P31-ADR-003`, `P31-ADR-004`, closed Phase 3 descriptor and `NoisyCircuit` ordering contracts.

---

## Scientific outcome (what “done” means for research)

Reviewers and future you can answer: **what mathematical object is a “fused noisy block,” how is it exact and CPTP, and how does it relate one-to-one to a stretch of sequential `GateOperation` / `NoiseOperation` semantics?** Ambiguity here invalidates correctness and publication claims.

## Novelty hypothesis

The Phase 3.1 novelty claim is **not** “SQUANDER uses superoperators” or “the
runtime can store channels in a different representation.” The intended claim is
that SQUANDER can perform **bounded exact channel-native fusion of contiguous
mixed gate+noise motifs** inside the partitioned noisy runtime and thereby
surpass the Phase 3 unitary-island fused baseline on a narrow, validation-ready
slice without relaxing exact mixed-state semantics.

---

## Given / When / Then

- **Given** a contiguous eligible pattern of gates and local noise channels on a
  bounded qubit support, with the frozen v1 slice being **1- and 2-qubit
  mixed motifs** built from `U3` / `CNOT` plus local single-qubit depolarizing,
  amplitude-damping, and phase-damping channels on those same qubits, allowing
  multiple successive gates and multiple local noise insertions in one block.
- **When** that pattern is lifted into the Phase 3.1 fusion IR (serialized or runtime handoff as agreed).
- **Then** the block maps to a **unique** CPTP map on that support whose action matches the sequential density evolution of the same operations under the canonical `NoisyCircuit` semantics, and non-eligible patterns are **not** represented as silently equivalent fused blocks.

---

## Assumptions and dependencies

- Sequential `NoisyCircuit` density evolution remains the semantic oracle (`P31-ADR-003`).
- Phase 3 partition ordering and descriptor contracts are not weakened; fusion only **compresses** admitted subsequences.
- Numeric tolerances for “representation matches oracle” tests are frozen under **P31-C-03** (this task may propose defaults for checklist closure).
- `P31-ADR-007` freezes the v1 support default around bounded mixed
  gate+noise motifs on total support of 1 or 2 qubits; this task turns that
  slice into an exact representation contract rather than reopening broader
  gate/noise coverage.
- This task does **not** by itself ship the full partitioned runtime path (Task 2) or benchmark bundles (Tasks 3–4).

---

## Required behavior

- The **primary** counted-claim representation is `kraus_bundle`; alternates are
  explicitly “internal optimization only” if allowed.
- Exact **composition** of two eligible blocks on compatible supports is defined
  by ordered Kraus-operator multiplication.
- **Trace preservation** for the primary form is checked through Kraus
  completeness residuals.
- A **representation-level invariant suite** is defined for the chosen primary
  form. Depending on the representation, that means at least one of:
  Kraus-completeness checks, Choi positivity plus trace-preservation checks, or
  PTM structural constraints that imply the claimed CPTP behavior.
- **Unsupported** fusion shapes are listed: anything outside the primary contract must not be implied as supported.
- The frozen v1 slice keeps the Phase 3 primitive gate/noise families and
  widens **fused eligibility**, not primitive surface breadth: contiguous 1- and
  2-qubit mixed motifs on the same support, including multiple successive gates
  and multiple interleaved local noise insertions in one fused block. Each
  counted Phase 3.1 fused block contains at least one noise operation; pure
  unitary islands remain the Phase 3 fused path.
- Traceability table: eligibility class → mathematical object → sequential reference segment.

---

## Unsupported behavior

- “Any equivalent CPTP form” without a single primary narrative for the baseline claim.
- Fused blocks that reorder or merge operations across semantics that change CPTP outcome relative to sequential reference.
- Silent widening of eligibility beyond the written support matrix.

---

## Acceptance evidence

- Written contract in `ADRs_PHASE_3_1.md` (**P31-ADR-004** Accepted) and/or an
  annex referenced from `DETAILED_PLANNING_PHASE_3_1.md` §7.
- Small-support **algebraic or numerical** checks: fused apply vs sequential reference on at least one hand-derived case per eligibility class (case IDs to align with **P31-C-04** when frozen).
- Representation-level invariant checks on the same microcases:
  - Kraus completeness residuals,
  - derived Choi positivity plus trace-preservation checks under the frozen
    `P31-ADR-008` thresholds.
- Design review note (short) listing rejected representation options and why.

---

## Affected interfaces (planned)

- Partition / runtime handoff types, serialization, or opcodes for fused CPTP blocks (exact paths TBD at implementation).
- Any public “fusion mode” naming is **prepared** for Task 2 / **P31-C-07** but need not be final here.

---

## Publication relevance

- Side Paper A / Phase 3.1 methods text can state **one** clear fusion object and how exactness is checked.
- Claim boundary: no performance or speedup language in Task 1 evidence—representation and exactness only.
