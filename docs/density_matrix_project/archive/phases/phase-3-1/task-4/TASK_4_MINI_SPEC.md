# Task 4 Mini-Spec: Performance and Diagnosis Evidence

**Phase planning traceability:** `DETAILED_PLANNING_PHASE_3_1.md` §8, Task 4.

**Layer 4 implementation plans (third vertical slice, Story P31-S09)**

- [`ENGINEERING_TASK_P31_S09_E01_IMPLEMENTATION_PLAN.md`](ENGINEERING_TASK_P31_S09_E01_IMPLEMENTATION_PLAN.md)

**Pre-implementation checklist traceability**

| P31-C row | Role in this task |
|-----------|-------------------|
| **P31-C-06** | **Primary closure target:** performance case set (structured families stressing noise density / locality), and explicit use of **baseline trio**: sequential, Phase 3 partitioned+fused, and the counted whole-workload Phase 3.1 hybrid path `phase31_channel_native_hybrid`. |
| **P31-C-08** | **Closed by** `P31-ADR-013`: existing `performance_evidence` family is extended with Phase 3.1 fields and the required `break_even_table` slice. |
| **P31-C-09** | **v1:** mandatory performance rows assume **scalar-only** builds; manifests record that policy (e.g. `build_flavor: scalar`). **Later branch:** if Task 6 adds TBB/SIMD rows, metadata per amended **P31-C-09**. |

Upstream: Task 2–3 (correct path attribution and mandatory slice green), `P31-ADR-005`.

---

## Scientific outcome

The phase answers Side Paper A’s performance question with **comparative, honest**
evidence: speedups, ties, or regressions versus **both** sequential density and
Phase 3 fused baselines, plus **diagnosis** when the kernel is not the
bottleneck. For the counted whole-workload slice, that evidence is carried by
the explicit hybrid runtime `phase31_channel_native_hybrid`, not by the strict
microcase-only path. That is the scientifically meaningful result, and even a
negative result remains a valid Phase 3.1 outcome per
`DETAILED_PLANNING_PHASE_3_1.md` §6.

For the current publication stance, the same evidence must also support a
**positive-methods claim** on the frozen v1 slice if such a claim is made:
the benchmark package must show where the richer mixed-motif class is actually
worth the added runtime complexity.

---

## Given / When / Then

- **Given** frozen **P31-C-06**, **P31-C-08**, and **P31-C-09** (v1:
  scalar-only mandatory bundles), correctness green on the slice that
  performance cases are allowed to claim, and the explicit hybrid runtime from
  Task 2.
- **When** the performance harness runs the frozen Phase 3.1 performance suite.
- **Then** each counted whole-workload case records comparable timings (and
  agreed memory metrics if in scope) for all applicable baselines, labels the
  hybrid execution path, records partition-route coverage, and includes enough
  context (qubit count, noise placement class, fusion granularity) to interpret
  wins or absence of wins.

---

## Assumptions and dependencies

- **Gate P31-G-2** from detailed planning: mandatory correctness slice green before performance claims (`DETAILED_PLANNING_PHASE_3_1.md` §11).
- No inflation of Phase 3 Paper 2 claims; Phase 3.1 rows are **additive** (`P31-ADR-002`).

---

## Required behavior

- **Baseline trio** explicitly in each row or in manifest defaults documented in
  **P31-C-06**, with `phase31_channel_native_hybrid` as the counted whole-
  workload Phase 3.1 baseline.
- Emit the Phase 3.1 counted-case metadata from `P31-ADR-013`, including
  `claim_surface_id`, `runtime_class`, `representation_primary`,
  `fused_block_support_qbits`, `contains_noise`, `counted_phase31_case`,
  `decision_class`, and hybrid route-summary fields.
- The frozen performance slice must include the frozen v1 richer
  eligibility class from `P31-ADR-007`: contiguous 1- and 2-qubit mixed motifs
  with multiple same-support local noise insertions, not only primitive
  microcases.
- The counted performance matrix follows the frozen `P31-ADR-010` family rules:
  primary families `phase31_pair_repeat` and `phase31_alternating_ladder`
  (`q ∈ {8,10}`, patterns `{periodic,dense}`, seeds
  `{20260318,20260319,20260320}`) plus sparse `layered_nearest_neighbor`
  controls.
- **Diagnosis fields** when speedup thresholds are not met (e.g. dominant overhead class—Python, memory, unfused structure—as far as the harness can observe).
- Counted whole-workload rows must emit hybrid route-coverage data:
  - `channel_native_partition_count`,
  - `phase3_routed_partition_count`,
  - `channel_native_member_count`,
  - `phase3_routed_member_count`,
  - and partition-level route records with stable `partition_runtime_class` and
    `partition_route_reason`.
- A required **decision artifact** accompanies the timing bundle:
  `break_even_table`, `justification_map`, or equivalent. It must classify the
  frozen performance cases as “Phase 3 sufficient,” “Phase 3.1 justified,” or
  “Phase 3.1 not justified yet,” using the measured results rather than prose
  alone.
- Positive-method closure is measured **versus the Phase 3 fused baseline**:
  at least one representative counted primary-family case must show `>= 1.2x`
  median wall-clock speedup or `>= 15%` peak-memory reduction with no
  correctness loss through the explicit hybrid Phase 3.1 runtime.
- Reproducibility: machine class, build type, relevant env vars; thread count **only** if amended **P31-C-09** allows non-scalar benchmark modes (Task 6 branch).

---

## Unsupported behavior

- Performance claims without sequential **and** Phase 3 fused comparison where both are applicable (`P31-ADR-005`).
- Performance claims on counted whole-workload families using the strict
  `phase31_channel_native` path rather than the explicit hybrid path.
- Cherry-picking only favorable qubit bands without documenting the full frozen **P31-C-06** set.
- Omitting hybrid route-coverage metadata on counted whole-workload rows.
- Omitting scalar-build attestation in mandatory bundles when **P31-C-09** v1 requires it; omitting SIMD/thread metadata when a later **P31-C-09** amendment requires it for optional rows.

---

## Acceptance evidence

- **performance_evidence** (or successor per **P31-C-08**) with version-bumped
  schema where required and with stable case IDs.
- Summary table suitable for paper “Results”:
  case ID → baseline A/B/C → route coverage → outcome → diagnosis tag.
- Required decision artifact: `break_even_table`, `justification_map`, or
  equivalent classification tied to the same case IDs.
- Optional: link to profiler snapshots referenced in Task 6 if that task runs.

---

## Affected interfaces

- Benchmark drivers, report generators, and partition-route aggregation logic
  for hybrid counted cases (paths TBD).
- CI or manual workflow docs for reproducing bundles.

---

## Publication relevance

- Results and limitations sections must **match** emitted bundles; negative or mixed results are first-class (`P31-ADR-005`, planning §6 success conditions).
