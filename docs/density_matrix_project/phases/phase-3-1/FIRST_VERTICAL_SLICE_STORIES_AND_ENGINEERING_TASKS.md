# Phase 3.1 — First Vertical Slice (Layers 3 and 4)

This document is the **implementation planning slice** for Phase 3.1: Tasks 1–2
plus the **smallest Task 3 closure** needed to prove correctness on one counted
case. It follows the spec-driven-development skill: Layer 2 mini-specs remain
authoritative for requirements; this file adds **behavioral stories** (Layer 3)
and **engineering tasks** (Layer 4) only for this slice.

**Do not** treat this as the full Task 3 matrix (`P31-ADR-009`, full Aer slice,
full `correctness_evidence` schema per `P31-ADR-013`). Those expand after this
slice produces runtime feedback.

## Source mini-specs (Layer 2)

| Task | Document |
|------|----------|
| 1 | [`task-1/TASK_1_MINI_SPEC.md`](task-1/TASK_1_MINI_SPEC.md) |
| 2 | [`task-2/TASK_2_MINI_SPEC.md`](task-2/TASK_2_MINI_SPEC.md) |
| 3 (slice only) | [`task-3/TASK_3_MINI_SPEC.md`](task-3/TASK_3_MINI_SPEC.md) |

## Slice boundary

**In**

- Primary representation `kraus_bundle` with composition and CPTP invariant
  checks on **at least one** hand-validated 1-qubit and **at least one**
  2-qubit eligible motif (`P31-ADR-004`, `P31-ADR-007`, `P31-ADR-008`).
- Runtime path identity `phase31_channel_native` /
  `execute_partitioned_density_channel_native(...)` with **hard errors** and
  auditable path attribution for unsupported channel-native requests
  (`P31-ADR-012`).
- **One** frozen counted microcase end-to-end: internal sequential oracle
  agreement at **P31-C-03** thresholds, plus representation-level invariant
  assertions recorded in a **minimal** reproducible form (tests + optional small
  structured stub aligned with the direction of `P31-ADR-013`).

**Out (deferred to full Task 3 / later slices)**

- Remaining `phase31_microcase_*` IDs, continuity anchors, performance cases as
  correctness rows.
- Full external Aer matrix from `P31-ADR-011` (slice may omit Aer until the
  chosen microcase is wired; full Task 3 requires Aer on the ADR-011 subset).
- Version-bumped production `correctness_evidence` bundles and CI regeneration
  policy at full schema breadth.

**Recommended first counted case ID for the slice:**  
`phase31_microcase_1q_u3_local_noise_chain` — smallest support; swap to another
`phase31_microcase_*` only if implementation order forces it, and record the
change here.

---

### Story P31-S01: Kraus-bundle fused block matches sequential `NoisyCircuit` semantics

**User/Research value**

- Establishes the **exact mathematical object** and its agreement with the
  sequential oracle before partitioning claims are meaningful (`TASK_1_MINI_SPEC`
  scientific outcome).

**Given / When / Then**

- **Given** a contiguous eligible v1 motif (1- or 2-qubit mixed gate+noise on
  the same support, at least one noise op, primitives per `P31-ADR-007`) lifted
  into the Phase 3.1 fusion IR.
- **When** the fused block is applied to a density state.
- **Then** the result matches sequential density evolution of the same operation
  list under canonical `NoisyCircuit` semantics within **P31-C-03**, and
  representation-level CPTP checks pass at **P31-ADR-008** policy for the primary
  `kraus_bundle` form.

**Scope**

- **In:** IR types / handoff shape, ordered composition, sequential reference
  comparison on representative micro-motifs, invariant suite on those motifs.
- **Out:** Full partition planner integration (Story P31-S02); full case
  registry (Story P31-S03 expansion).

**Acceptance signals**

- **Slice v1 (implemented):** one counted **1-qubit** mixed motif
  (`phase31_microcase_1q_u3_local_noise_chain`): fused channel-native apply vs
  sequential reference within **P31-C-03** (Frobenius family); Kraus
  completeness / Choi positivity + trace-preservation checks on the composed
  `kraus_bundle` at **P31-ADR-008** thresholds; plus regression coverage for
  parametric noise rates clamped like `NoisyCircuit` (same contract as sequential
  oracle).
- **Deferred (next increment):** a **2-qubit** motif on the same contract
  (requires `CNOT` and 4×4 Kraus / embedding in channel-native apply); explicitly
  **out of slice v1** implementation status. See
  [`SECOND_VERTICAL_SLICE_2Q_LOCAL_SUPPORT_STORIES_AND_ENGINEERING_TASKS.md`](SECOND_VERTICAL_SLICE_2Q_LOCAL_SUPPORT_STORIES_AND_ENGINEERING_TASKS.md).
- Short design note (or ADR pointer) listing rejected representation shortcuts.

**Traceability**

- Phase: `DETAILED_PLANNING_PHASE_3_1.md` §8 Task 1; **P31-C-01**, **P31-C-03**,
  **P31-C-04** (subset).
- ADRs: `P31-ADR-004`, `P31-ADR-007`, `P31-ADR-008`, `P31-ADR-003`.

#### Engineering tasks (Story P31-S01)

##### Engineering Task P31-S01-E01: Freeze the slice IR and handoff record

**Implements story**

- Story P31-S01

**Change type**

- docs | code

**Definition of done**

- Named Python (and if needed C++ boundary) types or records carry ordered
  `kraus_bundle` payload for fused blocks on 1–2 qubits, with explicit
  unsupported surface documented.
- `TASK_1_MINI_SPEC` traceability row: eligibility class → object → sequential
  segment is filled for the slice motifs.

**Execution checklist**

- [ ] Align field names with existing partition/fusion modules to avoid silent
      duplication.
- [ ] Document unsupported tiers that still hard-fail in channel-native mode
      (Story P31-S02).

**Evidence produced**

- Short IR note in code comments or `docs/...` pointer acceptable for slice;
      final API surface tracked toward `API_REFERENCE_PHASE_3_1.md` in Task 2.

**Risks / rollback**

- Risk: premature schema lock-in before Task 2 wiring.
- Mitigation: version internal record as slice-local until Task 3 schema work.

---

##### Engineering Task P31-S01-E02: Implement fusion lowering and sequential reference tests

**Implements story**

- Story P31-S01

**Change type**

- code | tests

**Definition of done**

- Eligible operation subsequences lower to `kraus_bundle` and apply correctly vs
  sequential `NoisyCircuit` reference for the slice motifs.

**Execution checklist**

- [ ] Unit-level tests independent of full partitioned harness.
- [ ] Residuals and validity checks assert **P31-C-03** / **P31-ADR-008**
      numerics.

**Evidence produced**

- Pytest (or project-standard) runs recorded; no performance claims.

**Risks / rollback**

- Risk: hidden reordering vs sequential reference.
- Mitigation: test mirrors exact operation order from reference builder.

---

### Story P31-S02: Channel-native mode is explicit, ordered, and non-silent

**User/Research value**

- Benchmarks and correctness are interpretable: either the **counted** path ran
  or the run failed with a **documented** condition (`TASK_2_MINI_SPEC`).

**Given / When / Then**

- **Given** partitioned execution with channel-native fusion **on** and Task 1
  IR available.
- **When** the runtime hits eligible fused regions or ineligible channel-native
  requests.
- **Then** global ordering matches sequential reference; eligible regions use
  the channel-native apply; ineligible patterns **error** with taxonomy
  consistent with **P31-C-07** (no silent fallback while advertising
  channel-native mode).

**Scope**

- **In:** Mode/flag surface, `phase31_channel_native` attribution, eligibility
  gate, deterministic failure tests.
- **Out:** Full `first_unsupported_condition` catalog across all future motifs;
      slice needs **representative** negative cases only.

**Acceptance signals**

- Integration test: supported small circuit exercises channel-native path with
  observable path label or equivalent audit hook.
- Integration test: intentionally unsupported circuit fails with stable
  exception/category.
- No silent downgrade to Phase 3 unitary-island-only path when channel-native
  mode is requested.

**Traceability**

- Phase: `DETAILED_PLANNING_PHASE_3_1.md` §8 Task 2; **P31-C-02**, **P31-C-07**,
  **P31-C-08** (path attribution direction).
- ADRs: `P31-ADR-012`, `P31-ADR-001`, `P31-ADR-003`, `P31-ADR-005`.

#### Engineering tasks (Story P31-S02)

##### Engineering Task P31-S02-E01: Wire `execute_partitioned_density_channel_native` and path reporting

**Implements story**

- Story P31-S02

**Change type**

- code | tests

**Definition of done**

- Counted runtime entry applies Story P31-S01 fused blocks inside partitions
  when eligible.
- Result or log surface exposes `phase31_channel_native` (or agreed equivalent)
  for audit.

**Execution checklist**

- [ ] Planner `requested_mode` remains `partitioned_density` per **P31-ADR-012**.
- [ ] Reuse existing exception families; extend vocabulary only as needed for
      slice errors.

**Evidence produced**

- Tests proving path attribution on a trivial eligible circuit.

**Risks / rollback**

- Risk: ambiguous overlap with `execute_partitioned_density_fused`.
- Mitigation: single documented discriminator in results/metadata.

---

##### Engineering Task P31-S02-E02: Eligibility preflight and negative matrix for the slice

**Implements story**

- Story P31-S02

**Change type**

- code | tests

**Definition of done**

- At least **two** negative cases (e.g. wrong support class, pure unitary island
  incorrectly requested as counted Phase 3.1 fused block if applicable) fail
  deterministically with documented codes/messages.

**Execution checklist**

- [ ] Cases align with `P31-ADR-007` wording (no silent widening).

**Evidence produced**

- Pytest markers or table in test module docstring listing case → expected
      failure.

**Risks / rollback**

- Risk: over-broad rejection blocking later motifs.
- Mitigation: slice cases are minimal; expand with Task 3 full matrix.

---

### Story P31-S03: One counted microcase closes the end-to-end correctness loop

**User/Research value**

- Delivers **reproducible proof** that the partitioned channel-native path
  matches the internal oracle on a **frozen** Phase 3.1 ID — the minimum
  credible “correctness package” step before scaling bundles (`TASK_3_MINI_SPEC`,
  slice interpretation).

**Given / When / Then**

- **Given** the frozen microcase ID chosen in the slice boundary section and a
  build with Stories P31-S01–S02 implemented.
- **When** the slice correctness gate runs.
- **Then** channel-native vs sequential agreement meets **P31-C-03**; fused-block
  invariants meet **P31-ADR-008**; outcomes are recorded in a minimal structured
  form suitable for later promotion to full `correctness_evidence`.

**Scope**

- **In:** One `phase31_microcase_*` end-to-end; internal oracle only for this
  slice unless Aer is already trivial to attach.
- **Out:** Remaining **P31-ADR-009** cases; full **P31-ADR-011** Aer rows; full
  **P31-ADR-013** bundle emission and schema version bump.

**Acceptance signals**

- Single test (or small harness module) keyed by stable case ID string.
- Explicit assertion table: Frobenius (or contract distance), trace validity,
  positivity floor, plus at least one representation invariant row for the
  fused block extracted from that run.
- Optional: minimal JSON-like dict written by test fixture — must not pretend
  to be the final **P31-C-08** schema without version field and checklist
  update.

**Traceability**

- Phase: `DETAILED_PLANNING_PHASE_3_1.md` §8 Tasks 2–3; **P31-C-03**,
  **P31-C-04** (one ID), **P31-C-08** (directional).
- ADRs: `P31-ADR-009` (subset), `P31-ADR-013` (future full alignment).

#### Engineering tasks (Story P31-S03)

##### Engineering Task P31-S03-E01: Build the slice microcase workload and E2E comparison

**Implements story**

- Story P31-S03

**Change type**

- code | tests

**Definition of done**

- One builder or fixture produces the same logical circuit for sequential
  reference and partitioned channel-native execution.
- Assertions enforce **P31-C-03** and **P31-ADR-008** on the final state (and
  fused block invariants where the harness exposes them).

**Execution checklist**

- [ ] Register case ID string consistently for later Task 3 matrix merge.
- [ ] Document in test module what is **not** yet covered (Aer, other IDs).

**Evidence produced**

- Passing test log; optional artifact file under project conventions.

**Risks / rollback**

- Risk: fixture drift from eventual benchmark builders.
- Mitigation: share builders with Task 3 harness when it lands.

---

##### Engineering Task P31-S03-E02: Record slice completion and Task 3 expansion hook

**Implements story**

- Story P31-S03

**Change type**

- docs

**Definition of done**

- This file records the chosen slice case ID, marks **first vertical slice
  implementation complete** when true, and points to full Task 3 for remaining
  IDs (see **Slice implementation status** at the end of this document). Do
  **not** use `PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md` for slice progress:
  that checklist is the closed Layer 1 gate only.

**Execution checklist**

- [ ] Update the **Slice implementation status** section when the slice is
      implemented (case ID, date or commit ref if useful, link to tests).
- [ ] Optional: one-line pointer in `DETAILED_PLANNING_PHASE_3_1.md` §10 to this
      file’s status section — not to the pre-implementation checklist.

**Evidence produced**

- Traceability note for reviewers in this file’s status section.

**Risks / rollback**

- Risk: mixing Layer 1 contract closure with implementation progress.
- Mitigation: slice closure lives only here; full Task 3 / bundle closure is
  tracked under `task-3/TASK_3_MINI_SPEC.md` and planning when that work ships.

---

## Story-to-engineering map

| Story | Engineering tasks |
|-------|-------------------|
| P31-S01 | P31-S01-E01, P31-S01-E02 |
| P31-S02 | P31-S02-E01, P31-S02-E02 |
| P31-S03 | P31-S03-E01, P31-S03-E02 |

## Suggested implementation order

1. P31-S01-E01 → P31-S01-E02 (representation grounded before runtime).
2. P31-S02-E01 → P31-S02-E02 (runtime + failures).
3. P31-S03-E01 → P31-S03-E02 (E2E proof + documentation hook).

Parallelism: P31-S01-E02 can start after E01 shapes stabilize; P31-S02 work can
prototype against S01 unit tests before full partition wiring.

---

## Slice implementation status

Update this section when P31-S03-E02 completes (implementation progress only; do
not fold into `PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md`).

| Field | Value |
|-------|-------|
| Slice case ID | `phase31_microcase_1q_u3_local_noise_chain` |
| First vertical slice implementation | **Complete** for 1q channel-native motifs (code: `execute_partitioned_density_channel_native`, Kraus composition + invariant in `squander/partitioning/noisy_runtime_channel_native.py`). **P31-S01** acceptance for a **2-qubit** fused motif is **not** claimed here until channel-native supports `CNOT` / 2q Kraus embedding (deferred increment; see `SECOND_VERTICAL_SLICE_2Q_LOCAL_SUPPORT_STORIES_AND_ENGINEERING_TASKS.md`). |
| Tests / evidence pointer | `tests/partitioning/test_partitioned_channel_native_phase31_slice.py` — run with `conda run -n qgd python -m pytest tests/partitioning/test_partitioned_channel_native_phase31_slice.py -q` |
| Full Task 3 expansion | `task-3/TASK_3_MINI_SPEC.md` — remaining `P31-ADR-009` IDs, Aer per `P31-ADR-011`, bundles per `P31-ADR-013` |
