# Task 2 Mini-Spec: Runtime Integration and Eligibility

**Phase planning traceability:** `DETAILED_PLANNING_PHASE_3_1.md` §8, Task 2.

**Pre-implementation checklist traceability**

| P31-C row | Role in this task |
|-----------|-------------------|
| **P31-C-01** | **Dependency:** runtime consumes the frozen primary representation from Task 1. |
| **P31-C-02** | **Dependency:** eligibility in code matches the frozen support matrix; unsupported combinations hard-fail when channel-native mode is selected. |
| **P31-C-07** | **Closed by** `P31-ADR-012`: planner mode remains `partitioned_density`; counted runtime/API surface uses `execute_partitioned_density_channel_native(...)` and runtime-path label `phase31_channel_native`, with existing exception families and extended `first_unsupported_condition` vocabulary. |
| **P31-C-08** | **Informs:** logs or manifests that prove which path executed (no silent fallback). |

Upstream: `P31-ADR-001`, `P31-ADR-003`, `P31-ADR-005`, Task 1 closure.

---

## Scientific outcome

Observable behavior is unambiguous: **when a user or harness requests the Phase 3.1 fusion mode, either eligible regions run through the new exact path with preserved global order, or the run fails loudly.** That is what makes comparative benchmarks (Task 4) and correctness bundles (Task 3) interpretable.

---

## Given / When / Then

- **Given** a partitioned noisy execution request with Phase 3.1 channel-native mode **on**, frozen **P31-C-02** eligibility, and Task 1 IR in place.
- **When** the runtime schedules operations through a partition stage that contains an eligible fused block.
- **Then** the global gate/noise order contract matches the sequential `NoisyCircuit` reference ordering for the full circuit, and ineligible channel-native requests **error** with a documented code/message (no silent downgrade to unitary-island-only or sequential path while still claiming channel-native mode).

---

## Assumptions and dependencies

- Task 1 representation is accepted (**P31-C-01** closed).
- `P31-ADR-007` defines the frozen v1 support slice: contiguous mixed
  gate+noise motifs on total support of 1 or 2 qubits using `U3` / `CNOT` plus
  local single-qubit depolarizing, amplitude-damping, and phase-damping
  channels on those same qubits, including multiple successive gates and
  multiple local noise insertions within one fused block.
- Phase 3 conservative fusion remains available as a distinct baseline (**P31-ADR-005** trio).
- Exact numeric checks use **P31-C-03** once frozen.

---

## Required behavior

- Explicit **mode** or **flag** surface (names frozen under **P31-C-07**) selecting channel-native fusion.
- Planner and descriptor entry continue to use
  `requested_mode="partitioned_density"`; the counted execution identity is the
  distinct runtime path `phase31_channel_native`.
- **Hard errors** for unsupported patterns when that mode is active (`P31-ADR-001`, `P31-ADR-003`).
- Auditable **path attribution**: evidence hooks or logs sufficient to show “channel-native block applied here” vs “unitary island / sequential step.”
- The v1 supported class includes the richer same-support mixed-motif slice from
  `P31-ADR-007`, not only single gate+single noise microcases.
- Counted Phase 3.1 fused blocks must contain at least one noise operation;
  pure unitary islands remain part of the Phase 3 fused baseline rather than
  the new counted claim surface.
- Ordering: no fusion that changes CPTP outcome vs sequential reference on the mandatory slice (validated in Task 3).

---

## Unsupported behavior

- Silent fallback from channel-native mode to another execution path while benchmarks or UI still claim channel-native fusion.
- Warning-only behavior for unsupported channel-native requests.
- Silently accepting motifs that exceed the v1 slice (support > 2 qubits,
  spectator-qubit effects, correlated noise, or broader primitive surfaces not
  frozen in the contract).
- Mode names that collide with Phase 3 “partitioned density” wording without distinction (`P31-ADR-002` additive labeling).

---

## Acceptance evidence

- Integration tests: supported case exercises new path; unsupported case fails deterministically with documented taxonomy (**P31-C-07**).
- At least one **end-to-end** small circuit comparing channel-native path vs sequential reference under **P31-C-03** (after frozen)—can overlap Task 3 if a single test satisfies both.
- Short interface note for `API_REFERENCE_PHASE_3_1.md` (when created): modes,
  runtime-path labels, Phase 3.1 `first_unsupported_condition` vocabulary, and
  defaults.

---

## Affected interfaces

- `squander/partitioning/` runtime entrypoints, fusion modules, and planner/runtime flags (exact symbols TBD).
- Public or harness-facing kwargs / env / config keys (**P31-C-07**).

---

## Publication relevance

- Methods text can describe **how** the new mode is invoked and how failures are classified—supports reproducibility and Side Paper A “when does this path apply?”
