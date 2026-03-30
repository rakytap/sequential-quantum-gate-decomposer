# Task 2 Mini-Spec: Runtime Integration and Eligibility

**Phase planning traceability:** `DETAILED_PLANNING_PHASE_3_1.md` §8, Task 2.

**Layer 4 implementation plans (second vertical slice, Story P31-S05)**

- [`ENGINEERING_TASK_P31_S05_E01_IMPLEMENTATION_PLAN.md`](ENGINEERING_TASK_P31_S05_E01_IMPLEMENTATION_PLAN.md)
- [`ENGINEERING_TASK_P31_S05_E02_IMPLEMENTATION_PLAN.md`](ENGINEERING_TASK_P31_S05_E02_IMPLEMENTATION_PLAN.md)

**Pre-implementation checklist traceability**

| P31-C row | Role in this task |
|-----------|-------------------|
| **P31-C-01** | **Dependency:** runtime consumes the frozen primary representation from Task 1. |
| **P31-C-02** | **Dependency:** eligibility in code matches the frozen support matrix; unsupported combinations hard-fail when channel-native mode is selected. |
| **P31-C-07** | **Closed by** `P31-ADR-012`: planner mode remains `partitioned_density`; strict runtime/API surface uses `execute_partitioned_density_channel_native(...)` / `phase31_channel_native`; hybrid whole-workload surface uses `execute_partitioned_density_channel_native_hybrid(...)` / `phase31_channel_native_hybrid`; existing exception families remain, with extended `first_unsupported_condition` vocabulary plus frozen hybrid route reasons. |
| **P31-C-08** | **Informs:** logs or manifests that prove which path executed (no silent fallback). |

Upstream: `P31-ADR-001`, `P31-ADR-003`, `P31-ADR-005`, Task 1 closure.

---

## Scientific outcome

Observable behavior is unambiguous:

- in **strict** mode, every partition is either fully eligible for the new exact
  path or the run fails loudly,
- in **hybrid** mode, eligible partitions run through the new exact path and
  Phase-3-supported but Phase-3.1-ineligible partitions run through the shipped
  Phase 3 exact path with explicit route attribution,
- unsupported-by-both partitions still fail loudly.

That is what makes comparative benchmarks (Task 4) and correctness bundles
(Task 3) interpretable.

---

## Given / When / Then

- **Given** a partitioned noisy execution request with either strict or hybrid
  Phase 3.1 mode **on**, frozen **P31-C-02** eligibility, and Task 1 IR in
  place.
- **When** the runtime schedules operations through a partition stage.
- **Then** the global gate/noise order contract matches the sequential
  `NoisyCircuit` reference ordering for the full circuit, and the selected mode
  behaves as follows:
  - strict mode: any ineligible partition **errors** with a documented code and
    no fallback,
  - hybrid mode: eligible partitions use the channel-native path; documented
    Phase-3-supported but Phase-3.1-ineligible partitions use the shipped Phase
    3 exact path with route attribution; unsupported-by-both partitions error.

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
- The runtime contract now has two explicit surfaces:
  - strict `phase31_channel_native` for fully eligible workloads,
  - hybrid `phase31_channel_native_hybrid` for whole workloads mixing eligible
    and Phase-3-supported but Phase-3.1-ineligible partitions.
- Counted Phase 3.1 fused blocks must contain at least one noise operation;
  pure unitary islands remain part of the Phase 3 fused baseline rather than
  the new counted claim surface.
- Strict mode hard-fails on any Phase-3.1-ineligible partition.
- Hybrid mode may route a partition to the shipped Phase 3 exact path only
  under the frozen route-policy reasons:
  - `pure_unitary_partition`,
  - `channel_native_noise_presence`,
  - `channel_native_qubit_span`,
  - `channel_native_support_surface`.
- Hybrid mode must **not** reroute partitions that fail because of
  representation, invariant, or runtime-execution errors in the channel-native
  path; those remain hard errors.
- Auditable **path attribution** is required at partition granularity:
  strict mode proves "channel-native block applied here," while hybrid mode
  proves "channel-native here, Phase 3 exact here, for this explicit reason."
- Ordering: no fusion that changes CPTP outcome vs sequential reference on the mandatory slice (validated in Task 3).

---

## Unsupported behavior

- Silent fallback from strict channel-native mode to another execution path
  while benchmarks or UI still claim strict channel-native execution.
- Hybrid routing without emitted route attribution or without a frozen
  route-policy reason.
- Warning-only behavior for unsupported channel-native requests.
- Silently accepting motifs that exceed the v1 slice (support > 2 qubits,
  spectator-qubit effects, correlated noise, or broader primitive surfaces not
  frozen in the contract).
- Routing representation, invariant, or channel-native runtime-execution
  failures through the Phase 3 path as if they were support-surface
  ineligibility.
- Mode names that collide with Phase 3 “partitioned density” wording without distinction (`P31-ADR-002` additive labeling).

---

## Acceptance evidence

- Integration tests:
  - strict supported case exercises the new path,
  - strict unsupported case fails deterministically with documented taxonomy,
  - hybrid supported whole-workload case records both channel-native and Phase 3
    routed partitions with stable reasons (**P31-C-07**).
- At least one **end-to-end** small circuit comparing channel-native path vs sequential reference under **P31-C-03** (after frozen)—can overlap Task 3 if a single test satisfies both.
- Short interface note for `API_REFERENCE_PHASE_3_1.md` (when created): modes,
  runtime-path labels, hybrid route reasons, Phase 3.1
  `first_unsupported_condition` vocabulary, and defaults.

---

## Affected interfaces

- `squander/partitioning/` runtime entrypoints, fusion modules, planner/runtime
  flags, and partition-route audit records (exact symbols TBD).
- Public or harness-facing kwargs / env / config keys (**P31-C-07**).

---

## Publication relevance

- Methods text can describe **how** the strict and hybrid modes are invoked, how
  hybrid routing is classified, and how failures surface—supports
  reproducibility and Side Paper A “when does this path apply?”
