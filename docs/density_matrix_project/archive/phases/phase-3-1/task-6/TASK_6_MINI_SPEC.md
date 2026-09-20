# Task 6 Mini-Spec: C++ Hot-Path Offload and Host Parallel / SIMD Evaluation (Later Branch)

**Phase planning traceability:** `DETAILED_PLANNING_PHASE_3_1.md` §8, Task 6 and §1.4.

**Program stance for Phase 3.1 v1:** **Task 6 is out of the v1 implementation gate.** Mandatory correctness and performance bundles use **scalar-only** evidence builds with **no TBB and no SIMD-specific benchmark rows** in mandatory artifacts. **P31-C-09** closes on that policy without delivering Task 6. This mini-spec defines the **later branch** if host acceleration is re-opened after channel-native fusion evidence exists.

**Pre-implementation checklist traceability**

| P31-C row | Role in this task (later branch) |
|-----------|----------------------------------|
| **P31-C-09** | **Closed for v1 without Task 6:** `P31-ADR-014` freezes the counted build policy as `phase31_scalar_only_v1` with `build_flavor = "scalar"`, `simd_enabled = false`, `tbb_enabled = false`, `thread_count = 1`, and `counted_claim_build = true`. Reopening **P31-C-09** for optional variants is a **contract amendment** when Task 6 starts. |
| **P31-C-03** | Accelerated builds must match sequential reference within frozen thresholds on the mandatory slice (or documented superset). |
| **P31-C-04** | Correctness cases re-run for any build flavor advertised in papers or default benchmarks. |
| **P31-C-06**, **P31-C-08** | A/B benchmark rows and performance manifests include build variant labels when variants exist. |

Upstream (when Task 6 runs): Tasks 3–4 complete for v1, `DETAILED_PLANNING_PHASE_3_1.md` §1.4, density module / CMake precedent.

---

## Scientific outcome (when this branch opens)

The project tests whether **host-side** numeric acceleration moves the
**measured** bottleneck on agreed workloads, or documents that Python, pybind,
or memory traffic still dominate. Under the current strict-plus-hybrid Phase 3.1
contract, the claim-carrying whole-workload path is the explicit **hybrid**
runtime, while the **strict** path remains the motif-proof sanity reference.
That distinction is **orthogonal** to the Side Paper A fusion question; results
must be **separately attributed** in prose and bundles (`P31-ADR-005`).

---

## Given / When / Then (later branch)

- **Given** v1 Phase 3.1 evidence already establishes channel-native (or
  negative-result) claims under scalar-only **P31-C-09**, and a **reopened**
  **P31-C-09** records allowed SIMD/TBB variants and metadata rules.
- **When** an accelerated build is enabled for profiling or optional benchmark rows.
- **Then** performance artifacts record **build flags, SIMD level, thread
  count**, correctness is re-validated for that flavor on the mandatory slice,
  and optional rows are **never** mixed into mandatory bundle schema without a
  checklist revision. Whole-workload performance variants identify whether they
  apply to the strict or hybrid runtime path.

---

## Assumptions and dependencies

- **v1:** No Task 6 engineering; **P31-C-09** documents scalar-only mandatory policy only.
- The counted whole-workload Phase 3.1 performance question belongs to the
  explicit **hybrid** runtime path; strict-path acceleration rows are supportive
  sanity data, not the primary whole-workload benchmark claim.
- Threading must not introduce nondeterminism that breaks **P31-C-03** unless explicitly allowed and bounded (planning §11).

---

## Required behavior (later branch only)

- Profile **before/after** on representative cases shared with Task 4, with at
  least:
  - one strict motif-proof case,
  - and one hybrid whole-workload case.
- Any accelerated path: mandatory-slice tests, documented in amended **P31-C-09**.
- Performance bundles: optional section or manifest extension for variant labels (`P31-C-08`); mandatory rows remain scalar-only unless the phase explicitly widens the contract.
- If Task 6 reports whole-workload speedup or bottleneck movement, it must do so
  against the explicit **hybrid** runtime path rather than collapsing strict and
  hybrid results into one opaque “Phase 3.1” number.

---

## Unsupported behavior

- Shipping TBB/SIMD rows inside **mandatory** Phase 3.1 v1 bundles while claiming the v1 checklist is closed on scalar-only policy.
- Claiming SIMD/TBB benefit without A/B rows and build metadata.
- Conflating Task 6 outcomes with **channel-native fusion** speedup claims without separate attribution.
- Reporting hybrid whole-workload acceleration using strict-only microcase
  evidence.

---

## Acceptance evidence

- **v1 (no Task 6):** `PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md` **P31-C-09** closed with explicit text: mandatory evidence builds are scalar reference; no TBB/SIMD in mandatory bundles; Task 6 deferred—no profiler requirement for Gate P31-G-1.
- **v1 migration policy:** even if Phase 3.1 later becomes the default evidence
  pipeline surface, the default counted bundles remain scalar-only until
  `P31-C-09` is explicitly re-opened; historical Phase 3 remains available via
  explicit legacy scripts/functions.
- **Later branch:** profiler snapshots or summarized tables; amended **P31-C-09**; optional bundle schema revision.
- **Later branch attribution:** optional Task 6 results name whether the row
  applies to:
  - strict `phase31_channel_native`,
  - hybrid `phase31_channel_native_hybrid`.

---

## Affected interfaces

- C++ `density_matrix` (or related) entrypoints, CMake options, TBB usage sites.
- Python bindings and dispatch hot paths.

---

## Publication relevance

- If Task 6 never runs: papers state host acceleration was **out of scope** for v1 evidence (scalar-only **P31-C-09**).
- If Task 6 runs: methods or appendix—**one paragraph** tied to amended
  **P31-C-09** and optional Task 4 rows, not a primary Phase 3.1 fusion claim;
  whole-workload acceleration language must be tied to the hybrid path
  explicitly.
