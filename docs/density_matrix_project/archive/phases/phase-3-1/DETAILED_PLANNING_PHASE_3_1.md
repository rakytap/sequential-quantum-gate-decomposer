# Detailed Planning for Phase 3.1

This document is the Phase 3.1 source of truth for scope, deliverable goals,
acceptance criteria, validation expectations, and publication evidence for
channel-native / superoperator fusion as a follow-on to the closed Phase 3
baseline.

Primary Phase 3.1 theme:

> extend the delivered noise-aware partitioned density runtime with a
> benchmark-justified channel-native or superoperator-native fusion path where
> the Phase 3 conservative fused baseline is insufficient, while preserving exact
> mixed-state semantics against the sequential `NoisyCircuit` reference.

This is a specification document, not an implementation log.

## Source-of-Truth Hierarchy

If multiple documents overlap, interpret them in the following order of authority
for Phase 3.1 (mirrors Phase 3 tiered structure, extended to Tier 5 for local
context; phase-local files win only
where they explicitly tighten Phase 3.1 scope).

### Tier 1: Strategic Planning Constraints

- `docs/density_matrix_project/planning/PLANNING.md` (§4 Phase 3.1, §5.1)
- `docs/density_matrix_project/planning/ADRs.md` (especially ADR-007)
- `docs/density_matrix_project/planning/PUBLICATIONS.md`
- `docs/density_matrix_project/planning/REFERENCES.md`

### Tier 2: Accepted Milestone Wording

- `docs/density_matrix_project/CHANGELOG.md`
- `docs/density_matrix_project/RESEARCH_ALIGNMENT.md`

### Tier 3: Closed Phase 3 Baseline (Extended, Not Reopened)

- `docs/density_matrix_project/ARCHITECTURE.md`
- `docs/density_matrix_project/README.md`
- `docs/density_matrix_project/phases/phase-3/DETAILED_PLANNING_PHASE_3.md`
- `docs/density_matrix_project/phases/phase-3/ADRs_PHASE_3.md`
- `docs/density_matrix_project/phases/phase-3/PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md`

### Tier 4: Phase 3.1 Working Contract Set (This Directory)

Layer 1 implementation contract and gate artifacts:

- `DETAILED_PLANNING_PHASE_3_1.md` (this file)
- `ADRs_PHASE_3_1.md`
- `PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md` (Gate P31-G-1; maps open items to frozen values)

Publication surfaces aligned with `PUBLICATIONS.md` Phase 3.1 / Side Paper A
(authored in parallel with planning per spec-driven workflow):

- `SHORT_PAPER_PHASE_3_1.md` (technical methods / validation posture)
- `SHORT_PAPER_NARRATIVE.md` (research arc positioning)
- `ABSTRACT_PHASE_3_1.md`
- `PAPER_PHASE_3_1.md`
- `CLOSURE_PLAN_PHASE_3_1.md` (evidence-first closeout playbook from remaining
  execution slices through publication sync)

### Tier 5: Legacy Or Supportive Context

- `docs/density_matrix_project/phases/phase-3/PHASE_3_ALGORITHM_LANDSCAPE.md`
- `docs/density_matrix_project/phases/phase-3/API_REFERENCE_PHASE_3.md` (API
  evolution for 3.1 documented via mini-specs and a phase-3.1 API reference when
  implementation exists)
- Exploratory notes not part of the accepted Phase 3.1 checklist closure

## 0.1 Starting Baseline After Phase 3

What is already closed before Phase 3.1 implementation is treated as contract-complete:

- Phase 3 delivered the canonical noisy planner surface, schema-backed partition
  descriptors, executable partitioned density runtime, and conservative
  unitary-island fusion on eligible substructures, with layered correctness and
  performance evidence on the audited Phase 3 support surface.
- Sequential `NoisyCircuit` density execution remains the internal semantic ground
  truth for any Phase 3.1 path.
- Phase 3 performance closure was diagnosis-grounded; channel-native /
  superoperator-native fusion was explicitly deferred as follow-on (see
  `P3-ADR-010`, PLANNING.md §5.1).
- Current implementation-backed Phase 3.1 progress already proves the
  **strict-plus-hybrid runtime design at thin-slice level**:
  - `phase31_channel_native` is implemented and validated on the 1-qubit slice
    and the bounded 2-qubit local-support slice,
  - `phase31_channel_native_hybrid` is implemented with explicit route
    attribution on the counted `phase2_xxz_hea_q4_continuity` anchor and on one
    frozen structured pilot,
    `phase31_pair_repeat_q8_periodic_seed20260318`.
- This means the open Phase 3.1 question is no longer "does the runtime design
  exist?" but "does the full frozen correctness and performance slice justify a
  bounded positive-methods closure relative to the Phase 3 fused baseline, or
  does it close more honestly as a benchmark-grounded decision study?"

What remains intentionally open for **full Phase 3.1 claim closure**:

- remaining counted **correctness** closure on the frozen slice:
  - `phase31_microcase_2q_multi_noise_entangler_chain`,
  - `phase31_microcase_2q_dense_same_support_motif`,
  - `phase2_xxz_hea_q6_continuity`,
  - required Aer rows from §10.4,
  - full `correctness_evidence` / `channel_invariants` /
    `partition_route_summary` migration under `P31-ADR-013`,
- remaining counted **performance** closure on the frozen slice:
  - the full `P31-ADR-010` matrix beyond the current pilot,
  - control-family closure,
  - emitted `break_even_table` / `justification_map`,
  - route-aware case summaries on the counted hybrid rows,
- the **publication-closure decision** after the emitted evidence exists and is
  reviewed against §6 and §13,
- optional **host-side C++/TBB/AVX** work only if the later Task 6 branch is
  explicitly opened; it is not part of the recommended immediate closure path.

## 0.1.1 Current State And Recommended Closure Path

This rebaseline note updates the **execution stance** for Phase 3.1 without
reopening the frozen contracts in §7 or the counted slices in §10.

- **Keep the frozen Phase 3.1 contract.** Do not broaden the support surface,
  relax thresholds, or reduce the counted slice because of partial
  implementation feedback.
- **Rebaseline the working approach to evidence-first closure.** The feature
  existence question has already been answered by the strict slice, the hybrid
  runtime, the counted `q4` continuity anchor, and the first structured pilot.
- **Treat the positive-methods outcome as allowed but not assumed.** Current
  evidence includes a real hybrid pilot row, but that row is presently
  overhead-dominant relative to the Phase 3 fused baseline; it is informative,
  not claim-closing.
- **Finish Tasks 3 and 4 before deciding publication closure.** The next phase
  of work should close the remaining correctness and performance contracts on
  the frozen slice and produce a pre-closure evidence summary that makes the
  publication choice mechanical rather than interpretive.
- **Keep Task 5 downstream and Task 6 separate.** Publication framing follows
  emitted evidence; host-acceleration work remains a later branch unless the
  phase contract is explicitly reopened.

## 0.2 Spec-Driven Development Principles for Phase 3.1

1. Close phase-wide representation, support matrix, and evidence contracts before
   implementation details drift (checklist Gate P31-G-1).
2. Separate **required** channel-native behavior from **unsupported** tiers:
   - strict channel-native mode has **no fallback**,
   - hybrid mode may route only through **explicit documented paths** with
     emitted route attribution, never through silent downgrade
   (`P31-ADR-003`, `P31-ADR-001`).
3. Keep Phase 3.1 **additive** relative to closed Phase 3 claims
   (`P31-ADR-002`).
4. Tie publication wording to emitted evidence bundles (`P31-ADR-005`).
5. Task mini-specs precede engineering work per `P31-ADR-001`.

## Traceability Matrix

| Upstream requirement | Phase 3.1 interpretation | Primary record |
|---|---|---|
| PLANNING.md §5.1 | Formal follow-on phase for channel-native / superoperator fusion | this document, §1 and §3 |
| ADR-007 | Execution and publication container when the branch opens | `ADRs_PHASE_3_1.md`; ADR-007 cross-link |
| P3-ADR-010 / Phase 3 closure | Does not reopen Phase 3 baseline claims; adds a new fusion tier | `P31-ADR-002` |
| Phase 3 diagnosis-grounded performance closure | Motivation: conservative fusion may leave headroom for native channel fusion | PLANNING.md Phase 3 evidence; this document §1 |
| Implementation readiness | Nine frozen contracts (P31-C-01..09) closed before code | `PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md`; this document §7 |

## 1. Purpose and Mission

Phase 3 delivered a native noise-aware planner and partitioned density runtime
with descriptor-local unitary-island fusion on eligible substructures, validated
against sequential `NoisyCircuit` and external references on the frozen Phase 3
workload surface. Performance closure was diagnosis-grounded: representative
structured workloads did not show positive-threshold speedups at 8 and 10 qubits
under the shipped fused path, with analysis pointing toward remaining unfused
structure and Python-level overhead as contributors, and toward more invasive
fusion architectures as a follow-on direction.

Phase 3.1 exists to **plan and bound** that follow-on: **channel-native,
superoperator-native, or IR-first fused execution** of noisy regions where such
representation is exact, auditable, and benchmark-justified relative to both
the sequential density baseline and the Phase 3 partitioned baseline.

The current Phase 3.1 design now has two explicit runtime layers:

- a **strict** motif-proof path, `phase31_channel_native`, for workloads whose
  partitions are all fully eligible for the bounded 1- and 2-qubit mixed-motif
  contract,
- an explicit **hybrid** whole-workload path,
  `phase31_channel_native_hybrid`, for continuity and structured benchmark
  cases, where eligible partitions execute channel-natively and
  Phase-3-supported but Phase-3.1-ineligible partitions execute through the
  shipped Phase 3 exact runtime with auditable route attribution.

### Novelty hypothesis and primary publication stance

Phase 3.1's candidate novelty is **not** the mere use of Kraus, PTM, or
Liouville representations in isolation; those are established open-system
tools. The intended claim is narrower and stronger: SQUANDER can execute
**bounded exact channel-native fusion of mixed gate+noise motifs inside a
partitioned noisy runtime**, first in strict motif-proof mode and then in an
explicit hybrid whole-workload mode, while preserving the canonical ordered
`NoisyCircuit` semantics and delivering a measurable benefit beyond Phase 3's
unitary-island fused baseline on a scientifically chosen workload slice.

For publication planning, Phase 3.1 now targets a **positive methods paper
first** on a bounded v1 slice. The earlier decision-study framing remains a
fallback closure mode if the targeted slice does not produce defensible wins.

### 1.1 Research literature relevant to exact noisy simulation and acceleration

Formal bibliographic entries and tags remain curated in
[`docs/density_matrix_project/planning/REFERENCES.md`](../../planning/REFERENCES.md).
The landscape note
[`../phase-3/PHASE_3_ALGORITHM_LANDSCAPE.md`](../phase-3/PHASE_3_ALGORITHM_LANDSCAPE.md)
interprets how those lines of work map onto Phase 3; Phase 3.1 inherits that
picture and emphasizes the **open-system / fusion** thread.

**Exact and HPC-oriented density-matrix simulation** (motivation for cost of
exact mixed-state evolution, communication, and memory locality):

- Li et al., density-matrix simulation at scale on clusters (`REFERENCES.md` §4).
- Doi and Horii, cache blocking / large-scale simulation (`REFERENCES.md` §4).
- QuEST, Qulacs, Atlas (`REFERENCES.md` §4) as reference simulators and, for
  Atlas, hierarchical GPU-oriented partitioning narratives.

**Circuit partitioning and gate fusion on graph-structured circuits** (most
published methods speak in terms of **dependency DAGs** or derived graphs, often
with **unitary gates** as the primary nodes):

- TDAG, GTQCP, QGo (`REFERENCES.md` §2): tree- or topology-aware cuts, synthesis-
  oriented partitioning, scalable optimization workflows.
- Hierarchical / multilevel partitioning (Fang et al., Burt et al.,
  `REFERENCES.md` §2): multi-level graph coarsening and refinement.

**Fusion as a first-class optimization object**:

- QMin and Nguyen et al. (`REFERENCES.md` §2): control-aware and runtime-oriented
  gate fusion for **state-vector** simulation; ideas transfer to density-matrix
  cost only after recalibration (see Phase 3 landscape).

**Open-system and superoperator viewpoints** (closest bibliographic support for
**channel-native** thinking in this repo today):

- Nielsen and Chuang, quantum operations formalism (`REFERENCES.md` §4):
  standard source for Kraus / Choi / superoperator semantics and CPTP
  conditions.
- Wood, Biamonte, and Cory (`REFERENCES.md` §4): explicit relationships between
  Kraus, Choi, Liouville, and process-matrix representations; useful for
  representation choice and invariant checks.
- QuTiP open-system framework (`REFERENCES.md` §4, tagged `[Future branch]`):
  superoperator / master-equation culture relevant to Phase 3.1 representation
  choices.

**Computational reuse and hardware-aware noisy simulation** (named as research
directions in `PHASE_3_ALGORITHM_LANDSCAPE.md`, including dynamic subcircuit reuse
and hardware-aware exact noisy runtime work): treat as **related work** to cite
once full entries are added to `REFERENCES.md`; they inform **workflow-level**
optimization (repeated circuit evaluations) as distinct from single-shot
partition semantics.

**Noise- or fidelity-aware partitioning in distributed / NISQ settings** (same
landscape note): useful for positioning and comparison, but often **not**
framed around exact dense mixed-state SQUANDER-style execution; use carefully
when claiming novelty.

### 1.2 Partitioning representations in the literature versus the canonical noisy planner surface

| Representation | Typical use in literature | Role in this project (Phase 3 / 3.1) |
|---|---|---|
| **Dependency DAG** | Nodes are operations (often gates); edges are data/control dependencies; TDAG / GTQCP / QGo-style algorithms cut or cluster this graph. | Natural **internal** view for “what can be parallelized or grouped” and for adapting `kahn` / `tdag` / `gtqcp` / `ilp*` **ideas**. Noise is **not** always a first-class node in prior art; Phase 3 makes it first-class in the **noisy** DAG equivalent. |
| **Linearized ordered stream** | Less often a published “algorithm object”; more often the **simulator IR** (time-ordered gates and channels). | The **canonical noisy planner surface** is contractually aligned with an ordered **`NoisyCircuit`-style** stream: explicit gate and **noise** operations in execution order, schema-backed for audit (`noisy_planner` / descriptor pipeline). |
| **Partition descriptors + runtime stages** | Some systems separate “plan” from “execute”; fusion papers emphasize **runtime** fusion. | Phase 3 fixes a handoff: descriptors preserve order and support; runtime (`noisy_runtime` / fusion) consumes them. Phase 3.1 adds **channel-native** execution for eligible **segments** of that same ordered semantics. |

**Planning consequence:** literature that assumes a **pure unitary DAG** should
not be copied verbatim into noisy claims. The scientifically honest mapping is:

- use DAG reasoning for **admissibility** (dependencies, support bounds),
- treat the **ordered noisy surface** as the **contractual** input to partitioning
  and validation,
- implement Phase 3.1 fusion on **contiguous or explicitly delimited** subsequences
  that respect that order (not reordering through noise in ways that would change
  CPTP semantics).

### 1.3 Candidate directions in SQUANDER to optimize exact noisy simulation further

The table below extends the strategy survey in `PHASE_3_ALGORITHM_LANDSCAPE.md`
§4 with a **Phase 3.1 / workflow** lens. It is a planning inventory, not a
commitment to implement every row.

| Candidate | Basis in SQUANDER today | Primary lever | Notes for Phase 3.1 and beyond |
|---|---|---|---|
| **Channel-native / superoperator fusion** | Partitioned runtime + density C++ kernels; Phase 3 unitary-island fusion only | Fuse small gate+noise regions as exact CPTP / superoperator blocks | **Core Phase 3.1** scope (`P31-ADR-004`); addresses noise-dense regions unitary fusion cannot absorb. |
| **Calibrated `ilp` / `ilp-fusion` / `ilp-fusion-ca` on noisy DAG** | `squander/partitioning/ilp.py` and fusion variants; noisy path uses separate planner surface | Density-aware or channel-aware **cost model** on noisy dependency graph | Design-space from Phase 3; Phase 3.1 can assume partitions are given and focus on **runtime** fusion, or co-design partition cost with new fusion eligibility. |
| **`kahn` / `tdag` / `gtqcp` adapted to noisy nodes** | Mature state-vector stack; noisy adaptation partially in design space | Structure-first cuts with noise-aware weights | Improves **where** cuts land when noise density varies; orthogonal to but composable with channel fusion. |
| **Multilevel / hypergraph partitioning** | Not primary noisy path today | Coarsen/refine graph or hypergraph of noisy operations | Optional later scale step (see Burt et al. in `REFERENCES.md`); high engineering cost. |
| **Control-aware fusion recalibrated for density** | `ilp-fusion-ca` lineage | Prefer fused shapes that minimize mixed-state kernel cost | Still mostly **unitary** fusion unless combined with channel-native blocks. |
| **Stronger C++ evolution kernels / layout** | `density_matrix` module, local multi-qubit application | SIMD (e.g. AVX), blocking, memory layout | Supports any fusion tier; see **§1.4** for TBB/AVX investigation framing; does not replace Phase 3.1 semantics work. |
| **Python / handoff overhead reduction** | Planner and runtime in Python; evidence bundles | Hot-path profiling, thinner handoff, compact plan blobs, C++ dispatch | Phase 3 diagnosis flagged overhead; **§1.4** lists concrete offload targets; can unlock wins alongside or before aggressive CPTP fusion. |
| **`qiskit` / `qiskit-fusion` / `bqskit-*` prepartition** | Comparison hooks in partitioning stack | External partition only | Benchmark and sanity comparison, not native claim surface. |
| **Computational reuse across repeated evaluations** | VQE and workflow repetition (Phase 2 anchor) | Cache fused superoperators or intermediate states across parameters | **Workflow** optimization; aligns with dynamic-reuse literature discussed in `PHASE_3_ALGORITHM_LANDSCAPE.md`; mostly **outside** minimum Phase 3.1 unless explicitly scoped. |
| **External reference alignment** | Qiskit Aer, QuEST, Qulacs in benchmarks | Correctness and performance baselines | Retain for Phase 3.1 evidence packages. |

**Traceability:** literature clusters above map to `REFERENCES.md` sections §1–§4
and §7.1; algorithm families map to `PHASE_3_ALGORITHM_LANDSCAPE.md` §2 and §4.
Phase 3.1 implementation should record which rows are **in scope** for the
frozen support matrix versus **explicitly deferred** in ADRs and the pre-
implementation checklist.

### 1.4 C++ offload, host parallelism (TBB), and vectorization (AVX)

Phase 3 performance closure already pointed to **Python-level and handoff
overhead** alongside conservative fusion coverage. Phase 3.1 should treat
**moving compute-bound work from Python into the existing C++ density module**
as a **secondary investigation axis for v1**, composable with channel-native
fusion if the main positive-method slice shows headroom. The primary v1
scientific claim remains the bounded channel-native fusion contract; host
acceleration is supporting diagnosis unless later evidence makes it the
dominant story.

**Intent:** keep schema-heavy planning, validation, and evidence emission in
Python where iteration and audit JSON matter; push **numeric hot paths**
(evolution, superoperator apply, packing/unpacking, repeated small-matrix work)
into C++ where **Intel TBB** (or equivalent task parallelism already used
elsewhere in SQUANDER’s C++ tree) and **AVX/AVX2**-style SIMD can be applied
without changing exact semantics.

**Candidate offload targets (planning inventory):**

| Hot zone (typical) | Why Python hurts | C++ direction |
|---|---|---|
| Per-step application of fused unitary or CPTP blocks to local density tiles | pybind and object churn per micro-step | Single C++ entry: apply fused superoperator / Kraus-squashed kernel on a dense slab with optional SIMD |
| Repeated allocation or copying of `numpy` views into kernels | GIL and allocation frequency | Pre-sized workspace buffers in C++; in-place evolution APIs where contract allows |
| Inner loops over partition stages or fusion regions | Interpreter overhead in tight loops | Stage dispatcher in C++ driven by a compact **plan blob** (indices, supports, op codes) built once from Python |
| Channel-native block composition (Kronecker / Pauli algebra on small supports) | Pure Python linear algebra | Small fixed-dimension `Eigen` or hand-unrolled SIMD blocks in `density_matrix` targets |

**Parallelism (TBB):** use parallel regions only where **dependencies permit**
(for example, independent micro-batches across **different** benchmark cases,
or embarrassingly parallel tensor slices **after** correctness review—not
speculative parallelization of a single sequential noisy circuit unless the
partition contract explicitly exposes independent substeps). Any threaded path
must preserve **bitwise or threshold-stable** agreement with the sequential
reference under the frozen numeric policy, and benchmark harnesses must record
**thread count** and build flags for reproducibility.

**Vectorization (AVX):** prioritize **local** dense operations (fixed small
`n`-qubit supports) where alignment and layout can be controlled; avoid claiming
AVX benefit until profiler and A/B builds show it moves the measured bottleneck.
Align with program precedent: optional kernel acceleration belongs in Phase 3.1
**when evidence shows it supports** the mixed-state partition/fusion story
(`PLANNING.md` Phase 3 technical focus on optional density-kernel work).

**Non-goals for this axis:** replacing the canonical noisy planner surface with
a C++-only IR in Phase 3.1; GPU-first offload as the default claim (remains a
separate track in `ARCHITECTURE.md` / `PLANNING.md` parallel hardware notes);
non-deterministic reductions that break regression comparison unless explicitly
allowed and bounded.

**Evidence expectation:** any C++/TBB/AVX change ships with **before/after**
profiles on the same representative cases used for Phase 3 diagnosis, and
performance bundles name **build configuration** (SIMD level, TBB on/off) beside
the baseline trio.

## 2. Prerequisites

Phase 3.1 may begin **planning** immediately. **Implementation** should assume:

- Phase 3 baseline contracts are delivered and frozen as documented in Phase 3
  artifacts (planner surface, descriptors, partitioned runtime, correctness and
  performance evidence packages on the audited support surface).
- Program-level decision: the research program accepts the engineering and
  validation cost of invasive fusion in exchange for a defensible methods
  contribution or negative result.

## 3. In Scope

- A documented **fusion representation contract** for CPTP / channel /
  superoperator blocks (exact, finite-dimensional) that can attach to the
  existing partitioned noisy pipeline without silently changing semantics.
- **Frozen v1 positive-method slice:** contiguous mixed
  gate+noise motifs on total support of 1 or 2 qubits, built from the Phase 3
  primitive gates (`U3`, `CNOT`) and local single-qubit depolarizing,
  amplitude-damping, and phase-damping channels on those same qubits. Multiple
  successive gates and multiple local noise insertions within one fused block
  are allowed; each eligible Phase 3.1 fused block must contain at least one
  noise operation. Spectator-qubit effects, correlated multi-qubit noise, and
  support beyond 2 qubits remain outside the v1 minimum claim.
- **Eligibility and unsupported** rules: which gate+noise patterns may be
  fused into channel-native blocks, which remain on the shipped Phase 3 exact
  path, and how failures surface:
  - **strict** channel-native mode hard-fails on any ineligible partition,
  - **hybrid** channel-native mode may route only Phase-3-supported but
    Phase-3.1-ineligible partitions to the shipped Phase 3 exact path with
    explicit route attribution,
  - unsupported-by-both cases still fail loudly.
- **Strict-plus-hybrid runtime design**:
  - strict `phase31_channel_native` for counted microcases and motif-proof
    validation,
  - hybrid `phase31_channel_native_hybrid` for bounded continuity anchors and
    structured whole-workload performance cases.
- **Correctness evidence**: exact agreement (within frozen numerical thresholds)
  with sequential `NoisyCircuit` density execution; explicit comparison or
  regression posture versus the Phase 3 partitioned path where both exist, with
  strict-mode evidence on the mixed-motif microcases and hybrid-mode evidence on
  continuity / whole-workload rows.
- **Performance evidence**: structured comparison versus Phase 3 fused baseline
  and sequential baseline on workloads that stress noise density and locality,
  using the explicit hybrid Phase 3.1 path on counted whole-workload cases and
  reporting route coverage together with honest wins, ties, and regressions.
- **Publication package**: claim boundary aligned with `PUBLICATIONS.md` Phase
  3.1 / Side Paper A positioning (decision study or follow-on methods paper).
- **Host-side performance engineering (investigation)**: benchmark-justified
  movement of compute-intensive paths from Python into the C++ density module,
  including optional **TBB** parallel regions and **AVX** (or equivalent SIMD)
  where correctness, determinism, and reproducibility constraints are met; claims
  tied to profiling and A/B evidence, not assumption.

## 4. Out of Scope and Non-Goals

- Phase 4 broader noisy VQE/VQA workflow surface, density-backend gradient
  routing, and optimizer studies (remain Phase 4).
- Stochastic trajectories, MPDO-style tensor methods, and other approximate
  scaling branches (PLANNING.md §5.2–5.3).
- Rewriting or weakening closed Phase 3 / Paper 2 claims; Phase 3.1 is an
  **additive** milestone with its own claim boundary.
- Hardware-specific acceleration tracks as the **primary** scientific claim of
  Phase 3.1 (GPU clusters, vendor-specific first-class narrative, and so on may
  appear only as supporting or later-track context).
- Note: **host** C++ with TBB/AVX to accelerate exact dense kernels is **not**
  excluded; it is an allowed supporting investigation under §3 and §1.4 when
  benchmark evidence supports it, distinct from claiming a new hardware platform
  paper.
- Full arbitrary CPTP fusion of unbounded regions without explicit support-tier
  vocabulary and evidence (must remain bounded and auditable).

## 5. Assumptions

- Exact dense density matrices remain the scientific anchor through Phase 3.1
  (aligned with program ADR-005 unless explicitly reopened at program level).
- Realistic local noise models remain the primary scientific target for
  benchmark narratives.
- Phase 3 machine-checkable evidence patterns (stable case IDs, mandatory vs
  optional evidence, no-fallback rules) are a **template** for Phase 3.1
  evidence design.

## 6. Success Conditions

Phase 3.1 is successful if:

- The **primary intended publication outcome** is a positive methods result on
  the frozen v1 slice: at least one **non-trivial** bounded mixed
  gate+noise eligibility class shows exact channel-native or
  superoperator-native benefit relative to the Phase 3 fused baseline, meeting
  the frozen positive-method threshold in `P31-ADR-010`. For whole-workload
  counted cases, that claim is carried by the explicit hybrid runtime path.
- If the targeted slice fails to show such a benefit, the phase may still close
  through a **benchmark-justified negative result** explaining why the richer
  fusion class does not pay off under the contract.
- The strict `phase31_channel_native` path is demonstrated on the counted
  mixed-motif microcases, and the explicit hybrid
  `phase31_channel_native_hybrid` path is demonstrated on the counted
  continuity / structured-workload slice with auditable partition routing.
- Exactness versus the sequential density baseline is **demonstrated** on the
  mandatory correctness slice for the frozen Phase 3.1 support matrix.
- Publication-facing documents state claims no stronger than the emitted
  evidence bundles allow.

## 7. Frozen Implementation Contracts (to be closed before coding)

The following must be frozen in `ADRs_PHASE_3_1.md` and
`PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md` before implementation is treated as
contract-complete. Each item maps to a checklist ID for auditability.

| Frozen contract | Checklist ID | Primary record |
|---|---|---|
| Primary fusion representation | P31-C-01 | `P31-ADR-004` Accepted |
| Support matrix (gates, channels, max support, patterns) | P31-C-02 | `P31-ADR-007` |
| Numerical correctness thresholds | P31-C-03 | `P31-ADR-008` |
| Mandatory correctness case set (IDs, bands, families) | P31-C-04 | `P31-ADR-009` |
| External reference slice (e.g. Aer) | P31-C-05 | `P31-ADR-011` |
| Performance case set; baseline trio usage | P31-C-06 | `P31-ADR-010` |
| Mode naming and API / error taxonomy | P31-C-07 | `P31-ADR-012` |
| Evidence bundle schema (extend vs new artifacts) | P31-C-08 | `P31-ADR-013` |
| Host build / SIMD / TBB / threading metadata for evidence | P31-C-09 | `P31-ADR-014` |

**Contract bullets (normative summary):**

- **Representation**: the counted v1 claim uses **Kraus bundles** as the primary
  exact representation for fused noisy blocks on bounded support. Cached
  Liouville / superoperator application is allowed only as an internal
  optimization once equivalence to the primary Kraus form is shown on the same
  block.
- **Support matrix**: gate families, noise channels, and partition patterns in
  scope for Phase 3.1 channel-native fusion (may start from Phase 3 surface and
  narrow or extend explicitly).
- **Provisional v1 slice**: keep the Phase 3 primitive gate/noise families, but
  widen fused eligibility to contiguous 1- and 2-qubit mixed motifs on the
  same support, allowing multiple successive gates and multiple interleaved
  local noise insertions inside one fused block. Each eligible block must
  contain at least one noise operation; pure unitary islands remain the Phase 3
  fused path rather than a new Phase 3.1 counted claim.
- **Numerical thresholds**: correctness tolerances consistent with Phase 3
  practice unless a documented stricter policy is required. Equality-style
  invariant residuals use `<= 1e-10`; positivity-style eigenvalue floors use
  `>= -1e-12`.
- **Baseline trio**: sequential `NoisyCircuit`, Phase 3 partitioned+fused path,
  and the relevant Phase 3.1 path:
  - strict `phase31_channel_native` on fully eligible mixed-motif cases,
  - hybrid `phase31_channel_native_hybrid` on counted whole-workload cases.
- **External reference slice**: Qiskit Aer is required on the four counted
  Phase 3.1 mixed-motif microcases and on `phase2_xxz_hea_q4_continuity`; it is
  not required on `q6` continuity or the counted 8/10-qubit performance cases.
- **Workflow anchor**: whether Phase 3.1 evidence reuses the Phase 2 continuity
  anchor and Phase 3 structured families only, or extends them (documented).
- **Mode naming and API surface**: keep planner `requested_mode` as
  `partitioned_density`; preserve the strict motif-proof helper
  `execute_partitioned_density_channel_native(...)` with runtime-path label
  `phase31_channel_native`; add the explicit hybrid whole-workload helper
  `execute_partitioned_density_channel_native_hybrid(...)` with runtime-path
  label `phase31_channel_native_hybrid`; retain
  `execute_partitioned_density_fused` for the Phase 3 unitary-island baseline.
  Strict mode hard-fails on any ineligible partition. Hybrid mode may route only
  Phase-3-supported but Phase-3.1-ineligible partitions through the shipped
  Phase 3 exact path, with emitted route records and no silent downgrade.
- **Evidence schema policy**: extend the existing `correctness_evidence` and
  `performance_evidence` trees with Phase 3.1 fields and required slices rather
  than creating a new top-level evidence family; include strict-vs-hybrid
  runtime identity and partition-route metadata for hybrid counted cases.
- **Host performance build and runtime policy**: which SIMD baseline is required
  for evidence builds. For v1 counted evidence, scalar-only builds are required:
  `build_policy_id = "phase31_scalar_only_v1"`, `build_flavor = "scalar"`,
  `simd_enabled = false`, `tbb_enabled = false`, `thread_count = 1`,
  `counted_claim_build = true`. Any TBB/SIMD variants remain later-branch
  artifacts unless this policy is re-opened.

## 8. Task Breakdown (Goals, Not Implementations)

Tasks are **outcomes**. Engineering tasks and mini-specs are created under each
goal during implementation planning.

Layer 2 mini-spec stubs (trace to §8 tasks and to **P31-C-01..09** in
`PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md`):

| Task | Mini-spec path |
|------|----------------|
| 1 | [`task-1/TASK_1_MINI_SPEC.md`](task-1/TASK_1_MINI_SPEC.md) |
| 2 | [`task-2/TASK_2_MINI_SPEC.md`](task-2/TASK_2_MINI_SPEC.md) |
| 3 | [`task-3/TASK_3_MINI_SPEC.md`](task-3/TASK_3_MINI_SPEC.md) |
| 4 | [`task-4/TASK_4_MINI_SPEC.md`](task-4/TASK_4_MINI_SPEC.md) |
| 5 | [`task-5/TASK_5_MINI_SPEC.md`](task-5/TASK_5_MINI_SPEC.md) |
| 6 | [`task-6/TASK_6_MINI_SPEC.md`](task-6/TASK_6_MINI_SPEC.md) |

**First vertical slice (Layers 3–4):** Behavioral stories and engineering tasks
for Tasks 1–2 plus a minimal Task 3 correctness gate are in
[`FIRST_VERTICAL_SLICE_STORIES_AND_ENGINEERING_TASKS.md`](FIRST_VERTICAL_SLICE_STORIES_AND_ENGINEERING_TASKS.md).

**Second vertical slice (2q / local-support increment):** the next bounded
Tasks 1–2 expansion plus a minimal Task 3 correctness gate for the frozen
2-qubit local-support contract are in
[`SECOND_VERTICAL_SLICE_2Q_LOCAL_SUPPORT_STORIES_AND_ENGINEERING_TASKS.md`](SECOND_VERTICAL_SLICE_2Q_LOCAL_SUPPORT_STORIES_AND_ENGINEERING_TASKS.md).
Bounded slice completion vs. deferred Task 3 / Task 4 work is recorded there under
**Slice implementation status** (not in `PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md`).

**Third vertical slice (strict plus hybrid whole-workload increment):** the next
bounded Tasks 2–4 expansion for the explicit hybrid runtime, one counted hybrid
continuity anchor, and one representative structured pilot case are in
[`THIRD_VERTICAL_SLICE_STRICT_PLUS_HYBRID_WHOLE_WORKLOAD_STORIES_AND_ENGINEERING_TASKS.md`](THIRD_VERTICAL_SLICE_STRICT_PLUS_HYBRID_WHOLE_WORKLOAD_STORIES_AND_ENGINEERING_TASKS.md).
This slice is the first thin end-to-end increment for the strict-plus-hybrid
Phase 3.1 contract and remains smaller than full Task 3 / Task 4 closure.

**Fourth vertical slice (remaining counted correctness closure):** the next
bounded Task 3 expansion for the remaining counted strict microcases, the
counted `q6` hybrid continuity anchor, the required Aer subset, and the
versioned correctness package migration is in
[`FOURTH_VERTICAL_SLICE_REMAINING_COUNTED_CORRECTNESS_STORIES_AND_ENGINEERING_TASKS.md`](FOURTH_VERTICAL_SLICE_REMAINING_COUNTED_CORRECTNESS_STORIES_AND_ENGINEERING_TASKS.md).
This is the recommended **actual next execution slice** after the completed
third slice and stops before the full Task 4 matrix and before publication
closure.

**Fifth vertical slice (full counted performance matrix closure):** the next
bounded Task 4 expansion for the full frozen `P31-ADR-010` matrix, route-aware
row emission, and the matrix-wide decision artifact is in
[`FIFTH_VERTICAL_SLICE_FULL_COUNTED_PERFORMANCE_MATRIX_STORIES_AND_ENGINEERING_TASKS.md`](FIFTH_VERTICAL_SLICE_FULL_COUNTED_PERFORMANCE_MATRIX_STORIES_AND_ENGINEERING_TASKS.md).

**Sixth vertical slice (pre-publication evidence review):** the final
pre-publication Layer 3 / 4 closure increment for the formal review artifact and
go / no-go state is in
[`SIXTH_VERTICAL_SLICE_PRE_PUBLICATION_EVIDENCE_REVIEW_STORIES_AND_ENGINEERING_TASKS.md`](SIXTH_VERTICAL_SLICE_PRE_PUBLICATION_EVIDENCE_REVIEW_STORIES_AND_ENGINEERING_TASKS.md).

**Recommended post-third-slice forward path (Layers 3-4, pre-publication-closure):**
the next tranche should **not** broaden scope. It should close the remaining
Task 3 / Task 4 evidence obligations on the frozen slice and stop once the
publication-closure decision can be made from emitted artifacts rather than
inference.

The execution order, closure-state rubric, and science-first writing workflow
for that final tranche are summarized in
[`CLOSURE_PLAN_PHASE_3_1.md`](CLOSURE_PLAN_PHASE_3_1.md). Treat that document as
the companion playbook for Story `P31-S10` through Task 5; it does not reopen
the frozen contracts in this file.

### Task 1: Fusion representation and IR contract

**Goal:** Define how bounded 1- and 2-qubit channel-native / superoperator
blocks for contiguous mixed gate+noise motifs are represented, serialized in
descriptors or runtime handoff, and validated for exact CPTP semantics.

**Success looks like:** Written contract in ADRs and planning; unsupported cases
explicit; traceability to sequential `NoisyCircuit` semantics.

**Evidence:** Design review record; small-case algebraic or reference checks as
specified in mini-specs.

### Task 2: Runtime integration and eligibility

**Goal:** Integrate the chosen representation into the partitioned noisy runtime
so that:

- fully eligible workloads can execute through the **strict** channel-native
  path without breaking ordering contracts,
- counted whole workloads can execute through an explicit **hybrid** path where
  eligible partitions use channel-native execution and Phase-3-supported but
  Phase-3.1-ineligible partitions use the shipped Phase 3 exact path with route
  attribution.

**Success looks like:** Explicit strict and hybrid execution identities;
hard errors for unsupported strict requests; explicit route metadata and
hard-fail behavior for unsupported-by-both hybrid requests.

**Evidence:** Tests and benchmark hooks as defined in mini-specs.

### Task 3: Correctness evidence package

**Goal:** Emit or extend machine-checkable correctness evidence for Phase 3.1,
including mandatory internal matrix and external micro-validation posture
consistent with Phase 3 culture, using strict mode for counted mixed-motif
microcases and hybrid mode for counted continuity / whole-workload cases.

**Success looks like:** Reproducible bundle with stable IDs; boundary cases
visible.

**Evidence:** `correctness_evidence` (or successor) artifacts; checklist closure.

### Task 4: Performance and diagnosis evidence

**Goal:** Measure runtime and memory versus sequential and Phase 3 fused
baselines on agreed structured families through the explicit hybrid Phase 3.1
path; record diagnosis when gains are absent together with route coverage.

**Success looks like:** Honest performance package; no inflation of Phase 3
claims.

**Evidence:** `performance_evidence` (or successor) artifacts plus a
`break_even_table` / `justification_map` recording where the frozen v1
slice is actually justified relative to Phase 3.

### Task 5: Publication and claim boundary

**Goal:** Align short paper, narrative, abstract, and full paper skeletons with
delivered evidence; map the Side Paper A question in `PUBLICATIONS.md` (verbatim
in `task-5/TASK_5_MINI_SPEC.md`) to concrete, evidence-backed answers.

**Success looks like:** Claim boundary section matches emitted bundles;
reviewer-facing limitations explicit.

**Evidence:** Updated paper docs; traceability table in this file; closure-state
decision and writing sequence recorded in `CLOSURE_PLAN_PHASE_3_1.md`.

### Task 6: C++ hot-path offload and host parallel / SIMD evaluation (later branch)

**Not in v1 gate:** Mandatory evidence uses **scalar-only** builds; **no TBB/SIMD**
rows in required bundles (**P31-C-09** closes without Task 6). See
`task-6/TASK_6_MINI_SPEC.md`.

**Goal (when branch opens):** Identify dominant Python and pybind overhead on
the Phase 3.1 and Phase 3 fused paths; implement or prototype **C++-resident**
hot paths where profiling justifies the move; evaluate **TBB** and **AVX** (or
project-standard SIMD) under the same correctness slice as channel-native work.

**Success looks like:** Documented profile deltas; reproducible build/run
metadata; optional speedups or an honest “kernel not the bottleneck” diagnosis
on the same representative workloads as Task 4.

**Evidence:** Profiler captures, A/B benchmark rows, and explicit note of SIMD
and threading settings in emitted performance bundles (optional / amended
**P31-C-09** only).

### Recommended Layer 3 / 4 Execution Path From The Current State

The stories below describe the recommended behavioral path from the current
implementation state **up to but excluding** the final publication-closure
decision. They preserve the frozen support surface, thresholds, and counted
case inventory.

### Story P31-S10: The remaining counted correctness slice closes under the frozen strict-plus-hybrid contract

**User/Research value**

- Converts Phase 3.1 exactness from a strong partial result into a full frozen
  counted slice.
- Prevents Task 4 performance interpretation from resting on an incompletely
  validated support surface.

**Given / When / Then**

- **Given** the strict motif-proof slice, the hybrid `q4` continuity anchor, and
  the thin structured pilot already implemented.
- **When** the remaining counted correctness rows and required external
  micro-validation rows run through the Phase 3.1 correctness harness.
- **Then** every frozen correctness case is either green with stable bundle
  metadata or explicitly recorded as a blocking failure on the frozen slice.

**Scope**

- **In:** the remaining strict counted 2-qubit microcases, counted hybrid
  `phase2_xxz_hea_q6_continuity`, the required Aer subset, and the
  `correctness_evidence` / `channel_invariants` / `partition_route_summary`
  package migration.
- **Out:** the structured performance matrix, publication framing, and Task 6.

**Acceptance signals**

- `phase31_microcase_2q_multi_noise_entangler_chain` passes through the strict
  path under §10.1.
- `phase31_microcase_2q_dense_same_support_motif` passes through the strict
  path under §10.1.
- `phase2_xxz_hea_q6_continuity` passes through the hybrid path with stable
  route-summary assertions.
- The Aer subset from §10.4 is emitted with versioned schema and stable case
  identifiers.
- The counted correctness bundle contains the required Phase 3.1 fields from
  §10.6.

**Traceability**

- Phase: §6, §8 Task 3, §10.1, §10.2, §10.4, §10.6.
- ADRs: `P31-ADR-003`, `P31-ADR-008`, `P31-ADR-009`, `P31-ADR-011`,
  `P31-ADR-013`.

#### Engineering tasks (Story P31-S10)

##### Engineering Task P31-S10-E01: Promote the remaining counted strict microcases and `q6` hybrid continuity row into claim-bearing correctness gates

**Implements story**

- Story P31-S10

**Change type**

- code | tests | validation automation

**Definition of done**

- The remaining counted strict microcases and the counted `q6` hybrid continuity
  row are green in reproducible pytest or harness form.
- Hybrid `q6` route-summary assertions are stable and reviewer-auditable.

**Execution checklist**

- [ ] Promote `phase31_microcase_2q_multi_noise_entangler_chain` to a
      claim-bearing exactness gate.
- [ ] Promote `phase31_microcase_2q_dense_same_support_motif` to a
      claim-bearing exactness gate.
- [ ] Add the counted `phase2_xxz_hea_q6_continuity` hybrid exactness gate with
      route-summary assertions.
- [ ] Record explicit blockers if any counted correctness row fails.

**Evidence produced**

- Passing tests or harness outputs for the remaining counted correctness rows.
- Updated case inventory or manifest inputs tied to the frozen IDs.

**Risks / rollback**

- Risk: `q6` continuity exposes a routing or scale behavior that invalidates the
  current partial-confidence boundary.
- Rollback/mitigation: keep the failure explicit and do not widen the support
  surface to compensate.

##### Engineering Task P31-S10-E02: Emit the versioned correctness package with Aer rows and Phase 3.1 slices

**Implements story**

- Story P31-S10

**Change type**

- validation automation | docs

**Definition of done**

- The required Aer subset is emitted for the frozen external slice only.
- The versioned `correctness_evidence` successor includes the required Phase
  3.1 fields plus `channel_invariants` and `partition_route_summary`.
- A stable Case ID -> internal pass -> Aer pass summary exists for review.

**Execution checklist**

- [ ] Wire the required Aer rows from §10.4 into the correctness package.
- [ ] Version-bump the package or case schema where Phase 3.1 fields are added.
- [ ] Emit `channel_invariants` for the counted strict microcases.
- [ ] Emit `partition_route_summary` for the counted hybrid rows.

**Evidence produced**

- Regeneratable correctness bundle for the full frozen counted correctness
  surface.
- Review-ready case summary table aligned to the same frozen IDs.

**Risks / rollback**

- Risk: partial schema migration obscures which rows are truly Phase 3.1-counted.
- Rollback/mitigation: prefer explicit successor schema names over silent
  reuse of Phase 3 package shapes.

### Story P31-S11: The frozen performance matrix yields route-aware decision rows

**User/Research value**

- Answers the whole-workload "when does channel-native fusion matter?" question
  on the frozen slice rather than on one pilot row.
- Separates local exactness success from workload-level justification.

**Given / When / Then**

- **Given** the mandatory counted correctness slice is green for rows that carry
  counted performance claims, or explicit blockers are recorded and the matrix
  is being run in diagnosis-only mode pending closure.
- **When** the full `P31-ADR-010` counted performance matrix runs through the
  explicit hybrid Phase 3.1 harness.
- **Then** each counted row records the baseline trio, route coverage, and a
  measured decision classification, with no publication framing yet attached.

**Scope**

- **In:** all remaining primary-family and control rows from §10.3, scalar-build
  metadata, route-coverage emission, and the required
  `break_even_table` / `justification_map`.
- **Out:** Task 5 publication updates and Task 6 host-acceleration work.

**Acceptance signals**

- All frozen counted rows from §10.3 are emitted with stable IDs.
- Each row records sequential, Phase 3 fused, and Phase 3.1 hybrid timings.
- Each row records hybrid route coverage and `decision_class`.
- The matrix emits a `break_even_table` or `justification_map` tied to the same
  case IDs.

**Traceability**

- Phase: §6, §8 Task 4, §10.3, §10.6, §10.7.
- ADRs: `P31-ADR-005`, `P31-ADR-006`, `P31-ADR-010`, `P31-ADR-013`,
  `P31-ADR-014`.

#### Engineering tasks (Story P31-S11)

##### Engineering Task P31-S11-E01: Expand the current hybrid pilot harness into the full frozen counted matrix

**Implements story**

- Story P31-S11

**Change type**

- benchmark harness | tests

**Definition of done**

- The current pilot runner expands to the full counted matrix from §10.3.
- Each emitted row carries the baseline trio and scalar-build metadata required
  by §10.7.

**Execution checklist**

- [ ] Add the remaining primary-family rows from `phase31_pair_repeat` and
      `phase31_alternating_ladder`.
- [ ] Add the control-family `layered_nearest_neighbor` rows.
- [ ] Preserve the frozen case IDs, seeds, patterns, and qubit bands from §10.3.
- [ ] Guard the inventory and iterator helpers with regression tests.

**Evidence produced**

- Reproducible benchmark rows for the full frozen counted matrix.
- Regression tests protecting the counted matrix inventory.

**Risks / rollback**

- Risk: harness expansion drifts from the frozen slice and reintroduces
  selection bias.
- Rollback/mitigation: derive rows from the frozen inventory and test the exact
  counts.

##### Engineering Task P31-S11-E02: Emit the route-aware decision artifact and machine-readable matrix summary

**Implements story**

- Story P31-S11

**Change type**

- validation automation | docs

**Definition of done**

- The matrix emits `break_even_table` / `justification_map` directly from the
  measured rows.
- A machine-readable summary table exists for later publication use without
  requiring manual reconstruction.

**Execution checklist**

- [ ] Emit `decision_class` for every counted performance row.
- [ ] Emit route-coverage counters and partition-level route records.
- [ ] Build the `break_even_table` / `justification_map` from measured results.
- [ ] Record explicit negative or mixed outcomes without rewriting the frozen
      success rule.

**Evidence produced**

- Full performance bundle with decision artifact.
- Review-ready case table: Case ID -> baselines -> route coverage -> decision.

**Risks / rollback**

- Risk: one favorable row is mistaken for matrix closure.
- Rollback/mitigation: keep the matrix-level artifact mandatory and include the
  control family in the same summary.

### Story P31-S12: A pre-closure evidence summary makes the publication decision mechanical

**User/Research value**

- Prevents ad hoc interpretation after the bundles land.
- Separates evidence closure from publication framing and venue strategy.

**Given / When / Then**

- **Given** the Task 3 and Task 4 artifacts exist for the frozen counted slice.
- **When** the pre-closure review runs.
- **Then** the phase records whether the evidence is
  `positive-methods-ready`, `decision-study-ready`, or `not-ready-yet`, with
  explicit blockers and without yet executing Task 5 publication closure.

**Scope**

- **In:** a concise phase-local evidence summary, blocker inventory, and status
  rubric tied to the emitted bundles.
- **Out:** actual paper rewrites, venue-specific framing, and final publication
  choice.

**Acceptance signals**

- One review table maps correctness coverage, performance coverage, and open
  blockers across the frozen slice.
- The phase-local summary explicitly states that publication closure is
  downstream of this evidence review.
- No new scientific claims are added beyond the emitted artifacts.

**Traceability**

- Phase: §6, §9, §10, §13.
- ADRs: `P31-ADR-005`, `P31-ADR-006`, `P31-ADR-013`.

#### Engineering tasks (Story P31-S12)

##### Engineering Task P31-S12-E01: Produce the pre-publication evidence review pack and go/no-go summary

**Implements story**

- Story P31-S12

**Change type**

- docs | validation automation

**Definition of done**

- One reviewer-facing summary exists for the full frozen counted slice.
- The summary records one of the three pre-closure states:
  `positive-methods-ready`, `decision-study-ready`, or `not-ready-yet`.
- Task 5 publication updates remain explicitly downstream of this summary.

**Execution checklist**

- [ ] Summarize correctness-slice coverage and blockers from Story P31-S10.
- [ ] Summarize performance-matrix coverage and decision artifacts from Story P31-S11.
- [ ] Record the pre-closure state without changing publication documents.
- [ ] Link the summary to the emitted bundle locations or stable builder entrypoints.

**Evidence produced**

- One pre-closure evidence summary suitable for phase review.
- Explicit go/no-go input for the later publication-closure decision.

**Risks / rollback**

- Risk: the status summary drifts into publication argument instead of evidence review.
- Rollback/mitigation: keep it artifact-indexed, case-ID-indexed, and
  explicitly pre-publication.

### Recommended implementation order after the third slice

1. `P31-S10-E01`
2. `P31-S10-E02`
3. `P31-S11-E01`
4. `P31-S11-E02`
5. `P31-S12-E01`

**Stop condition before publication closure**

Pause before Task 5 once Story P31-S12 is complete. At that point the phase
should have enough evidence to decide between the §6 / §13 closure modes
without broadening scope or inferring from partial rows.

## 9. Full-Phase Acceptance Criteria

- Phase 3.1 support matrix and representation choices are frozen and auditable.
- At least one real channel-native or superoperator-native execution mode is
  implemented and validated on the mandatory slice, and the whole-workload
  counted surface uses an explicit hybrid runtime contract rather than an
  undocumented downgrade, **or** a benchmark-grounded report documents why the
  branch does not proceed.
- No silent semantic fallback for modes that advertise channel-native fusion.
- Publication documents updated to match evidence; no contradiction with Phase 3
  closed claims.

## 10. Validation and Benchmark Matrix

Reuse the Phase 3 evaluation philosophy:

- **Internal baseline:** sequential `NoisyCircuit` density execution.
- **Regression / comparison:** Phase 3 partitioned runtime with unitary-island
  fusion where applicable.
- **External reference:** Qiskit Aer (or successor agreed in checklist) on the
  frozen micro-validation slice.
- **Phase 3.1 execution policy:** strict `phase31_channel_native` on counted
  mixed-motif microcases; explicit hybrid `phase31_channel_native_hybrid` on
  counted continuity and structured whole-workload rows.
- **Dimensions:** qubit count bands, noise placement (sparse / periodic / dense
  as relevant), locality, and fusion granularity.

### 10.1 Frozen Numeric Policy (`P31-C-03`)

Phase 3.1 adopts the following counted-claim thresholds:

- internal exactness versus sequential `NoisyCircuit`:
  - maximum Frobenius-norm density difference `<= 1e-10`,
- external exactness versus Qiskit Aer on the external slice:
  - maximum Frobenius-norm density difference `<= 1e-10`,
- state validity:
  - `|Tr(rho) - 1| <= 1e-10`,
  - `rho.is_valid(tol=1e-10)`,
- continuity-anchor observable agreement where continuity cases are counted:
  - maximum absolute energy error `<= 1e-8`,
- representation-level invariants:
  - equality-style residuals `<= 1e-10`,
  - positivity-style eigenvalue floors `>= -1e-12`.

### 10.2 Frozen Counted Correctness Slice (`P31-C-04`)

The counted Phase 3.1 correctness slice is:

- v1 mixed-motif microcases:
  - `phase31_microcase_1q_u3_local_noise_chain`,
  - `phase31_microcase_2q_cnot_local_noise_pair`,
  - `phase31_microcase_2q_multi_noise_entangler_chain`,
  - `phase31_microcase_2q_dense_same_support_motif`,
- bounded continuity anchors:
  - `phase2_xxz_hea_q4_continuity`,
  - `phase2_xxz_hea_q6_continuity`,
- and every counted performance case in §10.3.

Interpretation:

- the counted correctness slice is the claim-bearing Phase 3.1 v1 surface,
- counted mixed-motif microcases run through the **strict**
  `phase31_channel_native` path,
- counted continuity anchors and counted performance-carry-forward rows run
  through the explicit **hybrid** `phase31_channel_native_hybrid` path,
- inherited Phase 3 microcases outside this set may remain regression or
  unsupported-boundary checks, but they are not part of the counted positive
  methods claim unless explicitly promoted later.

### 10.3 Frozen Counted Performance Slice (`P31-C-06`)

The counted Phase 3.1 performance slice is:

- primary positive-method families:
  - `phase31_pair_repeat`,
  - `phase31_alternating_ladder`,
- control family:
  - `layered_nearest_neighbor`,
- qubit counts:
  - `8`, `10`,
- noise patterns:
  - primary families use `periodic` and `dense`,
  - control family uses `sparse`,
- seeds:
  - primary families use `20260318`, `20260319`, `20260320`,
  - control family uses `20260318`.

Counted-case inventory:

- primary positive-method counted cases:
  - `2 × 2 × 2 × 3 = 24`,
- control counted cases:
  - `1 × 2 × 1 × 1 = 2`.

Positive-method success rule:

- at least one representative counted `8`- or `10`-qubit case from the primary
  families must show either:
  - median wall-clock speedup `>= 1.2x` versus the Phase 3 fused baseline, or
  - peak-memory reduction `>= 15%` versus the Phase 3 fused baseline,
  - with no correctness loss under §10.1.

Every counted performance record must also include:

- sequential `NoisyCircuit`,
- Phase 3 partitioned+fused,
- Phase 3.1 channel-native hybrid,
- and a `break_even_table` / `justification_map` classification:
  - `phase3_sufficient`,
  - `phase31_justified`,
  - `phase31_not_justified_yet`.

### 10.4 Frozen External Reference Slice (`P31-C-05`)

Qiskit Aer density-matrix simulation is the required external exact reference
on the following counted Phase 3.1 slice:

- `phase31_microcase_1q_u3_local_noise_chain`,
- `phase31_microcase_2q_cnot_local_noise_pair`,
- `phase31_microcase_2q_multi_noise_entangler_chain`,
- `phase31_microcase_2q_dense_same_support_motif`,
- `phase2_xxz_hea_q4_continuity`.

Execution interpretation:

- the four `phase31_microcase_*` cases use the strict
  `phase31_channel_native` path,
- `phase2_xxz_hea_q4_continuity` uses the explicit hybrid
  `phase31_channel_native_hybrid` path.

Not required externally for the counted claim:

- `phase2_xxz_hea_q6_continuity`,
- counted 8- and 10-qubit performance cases,
- optional Task 6 build variants.

### 10.5 Frozen Mode Naming And API Surface (`P31-C-07`)

Phase 3.1 keeps the planner and descriptor entry contract:

- `requested_mode = "partitioned_density"` (`PARTITIONED_DENSITY_MODE`)

and adds two explicit Phase 3.1 execution identities:

- **strict** motif-proof path:
  - convenience helper:
    - `execute_partitioned_density_channel_native(...)`,
  - runtime-path label:
    - `phase31_channel_native`,
- **hybrid** whole-workload path:
  - convenience helper:
    - `execute_partitioned_density_channel_native_hybrid(...)`,
  - runtime-path label:
    - `phase31_channel_native_hybrid`.

Error-policy freeze:

- existing exception families remain:
  - `NoisyPlannerValidationError`,
  - `NoisyRuntimeValidationError`,
- existing high-level runtime categories remain:
  - `runtime_request`,
  - `descriptor_to_runtime_mismatch`,
  - `unsupported_runtime_operation`,
  - `unsupported_runtime_execution`,
- Phase 3.1 extends the `first_unsupported_condition` vocabulary with:
  - `channel_native_support_surface`,
  - `channel_native_noise_presence`,
  - `channel_native_qubit_span`,
  - `channel_native_representation`,
  - `channel_native_invariant_failure`,
  - `channel_native_runtime_execution`.

Hybrid-route policy freeze:

- strict `phase31_channel_native` keeps the existing no-fallback contract:
  any ineligible partition is a hard error,
- hybrid `phase31_channel_native_hybrid` may route a partition to the shipped
  Phase 3 exact path only when the partition is Phase-3-supported but
  Phase-3.1-ineligible for a documented support-surface reason,
- representation, invariant, or runtime-execution failures on a partition that
  was supposed to execute channel-natively remain hard errors rather than
  reroute conditions.

Required hybrid route reasons:

- `eligible_channel_native_motif`,
- `pure_unitary_partition`,
- `channel_native_noise_presence`,
- `channel_native_qubit_span`,
- `channel_native_support_surface`.

### 10.6 Frozen Evidence Packaging Policy (`P31-C-08`)

Phase 3.1 extends the existing evidence families:

- `benchmarks/density_matrix/artifacts/correctness_evidence/`
- `benchmarks/density_matrix/artifacts/performance_evidence/`

Required counted-case metadata:

- `claim_surface_id = "phase31_bounded_mixed_motif_v1"`,
- `runtime_class`,
- `representation_primary = "kraus_bundle"`,
- `fused_block_support_qbits`,
- `contains_noise`,
- `counted_phase31_case`.

Required additions under `correctness_evidence`:

- `channel_invariants` slice,
- invariant summary fields tied to the chosen primary representation,
- `partition_route_summary` slice for hybrid-counted cases.

Required additions under `performance_evidence`:

- `break_even_table` slice,
- `decision_class ∈ {phase3_sufficient, phase31_justified, phase31_not_justified_yet}`,
- explicit metrics relative to the Phase 3 fused baseline,
- route-coverage metadata for hybrid-counted cases, including:
  - `channel_native_partition_count`,
  - `phase3_routed_partition_count`,
  - `channel_native_member_count`,
  - `phase3_routed_member_count`.

Required runtime / route vocabularies:

- `runtime_class ∈ {"plain_partitioned_baseline", "phase3_unitary_island_fused",
  "phase31_channel_native", "phase31_channel_native_hybrid"}`,
- `partition_runtime_class ∈ {"phase31_channel_native",
  "phase3_unitary_island_fused", "phase3_supported_unfused"}`,
- `partition_route_reason` uses the frozen reasons in §10.5.

Versioning rule:

- any case or package record that gains required Phase 3.1 fields must use a
  version-bumped successor schema rather than silently reusing the old Phase 3
  schema string.

### 10.7 Frozen Host Build Policy (`P31-C-09`)

The counted Phase 3.1 v1 build policy is:

- `build_policy_id = "phase31_scalar_only_v1"`,
- `build_flavor = "scalar"`,
- `simd_enabled = false`,
- `tbb_enabled = false`,
- `thread_count = 1`,
- `counted_claim_build = true`.

Interpretation:

- counted correctness and performance bundles must record these fields,
- TBB/SIMD evidence is non-counted for v1 and belongs only to the Task 6 later
  branch,
- any future counted host-acceleration claim requires re-opening this policy.

### 10.8 Evidence-Pipeline Migration Policy (Default Phase 3.1, Explicit Phase 3 Legacy)

Phase 3.1 adopts the following staged migration for bundle builders and default
validation pipelines:

- **Stage A:** implement Phase 3.1 sibling builders and pipelines without
  mutating the current Phase 3 defaults,
- **Stage B:** once the Phase 3.1 flow is validated, switch the default bundle
  builders and `validation_pipeline.py` entrypoints to the Phase 3.1 surfaces.

Historical Phase 3 evidence remains available through **explicit legacy
scripts/functions**, not flags. Target naming pattern:

- bundle builders:
  - `build_phase3_correctness_package_payload()`,
  - `build_phase3_performance_benchmark_package_payload()`,
- validation pipelines:
  - `phase3_validation_pipeline.py` in each evidence tree (or equivalent
    explicit legacy module names).

This policy keeps “default” aligned with the current scientific frontier while
preserving reproducible access to the historical Phase 3 package.

## 11. Risks and Decision Gates

| Risk | Mitigation |
|---|---|
| Validation explosion for superoperator fusion | Start with minimal support; explicit unsupported tiers; reuse Phase 3 evidence machinery |
| Performance still dominated by Python or memory traffic | Profile early; scope claims to evidence; use §1.4 offload candidates and Task 6 |
| Threading (TBB) introduces nondeterminism or flaky benchmarks | Prefer deterministic sequential reference path; parallel modes opt-in with fixed thread counts and documented numeric policy |
| SIMD (AVX) alignment or dtype mismatches weaken correctness | Gate SIMD behind tests on the mandatory slice; fall back to scalar reference in harness when comparing |
| Scope creep into Phase 4 | P31-ADR-002; phase-level reviews |
| Confusion with Paper 2 | Phase 3.1 claims are separate; cross-link limitations |

**Gate P31-G-1:** Pre-implementation checklist closed (see companion document).

**Gate P31-G-2:** Mandatory correctness slice green before performance claims.

## 12. Non-Goals (Summary)

See §4. Repeated here for acceptance rubrics: no Phase 4 workflow expansion, no
approximate scaling thesis, no retroactive change to Phase 3 minimum deliverable
meaning.

## 13. Expected Outcome

Either (a) a defensible **follow-on methods contribution** showing where
channel-native fusion helps exact noisy simulation on training-relevant
workloads, or (b) a **decision-study outcome** that documents negative or mixed
results with the same rigor, preserving the exact-density scientific anchor.
