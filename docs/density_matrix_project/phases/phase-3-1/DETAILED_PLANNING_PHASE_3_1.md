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

What remains intentionally open at the start of Phase 3.1 (closure targets for
this phase):

- a **primary** exact representation for fused CPTP blocks and its descriptor /
  runtime handoff,
- **eligibility and unsupported** rules for channel-native fusion beyond
  unitary-island fusion,
- frozen **support matrix**, **numeric thresholds**, **mandatory correctness and
  performance case sets**, **evidence bundle schema**, and **mode / API naming**
  for Phase 3.1 modes,
- optional **host-side** C++/TBB/AVX policy for evidence builds when Task 6 is
  in play.

## 0.2 Spec-Driven Development Principles for Phase 3.1

1. Close phase-wide representation, support matrix, and evidence contracts before
   implementation details drift (checklist Gate P31-G-1).
2. Separate **required** channel-native behavior from **unsupported** tiers;
   no silent fallback for modes that advertise channel-native fusion
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

### Novelty hypothesis and primary publication stance

Phase 3.1's candidate novelty is **not** the mere use of Kraus, PTM, or
Liouville representations in isolation; those are established open-system
tools. The intended claim is narrower and stronger: SQUANDER can execute
**bounded exact channel-native fusion of mixed gate+noise motifs inside a
partitioned noisy runtime** while preserving the canonical ordered
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
  fused into channel-native blocks, which remain sequential or unitary-island
  fused only, and how failures surface (no silent fallback for claimed
  partitioned-channel-native modes).
- **Correctness evidence**: exact agreement (within frozen numerical thresholds)
  with sequential `NoisyCircuit` density execution; explicit comparison or
  regression posture versus the Phase 3 partitioned path where both exist.
- **Performance evidence**: structured comparison versus Phase 3 fused baseline
  and sequential baseline on workloads that stress noise density and locality,
  with honest reporting of wins, ties, and regressions.
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
  the frozen positive-method threshold in `P31-ADR-010`.
- If the targeted slice fails to show such a benefit, the phase may still close
  through a **benchmark-justified negative result** explaining why the richer
  fusion class does not pay off under the contract.
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
  and Phase 3.1 path (where applicable) for performance narratives.
- **External reference slice**: Qiskit Aer is required on the four counted
  Phase 3.1 mixed-motif microcases and on `phase2_xxz_hea_q4_continuity`; it is
  not required on `q6` continuity or the counted 8/10-qubit performance cases.
- **Workflow anchor**: whether Phase 3.1 evidence reuses the Phase 2 continuity
  anchor and Phase 3 structured families only, or extends them (documented).
- **Mode naming and API surface**: keep planner `requested_mode` as
  `partitioned_density`; expose the counted Phase 3.1 path through
  `execute_partitioned_density_channel_native(...)` and runtime-path label
  `phase31_channel_native`, while retaining `execute_partitioned_density_fused`
  for the Phase 3 unitary-island baseline.
- **Evidence schema policy**: extend the existing `correctness_evidence` and
  `performance_evidence` trees with Phase 3.1 fields and required slices rather
  than creating a new top-level evidence family.
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
so that eligible regions execute through the new path without breaking ordering
contracts.

**Success looks like:** Executable path behind explicit flags or modes;
hard errors for unsupported channel-native requests when that mode is selected.

**Evidence:** Tests and benchmark hooks as defined in mini-specs.

### Task 3: Correctness evidence package

**Goal:** Emit or extend machine-checkable correctness evidence for Phase 3.1,
including mandatory internal matrix and external micro-validation posture
consistent with Phase 3 culture.

**Success looks like:** Reproducible bundle with stable IDs; boundary cases
visible.

**Evidence:** `correctness_evidence` (or successor) artifacts; checklist closure.

### Task 4: Performance and diagnosis evidence

**Goal:** Measure runtime and memory versus sequential and Phase 3 fused
baselines on agreed structured families; record diagnosis when gains are absent.

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

**Evidence:** Updated paper docs; traceability table in this file.

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

## 9. Full-Phase Acceptance Criteria

- Phase 3.1 support matrix and representation choices are frozen and auditable.
- At least one real channel-native or superoperator-native execution mode is
  implemented and validated on the mandatory slice, **or** a benchmark-grounded
  report documents why the branch does not proceed.
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
- Phase 3.1 channel-native,
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

Not required externally for the counted claim:

- `phase2_xxz_hea_q6_continuity`,
- counted 8- and 10-qubit performance cases,
- optional Task 6 build variants.

### 10.5 Frozen Mode Naming And API Surface (`P31-C-07`)

Phase 3.1 keeps the planner and descriptor entry contract:

- `requested_mode = "partitioned_density"` (`PARTITIONED_DENSITY_MODE`)

and adds a distinct execution identity for the counted v1 path:

- convenience helper:
  - `execute_partitioned_density_channel_native(...)`,
- runtime-path label:
  - `phase31_channel_native`.

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
- invariant summary fields tied to the chosen primary representation.

Required additions under `performance_evidence`:

- `break_even_table` slice,
- `decision_class ∈ {phase3_sufficient, phase31_justified, phase31_not_justified_yet}`,
- explicit metrics relative to the Phase 3 fused baseline.

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
