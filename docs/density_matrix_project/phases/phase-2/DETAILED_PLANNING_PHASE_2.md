# Detailed Planning for Phase 2

This document is the Phase 2 source of truth for scope, task goals,
acceptance criteria, validation expectations, and research-facing deliverables.

Primary Phase 2 theme:

> turn the density-matrix module from a standalone exact noisy simulator into a
> usable, validated backend for noisy variational workflows.

This is a specification document, not an implementation log.

## 0.1 Revalidation After Phase 2 Tasks 1-6

The Phase 2 contract remains valid after implementation of Tasks 1 to 6. The
implemented slice now closes the frozen integrated-backend, validation, and
workflow-demonstration baseline for backend selection, observable evaluation,
bridge behavior, required local-noise validation, and the phase-level
publication-facing evidence package.

Implementation-backed status at this point:

- explicit backend selection is implemented on the VQE-facing API, with default
  `state_vector`, explicit `density_matrix`, and deterministic hard-error
  unsupported behavior,
- the guaranteed density path is generated `HEA` only, not broad manual circuit
  parity,
- the exact observable path is real-valued Hermitian energy evaluation through
  the existing sparse-Hamiltonian interface,
- the VQE-facing density-noise surface is ordered fixed local noise insertion
  for required models, not a general parametric noise-program interface,
- mandatory validation evidence now includes:
  - 1 to 3 qubit required micro-validation matrix,
  - one canonical workflow contract with stable workflow identity and explicit
    input/output fields, threshold metadata, deterministic parameter/trace
    policy metadata, and required unsupported-case field inventory,
  - required end-to-end 4 and 6 qubit workflow cases plus one bounded 4-qubit
    optimization trace with recorded runtime and peak memory,
  - 4 / 6 / 8 / 10 qubit required workflow matrix with 10 fixed parameter
    vectors per required size,
  - structured optional, deferred, and unsupported boundary evidence,
- and a complete workflow-facing publication bundle is archived at
  `benchmarks/density_matrix/artifacts/workflow_evidence/` with
  `workflow_publication_bundle.json` as the top-level manifest, linking the
  canonical workflow contract, end-to-end trace bundle, matrix baseline bundle,
  unsupported-workflow bundle, and interpretation bundle while preserving
  traceability to the underlying validation-evidence layers and now validating
  lower-story semantic closure in addition to artifact presence, expected
  status, and cross-artifact workflow identity.

This document therefore continues to describe the full Phase 2 contract, while
the notes below clarify the intentional scientific boundaries that remain in
place for later phases.

## 0.2 Learnings From Implemented Tasks

Phase 2 implementation produced several process and contract learnings that are
now part of the validation posture:

- hard-error boundaries at Python normalization, C++ density-noise
  specification checks, and density-anchor preflight are required to prevent
  silent fallback behavior,
- explicit support-tier vocabulary (`required`, `optional`, `deferred`,
  `unsupported`) is essential for keeping mandatory-baseline claims auditable,
- the workflow-scale exact-regime matrix should close independently of the
  bounded optimization trace, with the trace and documented 10-qubit anchor
  packaged as a separate evidence layer,
- machine-readable artifact manifests with stable case IDs and status checks are
  required for publication-facing reproducibility,
- lower evidence layers should consume canonical workflow-contract metadata such
  as thresholds, required qubit inventories, canonical trace identity, and
  unsupported-case field requirements instead of re-declaring them independently,
- phase-level metric-completeness and interpretation-guardrail bundles are
  required to prevent partial, optional, unsupported, or malformed evidence from
  inflating the main claim,
- only mandatory, complete, supported evidence may close the main Phase 2
  workflow claim; optional evidence remains supplemental and deferred or
  unsupported evidence remains boundary-only,
- whole-register depolarizing remains the only delivered optional workflow-
  facing noise baseline and must not be treated as equivalent to the required
  local-noise package,
- runtime and peak-memory must be recorded for mandatory workflow evidence even
  though they are not Phase 2 pass/fail thresholds.

## 1. Purpose

Phase 2 exists to bridge the current gap between:

- the standalone exact mixed-state backend already delivered in Phase 1, and
- the noisy training workflows required by the broader PhD research program on
  scalable training under realistic noise models.

Phase 2 is the first phase where the density-matrix work becomes a complete
scientific instrument rather than only an enabling module.

## 2. Source-of-Truth Hierarchy

If multiple existing documents overlap, interpret them in the following order of
authority for Phase 2.

### Tier 1: Strategic Planning Constraints

These documents define the authoritative Phase 2 intent and trade-off
boundaries:

- `docs/density_matrix_project/planning/PLANNING.md`
- `docs/density_matrix_project/planning/ADRs.md`
- `docs/density_matrix_project/planning/PUBLICATIONS.md`
- `docs/density_matrix_project/planning/REFERENCES.md`

### Tier 2: Existing Project Roadmap and Milestone Definitions

These documents define accepted milestone wording and must remain consistent
with the Phase 2 plan:

- `docs/density_matrix_project/CHANGELOG.md`
- `docs/density_matrix_project/RESEARCH_ALIGNMENT.md`

### Tier 3: Architecture and Existing Module Baseline

These documents describe the current delivered system and named integration
targets:

- `docs/density_matrix_project/ARCHITECTURE.md`
- `docs/density_matrix_project/README.md`
- `docs/density_matrix_project/SETUP.md`
- `docs/density_matrix_project/phases/phase-1/API_REFERENCE_PHASE_1.md`

### Tier 4: Legacy or Supportive Context

These sources are informative, but do not override the above:

- older top-level references to flat API documentation,
- existing branch-specific comments in examples or benchmark notes,
- exploratory planning notes that are not reflected in the Phase 2 planning set.

## 3. Traceability Matrix

| Requirement area | Upstream source | Phase 2 interpretation |
|---|---|---|
| Backend selection | `CHANGELOG.md`, `RESEARCH_ALIGNMENT.md`, `PLANNING.md` | Phase 2 must define and validate a selectable density-matrix execution path |
| Exact `Re Tr(H*rho)` Hermitian-energy path | `ARCHITECTURE.md`, `CHANGELOG.md`, `RESEARCH_ALIGNMENT.md`, `PUBLICATIONS.md` | Phase 2 must deliver validated exact real-valued Hermitian energy evaluation for the supported noisy VQE workflow |
| Bridge from the generated anchor VQE circuit into density execution | `ARCHITECTURE.md`, `CHANGELOG.md`, `PLANNING.md` | Phase 2 must specify how the generated `HEA` anchor workflow reaches the density backend without requiring Phase 3 partitioning/fusion or broad manual-circuit parity |
| Realistic local noise | `ADRs.md`, `PLANNING.md`, `PUBLICATIONS.md` | Phase 2 prioritizes local noise models required for first noisy training studies |
| Exact dense density matrices as reference | `ADRs.md`, `PLANNING.md` | Phase 2 remains exact-first and does not substitute approximate methods for the core deliverable |
| No density-aware partitioning/fusion yet | `PLANNING.md`, `ADRs.md` | Phase 2 explicitly excludes Phase 3 acceleration work from the core commitment |
| Publication readiness | `PUBLICATIONS.md`, `PLANNING.md` | Phase 2 must be documented and validated in a way that directly supports Paper 1 |

## 4. Spec-Driven Development Principles for Phase 2

Phase 2 should follow spec-driven development in the following sense:

1. Define contracts, scope boundaries, and success criteria before any
   implementation decisions are treated as fixed.
2. Separate required behavior from possible implementation choices.
3. Maintain explicit traceability from milestone goals to validation evidence.
4. Treat unsupported cases and deferred work as first-class documented outcomes.
5. Use publication evidence requirements to guide what counts as done.
6. Keep task descriptions goal-oriented rather than implementation-prescriptive.

In practice, this means Phase 2 documentation should answer:

- what must be true when Phase 2 ends,
- what is explicitly not part of Phase 2,
- how correctness and scientific usefulness will be demonstrated,
- and what evidence is required before Phase 2 can be declared complete.

## 5. Mission Statement

Phase 2 delivers an exact noisy backend integration layer for SQUANDER's
variational workflows, including:

- backend selection,
- exact noisy Hermitian-energy evaluation via `Re Tr(H*rho)`,
- generated-`HEA` circuit-to-density-backend bridging for the anchor workflow,
- realistic local noise support needed for initial studies,
- and validation strong enough to support the first major paper.

## 6. In Scope

The following are in scope for Phase 2.

### 6.1 Backend Integration

- A documented density-matrix backend path that can be selected within the
  Phase 2 anchor variational workflow, with
  `qgd_Variational_Quantum_Eigensolver_Base` as the required entry point.
- A clear user-facing and developer-facing contract for how backend choice is
  expressed and interpreted.
- Broader reuse in decomposition or other optimization interfaces is desirable,
  but it is not part of the Phase 2 completion gate.

### 6.2 Exact Noisy Observable Evaluation

- A documented and validated exact real-valued Hermitian-energy path for
  `Re Tr(H*rho)`.
- Support for that observable path sufficient for the canonical noisy XXZ VQE
  workflow.

### 6.3 Circuit and Representation Bridging

- A documented partial bridge from the generated default `HEA` circuit into the
  density-matrix execution path.
- Defined behavior for supported, unsupported, and no-fallback cases.
- A guarantee that the Phase 2 anchor VQE workflow can cross that bridge
  without requiring manual circuit rewriting beyond explicit noise insertion.

### 6.4 Workload-Driven Noise Expansion

- Noise-model support required for the first noisy training studies.
- Priority for realistic local noise over global toy noise.
- Whole-register depolarizing retained only as a regression or stress-test
  baseline, not as the main scientific workload.

### 6.5 Validation and Publication Evidence

- External reference comparison,
- reproducible benchmark definitions,
- and a publication-ready evidence package for Paper 1.

## 7. Out of Scope

The following are explicitly outside the Phase 2 core deliverable.

- density-aware partitioning and gate fusion,
- channel-native fusion or superoperator fusion,
- gradient-path completion for density-matrix optimization,
- AVX-focused density-kernel acceleration as a main milestone,
- stochastic trajectories or MPDO-based approximation paths,
- full `qgd_Circuit` gate parity,
- and noisy circuit re-synthesis as a primary result.

These topics may be mentioned as future work, but must not be presented as Phase
2 commitments.

## 8. Why Phase 2 Must Precede Phase 3

Phase 2 and Phase 3 are related, but they are not interchangeable as roadmap
milestones.

Phase 2 establishes the exact noisy backend as a usable scientific workflow.
Phase 3 optimizes that workflow through density-aware partitioning, fusion, and
acceleration. Some exploratory work may overlap across the two phases, but the
roadmap order itself is deliberate and should be preserved.

### 8.1 Convincing Technical Argument

From a technical perspective, Phase 3 depends on the existence of a stable target
to optimize.

At the start of Phase 2, the codebase already contains:

- a standalone density-matrix module,
- a separate state-vector partitioning and fusion subsystem,
- and named integration targets for backend selection and exact `Re Tr(H*rho)`
  Hermitian-energy evaluation in the
  variational workflow.

However, the density-matrix backend is not yet fully integrated into the main
workflow contract. Before Phase 2 completes, the project still lacks:

- a documented backend-selection path,
- a validated workflow-level `Re Tr(H*rho)` Hermitian-energy contract,
- a stable bridge from the generated anchor VQE circuit into density-matrix
  execution,
- and a clearly defined support matrix for noisy variational use cases.

If Phase 3 were moved ahead of Phase 2, the project would attempt to optimize a
path that is not yet fully specified at the workflow level. That creates several
technical risks:

- acceleration could target the wrong interface or data flow,
- performance work would lack a stable exact baseline inside the intended
  workflow,
- validation of partitioned or fused density execution would be weaker because
  the integrated reference path would still be incomplete,
- and benchmark choices could drift toward synthetic cases rather than the real
  noisy workflows the backend is supposed to serve.

In short, Phase 2 creates the backend contract and exact workflow baseline;
Phase 3 optimizes that baseline. Reversing the order would mean optimizing
before the target is fully defined and validated.

### 8.2 Convincing Scientific Argument

From a scientific perspective, the order is even more important.

The PhD theme is not merely high-performance simulation. It is scalable training
of quantum circuits under realistic noise. That means the first research
question to answer is:

- can SQUANDER perform exact noisy variational evaluation in a usable,
  reproducible, and scientifically trustworthy way?

Only after that question is answered does the next question become meaningful:

- can partitioning and fusion make that exact noisy workflow faster?

This order strengthens the publication narrative:

1. Phase 2 yields a first major paper on exact noisy backend integration.
2. Phase 3 yields a later methods paper on density-aware acceleration.
3. Later phases use the integrated and accelerated backend to study optimizer
   behavior and trainability under noise.

If the order were reversed, the project would risk producing a weaker Phase 3
story:

- faster than what exact integrated workflow?
- correct relative to what validated backend contract?
- useful for which noisy training workloads?

Without Phase 2 first, Phase 3 could become a performance prototype rather than
a strong methods result grounded in a real scientific workflow.

The roadmap therefore follows the order that maximizes scientific defensibility:

- first establish the research instrument,
- then optimize the instrument,
- then use the optimized instrument to answer higher-level questions about noisy
  optimization and trainability.

### 8.3 Practical Interpretation

The practical interpretation is:

- Phase 2 and Phase 3 may overlap in exploratory reading, benchmarking, or small
  prototypes,
- but they should not be swapped as milestone definitions,
- and Phase 2 completion remains a prerequisite for Phase 3 claims to be strong,
  reproducible, and publication-ready.

## 9. Assumptions

Phase 2 planning assumes:

- the current exact density-matrix backend remains the scientific reference path,
- the main exact target regime is around the current documented Phase 2
  acceptance range of about 10 qubits,
- the primary Phase 2 workflow anchor is noisy VQE for a 1D XXZ spin chain with
  local `Z` field using the default `HEA` ansatz,
- the existing state-vector backend remains intact and continues to serve as the
  default when density execution is not requested and as a comparison path where
  appropriate,
- at least one representative noisy VQA workflow can be specified and validated
  inside the exact regime,
- and Qiskit Aer remains the primary external reference simulator.

## 10. Success Conditions

Phase 2 is only scientifically successful if all of the following become true:

- the density-matrix backend can be selected explicitly in the anchor noisy VQE
  workflow,
- exact noisy expectation values required by that workflow are available and
  validated,
- the anchor noisy workflow is executable end-to-end at 4 to 6 qubits and
  benchmarkable at 8 to 10 qubits inside the exact regime,
- realistic local noise models are used in the core demonstrations,
- unsupported or deferred cases are explicitly documented,
- benchmark and validation results are strong enough to support Paper 1,
- and the resulting system is useful for the later optimizer and trainability
  studies rather than being only a software bridge.

### 10.1 Frozen Implementation Contract Decisions

The following decisions close the pre-implementation checklist and are now part
of the Phase 2 contract.

#### Backend Selection Decision

Phase 2 backend selection is an explicit workflow-level setting on
`qgd_Variational_Quantum_Eigensolver_Base` or an equivalent VQE-facing
configuration entry point.

Contract:

- Phase 2 allows exactly two backend modes: `state_vector` and
  `density_matrix`.
- `state_vector` remains the default when no backend is specified, so existing
  workflows stay backward compatible.
- `density_matrix` must be selected explicitly for any workflow that intends to
  claim exact noisy mixed-state execution.
- No implicit `auto` mode or silent fallback is part of the Phase 2 contract.
- If `density_matrix` is selected but the circuit, noise model, or observable is
  outside the documented support matrix, the workflow must raise a hard error
  before execution.
- If `state_vector` is selected for a workflow that requests Phase 2-only
  mixed-state behavior, that is also a hard error rather than an automatic
  backend swap.

Current implementation-backed clarification:

- The currently delivered user-facing backend switch is a keyword-only
  constructor-level `backend` argument on
  `qgd_Variational_Quantum_Eigensolver_Base`.
- Requests that keep `state_vector` while also configuring Phase 2-only
  mixed-state features such as ordered density noise now hard-error before
  execution rather than silently ignoring those features.

Trade-offs:

- Keeping `state_vector` as the default minimizes migration risk and preserves
  existing users.
- Requiring explicit density-matrix selection and hard errors sacrifices some
  convenience, but it prevents scientifically ambiguous fallback behavior in
  benchmarks and Paper 1 claims.

#### Observable Decision

Phase 2 supports exact real-valued Hamiltonian evaluation through
`E(theta) = Re Tr(H*rho(theta))`.

Contract:

- The required observable interface is the existing Hermitian sparse-Hamiltonian
  input used by `qgd_Variational_Quantum_Eigensolver_Base`.
- The mandatory benchmark Hamiltonian family is the 1D XXZ spin chain with
  optional local `Z` field, because it is both VQE-relevant and expressible in
  terms of `XX`, `YY`, `ZZ`, and local `Z` contributions.
- The implementation may internally use dense or sparse algebra, but the
  workflow contract is Hermitian energy evaluation, not a new generic observable
  API.
- The returned Phase 2 observable is the real energy. Any imaginary component
  larger than the documented tolerance is treated as a validation failure.
- Out of scope for the core Phase 2 contract are arbitrary non-Hermitian
  observables, general POVMs, batched multi-observable APIs, and shot-noise or
  readout estimation as the main acceptance path.

Current implementation-backed clarification:

- The current exact observable path is implemented through the existing
  sparse-Hamiltonian interface and evaluated on the density state directly.
- Executed evidence now covers the mandatory 1 to 3 qubit micro-validation
  matrix and the mandatory 4 / 6 / 8 / 10 qubit workflow-scale matrix against
  Qiskit Aer using the frozen thresholds.

Minimum proof of correctness:

- compare fixed-parameter energy evaluations against Qiskit Aer,
- demonstrate the observable path inside the anchor noisy VQE workflow,
- and enforce the numeric exactness thresholds defined below.

Trade-offs:

- Reusing the existing Hermitian Hamiltonian interface keeps Phase 2 compatible
  with current VQE code and makes validation straightforward.
- The cost of that choice is that a broader observable API is explicitly
  deferred instead of being half-supported.

#### Bridge Decision

The Phase 2 bridge is intentionally partial and workload-driven.

Contract:

- The mandatory and currently delivered source representation is the circuit
  generated by `qgd_Variational_Quantum_Eigensolver_Base.Generate_Circuit()`
  for the default `HEA` ansatz.
- User-supplied `qgd_Circuit` or `Gates_block` inputs are not part of the
  current Phase 2 closure claim; any future clean-lowering extension must be
  documented separately and must not widen the canonical workflow contract
  implicitly.
- The mandatory target representation is a density-matrix execution path built
  around `NoisyCircuit`, `GateOperation`, and explicit ordered `NoiseOperation`
  insertions.
- Phase 2 does not require full `qgd_Circuit` parity.
- Unsupported gates, fused blocks, or noise insertions outside the support
  matrix must fail deterministically before execution and must name the first
  unsupported operation rather than silently skipping or rewriting it.
- The guaranteed workflow behavior is that the anchor noisy VQE workflow can
  traverse this bridge without manual circuit rewriting beyond explicit noise
  specification.

Current implementation-backed clarification:

- The current guaranteed density path is generated `HEA` only.
- `HEA_ZYZ`, binary-imported gate lists, and custom `qgd_Circuit` /
  `Gates_block` sources are currently treated as unsupported on the density path
  rather than partially lowered.

Trade-offs:

- A partial bridge keeps implementation aligned with the real Phase 2 workload
  and avoids promising general circuit compatibility too early.
- The cost is that broader manual `qgd_Circuit` reuse remains a future extension
  instead of a Phase 2 guarantee.

#### Support Matrix Decision

Phase 2 freezes the following support surface.

| Area | Required in Phase 2 | Optional in Phase 2 | Deferred beyond core Phase 2 |
|---|---|---|---|
| Gate families | `U3`, `CNOT` | `H`, `X`, `Y`, `Z`, `S`, `Sdg`, `T`, `Tdg`, `SX`, `RX`, `RY`, `RZ`, `U1`, `U2`, `CZ`, `CH`, `CRX`, `CRY`, `CRZ`, `CP` when needed for tests or comparison microbenchmarks | full `qgd_Circuit` parity, multi-controlled gates, arbitrary custom `Gates_block` structures that cannot be lowered cleanly |
| Noise models | local single-qubit depolarizing, local amplitude damping, local phase damping / dephasing | whole-register depolarizing as a regression or stress-test baseline; generalized amplitude damping or coherent over-rotation only if a benchmark extension proves they are needed | correlated multi-qubit noise, readout noise as a density-backend feature, calibration-aware models, non-Markovian noise |
| Unsupported-case behavior | pre-execution hard error for mandatory workflows | documented opt-in extensions | silent fallback or silent omission of operations is not allowed |

Current implementation-backed clarification:

- The VQE-facing density-noise surface currently accepts ordered fixed local
  insertions only.
- The standalone `NoisyCircuit` module remains broader than the currently
  guaranteed VQE-backed density path, so planning and evidence should keep those
  two support surfaces distinct.
- The delivered optional workflow-facing noise evidence currently consists only
  of whole-register depolarizing regression or stress-test cases; generalized
  amplitude damping and coherent over-rotation remain undelivered extension
  ideas rather than part of the current publication claim.

Trade-offs:

- Requiring only the gate families needed by the anchor workflow keeps Phase 2
  bounded and implementable.
- Requiring local noise channels, including local depolarizing, increases the
  implementation surface beyond Phase 1, but it keeps Paper 1 anchored to
  realistic noise rather than scientifically weak whole-register toy models.

#### Workflow Anchor Decision

The Phase 2 anchor workflow is noisy VQE ground-state estimation for a 1D XXZ
spin chain with local `Z` field using `qgd_Variational_Quantum_Eigensolver_Base`
and the default `HEA` ansatz.

Contract:

- The anchor Hamiltonian family is limited to Hermitian nearest-neighbor XXZ
  models built from `XX`, `YY`, `ZZ`, and local `Z` terms.
- The anchor circuit path is generated `HEA` circuitry lowered through the
  Phase 2 bridge into density-matrix execution.
- The anchor noise path uses explicit local noise insertion with the required
  Phase 2 noise models.
- The anchor observable path is exact energy evaluation via `Re Tr(H*rho)`.
- The scale target is:
  - full end-to-end workflow execution, including at least one reproducible
    optimization trace, at 4 and 6 qubits,
  - fixed-parameter benchmark-ready behavior at 8 and 10 qubits,
  - with 10 qubits treated as the acceptance anchor for the exact regime, not
    as a hard theoretical ceiling.

Current implementation-backed clarification:

- The current executed anchor evidence includes mandatory fixed-parameter cases
  at 4, 6, 8, and 10 qubits (10 parameter vectors per required size), plus one
  bounded 4-qubit optimization trace.
- A documented 10-qubit anchor evaluation is present in the required workflow
  evidence bundle.

Minimum evidence:

- at least one reproducible optimization trace at 4 or 6 qubits,
- parameter-sweep evaluations at 4, 6, 8, and 10 qubits,
- and Qiskit Aer agreement for every mandatory benchmark case.

Trade-offs:

- Choosing a single noisy VQE anchor maximizes alignment with the current VQE
  interface and the Paper 1 narrative.
- The cost is that Phase 2 deliberately does not claim broad readiness for every
  noisy algorithm family.

#### Benchmark Minimum Decision

The Phase 2 minimum benchmark package is frozen as follows.

Mandatory:

- external baseline: Qiskit Aer density-matrix simulation,
- circuit classes:
  - 1 to 3 qubit micro-validation circuits that cover each required gate family
    and each required noise model individually and in mixed sequences,
  - anchor-workflow `HEA` VQE circuits for the XXZ Hamiltonian at 4, 6, 8, and
    10 qubits,
  - at least 10 fixed parameter vectors per mandatory workflow size,
  - at least one reproducible optimization trace on a 4- or 6-qubit anchor
    instance,
- metrics:
  - absolute energy error versus Qiskit Aer,
  - trace-preservation and density-validity checks,
  - runtime,
  - peak memory,
  - workflow completion status,
- reproducibility artifacts:
  - Hamiltonian specification,
  - ansatz type and depth,
  - noise insertion schedule and parameters,
  - random seeds or an explicit deterministic parameter-generation / trace
    policy when no randomness is required,
  - software versions or commit,
  - and raw benchmark outputs.

Current implementation-backed clarification:

- The implemented Phase 2 validation stack now emits a complete backend-explicit
  artifact set covering local correctness, canonical workflow definition,
  end-to-end 4/6 workflow evidence, workflow-scale exact-regime validation,
  unsupported-workflow boundaries, and interpretation guardrails.
- The top-level completeness, status, and workflow-identity checks are encoded
  in `workflow_publication_bundle.json`, which verifies mandatory artifact
  presence, expected status alignment, cross-artifact workflow identity, and
  lower-story semantic closure flags across the final Task 6 publication-facing
  evidence layers.

Optional:

- one secondary simulator baseline such as QuTiP, Qulacs, or QuEST if it
  materially strengthens Paper 1.

Trade-offs:

- This minimum package is strong enough to support Paper 1 without turning Phase
  2 into a multi-framework bake-off or a Phase 3 performance study.
- The downside is that broader simulator comparison is deliberately left
  optional rather than required.

#### Numeric Acceptance Threshold Decision

Phase 2 go or no-go thresholds are frozen as:

- exact operating regime: mandatory acceptance coverage at 4, 6, 8, and 10
  qubits, with 10 qubits required for at least one anchor-workflow evaluation
  case,
- micro-validation accuracy: maximum absolute energy error `<= 1e-10` versus
  Qiskit Aer on the 1 to 3 qubit required microcases,
- workflow-level accuracy: maximum absolute energy error `<= 1e-8` versus
  Qiskit Aer on the mandatory 4, 6, 8, and 10 qubit parameter-sweep cases,
- density validity: `rho.is_valid(tol=1e-10)` must pass and `|Tr(rho) - 1| <=
  1e-10` on recorded validation outputs,
- Hermitian observable consistency: `|Im Tr(H*rho)| <= 1e-10` for the exact
  observable path,
- workflow completeness:
  - explicit `density_matrix` selection works without fallback,
  - the 4- or 6-qubit anchor optimization trace completes end-to-end,
  - and all mandatory parameter-sweep cases complete without
    unsupported-operation workarounds,
- Paper 1 readiness threshold:
  - `100%` pass rate on the mandatory micro-validation matrix,
  - `100%` pass rate on the mandatory workflow benchmark set,
  - one documented 10-qubit anchor case,
  - and a complete reproducibility bundle.

Trade-offs:

- These thresholds are strict enough to make the Phase 2 paper defensible.
- They avoid introducing runtime or speedup pass or fail thresholds, because
  Phase 2 is about exact integrated correctness rather than acceleration.

## 11. Phase 2 Task Breakdown

Each task below is a goal, not an implementation recipe.

### Task 1: Backend Selection Contract

#### Goal

Define and validate how the density-matrix backend is selected and how that
selection interacts with the current state-vector-first workflows.

#### Why It Exists

Without a stable backend-selection contract, the density backend remains a
standalone tool instead of a usable research backend.

#### Success Looks Like

- backend choice is a documented first-class concept,
- expected user-facing behavior is defined,
- and interaction with existing workflows is unambiguous.

#### Evidence Required

- documented backend-selection semantics,
- defined supported entry points,
- and validation that the density path can actually be invoked in the intended
  workflow class.

### Task 2: Exact Noisy Expectation-Value Path

#### Goal

Define and validate an exact real-valued Hermitian-energy
`Re Tr(H*rho)` observable-evaluation path sufficient for the canonical noisy
variational workflow.

#### Why It Exists

Observable evaluation is the minimum scientific requirement for the density
backend to participate in VQA or similar training loops.

#### Success Looks Like

- the exact Hermitian-energy contract is documented,
- observable evaluation is numerically trustworthy within the supported XXZ
  workflow scope,
- and the canonical noisy VQE workflow depends on it successfully.

#### Evidence Required

- agreement against a trusted reference backend,
- documented Hermitian-energy scope assumptions,
- and benchmark or test evidence showing stable exact noisy `Re Tr(H*rho)`
  evaluation in the canonical workflow.

### Task 3: Circuit-to-Density Backend Bridge

#### Goal

Define how existing circuit and gate representations are translated or routed
into the density-matrix execution path.

#### Why It Exists

The current density backend is structurally separate from the main circuit path.
This task turns that separation into a documented, usable bridge rather than an
implicit gap.

#### Success Looks Like

- supported circuit-entry modes are documented,
- unsupported cases are defined,
- and the bridge is sufficient for the target noisy workflow.

#### Evidence Required

- explicit support matrix,
- defined behavior for unsupported gate/noise combinations,
- and at least one end-to-end workflow that depends on the bridge.

### Task 4: Phase 2 Noise Support Baseline

#### Goal

Define the realistic local noise models required for the first noisy training
studies and document what is mandatory, optional, and deferred.

#### Why It Exists

Phase 2 should not drift into either unrealistically weak toy-noise studies or
an unbounded effort to implement every possible noise model.

#### Success Looks Like

- the Phase 2 noise scope is finite,
- it is aligned with realistic local noisy training use cases,
- and it is sufficient for the first training demonstrations.

#### Evidence Required

- a noise support matrix,
- rationale for why each required model is included,
- and benchmark or validation plans tied to those models.

### Task 5: Validation Baseline

#### Goal

Define the minimum exactness and correctness evidence required before Phase 2 can
be considered scientifically usable.

#### Why It Exists

Phase 2 is exact-first. Its scientific value depends on trusted correctness.

#### Success Looks Like

- numerical correctness criteria are explicit,
- reference comparisons are specified,
- and acceptance thresholds are defined.

#### Evidence Required

- exactness comparison plan against Qiskit Aer,
- internal consistency checks,
- and a documented pass/fail interpretation for Phase 2 claims.

### Task 6: Noisy Workflow Demonstration Goal

#### Goal

Specify at least one end-to-end noisy workflow that Phase 2 must support.

#### Why It Exists

Phase 2 should be judged by usefulness to noisy training research, not only by
backend availability.

#### Success Looks Like

- one representative workflow is chosen,
- the workflow is executable inside the exact regime,
- and its artifacts are reproducible enough for publication.

#### Evidence Required

- workflow definition,
- expected input/output contract,
- and demonstration-ready validation or benchmark design.

### Task 7: Documentation and User-Facing Clarity

#### Goal

Document the Phase 2 support surface so that the backend can be used, reviewed,
and discussed without reverse-engineering implementation details.

#### Why It Exists

Paper 1 and future implementation both depend on clear contracts and explicit
scope boundaries.

#### Success Looks Like

- clear developer-facing and research-facing documentation exists,
- Phase 2 non-goals are visible,
- and future work is separated from current commitments.

#### Evidence Required

- coherent Phase 2 document bundle,
- terminology consistency,
- and explicit alignment with the planning and roadmap docs.

### Task 8: Paper 1 Evidence Package

#### Goal

Define the exact evidence required for the first major publication tied to Phase
2.

#### Why It Exists

Publication readiness should shape the phase, not be retrofitted after the fact.

#### Success Looks Like

- Paper 1 has a clear contribution boundary,
- required evidence is defined before implementation,
- and benchmark/validation demands are aligned with the publication strategy.

#### Evidence Required

- abstract-level claim set,
- short-paper structure,
- full-paper structure,
- and traceability from Phase 2 deliverables to publication sections.

## 12. Full-Phase Acceptance Criteria

Phase 2 is complete only if all of the following are true:

- the anchor noisy VQE workflow can execute the density-matrix backend path with
  explicit backend selection and no implicit fallback,
- `Re Tr(H*rho)` is validated for the supported Hermitian Hamiltonian class and
  meets the numeric thresholds in Section 10.1,
- exact noisy emulation is stable across the documented 4 to 10 qubit exact
  regime,
- the bridge from the generated `HEA` anchor circuit into density execution is
  documented, partial by design, and usable for the anchor workflow,
- realistic local noise support is sufficient for the documented first noisy
  training studies,
- at least one end-to-end noisy workflow is specified and supported at 4 to 6
  qubits, with 8 and 10 qubit evaluation cases recorded,
- and the publication evidence package for Paper 1 is complete enough to support
  abstract, short-paper, and full-paper drafting honestly.

These criteria are intentionally aligned with:

- `docs/density_matrix_project/CHANGELOG.md`
- `docs/density_matrix_project/RESEARCH_ALIGNMENT.md`
- `docs/density_matrix_project/planning/PLANNING.md`
- `docs/density_matrix_project/planning/PUBLICATIONS.md`

## 13. Validation and Benchmark Matrix

### 13.1 Primary External Baseline

- Qiskit Aer density-matrix simulation is the required primary reference.

### 13.2 Optional Secondary Baselines

- One additional simulator or framework may be used when it materially improves
  the publication evidence, but this is secondary to Qiskit Aer.
- Preferred secondary choices are QuTiP, Qulacs, or QuEST when one of them is
  easy to reproduce for the chosen benchmark subset.

### 13.3 Workload Classes

- 1 to 3 qubit micro-validation circuits that cover each required gate family
  and each required local noise model,
- the anchor `HEA` noisy VQE workflow for the XXZ Hamiltonian at 4, 6, 8, and
  10 qubits,
- at least 10 fixed parameter vectors per mandatory workflow size,
- and one reproducible 4- or 6-qubit optimization trace that exercises the
  full backend-selection, bridge, noise, and observable path.

### 13.4 Noise Classes

- local single-qubit depolarizing,
- local phase damping / dephasing,
- local amplitude damping,
- whole-register depolarizing only as the delivered optional regression or
  stress-test baseline,
- generalized amplitude damping or coherent unitary / over-rotation error only
  as future justified extensions beyond the current delivered evidence surface,
- and readout or shot-noise discussion only as secondary context rather than a
  core density-backend requirement.

### 13.5 Metrics

- numerical agreement with trusted references,
- observable error,
- trace-preservation and density-validity checks,
- runtime,
- peak memory footprint,
- stability of end-to-end execution,
- and reproducibility of the workflow and benchmark setup.

## 14. Risks

### Risk 1: Phase 2 Scope Drift

If Phase 2 starts to absorb density-aware partitioning, fusion, or gradient-path
work, it will lose focus and weaken Paper 1.

Mitigation:

- enforce out-of-scope boundaries,
- and treat acceleration as Phase 3 unless strictly required for Phase 2
  usability.

### Risk 2: Gate-Coverage Mismatch

The target workflow may depend on unsupported gate families.

Mitigation:

- use workload-driven gate support decisions,
- choose representative workflows compatible with the documented support matrix,
- and explicitly document unsupported cases.

### Risk 3: Noise-Scope Inflation

Trying to support too many realistic noise models too early can delay the first
scientifically useful backend integration.

Mitigation:

- require every Phase 2 noise model to justify itself via a target workflow or
  publication need.

### Risk 4: Weak Validation Package

If the reference comparisons are too narrow, Paper 1 will read as a software
note rather than a strong methods paper.

Mitigation:

- define publication evidence requirements before implementation,
- and keep Qiskit Aer validation central.

### Risk 5: Scientific Contribution Dilution

Phase 2 could become a pure integration effort without a strong research story.

Mitigation:

- keep the document bundle centered on exact noisy training usability and
  publication-grade evidence,
- not only on technical integration.

## 15. Decision Gates

### DG-1: Phase 2 Completion Gate

Question:

- does the integrated backend satisfy the explicit backend-selection contract,
  the numeric exactness thresholds, and the minimum benchmark package for the
  anchor noisy VQE workflow, including one 10-qubit evaluation case, strongly
  enough to underpin Paper 1?

If no:

- Phase 2 is not complete,
- even if partial backend integration exists.

### DG-2: Handoff to Phase 3

Question:

- are the remaining limitations primarily about performance and execution cost,
  rather than missing workflow integration?

If yes:

- proceed to density-aware partitioning, fusion, and acceleration.

If no:

- the project still has unresolved Phase 2 integration debt.

## 16. Non-Goals

To avoid later ambiguity, the following are explicit non-goals of this phase:

- proving density-aware partitioning benefit,
- delivering fused density-matrix execution,
- solving density-matrix gradients end to end,
- maximizing qubit count beyond the exact reference regime,
- and producing large-scale approximation results.

## 17. Expected Outcome

At the end of Phase 2, the project should have:

- an exact noisy backend that is integrated into the anchor variational
  workflow,
- a validated expectation-value path,
- a documented HEA-plus-local-noise workflow baseline for early studies,
- a reproducible evidence package for Paper 1,
- and a clean handoff into Phase 3, where performance acceleration becomes the
  main question.

That is the minimum outcome required for Phase 2 to count as a meaningful step
toward the broader PhD objective.
