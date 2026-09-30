# Initial requirements — M4 `canonical-attributed-energy`

> **Status:** v0.1 frozen baseline (owner-confirmed 2026-09-30), adversarial critique and confirmation pass recorded ·
> **Milestone:** M4 `canonical-attributed-energy` — Now, product walking skeleton ·
> **Traces:** CAP-001, CAP-003, CAP-005, CAP-007 · QA-001, QA-002, QA-005 (scoped), QA-008, QA-009 ·
> **Upstream:** [`PRODUCT_STATEMENT.md`](../../PRODUCT_STATEMENT.md) v0.4 and [`ROADMAP.md`](../../ROADMAP.md) v0.7,
> both confirmed by the owner on 2026-09-30 · **Owner skill:** `create-initreq-for-sdd` ·
> **Downstream:** `spec-driven-development` Steps 1–3 · **Not:** planning, ADRs, stories, tasks, or code.

## 1. Milestone scope and vision

- **Milestone:** M4 `canonical-attributed-energy`, the product walking skeleton of the new convention; the q4 `partitioned` row is the slice tracer.
- **Upstream traceability:** the header traces. Outcome and measure ([roadmap §4](../../ROADMAP.md#4-per-milestone-detail)): from one public VQE
  instance a researcher obtains exact noisy energy through bridge → planner → attributed runtime → exact core → observable; every route is
  labelled, every q4/q6/q8/q10 full state meets QA-001, and energy agrees with the public optimizer path and with Aer.
- **In scope:** the attributed-energy entry; route labels and records on every advertised route; the published entry-point support matrix and
  its pinned negatives; counted full-state, energy, and Aer rows; the regenerable M4 bundle; current-state doc updates at milestone close.
- **Out of scope:** new channels (M7); timing, cost selection, or performance claims (M6, M9A, M9); clamp migration (M10); legacy `NoiseChannel`
  changes; optimizer iterations through the Python runtime; gradients (M8B); declared dense schedules (M8); strict execution of the canonical
  whole workload; counted widths outside 4–10; changes to delivered C++ VQE error types or messages.
- **Target users:** the noisy-training researcher (primary); the SQUANDER maintainer, who needs the state-vector path untouched; the external
  reproducer, who regenerates the counted bundle.
- **Core problem:** the public density energy exists only on the C++ sequential path, and the Python planner/runtime stops at a state and a
  route summary. Nothing shows the two close on the same state and observable on every route; only hybrid partitions carry a route class;
  unsupported instances surface as bare exceptions; a strict rejection can fire after earlier partitions have run.
- **Success metrics:** 100 % of the 216 counted route rows meet QA-001, the derived energy tolerance, and QA-002; 100 % of executed partitions
  are labelled; 0 vacuous counted rows; every rejected matrix class has a pinned negative; 100 % of counted rows regenerate from one named
  lane; 0 upstream state-vector regressions.
- **Current-state context:** [`ARCHITECTURE_OVERVIEW.md`](../../ARCHITECTURE_OVERVIEW.md) and [`TECH_STACK.md`](../../TECH_STACK.md) constrain this
  milestone and are updated at its close, not here. The roadmap's architecture boundary holds: the C++ optimizer loop cannot call the Python
  planner/runtime, so M4 walks the product at one energy evaluation.

## 2. User journeys

**Primary — the noisy-training researcher.** (1) Builds the canonical noisy workflow on the density backend at q4. (2) Calls the
attributed-energy entry once with the instance, a pre-registered vector, and a route: `partitioned`, `fused`, or `hybrid`. (3) Receives
\(\mathrm{Re}\,\mathrm{Tr}(H\rho)\), the route record — requested and realized route, one label per executed partition, span budget — and, on
request, \(\rho\). (4) Checks it against `Optimization_Problem`, then cites the counted row, Aer check included, from the regenerated bundle.

**Alternate paths.**
- *Unsupported instance or call:* a named structured error before any partition executes; nothing is returned; the instance is unchanged.
- *Strict request on a counted anchor:* rejected at preflight, naming the first ineligible partition; no partition executes.
- *Nothing to fuse, or no eligible motif:* the partition carries its fallback label and reason; a vacuous row is never counted.
- *Aer absent:* the Aer lane fails with `missing_dependency`; the fast lane is unaffected.
- *Maintainer and reproducer:* the state-vector suite and default backend are unchanged; one lane regenerates the bundle.

## 3. Ubiquitous language (milestone glossary)

Extends the [product glossary](../../PRODUCT_STATEMENT.md#6-ubiquitous-language-seed-glossary); use these terms verbatim in tests, records, and code.

| Term | Meaning in M4 |
|------|---------------|
| **Canonical anchor** | The canonical noisy workflow at \(n\in\{4,6,8,10\}\) — open-chain XXZ (\(h=0.5\), \(J=1\)), generated `HEA` with one layer and one inner block, the delivered three-channel schedule at its delivered rates — paired with one pre-registered vector; 64 anchors. |
| **Boundary-rate anchor** | The same workflow at `set_00` with all three rates set to 0, or all to 1, at each counted width; 8 anchors realizing QA-002's boundary points. **Counted anchors:** the 64 plus these 8. |
| **Pre-registered vector set** | Per width: Phase 2 `set_00`–`set_09`, the all-zero identity vector, and five uniform draws on \([-\pi,\pi)\) from a fresh seed; 16 vectors, frozen and archived before counted data. |
| **Bridge; bridge-order reference** | The ordered gate+noise operations of `describe_density_bridge()`, the order the public energy lowers; the reference is a sequential `NoisyCircuit` built from them in that order with no planner output — M4's counted *sequential reference*. |
| **Public energy** | `Optimization_Problem(p)` on the same instance and vector: the delivered C++ sequential energy. |
| **Attributed-energy entry** | The public entry that takes one density-backend VQE instance, one parameter vector, and one requested route; one evaluation per call; it never drives an optimizer. |
| **Attributed energy** | \(\mathrm{Re}\,\mathrm{Tr}(H\rho)\) for the instance's own Hamiltonian on the state one route produced, returned with its imaginary part and the route record. |
| **Route** | `partitioned`, `fused` (unitary-island), `hybrid` (channel-native with labelled fallback), or `strict` (channel-native, proof mode). *Requested* is what the caller asked for; *realized* is what executed, derived from the route labels. |
| **Route label** | One label per executed partition from a published closed vocabulary — at least baseline, unitary-island fused, supported-but-unfused, and channel-native — plus the frozen fallback reason where the requested route did not apply. |
| **Route record** | Requested and realized route, route labels and reasons, span budget, partition and fused-region counts, workload identity (width, parameter count, source, noise schedule and rates, vector id), and energy parts. |
| **Span budget** | The planner's maximum partition width in qubits; 2 for every counted row. |
| **Non-vacuous row** | A counted `fused` row with ≥ 1 actually fused island, or a counted `hybrid` row with ≥ 1 channel-native partition. |
| **Counted route row; reference row** | One (counted anchor, route) execution carrying its state (QA-001), energy (\(\tau_E\)), and Aer (QA-002) checks: 72 × 3 = 216. A reference row checks the bridge-order reference against the public energy and Aer: 72. |
| **Derived energy tolerance** \(\tau_E(n)\) | \(\lVert H\rVert_F\cdot10^{-10}\), implied by QA-001's Frobenius bound through Cauchy–Schwarz; validators use the exact per-case value (≈ 6.3e-10, 1.6e-9, 3.8e-9, 8.7e-9 at q4, q6, q8, q10). |
| **Preflight; pre-mutation** | A rejection raised before any partition executes: no state, energy, or record is returned, and the caller-owned inputs (instance or descriptor set, and vector) are unchanged. |
| **Validator verdict code** | A frozen token the M4 bundle validator emits with a non-zero exit: `state_exactness_violation`, `energy_tolerance_violation`, `observable_imaginary_violation`, `aer_agreement_violation`, `vacuous_route_row`, `unlabelled_partition`, `unproven_matrix_row`, `unpinned_rejected_class`, `missing_provenance`, `classification_mismatch`, `missing_dependency`. |
| **Entry-point support matrix** | The published classification of every density entry as supported and counted, supported oracle, frozen slice only, exposed but non-claim-bearing (used by no counted row, covered by no M4 claim), or preflight-rejected. |

## 4. Requirements and acceptance criteria

### REQ-001 — Attributed energy from one public instance
**Upstream:** CAP-005, CAP-001, CAP-003 · QA-001, QA-005 (scoped) · **Evidence:** fast pytest (q4); correctness evidence pipeline.
*As a noisy-training researcher, I want one call that turns a canonical instance, a vector, and a route into an exact energy with its route
record, so that every noisy energy I cite names the path that produced it.*
- **Given** a counted anchor, **when** `partitioned`, `fused`, or `hybrid` is requested, **then** the entry returns a finite real energy with
  \(\lvert\mathrm{Im}\,\mathrm{Tr}(H\rho)\rvert\le10^{-10}\), the route record, and, on request, the \(\rho\) that produced it.
- *EARS:* The entry shall take the observable from the instance and accept no substitute. When called, it shall evaluate exactly one energy,
  and the instance's `describe_density_bridge()` metadata and the caller's vector shall be identical before and after the call.
- **Negative:** **Given** a counted anchor, **when** the route is `"auto"` or any other name outside the four routes, **then**
  `NoisyPlannerValidationError` (`mode` · `unsupported_route` · `attributed_energy_preflight`) is raised and nothing is returned.

### REQ-002 — Full-state exactness on every counted route
**Upstream:** CAP-001, CAP-003 · QA-001 · **Evidence:** fast pytest (q4 rows, sentinel); correctness evidence pipeline (all rows).
- **Given** each of the 72 counted anchors, **when** each counted route executes through the entry, **then** its state against the bridge-order
  reference meets \(\lVert\Delta\rho\rVert_F\le10^{-10}\), \(\lVert\Delta\rho\rVert_{\max}\le10^{-10}\), \(\lvert\mathrm{Tr}\,\rho-1\rvert\le10^{-10}\),
  and \(\lambda_{\min}(\rho)\ge-10^{-12}\) on 100 % of the 216 rows.
- *EARS:* The reference shall be built from the bridge with no planner output. If a counted state row fails, then every energy and Aer claim at
  that (width, route) is frozen until the disagreement is resolved (A6).
- **Negative:** **Given** the planted-defect sentinel — q4, `set_00`, the `local_depolarizing` rate changed from 0.1 to 0.101 on the planner
  surface only — **when** its `partitioned` state row is evaluated, **then** \(\lVert\Delta\rho\rVert_F\ge10^{-6}\) and the validator exits
  with `state_exactness_violation`, naming the row.

### REQ-003 — Energy closure with the public energy
**Upstream:** CAP-005, CAP-001 · QA-001 · **Evidence:** fast pytest (q4); correctness evidence pipeline.
- **Given** each counted route row and reference row, **when** its energy is compared with the public energy on the same vector, **then**
  \(\lvert E-E_\mathrm{public}\rvert\le\tau_E(n)\) on 100 % of the 288 rows, each recording its imaginary part, \(\lVert H\rVert_F\) computed from
  the instance's Hamiltonian, and \(\tau_E(n)\).
- **Negative:** **Given** a row whose \(\lvert\Delta E\rvert\) exceeds \(\tau_E(n)\), or whose imaginary part exceeds \(10^{-10}\), **when** the
  validator runs, **then** it exits with `energy_tolerance_violation` or `observable_imaginary_violation`, naming the row, and the bundle is
  not marked `pass`.

### REQ-004 — Route labels, route records, and non-vacuous rows
**Upstream:** CAP-003, CAP-007 · QA-005 (scoped) · **Evidence:** fast pytest (record schema, q4); correctness evidence pipeline (validator).
- **Given** any executed request on an advertised route, **then** 100 % of executed partitions carry exactly one label from the published
  vocabulary, and the record carries every route-record field.
- *EARS:* When a `fused` partition fuses no island, the system shall label it supported-but-unfused. When a `hybrid` partition is ineligible,
  the system shall label its fallback with the frozen reason (`pure_unitary_partition`, `channel_native_qubit_span`, or
  `channel_native_support_surface`). The realized route shall be derived from the labels, never copied from the request.
- **Given** a counted `fused` or `hybrid` row at span budget 2, **then** it is non-vacuous.
- **Negative:** **Given** a `hybrid` row planned at span budget 4, where no canonical partition is eligible, **when** the bundle validator runs,
  **then** it rejects the row with `vacuous_route_row`; a record with an unlabelled partition or an out-of-vocabulary label fails with
  `unlabelled_partition`.

### REQ-005 — Strict requests on the counted anchors
**Upstream:** CAP-003, CAP-001 · QA-005 (scoped) · **Evidence:** fast pytest (support-boundary suite); delivered Phase 3.1 slice tests.
- **Negative:** **Given** a counted anchor at span budget 2, **when** `strict` is requested, **then** `NoisyRuntimeValidationError`
  (`unsupported_runtime_operation` · `channel_native_noise_presence` · `runtime_preflight`) names the first ineligible partition,
  0 partitions execute, and no state, energy, or record is returned.
- *EARS:* While `strict` is requested, the system shall classify every partition before executing the first. Where every partition is
  eligible — the frozen Phase 3.1 ≤ 2-qubit motif slice — strict execution remains supported, every partition is labelled channel-native,
  and the delivered slice tests pass unchanged.

### REQ-006 — Preflight rejection of unsupported instances and calls
**Upstream:** CAP-005, CAP-001 · QA-005 (scoped) · **Evidence:** fast pytest (support-boundary suite, one pinned negative per row).
- **Negative:** **Given** an instance or call in the table, **when** the attributed-energy entry is called (for the last row, the direct runtime
  entry), **then** the named error — a delivered `ValueError`-family structured error with `to_dict()` — is raised before any partition
  executes, nothing is returned, and the caller-owned inputs (instance or descriptor set, and vector) are unchanged. Tokens marked *new* are
  introduced by M4; renaming one is a requirements change.

| Rejected input | Error · `category` · `first_unsupported_condition` · `failure_stage` |
|----------------|---------------------------------------------------------------------|
| Instance with the default or `backend="state_vector"` backend | `NoisyPlannerValidationError` · `backend` · `state_vector` · `attributed_energy_preflight` (new) |
| `Generate_Circuit` never called | `NoisyPlannerValidationError` · `source_type` · `unset` · `attributed_energy_preflight` (new) |
| Generated non-`HEA` ansatz (`HEA_ZYZ`) | `NoisyPlannerValidationError` · `source_type` · `generated_hea_zyz` · `attributed_energy_preflight` (new) |
| Custom gate structure (`set_Gate_Structure`) | `NoisyPlannerValidationError` · `source_type` · `custom_gate_structure` · `attributed_energy_preflight` (new) |
| Binary-imported gate structure | `NoisyPlannerValidationError` · `source_type` · `binary_import` · `attributed_energy_preflight` (condition as delivered) |
| Caller-set initial state (`set_Initial_State`) | `NoisyPlannerValidationError` · `initial_state` · `caller_initial_state` · `attributed_energy_preflight` (new) |
| Noise insertion after the last generated gate | `NoisyPlannerValidationError` · `noise_insertion` · `after_gate_index` · `attributed_energy_preflight` (category and condition as delivered) |
| Hamiltonian not \(2^n\times2^n\) | `NoisyPlannerValidationError` · `observable` · `hamiltonian_dimension` · `attributed_energy_preflight` (new) |
| Parameter vector of the wrong length | `NoisyRuntimeValidationError` · `runtime_request` · `parameter_count` · `runtime_preflight` (delivered) |
| Parameter vector not one-dimensional | `NoisyRuntimeValidationError` · `runtime_request` · `parameter_vector` · `runtime_preflight` (delivered) |
| Parameter vector with a NaN or ±∞ entry | `NoisyRuntimeValidationError` · `runtime_request` · `parameter_value` · `runtime_preflight` (new) |
| Direct runtime call with an unknown `runtime_path` | `NoisyRuntimeValidationError` · `runtime_request` · `runtime_path` · `runtime_preflight` (new) |

### REQ-007 — Published entry-point support matrix
**Upstream:** CAP-005, CAP-007 · QA-005 (scoped), QA-008 · **Evidence:** benchmark tests (`pytest benchmarks/density_matrix -v`); correctness evidence pipeline.
- **Given** the M4 bundle, **then** it publishes a machine-readable matrix carrying every row of the roadmap's M4 support matrix at the
  classification stated there, each linked to what proves it:

| Roadmap row (short) | Proved by |
|---------------------|-----------|
| Direct C++ `NoisyCircuit` · Python-bound `NoisyCircuit` | the optional C++ tests (delivered) and the public energy, which runs it · reference rows of REQ-002, REQ-003, REQ-009 |
| Planner → partitioned; → unitary-island fused; hybrid channel-native | REQ-002, REQ-003, REQ-004, REQ-009 |
| Strict channel-native whole-workload request | REQ-005 |
| Public density-backend VQE energy | REQ-003 (as comparator), REQ-008 |
| New attributed energy over planner/runtime | REQ-001, REQ-004, REQ-006 |
| Legacy `NoiseChannel`; clamping parametric-rate entries | this requirement: non-claim-bearing and unchanged |
| Other gates, noise names, sources, or modes | REQ-001, REQ-006, REQ-008, and the delivered planner, descriptor, and runtime negatives |

- *EARS:* The matrix shall classify attributed energy outside the counted anchors — other widths, rates, or schedules within the delivered
  vocabulary — as exposed but non-claim-bearing. No counted row shall use a legacy `NoiseChannel` or a clamping entry; M4 changes neither.
- **Negative:** **Given** a matrix row that claims support without counted rows, or a rejected class without a pinned negative, **when** the
  validator runs, **then** it exits with `unproven_matrix_row` or `unpinned_rejected_class`, naming the row.

### REQ-008 — Delivered public-VQE rejections pinned as delivered
**Upstream:** CAP-005 · QA-005 (scoped), QA-009 · **Evidence:** fast pytest (`tests/VQE/`).
- **Negative:** **Given** a public VQE construction or method call that hits a publicly reachable rejection class in the pinned rejection
  manifest — built from the raise sites of `_normalize_vqe_backend_name`, `_normalize_density_noise_spec`, `set_density_noise_specs`, and
  `validate_density_anchor_support`, with unreachable native branches recorded as such — **then** a pinned test asserts the delivered type
  and message token before any energy is returned; for example bare `Exception` "does not support gradient-based optimization", "supports
  only BAYES_OPT or COSINE", "after_gate_index exceeds generated gate count", "noise values must be in [0, 1]"; `ValueError` "Unsupported
  density-noise channel", "non-finite value"; `TypeError` "should be an integer".
- *EARS:* M4 shall not change these types or messages; where REQ-006 also rejects an input, the public path keeps its delivered error. A
  positive control confirms that `Optimization_Problem` on a counted anchor still returns the public energy.

### REQ-009 — External agreement with Qiskit Aer at every counted width and route
**Upstream:** CAP-001, CAP-005, CAP-007 · QA-002 · **Evidence:** Qiskit Aer external reference; correctness evidence pipeline with Aer installed.
- **Given** each of the 72 counted anchors, **when** Aer's density-matrix simulator evolves the bridge circuit once, **then** all 216 route rows
  and 72 reference rows meet \(\lVert\rho-\rho_\mathrm{Aer}\rVert_F<10^{-12}\) and \(\lvert E-E_\mathrm{Aer}\rvert\le10^{-10}\); boundary-rate anchors
  supply QA-002's boundary points and canonical anchors its interior points, at every width and route.
- *EARS:* Where the Aer lane runs, the bundle shall pin the `qiskit` and `qiskit-aer` versions and the Aer method. The frozen
  \(\lvert\Delta E\rvert\le10^{-10}\) closes QA-002's `[confirm]`.
- **Negative:** **Given** a row outside either bound, the validator exits with `aer_agreement_violation`, naming the row. **Given** an
  environment where `qiskit_aer` cannot be imported, **when** the Aer lane runs, **then** it exits with `missing_dependency`, naming the
  package, and writes no `pass` bundle; the fast lane imports no Qiskit module.

### REQ-010 — Regenerable counted evidence and continuity anchors
**Upstream:** CAP-007 · QA-008 · **Evidence:** correctness evidence pipeline; benchmark tests.
- **Given** a clean checkout at the pinned revision in `qgd`, **when** the named lane regenerates the M4 bundle, **then** every categorical
  field — labels, realized routes, verdicts, matrix classes, negative tokens — reproduces exactly and every residual stays within its frozen
  tolerance, on 100 % of rows.
- *EARS:* Every bundle shall pin the revision, tolerances, per-case \(\lVert H\rVert_F\) and \(\tau_E\), span budget, vector ids, values, and seed,
  rates and channel conventions, environment and dependency versions, and the claim boundary (counted anchors, three counted routes, no
  performance claim). As the continuity anchor, public energies at `set_00`–`set_09` shall match the delivered Phase 2 exact-regime
  workflow bundle within \(10^{-10}\).
- **Negative:** **Given** a manifest without the revision, seed, or span budget, **when** the validator runs, **then** it exits with
  `missing_provenance`, naming the field; a regenerated classification that differs from the committed bundle fails with
  `classification_mismatch`, naming the row.

### REQ-011 — Non-interference with the state-vector path
**Upstream:** CAP-005 · QA-009 · **Evidence:** CI (`pytest tests/` in `.github/workflows/ci.yml`); local `pytest -m "not density_matrix"`.
- **Given** any M4 slice, **when** CI and the local upstream lane run, **then** the state-vector suite passes with 0 regressions, the default
  VQE backend is `state_vector`, and a pinned state-vector `Optimization_Problem` energy, recorded before the first slice, is unchanged within
  \(10^{-12}\).
- *EARS:* The entry shall be additive: no delivered public signature or behavior changes beyond the tightenings REQ-004 to REQ-006 name,
  `Start_Optimization` stays on the C++ path, and a tightened behavior gets a new test rather than a weakened one.
- **Negative:** **Given** a state-vector instance, **when** the attributed entry is called, **then** it raises the REQ-006 `backend` error and
  never computes a density energy.

**Trace coverage** (`REQ-*` numbers). CAP-001 → 001, 002, 003, 005, 006, 009 · CAP-003 → 001, 002, 004, 005 · CAP-005 → 001, 003, 006, 007, 008,
009, 011 · CAP-007 → 004, 007, 009, 010 · QA-001 → 001, 002, 003 · QA-002 → 009 · QA-005 (scoped) → 001, 004–008 · QA-008 → 007, 010 ·
QA-009 → 008, 011.

## 5. Non-functional requirements

| NFR | Response measure | Lane that proves it |
|-----|------------------|---------------------|
| **Exactness** → QA-001 | 100 % of 216 route rows within REQ-002's bounds; 100 % of 288 energy rows within \(\tau_E(n)\) | fast pytest (q4); correctness evidence pipeline |
| **External agreement** → QA-002 | 100 % of 288 Aer rows: \(\lVert\Delta\rho\rVert_F<10^{-12}\), \(\lvert\Delta E\rvert\le10^{-10}\); boundary-rate and canonical anchors at every width and route | Qiskit Aer external reference |
| **Strictness (scoped)** → QA-005 | 100 % of rejected classes raise the named error with 0 partitions executed; 0 silent route substitutions or clamps; 100 % of executed partitions labelled; 0 vacuous counted rows | fast pytest (support-boundary suite); correctness evidence pipeline (validator) |
| **Reproducibility** → QA-008 | 100 % of counted rows regenerate from one lane; categorical fields identical; residuals within frozen tolerance; manifest complete per REQ-010 | correctness evidence pipeline; benchmark tests |
| **Non-interference** → QA-009 | 0 upstream regressions; default backend `state_vector`; pinned state-vector energy within \(10^{-12}\) | CI; local `pytest -m "not density_matrix"` |
| **Performance / scale** | No claim: 0 claim-bearing bundle fields cite `runtime_ms` or `peak_rss_kb`, which stay diagnostic | correctness evidence pipeline (validator) |
| **Observability** | 100 % of rejections raised by the attributed entry and the planner/runtime routes serialize through `to_dict()`, while REQ-008 and lane-bootstrap failures keep their pinned contracts; 100 % of route records serialize to JSON with a schema version | fast pytest |
| **Compatibility** | Python 3.13 in `qgd` and C++11; no new required dependency; `qiskit` and `qiskit-aer` stay optional and are version-pinned in Aer bundles | CI; Aer lane |
| **Current-state docs** | At close, `ARCHITECTURE_OVERVIEW.md` gains the attributed-energy flow and support boundary, and `TECH_STACK.md` the M4 lane | doc review at the M4 closeout |

Security, privacy, and accessibility are out of scope: M4 handles no personal data, secrets, network access, or user interface.

## 6. Operational boundaries

**Always**
- Run every lane in `qgd`; rebuild after any C++ or CMake change, and run the optional C++ tests when a slice touches C++.
- Write each requirement's failing test first, negative included; pin every rejection by class and all three tokens.
- Check the full state against the bridge-order reference before quoting any energy; record \(\lVert H\rVert_F\), \(\tau_E\), span budget, and vector id per row.
- Derive the realized route from the labels; name the lane in every evidence row; keep q6–q10 and Aer rows out of the fast lane.
- Use this glossary in code, tests, and records; keep the density path additive and opt-in; run `specs_check.sh` before claiming done.

**Ask first**
- Changing a frozen tolerance, the form of \(\tau_E\), the counted span budget, the route set, or the anchors, vectors, or seed once counted data exist.
- Adding a gate, channel, source, width, or route to the matrix, or reclassifying a matrix row.
- Changing a delivered public signature or a C++ VQE exception type or message, or adding an observable argument to the entry.
- Adding a required dependency, importing Qiskit outside the Aer lane, or letting the C++ optimizer call Python.
- Touching legacy `NoiseChannel` behavior or the parametric-rate clamps, which M10 owns.

**Never**
- Fall back silently — to the state-vector path, an unfused route, or another route — or return a partial result after a rejection.
- Count a vacuous row, copy the realized route from the request, or compare energies without the full state.
- Use timing in any M4 claim, or claim a speedup.
- Loosen, skip, or delete a test or tolerance; edit a delivered bundle or the frozen archive; cite a row the named lane does not regenerate.
- Edit `docs/specs/**` during code generation; record a contract gap as `task-<n>/STEP_4A_HANDBACK.md`.

## 7. Assumptions and open questions

**Owner decisions (2026-09-30).** QA-002's \(\lvert\Delta E\rvert\le10^{-10}\) is frozen; \(\tau_E(n)=\lVert H\rVert_F\cdot10^{-10}\); the bridge-order
reference is the counted oracle; counted rows use span budget 2 and must be non-vacuous; 16 vectors per width; a caller-set initial state is
rejected; delivered public-VQE rejections are pinned as delivered. At the checkpoint the owner confirmed the critique's identity-vector rename
and the 8 boundary-rate anchors (72 counted anchors), and froze v0.1.

**Assumptions**
- *Current state (2026-09-30 exploratory probe, non-counted).* At span 2 each width has 2 channel-native-eligible partitions, `fused` fuses
  4/6/8/10 islands at q4–q10, and `strict` raises `channel_native_noise_presence`; at the Phase 3 calibrated span 4 no partition is eligible.
  \(\lVert H\rVert_F\) is 6.325, 16.248, 38.367, 86.902. Delivered and probe residuals sit far inside every bound: partitioned vs sequential 0,
  route vs public energy ≤ 4.1e-15 including rates 0 and 1, Phase 2 Aer energy ≤ 1.8e-14 at q10. A planner change re-checks REQ-004 first.
- *Oracle independence.* The bridge and the public energy's lowering are separate C++ loops emitting the same order; Aer is independent of both.
- *Aer availability.* Aer's density-matrix method runs at q10 in `qgd`, as in Phase 2.
- *Claim boundary.* Other widths, rates, and schedules within the delivered vocabulary are non-claim-bearing; counting them needs a later milestone.

**Open questions (each closed by a Layer 1 decision)**
1. Where does the attributed-energy entry live, and what are its public route names and record schema version? (ADR candidate)
2. Is the span budget caller-selectable? If so, non-2 values are non-counted and values below the widest gate raise the delivered `partition_span` error.
3. Is the runtime state's \(\mathrm{Re}\,\mathrm{Tr}(H\rho)\) evaluated by the public energy's C++ evaluator, newly bound, or in Python? REQ-003 binds either. (ADR candidate)
4. How does the entry read the instance's Hamiltonian and detect a caller-set initial state, when the Python wrapper exposes neither? (ADR candidate)
5. What are the route-label strings, and where outside `docs/specs/` is the human-readable matrix published?
6. How do the roadmap's 2–3 slices split the REQ-006 table and the boundary-rate anchors, after the q4 `partitioned` tracer?
7. Which fresh seed generates the five uniform draws? It is frozen before counted data.

## Critique (adversarial pass, 2026-09-30)

`sdd-critic` plus an author pass raised 8 blocking and 1 non-blocking findings; a confirmation pass raised 2 more blocking (the owner decided the
boundary-rate anchors; REQ-008 now pins only publicly reachable classes) and 1 non-blocking. All are resolved:
- *Contradictions.* Observability demanded `to_dict()` for errors pinned as bare exceptions (now scoped); an unknown direct `runtime_path` ran the baseline (now a REQ-006 row; no caller passes one).
- *Untestable negatives.* Validator failures named no error (now frozen verdict codes); the planted-defect sentinel was vague (now q4 `set_00` with a \(10^{-6}\) witness; probe \(6.9\times10^{-4}\)).
- *Incomplete inventory.* REQ-008 now pins every publicly reachable rejection class from the four delivered validators' raise sites, not a hand-picked list.
- *Unmeasurable NFRs.* The 60 s lane budget and the security row had no provable measure and were removed; lane placement moved to Always.
- *False boundary.* Zero is interior to \([-\pi,\pi)\); QA-002's boundary points now come from rates 0 and 1 (probe: state ≤ 5.7e-16, energy ≤ 4.1e-15, hybrid non-vacuous).
- *Proof gaps.* Python reference rows cannot prove the direct C++ API (that row now cites the optional C++ tests); displayed \(\tau_E\) values are marked approximate.
- *Author checks.* \(\lVert H\rVert_F\cdot10^{-12}\le8.7\times10^{-11}<10^{-10}\) at q10, so QA-002's two bounds agree; non-mutation is proven on bridge metadata; all nine traces are covered. Residual risk: A6 and the eligibility facts, re-checked before counted data.

## Handoff

`spec-driven-development` Steps 1–3 ingest this file into Layer 1: `DETAILED_PLANNING_CANONICAL_ATTRIBUTED_ENERGY.md`, `ADRS_CANONICAL_ATTRIBUTED_ENERGY.md`
(open questions 1, 3, 4), and `PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md` (questions 2, 5–7), seeding the evidence matrix from REQ-001 to REQ-011 and
the NFR table, naming the q4 `partitioned` slice tracer, and ending with an implementation-ready verdict. Current-state docs are updated at M4 close.

## Change log

- **v0.1 (2026-09-30)** — Initial baseline from product statement v0.4 and roadmap v0.7: seven owner decisions, a critique and a confirmation pass, QA-002's energy bar frozen at \(10^{-10}\); frozen by the owner at the checkpoint.
