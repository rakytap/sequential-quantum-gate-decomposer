# Detailed planning — M-F5a `cpp-python-interop-profile`
> **Status:** Layer 1 v0.1 — ready for the Step 4a gate; implementation remains closed ·
> **Milestone:** M-F5a `cpp-python-interop-profile` ·
> **Owner skill:** `spec-driven-development` Steps 1–3 ·
> **Upstream:** `INITIAL_REQUIREMENTS.md` v0.1 (`06701153…`), product statement `ceb469c8`,
> roadmap `1cb3d20c`, RM ACCEPT-WITH-EDITS 2026-10-06 (plan `0dc77fec…`) ·
> **Traces:** REQ-001…009 · CAP-004, CAP-007 · QA-007, QA-008, QA-009 ·
> **Authorization:** Layer 1 only. Step 4a waits on the Tech Lead. Step 4b waits on code-ready ·
> **Decisions:** `ADRS_CPP_PYTHON_INTEROP_PROFILE.md` · **Readiness:** completion checklist ·
> **Baseline:** `feature/dm-perf-tuning` at `1b123a9a31235dd68d6c0a6ff9ba457c0112cd59`

## 1. Purpose and authority

M-F5a measures the language crossing on the density energy path and attributes time on the
shipped routes that return \(\rho\). Where an equal-work lower-boundary comparator exists,
the milestone reports an overhead ratio, uncertainty, components, and a QA-007 verdict.
Where it does not, the milestone reports attribution only.

Authority descends in this order: `PRODUCT_STATEMENT.md` → `ROADMAP.md` M-F5a →
`INITIAL_REQUIREMENTS.md` v0.1 → this plan and its ADRs → a future slice the Tech Lead
opens. Layer 1 may clarify the v0.1 baseline. It does not weaken E1, E2, or E3, and it
does not freeze the QA-007 10 % bar. The requirements header limits that file's own
writing scope; its handoff section is what sends Steps 1–3 here. These files authorize
no product code, no `task-1/` tree, and no push or pull request.

Registration task 3 is interop optimization. Partitioning and fusion remain task 2.
The milestone is not noise-aware partitioning.

## 2. Scope and non-goals

### In scope
- Language-boundary measurement of Python orchestration against the C++ density kernel.
- Per-operation nanoseconds per density-matrix entry, with uncertainty, on each measured path.
- A comparable-tier harness for the one public energy entry that has an equal-work pair.
- Attribution on the four shipped routes that return \(\rho\) and have no such pair.
- At most one binding or dispatch reduction, and only when the crossing is the material term.
- Updates, at milestone close only, to the existing `ARCHITECTURE_OVERVIEW.md` and
  `TECH_STACK.md`.

### Out of scope
- Kernel rewrites, fusion redesign, and noise-aware partitioning as a deliverable.
- AVX, GPU, and M-F5b. Optimizer loops and VQA training. New channels.
- An M-F1b timing commit, the M-F2 cost model, and M-F3 or M-F4.
- Any speedup sentence or at-least-1.2× sentence. GitHub Actions green as a gate.
- A push or pull request. An edit of ADR-F1A-006. A new public Python energy API.
- Quoting the historical 12–15 ns figure as an M-F5a result. Labelling a diagnosis
  "QA-007 met".

### Representation boundary
E-VQE stays the shipped Python `Optimization_Problem` on
`qgd_Variational_Quantum_Eigensolver_Base` with `backend="density_matrix"`. The lower
side is a harness-only call into the same C++ density branch. M-F5a adds no public
Python energy symbol.

## 3. Success conditions and stop rule

M-F5a succeeds only when all of the following hold together:

1. The counted set is E-VQE at widths 4, 6, and 8, plus R-base, R-fused, R-strict, and
   R-hybrid. R-oracle stays outside that set unless a row carries the E1 diagnosis sentence.
2. Each E-VQE width has at least 1000 warmed paired or interleaved calls, \(O\), its
   one-sided 95 % upper bound, the four components, per-operation nanoseconds per
   density-matrix entry, and a verdict of met or unmet-with-diagnosis. While the QA-007
   bar stays `[confirm]`, the bundle withholds "QA-007 met".
3. The four attribution-only routes publish orchestration time, the apply component,
   the same throughput, and uncertainty. They publish no overhead ratio.
4. The change set contains no interop reduction, or exactly one reduction confined to
   binding or dispatch. The A4 kill leaves the reduction absent and records CAP-004 as
   hold-the-line. Kernel, fusion, and planner sources stay unchanged either way.
5. One named sibling lane regenerates the interop bundle with REQ-006 provenance.
   `performance_evidence` and `benchmark_perf.py` stay unchanged.
6. The default backend remains state vector, density stays opt-in, and a later code close
   records rocky-local Tester CI. ADR-F1A-006 stays unedited.
7. The M3A disclosure stays 17/9/0. The archive stays unchanged. Both current-state
   documents name the delivered lane.

If the equal-work pair strips in-call allocate or build on one side only, that entry has
no QA-007 ratio and is reported attribution-only. If no harness can enter the same C++
`optimization_problem` density branch without a new public Python energy API, work stops
and returns to the Research Manager. The milestone does not invent that API.

## 4. Frozen contracts

| Contract | Frozen Layer 1 value | Trace |
|----------|----------------------|-------|
| Counted inventory | E-VQE at 4, 6, and 8 for a QA-007 verdict; R-base, R-fused, R-strict, R-hybrid attribution-only | REQ-001; ADR-F5A-001 |
| R-oracle (E1) | Excluded by default. A later slice may add one row only to label C++ `apply_to` for an E-VQE diagnosis, and the bundle must say so. Never an advertised route | REQ-001, REQ-004; ADR-F5A-001 |
| Exclude list | `Optimization_Problem_Batch`, `Optimization_Problem_Grad`, state-vector `Expectation_value_of_energy_real`, GQML `Optimization_Problem`, `NoisyCircuit.apply_to` as its own entry, helpers `density_energy` and `hermitian_energy_real` | REQ-001 |
| Equal-work pair (E3) | \(T_\mathrm{public}\): Python `Optimization_Problem`. \(T_\mathrm{lower}\): harness-only entry at C++ `optimization_problem` on the density branch, without the CPython wrapper. No new public Python energy API | REQ-003; ADR-F5A-002 |
| Equal work | In-call allocate, build, and `validate_density_anchor_support` sit on both sides or on neither. A one-sided strip has no QA-007 ratio | REQ-003; ADR-F5A-002 |
| QA-007 bar (E2) | One-sided 95 % upper bound on \(O\) at most 10 % stays `[confirm]` until the product owner freezes it in `INITIAL_REQUIREMENTS.md`. Layer 1 does not lock it | REQ-002; ADR-F5A-003 |
| A4 kill | Upper bound on \(O\) below 5 % at every 4/6/8 point: CAP-004 is hold-the-line and the reduction is absent. Distinct from the unfrozen 10 % bar | REQ-005; ADR-F5A-003, ADR-F5A-005 |
| Verdict label | "QA-007 met" is withheld while the bar is `[confirm]` or unmet. Unmet-with-diagnosis may close the research milestone and leaves QA-007 unmet | REQ-002; ADR-F5A-003 |
| Protocol | At least 1000 counted calls after discarded warm-up; paired or interleaved; repeated single calls, not the batch API. Estimator, warm-up count, affinity, and thread count are pinned before counted trials | REQ-002, REQ-006; ADR-F5A-004 |
| Throughput divisor | \(4^n\) complex elements of \(\rho\), times the operations the timed apply executes, including ordered local noise. Step 4a may record another divisor before counted trials | REQ-002, REQ-004; ADR-F5A-004 |
| Components | CPython wrapper; in-call C++ allocate and build; `NoisyCircuit.apply_to`; sparse energy contraction | REQ-002 |
| Apply labels | R-base: C++ `NoisyCircuit.apply_to`. R-strict: numpy Kraus. R-hybrid: the executed class | REQ-004 |
| Reduction | Zero or one change, confined to binding or dispatch, and only when the wrapper is the material term at every counted width and the A4 kill has not fired. The tracer takes no reduction | REQ-005; ADR-F5A-005 |
| Lane | Sibling `benchmarks/density_matrix/interop_profile/validation_pipeline.py` and `benchmarks/density_matrix/artifacts/interop_profile/`. Absent until a later slice creates them | REQ-006; ADR-F5A-004 |
| Untouched records | `benchmarks/density_matrix/performance_evidence/` and `benchmarks/density_matrix/benchmark_perf.py` | REQ-006 |
| Workload class | Supported generated-HEA density anchor: U3/CNOT and ordered local depolarizing, amplitude damping, and phase damping. Depth and noise schedule per width are pinned before counted trials. The frozen 26-case matrix is not the workload | REQ-002, REQ-008; ADR-F5A-006 |
| Widths | 4, 6, and 8 remain in the outcome, including width 8 at at least 1000 calls. A tracer may start at 4 | REQ-002; ADR-F5A-006, ADR-F5A-008 |
| Non-interference | State-vector default, density opt-in, rocky-local Tester CI at a later code close. ADR-F1A-006 is not edited. N8 stays deferred | REQ-007; ADR-F5A-007 |
| Historical boundary | M3A disclosure 17/9/0. `docs/density_matrix_project/archive/` unchanged against `1b123a9a` | REQ-008 |
| Current-state docs | Both documents update at milestone close only. `ROADMAP.md` waits for roadmap revalidation | REQ-009; ADR-F5A-007 |

## 5. Current-state findings

Read at `1b123a9a`. These locations confirm the v0.1 inventory. They are not a license to
edit the sources in this step.

- Python `Optimization_Problem` is `squander/VQA/qgd_Variational_Quantum_Eigensolver_Base.py:271`.
  It forwards to the base wrapper. The wrapper function
  `qgd_Variational_Quantum_Eigensolver_Base_Wrapper_Optimization_Problem` begins at
  `squander/VQA/qgd_VQE_Base_Wrapper.cpp:1041`. Argument parsing, `numpy2matrix_real`
  (`:1072`), the C++ call (`:1076`), and `Py_BuildValue` (`:1091`) are the crossing.
- `Variational_Quantum_Eigensolver_Base::optimization_problem` (`:1088`) is a public C++
  method. On `DENSITY_MATRIX_BACKEND` it calls private `evaluate_density_matrix_backend`
  (header `:169`, before `public:` at `:174`). That helper allocates \(\rho\), constructs
  `NoisyCircuit`, lowers the anchor, calls `apply_to`, and contracts with private
  `expectation_value_of_density_energy_real` (`:369`).
- `validate_density_anchor_support` runs at the start of `optimization_problem` (`:1091`).
  The gradient path requests gradient support and the density backend refuses it.
  `Optimization_Problem_Batch` reaches `Optimization_Interface::optimization_problem_batched`
  (`Optimization_Interface.cpp:940`), which has no density-backend branch.
- R-strict apply is numpy Kraus: `execute_partition_channel_native` calls
  `_apply_kraus_bundle` (`noisy_runtime_channel_native.py`). It does not call
  `NoisyCircuit.apply_to`. R-oracle is `execute_sequential_density_reference`
  (`noisy_runtime_core.py:1011`), which calls `circuit.apply_to`.
- `performance_evidence/records.py` stores `timing_mode` `median_3`. `benchmark_perf.py`
  is a different protocol. Neither is the M-F5a lane. The interop sibling is not on disk.
- `ARCHITECTURE_OVERVIEW.md` and `TECH_STACK.md` exist. Planning does not edit them.
  Both need an update at close: the interop lane, the measured-entry inventory, the claim
  boundary, and the rocky-local CI record.

## 6. Architecture boundary map

| Boundary | Owns | Inbound port | Outbound evidence or dependency | Decision |
|----------|------|--------------|--------------------------------|----------|
| VQE density entry | Public energy scalar \(\mathrm{Re}\,\mathrm{Tr}(H\rho)\) | Python `Optimization_Problem` with `backend="density_matrix"` | Wrapper crossing into C++ `optimization_problem` | ADR-F5A-001, ADR-F5A-002 |
| CPython wrapper | Argument parse, array conversion, result boxing | Shipped Python call | The crossing measured by \(O\) | ADR-F5A-002, ADR-F5A-005 |
| C++ density branch | `optimization_problem` density path, private evaluate and contraction | Harness-only lower call; the wrapper's C++ call | Allocate, build, `apply_to`, sparse contraction | ADR-F5A-002 |
| Partitioned runtime | R-base, R-fused, R-strict, R-hybrid | Public `execute_partitioned_density*` | Attribution records; no overhead ratio | ADR-F5A-001 |
| Sequential oracle | R-oracle | M-F1a reference entry | Excluded, or one labelled `apply_to` witness | ADR-F5A-001 |
| Interop evidence | Inventory, trials, components, provenance, verdict | Named sibling pipeline, once a later slice creates it | `artifacts/interop_profile/` | ADR-F5A-004 |
| State-vector path | Default backend | Existing VQE tests; later rocky-local Tester CI | Non-interference verdict | ADR-F5A-007 |
| Current-state docs | Shipped architecture and runnable lanes | Milestone closeout | Reader-facing current truth | ADR-F5A-007 |

The dependency direction is interop evidence → harness → shipped Python entry or the same
C++ `optimization_problem` density branch. The harness is an anti-corruption boundary: it
may call that C++ method and must not become a public Python energy API. Historical
`performance_evidence` rows are consumed as context and are not rewritten. M-F5a adds no
service, storage system, or deployment unit.

## 7. Milestone goals

| Goal | Outcome | Acceptance evidence | Requirements |
|------|---------|---------------------|--------------|
| G1 — freeze the inventory | The counted set matches the frozen table, and excluded symbols stay out | Interop inventory validator, once the lane exists | REQ-001 |
| G2 — deliver the E-VQE verdict | Each of 4, 6, and 8 carries the protocol, \(O\), the bound, components, and a lawful verdict label | Interop bundle rows | REQ-002 |
| G3 — keep the pair equal-work | The lower call is harness-only into the same density branch, with allocate, build, and support validation on both sides or on neither | Pair witness in the bundle | REQ-003 |
| G4 — attribute the four routes | Each route reports orchestration, apply component, throughput, and uncertainty, and publishes no \(O\) | Attribution records | REQ-004 |
| G5 — bound any reduction | The diff has zero reductions, or one binding/dispatch reduction that the material-term rule allows | Closeout diff review | REQ-005 |
| G6 — regenerate one bundle | The sibling command pins provenance, fails closed, and leaves M-F1b records untouched | Interop pipeline and a diff of the M-F1b paths | REQ-006 |
| G7 — protect the state-vector default | Density stays opt-in. The later code close records rocky-local Tester CI | `Test_VQE.test_explicit_state_vector_matches_legacy_default` plus that CI record | REQ-007 |
| G8 — hold the historical boundary | 17/9/0 and the archive stay put. No speedup sentence appears | Archive diff and bundle text check | REQ-008 |
| G9 — publish current operational truth | Both current-state documents name the lane and the claim boundary | Strict spec check at close | REQ-009 |

No Layer 2–4 artifact is created here. ADR-F5A-008 states what the first slice is when the
Tech Lead opens it.

## 8. Traceability

| Requirement | Layer 1 interpretation | Goals | ADRs |
|-------------|------------------------|-------|------|
| REQ-001 | One QA-007 entry and four attribution-only routes; R-oracle excluded unless the E1 sentence is present | G1 | ADR-F5A-001 |
| REQ-002 | Protocol, components, and a met or unmet-with-diagnosis label; "QA-007 met" stays withheld while the bar is `[confirm]` | G2 | ADR-F5A-003, ADR-F5A-004, ADR-F5A-006 |
| REQ-003 | Harness-only lower entry at the same C++ density branch; a one-sided strip drops the ratio | G3 | ADR-F5A-002 |
| REQ-004 | Attribution and throughput on R-base, R-fused, R-strict, and R-hybrid; no \(O\) | G4 | ADR-F5A-001, ADR-F5A-004 |
| REQ-005 | Zero or one binding/dispatch reduction; A4 kill and a non-material wrapper forbid it | G5 | ADR-F5A-005 |
| REQ-006 | Sibling lane, full provenance, M-F1b records untouched | G6 | ADR-F5A-004 |
| REQ-007 | State-vector default and rocky-local Tester CI; ADR-F1A-006 unchanged | G7 | ADR-F5A-007 |
| REQ-008 | 17/9/0, frozen archive, no speedup and no 12–15 ns result sentence | G8 | ADR-F5A-006, ADR-F5A-007 |
| REQ-009 | Current-state documents move at close, after the executable gates pass | G9 | ADR-F5A-007 |

## 9. Milestone evidence matrix

The interop command below is the lane a later slice creates. It is not on disk at this
revision, so it is not a current runnable. Static rows use paths that exist today.

| Trace id | Evidence type | Command or gate | Lane | Expected result |
|----------|---------------|-----------------|------|-----------------|
| REQ-001, REQ-002, REQ-003, REQ-004, QA-007 | interop bundle | `conda run -n qgd --no-capture-output python benchmarks/density_matrix/interop_profile/validation_pipeline.py` | interop evidence pipeline, created later | exit 0; inventory matches ADR-F5A-001; E-VQE rows carry \(O\), the bound, components, and a lawful label; attribution rows carry no \(O\) |
| REQ-005 | closeout diff review | `git diff --exit-code 1b123a9a -- squander/src-cpp/density_matrix squander/partitioning` | doc review | empty at close; kernel, fusion, and planner trees stay unchanged. The closeout names zero reduction diffs, or one diff confined to binding or dispatch |
| REQ-006, QA-008 | provenance and non-retcon | the interop command above, plus `git diff --exit-code 1b123a9a -- benchmarks/density_matrix/performance_evidence benchmarks/density_matrix/benchmark_perf.py` | interop lane plus repo review | bundle pins REQ-006 fields; both diffs empty |
| REQ-007, QA-009 | default-backend pin and later CI | `conda run -n qgd --no-capture-output pytest tests/VQE/test_VQE.py::Test_VQE::test_explicit_state_vector_matches_legacy_default -q`; rocky-local Tester CI lane at the later code close | fast pytest now; CI lane at code close | legacy default matches explicit `state_vector`; density stays opt-in; GitHub Actions green is not the semester record |
| REQ-008 | historical boundary | `git diff --exit-code 1b123a9a -- docs/density_matrix_project/archive` | repo review | empty; bundle text has no speedup sentence and no at-least-1.2× sentence |
| REQ-009 | spec fitness | `bash .cursor/skills/spec-driven-development/scripts/specs_check.sh --strict docs/specs/milestones/cpp-python-interop-profile` | spec lint | zero errors and zero warnings on this tree; at close, current-state docs match the lane |

## 10. Operational boundaries

Layer 1 adopts `INITIAL_REQUIREMENTS.md` §6 unchanged.

- **Always:** run product lanes in `qgd`; freeze inventory, workload class, the equal-work
  pair, and sampling before counted trials; keep R-oracle out unless the E1 sentence is in
  the bundle; publish no overhead ratio on an attribution-only route; leave 17/9/0 and the
  archive unchanged; update both current-state documents at close; run both spec checks.
- **Ask first:** freezing the QA-007 10 % bar (product owner); re-including R-oracle;
  choosing the harness mechanism; changing a frozen workload or sampling protocol after
  trials start; adding a public Python energy API; taking the bounded reduction; adding a
  runtime dependency.
- **Never:** add a public Python energy API; publish \(O\) on an attribution-only route;
  advertise R-oracle; label an unmet or still-`[confirm]` row "QA-007 met"; rewrite a
  kernel, fusion, or the planner; add AVX, GPU, a channel, an optimizer loop, or a VQA
  campaign; retcon M-F1b rows; edit ADR-F1A-006; quote 12–15 ns as an M-F5a result; state
  a speedup or an at-least-1.2× claim; treat GitHub Actions green as the semester gate;
  extend the archive; loosen, skip, or delete a test to make a lane pass.

## 11. Risks and decision gates

- **A4 is likely false on E-VQE.** Measure anyway. If the upper bound on \(O\) is below
  5 % at every 4/6/8 point, CAP-004 stays hold-the-line and the reduction is not made.
- **The pair is unequal.** A one-sided strip of allocate, build, or support validation
  drops the QA-007 ratio. The entry is then attribution-only.
- **No harness-only entry exists.** Stop and consult the Research Manager. Do not add a
  public Python energy API to create one.
- **The 10 % bar is still `[confirm]`.** The product owner freezes it in
  `INITIAL_REQUIREMENTS.md` before counted trials that apply the bar. Until then the
  bundle reports \(O\) and the bound and withholds "QA-007 met".
- **R-strict has no C++ `apply_to`.** Numpy Kraus time is labelled numpy Kraus.
- **Width 8 with at least 1000 calls is large.** The outcome keeps width 8. The tracer
  may start at 4. The milestone does not drop 8.
- **A later binding edit still owes QA-009.** Rocky-local Tester CI is that record.
  ADR-F1A-006 is not rewritten here.
- **Editing `performance_evidence` collides with M-F1b.** The sibling lane is mandatory.
- **Descriptor mapping.** Attribution routes consume planner descriptors. A later slice
  uses the same depth, noise schedule, and parameter vector as E-VQE. If a descriptor
  cannot represent that anchor, the slice hands back. It does not invent a second workload.
- **Call-site contradiction.** Step 4a may correct a symbol location only by citing the
  contradiction. The exclude list otherwise stays.

## 12. Expected outcome

The delivered milestone is a revision-pinned interop bundle from the sibling lane. E-VQE
at 4, 6, and 8 carries the protocol, components, uncertainty, and a lawful QA-007 verdict.
The four routes carry attribution and no overhead ratio. The bundle records zero or one
binding or dispatch reduction. Current-state documents match that lane. The M3A disclosure
stays 17/9/0. No speedup claim is made.

## 13. What this layer does not open

The first slice, when the Tech Lead opens it, is E-VQE at 4 qubits only (ADR-F5A-008).
Widths 6 and 8, the attribution-only routes, and any reduction are later slices. This
layer does not create `task-1/`, a closeout, or `CHANGE_CONTROL.md`. Step 4b stays closed
until a code-ready verdict says the equal-work pair, the inventory, the no-\(O\) rule, and
the kernel/fusion/AVX boundary are unchanged.
