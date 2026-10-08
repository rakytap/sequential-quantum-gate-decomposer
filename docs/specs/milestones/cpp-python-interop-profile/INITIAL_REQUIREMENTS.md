# Initial requirements — M-F5a `cpp-python-interop-profile`
> **Milestone:** M-F5a `cpp-python-interop-profile` · **Status:** draft v0.1 ·
> **Owner skill:** `create-initreq-for-sdd` ·
> **Upstream:** `PRODUCT_STATEMENT.md` at `ceb469c8` — CAP-004, CAP-007 ·
> QA-007, QA-008, QA-009; `ROADMAP.md` at `1cb3d20c` — M-F5a ·
> **Authorized scope:** requirements baseline only; Layer 1 and implementation stay closed ·
> **Downstream:** `spec-driven-development` Steps 1–3; Step 4a only if the Tech Lead opens the tracer; Step 4b waits on a code-ready verdict ·
> **RM edits locked here:** E1 R-oracle default exclude; E2 QA-007 10 % bar stays `[confirm]`; E3 harness-only lower boundary

## 1. Milestone scope and vision

- **Milestone:** M-F5a `cpp-python-interop-profile`. The delivered M3A disclosure stays 17/9/0.
  Registration task 3 (interop optimization). Partitioning and fusion remain task 2.
- **Upstream traceability:** CAP-004 low-overhead iterative evaluation and CAP-007 regenerable
  evidence; QA-007, QA-008, and QA-009.
- **Outcome:** at 4, 6, and 8 qubits, each public energy entry with an equal-work
  lower-boundary comparator receives at least 1000 warmed paired or interleaved calls,
  component attribution, uncertainty, and a QA-007 verdict; every route without that
  comparator receives attribution only.
- **In scope:** language-boundary measurement; per-operation nanoseconds per density-matrix
  entry, with uncertainty, on each measured path; a comparable-tier harness where an
  equal-work comparator exists; attribution where it does not; at most one binding or
  dispatch reduction, and only when that crossing is the material term; current-state doc
  updates at close.
- **Out of scope:** kernel rewrites; fusion redesign; noise-aware partitioning as a
  deliverable; AVX, GPU, and M-F5b; optimizer loops and VQA training; new channels; an M-F1b
  timing commit; the M-F2 cost model; M-F3 and M-F4; any speedup or at-least-1.2× sentence;
  GitHub Actions green as a gate; a push or pull request unless newly authorized; an edit of
  ADR-F1A-006; a new public Python energy API; quoting the historical 12–15 ns figure as an
  M-F5a result; labelling a diagnosis "QA-007 met".
- **RM accept edits:**
  - **E1.** R-oracle (`execute_sequential_density_reference`) is excluded from the attribution set by default. Step 4a may re-include it only to label C++ `apply_to` for an E-VQE diagnosis, and the bundle must say so. It is never a fifth advertised route.
  - **E2.** The QA-007 10 % bar stays `[confirm]`. This draft cites QA-007 and does not lock the product-owner bar. The product owner freezes that bar in this file, the first `INITIAL_REQUIREMENTS.md` that cites QA-007.
  - **E3.** The E-VQE lower boundary is a harness-only call into the same C++ density `optimization_problem` branch. This milestone adds no public Python energy API.
- **Target users:** researchers separating crossing cost from kernel cost; maintainers of the
  state-vector default; reproducers of one named lane.
- **Core problem:** later timing records and any interop reduction need a measured split
  between the Python crossing and the C++ kernel. A4 is likely false on the density energy
  entry and is measured before CAP-004 is treated as an optimization theme.
- **Success metrics:** E-VQE at 4, 6, and 8 carries the protocol, components, uncertainty, and a met or unmet-with-diagnosis verdict. The four attribution-only routes publish no overhead ratio. Each measured path publishes nanoseconds per density-matrix entry per operation and the comparison method. The bundle records zero or one binding or dispatch reduction. The M3A disclosure stays 17/9/0.
- **Depends on:** the M-F1a outcome, completeness recorded at `1b123a9a`.
- **Holds:** direction remains NARROW; 17/9/0 unchanged; no VQA campaign; semester CI for a
  later code close is rocky-local Tester CI; N8 stays deferred.
- **Current-state context:** CPU-only dense complex128 execution in `qgd`. Density is opt-in; the default backend is state vector. `ARCHITECTURE_OVERVIEW.md` and `TECH_STACK.md` exist and are updated at close. The new sibling lane leaves the Phase 3 median-of-three performance-evidence record and `benchmark_perf.py` untouched.

## 2. User journeys

- **Primary journey:** a researcher runs the named M-F5a lane in `qgd` at the pinned revision
  and receives a validated bundle. Each E-VQE row at 4, 6, and 8 reports at least 1000 warmed
  paired or interleaved calls, \(O\) and its one-sided 95 % upper bound, the component split,
  per-operation nanoseconds per density-matrix entry, and a verdict. Each attribution-only
  route reports orchestration time, the apply component, the same throughput, and uncertainty.
- **Reproduction journey:** the same command from a clean checkout reproduces categorical
  labels exactly and performance decisions against the bundle's stated margin, under the
  pinned host, CPU, compiler flags, dependencies, threading, affinity, warm-up, and sampling.
- **Diagnosis and refusal paths:** a missed frozen bar closes that row as unmet-with-diagnosis
  and withholds "QA-007 met". An upper bound below 5 % at every 4/6/8 point leaves CAP-004
  hold-the-line and omits the reduction. An excluded entry, an overhead ratio without an
  equal-work comparator, or R-oracle listed as an advertised route fails validation. A changed
  default backend, or a later code close without rocky-local Tester CI, blocks close. GitHub
  Actions green is not that record.

## 3. Ubiquitous language

This milestone inherits the product glossary and uses these narrower terms:

| Term | Meaning in M-F5a |
|------|------------------|
| **Public energy entry** | A shipped Python call that returns a noisy energy scalar. A state-returning route is not an energy entry. |
| **E-VQE** | `Optimization_Problem` on `qgd_Variational_Quantum_Eigensolver_Base` with `backend="density_matrix"`, returning \(\mathrm{Re}\,\mathrm{Tr}(H\rho)\). The only counted entry with an equal-work comparator. |
| **Equal-work lower-boundary comparator** | The same prebuilt evaluator, invoked so that only the language crossing differs. For E-VQE the lower side is the E3 harness-only call. |
| **Overhead ratio** \(O\) | \((T_\mathrm{public}-T_\mathrm{lower})/T_\mathrm{public}\). Uncertainty is the one-sided 95 % upper bound on \(O\). |
| **Attribution-only route** | R-base, R-fused, R-strict, or R-hybrid: a shipped path that returns \(\rho\) and has no equal-work pair. These rows publish no \(O\). |
| **R-base / R-fused / R-strict / R-hybrid** | `execute_partitioned_density`, `execute_partitioned_density_fused`, `execute_partitioned_density_channel_native`, `execute_partitioned_density_channel_native_hybrid`. |
| **R-oracle** | `execute_sequential_density_reference`, the M-F1a sequential oracle. Default-excluded from the attribution set (E1). Never an advertised M-F5a route. |
| **Harness-only lower call** | A measurement-harness invocation of the same C++ density `optimization_problem` branch. No new public Python energy symbol (E3). |
| **Equal work** | In-call allocate and build are on both sides or on neither. A one-sided strip has no QA-007 ratio. |
| **Bounded reduction** | At most one later change, confined to binding or dispatch, and only when the crossing is the material term. |
| **Interop bundle** | The counted artifact of the named sibling lane. Inventory and protocol are frozen before counted trials. |
| **A4 kill** | Upper bound on \(O\) below 5 % at every 4/6/8 point: CAP-004 becomes hold-the-line and the reduction is not made. Distinct from the unfrozen 10 % bar. |
| **M3A disclosure** | Frozen Phase 3.1 result 17/9/0. Unchanged here. The frozen 26-case matrix is not the M-F5a workload. |

## 4. Requirements and acceptance criteria

### REQ-001 — Frozen measured-entry inventory
- **Upstream:** M-F5a · CAP-004, CAP-007 · QA-007, QA-008.
- **Acceptance:** Given the pinned revision and a pre-trial inventory review, when the suite
  is registered, then the counted set is E-VQE at widths 4, 6, and 8 for a QA-007 verdict,
  plus R-base, R-fused, R-strict, and R-hybrid for attribution only. R-oracle stays outside
  that set (E1). Outside the outcome: `Optimization_Problem_Batch`, `Optimization_Problem_Grad`,
  state-vector `Expectation_value_of_energy_real`, GQML `Optimization_Problem`,
  `NoisyCircuit.apply_to` as its own entry, and the helpers `density_energy` and
  `hermitian_energy_real`.
- **Negative/error:** Given an excluded symbol, an overhead ratio with no equal-work
  comparator, a missing width among 4, 6, and 8, R-oracle as a fifth advertised route, or an
  R-oracle row that does not say it exists only to label C++ `apply_to` for an E-VQE
  diagnosis, when the bundle validates, then validation fails and emits no completeness claim.
- **Evidence route:** interop-profile inventory validator in `qgd`, plus the pre-trial review.

### REQ-002 — E-VQE QA-007 verdict at 4, 6, and 8 qubits
- **Upstream:** M-F5a · CAP-004 · QA-007.
- **Acceptance:** Given one prebuilt E-VQE evaluator per width, outside the timed region, on
  the supported generated-HEA density anchor (U3/CNOT and ordered local depolarizing,
  amplitude damping, and phase damping), when the harness runs, then each of 4, 6, and 8
  receives at least 1000 warmed paired or interleaved calls. State reset, allocation, output
  materialization, batching, build, threading, affinity, and warm-up are frozen for the ratio.
  The bundle reports \(O\), its one-sided 95 % upper bound, and the components (CPython
  wrapper, in-call C++ allocate and build, `NoisyCircuit.apply_to`, sparse energy
  contraction), including per-operation nanoseconds per density-matrix entry with the same
  sampling uncertainty. The verdict is met or unmet-with-diagnosis against the bar the product
  owner has frozen. While that bar remains `[confirm]` (E2), the bundle reports the ratio and
  the bound and withholds "QA-007 met".
- **Negative/error:** Given an unmet bar, fewer than 1000 warmed calls, a dropped width, or a
  "QA-007 met" label while the bar is `[confirm]` or unmet, when validation runs, then that
  row fails. Unmet-with-diagnosis may be recorded. The row is not relabelled as met.
- **Evidence route:** `conda run -n qgd --no-capture-output python benchmarks/density_matrix/interop_profile/validation_pipeline.py`
  once that sibling lane exists. The Phase 3 performance-evidence pipeline is a different lane.

### REQ-003 — Harness-only equal-work lower boundary
- **Upstream:** M-F5a · CAP-004 · QA-007.
- **Acceptance:** Given the E-VQE pair, when both sides are timed, then \(T_\mathrm{public}\)
  is shipped Python `Optimization_Problem` on the density backend and \(T_\mathrm{lower}\) is
  a harness-only call into the same C++ `optimization_problem` density branch (E3). This
  milestone adds no public Python energy API. Allocate and build sit on both sides or on
  neither. Pairing choice, warm-up count, affinity, and thread count are pinned in the bundle.
- **Negative/error:** Given a new public Python energy symbol, or a pair that strips allocate
  or build on one side only, when validation runs, then that pair is not equal-work. The entry
  is attribution-only, and a bundle that still publishes a QA-007 ratio for it fails.
- **Evidence route:** the interop-profile pair witness. Step 4a names the exact harness-only
  call site before any timed claim.

### REQ-004 — Attribution-only routes and per-operation throughput
- **Upstream:** M-F5a · CAP-004, CAP-007 · QA-008.
- **Acceptance:** Given R-base, R-fused, R-strict, and R-hybrid at the same widths and frozen
  workload shape, when the lane runs, then each row reports Python orchestration time, the
  apply component, per-operation nanoseconds per density-matrix entry, and uncertainty.
  R-base's apply component is C++ `NoisyCircuit.apply_to`. R-strict's is numpy Kraus.
  R-hybrid's label follows the executed class. No row publishes \(O\).
- **Negative/error:** Given an invented lower-boundary twin, an overhead ratio on any of these
  four routes, numpy Kraus time reported as C++ kernel time, or an R-oracle row without the
  E1 diagnosis sentence, when validation runs, then the bundle fails.
- **Evidence route:** interop-profile attribution records, in `qgd`.
- **Amendment ADR-F5A-011 (2026-10-08).** Where the strict contract refuses under the frozen workload, a required refusal row with recorded diagnosis satisfies REQ-004 for R-strict. The row carries no timings, ns/op, UB or O. Inventing strict timings or changing the anchor workload's noise remains forbidden.

### REQ-005 — At most one binding or dispatch reduction
- **Upstream:** M-F5a · CAP-004 · QA-007.
- **Acceptance:** Given a completed profile, when the milestone closes, then the change set
  has no interop reduction or exactly one reduction confined to binding or dispatch, and the
  bundle says which. The reduction is present only when the crossing is the material term.
  Under the A4 kill the reduction is absent and the bundle states that CAP-004 is
  hold-the-line. Kernel, fusion, and planner sources stay unchanged either way.
- **Negative/error:** Given a kernel rewrite, fusion redesign, AVX or GPU change, a second
  reduction, a planner change, or a reduction under the A4 kill, when closeout review runs,
  then the milestone does not close.
- **Evidence route:** interop-bundle claim boundary and the closeout diff review.

### REQ-006 — Revision-pinned interop bundle
- **Upstream:** M-F5a · CAP-007 · QA-008.
- **Acceptance:** Given a clean checkout at the recorded revision in `qgd`, when the named
  lane runs, then it regenerates `benchmarks/density_matrix/artifacts/interop_profile/`,
  exits nonzero on any missing or failing requirement, reproduces categorical results exactly,
  and reproduces performance decisions against the stated margin on 100 % of counted rows.
  The bundle pins revision, host, CPU, compiler and flags, dependencies, threading and
  affinity, warm-up, sampling protocol, workload, and claim boundary. `performance_evidence`
  rows and `benchmark_perf.py` stay unchanged.
- **Negative/error:** Given a missing provenance field, a stale or partial artifact, an
  unpinned sampling choice, or a write into the Phase 3 median-of-three record, when
  validation runs, then the bundle fails and cannot support the outcome.
- **Evidence route:** the REQ-002 command in `qgd`, plus a diff check that the M-F1b
  performance-evidence record is untouched.

### REQ-007 — State-vector non-interference and rocky-local CI
- **Upstream:** M-F5a · QA-009.
- **Acceptance:** Given the M-F5a change set, when a later code close records
  non-interference, then the default backend remains state vector, density stays opt-in, and
  the semester record is rocky-local Tester CI. This milestone does not rewrite ADR-F1A-006.
- **Negative/error:** Given a changed default backend, a state-vector regression, a close
  that treats GitHub Actions green as the semester gate, or an edit to ADR-F1A-006, when
  review runs, then the milestone does not close.
- **Evidence route:** rocky-local Tester CI at the later code close, plus pinned
  default-backend checks in `qgd`. This baseline has no code-close record yet.

### REQ-008 — Historical disclosure and anti-claims stay intact
- **Upstream:** M-F5a · CAP-007 · QA-008.
- **Acceptance:** Given M-F5a evidence, when it is published, then the M3A disclosure remains
  17/9/0, `docs/density_matrix_project/archive/` is unchanged, the frozen 26-case matrix is
  not the workload, and the bundle and closeout contain no speedup sentence, no at-least-1.2×
  sentence, and no use of the historical 12–15 ns figure as an M-F5a result.
- **Negative/error:** Given an edit, recount, or relabel of a frozen-matrix row, or a speedup
  or at-least-1.2× claim, when review runs, then validation fails.
- **Evidence route:** archive diff, bundle text checks, and closeout review.

### REQ-009 — Current-state documentation matches the delivered lane
- **Upstream:** M-F5a · CAP-007 · QA-008, QA-009.
- **Acceptance:** Given the executable gates pass, when the milestone closes, then the
  existing `ARCHITECTURE_OVERVIEW.md` and `TECH_STACK.md` name the interop lane, the
  measured-entry inventory, the claim boundary, and the rocky-local CI record. `ROADMAP.md`
  stays for `create-product-roadmap` revalidation after the closeout.
- **Negative/error:** Given a missing command, an omitted delivered boundary, or a speedup or
  GitHub Actions close gate in either current-state document, when strict spec checks and
  closeout review run, then M-F5a does not close.
- **Evidence route:** `bash .cursor/skills/spec-driven-development/scripts/specs_check.sh --strict`
  plus current-state-doc review at close.

## 5. Non-functional requirements

- **Interop overhead — QA-007:** E-VQE at 4, 6, and 8; at least 1000 warmed paired or
  interleaved calls; \(O\) on an equal-work pair; components and per-operation nanoseconds per
  density-matrix entry, with uncertainty. The one-sided 95 % upper bound on \(O\) is at most
  10 % `[confirm]`, owned by the product owner (E2). This draft does not lock that bar. A
  diagnosis may close the research milestone and leaves QA-007 unmet. The A4 kill stays the
  product-statement 5 % test. QA-007 bar frozen at 10% (product-statement default), RM ALIGN 2026-10-07, ratified by Zoltán as product owner 2026-10-08.
- **Reproducibility — QA-008:** 100 % of counted rows regenerate categorical results exactly;
  performance decisions reproduce against the stated margin under REQ-006 provenance. Every
  claim row names the interop lane.
- **Non-interference — QA-009:** at the later code close, zero state-vector regressions, the
  state-vector default unchanged, density still opt-in, rocky-local Tester CI as the semester
  record, and no new runtime dependency.
- **Scale:** widths 4, 6, and 8 remain, including width 8 at at least 1000 calls. A later
  tracer may start at 4. Widths 6 and 8 stay in the outcome.
- **Security / privacy / accessibility / i18n:** no new data obligation.
- **Architecture and tech-stack documentation:** both current-state documents update only
  after executable evidence passes.

## 6. Operational boundaries

- **Always do:** run product lanes in `qgd`; freeze inventory, workload class, equal-work
  pair, and sampling before counted trials; keep R-oracle out of the attribution set unless
  the bundle records the E1 diagnosis exception; publish no overhead ratio on attribution-only
  routes; leave 17/9/0 and the archive unchanged; update both current-state documents at
  close; run both spec checks.
- **Ask first:** freezing or changing the QA-007 10 % bar (product owner); re-including
  R-oracle; choosing the harness-only call site; changing a frozen workload or sampling
  protocol after trials start; adding a public Python energy API; taking the bounded
  reduction; adding a runtime dependency.
- **Never do:** add a public Python energy API; publish \(O\) on an attribution-only route;
  advertise R-oracle; label an unmet or still-`[confirm]` row "QA-007 met"; rewrite a kernel,
  fusion, or the planner; add AVX, GPU, a channel, an optimizer loop, or a VQA campaign;
  retcon M-F1b timing rows; edit ADR-F1A-006; quote 12–15 ns as an M-F5a result; state a
  speedup or an at-least-1.2× claim; treat GitHub Actions green as the semester gate; extend
  the frozen archive; loosen, skip, or delete a test to make a lane pass.

## 7. Assumptions and open questions

### Assumptions
- RM ACCEPT-WITH-EDITS (2026-10-06) authorizes this baseline and then
  `spec-driven-development` Steps 1–3. Step 4a stays closed until the Tech Lead opens the
  tracer. Step 4b waits on a code-ready verdict. This authorization includes no push and no
  pull request.
- Direction stays NARROW. The symbol inventory was read at `1b123a9a`. Step 4a may correct a
  call-site contradiction only by citing it. The exclude list otherwise stays.
- Widths are 4, 6, and 8, including width 8 at at least 1000 calls. The workload class is the
  generated-HEA density anchor in `ARCHITECTURE_OVERVIEW.md` flow 1. The frozen 26-case
  matrix is not that workload. The throughput divisor is \(4^n\) complex elements of \(\rho\)
  unless Step 4a records another before counted trials.
- The 5 % A4 kill is already in the product statement. Citing it here does not freeze the
  10 % bar. This file is the first `INITIAL_REQUIREMENTS.md` that cites QA-007, and the bar
  stays `[confirm]` until the product owner records the freeze here.

### Open questions
- **E2 bar.** Product owner: freeze the one-sided 95 % upper bound on \(O\) at most 10 %, or
  at another number, in this file before counted trials. Until then REQ-002 withholds "QA-007 met".
- **E3 call site.** Step 4a: which harness-only invocation of the C++ density
  `optimization_problem` branch is \(T_\mathrm{lower}\), with allocate and build on both sides
  or on neither? E3 already locks harness-only, the same branch, and no new public Python energy API.
- **E1 diagnosis label.** Step 4a: does E-VQE diagnosis need an R-oracle row solely to label
  C++ `apply_to`? The default remains exclude.
- **Workload numbers.** Step 4a, before counted trials: which depth and noise schedule are
  frozen per width? Widths remain 4, 6, and 8.
- **Protocol pins.** Step 4a: paired or interleaved, warm-up count, affinity, thread count,
  and the divisor, each pinned in the bundle under QA-008.

### Critique
- A4 moves the most scope. REQ-005 keeps any reduction behind the profile and the 5 % kill.
- The 10 % bar is untestable until the product owner freezes it. An early "met" label fails
  (E2, REQ-002). A one-sided build strip is not equal work (REQ-003). An R-oracle row without
  the E1 sentence fails (REQ-001, REQ-004).
- The interop lane is named for close and is absent in this slice (REQ-002, REQ-006).
  CAP-004, CAP-007, QA-007, QA-008, and QA-009 each have a `REQ-*`. No other CAP or QA is in scope.

### Handoff
Does this specification capture the intent for M-F5a? Which acceptance criteria are wrong,
missing, or need tighter edge-case coverage before a technical plan or implementation tasks
are generated?

Steps 1–3 of `spec-driven-development` ingest this file into
`DETAILED_PLANNING_CPP_PYTHON_INTEROP_PROFILE.md`, `ADRS_CPP_PYTHON_INTEROP_PROFILE.md`, and
`PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md`. Step 4a starts only if the Tech Lead opens the
tracer: E-VQE at 4 qubits, spec-only, ending in an explicit code-ready or not-ready verdict.
Widths 6 and 8 and the attribution-only routes are later slices. Step 4b waits on that
verdict. Consult the Research Manager again only if a later draft changes the equal-work
pair, adds an advertised energy entry, publishes \(O\) on an attribution-only route, or
proposes a kernel, fusion, or AVX change.

### Change log
- **v0.1 (2026-10-06):** initial baseline from roadmap `1cb3d20c`, product statement
  `ceb469c8`, and RM ACCEPT-WITH-EDITS (E1, E2, E3). QA-007 is cited with the 10 % bar still
  `[confirm]`.
- **2026-10-08, ADR-F5A-011:** REQ-004 amended for R-strict only. The original acceptance stays; the amendment line follows it. Trigger: Zoltán's option C1 sign-off via PhD Manager, recorded in `CHANGE_CONTROL.md`.
- **2026-10-08, milestone close:** QA-007 bar frozen at 10% (product-statement default), RM ALIGN 2026-10-07, ratified by Zoltán as product owner 2026-10-08. Zoltán, 13:01 CEST (UTC+2), via PhD Manager: "Yes, I ratify the 10% Python-overhead bar as frozen for M-F5a." No `REQ-*` text is removed.
