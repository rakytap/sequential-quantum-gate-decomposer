# Delivery stories — M-F5a task-1
> **Status:** code-ready · **Slice:** M-F5a task-1 ·
> **Scope:** E-VQE at 4 qubits. No reduction. No attribution routes. No widths 6 or 8 ·
> **Traces:** REQ-001…009 · CAP-004, CAP-007 · QA-007, QA-008, QA-009 ·
> **Gate:** code-ready. SDD stage `step-4b-authorized`. Step 4b has started; C1 is in flight (post-C1 fix revision)

### Delivery story: DS-1 — A 4-qubit E-VQE row with an equal-work ratio

**Stakeholder / system value**
- A researcher can see the language crossing on the one public density energy entry
  without a second energy API and without a milestone-wide claim.

**Given / When / Then**
- Given the frozen 4-qubit HEA evaluator in `TASK_1_MINI_SPEC.md` §2, built once
- When the harness runs 50 discarded warm-up pairs and 1000 counted alternating pairs
- Then the row reports mean `O`, the one-sided 95 % upper bound, the four components,
  the mean per-operation `apply_to` throughput and that same bound, and the QA-008 pins
- And the label "QA-007 met" is absent while the 10 % bar stays `[confirm]`

**Scope**
- In: E-VQE at 4 qubits; harness-only lower call; CLOCK_MONOTONIC pair
- Out: widths 6 and 8; batch and gradient entries; a returned energy from the harness

**Acceptance signals**
- Inventory matches ADR-F5A-001, with the batch exclusion reason corrected (F-1)
- `T_lower` is clocked inside C++ around `optimization_problem` on the same instance
- The timer flag and the six int64 fields are private members of `Variational_Quantum_Eigensolver_Base`. The wrapper calls one public C++ setter and one public C++ getter and returns no energy
- `support_outer` stays at the call before the density branch. Its clocks run only when the timer flag is on and the backend is density
- The four components partition `T_public` within the stated tolerance, after ADR-F5A-009
- The inner-timer flag is on for both sides of every warm-up pair and every counted pair
- The existing Aer node stays unchanged (`atol=1e-12` with NumPy's default `rtol=1e-5`, about 7.6e-6 at this cell). Flag-off versus flag-on bit-identity is the tight timer check
- The later counted run records `clean_start` true; this planning pack has no such run

**Traceability**
- Initial requirement(s): REQ-001, REQ-002, REQ-003, REQ-006
- Capability / quality attribute: CAP-004, CAP-007 / QA-007, QA-008
- Milestone planning reference(s): goals G1, G2, G3, G6
- ADR(s): ADR-F5A-001, ADR-F5A-002, ADR-F5A-003, ADR-F5A-004, ADR-F5A-006, ADR-F5A-009

### Delivery story: DS-2 — The tracer does not widen the milestone

**Stakeholder / system value**
- Later slices cannot mistake this cell for attribution evidence, a reduction, or a
  finished 4/6/8 verdict.

**Given / When / Then**
- Given this slice's change set
- When the row and the diff are reviewed
- Then no attribution-only route is timed, no overhead ratio is published for one,
  R-oracle is absent, and no binding or kernel diff is present
- And the claim boundary says `milestone_counted=false`

**Scope**
- In: the exclusions in the mini-spec §9 and the F-4 rules for later slices
- Out: implementing those later slices

**Acceptance signals**
- R-fused is not given an apply label here
- The material-term note (F-3) does not authorize a reduction

**Traceability**
- Initial requirement(s): REQ-004, REQ-005, REQ-008
- Capability / quality attribute: CAP-004, CAP-007 / QA-008
- Milestone planning reference(s): goals G4, G5, G8
- ADR(s): ADR-F5A-001, ADR-F5A-005, ADR-F5A-006, ADR-F5A-008

### Delivery story: DS-3 — State-vector behavior stays the default

**Stakeholder / system value**
- Measurement scaffolding must not become the default backend or a GitHub Actions gate.

**Given / When / Then**
- Given the later implementation, when non-interference is checked
- Then the legacy default backend is still state vector, density stays opt-in, and the
  semester record remains rocky-local Tester CI
- And current-state docs still wait for milestone close

**Scope**
- In: the existing state-vector pin and the ban on editing ADR-F1A-006
- Out: filing the rocky-local CI record (G-09) and editing `ARCHITECTURE_OVERVIEW.md`

**Acceptance signals**
- `test_explicit_state_vector_matches_legacy_default` still passes
- No workflow_dispatch or personal-access-token gate is added

**Traceability**
- Initial requirement(s): REQ-007, REQ-009
- Capability / quality attribute: QA-009, QA-008
- Milestone planning reference(s): goals G7, G9
- ADR(s): ADR-F5A-007
