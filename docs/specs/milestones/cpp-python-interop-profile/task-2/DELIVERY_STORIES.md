# Delivery stories — M-F5a task-2
> **Status:** code-ready · **Verdict:** Step 4b authorized · **Slice:** M-F5a task-2 ·
> **Scope:** E-VQE at 6 qubits on the task-1 harness. No width 8. No attribution routes. No reduction ·
> **Traces:** REQ-001…009 · CAP-004, CAP-007 · QA-007, QA-008, QA-009 ·
> **Gate:** SDD stage `step-4b-authorized`. QA-007 stays `[confirm]`. Milestone not complete ·
> **RM:** ACCEPT 2026-10-07. Developer not started. Binder `5bd81f9b…` at `c32b365e`

### Delivery story: DS-1 — A 6-qubit E-VQE row on the same equal-work pair

**Stakeholder / system value**
- A researcher can compare the language crossing at 6 qubits with the shipped 4-qubit
  row, on one harness, without a second energy API and without a milestone-wide claim.

**Given / When / Then**
- Given the frozen 6-qubit HEA evaluator in `TASK_2_MINI_SPEC.md` §2, built once
- When an authorized harness run executes 50 discarded warm-up pairs and 1000 counted alternating pairs
- Then the row reports mean `O`, including when that mean is negative, the one-sided
  95 % upper bound, the median, min, and max, the observational 20 µs tail count, the
  four components, and the mean per-operation `apply_to` throughput on divisor 73728
  with its one-sided 95 % upper bound
- And the label "QA-007 met" is absent while the 10 % bar stays `[confirm]`

**Scope**
- In: E-VQE at 6 qubits; the existing harness-only lower call; the Measure default for S-g
- Out: width 8; batch and gradient entries; a returned energy from the harness; a new CPU mask

**Acceptance signals**
- `parameter_count` 30, `operation_count` 18, `gate_count` 15, divisor 73728
- `T_lower` still comes from `harness_density_lower_ns` on the same instance
- The timer flag stays on for both sides of every warm-up pair and every counted pair
- No sample is dropped. A negative mean and `O_i` below −0.5 stay in the row. The tail count does not fail it
- Flag-off versus flag-on energy on this cell is bit-identical. The width-6 Aer oracle is `Test_VQE._get_density_backend_aer_reference` in the interop harness tests, with `|ΔE| ≤ 1e-12 + 1e-5·|E_Aer|`. The 4-qubit Aer node stays unchanged
- `milestone_counted=false`. The artifact is `interop_profile_bundle_w6.json`

**Traceability**
- Initial requirement(s): REQ-001, REQ-002, REQ-003, REQ-006
- Capability / quality attribute: CAP-004, CAP-007 / QA-007, QA-008
- Milestone planning reference(s): goals G1, G2, G3, G6 in `DETAILED_PLANNING_CPP_PYTHON_INTEROP_PROFILE.md`
- ADR(s): ADR-F5A-001, ADR-F5A-002, ADR-F5A-003, ADR-F5A-004, ADR-F5A-006, ADR-F5A-009

### Delivery story: DS-2 — The width-6 row does not consume the rest of the milestone

**Stakeholder / system value**
- The 4-qubit evidence stays intact, and later work cannot read this row as attribution,
  a reduction, width 8, or a finished QA-007 verdict.

**Given / When / Then**
- Given this slice's change set and both bundle files
- When the row and the diff are reviewed
- Then the task-1 bundle still hashes to `212f70386bf2a44711d29956c41bd3f0eea9ee2e284ace9c5403bc3d94ef934e`
- And no attribution route is timed, no overhead ratio is published for one, and no kernel, A4-kill, or reduction diff is present

**Scope**
- In: the exclusions in `TASK_2_MINI_SPEC.md` §6, the write-refusal rule, and the N-32 margin function
- Out: implementing width 8, the four routes, or the reduction

**Acceptance signals**
- `validate_interop_bundle` still rejects a width other than 4
- A width-6 write aimed at `interop_profile_bundle.json` is refused before any pair, and the refusal test uses a `tmp_path` copy
- `assert_mean_o_within_margin` enforces the 0.02 absolute margin on fixtures
- `performance_evidence/` and `benchmark_perf.py` are unchanged

**Traceability**
- Initial requirement(s): REQ-004, REQ-005, REQ-008
- Capability / quality attribute: CAP-004, CAP-007 / QA-008
- Milestone planning reference(s): goals G4, G5, G8
- ADR(s): ADR-F5A-001, ADR-F5A-005, ADR-F5A-006, ADR-F5A-008

### Delivery story: DS-3 — State-vector behavior stays the default

**Stakeholder / system value**
- The extra width must not become the default backend, a current-state-doc claim, or a
  GitHub Actions gate.

**Given / When / Then**
- Given the later implementation, when non-interference is checked
- Then the legacy default backend is still state vector, density stays opt-in, and the
  semester record remains rocky-local Tester CI
- And `ARCHITECTURE_OVERVIEW.md` and `TECH_STACK.md` still wait for milestone close

**Scope**
- In: the existing state-vector pin and the ban on editing ADR-F1A-006
- Out: filing the rocky-local CI record (G-09) and editing the current-state docs (G-08)

**Acceptance signals**
- `test_explicit_state_vector_matches_legacy_default` still passes
- No workflow_dispatch gate is added
- This slice does not claim QA-007 met and does not mark M-F5a complete. Stamp is code-ready at `c32b365e` (binder `5bd81f9b…`). The Developer is not started

**Traceability**
- Initial requirement(s): REQ-007, REQ-009
- Capability / quality attribute: QA-009, QA-008
- Milestone planning reference(s): goals G7, G9
- ADR(s): ADR-F5A-007
