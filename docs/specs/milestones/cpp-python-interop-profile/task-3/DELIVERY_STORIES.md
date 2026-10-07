# Delivery stories — M-F5a task-3
> **Status:** Step 4a draft · **Verdict:** not-ready · **Slice:** M-F5a task-3 ·
> **Scope:** E-VQE at 8 qubits on the existing harness. No routes. No reduction ·
> **Traces:** REQ-001…009 · CAP-004, CAP-007 · QA-007, QA-008, QA-009 ·
> **Gate:** SDD stage `step-4a`. QA-007 stays `[confirm]`. Milestone not complete ·
> **RM:** awaits ALIGN ACCEPT. No Step 4b

### Delivery story: DS-1 — An 8-qubit E-VQE row on the same equal-work pair

**Stakeholder / system value**
- The 4/6/8 energy set gains its missing width without a second energy API and
  without a milestone-wide claim.

**Given / When / Then**
- Given the frozen 8-qubit HEA evaluator in `TASK_3_MINI_SPEC.md` §2, built once
- When an authorized harness run executes 50 discarded warm-up pairs and 1000 counted alternating pairs
- Then the row reports mean `O`, including when that mean is negative, the one-sided 95 % upper bound, min, max, median, the observational 20 µs tail count, the four components, and the mean per-operation `apply_to` throughput on divisor 1572864 with its one-sided 95 % upper bound
- And the label "QA-007 met" is absent while the 10 % bar stays `[confirm]`

**Scope**
- In: E-VQE at 8 qubits; the existing harness-only lower call; Measure carried from S-g
- Out: attribution routes; a new CPU mask; a counted run in this draft

**Acceptance signals**
- `parameter_count` 42, `operation_count` 24, `gate_count` 21, divisor 1572864
- No sample is dropped. A negative mean and `O_i` below −0.5 stay in the row
- Aer oracle runs after `set_Optimized_Parameters`. Flag-off versus flag-on bit identity is not that oracle
- `milestone_counted=false`. The artifact is `interop_profile_bundle_w8.json`

**Traceability**
- Initial requirement(s): REQ-001, REQ-002, REQ-003, REQ-006
- Capability / quality attribute: CAP-004, CAP-007 / QA-007, QA-008
- Milestone planning reference(s): goals G1, G2, G3, G6
- ADR(s): ADR-F5A-001, ADR-F5A-002, ADR-F5A-003, ADR-F5A-004, ADR-F5A-006, ADR-F5A-009

### Delivery story: DS-2 — Width 8 does not close the milestone

**Stakeholder / system value**
- The new row cannot be read as attribution evidence, a reduction, an A4 kill, or a finished QA-007 verdict.

**Given / When / Then**
- Given this slice's change set and the two committed bundles
- When the row and the diff are reviewed
- Then both committed bundles keep their pinned sha256 values, no attribution route is timed, and no reduction diff is present

**Scope**
- In: the exclusions in `TASK_3_MINI_SPEC.md` §5 and the write-refusal rule
- Out: implementing the four routes or the reduction

**Acceptance signals**
- A width-8 write aimed at `interop_profile_bundle.json` or `interop_profile_bundle_w6.json` is refused before any pair
- A claim of "A4 kill", "hold-the-line", or "reduction taken" fails validation. "no reduction" does not

**Traceability**
- Initial requirement(s): REQ-004, REQ-005, REQ-008
- Capability / quality attribute: CAP-004, CAP-007 / QA-008
- Milestone planning reference(s): goals G4, G5, G8
- ADR(s): ADR-F5A-001, ADR-F5A-005, ADR-F5A-006, ADR-F5A-008

### Delivery story: DS-3 — State-vector behavior stays the default

**Stakeholder / system value**
- The extra width must not become the default backend or a GitHub Actions gate.

**Given / When / Then**
- Given the later implementation, when non-interference is checked
- Then the legacy default backend is still state vector and density stays opt-in
- And current-state docs still wait for milestone close

**Scope**
- In: the existing state-vector pin
- Out: G-08 and G-09

**Acceptance signals**
- `test_explicit_state_vector_matches_legacy_default` still passes
- This draft does not claim QA-007 met and does not mark M-F5a complete

**Traceability**
- Initial requirement(s): REQ-007, REQ-009
- Capability / quality attribute: QA-009, QA-008
- Milestone planning reference(s): goals G7, G9
- ADR(s): ADR-F5A-007
