# Delivery stories — M-F5a task-5
> **Status:** stamp draft · **Verdict:** awaiting APPROVE FOR STEP-4B · **Slice:** M-F5a task-5 ·
> **Scope:** three timed routes at widths 6 and 8. R-strict is a required refusal row at each width. No overhead ratio. No reduction ·
> **Traces:** REQ-001, REQ-004, REQ-005, REQ-006, REQ-007, REQ-008 · CAP-004, CAP-007 · QA-008, QA-009 ·
> **Gate:** uncommitted stage line `step-4b-authorized`. Developer not started. REQ-004 stays open. Milestone not complete ·
> **RM:** Q1b. ADR-F5A-010 wording amendment records `channel_native_noise_presence` at widths 6 and 8

### Delivery story: DS-1 — Width-6 and width-8 route rows

**Stakeholder / system value**
- The attribution inventory gains the three lawful routes at the two remaining widths, each with an honest R-strict refusal, and still without an overhead ratio.

**Given / When / Then**
- Given the width-6 and width-8 continuity anchors and ADR-F5A-010
- When a later authorized counted run calls the four entries at one width
- Then R-base, R-fused, and R-hybrid each report orchestration time, the apply component, and throughput whose numerator is that apply component divided by 73728 at width 6 or 1572864 at width 8, plus a one-sided 95 % bound on orchestration and on apply
- And the R-strict row has `status` `handback_refused`, cites `channel_native_noise_presence` and `98eec857`, and carries no number
- And `milestone_counted` is false

**Scope**
- In: the two bundles named in `TASK_5_MINI_SPEC.md` §2. The N-34 launch, including `unset PYTHONPATH`. The order is width 6, then its commit, then width 8. Each routes filename is bound to its width, and the committed width-4 routes file is not overwritten
- Out: R-oracle; a counted run in this draft; M-F1b

**Acceptance signals**
- A number on the R-strict row fails. A missing R-strict row fails. An overhead field on any row fails
- A reason that cites only `pure_unitary_partition` fails. The live raise code is `channel_native_noise_presence`

**Traceability**
- Initial requirement(s): REQ-001, REQ-004
- Capability / quality attribute: CAP-004, CAP-007 / QA-008
- Milestone planning reference(s): goals G1, G4
- ADR(s): ADR-F5A-001, ADR-F5A-004, ADR-F5A-010

### Delivery story: DS-2 — The new rows do not reduce or close the milestone

**Stakeholder / system value**
- Width-6 and width-8 route files cannot be read as a reduction, a finished four-route claim, or a finished milestone.

**Given / When / Then**
- Given this slice's change set and the four pinned bundles
- When the diff is reviewed
- Then `interop_profile_bundle.json` (`212f7038…`), `interop_profile_bundle_w6.json` (`5257bad2…`), `interop_profile_bundle_w8.json` (`1712dce9…`), and `interop_profile_bundle_routes_w4.json` (`6584be2b…`) are unchanged, and no reduction diff is present

**Scope**
- In: the exclusions in `TASK_5_MINI_SPEC.md` §5
- Out: binding or dispatch edits; current-state docs; M-F1b

**Acceptance signals**
- CAP-004 stays hold-the-line
- "M-F5a complete", "reduction shipped", "four-route shipped", and "REQ-004 met" are absent as claims

**Traceability**
- Initial requirement(s): REQ-005, REQ-006, REQ-008
- Capability / quality attribute: CAP-004 / QA-008
- Milestone planning reference(s): goals G4, G5, G8
- ADR(s): ADR-F5A-001, ADR-F5A-005, ADR-F5A-006

### Delivery story: DS-3 — State-vector behavior stays the default

**Stakeholder / system value**
- The extra rows must not become the default backend.

**Given / When / Then**
- Given a later implementation, when non-interference is checked
- Then the legacy default backend is still state vector and density stays opt-in

**Scope**
- In: the existing state-vector pin
- Out: G-08, G-09, Demo, and M-F1b

**Acceptance signals**
- `test_explicit_state_vector_matches_legacy_default` still passes
- This draft does not mark M-F5a complete

**Traceability**
- Initial requirement(s): REQ-007
- Capability / quality attribute: QA-009
- Milestone planning reference(s): goals G7, G9
- ADR(s): ADR-F5A-007
