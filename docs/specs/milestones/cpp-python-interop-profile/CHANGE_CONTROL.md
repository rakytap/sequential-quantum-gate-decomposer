# Change Control — REQ-004 refusal row satisfies R-strict
> **Governance evidence pack** for REQ-004 (R-strict under the frozen workload) and the success-check deferral (REQ-005 satisfied as no reduction). Traces: `REQ-004, REQ-005 → CAP-004 → roadmap assumption A4 → ADR-F5A-011`.
- **Status:** Approved · **Request date:** 2026-10-08 · **Owners:** Zoltán (decision); PhD Research Manager (pack); PhD Manager (relay); Tech Lead (applies) · **Related ADR:** ADR-F5A-011 (`ADR_AMENDMENTS_CPP_PYTHON_INTEROP_PROFILE.md`)
- **Milestone:** M-F5a `cpp-python-interop-profile` · **Scope:** REQ-004 for R-strict only. The milestone is not done · **Traces:** REQ-004, REQ-005 · CAP-004 · QA-007, QA-008

## 1. Request summary

| Item | Detail |
|------|--------|
| Baseline | REQ-004: R-base, R-fused, R-strict, and R-hybrid each report orchestration time, the apply component, per-operation ns, and uncertainty on the frozen workload |
| Reality | The frozen workload makes the strict contract refuse at 4, 6, and 8 (`channel_native_noise_presence`). ADR-F5A-010 already makes R-strict a required `handback_refused` row with no timings |
| Proposed | Option C1, below. Not a new workload and not invented timings |
| Scope of deviation | REQ-004 for R-strict only. R-base, R-fused, and R-hybrid keep the original acceptance |
| Outcomes preserved | The frozen anchor workload and its noise. No row publishes O. CAP-004 hold-the-line. `milestone_counted` false |

Where the strict contract refuses under the frozen workload, a required refusal row with recorded diagnosis satisfies REQ-004 for R-strict. The row carries no timings, ns/op, UB or O. Inventing strict timings or changing the anchor workload's noise remains forbidden.

## 2. Evidence supporting the request

| Evidence | Command / artifact | Result |
|----------|--------------------|--------|
| Strict raise | `STEP_4A_HANDBACK.md` `98eec8577b291588ccfa654fc9a0b4a6b9dde4428b631de57439a31d1d7b76cb` | `channel_native_noise_presence` |
| Width 4 | `interop_profile_bundle_routes_w4.json` at `5f63a9d6`, `6584be2bc49d55c995c54d1be9a7b7efb2b2dfe4cdd0004fd030b5596f9196aa` | R-strict `handback_refused`, no timings |
| Width 6 | `interop_profile_bundle_routes_w6.json` at `78ba7108`, `212758709a06c2805feef1181111ed980172877ad8b11e66f68bfafb9b11d1be` | R-strict `handback_refused`, no timings |
| Width 8 | `interop_profile_bundle_routes_w8.json` at `3feffb07`, `a9e375a1c4c7c39f908ed7136cc461403e315c232ec1a1d4d60fbf4da28fa84f` | R-strict `handback_refused`, no timings |
| Pack | `/workspace/phd/kb/briefs/2026-10-08-mf5a-option-c-pack.md` (`d038d340b96d00dc0388d078abcb420d9f758dad1bfb6beef2d1bdd7ec474031`) | option C1 recommended; decision recorded |

## 3. Governance outcome

| Field | Value |
|-------|-------|
| Outcome | Approved: option C1 |
| Meaning | The existing refusal rows at 4, 6, and 8 satisfy REQ-004 for R-strict. No new run |
| Formal approval | Zoltán; record in §6 |
| Approved-path steps | This Reviewer-gated amendment commit, separate from the milestone closeout. `CPP_PYTHON_INTEROP_PROFILE_CLOSEOUT.md` does not exist yet; when it is drafted, copy the ADR-F5A-011 Deferred paragraph into its Deferred section before the milestone review records its verdict. `milestone_counted` stays false |
| Backlog | C2 (strict-capable side workload): backlog, possible post-supervisor item, not started; not in M-F5a (Zoltán via PhD Manager and RM, 2026-10-08) |

Zoltán signed off the success-check deferral on 2026-10-08: no reduction, because the Python layer is shown not to be a bottleneck (UB (one-sided 95% upper bound on O, E-VQE bundles) ≤1.06% at 4/6/8: 1.0589% / 0.2149% / 0.0744%). Counted `overhead.upper_bound_95_O` is 0.010589466306384618, 0.0021490045376290055, and 0.0007435776878628046 on `ddde49ac:benchmarks/density_matrix/artifacts/interop_profile/interop_profile_bundle.json` (`212f70386bf2a44711d29956c41bd3f0eea9ee2e284ace9c5403bc3d94ef934e`), `6707892a:benchmarks/density_matrix/artifacts/interop_profile/interop_profile_bundle_w6.json` (`5257bad23e9fef4afad3f7b8b61f84c7cebd2ef8f85d95a94a02cb794d139d7d`), and `939d4908:benchmarks/density_matrix/artifacts/interop_profile/interop_profile_bundle_w8.json` (`1712dce97ae567460c9058c288c3a01879524dd3fb9cba71603a5f5785e1b21d`). CAP-004 is hold-the-line, and REQ-005 is satisfied as no reduction. `milestone_counted` stays false.

## 4. Fallback if change control refuses

Trigger: the sign-off is withdrawn. Response: REQ-004 for R-strict stays unmet on the frozen workload, and the choice among options C1, C2, and C3 returns to Zoltán through the Research Manager and the PhD Manager. Impact: M-F5a cannot formally close until that choice is made.

## 5. Risks and mitigations

| Risk | Mitigation |
|------|------------|
| The refusal row is read as a strict-route timing or as a defect | The row carries no timings. The bundles record the raise and the handback. The message is that strict mode refuses where it should |
| A later C2 figure is quoted as the anchor workload's strict cost | C2 (strict-capable side workload): backlog, possible post-supervisor item, not started; not in M-F5a (Zoltán via PhD Manager and RM, 2026-10-08). A figure from it is not this anchor's strict cost |
| The deferral is read as a reduction or a speed claim | REQ-005 is satisfied as no reduction. CAP-004 is hold-the-line. No speed claim |

## 6. Sign-off record

| Role | Name | Decision | Date | Change-request ID |
|------|------|----------|------|-------------------|
| Supervisor | Zoltán, via PhD Manager, relayed by the Tech Lead | "C1: accept the documented strict refusal, and I sign off the deferral: no reduction, the Python layer is shown not to be a bottleneck." | 2026-10-08 10:41 CEST (UTC+2) | option C1; ADR-F5A-011 |

## 7. Traceability

`CAP-004 → REQ-004 → task-4 DS-1, task-5 DS-1 → goal G4 → ADR-F5A-010, ADR-F5A-011 → A4`; `CAP-004 → REQ-005 → task-4 DS-2, task-5 DS-2 → goal G5 → ADR-F5A-005 → A4`. The milestone is not done. M-F1b stays closed.
