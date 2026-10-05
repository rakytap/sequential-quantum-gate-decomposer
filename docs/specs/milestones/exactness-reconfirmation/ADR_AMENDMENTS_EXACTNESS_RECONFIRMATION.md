# M-F1a ADR continuation — exactness-reconfirmation
> **Status:** accepted amendments and ADRs (2026-10-05) · **Milestone:** M-F1a `exactness-reconfirmation` ·
> **Continues:** `ADRS_EXACTNESS_RECONFIRMATION.md` (ADR-F1A-001…009) · **Scope:** ADR-F1A-008 Amend1, 010, 009 Amend1, 011

## ADR-F1A-008 — Amendment 1: stage-aware closeout check

**Status:** accepted (Research Manager decision record 2026-10-05 §(c): "acceptable as a
separate, Reviewer-gated tooling commit implementing ADR-F1A-008 exactly"). Amends the
rejected-alternatives list and the status clause "or the lint script"; decisions 1 and 3
stand; decision 2 stays replaced by ADR-F1A-009. Upstream alignment: REQ-004, REQ-007 ·
QA-008 · G-08.

**Change.** "Make the script stage-aware" is no longer rejected. One Reviewer-gated local
tooling commit, not a slice and without a CLOSEOUT, may enforce decision 1 mechanically. It
updates every companion text that says each unwaived warning becomes an error, and records
its throwaway-tree exercise in the skill `CHANGELOG.md`. Bounds:
1. Only `SLICE_MISSING_CLOSEOUT` is exempt from `--strict` promotion, and only while
   `task-<n>/ENGINEERING_TASKS.md` carries, within its first 12 lines, exactly one
   `**SDD stage:**` field whose value is `step-4a`. The finding is still printed as a
   warning. Normal mode is unchanged.
2. Any other value, a missing field, or a duplicate field fails closed: the finding is a
   strict error. Stage is never inferred from file presence, git state, or free-text status.
3. Every other finding keeps its normal and strict behavior. No waiver, `.sdd-lint.json`
   entry, or other suppression path is added.
4. The planning role sets `step-4b-authorized` in the writer pass that records the
   ADR-F1A-010 code-ready closure, so the value lands in C1.
5. The exercise shows: a `step-4a` slice without CLOSEOUT gives strict exit 0 and one
   warning; `step-4b-authorized`, a missing field, or a duplicate field gives strict exit
   nonzero; any other injected finding is a strict error at every stage; normal-mode output
   is unchanged.

---

## ADR-F1A-010 — Step 4b authorization for the remaining M-F1a slices

**Status:** accepted (Research Manager decision record 2026-10-05 §(b)). Replaces G-01's
per-cell authorization rule; the q4 authorization stays as history.

**Context.** G-01 required a separate Research Manager authorization for every cell beyond
q4. Research Manager opened Step 4a for every advertised route at 4/6/8/10 and for the drift
and comparator slices (§(c)), and delegated each slice's Step 4b start.

**Decision.** A slice in that scope enters Step 4b when Architect closes its Step 4a review
code-ready under ADR-F1A-008 and the verdict in its `ENGINEERING_TASKS.md` states that each
item below is unchanged. Anything not so stated returns to Research Manager first, as do any
oracle disagreement, QA-001 failure, route not advertised or not realizable at its anchor,
and proposed scope or oracle change.
1. The oracle, `execute_sequential_density_reference`.
2. QA-001 per ADR-F1A-002 and the regeneration comparators, apart from the single allowlist
   entry in ADR-F1A-009 Amendment 1.
3. The counted set: ADR-F1A-001 counting and realization rules and, once frozen, the reviewed
   manifest. No counted run precedes the freeze.
4. The scope and the G-07 exit rule. Adding a required M-F1a sibling suite is not a G-07
   change. Changing an exclusion, dropping, skipping, or demoting an included suite, or
   changing the exit rule is.

**Rationale.** Research Manager keeps every decision that can move the claim; Architect
keeps the code-ready judgement it already owns.

**Consequences.** Checklist G-01 cites this ADR. Tech Lead performs each handoff. Wording
stays "baseline route verified" or "`<route>` route verified at q`<n>`"; "exactness
reconfirmed" waits for the milestone closeout and the Research Manager report. No push and
no PR.

**Rejected alternatives.** A Research Manager round per slice: withdrawn by the decision
record. Delegating any of the four items: each one defines the claim.

**Upstream alignment:** REQ-001, REQ-002, REQ-004 · QA-001, QA-008 · G-01, G-03.

---

## ADR-F1A-009 — Amendment 1: per-slice two-commit close

**Status:** accepted (Research Manager decision record 2026-10-05 §(b) "applies per slice",
§(c) comparator). Replaces only the "q4 baseline cell only" scope; steps (a)–(g), the
pre-(d) gate, and the Unchanged clause stand. Upstream alignment: REQ-004, REQ-007 · QA-008
· ADR-F1A-004, ADR-F1A-008, ADR-F1A-010.

**Change.** Steps (a)–(g) apply to every M-F1a slice whose evidence records `clean_start`,
each with its own C1, counted run, real CLOSEOUT, Reviewer evidence review, C2, and
clean-C2 regeneration. Layer 1 amendment, process-docs, and tooling commits are single
Reviewer-gated local commits, not slices. For a slice that adds no counted cell, the
pre-(d) gate is a written statement, with the diff stat, that no runtime or oracle code path
changed. A change to code, tests, or planning docs between C1 and the counted run restarts
the slice at (a). A CLOSEOUT cannot name the commit that adds it, so the next planning
writer pass records each C2 in the checklist.

**Cross-revision regeneration.** A rerun at a revision other than a bundle's recorded one
compares every recorded field as ADR-F1A-004 requires, except
`provenance.implementation_revision`, the only allowlisted field. Derived fields (`status`,
`summary`, `regeneration`) and non-counted context are recomputed, not compared. This is the
one governed deviation from ADR-F1A-004 and task-1 §3.3; the decision record is its
sign-off, so no `CHANGE_CONTROL.md` is opened. REQ-004 is unchanged: at the recorded
revision every field compares exactly.

**Consequences.**
1. Each counted run also regenerates every earlier M-F1a sibling. Each must pass, subject
   to consequence 2; its rewritten file is restored to the committed bytes and never committed.
2. If an earlier sibling fails, the slice stops before (d). An environment or extension
   identity mismatch goes to Tech Lead; any other mismatch goes to Research Manager. A
   re-baseline is its own Step 4a item, never folded into another slice's C2. Until Slice
   B's allowlist lands, in Slice A only, a q4 regeneration failure whose full field diff
   against the committed bundle equals exactly the TST-4 difference set (first_mismatch
   `cases[0].provenance.implementation_revision`) is the expected outcome, not a stop; the
   regenerated q4 file is restored to committed bytes and never committed.
3. The 2026-10-05 drift-lineage HISTORICAL-RECORD BLOCKER is dispositioned by Research
   Manager's approval of Slice A and ADR-F1A-011; the first permitted pipeline run is Slice
   A's step (c) from its C1, which contains the verify-only change. Before that C1 only
   stubbed tests run. Before Slice A's C1, rewritten historical bundles are restored after
   every run and never committed, and the slice CLOSEOUT records it. From Slice A's C1 on,
   the REQ-007 eight-path diff runs before any restore and a non-empty result is a defect;
   the restore then stays as the safety check until Research Manager revises it, and the
   slice CLOSEOUT records each restore.

---

## ADR-F1A-011 — Historical suites are verify-only under the M-F1a command

**Status:** accepted (Research Manager, 2026-10-05, via Tech Lead, "exactly as designed").
Provenance: Tech Lead restore addendum (2026-10-05) is the restore consequence below. The
eight-bundle restore stands until Research Manager revises it. From Slice A C1, the REQ-007
eight-path check runs before that restore, and a non-empty result is a defect. The
`test-density-matrix` skill rule and this ADR are consistent.
Final text, 2026-10-05, including that addendum.

**Context.** The single M-F1a command (`validation_pipeline.py`, `run_pipeline` `:104-120`,
`_write_slice_bundle` `:52-53`) builds every registered suite and writes every bundle
unconditionally. Six of those bundles are Phase-3 evidence (Phase 3 Task 6 Stories 2–7:
`correctness_package`, `output_integrity`, `runtime_classification`, `sequential_correctness`,
`external_correctness`, `unsupported_boundary`). The frozen Phase 3 paper, short paper, and
abstract quote their 25/4/17 counts, and Task 8 names two of them as Paper-2 entry points. Two
more registered outputs, `correctness_matrix` and `summary_consistency`, are rewritten
byte-identically. Every run rewrites the six with content that is not in the historical record:
`partitions[*].members` (from `ac20a8e1`), reworded case 7/11 strings, floating-point drift of at
most 6.7e-16, and runtime and RSS fields. Committing after a run would rewrite Phase-3 evidence,
which ADR-F1A-005 forbids. The committed bytes at `a2928bf1` are still identical to `fb865857`.
REQ-007's static diff does not cover these paths today. Lineage report:
`2026-10-05-mf1a-drift-lineage-report.md`.

**Decision.**
1. **Write allowlist.** The M-F1a command writes a bundle only for suites explicitly registered as
   M-F1a siblings. Registration is an explicit per-entry field in the suite registry; a suite name
   pattern, path prefix, or default does not count. A suite without that field is verify-only, so
   new or unknown suites fail closed.
2. **Verify-only means built, not written.** Every verify-only suite is still built in memory, its
   status is returned exactly as today, and it feeds G-07 and `summary_consistency` unchanged.
   `g07_exit_passes` is not modified. The command reports each such suite as "verified, not
   written".
3. **No historical comparison.** Verify-only suites are not compared against their committed bytes,
   and drift does not affect the exit code. Drift is a known pre-existing condition, not an M-F1a
   finding.
4. **Opt-in out-of-repo output.** `--historical-output-dir <path>` may write verify-only bundles for
   inspection, but only to a path that resolves (after symlinks) outside the repository working
   tree. A path inside the repo is refused with a nonzero exit before any suite is built. Outputs
   written this way are never evidence and are never committed.
5. **REQ-007 static diff.** The eight historical artifact paths under
   `benchmarks/density_matrix/artifacts/correctness_evidence/<suite>/` are added to the REQ-007
   historical-boundary static diff, so any tracked or untracked change to them fails that check.
6. **Scope of change.** Only `validation_pipeline.py` and
   `tests/partitioning/evidence/test_correctness_evidence.py` change. No generator, no
   `squander/`, no `evidence_io.py`, and none of the historical JSON files.
7. **Required tests.**
   (a) After a run, all eight historical bundles are byte-identical to their pre-run state.
   (b) Every registered M-F1a sibling is still written.
   (c) All registered statuses are still returned, with the same G-07 result as before.
   (d) An in-repo `--historical-output-dir`, including through a symlink, is refused before
       building.
   (e) A suite registered without the sibling field is not written.

**Known hazard (Research Manager requirement, recorded in the slice A CLOSEOUT).** These entry
points still rewrite historical artifacts in place and are outside the M-F1a command: the
standalone per-suite CLIs, `phase31_validation_pipeline.py`, and the `performance_evidence`
validation pipeline. Until a later decision, nobody runs them on rocky. This ADR does not change
them.

**Rationale.** It removes the only M-F1a write path into Phase-3 evidence without changing the
G-07 contract, the oracle, tolerances, counted set, or scope. It needs no new comparator and no
gitignore exclusion, so ADR-F1A-009's clean-start rule holds.

**Consequences.**
- Once this ADR's change lands (Slice A C1), the M-F1a pipeline no longer writes the eight
  historical bundles. The standing restore of those paths remains as a safety check until
  Research Manager revises it; from that C1 on, the REQ-007 eight-path diff runs before any
  restore and a non-empty result is a defect. That restore rule is ADR-F1A-009 Amendment 1,
  consequence 3.
- `TECH_STACK.md:43` is updated at milestone close, under ADR-F1A-007.
- Regenerating and committing the six bundles stays forbidden. It needs a separate ADR-F1A-005
  decision.
- Slice A runs under ADR-F1A-008 (code-ready) and then the ADR-F1A-009 two-commit close. The
  comparator slice and the inventory counted runs start only after slice A closes.

**Rejected alternatives.**
- Write to temp and compare: needs a new historical comparator, would flag all six today, and
  gating on it changes G-07 and turns M-F1a red for a pre-M-F1a problem.
- A separate in-repo output directory: either dirties `clean_start` or needs a gitignore
  exclusion, which ADR-F1A-009 rejects.
- Regenerate and commit the six bundles: rewrites Phase-3 evidence (ADR-F1A-005).
- A write denylist of the known historical suites: fails open for any suite added later.

**Upstream alignment:** REQ-004, REQ-007 · CAP-007 · QA-008 · ADR-F1A-004, ADR-F1A-005,
ADR-F1A-009 · G-07.
