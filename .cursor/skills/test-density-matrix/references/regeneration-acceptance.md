# Regeneration acceptance (revision-only mismatch)

Read when a slice's ADRs or mini-spec require accepting a regeneration run at a commit
after the one that produced the committed evidence bundle. Milestone-specific thresholds
and field lists live in that milestone's ADRs; this file describes the usual shape.

## Contents

- Expected exit status
- Acceptance checklist
- Restore after acceptance

## Expected exit status

Regenerating at a commit after the one that produced the committed bundle is often
**expected** to exit non-zero, with `status=fail` and
`summary.first_failure=regeneration`, because `provenance.implementation_revision`
changes. Run once with no env overrides.

## Acceptance checklist

Accept only if all of the following hold (adjust metric names to the milestone ADR):

- QA-001-style metrics (Frobenius, max-abs, trace residual) match the committed bundle
  within the frozen tolerances, and any minimum-eigenvalue witness meets its floor.
- Route and label fields are identical.
- Provenance and clean-start flags match the ADR (for example `qa001_pass`,
  `provenance_pass`, `clean_start` true, `dirty_paths` empty, and
  `implementation_revision` equals HEAD).
- Other suites named in the ADR remain `pass`.
- An unfiltered recursive field diff (regenerated vs committed) shows **only** the
  differences the milestone ADR lists as revision-only (typically implementation revision,
  regeneration status fields, and related provenance). Any other difference is a fail.

## Restore after acceptance

After acceptance, restore every rewritten file to the HEAD bytes, as in the snapshot and
restore section of `SKILL.md`. Regeneration outputs are never committed.
