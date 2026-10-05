# Regeneration acceptance (revision-only mismatch)

Read when a slice's ADRs or mini-spec require accepting a regeneration run at a commit
after the one that produced the committed evidence bundle. Milestone-specific thresholds
and field lists live in that milestone's ADRs; this file describes the usual shape.

## Contents

- Expected exit status
- Acceptance checklist
- Restore after acceptance

## Expected exit status

The exit status depends on the bundle's comparator. When the milestone ADR allowlists
`provenance.implementation_revision` (for M-F1a, ADR-F1A-009 Amendment 1), a difference
in only that field passes: exit 0, `status` pass, `regeneration.pass` true, and
`first_mismatch` null. Otherwise, expect the non-zero shape that milestone ADR documents.
Never widen an allowlist to make a run pass. Run once with no env overrides.

## Acceptance checklist

Accept only if all of the following hold (adjust metric names to the milestone ADR):

- Metrics are within the frozen comparators.
- Route and label fields are identical.
- Provenance and clean-start flags are as the ADR requires.
- `implementation_revision` equals HEAD.
- The other suites named in the ADR remain `pass`.
- The unfiltered recursive diff shows only the allowlisted field and the derived fields
  the ADR names. For example, `regeneration.prior_present` changes from false to true
  when the committed bundle was written with no prior. Any other difference fails.
- Before the run, the loaded extension sha256 equals the committed identity.

## Restore after acceptance

After acceptance, restore every rewritten file to the HEAD bytes, as in the snapshot and
restore section of `SKILL.md`. Regeneration outputs are never committed.
