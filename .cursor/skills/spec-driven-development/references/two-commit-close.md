# Two-commit slice close (clean-start evidence)

Read when a slice's counted evidence records `clean_start` and Step 4b must close
without dirtying the regeneration run. Precedent: ADR-F1A-009, M-F1a q4 tracer.

## Contents

- Commit and review order
- Planning-doc header sync before Reviewer (a)
- Porcelain check before a local commit
- Independence gate (G-10)

## Commit and review order

For that slice, this order replaces "write the closeout, Reviewer, then commit":

1. **(a) Reviewer implementation review** of the uncommitted implementation diff.
2. **(b) Local implementation commit C1**: planning docs, implementation, and tests only.
   C1 holds no generated artifact and no `CLOSEOUT.md`. An optional planning-docs C0 may
   precede C1.
3. **(c) Counted clean-start run**: from an empty `git status --porcelain` at C1, run the
   slice's single evidence command once and keep its counted evidence.
4. **Pre-(d) independence gate**: Tester confirms in writing that the oracle and the cell
   are independent. The note names the code paths and objects on each side, shows that
   the oracle is not the cell's own output read back, and explains any bitwise agreement
   (`references/repo-gotchas.md` § Oracle independence (G-10)). If independence is not
   established, stop: write no `CLOSEOUT.md` and make no C2.
5. **(d) Real slice close**: write `task-<n>/CLOSEOUT.md` citing C1. Normal and
   `--strict` `specs_check.sh` and traceability must then be fully clean.
6. **(e) Reviewer evidence review** of the counted evidence, the real closeout, and the
   clean verification results.
7. **(f) Local evidence commit C2**: the counted bundle and `CLOSEOUT.md`.
8. **(g) Clean-C2 regeneration**: Tester reruns regeneration from a clean C2, accepts it
   under the slice's ADRs and mini-spec regeneration rule (including any revision-only
   mismatch policy the milestone documents), and restores every generated output.
   Regeneration outputs are never committed.

No push or pull request is part of this sequence. Clean-start rules, including how to park
a dirty non-counted run: `practices-testing.md` § Clean-start evidence and slice-close order.

## Planning-doc header sync

Before the Reviewer implementation review (a), sweep `docs/specs/milestones/<slug>/`: the
checklist, every `task-<n>/` mini-spec, stories, and engineering tasks. Bring each context
header and each authorization or status line to the current position. No header may still
say "C1 awaits Reviewer" or "Step 4b blocked" once authorization has moved. A quick check is
`rg -n "awaits Reviewer|Step 4b blocked|not committed|uncommitted|until the Reviewer gate|in this .*draft" docs/specs/milestones/<slug>/`. The
synced docs belong in C1 (or C0), so they do not dirty the counted run.

## Porcelain check before a local commit

Before a C0, C1, C2, or docs commit, from the repo root:

```bash
unset PYTHONPATH
git status --porcelain --untracked-files=all
git diff --cached --name-only
git diff HEAD -- benchmarks/density_matrix/artifacts
git diff HEAD -- squander
```

Porcelain equals the intended add list and nothing else. The index stays empty
until `git add -- <path> …`. The artifacts directory and `squander/` match HEAD
unless this commit is the evidence commit that adds the counted bundle. This is
the procedure for `AGENTS.md` non-negotiable 7.
