# Testing, evidence, and readiness practices for SDD

Read when writing acceptance criteria, an evidence matrix, a `QA-*` fitness function, or
when gating a slice in or out. This is where a spec becomes provable.

## Contents

- BDD and specification by example
- Testing strategy
- Evidence matrix
- Quality attributes as fitness functions
- Definition of Ready
- Definition of Done
- Verification commands

## BDD and specification by example

- Carry the `Given / When / Then` acceptance from each `REQ-*` into executable
  specifications — the living documentation. Acceptance criteria are written before
  implementation and become automated acceptance tests (ATDD).
- Express the **negative space** as scenarios too: unsupported behavior, error semantics,
  and refusals are part of the contract.

## Testing strategy

- Favor a **test pyramid / testing trophy**: many fast unit tests on the domain, fewer
  integration tests at ports and adapters, and a thin layer of end-to-end acceptance
  tests per slice. Avoid the ice-cream cone (mostly end-to-end).
- Practice **TDD** (red → green → refactor) for domain logic, **contract tests** at
  service and integration boundaries, and **characterization tests** before changing
  untested legacy code.
- Each delivery story's acceptance signals must map to a reproducible test command or
  check. "Done" means deployable with green checks.

## Evidence matrix

Maintain an evidence matrix for every non-trivial slice. Each `REQ-*`, each relevant
`QA-*`, and each major boundary decision maps to:

| Trace id | Evidence type | Command / CI gate | Expected result | Owner artifact |
|----------|---------------|-------------------|-----------------|----------------|

Rules that make the matrix mechanically checkable:

- The command cell names something runnable — a `make` target, a `pytest` path, a
  script, or a named CI gate — or an explicitly declared non-executable evidence type
  (`doc review`, `sign-off`).
- A row may defer to another row or artifact, as long as it points somewhere.
- A `make` target named as evidence must exist in the `Makefile`; a `tests/…` path must
  exist on disk. When a later milestone renames either, the specs that cite it are stale
  evidence.
- Distinguish the blocking hermetic lane from opt-in lanes, and say which lane each row
  runs in.

`check_traceability.py` verifies all of the above.

## Quality attributes as fitness functions

Turn every relevant `QA-*` or NFR into an automated fitness function and cite the `QA-*`
id: a performance budget, an architecture test enforcing the dependency rule, a
security/SAST gate, an availability probe, a containment test proving a dependency stays
behind its adapter. These guard quality attributes continuously instead of through one-off
review. A `QA-*` asserted but never enforced by a check is an evidence gap.

## Definition of Ready

A slice may start when:

- acceptance criteria are clear and testable;
- dependencies are known;
- contracts are agreed;
- required boundary-map and ADR decisions are in place;
- current-state docs have been read and any update need is known;
- the Layer 4 plan is code-ready (objective done criteria, named tests, evidence rows,
  cited `QA-*` fitness functions).

## Definition of Done

A slice is done when:

- acceptance tests are green;
- fitness functions pass;
- specs and docs are updated, including `ARCHITECTURE_OVERVIEW.md` and `TECH_STACK.md`
  when affected;
- the slice is deployable;
- release and rollback expectations are known;
- observability and security/privacy checks are satisfied where relevant;
- `task-<n>/CLOSEOUT.md` records the verdict and the reproduce commands.

The **slice tracer** additionally proves the deployment path end-to-end.

## Verification commands

Run these before claiming a slice or milestone is complete. They are cheap, hermetic, and
dependency-free (stdlib only):

```bash
bash .cursor/skills/spec-driven-development/scripts/specs_check.sh            # both linters, errors only
bash .cursor/skills/spec-driven-development/scripts/specs_check.sh --strict   # every unwaived warning becomes an error
```

Add `--no-waivers` to see the full debt — size budgets, stale evidence commands, legacy
naming — that the recorded waivers suppress.

To scope a check to the milestone in hand:

```bash
bash .cursor/skills/spec-driven-development/scripts/specs_check.sh docs/specs/milestones/<slug>
```

The product suites themselves run in the `qgd` conda environment and are described in
`docs/specs/TECH_STACK.md`; the `test-density-matrix` skill runs them end to end. Name
the lane in every evidence row: fast pytest (`-m "not slow"`), `slow`, a benchmark
evidence pipeline (`benchmarks/density_matrix/<tree>/validation_pipeline.py`), the
optional C++ tests (`QGD_CTEST=1`), or the Qiskit Aer external reference.

Both exit non-zero on errors and print `--format json` for programmatic use. A finding you
intend to keep must be waived in `docs/specs/.sdd-lint.json` with a reason — never
silenced by deleting the check.
