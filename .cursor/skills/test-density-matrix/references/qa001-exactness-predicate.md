# QA-001 exactness predicate (lambda_min witness)

Read when a counted density-matrix predicate cites QA-001 (ADR-F1A-002,
`docs/specs/milestones/exactness-reconfirmation/ADRS_EXACTNESS_RECONFIRMATION.md`).

## Predicate

Compute `lambda_min(rho)` only with the existing `DensityMatrix.eigenvalues()` contract.
LAPACK `zheev` reads the upper triangle of rho as stored, and the minimum returned
eigenvalue is `lambda_min`. Check finiteness of every entry and residual first. A solver
failure or a non-finite eigenvalue fails the row.

Do not symmetrize: `(rho + rho^dagger)/2` is forbidden. Do not add a Hermiticity gate or
cite QA-010. This is an eigensolver convention, not a Hermiticity check.

The thresholds are frozen: `||delta rho||_F <= 1e-10`, max-abs `<= 1e-10`, `|Tr rho - 1| <= 1e-10`,
`lambda_min >= -1e-12`, all against the sequential oracle. Record each value and its
pass/fail. Older `rho_is_valid`, energy, and Aer fields may appear as context but never
decide the counted status.
