# Repository gotchas (density-matrix track)

Read when implementing or closing a slice in this repository.

## Contents

- Conda, rebuild, and lanes
- Exact baseline and external reference
- Oracle independence (G-10)
- Archive convention

## Conda, rebuild, and lanes

Tests, benchmarks, and examples run in the **`qgd` conda environment**
(`conda run -n qgd --no-capture-output pytest …`). A counted run, a regeneration,
and a (g) rerun use that same `conda run -n qgd --no-capture-output` form from the
mini-spec, never the environment's `python` binary on its own. The categorical
check and the one-attempt rule are in `practices-testing.md` § One attempt and an
interrupted launch. C++ or CMake changes need a rebuild
(`clean-rebuild` skill); `test-density-matrix` runs the suites; `TECH_STACK.md` lists the
lanes. Evidence rows name their lane: fast pytest (`tests/density_matrix`,
`tests/partitioning`, `tests/VQE`, `-m "not slow"`), `slow`, a benchmark evidence pipeline
(`benchmarks/density_matrix/*/validation_pipeline.py`), the optional C++ tests
(`QGD_CTEST=1`), or the Qiskit Aer external reference.

## Exact baseline and external reference

The sequential `NoisyCircuit` executor is the exact baseline every partitioned, fused,
or new backend path is validated against; Aer is the external reference. A `QA-*` about
exactness becomes a fitness test against that baseline.

## Oracle independence (G-10)

Oracle independence proves separate execution, not separate kernels. The cell and the
sequential oracle are separate calls, and each allocates its own `DensityMatrix`. Neither
reads the other's output back, so bitwise agreement is legitimate and the closeout records
its reason. Both still share `_build_runtime_circuit` lowering and the `NoisyCircuit` gate
and noise kernels. A kernel-level bug therefore appears on both sides, and this oracle
cannot detect it. Record that limitation in the closeout. Do not redefine or replace the
oracle to close it.

## Archive convention

The delivered phase trees under `docs/density_matrix_project/archive/` use an earlier
convention. Read them for history; never extend them or copy their naming into
`docs/specs/` (mapping: `artifact-map.md`). Revisions are forward-only.
