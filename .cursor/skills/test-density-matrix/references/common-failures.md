# Common density-matrix test failures

Read when a smoke import, pytest, build, or C++ test step fails. Prefer
`docs/density_matrix_project/SETUP.md` when it disagrees.

## Contents

- pybind11 not found
- TBB headers or libs not found
- Missing squander.density_matrix module
- Qiskit on Python 3.13
- Wrong C++ test binary path

## pybind11 not found

```bash
conda activate qgd
pip install pybind11
python setup.py build_ext
```

## TBB headers or libs not found

```bash
conda install -y tbb-devel -c conda-forge
export TBB_INC_DIR=~/.conda/envs/qgd/include
export TBB_LIB_DIR=~/.conda/envs/qgd/lib
rm -rf _skbuild
python setup.py build_ext
```

## Missing squander.density_matrix module

```bash
python setup.py build_ext
python -m pip install -e .
ls squander/density_matrix/_density_matrix_cpp*.so
```

## Qiskit on Python 3.13

```bash
conda install -y qiskit qiskit-aer -c conda-forge
```

## Wrong C++ test binary path

The C++ test binary is built under `_skbuild/*/cmake-build/...`, not `./test_standalone/`
in this workflow. Use the optional C++ test workflow in `SKILL.md` (including `rm -rf
_skbuild` before build, then the discovered `_skbuild/*/.../test_density_matrix_cpp` path).
