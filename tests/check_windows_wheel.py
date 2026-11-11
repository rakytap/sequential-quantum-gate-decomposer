"""Check the installed Windows package's native modules and public OSR ABI."""

import argparse
import ctypes
import sys
from pathlib import Path

import squander


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-venv", action="store_true",
                        help="Require the wheel to be installed inside a virtual environment")
    args = parser.parse_args()
    package = Path(squander.__file__).resolve().parent
    if args.require_venv:
        prefix = Path(sys.prefix).resolve()
        if sys.prefix == sys.base_prefix or prefix not in package.parents:
            raise RuntimeError(f"Expected a wheel in {prefix}, imported {package}")
    # delvewheel may move/rename the core DLL into squander.libs.
    candidates = sorted(package.rglob("*qgd*.dll"))
    candidates += sorted((package.parent / "squander.libs").glob("*qgd*.dll"))
    if not candidates:
        raise RuntimeError(f"No qgd DLL in the installed package at {package}")
    core = ctypes.CDLL(str(candidates[0]))
    for name in ("qgd_osr_logical_factor", "qgd_osr_phase_polynomial_factor",
                 "qgd_osr_parity_networks"):
        getattr(core, name)
    print(f"Loaded {squander.__file__}; checked OSR exports in {core._name}")


if __name__ == "__main__":
    main()
