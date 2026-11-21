"""Exact fixed-target motifs; the compiler reproves every proposed circuit."""

from functools import lru_cache
import json
from pathlib import Path

import sympy as sp


@lru_cache(maxsize=1)
def _templates():
    data = json.loads(Path(__file__).with_suffix(".json").read_text())
    return {(row["gate"], row["basis"]):row["candidates"] for row in data["entries"]}


def fixed_template_candidates(name, parameters, basis):
    if parameters:
        return []
    result = []
    for candidate in _templates().get((name, basis.name), ()):
        sequence = tuple((op["gate"], *(sp.sympify(p) for p in op["parameters"]), tuple(op["wires"]))
                         for op in candidate["operations"])
        factor = sp.sympify(candidate["factor"])
        if factor.free_symbols or factor.atoms(sp.Float) or any(
                p.free_symbols or p.atoms(sp.Float) for op in sequence for p in op[1:-1]):
            raise ValueError("Fixed templates must contain exact constant angles")
        result.append((sequence, factor))
    return result
