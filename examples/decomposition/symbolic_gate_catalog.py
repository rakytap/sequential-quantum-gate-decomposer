#!/usr/bin/env python3
"""Inspect the exact symbolic catalog without building the native backend."""

import argparse
import dataclasses
import json
from pathlib import Path
import sys

# These modules deliberately support standalone import for symbolic-only work.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "squander/decomposition"))
from symbolic_gates import GATE_ALIASES, GATE_DEFINITIONS, gate_definition, make_gate
from symbolic_bases import BASIS_SETS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", action="append", default=[], help="Inspect a gate's signature; repeatable")
    parser.add_argument("--list-gates", action="store_true", help="Include all catalog names")
    parser.add_argument("--cliffords", action="store_true", help="Validate all 192 Cliffords and include exact group tables")
    parser.add_argument("--output", type=Path, help="Write JSON to this file instead of stdout")
    args = parser.parse_args()
    report = {
        "gate_definitions": len(GATE_DEFINITIONS),
        "provider_and_common_aliases": len(GATE_ALIASES),
        "wire_convention": "local wire 0 is least significant; sequences are in circuit order",
        "basis_presets": [dataclasses.asdict(basis) for basis in BASIS_SETS.values()],
    }
    if args.list_gates:
        report["gates"] = sorted(GATE_DEFINITIONS)
        report["aliases"] = GATE_ALIASES
    if args.gate:
        report["selected"] = []
        for name in args.gate:
            definition = gate_definition(name)
            item = dataclasses.asdict(definition)
            if not definition.parameters:
                item["matrix"] = str(make_gate(name))
            report["selected"].append(item)
    if args.cliffords:
        from symbolic_clifford import analyze_single_qubit_cliffords
        analysis = analyze_single_qubit_cliffords()
        report["clifford_analysis"] = dataclasses.asdict(analysis)
    text = json.dumps(report, indent=2, default=str) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
