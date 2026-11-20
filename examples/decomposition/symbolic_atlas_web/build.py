"""Build a standalone atlas from the archive used by the PDF."""

import ast
import base64
import gzip
import hashlib
import json
from pathlib import Path
import sys


SOURCE_SHA256 = "f0e67c123e43d26c999cc39ef5865d153ceb7f9ad5742b5d9be35f76b8492495"
ARCHIVE = "benchmarks/symbolic/synthesis_reference_20261107_referee.json.gz"
OPERATORS = {
    ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.Pow: "**",
    ast.USub: "-", ast.Invert: "~", ast.BitAnd: "&",
    ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=",
}
FUNCTIONS = {
    "sin", "cos", "sqrt", "Abs", "atan2", "Piecewise", "Eq", "atan", "floor",
    "exp", "sinc", "acos", "Mod", "log", "Ne", "conjugate", "re", "im",
}


def expression_node(node):
    if isinstance(node, ast.Constant) and type(node.value) in (int, float, bool):
        return node.value
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.UnaryOp):
        return [OPERATORS[type(node.op)], expression_node(node.operand)]
    if isinstance(node, ast.BinOp):
        return [OPERATORS[type(node.op)], expression_node(node.left), expression_node(node.right)]
    if isinstance(node, ast.Compare) and len(node.ops) == 1:
        return [OPERATORS[type(node.ops[0])], expression_node(node.left),
                expression_node(node.comparators[0])]
    if isinstance(node, ast.Call) and node.func.id in FUNCTIONS and not node.keywords:
        return [node.func.id, *(expression_node(arg) for arg in node.args)]
    if isinstance(node, ast.Tuple):
        return [expression_node(item) for item in node.elts]
    raise ValueError(ast.dump(node))


def free_symbols(value):
    if isinstance(value, (list, tuple)):
        return set().union(*(free_symbols(item) for item in value))
    return {str(symbol) for symbol in getattr(value, "free_symbols", ())}


def build(repo):
    compressed = (repo / ARCHIVE).read_bytes()
    if hashlib.sha256(compressed).hexdigest() != SOURCE_SHA256:
        raise ValueError("Archive does not match the PDF atlas.")
    archive = json.loads(gzip.decompress(compressed))
    sys.path.insert(0, str(repo / "squander" / "decomposition"))
    from symbolic_archive import symbolic_cases
    from symbolic_bases import BASIS_SETS
    from symbolic_gates import gate_definition, make_gate

    targets = {}
    for name, parameters, scope in symbolic_cases():
        width = gate_definition(name).num_qubits
        if width is None:
            width = make_gate(name, *parameters).rows.bit_length() - 1
        targets[name] = {
            "width": width,
            "scope": scope,
            "inputs": sorted(free_symbols(parameters)),
            "convention": gate_definition(name).convention,
        }

    expressions = []
    expression_indices = {}

    def index(source):
        if source not in expression_indices:
            expression_indices[source] = len(expressions)
            expressions.append([source, expression_node(ast.parse(source, mode="eval").body)])
        return expression_indices[source]

    def mode(entry, status):
        if not entry or status != "verified_symbolic":
            entry = entry or {}
            return {"status": status, "reason": entry.get("explanation") or entry.get("reason")}
        factor = (entry["virtual_gp"] or {}).get("factor")
        result = {
            "status": status,
            "operations": [[op["gate"], op["wires"], [index(p) for p in op["parameters"]]]
                           for op in entry["operations"]],
            "entanglers": entry["cost"]["entanglers"],
            "locals": entry["cost"]["single_qubit_gates"],
        }
        if factor is not None:
            result["factor"] = index(factor)
        program = entry.get("expression_program")
        if program:
            if program["format"] != "exact_scalar_dag_v1":
                raise ValueError(program["format"])
            result["bindings"] = [[b["symbol"], index(b["expression"])] for b in program["bindings"]]
        return result

    rows = {name: {} for name in targets}
    for row in archive["rows"]:
        statuses = row["phase_statuses"]
        rows[row["gate"]][row["basis"]] = [
            mode(row["phase_exact"], statuses["phase_exact"]),
            None if row["projective_omitted_as_no_better"]
            else mode(row["projective"], statuses["projective"]),
        ]
    assert set(targets) == set(archive["case_names"])
    assert all(set(results) == set(BASIS_SETS) for results in rows.values())
    bases = [{
        "name": name, "entangler": basis.entangler, "locals": basis.single_qubit_gates,
        "classification": basis.universality, "notes": basis.notes,
    } for name, basis in sorted(BASIS_SETS.items())]
    return {
        "targets": targets, "bases": bases, "rows": rows, "expressions": expressions,
        "source": archive["git_revision"], "sha256": SOURCE_SHA256,
    }


if __name__ == "__main__":
    src = Path(__file__).resolve().parent
    repo = src.parents[2]
    data = build(repo)
    packed = gzip.compress(json.dumps(data, separators=(",", ":")).encode(), mtime=0)
    page = (src / "index.html").read_text()
    for name, replacement in {
        "CSS": (src / "atlas.css").read_text(),
        "QASM": (src / "qasm.js").read_text(),
        "APP": (src / "atlas.js").read_text(),
        "DATA": base64.b64encode(packed).decode(),
    }.items():
        page = page.replace(f"<!-- {name} -->", replacement)
    (repo / "docs" / "atlas" / "index.html").write_text(page, encoding="utf-8")
