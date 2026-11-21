"""Exact-expression exchange with UniversalQCompiler's native Mathematica API.

The upstream Python bridge coerces angles to floats. This protocol avoids that
bridge. Expression import is a whitelist decoder, never eval/sympify on input.
"""

import json
from pathlib import Path
import shutil
import sympy as sp
from sympy.printing.mathematica import mathematica_code

if __package__:
    from .symbolic_gates import make_gate
else:
    from symbolic_gates import make_gate


UQC_SOURCE = dict(repository="https://github.com/Q-Compiler/UniversalQCompiler",
                  revision="c72cd84a537786756ae8490c07f3f599aa411323",
                  paper="https://arxiv.org/abs/1904.01072",
                  license="Apache-2.0")


def runtime_status():
    executable = next((shutil.which(name) for name in ("wolframscript", "WolframKernel", "MathKernel", "math")
                       if shutil.which(name)), None)
    return dict(status="available" if executable else "not_run_missing_wolfram_kernel",
                executable=executable, source=UQC_SOURCE,
                symbolic_parameter_support="must_be_verified_per_returned_expression",
                phase_policy="reconstruct_matrix_and_infer_scalar; no free GP in exact mode")


def export_job(name, parameters, *, job_id=None, timeout=60):
    target = make_gate(name, *parameters)
    width = target.rows.bit_length()-1
    variables = sorted(target.free_symbols, key=str)
    mapping = {symbol: sp.Symbol(f"p{i}", real=True) for i,symbol in enumerate(variables)}
    target_wolfram = target.xreplace(mapping)
    return dict(id=job_id or name, gate=name, parameters=[str(p) for p in parameters],
                num_qubits=width, symbolic_variables={str(v):str(k) for k,v in mapping.items()},
                matrix_input=mathematica_code(target_wolfram.tolist()),
                method="ZYZDec" if width == 1 else "DecUnitary2Qubits" if width == 2 else "QSD",
                timeout_seconds=timeout, ancillas=0, numerical_substitution=False,
                input_wire_convention="UQC wire 1 is most significant; map label j to width-j",
                gate_order="native list order, as CreateIsometryFromList multiplies it")


_FUNCTIONS = {
    "Plus": sp.Add, "Times": sp.Mul, "Power": sp.Pow,
    "Sin": sp.sin, "Cos": sp.cos, "Tan": sp.tan, "Cot": sp.cot, "Sec": sp.sec, "Csc": sp.csc,
    "Exp": sp.exp, "Log": sp.log, "Abs": sp.Abs, "Arg": sp.arg,
    "Conjugate": sp.conjugate, "Re": sp.re, "Im": sp.im,
    "ArcSin": sp.asin, "ArcCos": sp.acos, "ArcTan": sp.atan,
}


def decode_expression(node, variables, *, max_nodes=10000):
    """Decode only exact arithmetic/function ASTs and declared real symbols."""
    budget = [max_nodes]
    def decode(value):
        budget[0] -= 1
        if budget[0] < 0 or not isinstance(value, dict):
            raise ValueError("Invalid or oversized UQC expression")
        operation = value.get("op")
        if operation == "Integer":
            text = value["value"]
            if not isinstance(text,str) or len(text) > 10000 or not text.lstrip("-").isdigit():
                raise ValueError("Invalid UQC integer")
            return sp.Integer(text)
        if operation == "Rational":
            numerator, denominator = (decode(x) for x in value["args"])
            if not numerator.is_Integer or not denominator.is_Integer or denominator == 0:
                raise ValueError("Invalid UQC rational")
            return sp.Rational(numerator, denominator)
        if operation in {"Pi", "I", "E"}:
            return {"Pi":sp.pi,"I":sp.I,"E":sp.E}[operation]
        if operation == "Symbol":
            if value["name"] not in variables:
                raise ValueError("Undeclared symbol in UQC output")
            return variables[value["name"]]
        if operation == "Float":
            raise ValueError("UQC returned a floating angle; this is not a symbolic result")
        if operation not in _FUNCTIONS:
            raise ValueError(f"Unsupported UQC expression head {operation!r}")
        arguments = tuple(decode(x) for x in value["args"])
        if operation == "ArcTan" and len(arguments) == 2:
            # Mathematica ArcTan[x,y] is atan2(y,x).
            return sp.atan2(arguments[1], arguments[0])
        return _FUNCTIONS[operation](*arguments)
    return decode(node)


def import_native_result(job, result, *, variables):
    """Preserve symbolic angles, native list order, and exact wire convention.

    UQC rotations use +i sin(theta/2), opposite to our Rx/Ry/Rz convention.
    Tags outside the unitary CNOT/rotation model (including ancillas and
    measurements) are rejected rather than silently omitted.
    """
    if result.get("id") != job["id"] or result.get("status") != "completed":
        raise ValueError("UQC result is not a completed matching job")
    width = job["num_qubits"]
    operations = []
    def wire(label):
        if not isinstance(label, int) or isinstance(label,bool) or not 1 <= label <= width:
            raise ValueError("UQC output uses an ancillary or invalid wire")
        return width-label
    for entry in result["gates"]:
        tag = entry["tag"]
        if tag == 0:
            operations.append(("CX", (wire(entry["control"]), wire(entry["target"]))))
        elif tag in (1,2,3):
            angle = decode_expression(entry["angle"], variables)
            operations.append(({1:"Rx",2:"Ry",3:"Rz"}[tag], -angle, (wire(entry["wire"]),)))
        else:
            raise ValueError(f"Unsupported UQC gate tag {tag!r}; ancillas/measurements are forbidden")
    if any(len(set(op[-1])) != len(op[-1]) for op in operations):
        raise ValueError("UQC output has coinciding control/target wires")
    return tuple(operations)


def write_job_bundle(path, cases, *, timeout=60):
    path = Path(path)
    data = dict(schema_version=1, source=UQC_SOURCE, runtime=runtime_status(),
                jobs=[export_job(name, args, timeout=timeout) for name,args in cases])
    path.write_text(json.dumps(data,indent=2)+"\n")
    return data
