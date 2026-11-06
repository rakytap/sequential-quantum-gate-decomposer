"""Lossless float64 parameter serialization for native rewrite audits."""
import math


def dump_native_audit_qasm(circuit):
    """Keep Qiskit's gate declarations, but emit exact decimal angle literals.

    Qiskit's human-readable pi expressions can round a value just below a
    rotation period to the period itself. Importing that expression and
    applying native periodic extraction then changes the recorded gate stream.
    Python's float repr round-trips the actual binary64 value instead.
    """
    from qiskit import qasm2

    if circuit.num_clbits or len(circuit.qregs) != 1:
        raise ValueError(
            "Native audit circuits require one quantum register and no classical bits.")
    lines = qasm2.dumps(circuit).splitlines()
    registers = [index for index, line in enumerate(lines)
                 if line.startswith("qreg ")]
    if len(registers) != 1:
        raise ValueError("Native audit QASM must declare one quantum register.")
    body = []
    register = circuit.qregs[0].name
    for instruction in circuit.data:
        operation = instruction.operation
        if instruction.clbits or operation.num_clbits:
            raise ValueError("Native audit operations cannot address classical bits.")
        angles = []
        for angle in operation.params:
            value = float(angle)
            if not math.isfinite(value):
                raise ValueError("Native audit angles must be finite.")
            angles.append(repr(value))
        arguments = "(" + ",".join(angles) + ")" if angles else ""
        wires = ",".join(
            register + "[" + str(circuit.find_bit(qubit).index) + "]"
            for qubit in instruction.qubits)
        body.append(operation.name + arguments + " " + wires + ";")
    return "\n".join(lines[:registers[0] + 1] + body)

