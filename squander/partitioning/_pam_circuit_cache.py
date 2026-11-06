"""Bounded, routing-scoped conversion of native OSR columns for PAM.

The exact exported QASM text is the cache key. Every caller receives an
independent circuit copy before any endian permutation or PAM mutation.
No synthesis result, objective, mapping decision, or tolerance is cached.
"""
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
import os


_active_cache = ContextVar("squander_pam_circuit_parser_cache", default=None)


def _decode(qasm):
    from bqskit.ir.lang.qasm2 import OPENQASM2Language

    return OPENQASM2Language().decode(qasm)


def native_pam_circuit(circuit, parameters, *, compensate_endianness):
    """Convert flat, fully parameterized U3/CNOT columns without QASM.

    Return None for other gates or parameterizations so the existing basis
    conversion and parser remain authoritative. Every successful conversion
    creates fresh operations and parameters; it cannot mutate the native
    column or a cached template.
    """
    import numpy as np
    from bqskit import Circuit
    from bqskit.ir.gates import CNOTGate, U3Gate
    from squander.gates.gates_Wrapper import CNOT, U3

    gates = circuit.get_Gates()
    if any(type(gate) not in (U3, CNOT) or
           (type(gate) is U3 and gate.get_Parameter_Num() != 3)
           for gate in gates):
        return None
    parameters = np.asarray(parameters, dtype=np.float64)
    width = circuit.get_Qbit_Num()
    converted = Circuit(width)
    u3 = U3Gate()
    cnot = CNOTGate()

    def wire(qubit):
        return width - 1 - qubit if compensate_endianness else qubit

    for gate in gates:
        target = wire(gate.get_Target_Qbit())
        if type(gate) is CNOT:
            converted.append_gate(
                cnot, (wire(gate.get_Control_Qbit()), target))
        else:
            # Native extraction already expands the internal half-angle.
            converted.append_gate(
                u3, target, list(gate.Extract_Parameters(parameters)))
    return converted


def copy_resolved_pam_circuit(circuit):
    """Copy flat U3/CNOT columns without recursive graph copying.

    Rebuild operations and their parameter lists using BQSKit's public API.
    Unknown gates and noncompact schedules retain the original deep copy.
    Both paths return an independent circuit; no cached template is mutated.
    """
    from bqskit import Circuit
    from bqskit.ir.gates import CNOTGate, U3Gate

    operations = list(circuit.operations_with_cycles())
    if any(type(operation.gate) not in (CNOTGate, U3Gate)
           for _, operation in operations):
        return circuit.copy()
    copied = Circuit(circuit.num_qudits, circuit.radixes)
    gates = {}
    for cycle, operation in operations:
        kind = type(operation.gate)
        if kind not in gates:
            gates[kind] = kind()
        actual_cycle = copied.append_gate(
            gates[kind], operation.location, list(operation.params))
        if actual_cycle != cycle:
            return circuit.copy()
    return copied

class _ParserCache:
    def __init__(self, max_entries):
        self.pid = os.getpid()
        self.max_entries = max_entries
        self.templates = OrderedDict()
        self.hits = 0
        self.misses = 0

    def decode(self, qasm):
        if qasm in self.templates:
            self.hits += 1
            template = self.templates[qasm]
            self.templates.move_to_end(qasm)
        else:
            self.misses += 1
            template = _decode(qasm)
            self.templates[qasm] = template
            if len(self.templates) > self.max_entries:
                self.templates.popitem(last=False)
        return copy_resolved_pam_circuit(template)


def decode_pam_qasm(qasm):
    """Decode normally outside a route; clone a template inside its scope."""
    cache = _active_cache.get()
    if cache is None or cache.pid != os.getpid():
        return _decode(qasm)
    return cache.decode(qasm)


@contextmanager
def pam_circuit_conversion_cache(max_entries=256):
    """Release all retained templates when the routing invocation ends."""
    if (not isinstance(max_entries, int) or isinstance(max_entries, bool)
            or max_entries <= 0):
        raise ValueError("max_entries must be a positive integer.")
    cache = _ParserCache(max_entries)
    token = _active_cache.set(cache)
    try:
        yield cache
    finally:
        _active_cache.reset(token)
        cache.templates.clear()
