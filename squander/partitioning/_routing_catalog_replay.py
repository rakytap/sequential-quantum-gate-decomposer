"""Memory bounded path catalogue loading and exact physical-frame expansion.

This representation changes storage only. It retains every archived local
program and validates fallback coverage before removing translated copies.
It never invokes synthesis. The ordinary full loader remains the default.
"""
from dataclasses import replace
import gzip
import hashlib
from pathlib import Path

from squander.decomposition._rewrite_audit_stream import _JSONReader


def bind_catalog_source(catalog, circuit, parameters, *, max_parameter_drift=1e-12):
    """Restore lossless audit angles after an older QASM cache rounded them.

    Cached implementations and all their prices are unchanged. The selected
    programs must still pass the ordinary local-unitary and exact-cover proof
    against this exact source; a rounded cache never becomes the authority for
    the experiment's input gate stream.
    """
    import numpy as np
    from squander.gates.qgd_Circuit import qgd_Circuit

    gates = circuit.get_Gates()
    def structure(value):
        return tuple((g.get_Name(), tuple(g.get_Involved_Qbits()), g.get_Parameter_Num())
                     for g in value.get_Gates())
    if circuit.get_Qbit_Num() != catalog.circuit.get_Qbit_Num() or structure(circuit) != structure(catalog.circuit):
        raise ValueError("Archived audit and routing catalogue have different gate structures.")
    parameters = np.asarray(parameters, dtype=np.float64)
    cached = np.asarray(catalog.parameters, dtype=np.float64)
    if (parameters.shape != cached.shape or not np.all(np.isfinite(parameters))
            or not np.all(np.isfinite(cached))
            or np.any(np.abs(parameters - cached) > max_parameter_drift)):
        raise ValueError("Routing catalogue differs from the audited source beyond export roundoff.")
    alternatives = {}
    for partition, values in catalog.alternatives.items():
        logicals = values[0].logical_qubits
        source = qgd_Circuit(circuit.get_Qbit_Num())
        blocks = []
        for gate_id in catalog.gate_orders[partition]:
            gate = gates[gate_id]
            start = gate.get_Parameter_Start_Index()
            blocks.append(parameters[start:start + gate.get_Parameter_Num()])
            source.add_Gate(gate)
        source = source.Remap_Qbits({q: i for i, q in enumerate(logicals)}, len(logicals)).get_Flat_Circuit()
        source_parameters = np.concatenate(blocks) if blocks else np.empty(0)
        payloads = {}
        rebound = []
        for alternative in values:
            key = id(alternative.payload)
            if key not in payloads:
                payloads[key] = replace(alternative.payload, source_circuit=source,
                                        source_parameters=source_parameters)
            rebound.append(replace(alternative, payload=payloads[key]))
        alternatives[partition] = tuple(rebound)
    return replace(catalog, circuit=circuit, parameters=parameters, alternatives=alternatives)


def load_compact_path_archive(path, expected_sha256=None):
    from . import routing

    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    if expected_sha256 is not None and digest.hexdigest() != expected_sha256.lower():
        raise ValueError("Routing OSR catalog SHA-256 mismatch.")
    archive = {}
    coverage_by_width = {}
    with gzip.open(path, "rt", encoding="utf-8") as source:
        reader = _JSONReader(source)
        reader.expect("{")
        while reader.peek() != "}":
            key = reader.value()
            reader.expect(":")
            if key not in ("partitions", "payloads"):
                archive[key] = reader.value()
            else:
                records = []
                reader.expect("[")
                while reader.peek() != "]":
                    record = reader.value()
                    if key == "payloads":
                        records.append(record)
                        if reader.peek() != "]":
                            reader.expect(",")
                        continue
                    width = len(record["logical_qubits"])
                    if not 1 <= width <= 3:
                        raise ValueError("Compact path replay supports blocks up to three qubits.")
                    # The archive is sorted: device topology follows partitions.
                    # Store compact transition keys for validation once read.
                    transitions = {(tuple(a["input_physical"]), tuple(a["output_physical"]))
                                   for a in record["alternatives"]}
                    if width in coverage_by_width and transitions != coverage_by_width[width]:
                        raise ValueError("Routing partitions have inconsistent boundary coverage.")
                    coverage_by_width.setdefault(width, transitions)
                    programs = {}
                    for alternative in record["alternatives"]:
                        programs.setdefault(alternative["payload"], alternative)
                    record["alternatives"] = list(programs.values())
                    records.append(record)
                    if reader.peek() != "]":
                        reader.expect(",")
                reader.expect("]")
                archive[key] = records
            if reader.peek() != "}":
                reader.expect(",")
        reader.expect("}")
    if (archive.get("schema") != "squander-routing-osr-catalog"
            or archive.get("schema_version") != routing.ROUTING_OSR_CATALOG_SCHEMA_VERSION
            or not archive.get("complete")):
        raise ValueError("Unsupported or incomplete routing catalogue.")
    topology = tuple(tuple(edge) for edge in archive["topology"])
    n = 1 + max(max(edge) for edge in topology)
    if routing._path_topology_order(topology, n) is None:
        raise ValueError("Compact physical frames require a path topology.")
    for width, transitions in coverage_by_width.items():
        if routing._expected_boundary_transition_keys(topology, n, width) - transitions:
            raise ValueError("Routing catalogue is missing fallback boundary coverage.")
    position = {physical: i for i, physical in enumerate(routing._path_topology_order(topology, n))}
    for record in archive["partitions"]:
        width = len(record["logical_qubits"])
        for alternative in record["alternatives"]:
            incoming = tuple(position[p] for p in alternative["input_physical"])
            outgoing = tuple(position[p] for p in alternative["output_physical"])
            offset = min(incoming)
            if set(incoming) != set(range(offset, offset + width)) or set(incoming) != set(outgoing):
                raise ValueError("Cached path program occupies a noncontiguous frame.")
            alternative["input_physical"] = [p - offset for p in incoming]
            alternative["output_physical"] = [p - offset for p in outgoing]
    return archive


def expand_path_cover(catalog, selected_partitions):
    """Materialize all translated/reflected frames for a selected cover only."""
    from . import routing

    if not catalog.implicit_path_frames:
        return catalog
    path = routing._path_topology_order(catalog.topology, catalog.circuit.get_Qbit_Num())
    if path is None:
        raise ValueError("Expected path topology.")
    edges = {frozenset(edge) for edge in catalog.topology}
    result = dict(catalog.alternatives)
    for partition in selected_partitions:
        by_boundary = {}
        for alternative in catalog.alternatives[partition]:
            width = len(alternative.logical_qubits)
            start = min(alternative.input_physical)
            incoming = tuple(p - start for p in alternative.input_physical)
            outgoing = tuple(p - start for p in alternative.output_physical)
            if set(incoming) != set(range(width)) or set(outgoing) != set(range(width)):
                raise ValueError("Compact catalogue frame is not contiguous.")
            for offset in range(len(path) - width + 1):
                for reflection in (False, True):
                    def physical(values):
                        return tuple(path[offset + (width - 1 - v if reflection else v)] for v in values)
                    pin, pout = physical(incoming), physical(outgoing)
                    embedding = tuple(pin[i] for i in alternative.payload.input_assignment)
                    if any(frozenset((embedding[u], embedding[v])) not in edges
                           for u, v in alternative.payload.topology):
                        continue
                    candidate = replace(alternative, input_physical=pin, output_physical=pout)
                    key = pin, pout
                    previous = by_boundary.get(key)
                    if previous is None or (candidate.cnot_count, candidate.single_qubit_count) < (
                            previous.cnot_count, previous.single_qubit_count):
                        by_boundary[key] = candidate
        result[partition] = tuple(by_boundary.values())
    return replace(catalog, alternatives=result, implicit_path_frames=False)
