"""Native identity-placement screens must measure the current residual."""

import ctypes
import re

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from squander.IO_interfaces.Qiskit_IO import get_Qiskit_Circuit
from squander.decomposition.qgd_Wide_Circuit_Optimization import (
    qgd_Wide_Circuit_Optimization as WCO,
)


@pytest.mark.parametrize("width", [3, 4])
@pytest.mark.parametrize("hs_guidance", [False, True])
def test_placement_screen_measures_residual_not_unset_optimizer_error(
    width, hs_guidance, capfd,
):
    source = QuantumCircuit(width)
    for qubit in range(width):
        source.u(0.13 * (qubit + 1), -0.17, 0.23, qubit)
    source.cx(0, 1)
    for qubit in range(width):
        source.u(-0.21, 0.07 * (qubit + 1), -0.11, qubit)
    target = Operator(source).data
    graph = [(qubit, qubit + 1) for qubit in range(width - 1)]
    optimizer = WCO({
        "strategy": "TreeSearch",
        "use_osr": True,
        "use_graph_search": True,
        "max_partition_size": width,
        "synthesis_acceptance_tolerance": 1e-10,
        "parallel": 0,
        "verbosity": 2,
        "random_seed": 20260926,
        "osr_use_hilbert_schmidt_guidance": hs_guidance,
        # Exercise screening itself, not a certified shortcut that can
        # legitimately finish this target before any placement is screened.
        "osr_certified_terminal_completion": False,
        "osr_geometric_completion_motifs": False,
        "osr_physical_cover_motifs": False,
        "osr_certified_cnot_deflation": False,
        "osr_placement_objective_ordering": True,
        "osr_trace_candidate_placements": True,
    })
    results = optimizer.DecomposePartition(
        target, optimizer.config, mini_topology=graph,
    )
    output = capfd.readouterr()
    objectives = [
        float(value) for value in re.findall(
            r"OSR placement screen: cnots=\d+, residual=\d+, objective=([^\s]+)",
            output.out + output.err,
        )
    ]
    assert objectives, "Native placement-screen diagnostics were not emitted"
    assert all(np.isfinite(value) and abs(value) < 2.0 for value in objectives)
    if hs_guidance:
        assert max(objectives) - min(objectives) > 1e-8
    assert results
    for circuit, parameters in results:
        exported = get_Qiskit_Circuit(circuit, parameters)
        native = circuit.get_Matrix(np.asarray(parameters, dtype=np.float64))
        matrix = Operator(exported).data
        for left, right in ((target, native), (target, matrix), (native, matrix)):
            error = max(0.0, 1.0 - abs(np.vdot(left, right)) ** 2 / 4 ** width)
            assert error <= 1e-10
        assert exported.count_ops().get("cx", 0) == 1
        for instruction in exported.data:
            if instruction.operation.name == "cx":
                support = frozenset(
                    exported.find_bit(qubit).index for qubit in instruction.qubits
                )
                assert support in {frozenset(edge) for edge in graph}


@pytest.mark.parametrize("width", [3, 4])
@pytest.mark.parametrize("continuation", [
    "osr_bidirectional_profile_refinement",
    "osr_certified_terminal_completion",
    "osr_certified_cnot_deflation",
])
def test_opt_in_two_sided_refinement_keeps_certified_fast_path(
    width, continuation, capfd,
):
    source = QuantumCircuit(width)
    for qubit in range(width):
        source.u(0.13 * (qubit + 1), -0.17, 0.23, qubit)
    source.cx(0, 1)
    for qubit in range(width):
        source.u(-0.21, 0.07 * (qubit + 1), -0.11, qubit)
    target = Operator(source).data
    graph = [(qubit, qubit + 1) for qubit in range(width - 1)]
    optimizer = WCO({
        "strategy": "TreeSearch",
        "use_osr": True,
        "use_graph_search": True,
        "max_partition_size": width,
        "synthesis_acceptance_tolerance": 1e-10,
        "parallel": 0,
        "verbosity": 2,
        "random_seed": 20260926,
        continuation: True,
        "osr_trace_candidate_placements": True,
    })
    results = optimizer.DecomposePartition(
        target, optimizer.config, mini_topology=graph,
    )
    captured = capfd.readouterr()
    frames = set(re.findall(
        r"OSR fixed-profile refinement: path_depth=\d+, frame=(input|output)",
        captured.out + captured.err,
    ))
    assert not frames, "A certified one-CNOT fit should not be refined"
    assert results
    for circuit, parameters in results:
        exported = get_Qiskit_Circuit(circuit, parameters)
        native = circuit.get_Matrix(np.asarray(parameters, dtype=np.float64))
        matrix = Operator(exported).data
        for left, right in ((target, native), (target, matrix), (native, matrix)):
            error = max(0.0, 1.0 - abs(np.vdot(left, right)) ** 2 / 4 ** width)
            assert error <= 1e-10
        assert exported.count_ops().get("cx", 0) == 1
        for instruction in exported.data:
            if instruction.operation.name == "cx":
                support = frozenset(
                    exported.find_bit(qubit).index for qubit in instruction.qubits
                )
                assert support in {frozenset(edge) for edge in graph}



@pytest.mark.parametrize("width", [3, 4])
def test_explicit_projection_restart_portfolio_is_honored_on_both_widths(
    width, capfd,
):
    source = QuantumCircuit(width)
    for qubit in range(width):
        source.u(0.13 * (qubit + 1), -0.17, 0.23, qubit)
    source.cx(0, 1)
    for qubit in range(width):
        source.u(-0.21, 0.07 * (qubit + 1), -0.11, qubit)
    target = Operator(source).data
    graph = [(qubit, qubit + 1) for qubit in range(width - 1)]
    optimizer = WCO({
        "strategy": "TreeSearch",
        "use_osr": True,
        "use_graph_search": True,
        "max_partition_size": width,
        "synthesis_acceptance_tolerance": 1e-10,
        "parallel": 0,
        "verbosity": 2,
        "random_seed": 20260926,
        "osr_hs_projection_restarts": 3,
        # Keep this test on the requested restart path rather than allowing
        # an unrelated certified continuation to finish the target first.
        "osr_certified_terminal_completion": False,
        "osr_geometric_completion_motifs": False,
        "osr_physical_cover_motifs": False,
        "osr_certified_cnot_deflation": False,
        "osr_eager_projection_restarts": True,
        "osr_trace_candidate_placements": True,
    })
    results = optimizer.DecomposePartition(
        target, optimizer.config, mini_topology=graph,
    )
    captured = capfd.readouterr()
    restarts = {
        int(value) for value in re.findall(
            r"OSR projection restart: path_depth=\d+, restart_index=(\d+)",
            captured.out + captured.err,
        )
    }
    assert {1, 2} <= restarts
    assert results
    for circuit, parameters in results:
        exported = get_Qiskit_Circuit(circuit, parameters)
        native = circuit.get_Matrix(np.asarray(parameters, dtype=np.float64))
        matrix = Operator(exported).data
        for left, right in ((target, native), (target, matrix), (native, matrix)):
            error = max(0.0, 1.0 - abs(np.vdot(left, right)) ** 2 / 4 ** width)
            assert error <= 1e-10
        assert exported.count_ops().get("cx", 0) == 1
        for instruction in exported.data:
            if instruction.operation.name == "cx":
                support = frozenset(
                    exported.find_bit(qubit).index for qubit in instruction.qubits
                )
                assert support in {frozenset(edge) for edge in graph}


@pytest.mark.parametrize("width", [3, 4])
@pytest.mark.parametrize("inverse", [False, True])
def test_certified_terminal_continuation_uses_matrix_only_and_strict_accuracy(
    width, inverse, capfd,
):
    rng = np.random.default_rng(20261005)
    source = QuantumCircuit(width)
    edges = [(0, 1), (0, 2), (1, 2)]
    for digit in (2, 0, 1, 2):
        first, second = edges[digit]
        source.u(*rng.uniform(-np.pi, np.pi, 3), first)
        source.u(*rng.uniform(-np.pi, np.pi, 3), second)
        source.cx(first, second)
    for qubit in range(width):
        source.u(*rng.uniform(-np.pi, np.pi, 3), qubit)
    target = Operator(source.inverse() if inverse else source).data
    graph = [
        (first, second)
        for first in range(width) for second in range(first + 1, width)
    ]
    optimizer = WCO({
        "strategy": "TreeSearch",
        "use_osr": True,
        "use_graph_search": True,
        "max_partition_size": width,
        "synthesis_acceptance_tolerance": 1e-10,
        "parallel": 0,
        "verbosity": 2,
        "random_seed": 20260926,
        "osr_certified_terminal_completion": True,
        "osr_trace_candidate_placements": True,
    })
    results = optimizer.DecomposePartition(
        target, optimizer.config, mini_topology=graph,
    )
    captured = capfd.readouterr()
    certificates = re.findall(
        r"OSR terminal completion: parent_depth=\d+, frame=(?:input|output)"
        r", edge=\d+, cnots=(\d+), floor=(\d+), hs=([^\s]+)"
        r", certified=1",
        captured.out + captured.err,
    )
    product_certificates = re.findall(
        r"OSR product-cut completion: spectator=\d+, cnots=(\d+), "
        r"floor=(\d+), hs=([^\s]+), certified=1",
        captured.out + captured.err,
    )
    assert results
    for cnots, floor, hs in certificates + product_certificates:
        assert int(cnots) == int(floor)
        assert float(hs) < 1e-10
    for circuit, parameters in results:
        exported = get_Qiskit_Circuit(circuit, parameters)
        native = circuit.get_Matrix(np.asarray(parameters, dtype=np.float64))
        matrix = Operator(exported).data
        for left, right in ((target, native), (target, matrix), (native, matrix)):
            error = max(0.0, 1.0 - abs(np.vdot(left, right)) ** 2 / 4 ** width)
            assert error <= 1e-10
        assert exported.count_ops().get("cx", 0) <= 4
        for instruction in exported.data:
            if instruction.operation.name == "cx":
                support = frozenset(
                    exported.find_bit(qubit).index for qubit in instruction.qubits
                )
                assert support in {frozenset(edge) for edge in graph}


def test_geometric_motifs_rescue_full_rank_three_qubit_root_without_source_seeds(
    capfd,
):
    # Only this matrix reaches native synthesis, not a source circuit/ansatz.
    rng = np.random.default_rng(20261007)
    raw = rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8))
    target, _ = np.linalg.qr(raw)
    graph = [(0, 1), (1, 2)]
    optimizer = WCO({
        "strategy": "TreeSearch",
        "use_osr": True,
        "use_graph_search": True,
        "max_partition_size": 3,
        "synthesis_acceptance_tolerance": 1e-10,
        "parallel": 0,
        "verbosity": 2,
        "random_seed": 20260926,
        "osr_geometric_completion_motifs": True,
        "osr_trace_candidate_placements": True,
    })
    results = optimizer.DecomposePartition(
        target, optimizer.config, mini_topology=graph,
    )
    captured = capfd.readouterr()
    certificates = re.findall(
        r"OSR geometric completion: cnots=(\d+), floor=(\d+), "
        r"bound=(\d+), hs=([^\s]+), certified=1",
        captured.out + captured.err,
    )
    assert certificates
    assert results
    for cnots, floor, bound, hs in certificates:
        assert int(floor) <= int(cnots) <= int(bound)
        assert float(hs) < 1e-10
    for circuit, parameters in results:
        exported = get_Qiskit_Circuit(circuit, parameters)
        native = circuit.get_Matrix(np.asarray(parameters, dtype=np.float64))
        matrix = Operator(exported).data
        for left, right in ((target, native), (target, matrix), (native, matrix)):
            error = max(0.0, 1.0 - abs(np.vdot(left, right)) ** 2 / 4 ** 3)
            assert error <= 1e-10
        assert exported.count_ops().get("cx", 0) <= 14
        for instruction in exported.data:
            if instruction.operation.name == "cx":
                support = frozenset(
                    exported.find_bit(qubit).index for qubit in instruction.qubits
                )
                assert support in {frozenset(edge) for edge in graph}

def test_geometric_refinement_preserves_structured_osr_certificate():
    rng = np.random.default_rng(20261009)
    source = QuantumCircuit(3)
    graph = [(0, 1), (0, 2), (1, 2)]
    for digit in (0, 0, 1, 0, 0):
        first, second = graph[digit]
        source.u(*rng.uniform(-np.pi, np.pi, 3), first)
        source.u(*rng.uniform(-np.pi, np.pi, 3), second)
        source.cx(first, second)
    for qubit in range(3):
        source.u(*rng.uniform(-np.pi, np.pi, 3), qubit)
    target = Operator(source).data
    counts = []
    for enabled in (False, True):
        optimizer = WCO({
            "strategy": "TreeSearch",
            "use_osr": True,
            "use_graph_search": True,
            "max_partition_size": 3,
            "synthesis_acceptance_tolerance": 1e-10,
            "parallel": 0,
            "verbosity": 0,
            "random_seed": 20260926,
            "osr_geometric_completion_motifs": enabled,
        })
        results = optimizer.DecomposePartition(
            target, optimizer.config, mini_topology=graph,
        )
        assert results
        verified_counts = []
        for circuit, parameters in results:
            exported = get_Qiskit_Circuit(circuit, parameters)
            native = circuit.get_Matrix(np.asarray(parameters, dtype=np.float64))
            matrix = Operator(exported).data
            for left, right in ((target, native), (target, matrix), (native, matrix)):
                error = max(0.0, 1.0 - abs(np.vdot(left, right)) ** 2 / 4 ** 3)
                assert error <= 1e-10
            verified_counts.append(exported.count_ops().get("cx", 0))
        counts.append(min(verified_counts))
    assert counts[1] <= counts[0]


@pytest.mark.parametrize("sign", [1, -1])
def test_cnot_deflation_product_seed_attains_schmidt_overlap_bound(sign):
    cnot = QuantumCircuit(2)
    cnot.cx(1, 0)
    product = QuantumCircuit(2)
    product.rx(sign * np.pi / 2, 0)
    product.rz(sign * np.pi / 2, 1)
    target = Operator(cnot).data
    approximation = Operator(product).data
    overlap = abs(np.vdot(target, approximation)) / 4
    assert overlap == pytest.approx(1 / np.sqrt(2), abs=1e-14)
    assert 1 - overlap ** 2 == pytest.approx(0.5, abs=1e-14)


def test_native_cnot_deflation_reduces_only_strictly_certified_incumbents(capfd):
    # Test deletion itself on a certified redundant incumbent. A matrix-only
    # search may now find the smaller chart directly, legitimately bypassing
    # deflation, so its route must not determine component-test coverage.
    library = None
    for name in ("libqgd.so", "libqgd.dylib", "qgd.dll"):
        try:
            candidate = ctypes.CDLL(name)
            deletion = candidate.qgd_osr_refit_cnot_deletion
        except (OSError, AttributeError):
            continue
        library = candidate
        break
    assert library is not None, "Native CNOT deletion component is unavailable"
    deletion.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_double, ctypes.c_void_p, ctypes.c_int,
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_int,
    ]
    deletion.restype = ctypes.c_int
    incumbent = QuantumCircuit(4)
    incumbent.cx(1, 0)
    incumbent.u(0.0, 0.0, np.pi / 2, 0)
    incumbent.cx(1, 0)
    incumbent_matrix = Operator(incumbent).data
    inverse_matrix = np.ascontiguousarray(incumbent_matrix.conj().T)
    targets = np.array([0, 0], dtype=np.int32)
    controls = np.array([1, 1], dtype=np.int32)
    angles = np.zeros(24, dtype=np.float64)
    angles[8] = np.pi / 2
    reduced_angles = np.empty(18, dtype=np.float64)
    hs, process = ctypes.c_double(), ctypes.c_double()
    certified = False
    for removed in range(2):
        for branch in (1, -1):
            status = deletion(
                inverse_matrix.ctypes.data, 4, 16,
                targets.ctypes.data, controls.ctypes.data, 2,
                angles.ctypes.data, len(angles), removed, branch, 2, 1000, 20260926,
                1e-10, reduced_angles.ctypes.data, len(reduced_angles),
                ctypes.byref(hs), ctypes.byref(process), 1,
            )
            assert status in (0, 1)
            if status == 1:
                certified = True
                break
        if certified:
            break
    assert certified, "Native deletion did not certify the one-CNOT chart"
    assert np.all(np.isfinite(reduced_angles))
    assert hs.value < 1e-10
    assert process.value <= 1e-10
    reduced = QuantumCircuit(4)
    for start, wire in ((0, 0), (3, 1)):
        theta, phi, lam = reduced_angles[start:start + 3]
        reduced.u(2 * theta, phi, lam, wire)
    reduced.cx(1, 0)
    for wire in range(4):
        theta, phi, lam = reduced_angles[6 + 3 * wire:9 + 3 * wire]
        reduced.u(2 * theta, phi, lam, wire)
    reduced_matrix = Operator(reduced).data
    phase = np.exp(-1j * np.angle(np.vdot(incumbent_matrix, reduced_matrix)))
    squared = np.linalg.norm(incumbent_matrix - phase * reduced_matrix) ** 2 / 16
    assert squared - squared ** 2 / 4 <= 1e-10
    assert reduced.count_ops().get("cx", 0) == incumbent.count_ops().get("cx", 0) - 1

    rng = np.random.default_rng(20261005)
    source = QuantumCircuit(4)
    edges = [(0, 1), (0, 2), (1, 2)]
    for digit in (2, 0, 1, 2):
        first, second = edges[digit]
        source.u(*rng.uniform(-np.pi, np.pi, 3), first)
        source.u(*rng.uniform(-np.pi, np.pi, 3), second)
        source.cx(first, second)
    for qubit in range(4):
        source.u(*rng.uniform(-np.pi, np.pi, 3), qubit)
    target = Operator(source.inverse()).data
    graph = [(a, b) for a in range(4) for b in range(a + 1, 4)]
    optimizer = WCO({
        "strategy": "TreeSearch",
        "use_osr": True,
        "use_graph_search": True,
        "max_partition_size": 4,
        "synthesis_acceptance_tolerance": 1e-10,
        "parallel": 0,
        "verbosity": 2,
        "random_seed": 20260926,
        "use_float": False,
        "osr_certified_terminal_completion": False,
        "osr_geometric_completion_motifs": False,
        "osr_physical_cover_motifs": False,
        "osr_certified_cnot_deflation": True,
        "osr_cnot_deflation_basin_hops": 2,
        "osr_trace_candidate_placements": True,
    })
    # Supply only the matrix and device graph, never source angles/structure.
    results = optimizer.DecomposePartition(
        target, optimizer.config, mini_topology=graph,
    )
    captured = capfd.readouterr()
    certificates = re.findall(
        r"OSR CNOT deflation: from=(\d+), to=(\d+), removed=\d+, "
        r"branch=-?1, hs=([^\s]+), process_infidelity=([^\s]+), certified=1",
        captured.out + captured.err,
    )
    assert certificates
    assert results
    for parent, child, hs, error in certificates:
        assert int(child) == int(parent) - 1
        assert float(hs) < 1e-10
        assert float(error) <= 1e-10
    for circuit, parameters in results:
        exported = get_Qiskit_Circuit(circuit, parameters)
        native = circuit.get_Matrix(np.asarray(parameters, dtype=np.float64))
        matrix = Operator(exported).data
        for left, right in ((target, native), (target, matrix), (native, matrix)):
            error = max(0.0, 1.0 - abs(np.vdot(left, right)) ** 2 / 4 ** 4)
            assert error <= 1e-10
        assert exported.count_ops().get("cx", 0) <= 6
        for instruction in exported.data:
            if instruction.operation.name == "cx":
                support = frozenset(
                    exported.find_bit(qubit).index for qubit in instruction.qubits
                )
                assert support in {frozenset(edge) for edge in graph}
