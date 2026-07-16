"""
Wide-circuit optimization: partition large circuits into subcircuits, re-decompose
them, and optionally route or fuse results according to configuration.
"""

from squander.decomposition.qgd_N_Qubit_Decompositions_Wrapper import (
    qgd_N_Qubit_Decomposition_adaptive as N_Qubit_Decomposition_adaptive,
    qgd_N_Qubit_Decomposition_Tree_Search as N_Qubit_Decomposition_Tree_Search,
    qgd_N_Qubit_Decomposition_Tabu_Search as N_Qubit_Decomposition_Tabu_Search,
)
from squander import N_Qubit_Decomposition_custom, N_Qubit_Decomposition
from squander.gates.qgd_Circuit import qgd_Circuit as Circuit
from squander.utils import CompareCircuits

import numpy as np
from qiskit import QuantumCircuit

from typing import List, Callable, Tuple, Optional, Set, Dict, Any, cast, Union

import multiprocessing as mp
from multiprocessing import Process, Pool, parent_process
import os, contextlib, collections, time


from squander.partitioning.partition import PartitionCircuit
from squander.partitioning.tools import translate_param_order, build_dependency
from squander.synthesis.qgd_SABRE import qgd_SABRE as SABRE

try:
    from bqskit.compiler.basepass import BasePass as _BQSKitBasePass
    from bqskit.passes.synthesis.synthesis import SynthesisPass as _BQSKitSynthesisPass
except Exception:
    _BQSKitBasePass = object
    _BQSKitSynthesisPass = object


_SQUANDER_BQSKIT_SYNTHESIS_CONFIG = None

_SQUANDER_NATIVE_STRATEGIES = frozenset(
    ("TreeSearch", "TabuSearch", "Adaptive", "Custom")
)

SQUANDER_FLOAT64_TOLERANCE = 1e-10
SQUANDER_FLOAT32_TOLERANCE = 1e-8
BQSKIT_FLOAT64_SYNTHESIS_VALIDATION_TOLERANCE = 1e-8
BQSKIT_FLOAT32_SYNTHESIS_VALIDATION_TOLERANCE = 1e-8
CIRCUIT_FLOAT64_VALIDATION_TOLERANCE = 1e-8
CIRCUIT_FLOAT32_VALIDATION_TOLERANCE = 1e-6


def _config_uses_float32(config):
    return bool(config.get("use_float", False))


def _default_squander_tolerance(config):
    return (
        SQUANDER_FLOAT32_TOLERANCE
        if _config_uses_float32(config)
        else SQUANDER_FLOAT64_TOLERANCE
    )


def _default_bqskit_synthesis_validation_tolerance(config):
    return (
        BQSKIT_FLOAT32_SYNTHESIS_VALIDATION_TOLERANCE
        if _config_uses_float32(config)
        else BQSKIT_FLOAT64_SYNTHESIS_VALIDATION_TOLERANCE
    )


def _default_circuit_validation_tolerance(config):
    return (
        CIRCUIT_FLOAT32_VALIDATION_TOLERANCE
        if _config_uses_float32(config)
        else CIRCUIT_FLOAT64_VALIDATION_TOLERANCE
    )


def _squander_validation_tolerance(config):
    return config.get(
        "tolerance",
        _default_squander_tolerance(config),
    )


def _circuit_validation_tolerance(config):
    """Return the allowed whole-circuit infidelity for state-vector checks."""

    return config.get(
        "circuit_validation_tolerance",
        _default_circuit_validation_tolerance(config),
    )


def _bqskit_synthesis_validation_tolerance(config):
    return config.get(
        "bqskit_synthesis_validation_tolerance",
        _default_bqskit_synthesis_validation_tolerance(config),
    )


def _copy_bqskit_synthesis_config(config):
    """Copy only plain data needed by BQSKit worker processes."""

    def copy_value(value):
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, tuple):
            copied = [copy_value(item) for item in value]
            return tuple(item for item in copied if item is not _SKIP_CONFIG_VALUE)
        if isinstance(value, list):
            copied = [copy_value(item) for item in value]
            return [item for item in copied if item is not _SKIP_CONFIG_VALUE]
        if isinstance(value, dict):
            copied = {}
            for key, item in value.items():
                copied_item = copy_value(item)
                if copied_item is not _SKIP_CONFIG_VALUE:
                    copied[key] = copied_item
            return copied
        return _SKIP_CONFIG_VALUE

    copied_config = {}
    for key, value in config.items():
        copied_value = copy_value(value)
        if copied_value is not _SKIP_CONFIG_VALUE:
            copied_config[key] = copied_value
    return copied_config


_SKIP_CONFIG_VALUE = object()


# ---------------------------------------------------------------------------
# Helper: insert a SWAP as 3 CNOTs so BQSKit's scoring function weights
# them honestly (3 two-qubit ops instead of 1).  SEQPAM then avoids
# unnecessary SWAP insertions.
# ---------------------------------------------------------------------------
def _add_swap_as_cnots(circuit, a, b):
    """Append CNOT(a,b); CNOT(b,a); CNOT(a,b) — equivalent to SWAP(a,b)."""
    from bqskit.ir.gates import CNOTGate
    circuit.append_gate(CNOTGate(), [a, b])
    circuit.append_gate(CNOTGate(), [b, a])
    circuit.append_gate(CNOTGate(), [a, b])


# ---------------------------------------------------------------------------
# Module-level EAPP monkey-patch for SWAP fallback.
# BQSKit's Compiler starts a runtime server via Popen([sys.executable, ...]),
# a fresh Python process.  Class-level monkey-patches applied in the parent
# are invisible there.  We use an environment variable that the Popen child
# inherits; when this module is imported inside a worker, the env var triggers
# the patch.
# ---------------------------------------------------------------------------
def _append_topology_safe(new_c, op, topo_edges, width):
    """Append *op* to *new_c*, using SWAP bridges for edges not in *topo_edges*.

    For gates with ≥3 qubits, decomposes via :func:`squander.utils.circuit_to_CNOT_basis`
    and recurses on each resulting gate.
    """

    loc = list(op.location)
    gate = op.gate
    params = list(op.params) if op.params else None

    if gate.num_qudits == 1:
        if params:
            new_c.append_gate(gate, loc, params)
        else:
            new_c.append_gate(gate, loc)
        return

    if gate.num_qudits == 2:
        u, v = loc[0], loc[1]
        if (u, v) in topo_edges:
            if params:
                new_c.append_gate(gate, [u, v], params)
            else:
                new_c.append_gate(gate, [u, v])
            return
        # Edge not in topology — find shortest SWAP path u↔v via BFS.
        adj = {i: set() for i in range(width)}
        for a, b in topo_edges:
            adj[a].add(b)
            adj[b].add(a)
        from collections import deque
        parent = {v: None}
        q = deque([v])
        while q:
            node = q.popleft()
            if node == u:
                break
            for nb in adj.get(node, set()):
                if nb not in parent:
                    parent[nb] = node
                    q.append(nb)
        if u not in parent:
            # Cannot bridge this edge on the given topology.
            raise ValueError(f"Cannot bridge ({u},{v}) on topology")
        # Reconstruct path v -> ... -> u, then SWAP v along the path until it
        # is adjacent to u, apply the gate, and unwind those same SWAPs.
        path = [u]
        node = u
        while parent[node] is not None:
            node = parent[node]
            path.append(node)
        path = list(reversed(path))
        swaps = list(zip(path[:-2], path[1:-1]))
        cur = v
        for a, b in swaps:
            _add_swap_as_cnots(new_c, a, b)
            cur = b
        if params:
            new_c.append_gate(gate, [u, cur], params)
        else:
            new_c.append_gate(gate, [u, cur])
        for a, b in reversed(swaps):
            _add_swap_as_cnots(new_c, a, b)
        return

    # gate.num_qudits >= 3: decompose to CNOT basis via Squander's utility
    from bqskit.ir.lang.qasm2 import OPENQASM2Language
    from qiskit import qasm2

    # 1) Build a minimal BQSKit circuit containing just this gate
    from bqskit import Circuit as _BQCircuit
    tmp_bq = _BQCircuit(width)
    if params:
        tmp_bq.append_gate(gate, loc, params)
    else:
        tmp_bq.append_gate(gate, loc)

    # 2) Encode to QASM, then decode via Squander
    qasm_str = OPENQASM2Language().encode(tmp_bq)
    from squander import Qiskit_IO as _QIO
    qiskit_tmp = qasm2.loads(qasm_str)
    sq_tmp, sq_params = _QIO.convert_Qiskit_to_Squander(qiskit_tmp)

    # 3) Decompose to CNOT basis
    from squander.utils import circuit_to_CNOT_basis
    sq_decomp, sq_decomp_params = circuit_to_CNOT_basis(sq_tmp, sq_params)

    # 4) Convert back to BQSKit and recurse on each gate
    qiskit_decomp = _QIO.get_Qiskit_Circuit(sq_decomp, sq_decomp_params)
    bq_decomp = OPENQASM2Language().decode(qasm2.dumps(qiskit_decomp))
    for bq_op in bq_decomp:
        _append_topology_safe(new_c, bq_op, topo_edges, width)


def _bqskit_location_respects_topology(location, topo_edges):
    """Return true if ``location`` can be hosted by ``topo_edges``."""
    loc = tuple(int(q) for q in location)
    if len(loc) <= 1:
        return True
    if len(loc) == 2:
        return (loc[0], loc[1]) in topo_edges or (loc[1], loc[0]) in topo_edges

    wanted = set(loc)
    seen = {loc[0]}
    stack = [loc[0]]
    adjacency = {q: set() for q in wanted}
    for u, v in topo_edges:
        if u in wanted and v in wanted:
            adjacency[u].add(v)
            adjacency[v].add(u)
    while stack:
        cur = stack.pop()
        for nxt in adjacency.get(cur, ()):
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return wanted <= seen


def _assert_circuit_respects_topology(circuit, topo_edges):
    """Raise AssertionError if ``circuit`` violates ``topo_edges``.

    Topology violations indicate a critical logic bug — the circuit cannot
    physically execute on the target hardware.  Execution must stop
    immediately so the root cause can be investigated and fixed.
    """
    for op in circuit:
        if op.gate.num_qudits <= 1:
            continue
        if not _bqskit_location_respects_topology(op.location, topo_edges):
            raise AssertionError(
                f"BUG: circuit contains {op.gate.name} on {list(op.location)}, "
                f"outside topology {sorted(topo_edges)}."
            )


def _fallback_circuit_for_permutation(original_circuit, graph, pi, po):
    """Build a topology-valid fallback for ``Po.T @ U @ Pi``.

    ``original_circuit`` is the block circuit passed into BQSKit's
    EmbedAllPermutationsPass.  ``graph`` is the block-local coupling graph
    selected by EAPP for this synthesis attempt.
    """
    from bqskit import Circuit as _BQCircuit

    width = original_circuit.num_qudits
    if len(pi) != width or len(po) != width:
        raise _SquanderSynthesisFailed(
            f"Permutation width mismatch for fallback: {pi}, {po}, width={width}."
        )

    topo_edges = set()
    for u, v in graph:
        topo_edges.add((u, v))
        topo_edges.add((v, u))

    fallback = _BQCircuit(width, original_circuit.radixes)

    for a, b in _topo_perm_to_swaps(pi, topo_edges, width):
        if (a, b) not in topo_edges:
            raise _SquanderSynthesisFailed(
                f"Cannot realize input permutation {pi} on topology {sorted(topo_edges)}."
            )
        _add_swap_as_cnots(fallback, a, b)

    for op in original_circuit:
        _append_topology_safe(fallback, op, topo_edges, width)

    po_inv = tuple(po.index(k) for k in range(width))
    for a, b in _topo_perm_to_swaps(po_inv, topo_edges, width):
        if (a, b) not in topo_edges:
            raise _SquanderSynthesisFailed(
                f"Cannot realize output permutation {po} on topology {sorted(topo_edges)}."
            )
        _add_swap_as_cnots(fallback, a, b)

    _assert_circuit_respects_topology(fallback, topo_edges)
    return fallback


async def _squander_synthesize_or_fallback(
    inner_synthesis,
    target,
    target_data,
    original_circuit,
    graph,
    pi,
    po,
):
    """Run Squander synthesis, falling back only for explicit Squander misses."""
    try:
        return await inner_synthesis.synthesize(target, target_data)
    except _SquanderSynthesisFailed:
        return _fallback_circuit_for_permutation(original_circuit, graph, pi, po)


def _patch_eapp_if_needed():
    """Monkey-patch EAPP.run to catch Squander OSR failures per permutation.

    IMPORTANT: This patch fully replaces ``EmbedAllPermutationsPass.run``.
    It was written against BQSKit's internal EAPP implementation as of
    the pip-installed version (see pyproject.toml / requirements for the
    exact version).  If BQSKit changes its EAPP internals (scoring function,
    subtopology selection, permutation handling, or pass data keys), this
    patch may silently diverge and should be re-audited against the new
    BQSKit source.
    """
    import os as _os
    if not _os.environ.get('_SQUANDER_EAPP_FALLBACK_PATCH'):
        return

    from bqskit.passes.mapping.embed import EmbedAllPermutationsPass as __EAPP
    if getattr(__EAPP.run, "_squander_fallback_patch", False):
        return

    async def __patched_eapp_run(self, circuit, data):
        import copy as _copy
        import itertools as _it
        import logging as _logging
        from bqskit.compiler.machine import MachineModel as _MachineModel
        from bqskit.passes.mapping.topology import SubtopologySelectionPass as _STSP
        from bqskit.qis.graph import CouplingGraph as _CouplingGraph
        from bqskit.qis.permutation import PermutationMatrix as _PermutationMatrix
        from bqskit.runtime import get_runtime as _get_runtime

        _logger = _logging.getLogger("bqskit.passes.mapping.embed")
        utry = data.target

        if not all(r == utry.radixes[0] for r in utry.radixes):
            raise NotImplementedError(
                'PermutationAwareSynthesisPass only supports unitaries '
                'with the same radix on all qudits currently.',
            )

        width = utry.num_qudits
        perms = list(_it.permutations(range(width)))
        no_perm = [tuple(range(width))]
        Pis = [
            _PermutationMatrix.from_qudit_location(width, utry.radixes[0], p)
            for p in perms
        ]
        Pos = [
            _PermutationMatrix.from_qudit_location(width, utry.radixes[0], p)
            for p in perms
        ]

        if self.input_perm and self.output_perm:
            permsbyperms = list(_it.product(perms, perms))
            targets = [Po.T @ utry @ Pi for Pi, Po in _it.product(Pis, Pos)]
        elif self.input_perm:
            permsbyperms = list(_it.product(perms, no_perm))
            targets = [utry @ Pi for Pi in Pis]
        elif self.output_perm:
            permsbyperms = list(_it.product(no_perm, perms))
            targets = [Po.T @ utry for Po in Pos]
        else:
            _logger.warning('No permutation is being used in PAS.')
            permsbyperms = list(_it.product(no_perm, no_perm))
            targets = [utry]

        if self.vary_topology and width != 1:
            if _STSP.key not in data:
                raise RuntimeError(
                    'Cannot find subtopologies, try running a'
                    ' SubtopologySelectionPass first.',
                )
            if width not in data[_STSP.key]:
                raise RuntimeError(
                    'Subtopology information for block size'
                    f' {width} is not available.',
                )
            graphs = data[_STSP.key][width]
        else:
            graphs = [_CouplingGraph.all_to_all(width)]

        datas = []
        for graph in graphs:
            model = _MachineModel(
                circuit.num_qudits, graph,
                data.gate_set, data.model.radixes,
            )
            target_data = _copy.deepcopy(data)
            target_data.model = model
            datas.append(target_data)

        extended_targets = []
        extended_datas = []
        extended_graphs = []
        extended_perms = []
        original_circuits = []
        for target_index, target in enumerate(targets):
            for graph_index, graph in enumerate(graphs):
                extended_targets.append(target)
                extended_datas.append(datas[graph_index])
                extended_graphs.append(graph)
                extended_perms.append(permsbyperms[target_index])
                original_circuits.append(circuit)

        circuits = await _get_runtime().map(
            _squander_synthesize_or_fallback,
            [self.inner_synthesis] * len(extended_targets),
            extended_targets,
            extended_datas,
            original_circuits,
            extended_graphs,
            [perm[0] for perm in extended_perms],
            [perm[1] for perm in extended_perms],
        )

        perm_data = {}
        all_perms = list(_it.permutations(range(width)))
        for i, synthesized in enumerate(circuits):
            graph = extended_graphs[i]
            perm = extended_perms[i]

            if graph not in perm_data:
                perm_data[graph] = {}

            if perm in perm_data[graph]:
                s1 = self.scoring_fn(perm_data[graph][perm])
                s2 = self.scoring_fn(synthesized)
                if s2 < s1:
                    perm_data[graph][perm] = synthesized
            else:
                perm_data[graph][perm] = synthesized

            for univ_perm in all_perms[1:]:
                renumber_c = synthesized.copy()
                renumber_c.renumber_qudits(univ_perm)
                new_pi = tuple(univ_perm[j] for j in perm[0])
                new_pf = tuple(univ_perm[j] for j in perm[1])
                new_graph = renumber_c.coupling_graph
                if new_graph not in perm_data:
                    perm_data[new_graph] = {}

                new_perm = (new_pi, new_pf)
                if new_perm not in perm_data[new_graph]:
                    perm_data[new_graph][new_perm] = renumber_c
                else:
                    s1 = self.scoring_fn(perm_data[new_graph][new_perm])
                    s2 = self.scoring_fn(renumber_c)
                    if s2 < s1:
                        perm_data[new_graph][new_perm] = renumber_c

        if circuit.gate_set.issubset(data.model.gate_set):
            for univ_perm in _it.permutations(range(width)):
                uperm = (univ_perm, univ_perm)
                renumber_c = circuit.copy()
                renumber_c.renumber_qudits(univ_perm)
                new_graph = renumber_c.coupling_graph
                new_score = self.scoring_fn(renumber_c)
                for graph, graph_data in perm_data.items():
                    if all(e in graph for e in new_graph):
                        if uperm not in graph_data:
                            graph_data[uperm] = renumber_c
                        elif new_score < self.scoring_fn(graph_data[uperm]):
                            graph_data[uperm] = renumber_c

        data['permutation_data'] = perm_data

    __patched_eapp_run._squander_fallback_patch = True
    __EAPP.run = __patched_eapp_run


_patch_eapp_if_needed()


class SquanderPartitioner(_BQSKitBasePass):
    """BQSKit pass: replace circuit body with Squander ILP partition blocks."""

    def __init__(self, max_partition_size):
        super().__init__()
        self.max_partition_size = max_partition_size

    async def run(self, circuit, data=None):
        from qiskit import qasm2, QuantumCircuit
        from squander import Qiskit_IO
        from bqskit import Circuit as BQSKitCircuit
        from bqskit.ir.lang.qasm2 import OPENQASM2Language

        try:
            circ_qiskit = QuantumCircuit.from_qasm_str(
                OPENQASM2Language().encode(circuit)
            )
        except Exception:
            # Circuit contains gates that can't be QASM-encoded (e.g.
            # ConstantUnitaryGate from a prior pass).  Keep as-is.
            return

        circ, orig_parameters = Qiskit_IO.convert_Qiskit_to_Squander(circ_qiskit)
        partitioned_circuit, parameters, _ = PartitionCircuit(
            circ, orig_parameters, self.max_partition_size, strategy="ilp"
        )
        partitioned_circuit_bqskit = BQSKitCircuit(circ.get_Qbit_Num())
        for subcircuit in partitioned_circuit.get_Gates():
            if not isinstance(subcircuit, Circuit):
                raise RuntimeError(
                    "Squander ILP partitioning returned a non-block gate; "
                    "BQSKit SEQPAM requires partition blocks."
                )

            involved_qbits = sorted(subcircuit.get_Qbits())
            qbit_map = {qbit: idx for idx, qbit in enumerate(involved_qbits)}
            subcircuit_parameters = parameters[
                subcircuit.get_Parameter_Start_Index() :
                subcircuit.get_Parameter_Start_Index() + subcircuit.get_Parameter_Num()
            ]
            remapped_subcircuit = subcircuit.Remap_Qbits(qbit_map, len(involved_qbits))
            subcircuit_qiskit = Qiskit_IO.get_Qiskit_Circuit(
                remapped_subcircuit.get_Flat_Circuit(),
                np.asarray(subcircuit_parameters, dtype=np.float64),
            )
            subcircuit_bqskit = OPENQASM2Language().decode(qasm2.dumps(subcircuit_qiskit))
            partitioned_circuit_bqskit.append_circuit(
                subcircuit_bqskit,
                involved_qbits,
                True,
                True,
            )
        circuit.become(partitioned_circuit_bqskit, False)


class SquanderSynthesisPass(_BQSKitSynthesisPass):
    """BQSKit synthesis pass: optimize partition blocks with Squander.

    Raises _SquanderSynthesisFailed when the configured Squander synthesis
    strategy cannot produce a valid circuit for the requested subtopology. The
    monkey-patched EmbedAllPermutationsPass catches this and installs a
    SWAP-correct original-block fallback.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        cfg = _SQUANDER_BQSKIT_SYNTHESIS_CONFIG
        if not cfg:
            # Workers spawned via Popen inherit env vars but not Python
            # globals.  The main process serializes the config to
            # _SQUANDER_BQSKIT_CONFIG before spawning workers.
            import os as _os, json as _json
            _env = _os.environ.get('_SQUANDER_BQSKIT_CONFIG')
            if _env:
                cfg = _json.loads(_env)
        self.config = dict(cfg or {})

    @staticmethod
    def _data_topology(data, qbit_num):
        """Return block subtopology from *data*.

        BQSKit labels are reversed when circuits are converted through
        Squander/Qiskit, so the topology supplied to Squander is reversed too.
        """
        if data is None or getattr(data, "model", None) is None:
            return None

        edges = []
        for u, v in data.model.coupling_graph:
            if u == v:
                continue
            edges.append((qbit_num - 1 - int(u), qbit_num - 1 - int(v)))

        all_edges = {
            frozenset((i, j))
            for i in range(qbit_num)
            for j in range(i + 1, qbit_num)
        }
        edge_set = {frozenset(edge) for edge in edges}
        if edge_set == all_edges:
            return None
        return edges

    @staticmethod
    def _topology_edges_from_data(data):
        """Return directed topology edges from BQSKit pass data."""
        if data is None or getattr(data, "model", None) is None:
            return None
        topo_edges = set()
        for u, v in data.model.coupling_graph:
            topo_edges.add((int(u), int(v)))
            topo_edges.add((int(v), int(u)))
        return topo_edges

    async def synthesize(self, target, data=None):
        from qiskit import qasm2
        from squander import Qiskit_IO
        from bqskit.ir.lang.qasm2 import OPENQASM2Language
        from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix

        target_matrix = np.asarray(target)
        qbit_num = target.num_qudits
        mini_topology = self._data_topology(data, qbit_num)

        config = {
            **self.config,
            "topology": mini_topology,
        }

        candidates = qgd_Wide_Circuit_Optimization.DecomposePartition(
            target_matrix,
            config,
            mini_topology=mini_topology,
        )
        if len(candidates) == 0:
            tolerance = config.get("tolerance", _default_squander_tolerance(config))
            raise _SquanderSynthesisFailed(
                f"Squander synthesis failed for {qbit_num}-qubit block "
                f"at tolerance {tolerance}."
            )

        optimized_circuit, optimized_parameters = (
            qgd_Wide_Circuit_Optimization.CompareAndPickCircuits(
                [candidate[0] for candidate in candidates],
                [candidate[1] for candidate in candidates],
            )
        )

        optimized_qiskit = Qiskit_IO.get_Qiskit_Circuit(
            optimized_circuit.get_Flat_Circuit(),
            np.asarray(optimized_parameters, dtype=np.float64),
        )
        synthesized = OPENQASM2Language().decode(qasm2.dumps(optimized_qiskit))

        # The QASM round-trip preserves qubit labels but changes the physical
        # interpretation (Squander MSB=0 → BQSKit LSB=0).  Renumber qudits to
        # compensate: Squander qubit k (MSB=0) → BQSKit qubit (qbit_num-1-k).
        if qbit_num > 1:
            synthesized.renumber_qudits(
                [qbit_num - 1 - i for i in range(qbit_num)]
            )

        topo_edges = self._topology_edges_from_data(data)
        if topo_edges is not None:
            _assert_circuit_respects_topology(synthesized, topo_edges)

        if self.config.get("bqskit_distance_test", False):
            target_unitary = UnitaryMatrix(target)
            distance = target_unitary.get_distance_from(synthesized.get_unitary())
            tol = _bqskit_synthesis_validation_tolerance(self.config)
            if distance > tol:
                raise _SquanderSynthesisFailed(
                    f"BQSKit synthesis validation failed: {distance:.2e} > {tol:.2e}"
                )

        return synthesized


class _SquanderSynthesisFailed(Exception):
    """Raised when Squander cannot synthesize a partition block."""


def _topo_perm_to_swaps(pi, topo_edges, width):
    """Decompose permutation *pi* into SWAPs using only edges in *topo_edges*.

    Uses BFS on the topology graph to find a SWAP sequence that implements
    the permutation.  Returns a list of (u, v) pairs valid in *topo_edges*.
    """
    # Build adjacency list from topo_edges (undirected)
    adj = {i: set() for i in range(width)}
    for u, v in topo_edges:
        adj[u].add(v)
        adj[v].add(u)

    # Greedy: for each position i, bring the target qubit pi[i] to position i
    # by routing through the topology graph.
    current = list(range(width))  # current[pos] = which qubit is at pos
    swaps = []
    for i in range(width):
        target = pi[i]
        if current[i] == target:
            continue
        # Find where target currently is
        target_pos = current.index(target)
        # BFS from target_pos to i, finding shortest path of SWAPs
        from collections import deque
        parent = {target_pos: None}
        q = deque([target_pos])
        while q:
            u = q.popleft()
            if u == i:
                break
            for v in adj[u]:
                if v not in parent:
                    parent[v] = u
                    q.append(v)
        # Reconstruct path and apply SWAPs
        if i not in parent:
            raise _SquanderSynthesisFailed(
                f"Cannot realize permutation {pi} on disconnected topology "
                f"{sorted(topo_edges)}."
            )
        path = []
        v = i
        while parent[v] is not None:
            path.append(v)
            v = parent[v]
        path.append(target_pos)
        # Apply SWAPs along the path (reverse order to bring target to i)
        for k in range(len(path) - 1, 0, -1):
            a, b = path[k], path[k - 1]
            swaps.append((a, b))
            # Update current positions
            current[a], current[b] = current[b], current[a]
    return swaps


@contextlib.contextmanager
def patched_seqpam_workflow_classes(bqskit_compile_module, use_squander_partitioner, config):
    """Patch BQSKit workflow factories to use Squander passes.

    Replaces QSearch/LEAP with ``SquanderSynthesisPass`` only when the selected
    decomposition strategy is Squander-native. External strategies such as
    ``bqskit`` and ``qiskit`` keep BQSKit's synthesis passes; otherwise they
    would be forwarded to Squander's ``DecomposePartition`` and fail as
    unsupported. Squander failures are caught by the EAPP patch and replaced
    with SWAP-correct fallbacks.
    """

    global _SQUANDER_BQSKIT_SYNTHESIS_CONFIG

    import os as _os, json as _json

    original_quick = bqskit_compile_module.QuickPartitioner
    original_qsearch = bqskit_compile_module.QSearchSynthesisPass
    original_leap = bqskit_compile_module.LEAPSynthesisPass
    original_config = _SQUANDER_BQSKIT_SYNTHESIS_CONFIG
    original_config_env = _os.environ.get('_SQUANDER_BQSKIT_CONFIG')
    try:
        cfg = _copy_bqskit_synthesis_config(config)
        _SQUANDER_BQSKIT_SYNTHESIS_CONFIG = cfg
        # Also store in env var so worker processes (Popen) inherit it
        _os.environ['_SQUANDER_BQSKIT_CONFIG'] = _json.dumps(cfg)
        if use_squander_partitioner:
            bqskit_compile_module.QuickPartitioner = SquanderPartitioner
        if config.get("strategy") in _SQUANDER_NATIVE_STRATEGIES:
            bqskit_compile_module.QSearchSynthesisPass = SquanderSynthesisPass
            bqskit_compile_module.LEAPSynthesisPass = SquanderSynthesisPass
        yield
    finally:
        bqskit_compile_module.QuickPartitioner = original_quick
        bqskit_compile_module.QSearchSynthesisPass = original_qsearch
        bqskit_compile_module.LEAPSynthesisPass = original_leap
        _SQUANDER_BQSKIT_SYNTHESIS_CONFIG = original_config
        if original_config_env is None:
            _os.environ.pop('_SQUANDER_BQSKIT_CONFIG', None)
        else:
            _os.environ['_SQUANDER_BQSKIT_CONFIG'] = original_config_env


def extract_subtopology(involved_qbits, qbit_map, config):
    """Return topology edges restricted to ``involved_qbits``, with indices remapped via ``qbit_map``.

    Args:
        involved_qbits: Qubit labels present in a partition.
        qbit_map: Maps original qubit index to local index (0..n-1).
        config: Configuration dict containing ``topology`` as a list of edges.

    Returns:
        List of ``(u, v)`` pairs in local indices, each edge fully inside the partition.
    """
    mini_topology = []
    for edge in config["topology"]:
        if edge[0] in involved_qbits and edge[1] in involved_qbits:
            mini_topology.append((qbit_map[edge[0]], qbit_map[edge[1]]))
    return mini_topology


# Universal gate decomposition dictionary.
# Each gate maps to its exact breakdown into {CNOT, H, RX, RY, RZ, ...} basis
# as defined by circuit_to_CNOT_basis in squander/utils.py.
# Native single-qubit gates and CNOT map to themselves with count 1.
_GATE_DECOMPOSITION = {
    # --- native gates (do not decompose) ---
    "CNOT":  {"CNOT": 1},
    "H":     {"H": 1},
    "X":     {"X": 1},
    "Y":     {"Y": 1},
    "Z":     {"Z": 1},
    "S":     {"S": 1},
    "Sdg":   {"Sdg": 1},
    "T":     {"T": 1},
    "Tdg":   {"Tdg": 1},
    "SX":    {"SX": 1},
    "SXdg":  {"SXdg": 1},
    "RX":    {"RX": 1},
    "RY":    {"RY": 1},
    "RZ":    {"RZ": 1},
    "R":     {"R": 1},
    "U1":    {"U1": 1},
    "U2":    {"U2": 1},
    "U3":    {"U3": 1},
    # --- decomposed gates (counts from circuit_to_CNOT_basis) ---
    "CH":    {"CNOT": 1, "RY": 2},                                    # RY + CNOT + RY
    "CZ":    {"CNOT": 1, "H": 2},                                      # H + CNOT + H
    "SYC":   {"CNOT": 3, "U1": 3},                                     # U1 + U1 + CNOT + U1 + CNOT + CNOT
    "CRY":   {"CNOT": 2, "RY": 2},                                     # CNOT + RY + CNOT + RY
    "CU":    {"CNOT": 2, "U1": 1, "RZ": 3, "RY": 2},                  # U1 + RZ + RY + CNOT + RY + RZ + CNOT + RZ
    "CR":    {"CNOT": 2, "RZ": 2, "RY": 2},                            # RZ + CNOT + RY + CNOT + RY + RZ
    "CROT":  {"CNOT": 2, "RZ": 3, "RY": 2},                            # RZ + RY + CNOT + RZ + CNOT + RY + RZ
    "CRX":   {"CNOT": 2, "H": 2, "RZ": 2},                             # H + CNOT + RZ + CNOT + RZ + H
    "CRZ":   {"CNOT": 2, "RZ": 2},                                     # CNOT + RZ + CNOT + RZ
    "CP":    {"CNOT": 2, "U1": 3},                                     # U1 + CNOT + U1 + CNOT + U1
    "CCX":   {"CNOT": 6, "H": 2, "T": 4, "Tdg": 3},                   # standard Toffoli: 7 CNOTs + 8 single-qubit
    "CSWAP": {"CNOT": 7, "H": 1, "T": 5, "Tdg": 2, "SX": 1, "Sdg": 1, "S": 1},  # Fredkin
    "SWAP":  {"CNOT": 3},                                              # CNOT + CNOT + CNOT
    "RXX":   {"CNOT": 2, "RX": 1},                                     # CNOT + RX + CNOT
    "RYY":   {"CNOT": 2, "RX": 4, "RZ": 1},                            # RX + RX + CNOT + RZ + CNOT + RX + RX
    "RZZ":   {"CNOT": 2, "RZ": 1},                                     # CNOT + RZ + CNOT
}

# Backward-compatible: CNOT-equivalent cost (number of CNOTs in decomposition).
CNOT_COUNT_DICT = {g: d.get("CNOT", 0) for g, d in _GATE_DECOMPOSITION.items()}


def CNOTGateCount(circ: Circuit, max_gates: int = 0) -> int:
    """Compute weighted two-qubit gate count for a circuit.

    The base count is the CNOT-equivalent cost derived from ``CNOT_COUNT_DICT``.
    When ``max_gates > 0``, the function returns a lexicographic-style score:
    ``two_qubit_cost * max_gates + single_qubit_gate_count``.

    Args:
        circ: Squander circuit representation.
        max_gates: Weight multiplier for the two-qubit cost term.

    Returns:
        Integer gate-cost score used by optimization heuristics.
    """
    assert isinstance(circ, Circuit), \
        "The input parameters should be an instance of Squander Circuit"
    gate_counts = circ.get_Gate_Nums()
    num_cnots = sum(
        CNOT_COUNT_DICT.get(gate, 0) * count for gate, count in gate_counts.items()
    )
    if max_gates > 0:
        return num_cnots * max_gates + sum(
            y for x, y in gate_counts.items() if CNOT_COUNT_DICT.get(x, -1) <= 0
        )
    return num_cnots


def SingleQubitGateCount(circ: Circuit) -> int:
    """Count single-qubit gates in a circuit (U3, H, RX, RY, RZ, etc.).

    Uses _GATE_DECOMPOSITION to count non-CNOT gates in each gate's breakdown.

    Args:
        circ: Squander circuit representation.

    Returns:
        Total number of single-qubit gate operations when fully decomposed.
    """
    gate_counts = circ.get_Gate_Nums()
    total = 0
    for gate, count in gate_counts.items():
        decomp = _GATE_DECOMPOSITION.get(gate, {})
        total += count * sum(v for k, v in decomp.items() if k != "CNOT")
    return total


def TotalRawGateCount(circ: Circuit) -> int:
    """Total number of raw gate operations (single-qubit + multi-qubit).

    Args:
        circ: Squander circuit representation.

    Returns:
        Total gate operation count.
    """
    return sum(circ.get_Gate_Nums().values())


def CircuitGateStats(circ: Circuit) -> dict:
    """Return comprehensive gate statistics for a circuit.

    Uses _GATE_DECOMPOSITION to compute fully-decomposed gate counts.

    Returns dict with keys: cnot_equiv, single_qubit, total_raw, qubits,
    and gate_breakdown (per-gate-type raw counts).
    """
    gate_counts = circ.get_Gate_Nums()
    cnot_equiv = sum(
        CNOT_COUNT_DICT.get(g, 0) * c for g, c in gate_counts.items()
    )
    single = 0
    for g, c in gate_counts.items():
        decomp = _GATE_DECOMPOSITION.get(g, {})
        single += c * sum(v for k, v in decomp.items() if k != "CNOT")
    total = sum(gate_counts.values())
    return {
        "cnot_equiv": cnot_equiv,
        "single_qubit": single,
        "total_raw": total,
        "qubits": circ.get_Qbit_Num(),
        "gate_breakdown": dict(gate_counts),
    }


class qgd_Wide_Circuit_Optimization:
    """Optimize wide (many-qubit) circuits via partitioning and subcircuit decomposition.

    Supports multiple decomposition strategies, optional global recombination (ILP),
    and routing when the circuit does not match the target topology.
    """

    def __init__(self, config):
        """Validate and store wide-circuit optimization ``config`` (strategy, topology, partitioning, tolerances)."""

        config.setdefault("strategy", "TreeSearch")
        config.setdefault("parallel", 0)
        config.setdefault("verbosity", 0)
        config.setdefault("use_float", False)
        config.setdefault("tolerance", _default_squander_tolerance(config))
        config.setdefault(
            "circuit_validation_tolerance",
            _default_circuit_validation_tolerance(config),
        )
        config.setdefault(
            "bqskit_synthesis_validation_tolerance",
            _default_bqskit_synthesis_validation_tolerance(config),
        )
        config.setdefault("test_subcircuits", False)
        config.setdefault("test_final_circuit", True)
        config.setdefault("max_partition_size", 3)
        config.setdefault("topology", None)
        config.setdefault("partition_strategy", "ilp")
        config.setdefault("auto_expand_partition_size", True)
        config.setdefault("force_small_circuit_validation", True)

        # testing the fields of config
        strategy = config["strategy"]
        allowed_startegies = [
            "TreeSearch",
            "TabuSearch",
            "Adaptive",
            "qiskit",
            "bqskit",
        ]
        if not strategy in allowed_startegies:
            raise Exception(
                f"The decomposition startegy should be either of {allowed_startegies}, got {strategy}."
            )

        parallel = config["parallel"]
        allowed_parallel = [0, 1, 2]
        if not parallel in allowed_parallel:
            raise Exception(
                f"The parallel configuration should be either of {allowed_parallel}, got {parallel}."
            )

        verbosity = config["verbosity"]
        if not isinstance(verbosity, int):
            raise Exception(f"The verbosity parameter should be an integer.")

        tolerance = config["tolerance"]
        if not isinstance(tolerance, float):
            raise Exception(f"The tolerance parameter should be a float.")

        use_float = config["use_float"]
        if not isinstance(use_float, bool):
            raise Exception(f"The use_float parameter should be a bool.")

        bqskit_synthesis_validation_tolerance = config[
            "bqskit_synthesis_validation_tolerance"
        ]
        if not isinstance(bqskit_synthesis_validation_tolerance, float):
            raise Exception(
                "The bqskit_synthesis_validation_tolerance parameter should be a float."
            )

        circuit_validation_tolerance = config["circuit_validation_tolerance"]
        if not isinstance(circuit_validation_tolerance, float):
            raise Exception(
                "The circuit_validation_tolerance parameter should be a float."
            )

        test_subcircuits = config["test_subcircuits"]
        if not isinstance(test_subcircuits, bool):
            raise Exception(f"The test_subcircuits parameter should be a bool.")

        test_final_circuit = config["test_final_circuit"]
        if not isinstance(test_final_circuit, bool):
            raise Exception(f"The test_final_circuit parameter should be a bool.")

        max_partition_size = config["max_partition_size"]
        if not isinstance(max_partition_size, int):
            raise Exception(f"The max_partition_size parameter should be an integer.")

        self.config = config

        self.max_partition_size = max_partition_size

    @staticmethod
    def partition_tree_level_max(config, subcircuit, reduction=1):
        """Return the tree-search depth used for partition-local rewrites."""

        target_depth = max(0, CNOTGateCount(subcircuit, 0) - reduction)
        configured_limit = config.get("partition_tree_level_max", None)
        if configured_limit is None:
            configured_limit = target_depth
        return min(target_depth, int(configured_limit))

    def ConstructCircuitFromPartitions(
        self, circs: List[Circuit], parameter_arrs: List[List[np.ndarray]]
    ) -> Tuple[Circuit, np.ndarray]:
        """Concatenate optimized partition circuits into a single wide circuit.

        Args:
            circs: Partition circuits in execution order.
            parameter_arrs: Parameter arrays corresponding to ``circs``.

        Returns:
            Tuple of ``(wide_circuit, wide_parameters)``.
        """

        if not isinstance(circs, list):
            raise Exception("First argument should be a list of squander circuits")

        if not isinstance(parameter_arrs, list):
            raise Exception("Second argument should be a list of numpy arrays")

        if len(circs) != len(parameter_arrs):
            raise Exception("The first two arguments should be of the same length")

        qbit_num = circs[0].get_Qbit_Num()

        wide_parameters = np.concatenate(parameter_arrs, axis=0)

        wide_circuit = Circuit(qbit_num)

        for circ in circs:
            wide_circuit.add_Circuit(circ)

        assert (
            wide_circuit.get_Parameter_Num() == wide_parameters.size
        ), f"Mismatch in the number of parameters: {wide_circuit.get_Parameter_Num()} vs {wide_parameters.size}"

        return wide_circuit, wide_parameters

    @staticmethod
    def DecomposePartition(
        Umtx: np.ndarray, config: dict, mini_topology=None, structure=None
    ) -> list[tuple[Circuit, np.ndarray]]:
        """Decompose a unitary ``Umtx`` (e.g. from a partition) using ``config['strategy']``.

        Args:
            Umtx: Complex unitary matrix.
            config: Must include ``strategy``, ``tolerance``, ``verbosity``, etc.
            mini_topology: Optional hardware couplers for topology-aware decomposers.
            structure: Required gate structure when ``strategy == "Custom"``.

        Returns:
            Normally ``[(circuit, parameters)]`` on success, or ``[]`` if the
            decomposition error exceeds ``tolerance``. If
            ``config.get('stop_first_solution')`` is false, returns
            ``cDecompose.all_solutions`` from the underlying decomposer instead of
            a single best pair.
        """
        strategy = config["strategy"]
        if strategy == "TreeSearch":
            cDecompose = N_Qubit_Decomposition_Tree_Search(
                Umtx.conj().T, config=config, accelerator_num=0, topology=mini_topology
            )
        elif strategy == "TabuSearch":
            cDecompose = N_Qubit_Decomposition_Tabu_Search(
                Umtx.conj().T, config=config, accelerator_num=0, topology=mini_topology
            )
        elif strategy == "Adaptive":
            cDecompose = N_Qubit_Decomposition_adaptive(
                Umtx.conj().T,
                level_limit_max=5,
                level_limit_min=1,
                topology=mini_topology,
            )
        elif strategy == "Custom":
            cDecompose = N_Qubit_Decomposition_custom(
                Umtx.conj().T, config=config, accelerator_num=0
            )
            assert (
                structure is not None
            ), "Custom decomposition strategy requires a gate structure to be provided."
            cDecompose.set_Gate_Structure(structure)
        else:
            raise Exception(f"Unsupported decomposition type: {strategy}")

        tolerance = config["tolerance"]
        cDecompose.set_Verbose(config["verbosity"])
        cDecompose.set_Cost_Function_Variant(3)
        cDecompose.set_Optimization_Tolerance(tolerance)

        # adding new layer to the decomposition until threshold
        cDecompose.set_Optimizer("BFGS")

        # starting the decomposition
        try:
            cDecompose.Start_Decomposition()
        except Exception as e:
            # print(e)
            raise e
            # return []
        if not config.get("stop_first_solution", True):
            return cDecompose.all_solutions

        squander_circuit = cDecompose.get_Circuit()
        parameters = cDecompose.get_Optimized_Parameters()
        assert parameters is not None

        if strategy == "Custom":
            err = cDecompose.Optimization_Problem(parameters)
            it = 0
            while err > tolerance and it < 20:
                cDecompose.set_Optimized_Parameters(
                    np.random.rand(cDecompose.get_Parameter_Num()) * (2 * np.pi)
                )
                cDecompose.Start_Decomposition()
                parameters = cDecompose.get_Optimized_Parameters()
                err = cDecompose.Optimization_Problem(parameters)
                it += 1
            if err > tolerance or it != 0:
                print("Decomposition error: ", err, it)
        else:
            err = cDecompose.get_Decomposition_Error()
        # print( "Decomposition error: ", err )
        if tolerance < err:
            # raise Exception(f"Decomposition error {err} exceeds the tolerance {tolerance}.")
            return []

        return [(squander_circuit, parameters)]

    @staticmethod
    def CompareAndPickCircuits(
        circs: List[Circuit],
        parameter_arrs: List[np.ndarray],
        metric: Callable[[Circuit], int] = CNOTGateCount,
    ) -> tuple[Circuit, np.ndarray]:
        """Select the circuit with the lowest ``metric`` value.

        Args:
            circs: Candidate Squander circuits (same length as ``parameter_arrs``).
            parameter_arrs: Parameter vectors aligned with ``circs``.
            metric: Scalar cost functional; lower is better. Defaults to ``CNOTGateCount``.

        Returns:
            ``(best_circuit, best_parameters)`` for the minimizing index.
        """

        if not isinstance(circs, list):
            raise Exception("First argument should be a list of squander circuits")

        if not isinstance(parameter_arrs, list):
            raise Exception("Second argument should be a list of numpy arrays")

        if len(circs) != len(parameter_arrs):
            raise Exception("The first two arguments should be of the same length")

        metrics = [metric(circ) for circ in circs]

        metrics = np.array(metrics)

        min_idx = np.argmin(metrics)

        return circs[min_idx], parameter_arrs[min_idx]

    @staticmethod
    def PartitionDecompositionProcess(
        subcircuit: Circuit,
        subcircuit_parameters: np.ndarray,
        config: dict,
        structure=None,
    ) -> Tuple[Circuit, np.ndarray]:
        """Decompose one partition subcircuit (multiprocessing-safe entry point).

        Args:
            subcircuit: Subcircuit acting on a subset of the wide register.
            subcircuit_parameters: Flat parameter vector slice for ``subcircuit``.
            config: Same keys as wide optimization (``strategy``, ``topology``, etc.).
            structure: Optional fixed gate structure when ``strategy == "Custom"``.

        Returns:
            Tuple of ``(decomposed_circuit, decomposed_parameters)`` pairs, each
            remapped back to the original qubit indices of ``subcircuit``.
        """

        qbit_num_orig_circuit = subcircuit.get_Qbit_Num()

        involved_qbits = subcircuit.get_Qbits()

        qbit_num = len(involved_qbits)

        # create qbit map:
        qbit_map = {}
        for idx in range(len(involved_qbits)):
            qbit_map[involved_qbits[idx]] = idx
        mini_topology = None
        if config["topology"] is not None:
            mini_topology = extract_subtopology(involved_qbits, qbit_map, config)
        # remap the subcircuit to a smaller qubit register
        remapped_subcircuit = subcircuit.Remap_Qbits(qbit_map, qbit_num)

        if not structure is None:
            structure = structure.Remap_Qbits(qbit_map, qbit_num)

        # get the unitary representing the circuit
        unitary = remapped_subcircuit.get_Matrix(
            np.asarray(subcircuit_parameters, dtype=np.float64)
        )

        # decompose a small unitary into a new circuit
        all_decomposed = qgd_Wide_Circuit_Optimization.DecomposePartition(
            unitary, config, mini_topology, structure=structure
        )
        # create inverse qbit map:
        inverse_qbit_map = {}
        for key, value in qbit_map.items():
            inverse_qbit_map[value] = key
        result = []
        for decomposed_circuit, decomposed_parameters in all_decomposed:

            # remap the decomposed circuit in order to insert it into a large circuit
            new_subcircuit = decomposed_circuit.Remap_Qbits(
                inverse_qbit_map, qbit_num_orig_circuit
            )

            if config["test_subcircuits"]:
                CompareCircuits(
                    subcircuit,
                    subcircuit_parameters,
                    new_subcircuit,
                    decomposed_parameters,
                    parallel=config["parallel"],
                    tolerance=_squander_validation_tolerance(config)
                )

            new_subcircuit = new_subcircuit.get_Flat_Circuit()
            result.append((new_subcircuit, decomposed_parameters))
        return tuple(result)

    @staticmethod
    def build_partition_topo_deps(allparts):
        """Order partition gate-sets by dependencies and build a reverse-dependency map.

        Args:
            allparts: List of sets of gate indices, one per partition.

        Returns:
            ``(ordered_parts, rg_new)`` where ``ordered_parts`` lists partitions in
            topological order and ``rg_new`` maps each new index to predecessors.
        """
        gate_to_parts = {}
        for i, part in enumerate(allparts):
            for gate in part:
                gate_to_parts.setdefault(gate, set()).add(i)
        g = {i: set() for i in range(len(allparts))}
        rg = {i: set() for i in range(len(allparts))}
        for i, part in enumerate(allparts):
            for gate in part:
                for other_part in gate_to_parts[gate]:
                    if other_part != i and (
                        len(part & allparts[other_part]) > 0
                        and (len(part) < len(allparts[other_part]))
                        or part < allparts[other_part]
                    ):
                        g[i].add(other_part)
                        rg[other_part].add(i)
        rg_ret = {i: set(rg[i]) for i in range(len(allparts))}
        S = collections.deque(m for m in rg if len(rg[m]) == 0)
        L = []
        while S:
            n = S.popleft()
            L.append(n)
            for m in set(g[n]):
                g[n].remove(m)
                rg[m].remove(n)
                if len(rg[m]) == 0:
                    S.append(m)
        if len(L) != len(allparts):
            raise ValueError("Dependency graph is not a DAG")
        neworder = {old: new for new, old in enumerate(L)}
        rg_ret = {
            neworder[i]: set(neworder[j] for j in rg_ret[i])
            for i in range(len(allparts))
        }
        return [
            allparts[i] for i in L
        ], rg_ret  # return partitions in dependency order and dependencies

    @staticmethod
    def make_all_partition_circuit(circ, orig_parameters, max_partition_size):
        """ILP-based partitioning: flatten ``circ`` into a circuit of sub-circuits with concatenated parameters.

        Returns:
            ``(partitioned_circuit, parameters, recombine_info, part_deps)`` for later fusion in
            ``recombine_all_partition_circuit``.
        """
        from squander.partitioning.ilp import get_all_partitions, _get_topo_order

        allparts, g, go, rgo, single_qubit_chains, gate_to_qubit, gate_to_tqubit = (
            get_all_partitions(circ, max_partition_size)
        )
        qbit_num_orig_circuit = circ.get_Qbit_Num()
        gate_dict = {i: gate for i, gate in enumerate(circ.get_Gates())}
        single_qubit_chains_pre = {x[0]: x for x in single_qubit_chains if rgo[x[0]]}
        single_qubit_chains_post = {x[-1]: x for x in single_qubit_chains if go[x[-1]]}
        single_qubit_chains_prepost = {
            x[0]: x
            for x in single_qubit_chains
            if x[0] in single_qubit_chains_pre and x[-1] in single_qubit_chains_post
        }
        partitioned_circuit = Circuit(qbit_num_orig_circuit)
        params = []
        allparts, part_deps = qgd_Wide_Circuit_Optimization.build_partition_topo_deps(
            allparts
        )
        for part in allparts:
            surrounded_chains = {
                t
                for s in part
                for t in go[s]
                if t in single_qubit_chains_prepost
                and go[single_qubit_chains_prepost[t][-1]]
                and next(iter(go[single_qubit_chains_prepost[t][-1]])) in part
            }
            gates = frozenset.union(
                part, *(single_qubit_chains_prepost[v] for v in surrounded_chains)
            )
            # topo sort part + surrounded chains
            c = Circuit(qbit_num_orig_circuit)
            for gate_idx in _get_topo_order(
                {x: go[x] & gates for x in gates},
                {x: rgo[x] & gates for x in gates},
                gate_to_qubit,
            ):
                c.add_Gate(gate_dict[gate_idx])
                start = gate_dict[gate_idx].get_Parameter_Start_Index()
                params.append(
                    orig_parameters[
                        start : start + gate_dict[gate_idx].get_Parameter_Num()
                    ]
                )
            partitioned_circuit.add_Circuit(c)
        for chain in single_qubit_chains:
            c = Circuit(qbit_num_orig_circuit)
            for gate_idx in chain:
                c.add_Gate(gate_dict[gate_idx])
                start = gate_dict[gate_idx].get_Parameter_Start_Index()
                params.append(
                    orig_parameters[
                        start : start + gate_dict[gate_idx].get_Parameter_Num()
                    ]
                )
            partitioned_circuit.add_Circuit(c)
        parameters = np.concatenate(params, axis=0)
        return (
            partitioned_circuit,
            parameters,
            (allparts, g, go, rgo, single_qubit_chains, gate_to_qubit, gate_to_tqubit),
            part_deps,
        )

    @staticmethod
    def strip_single_qubit_head_tails(circ, params):
        """Drop single-qubit gates that sit only at the head or tail of the dependency DAG.

        Args:
            circ: Input circuit.
            params: Flat parameter array for ``circ``.

        Returns:
            ``(new_circuit, new_params)`` with head/tail single-qubit gates removed.
        """
        gate_dict, g, rg, gate_to_qubit, _ = build_dependency(circ)
        newcirc = Circuit(circ.get_Qbit_Num())
        new_params = []
        for i in gate_dict:
            gate = gate_dict[i]
            if len(gate_to_qubit[i]) == 1 and (len(g[i]) == 0 or len(rg[i]) == 0):
                continue
            newcirc.add_Gate(gate)
            start_idx = gate.get_Parameter_Start_Index()
            new_params.append(params[start_idx : start_idx + gate.get_Parameter_Num()])
        return newcirc, (
            np.empty((0,), dtype=np.float64)
            if len(new_params) == 0
            else np.concatenate(new_params, axis=0)
        )

    @staticmethod
    def get_fingerprint(circ, params):
        """Hashable signature of gate layout and parameters (for decomposition caching).

        Args:
            circ: Squander circuit.
            params: Parameter array associated with ``circ``.

        Returns:
            Tuple usable as a dict key for memoizing decompositions.
        """
        return tuple(
            (gate.get_Name(), tuple(gate.get_Involved_Qbits()))
            for gate in circ.get_Gates()
        ) + tuple(params)

    @staticmethod
    def recombine_all_partition_circuit(
        circ, optimized_subcircuits, optimized_parameter_list, recombine_info
    ):
        """Reorder optimized partitions to respect global gate dependencies.

        Args:
            circ: Original flat circuit (for topological ordering context).
            optimized_subcircuits: One optimized subcircuit per partition slot.
            optimized_parameter_list: Parameter lists aligned with ``optimized_subcircuits``.
            recombine_info: Tuple from ``make_all_partition_circuit`` (ILP metadata).

        Returns:
            ``(reordered_circuits, reordered_parameter_lists)`` in execution order.
        """
        from squander.partitioning.ilp import (
            topo_sort_partitions,
            ilp_global_optimal,
            recombine_single_qubit_chains,
        )

        allparts, g, go, rgo, single_qubit_chains, gate_to_qubit, gate_to_tqubit = (
            recombine_info
        )
        max_gates = sum(
            sum(y for x, y in c.get_Gate_Nums().items() if CNOT_COUNT_DICT.get(x, -1) <= 0)
            for c in optimized_subcircuits[: len(allparts)]
        )
        weights = [
            CNOTGateCount(circ, max_gates)
            for circ in optimized_subcircuits[: len(allparts)]
        ]
        L, fusion_info = ilp_global_optimal(allparts, g, weights=weights)
        struct_idxs = list(L)
        parts = recombine_single_qubit_chains(
            go,
            rgo,
            single_qubit_chains,
            gate_to_tqubit,
            [allparts[i] for i in L],
            fusion_info,
            surrounded_only=True,
        )
        single_qubit_chain_idx = {
            frozenset(chain): idx + len(allparts)
            for idx, chain in enumerate(single_qubit_chains)
        }
        for extrapart in parts[len(struct_idxs) :]:
            struct_idxs.append(single_qubit_chain_idx[frozenset(extrapart)])
        L = topo_sort_partitions(circ, parts)
        return [optimized_subcircuits[struct_idxs[i]] for i in L], [
            optimized_parameter_list[struct_idxs[i]] for i in L
        ]

    def OptimizeWideCircuit(
        self, circ: Circuit, parameters: np.ndarray
    ) -> Tuple[Circuit, np.ndarray]:
        """Top-level wide-circuit pass: optional routing, then Qiskit / BQSKit / Squander partition optimization.

        Sets ``self.config`` timing and intermediate circuit keys (e.g. ``routed_circuit``, ``optimization_time``).
        """
        if not qgd_Wide_Circuit_Optimization.is_valid_routing(
            circ, self.config["topology"]
        ):

            print("fixing topology in the circuit")
            topo = self.config["topology"]
            self.config["topology"] = None
            strat = self.config["strategy"]
            self.config["strategy"] = self.config["pre-opt-strategy"]

            print("Optimizing circuit with all-to-all (a2a) connectivity")
            circ, parameters = self.OptimizeWideCircuit(circ, parameters)
            self.config["all_to_all_optimization_time"] = self.config[
                "optimization_time"
            ]
            self.config["all_to_all_circuit"] = circ
            self.config["all_to_all_parameters"] = parameters
            self.config["strategy"] = strat
            self.config["topology"] = topo
            start_time = time.time()

            print("Routing circuit to fix the topology")
            circ, parameters = self.route_circuit(circ, parameters)
            self.config["routing_time"] = time.time() - start_time
            self.config["routed_circuit"] = circ
            self.config["routed_parameters"] = parameters
        else:
            if self.config["topology"] is not None:
                print("No additional routing is needed on the circuit")

        start_time = time.time()
        if self.config["strategy"] == "bqskit":
            print("Optimizing circuit with BQSkit")
            from squander import Qiskit_IO
            from bqskit import compile

            from bqskit.compiler.machine import MachineModel
            from bqskit.compiler import Compiler
            from bqskit.ir.lang.qasm2 import OPENQASM2Language
            from qiskit import qasm2, QuantumCircuit

            from bqskit.passes import SetModelPass
            from bqskit.compiler.compile import (
                build_multi_qudit_retarget_workflow,
                build_resynthesis_optimization_workflow,
                build_single_qudit_retarget_workflow,
                build_gate_deletion_optimization_workflow,
                LogErrorPass,
            )

            # Build BQSKit machine model from your topology
            model = MachineModel(circ.get_Qbit_Num(), self.config["topology"])

            # Convert squander circuit → qiskit → BQSKit
            # (BQSKit has a from_qiskit helper if you go via Qiskit IR)
            circo = Qiskit_IO.get_Qiskit_Circuit(
                circ, np.asarray(parameters, dtype=np.float64)
            )

            bqskit_circ = OPENQASM2Language().decode(qasm2.dumps(circo))

            compilation_workflow = [
                SetModelPass(model),  # attach hardware model to circuit
                build_multi_qudit_retarget_workflow(
                    4, max_synthesis_size=self.max_partition_size
                ),
                build_resynthesis_optimization_workflow(
                    4, max_synthesis_size=self.max_partition_size, iterative=True
                ),
                build_single_qudit_retarget_workflow(
                    4, max_synthesis_size=self.max_partition_size
                ),
                build_gate_deletion_optimization_workflow(
                    4, max_synthesis_size=self.max_partition_size, iterative=True
                ),
                LogErrorPass(),
            ]

            with Compiler() as compiler:
                routed_bqskit_circ, pass_data = compiler.compile(
                    bqskit_circ, compilation_workflow, True
                )

                default = list(range(bqskit_circ.num_qudits))
                initial_map = pass_data.get("initial_mapping", default)
                final_map = pass_data.get("final_mapping", default)

            # Convert back: BQSKit → Qiskit → Squander
            circuit_qiskit = QuantumCircuit.from_qasm_str(
                OPENQASM2Language().encode(routed_bqskit_circ)
            )
            newcirc, newparameters = Qiskit_IO.convert_Qiskit_to_Squander(
                circuit_qiskit
            )

            qgd_Wide_Circuit_Optimization.check_valid_routing(
                newcirc, self.config["topology"]
            )
            print("OptimizeWideCircuit::check_compare_circuits")
            self.check_compare_circuits(circ, parameters, newcirc, newparameters)
            circ, parameters = newcirc, newparameters

        elif self.config["strategy"] == "qiskit":
            print("Optimizing circuit with Qiskit")
            from squander import Qiskit_IO
            from qiskit import transpile
            from qiskit.transpiler import CouplingMap
            from squander.gates import gates_Wrapper as gate

            SUPPORTED_GATES_NAMES = {
                n.lower().replace("cnot", "cx")
                for n in dir(gate)
                if not n.startswith("_")
                and issubclass(getattr(gate, n), gate.Gate)
                and n not in ("Gate", "CROT", "CR", "SYC", "CCX", "CSWAP")
            }
            circo = Qiskit_IO.get_Qiskit_Circuit(
                circ, np.asarray(parameters, dtype=np.float64)
            )
            coupling_map = (
                None
                if self.config["topology"] is None
                else CouplingMap([[i, j] for i, j in self.config["topology"]])
            )
            circuit_qiskit = transpile(
                circo,
                basis_gates=SUPPORTED_GATES_NAMES,
                coupling_map=coupling_map,
                optimization_level=3,
            )
            newcirc, newparameters = Qiskit_IO.convert_Qiskit_to_Squander(
                circuit_qiskit
            )
            qgd_Wide_Circuit_Optimization.check_valid_routing(
                newcirc, self.config["topology"]
            )
            print("OptimizeWideCircuit::check_compare_circuits")
            self.check_compare_circuits(circ, parameters, newcirc, newparameters)
            circ, parameters = newcirc, newparameters
        else:

            print("Optimizing circuit with Squander")
            part_size_start = self.max_partition_size
            part_size_end = self.max_partition_size
            if self.config.get("auto_expand_partition_size", True) and (
                self.config.get("use_osr", False)
                or self.config.get("use_graph_search", False)
            ):
                part_size_end = min(4, circ.get_Qbit_Num())
            count = CNOTGateCount(circ, 0)
            fingerprint_dict = {}
            for max_part_size in range(part_size_start, part_size_end + 1):
                # instantiate the object for optimizing wide circuits
                wide_circuit_optimizer = qgd_Wide_Circuit_Optimization(
                    {**self.config, "max_partition_size": max_part_size}
                )
                while True:
                    # run circuit optimization
                    circ_flat, parameters = (
                        wide_circuit_optimizer.InnerOptimizeWideCircuit(
                            circ, parameters, fingerprint_dict=fingerprint_dict
                        )
                    )
                    circ = circ_flat.get_Flat_Circuit()
                    newcount = CNOTGateCount(circ, 0)
                    no_improve = newcount >= count
                    count = newcount
                    if no_improve:
                        break
        self.config["optimization_time"] = time.time() - start_time
        return circ, parameters

    def InnerOptimizeWideCircuit(
        self, circ: Circuit, orig_parameters: np.ndarray, fingerprint_dict=None
    ) -> Tuple[Circuit, np.ndarray]:
        """Optimize one pass of wide-circuit partition decomposition.

        The circuit is converted to a CNOT basis, partitioned, each partition is
        optimized (possibly in parallel), and then reconstructed into one circuit.

        Args:
            circ: Input circuit to optimize.
            orig_parameters: Parameter array associated with ``circ``.
            fingerprint_dict: Optional decomposition cache shared across passes.

        Returns:
            Tuple of ``(optimized_circuit, optimized_parameters)``.
        """
        from squander.utils import circuit_to_CNOT_basis

        circ, orig_parameters = circuit_to_CNOT_basis(circ, orig_parameters)
        max_gates = sum(
            y for x, y in circ.get_Gate_Nums().items() if CNOT_COUNT_DICT.get(x, -1) <= 0
        )

        global_min = self.config.get("global_min", True)
        if global_min:
            partitioned_circuit, parameters, recombine_info, part_deps = (
                qgd_Wide_Circuit_Optimization.make_all_partition_circuit(
                    circ, orig_parameters, self.max_partition_size
                )
            )

        else:
            partitioned_circuit, parameters, _ = PartitionCircuit(
                circ,
                orig_parameters,
                self.max_partition_size,
                strategy=self.config["partition_strategy"],
            )
            part_deps = None

        subcircuits = partitioned_circuit.get_Gates()

        # subcircuits = subcircuits[9:10]

        in_parent = parent_process() is not None

        if not in_parent:
            print(len(subcircuits), "partitions found to optimize")

        # the list of optimized subcircuits
        optimized_subcircuits: List[Optional[Circuit]] = [None] * len(subcircuits)

        # the list of parameters associated with the optimized subcircuits
        optimized_parameter_list: List[Optional[List[np.ndarray]]] = [None] * len(
            subcircuits
        )

        # list of AsyncResult objects
        async_results = [None] * len(subcircuits)

        total_opt = [0]

        def process_result(partition_idx):
            """Finalize async decomposition for partition ``partition_idx`` and update caches / lists."""
            if optimized_subcircuits[partition_idx] is not None:
                return
            subcircuit = subcircuits[partition_idx]
            # callback on the master process to compare the decomposed and original subcircuit
            start_idx = subcircuit.get_Parameter_Start_Index()
            subcircuit_parameters = parameters[
                start_idx : start_idx + subcircuit.get_Parameter_Num()
            ]
            fingerprint = (
                None
                if fingerprint_dict is None
                else qgd_Wide_Circuit_Optimization.get_fingerprint(
                    subcircuit, subcircuit_parameters
                )
            )
            callback_fnc = lambda x: self.CompareAndPickCircuits(
                [subcircuit, *(z[0] for z in x)],
                [subcircuit_parameters, *(z[1] for z in x)],
                lambda c: CNOTGateCount(c, max_gates),
            )
            if fingerprint_dict is not None and fingerprint in fingerprint_dict:
                new_subcircuit, new_parameters = fingerprint_dict[fingerprint]
            else:
                new_subcircuit, new_parameters = callback_fnc(
                    async_results[partition_idx][0](*async_results[partition_idx][1])
                    if in_parent
                    else async_results[partition_idx].get(timeout=None)
                )

                if subcircuit != new_subcircuit:
                    print(
                        "original subcircuit:    ",
                        subcircuit.get_Gate_Nums(),
                        partition_idx,
                    )
                    print("reoptimized subcircuit: ", new_subcircuit.get_Gate_Nums())
                if fingerprint_dict is not None:
                    fingerprint_dict[fingerprint] = (new_subcircuit, new_parameters)
                    fingerprint_dict[
                        qgd_Wide_Circuit_Optimization.get_fingerprint(
                            new_subcircuit, new_parameters
                        )
                    ] = (new_subcircuit, new_parameters)
                    trim_subcirc, trim_parameters = (
                        qgd_Wide_Circuit_Optimization.strip_single_qubit_head_tails(
                            new_subcircuit, new_parameters
                        )
                    )
                    fingerprint_dict[
                        qgd_Wide_Circuit_Optimization.get_fingerprint(
                            trim_subcirc, trim_parameters
                        )
                    ] = (trim_subcirc, trim_parameters)
            if total_opt[0] % 100 == 99:
                print(total_opt[0] + 1, "partitions optimized")
            total_opt[0] += 1
            optimized_subcircuits[partition_idx] = new_subcircuit
            optimized_parameter_list[partition_idx] = new_parameters

        with (
            contextlib.nullcontext() if in_parent else Pool(processes=mp.cpu_count())
        ) as pool:
            remaining = list(range(len(subcircuits)))
            while remaining:
                still_remaining = []
                #  code for iterate over partitions and optimize them
                for partition_idx in remaining:
                    subcircuit = subcircuits[partition_idx]

                    # isolate the parameters corresponding to the given sub-circuit
                    start_idx = subcircuit.get_Parameter_Start_Index()
                    end_idx = start_idx + subcircuit.get_Parameter_Num()
                    subcircuit_parameters = parameters[start_idx:end_idx]

                    fingerprint = (
                        None
                        if fingerprint_dict is None
                        else qgd_Wide_Circuit_Optimization.get_fingerprint(
                            subcircuit, subcircuit_parameters
                        )
                    )
                    if fingerprint_dict is not None and fingerprint in fingerprint_dict:
                        (
                            optimized_subcircuits[partition_idx],
                            optimized_parameter_list[partition_idx],
                        ) = fingerprint_dict[fingerprint]
                        continue
                    if part_deps is not None and partition_idx in part_deps:
                        any_optimized, any_remaining = False, False
                        for dep_idx in part_deps[partition_idx]:
                            if optimized_subcircuits[dep_idx] is None and (
                                async_results[dep_idx] is None
                                or not isinstance(async_results[dep_idx], tuple)
                                and not async_results[dep_idx].ready()
                            ):
                                any_remaining = True
                                continue
                            elif optimized_subcircuits[dep_idx] is None:
                                process_result(dep_idx)

                            optimized_subcircuits_loc = optimized_subcircuits[dep_idx]
                            assert isinstance(optimized_subcircuits_loc, Circuit)
                            assert optimized_subcircuits_loc is not None

                            if CNOTGateCount(optimized_subcircuits_loc) < CNOTGateCount(
                                subcircuits[dep_idx]
                            ):  # if the dependency partition was optimized, skip
                                any_optimized = True
                                break
                        if any_optimized:
                            optimized_subcircuits[partition_idx] = subcircuit
                            optimized_parameter_list[partition_idx] = (
                                subcircuit_parameters
                            )
                            continue
                        if any_remaining:
                            still_remaining.append(partition_idx)
                            continue
                    # call a process to decompose a subcircuit
                    config = {
                        **self.config,
                        "tree_level_max": qgd_Wide_Circuit_Optimization.partition_tree_level_max(
                            self.config, subcircuit
                        ),
                    }
                    fargs = (
                        self.PartitionDecompositionProcess,
                        (subcircuit, subcircuit_parameters, config, None),
                    )
                    # print("Dispatching", subcircuit.get_Involved_Qubits(), "qubits with", CNOGateCount(subcircuit, 0), "CNOT gates, partition ", partition_idx)
                    async_results[partition_idx] = (
                        fargs if in_parent else pool.apply_async(*fargs)  # type: ignore[union-attr]
                    )
                if len(remaining) == len(still_remaining):
                    time.sleep(0.1)
                remaining = still_remaining
            #  code for iterate over async results and retrieve the new subcircuits
            for partition_idx in range(len(subcircuits)):
                process_result(partition_idx)

        # construct the wide circuit from the optimized subcircuits
        if global_min:
            optimized_subcircuits, optimized_parameter_list = (
                qgd_Wide_Circuit_Optimization.recombine_all_partition_circuit(
                    circ,
                    optimized_subcircuits,
                    optimized_parameter_list,
                    recombine_info,
                )
            )

        if any(c is None for c in optimized_subcircuits) or any(
            p is None for p in optimized_parameter_list
        ):
            raise RuntimeError(
                "Internal error: some partitions were not optimized before reconstruction."
            )
        wide_circuit, wide_parameters = self.ConstructCircuitFromPartitions(
            cast(List[Circuit], optimized_subcircuits),
            cast(List[List[np.ndarray]], optimized_parameter_list),
        )

        if not in_parent:
            print("original circuit:    ", circ.get_Gate_Nums())
            print("reoptimized circuit: ", wide_circuit.get_Gate_Nums())

        qgd_Wide_Circuit_Optimization.check_valid_routing(
            wide_circuit, self.config["topology"]
        )
        self.check_compare_circuits(
            circ,
            orig_parameters,
            wide_circuit,
            wide_parameters,
            label="InnerOptimizeWideCircuit",
        )

        return wide_circuit, wide_parameters

    @staticmethod
    def all_to_all_topology(num_qubits):
        """Undirected all-to-all coupler list for ``num_qubits`` qubits."""
        return [(i, j) for i in range(num_qubits) for j in range(i + 1, num_qubits)]

    @staticmethod
    def linear_topology(num_qubits):
        """Path graph couplers ``(i, i+1)``."""
        return [(i, i + 1) for i in range(num_qubits - 1)]

    @staticmethod
    def star_topology(num_qubits):
        """Star graph: hub qubit ``0`` connected to all others."""
        return [(0, i) for i in range(1, num_qubits)]

    @staticmethod
    def ring_topology(num_qubits):
        """Ring couplers including wrap-around ``(n-1, 0)``."""
        return [(i, (i + 1) % num_qubits) for i in range(num_qubits)]

    @staticmethod
    def lattice_topology(x_qbits, y_qbits):
        """2D grid of size ``x_qbits`` by ``y_qbits`` with nearest-neighbor horizontal and vertical edges."""
        return [
            (i * x_qbits + j, i * x_qbits + (j + 1))
            for i in range(y_qbits)
            for j in range(x_qbits - 1)
        ] + [
            (i * x_qbits + j, (i + 1) * x_qbits + j)
            for i in range(y_qbits - 1)
            for j in range(x_qbits)
        ]

    @staticmethod
    def heavy_hexagonal_topology(rows, cols):
        """Build a finite heavy-hex coupling list (honeycomb with subdivided edges).

        Args:
            rows: Number of rows in the brick-wall honeycomb patch.
            cols: Number of columns in the patch.

        Returns:
            List of undirected edges ``(u, v)``. The first ``rows * cols`` qubit
            indices are honeycomb vertices; each original edge introduces one
            additional degree-2 qubit on the subdivided link.
        """

        def vid(r, c):
            """Linear index for honeycomb vertex at row ``r``, column ``c``."""
            return r * cols + c

        # Underlying honeycomb / brick-wall edges
        base_edges = []

        for r in range(rows):
            for c in range(cols):
                # Vertical brick-wall edges
                if r + 1 < rows:
                    base_edges.append((vid(r, c), vid(r + 1, c)))

                # Alternating horizontal edges
                if c + 1 < cols and ((r + c) % 2 == 0):
                    base_edges.append((vid(r, c), vid(r, c + 1)))

        # Subdivide every honeycomb edge by inserting a qubit
        next_id = rows * cols
        heavy_edges = []

        for u, v in base_edges:
            w = next_id
            next_id += 1
            heavy_edges.append((u, w))
            heavy_edges.append((w, v))

        return heavy_edges

    @staticmethod
    def sycamore_topology():
        """Approximate Sycamore-like 6x9 grid topology (simplified; ignores known dead qubits)."""
        return qgd_Wide_Circuit_Optimization.lattice_topology(
            6, 9
        )  # there is a defective qubit at (0, 3) in the sycamore chip, but we ignore it here for simplicity

    @staticmethod
    def is_valid_routing(wide_circuit, topo):
        """True if every multi-qubit gate's qubits lie in a connected subgraph of undirected ``topo``."""
        if topo is None:
            return True

        import itertools

        topo_set = {frozenset(edge) for edge in topo}

        def qubits_connected(qubits):
            """Whether pairwise couplers in ``topo_set`` connect all qubits in ``qubits``."""
            if len(qubits) <= 1:
                return True
            edges = {
                frozenset((q1, q2))
                for q1, q2 in itertools.combinations(qubits, 2)
                if frozenset((q1, q2)) in topo_set
            }
            if len(edges) == 0:
                return False
            cur_set = set(edges.pop())
            while edges:
                next_edge = next((e for e in edges if len(e & cur_set) > 0), None)
                if next_edge is None:
                    return False
                cur_set |= next_edge
                edges.remove(next_edge)
            return set(qubits) <= cur_set

        return all(
            qubits_connected(gate.get_Involved_Qbits())
            for gate in wide_circuit.get_Flat_Circuit().get_Gates()
            if len(gate.get_Involved_Qbits()) > 1
        )

    @staticmethod
    def check_valid_routing(wide_circuit, topo):
        """Assert ``is_valid_routing``; raises if any gate violates ``topo``."""
        if not qgd_Wide_Circuit_Optimization.is_valid_routing(wide_circuit, topo):
            import itertools, sys
            topo_set = {frozenset(e) for e in topo}
            for gate in wide_circuit.get_Flat_Circuit().get_Gates():
                qbits = gate.get_Involved_Qbits()
                if len(qbits) <= 1:
                    continue
                edges = {frozenset((q1,q2)) for q1,q2 in itertools.combinations(qbits,2) if frozenset((q1,q2)) in topo_set}
                if not edges:
                    sys.stderr.write(f'ROUTING_VIOLATION: {type(gate).__name__} on {qbits} topo={topo}\n')
                    sys.stderr.flush()
                    break
            raise AssertionError("Final circuit contains gates that do not respect the routing constraints.")

    def check_compare_circuits(
        self,
        circ,
        orig_parameters,
        wide_circuit,
        wide_parameters,
        routing=False,
        forced_test=False,
        label=None,
    ):
        """Optionally verify equivalence of ``circ`` and ``wide_circuit`` via ``CompareCircuits``.

        Args:
            circ: Original circuit.
            orig_parameters: Parameters for ``circ``.
            wide_circuit: Optimized or routed circuit.
            wide_parameters: Parameters for ``wide_circuit``.
            routing: If true and initial/final mappings exist in ``self.config``,
                pass them to ``CompareCircuits`` for layout-aware comparison.
            forced_test: If true, run the comparison even when ``test_final_circuit``
                is false in config.

        ``self.config['circuit_validation_tolerance']`` is an infidelity
        threshold for this whole-circuit state-vector check. It is deliberately
        separate from ``self.config['tolerance']``, which controls block
        synthesis and block-level validation.
        """
        forced_test = forced_test or (
            self.config.get("force_small_circuit_validation", True)
            and circ.get_Qbit_Num() <= 12
        )
        if self.config["test_final_circuit"] or forced_test:
            if label is not None:
                print(f"{label}: check_compare_circuits")
            tolerance = _circuit_validation_tolerance(self.config)
            if (
                routing
                and self.config.get("initial_mapping", None) is not None
                and self.config.get("final_mapping", None) is not None
            ):
                CompareCircuits(
                    circ,
                    orig_parameters,
                    wide_circuit,
                    wide_parameters,
                    initial_mapping=self.config["initial_mapping"],
                    final_mapping=self.config["final_mapping"],
                    tolerance=tolerance,
                    parallel=0,
                )
            else:
                CompareCircuits(
                    circ,
                    orig_parameters,
                    wide_circuit,
                    wide_parameters,
                    tolerance=tolerance,
                )

    def route_circuit(self, circ: Circuit, orig_parameters: np.ndarray):
        """Map ``circ`` onto ``self.config['topology']`` using the configured router.

        The strategy is ``self.config['routing-strategy']``, e.g. ``seqpam-ilp``,
        ``seqpam-quick``, ``bqskit-sabre``, ``light-sabre`` (Qiskit), or ``sabre``
        (Squander). Writes ``initial_mapping`` and ``final_mapping`` into
        ``self.config`` when the backend provides them.

        Args:
            circ: Circuit before routing.
            orig_parameters: Parameter vector for ``circ``.

        Returns:
            ``(routed_circuit, routed_parameters)`` laid out for ``self.config['topology']``.
        """
        strategy = self.config.get("routing-strategy", "seqpam-ilp")

        if strategy in ("seqpam-ilp", "seqpam-quick", "bqskit-sabre"):
            from squander import Qiskit_IO
            import bqskit.compiler.compile as bqskit_compile_module
            from bqskit.compiler import Compiler
            from bqskit.compiler.compile import (
                build_sabre_mapping_workflow,
                build_seqpam_mapping_optimization_workflow,
            )

            from bqskit.passes import (
                SetModelPass,
            )
            from bqskit.compiler.machine import MachineModel
            from bqskit.ir.lang.qasm2 import OPENQASM2Language
            from qiskit import qasm2, QuantumCircuit

            # Build BQSKit machine model from your topology
            model = MachineModel(circ.get_Qbit_Num(), self.config["topology"])

            # Convert squander circuit → qiskit → BQSKit
            # (BQSKit has a from_qiskit helper if you go via Qiskit IR)
            circo = Qiskit_IO.get_Qiskit_Circuit(
                circ, np.asarray(orig_parameters, dtype=np.float64)
            )

            bqskit_circ = OPENQASM2Language().decode(qasm2.dumps(circo))
            # Customizable knobs

            if strategy == "seqpam-ilp":
                # Routing-only SEQPAM pass pipeline. Patch the classes BQSKit's
                # workflow factory instantiates, so we do not depend on the private
                # shape of the returned Workflow.
                with patched_seqpam_workflow_classes(
                    bqskit_compile_module,
                    use_squander_partitioner=True,
                    config=self.config,
                ):
                    mainflow = build_seqpam_mapping_optimization_workflow(
                        block_size=3  # SEQPAM uses 3-qubit blocks only
                    )
            elif strategy == "seqpam-quick":
                # Keep BQSKit's QuickPartitioner. QSearch/LEAP are replaced
                # only when the configured optimizer is Squander-native.
                with patched_seqpam_workflow_classes(
                    bqskit_compile_module,
                    use_squander_partitioner=False,
                    config=self.config,
                ):
                    mainflow = build_seqpam_mapping_optimization_workflow(
                        block_size=3  # SEQPAM uses 3-qubit blocks only
                    )
            elif strategy == "bqskit-sabre":
                mainflow = build_sabre_mapping_workflow()
            else:
                raise ValueError(f"Unsupported BQSKit routing strategy: {strategy}")

            routing_workflow = [
                SetModelPass(model),  # attach hardware model to circuit
                mainflow,
            ]

            # EAPP monkey-patch catches Squander OSR failures per permutation
            # and installs a SWAP-correct fallback in BQSKit worker processes.
            import os as _os, json as _json
            old_patch_env = _os.environ.get('_SQUANDER_EAPP_FALLBACK_PATCH')
            old_config_env = _os.environ.get('_SQUANDER_BQSKIT_CONFIG')
            _os.environ['_SQUANDER_EAPP_FALLBACK_PATCH'] = '1'
            _os.environ['_SQUANDER_BQSKIT_CONFIG'] = _json.dumps(
                _copy_bqskit_synthesis_config(self.config)
            )
            _patch_eapp_if_needed()
            try:
                with Compiler() as compiler:
                    routed_bqskit_circ, pass_data = compiler.compile(
                        bqskit_circ, routing_workflow, True
                    )
            finally:
                if old_patch_env is None:
                    _os.environ.pop('_SQUANDER_EAPP_FALLBACK_PATCH', None)
                else:
                    _os.environ['_SQUANDER_EAPP_FALLBACK_PATCH'] = old_patch_env
                if old_config_env is None:
                    _os.environ.pop('_SQUANDER_BQSKIT_CONFIG', None)
                else:
                    _os.environ['_SQUANDER_BQSKIT_CONFIG'] = old_config_env

            # Convert back: BQSKit → Qiskit → Squander
            circuit_qiskit_routed = QuantumCircuit.from_qasm_str(
                OPENQASM2Language().encode(routed_bqskit_circ)
            )
            Squander_remapped_circuit, parameters_remapped_circuit = (
                Qiskit_IO.convert_Qiskit_to_Squander(circuit_qiskit_routed)
            )
            self.config["initial_mapping"] = list(pass_data.initial_mapping)
            self.config["final_mapping"] = list(pass_data.final_mapping)

        elif strategy == "light-sabre":
            from squander import Qiskit_IO
            from qiskit import transpile
            from qiskit.transpiler.preset_passmanagers import (
                generate_preset_pass_manager,
            )
            from qiskit.transpiler.passes import SabreLayout, SabreSwap
            from qiskit.transpiler import PassManager, CouplingMap
            from squander.gates import gates_Wrapper as gate

            # SUPPORTED_GATES_NAMES = {n.lower().replace("cnot", "cx") for n in dir(gate) if not n.startswith("_") and issubclass(getattr(gate, n), gate.Gate) and n not in ("Gate", "CROT", "CR", "SYC", "CCX", "CSWAP")}
            circo = Qiskit_IO.get_Qiskit_Circuit(
                circ, np.asarray(orig_parameters, dtype=np.float64)
            )
            coupling_map = [[i, j] for i, j in self.config["topology"]]
            # circuit_qiskit_sabre = transpile(circo, basis_gates=SUPPORTED_GATES_NAMES, coupling_map=coupling_map, optimization_level=0)
            coupling_map = CouplingMap(coupling_map)
            # Customizable SABRE parameters
            sabre_seed = self.config.get("sabre_seed", 42)
            sabre_trials = self.config.get("sabre_trials", 5)  # layout trials
            swap_trials = self.config.get("sabre_swap_trials", sabre_trials)
            heuristic = self.config.get(
                "sabre_heuristic", "decay"
            )  # "basic" | "lookahead" | "decay"

            layout_pass = SabreLayout(
                coupling_map,
                seed=sabre_seed,
                max_iterations=sabre_trials,
                swap_trials=swap_trials,
            )
            swap_pass = SabreSwap(
                coupling_map,
                heuristic=heuristic,
                seed=sabre_seed,
                trials=swap_trials,
            )

            pm = PassManager(
                [
                    layout_pass,  # find initial qubit mapping via SABRE
                    swap_pass,  # insert SWAP gates for routing
                ]
            )
            circuit_qiskit_sabre = pm.run(circo)
            Squander_remapped_circuit, parameters_remapped_circuit = (
                Qiskit_IO.convert_Qiskit_to_Squander(circuit_qiskit_sabre)
            )
            self.config["initial_mapping"] = (
                circuit_qiskit_sabre.layout.initial_index_layout()
            )
            self.config["final_mapping"] = (
                circuit_qiskit_sabre.layout.final_index_layout()
            )
        elif strategy == "sabre":
            sabre = SABRE(circ, self.config["topology"])
            (
                Squander_remapped_circuit,
                parameters_remapped_circuit,
                pi,
                final_pi,
                swap_count,
            ) = sabre.map_circuit(orig_parameters)
            self.config["initial_mapping"] = pi
            self.config["final_mapping"] = final_pi
        qgd_Wide_Circuit_Optimization.check_valid_routing(
            Squander_remapped_circuit, self.config["topology"]
        )

        print("checking circuit after routing")
        print(self.config)
        self.check_compare_circuits(
            circ,
            orig_parameters,
            Squander_remapped_circuit,
            parameters_remapped_circuit,
            routing=True,
            forced_test=True,
            label="route_circuit",
        )
        return Squander_remapped_circuit, parameters_remapped_circuit
