"""Fast phase-resolved Clifford shortest paths, with an explicit proof scope.

For fixed Clifford primitives this exhausts the allowed one-qubit circuits.
For continuous primitives it certifies the listed Clifford-preserving
specializations and macro words, not arbitrary non-Clifford intermediates.
"""

from functools import lru_cache
import heapq
import itertools
import hashlib
import json
from pathlib import Path
import sympy as sp

if __package__:
    from .symbolic_gates import make_gate, gate_definition
    from .symbolic_clifford import analyze_single_qubit_cliffords, CliffordAnalysis, _zero
    from .symbolic_contract import parameter_score, encode_sequence
else:
    from symbolic_gates import make_gate, gate_definition
    from symbolic_clifford import analyze_single_qubit_cliffords, CliffordAnalysis, _zero
    from symbolic_contract import parameter_score, encode_sequence


def _fingerprint(matrix):
    # This only indexes candidates. Every match is then checked exactly.
    return tuple((round(float(sp.re(x).evalf()), 10), round(float(sp.im(x).evalf()), 10)) for x in matrix)


@lru_cache(maxsize=1)
def group_data():
    directory = Path(__file__).parent
    cache = directory / "symbolic_clifford_group.json"
    analysis = None
    if cache.exists():
        data = json.loads(cache.read_text())
        if all(hashlib.sha256((directory/name).read_bytes()).hexdigest() == digest
               for name, digest in data["source_sha256"].items()):
            data = data["analysis"]
            analysis = CliffordAnalysis(tuple(data["representatives"]), tuple(map(tuple,data["elements"])),
                                       tuple(map(tuple,data["multiplication"])), tuple(map(tuple,data["multiplication_phases"])),
                                       tuple(data["inverse_indices"]))
    if analysis is None:
        analysis = analyze_single_qubit_cliffords()
    matrices = {}
    names = {}
    for name, index, phase in analysis.elements:
        state = index*8+phase
        matrices[state] = make_gate(name)
        names[name] = state
    fingerprints = {_fingerprint(matrix): state for state, matrix in matrices.items()}
    identity = names["I"]
    return analysis, matrices, fingerprints, names, identity


def multiply_states(left, right):
    analysis = group_data()[0]
    a, pa = divmod(left, 8)
    b, pb = divmod(right, 8)
    return 8*analysis.multiplication[a][b]+(pa+pb+analysis.multiplication_phases[a][b]) % 8


def exact_state(matrix):
    return _exact_state(sp.ImmutableMatrix(matrix))


@lru_cache(maxsize=8192)
def _exact_state(matrix):
    _, matrices, fingerprints, _, _ = group_data()
    if matrix.free_symbols:
        return None
    state = fingerprints.get(_fingerprint(matrix))
    if state is not None and all(_zero(a-b) for a,b in zip(matrix, matrices[state])):
        return state
    return None


def _specializations(name):
    definition = gate_definition(name)
    if not definition.parameters:
        yield ()
        return
    quarter = tuple(sp.pi*sp.Rational(k, 4) for k in range(-4, 4))
    half = tuple(sp.pi*sp.Rational(k, 2) for k in range(-4, 4))
    if name in {"U", "U3"}:
        yield from itertools.product(half, quarter, quarter)
    elif name == "U2":
        yield from itertools.product(quarter, quarter)
    elif name in {"P", "U1", "Rx", "Ry", "Rz"}:
        yield from ((x,) for x in (quarter if name in {"P", "U1"} else half))
    elif name in {"GPi", "GPi2"}:
        yield from ((sp.Rational(k, 8),) for k in range(-4, 4))
    elif name == "R":
        yield from itertools.product(half, quarter)
    elif name == "PhasedX":
        yield from itertools.product((sp.Rational(k,2) for k in range(-4,4)),
                                    (sp.Rational(k,4) for k in range(-4,4)))
    elif name == "PhasedXZ":
        yield from itertools.product((sp.Rational(k,2) for k in range(-2,2)),
                                    (sp.Rational(k,2) for k in range(-2,2)),
                                    (sp.Rational(k,4) for k in range(-4,4)))


def _ticks(sequence):
    ticks = sp.simplify(500*parameter_score(sequence))
    if not ticks.is_Integer:
        raise ValueError("Clifford proof alphabet requires an exact integral cost grid")
    return int(ticks)


def _dijkstra(edges):
    identity = group_data()[-1]
    distance, predecessor = {identity: 0}, {}
    queue = [(0, identity)]
    while queue:
        cost, state = heapq.heappop(queue)
        if cost != distance[state]:
            continue
        for edge_index, edge in enumerate(edges):
            target = multiply_states(edge[0], state)
            proposal = cost+edge[1]
            if proposal < distance.get(target, 10**18):
                distance[target] = proposal
                predecessor[target] = state, edge_index
                heapq.heappush(queue, (proposal, target))
    return distance, predecessor


@lru_cache(maxsize=64)
def solve_clifford_alphabet(local_gates):
    """Solve all 192 exact phases and all 24 projective classes together."""
    local_gates = tuple(local_gates)
    edges_by_state, nonclifford = {}, []
    for name in local_gates:
        for parameters in _specializations(name):
            operation = (name, *parameters, (0,))
            matrix = make_gate(name, *parameters)
            state = exact_state(matrix)
            if state is None:
                if name in {"T", "Tdg", "P", "U1"}:
                    nonclifford.append((operation, matrix))
                continue
            cost = _ticks((operation,))
            if state != group_data()[-1] and (state not in edges_by_state or cost < edges_by_state[state][1]):
                edges_by_state[state] = state, cost, (operation,)
    edges = list(edges_by_state.values())
    distance, predecessor = _dijkstra(edges)
    def path(state):
        blocks = []
        while state in predecessor:
            previous, edge = predecessor[state]
            blocks.append(edges[edge][2])
            state = previous
        return tuple(op for block in reversed(blocks) for op in block)
    # Include phase-producing T-Pauli-T words, which a Clifford-only node
    # graph would otherwise miss. The word alphabet remains explicit.
    for pauli in ("X", "Y", "Z"):
        middle = group_data()[3][pauli]
        if middle not in distance:
            continue
        middle_ops, middle_matrix = path(middle), make_gate(pauli)
        for (first, a), (last, b) in itertools.product(nonclifford, repeat=2):
            state = exact_state(b*middle_matrix*a)
            if state is None or state == group_data()[-1]:
                continue
            operations = (first,)+middle_ops+(last,)
            cost = _ticks(operations)
            if state not in edges_by_state or cost < edges_by_state[state][1]:
                edges_by_state[state] = state, cost, operations
    edges = list(edges_by_state.values())
    distance, predecessor = _dijkstra(edges)
    fixed_clifford_alphabet = all(not gate_definition(name).parameters and exact_state(make_gate(name)) is not None
                                   for name in local_gates)
    result = dict(local_gates=local_gates, edges=edges, distances=distance, predecessors=predecessor,
                  proof_scope="all_circuits_in_supplied_fixed_clifford_basis" if fixed_clifford_alphabet
                  else "listed_clifford_preserving_words_only", global_local_optimality=fixed_clifford_alphabet)
    verify_shortest_path_certificate(result)
    return result


def clifford_path(solution, state):
    blocks = []
    while state in solution["predecessors"]:
        state, edge_index = solution["predecessors"][state]
        blocks.append(solution["edges"][edge_index][2])
    return tuple(op for block in reversed(blocks) for op in block)


def verify_shortest_path_certificate(solution):
    """Check attainability plus every Bellman inequality and closure edge."""
    identity = group_data()[-1]
    distances, edges = solution["distances"], solution["edges"]
    if distances.get(identity) != 0:
        raise ValueError("Invalid Clifford identity distance")
    for state,weight,operations in edges:
        actual=sp.eye(2)
        for operation in operations:
            actual=make_gate(operation[0],*operation[1:-1])*actual
        if exact_state(actual)!=state or _ticks(operations)!=weight or weight<=0:
            raise ValueError("Invalid Clifford edge identity or weight")
    for state, cost in distances.items():
        if state != identity:
            previous, edge_index = solution["predecessors"][state]
            edge = edges[edge_index]
            if multiply_states(edge[0], previous) != state or distances[previous]+edge[1] != cost:
                raise ValueError("Invalid Clifford predecessor witness")
        for target, weight, _ in edges:
            end = multiply_states(target, state)
            if end not in distances or distances[end] > cost+weight:
                raise ValueError("Clifford certificate violates a Bellman inequality")
    return True


def export_certificate(solution):
    return dict(local_gates=list(solution["local_gates"]), proof_scope=solution["proof_scope"],
                global_local_optimality=solution["global_local_optimality"], cost_unit="1/500 legacy gate score",
                identity_state=group_data()[-1], reachable_states=len(solution["distances"]),
                distances={str(k):v for k,v in solution["distances"].items()},
                predecessors={str(k):list(v) for k,v in solution["predecessors"].items()},
                edges=[dict(target=state, cost_ticks=weight, operations=encode_sequence(ops)) for state,weight,ops in solution["edges"]])
