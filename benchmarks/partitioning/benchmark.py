import itertools
from unittest import result
from squander.partitioning.partition import PartitionCircuitQasm
from qiskit import QuantumCircuit
from squander.partitioning.tools import qiskit_to_squander_name, get_qubits, total_float_ops
from squander import utils

import timeit
import glob
import os

USE_ILP = True

MAX_GATES_ALLOWED = 1024**2 

METHOD_NAMES = [
    "kahn", 
    "tdag",
    "gtqcp",
    "qiskit",
    "qiskit-fusion",
    "bqskit-Quick",
    # "bqskit-Greedy", 
    # "bqskit-Scan",
    # "bqskit-Cluster", 
] + (["ilp", "ilp-fusion", #"ilp-fusion-ca"
      ] if USE_ILP else [])

from squander.gates import gates_Wrapper as gate
SUPPORTED_GATES = {x for n in dir(gate) for x in (getattr(gate, n),) if not n.startswith("_") and issubclass(x, gate.Gate) and n != "Gate"}
SUPPORTED_GATES_NAMES = {n for n in dir(gate) if not n.startswith("_") and issubclass(getattr(gate, n), gate.Gate) and n != "Gate"}

def projectq_import_qasm(filename, eng, initial_state=None):
    import re, math
    from projectq.ops import H, X, Y, Z, S, SqrtX, Sdag, T, Tdag, R, Rx, Ry, Rz, CNOT, CZ
    def _eval_angle(expr):
        # supports e.g. "pi/2", "3*pi/4", numeric literals
        expr = expr.strip().lower().replace("pi", f"({math.pi})")
        return eval(expr, {"__builtins__": {}}, {})
    # Minimal OpenQASM-2 parser for common qelib1 gates.
    # Ignores: creg/measure/barrier/includes/custom-gates. Extend as needed.
    _rx = re.compile(r"rx\(([^)]+)\)\s+q\[(\d+)\];", re.I)
    _ry = re.compile(r"ry\(([^)]+)\)\s+q\[(\d+)\];", re.I)
    _rz = re.compile(r"rz\(([^)]+)\)\s+q\[(\d+)\];", re.I)
    _u3 = re.compile(r"u3\(([^,]+),([^,]+),([^)]+)\)\s+q\[(\d+)\];", re.I)
    _one = re.compile(r"(h|x|y|z|s|t|sdg|tdg|sx|r)\s+q\[(\d+)\];", re.I)
    _cx  = re.compile(r"cx\s+q\[(\d+)\],\s*q\[(\d+)\];", re.I)
    _cz  = re.compile(r"cz\s+q\[(\d+)\],\s*q\[(\d+)\];", re.I)
    _meas   = re.compile(r"measure\s+q\[(\d+)\]\s*->\s*([A-Za-z_]\w*)\[(\d+)\];", re.I)
    _creg   = re.compile(r"creg\s+([A-Za-z_]\w*)\[(\d+)\];", re.I)
    _qreg = re.compile(r"qreg\s+q\[(\d+)\];", re.I)
    with open(filename, "r") as fh:
        lines = [ln.strip() for ln in fh if ln.strip() and not ln.startswith("//")]
    nq = None
    for ln in lines:
        m = _qreg.search(ln)
        if m:
            nq = int(m.group(1)); break
    qureg = eng.allocate_qureg(nq)
    if not initial_state is None:
        eng.flush()
        eng.backend.set_wavefunction(initial_state.tolist(), qureg)
    replay = []
    for ln in lines:
        if ln.lower().startswith(("openqasm", "include", "qreg", "creg", "barrier", "measure")):
            continue
        m = _one.match(ln)
        if m:
            gate, i = m.group(1).lower(), int(m.group(2))
            replay.append(({ "h":H, "x":X, "y":Y, "z":Z, "s":S, "t":T, "sdg":Sdag, "tdg":Tdag, "sx":SqrtX, "r":R }[gate], qureg[i]))
            continue
        m = _rx.match(ln)
        if m: replay.append((Rx(_eval_angle(m.group(1))), qureg[int(m.group(2))])); continue
        m = _ry.match(ln)
        if m: replay.append((Ry(_eval_angle(m.group(1))), qureg[int(m.group(2))])); continue
        m = _rz.match(ln)
        if m: replay.append((Rz(_eval_angle(m.group(1))), qureg[int(m.group(2))])); continue
        m = _u3.match(ln)
        if m:
            theta, phi, lam, i = (_eval_angle(m.group(1)), _eval_angle(m.group(2)), _eval_angle(m.group(3)), int(m.group(4)))
            # Decompose U3(θ, φ, λ) = Rz(φ) Ry(θ) Rz(λ)
            replay.append((Rz(phi), qureg[i])); replay.append((Ry(theta), qureg[i])); replay.append((Rz(lam), qureg[i]))
            continue
        m = _cx.match(ln)
        if m: replay.append((CNOT, (qureg[int(m.group(1))], qureg[int(m.group(2))]))); continue
        m = _cz.match(ln)
        if m: replay.append((CZ, (qureg[int(m.group(1))], qureg[int(m.group(2))]))); continue
        # Unknown/unsupported line types are silently skipped. Add more patterns as needed.
    return qureg, replay
def preprocess_qasm_angles(qasm: str) -> str:
    import re, ast, math

    _ALLOWED_NODES = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Num, ast.Load,
                    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd,
                    ast.Name, ast.Constant, ast.Call)
    _ALLOWED_NAMES = {"pi": math.pi}
    # If you also want 'tau' or 'e': add {"tau": 2*math.pi, "e": math.e}

    def _safe_eval(expr: str) -> float:
        """Evaluate a simple arithmetic expression with 'pi' safely."""
        # normalize unicode minus, strip spaces
        expr = expr.replace("−", "-").strip()
        node = ast.parse(expr, mode="eval")
        for n in ast.walk(node):
            if not isinstance(n, _ALLOWED_NODES):
                raise ValueError(f"Disallowed token in angle expression: {type(n).__name__}")
            if isinstance(n, ast.Name) and n.id not in _ALLOWED_NAMES:
                raise ValueError(f"Unknown name in angle expression: {n.id}")
            if isinstance(n, ast.Call):  # disallow function calls
                raise ValueError("Function calls not allowed in angle expressions")
        return float(eval(compile(node, "<angle>", "eval"), {"__builtins__": {}}, _ALLOWED_NAMES))

    _angle_gate = r"(u1|u2|u3|rx|ry|rz|p|cp|r|crx|cry|crz)"
    # Matches gate(params) with any content until the closing ')'
    _param_pat = re.compile(rf"\b{_angle_gate}\s*\(([^)]+)\)", re.IGNORECASE)

    def _format_float(x: float) -> str:
        # 17 digits -> round-trip for double; avoid scientific noise where possible
        return f"{x:.17g}"
    """Replace symbolic angle expressions (with 'pi') by numeric literals."""
    def _repl(m: re.Match) -> str:
        gate = m.group(1)
        params = m.group(2)
        # split on commas that are not nested (these are flat anyway)
        parts = [p.strip() for p in params.split(",")]
        new_parts = [_format_float(_safe_eval(p)) for p in parts]
        return f"{gate}(" + ",".join(new_parts) + ")"
    return _param_pat.sub(_repl, qasm)
import numpy as np
def normalize_state(state):
  return state / np.linalg.norm(state)
def state_vector_equivalence(psi, phi):
  """Checks if two quantum state vectors are equal up to a global phase."""
  # Normalize to prevent issues if vectors are not exactly normalized
  psi = normalize_state(psi)
  phi = normalize_state(phi)
  
  # Compute relative phase using the first nonzero entry
  phase_factor = np.vdot(psi, phi)  # Inner product
  phase = np.angle(phase_factor)
  
  # Compare after removing global phase
  return np.allclose(psi * np.exp(-1j * phase), phi)
def test_simulation(max_qubits = 4, random_initial_state=True):
    import numpy as np
    SVSIM_METHOD_NAMES = ["SQUANDER", "Qiskit", "qulacs", "Cirq", "ProjectQ"]
    files = glob.glob("benchmarks/partitioning/test_circuit/*.qasm")
    print(f"Total QASM: {len(files)}, max qubits: {max_qubits}")
    for filename in files:
        qc = QuantumCircuit.from_qasm_file(filename)
        qc_gates_names = {qiskit_to_squander_name(inst.operation.name) for inst in qc.data}
        num_gates = len(qc.data)
        fname = os.path.basename(filename)
        if num_gates > MAX_GATES_ALLOWED or not qc_gates_names.issubset(SUPPORTED_GATES_NAMES) or filename.endswith("_qsearch.qasm") or filename.endswith("_squander.qasm"):
            print(f"Skipping {fname}; qubits {qc.num_qubits} gates {qc_gates_names} num_gates {num_gates}")
            continue
        qbit_num = qc.num_qubits
        matrix_size = 1 << qbit_num
        print(f"Testing {fname}; qubits {qbit_num} num_gates {num_gates} matrix_size {matrix_size}")
        if (random_initial_state ) :
            initial_state_real = np.random.uniform(-1.0,1.0, (matrix_size,) )
            initial_state_imag = np.random.uniform(-1.0,1.0, (matrix_size,) )
            initial_state = initial_state_real + initial_state_imag*1j
            initial_state = normalize_state(initial_state)
        else:
            initial_state = np.zeros( (matrix_size), dtype=np.complex128 )
            initial_state[0] = 1.0 + 0j
        res = {}
        for method in SVSIM_METHOD_NAMES:
            output = [None, None]
            if method == "SQUANDER":
                def f():
                    transformed_state = initial_state.copy()
                    circ, params, _ = PartitionCircuitQasm( filename, max_qubits, strategy="ilp-fusion" )
                    circ.set_min_fusion(14)
                    def run():
                        circ.apply_to(params, transformed_state)
                        output[0] = transformed_state
                    output[1] = run
            elif method == "Qiskit":
                import qiskit_aer as Aer
                from qiskit import transpile
                def f():
                    circuit_qiskit = QuantumCircuit.from_qasm_file(filename)
                    init = QuantumCircuit(qbit_num)
                    init.initialize( initial_state )
                    circuit_qiskit = circuit_qiskit.compose(init, front=True)
                    circuit_qiskit.save_statevector()
                    backend = Aer.AerSimulator(method='statevector', fusion_enable=True, fusion_max_qubit=max_qubits)
                    compiled_circuit = transpile(circuit_qiskit, backend)
                    def run():
                        result = backend.run(compiled_circuit).result()
                        transformed_state = result.get_statevector(compiled_circuit)
                        output[0] = np.array(transformed_state)
                    output[1] = run
            elif method == "qulacs":
                from qulacs import QuantumState, converter
                import qulacs
                def f():
                    state = QuantumState(qbit_num)
                    state.load( initial_state )
                    with open(filename, 'r') as file:
                        circuit_qulacs = converter.convert_QASM_to_qulacs_circuit([preprocess_qasm_angles(x) for x in file.readlines() if not x.startswith("creg")])
                    def run():
                        circuit_qulacs.update_quantum_state( state )
                        transformed_state = state.get_vector()
                        output[0] = transformed_state
                    output[1] = run
            elif method == "Cirq":
                import cirq #pip install cirq, ply, qsimcirq
                from cirq.contrib.qasm_import import circuit_from_qasm
                import qsimcirq
                def f():
                    with open(filename, 'r') as file:
                        circuit_cirq = circuit_from_qasm(file.read())
                    #simulator = cirq.Simulator()
                    #initial_state_cirq = cirq.StateVectorSimulationState(initial_state=initial_state, qubits=[cirq.NamedQubit(f"q_{i}") for i in range(qbit_num-1,-1,-1)])
                    options = qsimcirq.QSimOptions(max_fused_gate_size=max_qubits, cpu_threads=os.cpu_count())
                    qs = qsimcirq.QSimSimulator(options)
                    state = initial_state.astype(np.complex64)
                    def run():
                        result = qs.simulate(circuit_cirq, qubit_order=[cirq.NamedQubit(f"q_{i}") for i in range(qbit_num-1,-1,-1)], initial_state=state)
                        #result = simulator.simulate(circuit_cirq, initial_state=initial_state_cirq)
                        transformed_state = result.final_state_vector
                        output[0] = transformed_state
                    output[1] = run
            elif method == "ProjectQ":
                from projectq import MainEngine
                from projectq.backends import Simulator
                from projectq.ops import Measure, All
                def f():
                    eng = MainEngine(Simulator(gate_fusion=True))
                    qureg, replay = projectq_import_qasm(filename, eng, initial_state)
                    def run():
                        for op in replay: op[0] | op[1]
                        eng.flush() #Execute; ProjectQ performs fusion inside the C++ backend
                        _, psi = eng.backend.cheat()
                        transformed_state = np.array(psi, dtype=complex)
                        output[0] = transformed_state
                        All(Measure) | qureg
                    output[1] = run
            print(method)
            tpart = timeit.timeit(f, number=1)
            trun = timeit.timeit(output[1], number=1)
            res[method] = (tpart, trun, output[0])
            #assert state_vector_equivalence(res["SQUANDER"][2], res[method][2])
            overlap = np.abs(np.dot(res["SQUANDER"][2], np.conj(res[method][2])))
            assert np.isclose(overlap, 1.0, atol=1e-3), overlap
            #assert np.linalg.norm(res["SQUANDER"][2] - res[method][2]) < 1e-8
        print({x: res[x][:2] for x in res})

def test_partitions(max_qubits = 4):
    """
    Benchmark partitioning strategies on QASM circuits
    Args:
        
        max_qubits: Max qubits per partition
    """
    files = glob.glob("benchmarks/partitioning/test_circuit/*.qasm")
    print(f"Total QASM: {len(files)}, max qubits: {max_qubits}")
    allfiles = {}
    for filename in files:
        qc = QuantumCircuit.from_qasm_file(filename)
        qc_gates_names = {qiskit_to_squander_name(inst.operation.name) for inst in qc.data}
        num_gates = len(qc.data)
        fname = os.path.basename(filename)
        if num_gates > MAX_GATES_ALLOWED or not qc_gates_names.issubset(SUPPORTED_GATES_NAMES) or filename.endswith("_qsearch.qasm") or filename.endswith("_squander.qasm"):
            print(f"Skipping {fname}; qubits {qc.num_qubits} gates {qc_gates_names} num_gates {num_gates}")
            continue
        circ, _ = utils.qasm_to_squander_circuit(filename)
        gate_dict = {i: gate for i, gate in enumerate(circ.get_Gates())}
        gate_to_qubit = { i: get_qubits(g) for i, g in gate_dict.items() }
        gate_to_tqubit = { i: g.get_Target_Qbit() for i, g in gate_dict.items() }
        print(fname, qc.num_qubits, num_gates, f"k={max_qubits}",
              "Max. float ops:", total_float_ops(qc.num_qubits, max_qubits, gate_to_qubit, None, [[i] for i in gate_dict]),
              "Control-aware Max. float ops:", total_float_ops(qc.num_qubits, max_qubits, gate_to_qubit, gate_to_tqubit, [[i] for i in gate_dict]))
        res = {}
        for method in METHOD_NAMES:
            print(method)
            ls = []
            def f():
                ls.extend(PartitionCircuitQasm( filename, max_qubits, method ))
            t = timeit.timeit(f, number=1)
            partitioned_circuit, parameters, L = ls
            res[method] = len(partitioned_circuit.get_Gates()), t, total_float_ops(qc.num_qubits, max_qubits, gate_to_qubit, None, L), total_float_ops(qc.num_qubits, max_qubits, gate_to_qubit, gate_to_tqubit, L)
        allfiles[fname] = (qc.num_qubits, num_gates, res)
        print(fname, allfiles[fname])
    import json
    print(json.dumps(allfiles, indent=2))    
    import matplotlib.pyplot as plt
    sorted_items = sorted(allfiles.items(), key=lambda item: (item[1][0], item[1][1]))
    circuits_sorted = [name for name, _ in sorted_items]
    circuits_labels = [name.replace(".qasm", f" ({x[0]}, {x[1]})") for name, x in sorted_items]
    print("Circuit Name & Qubit Count & Gate Count & " + " & ".join(METHOD_NAMES) + r"\\")
    if USE_ILP:
        ilpbest = [name for name in circuits_sorted if allfiles[name][2]["ilp"][0] != min(allfiles[name][2][method][0] for method in METHOD_NAMES if method != "ilp")]
        ilptie = [name for name in circuits_sorted if allfiles[name][2]["ilp"][0] == min(allfiles[name][2][method][0] for method in METHOD_NAMES if method != "ilp")]
        print("\n".join(" & ".join((name.replace(".qasm", "").replace("_", r"\_"), str(allfiles[name][0]), str(allfiles[name][1]),
                                    *((r"\textbf{" if allfiles[name][2][method][0] == m else "") + str(allfiles[name][2][method][0]) + (r"}" if allfiles[name][2][method][0] == m else "") for method in METHOD_NAMES))) + r"\\" for name in ilpbest for m in (min(allfiles[name][2][method][0] for method in METHOD_NAMES),)))
        print("\n".join(" & ".join((name.replace(".qasm", "").replace("_", r"\_"), str(allfiles[name][0]), str(allfiles[name][1]),
                                    *((r"\textbf{" if allfiles[name][2][method][0] == m else "") + str(allfiles[name][2][method][0]) + (r"}" if allfiles[name][2][method][0] == m else "") for method in METHOD_NAMES))) + r"\\" for name in ilptie for m in (min(allfiles[name][2][method][0] for method in METHOD_NAMES),)))
    else:
        print("\n".join(" & ".join((name.replace(".qasm", "").replace("_", r"\_"), str(allfiles[name][0]), str(allfiles[name][1]),
                                    *((r"\textbf{" if allfiles[name][2][method][0] == m else "") + str(allfiles[name][2][method][0]) + (r"}" if allfiles[name][2][method][0] == m else "") for method in METHOD_NAMES))) + r"\\" for name in circuits_sorted for m in (min(allfiles[name][2][method][0] for method in METHOD_NAMES),)))
    markers = ["o", "*", "D", "s", "+", "<", ">", "v", "^"]
    for j in range(4):
        title = ["Total Partitions", "Runtime Performance", "Total Float Ops for Fusion", "Total Float Ops for Control-Aware Fusion"][j]
        y_label = ["Partition Count", "Time in seconds", "Number of Float Ops", "Number of Float Ops"][j]
        for i, strat in enumerate(METHOD_NAMES):
            y = [allfiles[name][2][strat][j] for name in circuits_sorted]
            plt.plot(circuits_labels, y, marker=markers[i], label=strat)
        plt.xlabel("Circuit (sorted by qubits, gates)")
        plt.ylabel(y_label, fontsize=6)
        plt.title(f"{y_label} - Maximum k={max_qubits} Qubits per Partition")
        plt.xticks(rotation=45, ha="right")
        plt.legend()
        plt.tight_layout()
        plt.grid(True)
        plt.savefig(f"{title.replace(' ', '_')}-{max_qubits}-max_qubit.svg", format="svg", transparent=True)
        plt.clf()

if __name__ == "__main__":
    for max_qubits in range(3, 6):
        pass #test_partitions(max_qubits)
    test_simulation(5)
