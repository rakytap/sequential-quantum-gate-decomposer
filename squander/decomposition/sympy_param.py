from __future__ import annotations

from deap import base, creator, tools, gp, algorithms
import numpy as np
import functools, itertools, operator, random
import sympy
import multiprocessing

#A Symbolic Search Framework for Exact Multi-Basis Quantum Gate Decomposition
def get_param_space(params):
    if len(params) == 0: return [()]
    #num_samples = 3 #this should be an odd number to avoid sampling only at pi/2**x
    #return list(itertools.product(*[np.linspace(0, np.pi*2*x[0], num=num_samples, endpoint=False) for x in params]))
    num_samples = int(sympy.nextprime(2*len(params)+1))
    A = [pow(2, i, num_samples) for i in range(len(params))]
    return [tuple((i*A[j] / num_samples) * np.pi*2*(x[0] if x[0] is not None else 1) for j, x in enumerate(params)) for i in range(num_samples)]

def determine_CNOT_structure(Umtx, params):
    if __package__:
        from .guided_tree import N_Qubit_Decomposition_Guided_Tree
    else:
        from guided_tree import N_Qubit_Decomposition_Guided_Tree
    allU = [] #2k+1 samples needed per Fourier analysis
    paramspace = get_param_space(params)
    for pos in paramspace:
        allU.append(np.array(Umtx.subs({vardict[x[1]]: y for x, y in zip(params, pos)}).evalf()).astype(np.complex128))
    config = {'tree_level_max': 4 if len(Umtx) < 8 else 14, 'stop_first_solution': True, 'tolerance': 1e-10}
    optim = N_Qubit_Decomposition_Guided_Tree(allU, config, 0, None, paramspace=paramspace, paramscale=[x[0] for x in params])
    optim.set_Optimizer("BFGS2")
    optim.set_Verbose(0)
    optim.Start_Decomposition()
    cnot_structure = [(gate.get_Target_Qbit(), gate.get_Control_Qbit()) for gate in optim.get_Circuit().get_Gates() if gate.get_Name() == "CNOT"]
    print("CNOT structure:", cnot_structure, "num_samples:", len(paramspace))
    return cnot_structure

if __package__:
    from .symbolic_gates import *
else:
    from symbolic_gates import *


QUBIT = int #np.int32

class ParamIndex:
    def __init__(self, index):
        self.index = index
    def __hash__(self): return hash(self.index)
    def __eq__(self, other): return self.index == other.index
    def __repr__(self): return f"ParamIndex({self.index})"
class AngleScale:
    def __init__(self, num, scale):
        self.num = num
        self.scale = scale
    def __hash__(self): return hash((self.num, self.scale))
    def __eq__(self, other): return self.num == other.num and self.scale == other.scale
    def __repr__(self): return f"AngleScale_{'m' if self.num < 0 else ''}{abs(self.num)}_{self.scale}"
class ParamIndexSum:
    def __init__(self):
        self.params = []
    def add_param(self, param):
        self.params.append(param)
        return self
    def add_params(self, params):
        self.params.extend(params)
        return self
    def to_sympy(self, params):
        return sum([(2*sympy.pi if index.index == -1 else params[index.index]) * scale.num / scale.scale for index, scale in self.params])
    def to_qiskit(self, params):
        return sum([(2*np.pi if index.index == -1 else params[index.index]) * (int(scale.num) / int(scale.scale)) for index, scale in self.params])
    def __hash__(self): return hash(tuple(self.params))
    def __eq__(self, other): return self.params == other.params
    def __lt__(self, other): return len(self.params) < len(other.params)
    def __repr__(self): return f"ParamIndexSum({self.params})"
    def __str__(self):
        return " + ".join([f"{'2π' if index.index == -1 else 'ARG'+str(index.index)}*{scale.num}/{scale.scale}" for index, scale in self.params])
def make_angle(index : ParamIndex, scale : AngleScale):
    return ParamIndexSum().add_param((index, scale))
def angle_sum(angle1, angle2):
    return ParamIndexSum().add_params(angle1.params).add_params(angle2.params)
ANGLE = ParamIndexSum
class CircuitBuilder:
    def __init__(self):
        self.gates = []
    def add_gate(self, gate):
        self.gates.append(gate)
        return self
    def add_circuit(self, circuit):
        self.gates.extend(circuit.gates)
        return self
    def to_sympy(self, params):
        gate_ops = []
        for gate in self.gates:
            gate_name, qubits, gate_params = gate
            nqbits = gate_descs[gate_name][0]
            if nqbits == 0:
                gate_ops.append( (gate_name, *[param.to_sympy(params) for param in gate_params], len(qubits), qubits) )
            else: gate_ops.append( (gate_name, *[param.to_sympy(params) for param in gate_params], qubits) )
        return gate_ops
    def to_squander(self, params, num_qubits):
        from squander import Circuit
        circ = Circuit(num_qubits)
        params = []
        for gate in self.gates:
            gate_name, qubits, gate_params = gate
            nqbits, nparams = gate_descs[gate_name][:2]
            gen_func = "add_" + gate_name.upper()
            getattr(circ, gen_func)(*qubits)
            for param in gate_params:
                params.append(param.to_sympy(params))
        return circ, params
    def __repr__(self): return f"CircuitBuilder({self.gates})"

def gen_gate_qubits(name, qubits, *args):
    return CircuitBuilder().add_gate((name, qubits, args))
def gen_gate(name, *args):
    l = len(gate_descs[name][1])
    return CircuitBuilder().add_gate((name, args[l:], args[:l]))
def sequence(circuit1, circuit2):
    return CircuitBuilder().add_circuit(circuit1).add_circuit(circuit2)

def pass_through(x): return x
def randfloat_in_range(lo, hi): return ((hi-lo)*np.random.rand()-lo)

COST_METHOD_TRACE, COST_METHOD_FROBENIUS, COST_METHOD_HILBERT_SCHMIDT = 0, 1, 2
def expr_tree_size(expr: sympy.Expr) -> int:
    """Total number of nodes in the expression tree."""
    return sum(1 for _ in sympy.preorder_traversal(expr))

def expr_tree_depth(expr: sympy.Expr) -> int:
    """Maximum depth of the expression tree."""
    if not expr.args:
        return 1
    return 1 + max(expr_tree_depth(arg) for arg in expr.args)

def expr_leaf_count(expr: sympy.Expr) -> int:
    """Number of leaf nodes (atoms) in the expression tree."""
    if not expr.args:
        return 1
    return sum(expr_leaf_count(arg) for arg in expr.args)

def term_factor_count(term):
    if term == 0 or sympy.Abs(term) == 1:
        return 0

    coeff, rest = term.as_coeff_Mul()

    # If rest == 1 → pure constant
    if rest == 1:
        return 1

    # Break multiplicative structure
    factors = sympy.Mul.make_args(rest)

    var_count = 0
    const_present = False

    for f in factors:
        if f.free_symbols:
            # depends on parameters
            var_count += 1
        else:
            # pure constant factor (exp(I*pi/4), -1, etc.)
            if f == sympy.I:
                continue  # ignore pure imaginary unit as "constant"
            const_present = True

    if sympy.Abs(coeff) != 1/2 and sympy.Abs(coeff) != 1:
        const_present = True
    #print(sympy.Abs(coeff), const_present, factors, var_count)
    if var_count == 0:
        return 1  # constant only

    return var_count + (1 if const_present else 0)

def expr_structural_cost(expr: sympy.Expr,
                         weight_ops: float = 1.0,
                         weight_depth: float = 0.5,
                         weight_size: float = 0.2,
                         weight_leaves: float = 0.1) -> float:
    """
    Structural 'simplicity' cost for a SymPy expression.
    Lower is 'simpler'. Zero expression gets cost 0.
    """
    # Fast exact zero check
    if expr == 0:
        return 0.0
    terms = sympy.Add.make_args(expr)
    term_counts = [term_factor_count(term) for term in terms]
    num_terms = sum(1 if x != 0 else 0 for x in term_counts)
    #print(terms, [term_factor_count(term) for term in terms])
    num_factors = sum(term_counts)
    return 16.0 + num_terms * weight_ops + num_factors * weight_depth
    #ops = sympy.count_ops(expr, visual=False)
    #depth = expr_tree_depth(expr)
    #size = expr_tree_size(expr)
    #leaves = expr_leaf_count(expr)
    #return 1.0 + (weight_ops * ops +
    #        weight_depth * depth +
    #        weight_size * size +
    #        weight_leaves * leaves)
def build_osr_matrix(U, n, A):
    A = list(reversed(A))
    B = list(sorted(set(range(n)) - set(A), reverse=True))
    A, B = [n-1-q for q in A], [n-1-q for q in B]
    dA = 1 << len(A)
    dB = 1 << len(B)
    U = sympy.Array(U)
    return sympy.permutedims(U.reshape(*([2]*(2*n))), tuple(A) + tuple(t+n for t in A) + tuple(B) + tuple(t+n for t in B)).reshape(dA*dA, dB*dB).tomatrix()

def extract_exp_i_phase(expr):
    """
    Try to write expr = rest * exp(I*t) where exp(I*t) is a multiplicative factor.
    Returns t if found, else None.
    Conservative: only succeeds if an exp(...) factor exists.
    """
    expr = sympy.sympify(expr)
    if expr.is_zero:
        return None

    # Pull out exp(...) factors multiplicatively
    rest, exp_part = expr.as_independent(sympy.exp, as_Add=False)

    # exp_part is product of exp(...) terms (or 1)
    if exp_part == 1:
        return None

    # Collapse product of exp(a)*exp(b) -> exp(a+b)
    exp_part = sympy.powsimp(exp_part, force=True)

    # Now accept only exp(I*t) exactly (or exp(-I*t))
    if exp_part.func is sympy.exp:
        arg = sympy.simplify(exp_part.args[0])
        # Check if arg is I*t with t real-ish symbolically
        t = sympy.simplify(arg / sympy.I)
        # Require that arg == I*t exactly (no remainder)
        if sympy.simplify(arg - sympy.I*t) == 0:
            return t

    return None
def dephase_matrix_by_first_clean_exp(U):
    """
    If we find a pivot entry with a clean multiplicative exp(I*t) factor,
    multiply entire U by exp(-I*t). Otherwise return U unchanged.
    """
    if any(U[i,i] == 1 for i in range(U.shape[0])): return U, None
    for e in list(U):  # row-major
        t = extract_exp_i_phase(e)
        if t is not None:
            phase = sympy.exp(-sympy.I*t)
            return (phase * U).applyfunc(quantsimp), t
    return U, None
def get_eval_circ(unitary, param_syms, compiled_circuit):
    num_qubits = unitary.shape[0].bit_length() - 1
    spcirc = compiled_circuit.to_sympy(param_syms)
    rescirc = compile_gates(num_qubits, [(unitary, list(range(num_qubits)))] + spcirc)
    return num_qubits, rescirc
def eval_osrcost(num_qubits, rescirc, params, param_syms):
    if num_qubits == 1: return 0.0
    if __package__:
        from .guided_tree import N_Qubit_Decomposition_Guided_Tree
    else:
        from guided_tree import N_Qubit_Decomposition_Guided_Tree
    tol = 1e-3
    Fnorm = np.sqrt(1<<num_qubits)
    rescircnps = [np.array(rescirc.evalf(subs={x: y for x, y in zip(param_syms, pos)})).astype(np.complex128) for pos in get_param_space(params)]
    all_cuts = list(N_Qubit_Decomposition_Guided_Tree.unique_cuts(num_qubits))
    topology = [(i, j) for i in range(num_qubits) for j in range(i+1, num_qubits)]
    osr_results = []    
    for rescircnp in rescircnps:
        osr_result = []
        for cut in all_cuts:
            #R = build_osr_matrix(rescirc, num_qubits, cut)
            #p1 = sum(R.multiply_elementwise(R.conjugate()))
            """
            g = R * dagger_gate(R)
            p1 = sympy.Trace(g).rewrite(sympy.Sum)
            p2 = sum(g.multiply_elementwise(g.conjugate()))
            kappa = 1.0 - complex((p2 / (p1 * p1)).simplify().evalf()).real
            osrcosts.append(R.rank()+kappa)
            """        
            rank, s = N_Qubit_Decomposition_Guided_Tree.operator_schmidt_rank(rescircnp, num_qubits, cut, Fnorm, tol)
            num_cnot = N_Qubit_Decomposition_Guided_Tree.ceil_log2(rank)
            #s2 = [x*x for x in s]
            #ss2 = sum(s2)
            #kappa = 1.0 - sum(x*x for x in s2)/(ss2*ss2)
            kappa = N_Qubit_Decomposition_Guided_Tree.dyadic_loss(s, N_Qubit_Decomposition_Guided_Tree.ceil_log2(len(s)))
            osr_result.append((num_cnot, kappa))
        osr_results.append(N_Qubit_Decomposition_Guided_Tree.solve_best_min_cnots(num_qubits, all_cuts, osr_result, topology, use_surplus=True))
    osrcost = sum(max(osr_results)) #sum(sum(x) for x in osr_results)/len(osr_results)
    return osrcost
def eval_symcost(num_qubits, rescirc, allow_global_phase=False):
    if allow_global_phase:
        if all(rescirc[i,j] == (rescirc[0,0] if i==j else 0) for i in range(1<<num_qubits) for j in range(1<<num_qubits)):
            return 0.0
        #diag = sympy.Trace(rescirc).rewrite(sympy.Sum)
        #gamma = diag / sympy.Abs(diag) if diag != 0 else 1 #canonical phase
        #rescirc = (rescirc / gamma).applyfunc(quantsimp)
    rescirc = rescirc - sympy.eye(1<<num_qubits)
    #rescirc = sympy.simplify(rescirc)
    #print(rescirc)
    symcost = sum(expr_structural_cost(expr) for expr in rescirc) / len(rescirc)
    return symcost
def get_gatecost(compiled_circuit, param_syms):
    param_cost = 0.2
    gatecost = sum((1<<len(x[1]))+sum((complex(sympy.log(sympy.Abs(z.to_sympy(param_syms).simplify().as_numer_denom()[1]), 2).evalf()).real+0.01)*param_cost for z in x[2]) for x in compiled_circuit.gates)
    return gatecost
def eval_circ(unitary, params, compiled_circuit, symonly=False, allow_global_phase=False):
    param_syms = [vardict[x[1]] for x in params]
    num_qubits, rescirc = get_eval_circ(unitary, param_syms, compiled_circuit)
    #rescirc = dephase_matrix_by_first_clean_exp(rescirc)[0]    
    if not symonly:
        tr = sympy.Trace(rescirc).rewrite(sympy.Sum)
        cost = 0.0
        paramspace = get_param_space(params)
        for pos in paramspace:
            #.rewrite(sympy.exp)
            cost += 1.0 - complex(tr.evalf(subs={x: y for x, y in zip(param_syms, pos)})).real / (1<<num_qubits)
        cost = cost / len(paramspace)
    else: cost = 0.0
    symcost = eval_symcost(num_qubits, rescirc, allow_global_phase=allow_global_phase)
    osrcost = eval_osrcost(num_qubits, rescirc, params, param_syms)
    gatecost = get_gatecost(compiled_circuit, param_syms)
    if osrcost == 0.0 and (symcost == 0.0 or cost == 0.0):
        print("Found exact solution:", compiled_circuit, osrcost, symcost, cost, gatecost)
        print(rescirc)
    return (osrcost, symcost, cost, gatecost), rescirc
def evaluate(individual, pset, hof, unitary, params, num_qubits, ansatz, cost_func_method):
    #print(individual)
    compiled_circuit = CircuitBuilder()
    count = 0
    for part in ansatz:
        if part is not None:
            angles = [gp.compile(expr=individual[count+i], pset=pset) for i in range(part[0])]
            if len(params) > 0: angles = [angle(*[ParamIndex(i) for i in range(len(params))]) for angle in angles]
            count += part[0]
            compiled_circuit.add_circuit(part[1](*angles))
        else:
            cc = gp.compile(expr=individual[count], pset=pset)
            if len(params) > 0: cc = cc(*[ParamIndex(i) for i in range(len(params))])
            compiled_circuit.add_circuit(cc)
            count += 1
    #print(compiled_circuit)
    (osrcost, symcost, cost, gatecost), rescirc = eval_circ(unitary, params, compiled_circuit, allow_global_phase=True)
    if cost == 0.0 or symcost == 0:
        hof.clear()
        hof.insert(individual)
    import math
    if math.isnan(cost) or math.isnan(symcost):
        print("NaN cost detected:", [str(x) for x in individual], osrcost, symcost, cost, rescirc, unitary)
    #print(cost)
    #if cost < 0.0: print(individual, "Negative cost!", cost, tr, rescirc, unitary)
    return osrcost, symcost, cost, gatecost #return osrcost, symcost, cost, gatecost

def find_best_orientation(cnot_structure, Umtx, param_info):
    results = []
    for x in itertools.product(*[(x, tuple(reversed(x))) for x in cnot_structure]):
        compiled_circuit = CircuitBuilder()
        for qbits in x:
            compiled_circuit.add_circuit(gen_gate("CNOT", *qbits))
        results.append((eval_circ(Umtx, param_info, compiled_circuit, symonly=True), x))
    return min(results, key=lambda x: x[0][0])[1]

def gen_ansatz(gate, scale_max, layers, basis, u3_ansatz=False, gp_gate=False):
    if isinstance(gate, str):
        param_info = gate_descs[gate][1]
        Utarget = make_gate(gate, *[vardict[x[1]] for x in param_info], *((1,) if gate == "GP" else ()))
    else:
        param_info = gate[1]
        Utarget = gate[2](*[vardict[x[1]] for x in param_info])
    #Umtx = make_inverse(Utarget).applyfunc(gateprep)
    Umtx = dagger_gate(Utarget).applyfunc(gateprep)
    print("Target:", Utarget, "Dagger:", Umtx)
    num_qubits = Umtx.shape[0].bit_length() - 1
    assert 'CNOT' in basis
    print(gate, "num_qubits:", num_qubits, "scale_max:", scale_max, "num_params:", len(param_info), "layers:", layers)
    if gate in ("GP",): cnot_structure = []
    elif gate in ("CH", "CZ"): cnot_structure = [(0, 1)]
    elif gate in ("Rxy", "CRX", "CRY", "CRZ", "CP", "CS", "CSX", "CU3", "CU", "Rxx", "Ryy", "Rzz", "iSWAP", "SiSWAP"): cnot_structure = [(0, 1), (0, 1)]
    elif gate in ("SYC", "SWAP", "SSWAP",): cnot_structure = [(0, 1), (0, 1), (0, 1)]
    elif gate in ("Peres",): cnot_structure = [(0, 2), (0, 2), (1, 2), (0, 1), (1, 2)]
    elif gate in ("CCZ", "CCX"): cnot_structure = [(1, 2), (0, 2), (1, 2), (0, 2), (0, 1), (0, 1)]
    elif gate in ("CSWAP",): cnot_structure = [(2, 1), (0, 2), (1, 2), (0, 2), (0, 1), (0, 1), (2, 1)]
    elif num_qubits > 1: cnot_structure = determine_CNOT_structure(Umtx, param_info)
    else: cnot_structure = []
    if len(cnot_structure) and False:
        cnot_structure = find_best_orientation(cnot_structure, Umtx, param_info)
        print("CNOT structure:", cnot_structure)
    if u3_ansatz:
        ansatz, regions = [], []
        for x, y in cnot_structure:
            ansatz.append((3, functools.partial(gen_gate_qubits, 'U3', [x])))
            ansatz.append((3, functools.partial(gen_gate_qubits, 'U3', [y])))
            regions.extend([False]*6)
            ansatz.append((0, functools.partial(gen_gate_qubits, 'CNOT', [x, y])))
        for i in range(num_qubits):
            ansatz.append((3, functools.partial(gen_gate_qubits, 'U3', [i])))
            regions.extend([False]*3)
    else:
        ansatz, regions = [None], [True]
        for x, y in cnot_structure:
            ansatz.extend(((0, functools.partial(gen_gate_qubits, 'CNOT', [x, y])), None))
            regions.append(True)
    if gp_gate:
        ansatz.append((1, functools.partial(gen_gate_qubits, 'GP', tuple(range(num_qubits)))))
        regions.append(False)
    return ansatz, regions, num_qubits, Umtx, param_info
def decompose_unitary(gate, scale_max, layers=4, basis=('CNOT', 'S', 'Sdg', 'T', 'Tdg', 'Ry', 'Rz', 'H')):
    ansatz, regions, num_qubits, Umtx, param_info = gen_ansatz(gate, scale_max, layers, basis, u3_ansatz=False)

    pset = gp.PrimitiveSetTyped("MAIN", [ParamIndex for _ in param_info], CircuitBuilder)
    psetangle = gp.PrimitiveSetTyped("MAIN", [ParamIndex for _ in param_info], ANGLE)
    for name in basis:
        if gate_descs[name][0] == 1:
            pset.addPrimitive(functools.partial(gen_gate, name), [ANGLE]*len(gate_descs[name][1])+[QUBIT]*gate_descs[name][0], CircuitBuilder, name=name)
    #pset.addPrimitive(CNOT_gate, [QUBIT, QUBIT], CircuitBuilder, name="CNOT")
    #for qubit1 in range(num_qubits):
    #    for qubit2 in range(num_qubits):
    #        if qubit1 == qubit2: continue
    #        pset.addPrimitive(functools.partial(CNOT_gate, control_qubit=qubit1, target_qubit=qubit2), [], CircuitBuilder, name=f"CNOT_{qubit1}_{qubit2}")
    #pset.addPrimitive(CRY_gate, [ANGLE, QUBIT, QUBIT], CircuitBuilder, name="CRY")
    pset.addPrimitive(sequence, [CircuitBuilder, CircuitBuilder], CircuitBuilder, name="sequence")
    pset.addEphemeralConstant("qubit_index", functools.partial(np.random.randint, 0, num_qubits), ret_type=QUBIT)

    for ps in (pset, psetangle):
        ps.addPrimitive(make_angle, [ParamIndex, AngleScale], ANGLE, name="make_angle")
        ps.addPrimitive(angle_sum, [ANGLE, ANGLE], ANGLE, name="angle_sum")
        
        ps.addTerminal(ParamIndexSum().add_param((ParamIndex(-1), AngleScale(scale_max, scale_max))), ANGLE, name="twopi_angle")
        ps.addTerminal(ParamIndex(-1), ParamIndex, name="twopi")
    
        # Avoid generating zero scale to prevent division by zero in angle construction
        for i in range(-scale_max+1, scale_max + 1):
            if i == 0: continue
            ps.addTerminal(AngleScale(i, scale_max), AngleScale, name=f"AngleScale_{'m' if i < 0 else ''}{abs(i)}")

    import operator
    # Set up the DEAP framework
    def init_individual(regions):
        return creator.Individual([gp.PrimitiveTree((toolbox if is_circ else toolboxangle).expr()) for is_circ in regions])
    creator.create("FitnessMin", base.Fitness, weights=(-1.0, -0.1, -0.01, -0.001))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    tree_depth = num_qubits * (num_qubits - 1) * layers
    # Register the genetic operators
    toolbox = base.Toolbox()
    toolboxangle = base.Toolbox()
    def bounded_expr(pset, min_, max_, max_height=tree_depth):
        while True:
            expr = gp.genHalfAndHalf(pset=pset, min_=min_, max_=max_)
            # Temporarily build an individual to measure height
            ind = gp.PrimitiveTree(expr)
            if ind.height <= max_height:
                return expr
    def mutate_multi(ind, expr, pset):
        i = random.randrange(len(ind))
        ind[i], = gp.mutUniform(ind[i], expr=expr, pset=pset)
        return (ind,)    
    for tb, ps in ((toolbox, pset), (toolboxangle, psetangle)):
        tb.register("compile", gp.compile, pset=ps)
        tb.register("expr", bounded_expr, pset=ps, min_=0, max_=layers)
        tb.register("expr_mut", gp.genHalfAndHalf, pset=ps, min_=0, max_=layers)
        tb.register("mutate", mutate_multi, expr=tb.expr_mut, pset=ps)
    toolbox.register("individual", init_individual, regions=regions)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    hof = tools.HallOfFame(1)
    #toolbox/pset contains wrapper functions
    cost_func_method = COST_METHOD_TRACE
    toolbox.register("evaluate", functools.partial(evaluate, pset=pset, hof=hof, unitary=Umtx, params=param_info, num_qubits=num_qubits, ansatz=ansatz, cost_func_method=cost_func_method))
    # Prefer parsimony to control tree bloat
    #toolbox.register("select", tools.selDoubleTournament, fitness_size=7, parsimony_size=1.4, fitness_first=True)
    toolbox.register("select", tools.selNSGA2)
    def mate_multi(ind1, ind2, termpb=0.9):
        i = random.randrange(len(ind1))
        ind1[i], ind2[i] = gp.cxOnePointLeafBiased(ind1[i], ind2[i], termpb=termpb)
        return ind1, ind2

    toolbox.register("mate", mate_multi, termpb=0.9)
    npopulation = 32*32
    def individual_height(ind):
        return max(tree.height for tree in ind)
    toolbox.decorate("mate", gp.staticLimit(key=individual_height, max_value=tree_depth))
    toolbox.decorate("mutate", gp.staticLimit(key=individual_height, max_value=tree_depth))
    toolbox.register("generate", toolbox.population, n=npopulation)
    pop = toolbox.generate()
    #pop.insert(0, creator.Individual([gp.PrimitiveTree.from_string("Ry(make_angle(twopi, AngleScale_8), 1)", pset), gp.PrimitiveTree.from_string("Ry(make_angle(twopi, AngleScale_m8), 1)", pset), gp.PrimitiveTree.from_string("twopi_angle", pset)]))
    #pop.insert(0, creator.Individual(gp.PrimitiveTree.from_string("sequence(sequence(H(1), CNOT_0_1), H(1))", pset)))
    #pop.insert(0, creator.Individual(gp.PrimitiveTree.from_string("sequence(sequence(H(0), H(1)), sequence(sequence(CNOT_0_1, Rz(make_angle(ARG0, AngleScale_1), 1)), sequence(CNOT_0_1, sequence(H(0), H(1)))))", pset)))
    #pop.insert(0, creator.Individual(gp.PrimitiveTree.from_string("sequence(sequence(sequence(Rz(make_angle(twopi, AngleScale_m4), 1), CNOT_0_1), Rz(make_angle(twopi, AngleScale_m4), 1)), sequence(sequence(Ry(make_angle(twopi, AngleScale_8), 1), CNOT_0_1), sequence(Ry(make_angle(twopi, AngleScale_m8), 1), sequence(Rz(make_angle(twopi, AngleScale_2), 1), P(make_angle(twopi, AngleScale_4), 0)))))", pset)))
    #for x in pop: print(x)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)
    algorithms.eaSimple(pop, toolbox, cxpb=0.6, mutpb=0.3, ngen=100000, stats=stats, halloffame=hof, verbose=True)
    #algorithms.eaMuPlusLambda(pop, toolbox, mu=100, lambda_=1000, cxpb=0.6, mutpb=0.3, ngen=100000, stats=stats, halloffame=hof, verbose=True)
    #algorithms.eaMuCommaLambda(pop, toolbox, mu=100, lambda_=1000, cxpb=0.6, mutpb=0.3, ngen=100000, stats=stats, halloffame=hof, verbose=True)
    #algorithms.eaGenerateUpdate(toolbox, ngen=100000, stats=stats, halloffame=hof, verbose=True)
    print("Best individual:", hof[0])
    print("Best individual fitness:", hof[0].fitness.values)

def reconstruct_path(cameFrom, current):
    total_path = [current]
    while current in cameFrom:
        current = cameFrom[current]
        total_path.append(current)
    return list(reversed(total_path))
def A_star(starts, isGoal, h, getNeighbors, d):
    openSet, cameFrom, gScore, fScore = set(starts), {}, {start: 0 for start in starts}, {start: h(start) for start in starts}
    while openSet:
        current = min(openSet, key=fScore.__getitem__)
        if isGoal(current): return reconstruct_path(cameFrom, current)
        openSet.remove(current)
        for neighbor in getNeighbors(current):
            tentative_gScore = gScore[current] + d(current, neighbor)
            if neighbor not in gScore or tentative_gScore < gScore[neighbor]:
                cameFrom[neighbor] = current
                gScore[neighbor] = tentative_gScore
                fScore[neighbor] = tentative_gScore + h(neighbor)
                if neighbor not in openSet: openSet.add(neighbor)
    return None
def key_conv(x):
    return x
    re, im = x.as_real_imag()
    #return sympy.Trace(sympy.Abs(re)+sympy.Abs(im)*sympy.I).rewrite(sympy.Sum)
    return sympy.Add(*(sympy.Abs(re)+sympy.Abs(im)*sympy.I))
    #return sympy.Abs(sympy.Add(*(re+im)))
def best_first(starts, is_goal, neighbors, eval_E, canonicalize, find_min=False):
    import heapq
    p = multiprocessing.Pool()
    # starts: iterable of nodes (can be just [start_node])
    pq = []
    best_E_by_key = {}   # dominance pruning per canonical state

    for s in starts:
        E, k = eval_E(s)
        k = key_conv(k)
        best_E_by_key[k] = E
        heapq.heappush(pq, (E, s))
    bestE, best = None, None
    while pq:
        E, node = heapq.heappop(pq)

        if is_goal(E):
            bestE, best = E, node
            if not find_min: return node
            continue

        #if any(x in best_E_by_key and best_E_by_key[x] <= E for x in canonicalize(node)):
        #    continue

        _, k = eval_E(node)
        knb = key_conv(k)
        if knb in best_E_by_key and E > best_E_by_key[knb]:
            continue

        nbrs = neighbors(node, E, bestE)
        if len(nbrs) == 0: continue
        print(E, node, k, knb)
        for nb, (Enb, knb) in zip(nbrs, p.map(eval_E, nbrs)):
        #for nb in nbrs:
        #    Enb, knb = eval_E(nb)
            knb = key_conv(knb)

            if knb in best_E_by_key and Enb >= best_E_by_key[knb]:
                continue

            best_E_by_key[knb] = Enb
            heapq.heappush(pq, (Enb, nb))

    return best
def make_circ(ansatz, node, num_qubits):
    compiled_circuit = CircuitBuilder()
    count = 0
    for part in ansatz:
        if part is not None:
            angles = node[count][:part[0]]
            #if len(param_info) > 0: angles = [angle(*[ParamIndex(i) for i in range(len(param_info))]) for angle in angles]
            count += part[0]
            compiled_circuit.add_circuit(part[1](*angles))
        else:
            for _ in range(num_qubits):
                cc = CircuitBuilder()
                for x in node[count]:
                    cc.add_circuit(gen_gate(*x))
                #if len(param_info) > 0: cc = cc(*[ParamIndex(i) for i in range(len(param_info))])
                compiled_circuit.add_circuit(cc)
                count += 1
    #compiled_circuit = CircuitBuilder().add_circuit(gen_gate("Ry", make_angle(ParamIndex(-1), AngleScale(1, 8)), 1)).add_circuit(gen_gate("CNOT", 0, 1)).add_circuit(gen_gate("Ry", make_angle(ParamIndex(-1), AngleScale(-1, 8)), 1)) #CH
    #compiled_circuit = CircuitBuilder().add_circuit(gen_gate("Rz", make_angle(ParamIndex(0), AngleScale(1, 2)), 0)).add_circuit(gen_gate("CNOT", 0, 1)).add_circuit(gen_gate("Rz", make_angle(ParamIndex(0), AngleScale(-1, 2)), 1)).add_circuit(gen_gate("CNOT", 0, 1)).add_circuit(gen_gate("Rz", make_angle(ParamIndex(0), AngleScale(1, 2)), 1)) #CP
    #compiled_circuit = CircuitBuilder().add_circuit(gen_gate("CNOT", 0, 1)).add_circuit(gen_gate("Rz", make_angle(ParamIndex(0), AngleScale(-1, 2)), 1)).add_circuit(gen_gate("CNOT", 0, 1)).add_circuit(gen_gate("Rz", make_angle(ParamIndex(0), AngleScale(1, 2)), 1)) #CP
    #compiled_circuit = CircuitBuilder().add_circuit(gen_gate("Rz", make_angle(ParamIndex(0), AngleScale(1, 2)), 0)).add_circuit(gen_gate("CNOT", 0, 1)).add_circuit(gen_gate("CNOT", 0, 1)) #CP
    #compiled_circuit = CircuitBuilder().add_circuit(gen_gate("Rz", make_angle(ParamIndex(2), AngleScale(1, 1)), 0)).add_circuit(gen_gate("Ry", make_angle(ParamIndex(0), AngleScale(1, 1)), 0)).add_circuit(gen_gate("Rz", make_angle(ParamIndex(1), AngleScale(1, 1)), 0)).add_circuit(gen_gate("Rz", make_angle(ParamIndex(1), AngleScale(1, 1)), 0))
    #compiled_circuit = CircuitBuilder().add_circuit(gen_gate("Rz", make_angle(ParamIndex(2), AngleScale(1, 1)), 0))#.add_circuit(gen_gate("Rz", make_angle(ParamIndex(1), AngleScale(1, 1)), 0))
    #'SiSWAP': lambda: compile_gates(2, [("Ry", sympy.pi/2, [0]), ("S", [1]), ("CNOT", [0, 1]), ("Ry", sympy.pi/4, [0]), ("Ry", sympy.pi/4, [1]), ("CNOT", [0, 1]), ("Ry", -sympy.pi/2, [0]), ("Sdg", [1])]).applyfunc(textbook_simp),
    #compiled_circuit = CircuitBuilder().add_circuit(gen_gate("S", 1)).add_circuit(gen_gate("CNOT", 0, 1)).add_circuit(gen_gate("CNOT", 0, 1)).add_circuit(gen_gate("Sdg", 1))
    #compiled_circuit = CircuitBuilder().add_circuit(gen_gate("CNOT", 0, 1)).add_circuit(gen_gate("Ry", make_angle(ParamIndex(-1), AngleScale(1, 8)), 0)).add_circuit(gen_gate("Ry", make_angle(ParamIndex(-1), AngleScale(1, 8)), 1)).add_circuit(gen_gate("CNOT", 0, 1))
    #compiled_circuit = CircuitBuilder().add_circuit(gen_gate("Ry", make_angle(ParamIndex(-1), AngleScale(1, 4)), 0)).add_circuit(gen_gate("CNOT", 0, 1)).add_circuit(gen_gate("CNOT", 0, 1)).add_circuit(gen_gate("Ry", make_angle(ParamIndex(-1), AngleScale(-1, 4)), 0))
    #compiled_circuit = CircuitBuilder().add_circuit(gen_gate("S", 1)).add_circuit(gen_gate("Ry", make_angle(ParamIndex(-1), AngleScale(1, 4)), 0)).add_circuit(gen_gate("CNOT", 0, 1)).add_circuit(gen_gate("Ry", make_angle(ParamIndex(-1), AngleScale(1, 8)), 0)).add_circuit(gen_gate("Ry", make_angle(ParamIndex(-1), AngleScale(1, 8)), 1)).add_circuit(gen_gate("CNOT", 0, 1)).add_circuit(gen_gate("Ry", make_angle(ParamIndex(-1), AngleScale(-1, 4)), 0)).add_circuit(gen_gate("Sdg", 1))
    #compiled_circuit = CircuitBuilder().add_circuit(gen_gate("S", 1)).add_circuit(gen_gate("CNOT", 0, 1)).add_circuit(gen_gate("Ry", make_angle(ParamIndex(-1), AngleScale(1, 8)), 0)).add_circuit(gen_gate("Ry", make_angle(ParamIndex(-1), AngleScale(1, 8)), 1)).add_circuit(gen_gate("CNOT", 0, 1)).add_circuit(gen_gate("Sdg", 1))
    return compiled_circuit

#@lru_cache(None)
def eval(Umtx, param_info, ansatz, allow_global_phase, node):
    compiled_circuit = make_circ(ansatz, node, Umtx.shape[0].bit_length()-1)
    (osrcost, symcost, cost, gatecost), U = eval_circ(Umtx, param_info, compiled_circuit, False, allow_global_phase=allow_global_phase)
    #print(compiled_circuit, osrcost, symcost, cost, gatecost, U)
    return (osrcost, symcost, cost, gatecost), sympy.ImmutableDenseMatrix(U)
def gen_clifford():
    all_clifford = [x+y for x, y in itertools.product((('I',), ('H',), ('S',), ('H', 'S'), ('S', 'H'), ('Sx',)), (('I',), ('X',), ('Y',), ('Z',)))]
    #all_clifford = [x+y for x, y in itertools.product((('I',), ('H',), ('S',), ('H', 'S'), ('S', 'H'), ('H', 'S', 'H')), (('I',), ('H', 'S', 'S', 'H'), ('H', 'S', 'S', 'H', 'S', 'S'), ('S', 'S')))]
    clifford_gates = {}
    phases = tuple(make_angle(ParamIndex(-1), AngleScale(i, 8)) for i in range(8))
    print("Phases:", phases)
    for cliff in all_clifford:
        for i in range(len(phases)):
            cb = CircuitBuilder()
            for gate in (gen_gate(x, 0) for x in cliff):
                cb.add_circuit(gate)
            cb.add_circuit(gen_gate_qubits("GP", [0], phases[i]))
            mat = sympy.ImmutableMatrix(compile_gates(1, cb.to_sympy([])).applyfunc(textbook_simp))
            res = decompose_unitary_search((1, [], lambda: mat), 8, basis=('CNOT', 'H', 'S', 'Sdg', 'Sx', 'Sxdg', 'X', 'Y', 'Z', 'T', 'Tdg', 'P', 'Rx', 'Ry', 'Rz', 'U2', 'U3'), allow_global_phase=False, find_min=True)[1]
            name = "".join(cliff).replace('I', '').replace("SZ", "Sdg").replace("SxX", "Sxdg")
            if name == "": name = "I"
            clifford_gates[mat] = (name+(f"_q{i}" if i!=0 else ""), res)
            print(cliff, i, mat)
    subs = {sympy.I: sympy.Symbol("sympy.I", real=True),
            sympy.pi: sympy.Symbol("sympy.pi", real=True)}
    for gate in clifford_gates:
        if clifford_gates[gate][0] in basic_clifford: continue
        strmat = str(gate.subs(subs).applyfunc(sympy.factor_terms)) #.replace(sympy.sqrt(2)/2, 1/sympy.sqrt(2))
        print(f'    \"{clifford_gates[gate][0]}\": [{", ".join("("+repr(x[0])+", "+", ".join(str(y) for y in x[1:-1])+(", " if len(x[1:-1]) else "")+str(x[-1])+")" for x in clifford_gates[gate][1])}],')
    for gate in clifford_gates:
        if clifford_gates[gate][0] in basic_clifford: continue
        strmat = str(gate.subs(subs).applyfunc(sympy.factor_terms)) #.replace(sympy.sqrt(2)/2, 1/sympy.sqrt(2))
        print(f'    \"{clifford_gates[gate][0]}\": (1, [], lambda: sympy.{strmat.replace("sqrt", "sympy.sqrt")}.applyfunc(sympy.factor_terms)),')
    s, count = "", 0
    for gate in clifford_gates:
        rev = sympy.ImmutableMatrix(compile_gates(1, [(dagger_gate(gate), [0])]).applyfunc(textbook_simp))
        s += f'({repr(clifford_gates[gate][0])}, {repr(clifford_gates[rev][0])}), '
        count += 1
        if count == len(phases):
            print(s)
            s, count = "", 0

#not maximally entangling: CS, CT
#maximally entangling: CNOT, CZ, CY, CH, iSWAP, SiSWAP (partial), XX(pi/4), exp(i pi/4 (X⊗X + Y⊗Y)), SSWAP (partial)
#maximally entangling only at specific angles: CRX, CRY, CRZ, CP, Rxy, iSWAP_pow
def decompose_unitary_search(orig_gate, scale_max, layers=4, basis=('CNOT',)+single_gate_clifford+('T', 'Tdg', 'P', 'Rx', 'Ry', 'Rz', 'U2'), allow_global_phase=True, find_min=False):
    ansatz, regions, num_qubits, Umtx, param_info = gen_ansatz(orig_gate, scale_max, layers, basis)
    no_ansatz = False
    if no_ansatz: ansatz, regions = [None], [True]
    params = [vardict[x[1]] for x in param_info]
    num_regions = len(regions)*num_qubits
    def gen_angles(s, reversed=False): #we use a half open interval based on (−π,π].
        return [make_angle(ParamIndex(i), AngleScale(scale*s, scale_max*(1 if i==-1 else param_info[i][0]))) for i in range(-1, len(param_info)) for scale in (range(scale_max//2-1, -scale_max//2-1, -1) if reversed else range(-scale_max//2+1, scale_max//2+1)) if scale != 0 or i == 0]
    def sympy_to_paramsum(expr, s):
        res = ParamIndexSum()
        for term in sympy.Add.make_args(expr):
            if term.is_constant():
                if term <= s*sympy.pi: term += s*2*sympy.pi
                if term > s*sympy.pi: term -= s*2*sympy.pi
                term = term/(2*sympy.pi)
                num, denom = term.as_numer_denom()
                res = angle_sum(res, make_angle(ParamIndex(-1), AngleScale(num*scale_max/denom, scale_max)))
            else:
                coeff, rest = term.as_coeff_Mul()
                paramidx = params.index(rest)
                num, denom = coeff.as_numer_denom()
                res = angle_sum(res, make_angle(ParamIndex(paramidx), AngleScale(num*scale_max*param_info[paramidx][0]/denom, scale_max*param_info[paramidx][0])))
        return res
    @functools.lru_cache(None)
    def reverse_gate(gate, angles):
        if len(gate_inverses[gate]) == 1: return gate_inverses[gate][0], angles
        target_gate, angle_mapper = gate_inverses[gate]
        mapped_angles = tuple(sympy_to_paramsum(z, gate_descs[target_gate][1][i][0]) for i, z in enumerate(angle_mapper(*(x.to_sympy(params) for x in angles))))
        #print(gate, angles, target_gate, mapped_angles)
        test = compile_gates(1, [(gate, *[x.to_sympy(params) for x in angles], (0,)), (target_gate, *[x.to_sympy(params) for x in mapped_angles], (0,))]).applyfunc(textbook_simp)
        assert test == sympy.eye(2), test
        return target_gate, mapped_angles
    def gen_gate_angles(gate):
        return itertools.product(*[gen_angles(x[0]) for x in gate_descs[gate][1]])
    #comp_angles = {make_angle(ParamIndex(i), AngleScale(scale, scale_max)): make_angle(ParamIndex(i), AngleScale(-scale, scale_max)) for i in range(-1, len(param_info)) for scale in range(-scale_max, scale_max+1) if scale != 0}
    gatecheck = {}
    for gate in basis:
        if gate_descs[gate][0] == 1:
            for angles in gen_gate_angles(gate):
                mat = sympy.ImmutableMatrix(compile_gates(1, [(gate, *[x.to_sympy(params) for x in angles], (0,))]).applyfunc(textbook_simp))
                if mat == sympy.eye(2): continue
                if not mat in gatecheck: gatecheck[mat] = []
                gatecheck[mat].append((gate, angles))
    subs = {sympy.I: sympy.Symbol("sympy.I", real=True),
            sympy.pi: sympy.Symbol("sympy.pi", real=True)}
    # for mat in gatecheck:
    #     if len(gatecheck[mat]) > 1:
    #         primgate = next(iter(x for x in gatecheck[mat] if len(x[1]) == 0), next(iter(gatecheck[mat])))
    #         if primgate is not None:
    #             print(f'\"{primgate[0] + ("" if len(primgate[1]) == 0 else "_" + "_".join(str(x.to_sympy(params)/sympy.pi*scale_max).replace("-", "m") for x in primgate[1]))}\": ({", ".join(["(" + repr(x[0]) + ", (" + ", ".join([str(z.to_sympy(params).subs(subs)) for z in x[1]]) + ",))" for x in gatecheck[mat] if x!=primgate or len(x[1]) != 0])},),')
        
    unique_gates = {min(z, key=lambda x: get_gatecost(gen_gate(x[0], *x[1], 0), params)) for z in gatecheck.values()}
    all_gates = [tuple(((gate, *angles, qbit),) if i==region*num_qubits+qbit else () for i in range(num_regions))
                 for gate in basis if gate_descs[gate][0] == 1
                 for region in range(len(regions)) if regions[region]
                 for qbit in range(num_qubits)
                 for angles in gen_gate_angles(gate) if (gate, angles) in unique_gates
            ]
    """
    motifs = [((SINGLE_QUBIT_GATES, ANY_QUBIT_POS),),
        ((SINGLE_QUBIT_GATES, ANY_QUBIT_POS), (SAME_GATE, ANY_OTHER_QUBIT_POS),),
        ((SINGLE_QUBIT_GATES, ANY_QUBIT_POS), (INV_GATE, SAME_POS),),
        ((TWO_QUBIT_GATES, ANY_TWOQUBIT_POS),),
        ((TWO_QUBIT_GATES, ANY_TWOQUBIT_POS), (TWO_QUBIT_GATES, ANY_COMMON_TWOQUBIT_POS),),
        ((TWO_QUBIT_GATES, ANY_TWOQUBIT_POS), (TWO_QUBIT_GATES, SAME_REV_POS), (TWO_QUBIT_GATES, SAME_POS))] #SWAP motif
    """
    if no_ansatz: #1, 2 and 3 CNOT motifs
        all_gates.extend([tuple((("CNOT", qbit1, qbit2),) if i==region*num_qubits+qbit1 or i==region*num_qubits+qbit2 else () for i in range(num_regions)) for qbit1 in range(num_qubits) for qbit2 in range(num_qubits) if qbit1 != qbit2
                                    for region in range(len(regions)) if regions[region]])
        all_gates.extend([tuple((("CNOT", qbit1, qbit2),("CNOT", qbit3, qbit4)) if i==region*num_qubits+qbit1 or i==region*num_qubits+qbit2 else () for i in range(num_regions)) for qbit1 in range(num_qubits) for qbit2 in range(num_qubits) for qbit3 in range(num_qubits) for qbit4 in range(num_qubits) if qbit1 != qbit2 and qbit3 != qbit4 and len({qbit1, qbit2} & {qbit3, qbit4}) != 0
                                    for region in range(len(regions)) if regions[region]])
        all_gates.extend([tuple((("CNOT", qbit1, qbit2),("CNOT", qbit2, qbit1),("CNOT", qbit1, qbit2)) if i==region*num_qubits+qbit1 or i==region*num_qubits+qbit2 else () for i in range(num_regions)) for qbit1 in range(num_qubits) for qbit2 in range(num_qubits) if qbit1 != qbit2
                                    for region in range(len(regions)) if regions[region]]) #SWAP motif
        all_gates.extend([tuple(((gate1, *angles1, qbit1), ("CNOT", qbit1, qbit2), (gate2, *angles2, qbit1)) if i==region*num_qubits+qbit1 else () for i in range(num_regions)) for qbit1 in range(num_qubits) for qbit2 in range(num_qubits) if qbit1 != qbit2
                                    for region in range(len(regions)) if regions[region]
                                    for gate1 in gate_inverses if gate1 in basis
                                    for angles1 in gen_gate_angles(gate1) if (gate1, angles1) in unique_gates
                                    for gate2, angles2 in (reverse_gate(gate1, angles1),)                        ])
        all_gates.extend([tuple(((gate1, *angles1, qbit2), ("CNOT", qbit1, qbit2), (gate2, *angles2, qbit2)) if i==region*num_qubits+qbit1 else () for i in range(num_regions)) for qbit1 in range(num_qubits) for qbit2 in range(num_qubits) if qbit1 != qbit2
                                    for region in range(len(regions)) if regions[region]
                                    for gate1 in gate_inverses if gate1 in basis
                                    for angles1 in gen_gate_angles(gate1) if (gate1, angles1) in unique_gates
                                    for gate2, angles2 in (reverse_gate(gate1, angles1),)
                        ])
        all_gates.extend([tuple((("CNOT", qbit1, qbit2),(gate, *angles, qbit1),(gate, *angles, qbit2),("CNOT", qbit3, qbit4)) if i==region*num_qubits+qbit1 or i==region*num_qubits+qbit2 else () for i in range(num_regions)) for qbit1 in range(num_qubits) for qbit2 in range(num_qubits) for qbit3 in range(num_qubits) for qbit4 in range(num_qubits) if qbit1 != qbit2 and qbit3 != qbit4 and len({qbit1, qbit2} & {qbit3, qbit4}) != 0
                                    for region in range(len(regions)) if regions[region]
                                    for gate in basis if gate_descs[gate][0] == 1
                                    for angles in gen_gate_angles(gate) if (gate, angles) in unique_gates
                                    ])
    if num_qubits > 1:
        all_gates.extend([tuple(((gate, *angles, qbit1),) if i==region*num_qubits+qbit1 else ((gate, *angles, qbit2),) if i==region*num_qubits+qbit2 else () for i in range(num_regions))
                        for gate in basis if gate_descs[gate][0] == 1
                        for region in range(len(regions)) if regions[region]
                        for qbit1 in range(num_qubits) for qbit2 in range(qbit1+1, num_qubits)
                        for angles in gen_gate_angles(gate) if (gate, angles) in unique_gates
                ])
        all_gates.extend([tuple(((gate1, *angles1, qbit),) if i==region1*num_qubits+qbit else ((gate2, *angles2, qbit),) if i==region2*num_qubits+qbit else () for i in range(num_regions))
                        for gate1 in gate_inverses if gate1 in basis
                        for region1 in range(len(regions)) if regions[region1]
                        #for region2 in (next(iter(region2 for region2 in range(region1+1, len(regions)) if regions[region2]), None),) if region2 is not None
                        for region2 in range(region1+1, len(regions)) if regions[region2]
                        for qbit in range(num_qubits)
                        for angles1 in gen_gate_angles(gate1) if (gate1, angles1) in unique_gates
                        for gate2, angles2 in (reverse_gate(gate1, angles1),)
                ])
    #print(all_gates)

    #startcost = eval_circ(Umtx, param_info, CircuitBuilder())[0]
    starts = list(itertools.product(*[(((),) if region else ((x,) for x in gen_angles(1))) for region in regions for qbit in range(num_qubits if region else 1)]))
    starts[0] = ((), (('Tdg', 1,),), (('T', 2,),), (), (('T', 1,),), (('Rx', sympy.pi/2, 2,), ('T', 2,)), (), (), (('T', 2,),),
                 (), (), (('Tdg', 2,),), (), (), (('T', 2,), ('Rx', -sympy.pi/4, 2,), ('Ry', -sympy.pi/2, 2,)), (), (('T', 1,),), (),
                 (), (), (), (('Tdg', 0,),), (('Tdg', 1,),), ())
    starts[0] = ((), (('Tdg', 1,),), (('T', 2,),), (), (('T', 1,),), (('T', 2,),), (), (), (('T', 2,),),
                 (), (), (('Tdg', 2,),), (), (), (('T', 2,),), (), (('T', 1,),), (),
                 (), (), (), (('Tdg', 0,),), (('Tdg', 1,),), ())
    #'CSWAP': lambda: compile_gates(3, [('Tdg', (1,)), ('T', (2,)), ('CNOT', [2, 1]), ('T', (1,)), ('Rx', sympy.pi/2, (2,)), ('T', (2,)), ('CNOT', [0, 2]), ('T', (2,)), ('CNOT', [1, 2]), ('Tdg', (2,)), ('CNOT', [0, 2]), ('T', (2,)), ('Rx', -sympy.pi/4, (2,)), ('Ry', -sympy.pi/2, (2,)), ('CNOT', [0, 1]), ('T', (1,)), ('CNOT', [0, 1]), ('CNOT', [2, 1]), ('Tdg', (0,)), ('Tdg', (1,)), ('GP', -sympy.pi/8, 3, [0, 1, 2])]),
    #res = A_star(starts, lambda node: eval(node) == 0, lambda node: eval(node),
    #       lambda node: [tuple(x+((gate[1:],) if gate[0] == i else ()) for i, x in enumerate(node)) for gate in gates],
    #       lambda node1, node2: abs(eval(node2) - eval(node1)))
    gate_costs = {gates: sum(get_gatecost(gen_gate(x[0], *x[1:-1], 0), params) for region in gates for x in region) for gates in all_gates}
    min_gate_costs = min(gate_costs.values())
    def get_neighbors(node, E, bestE):
        if bestE is not None and bestE[3] <= E[3] + min_gate_costs: return []
        try_gates = all_gates if bestE is None else (gates for gates in all_gates if bestE[3] > E[3]+gate_costs[gates])
        return [tuple(x[:i]+y+x[i:] for x, y, i in zip(node, gates, pos)) for gates in try_gates
                for pos in itertools.product(*[range((len(x) if len(gates[i]) > 0 else 0)+1) for i, x in enumerate(node)])]
    if num_qubits == 1:
        pauli_dressings = [
            CircuitBuilder().add_circuit(gen_gate(g, qubit)) for g in basic_pauli
            for qubit in range(num_qubits)
        ]
    elif num_qubits > 1:
        pauli_dressings = [
            CircuitBuilder().add_circuit(gen_gate(g1, qubit1)).add_circuit(gen_gate(g2, qubit2)) for g1, g2 in itertools.product(basic_pauli, repeat=2)
            for qubit1 in range(num_qubits) for qubit2 in range(qubit1+1, num_qubits)
        ]

    def canonicalize(node):
        return
        _, _, U = eval(Umtx, param_info, ansatz, allow_global_phase, node)  # populate cache
        preUs = [compile_gates(num_qubits, pauli_dressing_pre.to_sympy(params) + [(U, list(range(num_qubits)))]) for pauli_dressing_pre in pauli_dressings]
        yield from (mat for pauli_dressing in pauli_dressings for preU in preUs for mat in (sympy.ImmutableMatrix(eval_circ(preU, param_info, pauli_dressing, True, allow_global_phase=allow_global_phase)[2]),) if mat != U)
        #yield from (mat for pauli_dressing in pauli_dressings for mat in (sympy.ImmutableMatrix(eval_circ(U, param_info, pauli_dressing, True, allow_global_phase=allow_global_phase)[2]),) if mat != U)

    res = best_first(starts, lambda E: E[0] == 0.0 and (E[1] == 0.0 or E[2] == 0.0), get_neighbors,
                     functools.partial(eval, Umtx, param_info, ansatz, allow_global_phase), canonicalize, find_min=find_min)
    if res is None: return res
    if not find_min and not no_ansatz:
        basis_no_clifford = tuple(g for g in basis if g not in clifford_descs)
        print(res)
        def remap_qubits(x, qbit): return tuple(z[:-1] + (qbit,) for z in x)
        res = tuple(region if len(region) <= 1 else remap_qubits(decompose_unitary_search((1, param_info, lambda *args: compile_gates(1, make_circ([None], (remap_qubits(region, 0),), 1).to_sympy(args))), scale_max, layers, basis=basis_no_clifford, find_min=True)[0][0], i % num_qubits)
                    for i, region in enumerate(res))
        print(res)
    invparams = {vardict[x[1]]: x[1] for x in param_info}
    invparams.update(subs)
    def circ_to_code(circ):
        return ", ".join(f"({gate}({', '.join(str(x.to_sympy(params)) for x in angles)}), ({', '.join(str(q) for q in qubits)},))" for gate, qubits, angles in circ.gates)
    def circ_to_compile(circ):
        return tuple((gate, *(str(x.to_sympy(params).subs(invparams)) for x in angles), qubits) for gate, qubits, angles in circ.gates)
    def circ_to_qiskit(circ):
        from qiskit.circuit import QuantumCircuit, Parameter
        qc = QuantumCircuit(num_qubits)
        namedict = {"CNOT": "cx", "H": "h", "S": "s", "Sdg": "sdg", "Sx": "sx", "Sxdg": "sxdg",
                    "T": "t", "Tdg": "tdg", "X": "x", "Y": "y", "Z": "z", "P": "p", "Rx": "rx", "Ry": "ry", "Rz": "rz", "U3": "u"}
        p = [Parameter(str(x)) for x in params]
        for gate, qubits, angles in circ.gates:
            if gate == "U2": gate, angles = "U3", [make_angle(ParamIndex(-1), AngleScale(1, 4)), *angles]
            if gate in namedict:
                #if isinstance(namedict[gate], tuple):
                #    for g in namedict[gate]:
                #        getattr(qc, g)(*(x.to_qiskit(p) for x in angles), *qubits)
                #else:
                getattr(qc, namedict[gate])(*(x.to_qiskit(p) for x in angles), *qubits)
            else: assert False, f"Unknown gate {gate}"
        return qc
    fincirc = make_circ(ansatz, res, num_qubits)
    #print(circ_to_code(fincirc))
    compress = circ_to_compile(fincirc)
    print(orig_gate, compress)
    print(circ_to_qiskit(fincirc).draw())

    return res, compress

#decompose_unitary("CS", 16, 2)
#decompose_unitary("SYC", 48, 2)
#decompose_unitary("CCX", 1, 2)
#decompose_unitary("Rxy", 8, 2)
#decompose_unitary("SSWAP", 1, 2)
#decompose_unitary("SiSWAP", 8, 2)
#decompose_unitary("CRY", 2, 2)
#decompose_unitary("iSWAP_pow", 8, 2)
#decompose_unitary("Rxx", 2, 2)
#decompose_unitary("CZ", 2, 2)
#decompose_unitary("CH", 8, 2)
#decompose_unitary_search("CY", 1)
#decompose_unitary_search("CZ", 1)
#decompose_unitary_search("CH", 8)
#decompose_unitary_search("CP", 8)
#decompose_unitary_search("Rxx", 1)
#decompose_unitary_search("Ryy", 1)
#decompose_unitary_search("Rzz", 1)
#decompose_unitary_search("CS", 1)
#decompose_unitary_search("Rxy", 8)
#decompose_unitary_search("SWAP", 1)
#decompose_unitary_search("SSWAP", 8)
#decompose_unitary_search("CSX", 8)
#decompose_unitary_search("iSWAP", 8)
#decompose_unitary_search("SiSWAP", 8)
#decompose_unitary_search("CRX", 4)
#decompose_unitary_search("CRY", 4)
#decompose_unitary_search("CRZ", 4)
#decompose_unitary_search("SYC", 24)
if __name__ == "__main__":
    decompose_unitary_search("CSWAP", 8, allow_global_phase=False)
#decompose_unitary_search("CSWAP", 8, basis=cnot_full_clifford_t, allow_global_phase=False)
#decompose_unitary_search("CU3", 4)
#decompose_unitary_search("U3", 2, allow_global_phase=False)
#decompose_unitary_search("GP", 1)
#decompose_unitary_search("P", 1)
#decompose_unitary_search("Deutsch", 1)
#decompose_unitary_search("fFredkin", 8, basis=cnot_full_clifford_t)
#decompose_unitary_search("Peres", 8, basis=cnot_full_clifford_t)
#decompose_unitary_search("CCX", 1, basis=cnot_full_clifford_t)
#decompose_unitary_search("CCZ", 1)
#decompose_unitary_search((1, [], lambda: compile_gates(1, [("Rx", -3*sympy.pi/2, [0]), ("Rz", -sympy.pi/4, [0]), ("H", [0]), ("X", [0]), ("Rz", -3*sympy.pi/2, [0])])), 8)
#gen_clifford()
