import sys, os

sys.setrecursionlimit(sys.getrecursionlimit()*2*4)

import numpy as np
from numba import njit
from numba.np.unsafe.ndarray import to_fixed_tuple
from functools import lru_cache
from scipy.stats import unitary_group

import groq.api as g
import groq.api.instruction as inst
import groq.api.nn as nn
import groq.tensor as tensor
import groq.runner.tsp as tsp
from groq.common import print_utils
try:
    import groq.runtime as runtime
except ImportError:
    # raise ModuleNotFoundError("groq.runtime")
    print('Error: ModuleNotFoundError("groq.runtime")')

def qiskit_oracle(unitary, qbit_num, parameters, target_qbits, control_qbits, usefloat=True):
    from qiskit import Aer
    from qiskit import QuantumCircuit, execute
    backend = Aer.get_backend('unitary_simulator')
    if usefloat: backend.set_option("precision", "single")
    circuit = QuantumCircuit(qbit_num)
    circuit.unitary(unitary, [i for i in range(qbit_num)])
    for param, target_qbit, control_qbit in zip(parameters, target_qbits, control_qbits):
        if control_qbit is None or target_qbit == control_qbit:
            circuit.u(param[0]*2, param[1], param[2], target_qbit)
        else:
            circuit.cry(param[0]*2, control_qbit, target_qbit)
    job = execute(circuit, backend)
    result=job.result()
    U3_qiskit = result.get_unitary(circuit)
    U3_qiskit = np.asarray(U3_qiskit)
    return U3_qiskit
@njit
def make_u3(parameters):
    return np.array(
        [[np.cos(parameters[0]*2/2), -np.exp(parameters[2]*1j)*np.sin(parameters[0]*2/2)],
         [np.exp(parameters[1]*1j)*np.sin(parameters[0]*2/2), np.exp((parameters[1]+parameters[2])*1j)*np.cos(parameters[0]*2/2)]])
@njit
def make_ry(parameters):
    return make_u3([parameters[0], 0, 0])
    #return np.array(
    #    [[np.cos(parameters[0]*2/2), -np.sin(parameters[0]*2/2)],
    #     [np.sin(parameters[0]*2/2), np.cos(parameters[0]*2/2)]])
@njit
def make_controlled(gate):
    return np.block([[np.eye(2), np.zeros((2, 2))], [np.zeros((2, 2)), gate]]) #[np.ix_(*([[0,2,1,3]]*2))]
@njit
def make_cry(parameters):
    return make_ry(parameters) #make_controlled(make_ry(parameters))
@njit
def twoByTwoFloat(A, B):
    res = np.empty(B.shape, dtype=B.dtype)
    for j in range(2):
        for i in range(B.shape[1]):
            res[j,i] = (np.real(A[j,0])*np.real(B[0,i])-np.imag(A[j,0])*np.imag(B[0,i])) + (np.real(A[j,1])*np.real(B[1,i])-np.imag(A[j,1])*np.imag(B[1,i]))
            res[j,i] += ((np.real(A[j,0])*np.imag(B[0,i])+np.imag(A[j,0])*np.real(B[0,i])) + (np.real(A[j,1])*np.imag(B[1,i])+np.imag(A[j,1])*np.real(B[1,i]))) * 1j
            #((np.real(A[j,0])*np.imag(B[0,i])+np.real(A[j,1])*np.imag(B[1,i])) + (np.imag(A[j,0])*np.real(B[0,i])+np.imag(A[j,1])*np.real(B[1,i]))) * 1j
    return res
#@njit
def apply_to_qbit(unitary, num_qbits, target_qbit, control_qbit, gate):
    pow2qb = 1 << num_qbits
    t = np.arange(num_qbits)
    if not control_qbit is None:
        t[:-1] = np.roll(t[:-1], (target_qbit - control_qbit) % num_qbits)
        gate = make_controlled(gate)
    t = np.roll(t, -target_qbit)
    idxs = np.arange(pow2qb).reshape(*((2,)*num_qbits)).transpose(t).flatten().tolist()
    return np.kron(np.eye(pow2qb>>(1 if control_qbit is None else 2), dtype=np.bool_), gate)[np.ix_(idxs, idxs)].astype(unitary.dtype) @ unitary
@lru_cache
def make_apply_to_qbit_loop(num_qbits):
    twos = tuple([2]*num_qbits)
    @njit
    def apply_to_qbit_loop(unitary, _, target_qbit, control_qbit, gate):
        pow2qb = 1 << num_qbits
        t = np.roll(np.arange(num_qbits), target_qbit)
        idxs = np.arange(pow2qb).reshape(twos).transpose(to_fixed_tuple(t, num_qbits)).copy().reshape(-1, 2) #.reshape(*([2]*num_qbits)).transpose(t).reshape(-1, 2)
        for pair in (idxs if control_qbit is None else idxs[(idxs[:,0] & (1<<control_qbit)) != 0,:]):
            unitary[pair,:] = twoByTwoFloat(gate, unitary[pair,:])
            #unitary[pair,:] = gate @ unitary[pair,:]
        return unitary
    return apply_to_qbit_loop
def process_gates32(unitary, num_qbits, parameters, target_qbits, control_qbits):
    return process_gates(unitary.astype(np.complex64), num_qbits, parameters, target_qbits, control_qbits).astype(np.complex128)
def process_gates(unitary, num_qbits, parameters, target_qbits, control_qbits):
    if unitary.dtype == np.dtype(np.complex128): unitary = np.copy(unitary)
    return process_gates_loop(unitary, num_qbits, parameters, target_qbits, control_qbits, make_apply_to_qbit_loop(num_qbits)) #apply_to_qbit
@njit
def process_gates_loop(unitary, num_qbits, parameters, target_qbits, control_qbits, apply_to_qbit_func):
    for param, target_qbit, control_qbit in zip(parameters, target_qbits, control_qbits):
        unitary = apply_to_qbit_func(unitary, num_qbits, target_qbit, None if control_qbit == target_qbit else control_qbit, (make_u3(param) if control_qbit is None or control_qbit==target_qbit else make_cry(param)).astype(unitary.dtype))
    return unitary
def test():
    print([np.trace(np.real(process_gates(np.eye(1 << 9) + 0j, 9, np.array([[(25+i+d)%64, (50+i)%64, (55+i)%64] for i in range(20)]), np.array([i % 9 for i in range(20)]), np.array([i % 9 for i in range(20)])))) for d in range(4)])
    num_qbits, use_identity = 5, False
    pi = np.pi
    parameters = np.array( [pi/2*0.32, pi*1.2, pi/2*0.89])
    pow2qb = 1 << num_qbits
    unitary = np.eye(pow2qb) + 0j if use_identity else unitary_group.rvs(pow2qb)
    for i in range(num_qbits):
        for j in range(num_qbits):
            target_qbits, control_qbits = np.array([i, (i+1)%num_qbits, i]), np.array([i, i, j])
            gateparams = np.repeat(parameters.reshape(1,3), 3, axis=0)
            actual, oracle = qiskit_oracle(unitary, num_qbits, gateparams, target_qbits, control_qbits), process_gates(unitary, num_qbits, gateparams, target_qbits, control_qbits)
            assert np.allclose(actual, oracle), (i, j, actual, oracle)
#test()
WEST, EAST = 0, 1
s16rangeW = list(range(25, 27+1))+list(range(29, 37+1))+list(range(39,42+1))
s16rangeE = list(range(26, 27+1))+list(range(29,42+1))
s16rangeW2 = list(range(6, 15+1))+list(range(17, 19+1))+list(range(21, 23+1))
s16rangeE2 = list(range(7, 15+1))+list(range(17, 19+1))+list(range(21, 23+1))+[25]
s8range = list(range(17, 19+1))+list(range(21, 23+1))+list(range(25, 26+1))
s8range2 = [27]+list(range(29, 35+1))
def rev_alu(x, do_rev): return (x//4*4)+3-x%4 if do_rev else x
def get_slice1(drctn, start, bank=0):
    return "-1, H1(" + ("W" if drctn==WEST else "E") + "), S1(" + str(start) + "), B1(" + str(bank) + ")"
def get_slice2(drctn, start, end, bank=0):
    return "-1, H1(" + ("W" if drctn==WEST else "E") + "), S2(" + str(start) + "-" + str(end) + "), B1(" + str(bank) + ")"
def get_slice4(drctn, start, end, bank=0):
    return "-1, H1(" + ("W" if drctn==WEST else "E") + "), S4(" + str(start) + "-" + str(end) + "), B1(" + str(bank) + ")"
def get_slice8(drctn, start, end, bank=0):
    return "-1, H1(" + ("W" if drctn==WEST else "E") + "), S8(" + str(start) + "-" + str(end) + "), B1(" + str(bank) + ")"
def get_slice16(drctn, slices, bank=0):
    #return "-1, H1(" + ("W" if drctn==WEST else "E") + "), S16(" + ",".join(str(x) for x in slices) + "), B1(" + str(bank) + ")"
    return "-1, H1(" + ("W" if drctn==WEST else "E") + "), S16(" + str(min(slices)) + "-" + str(max(slices)) + "), B1(" + str(bank) + ")"
def compile_unit_test(name):
    print_utils.infoc("\nCompiling model ...")
    # Compile program to generate IOP. Also generate groqview JSON dump file and
    # check for potential stream conflicts.
    iop_file = g.compile(
        base_name=name, gen_vis_data=True, check_stream_conflicts=True, skip_assembler=False #tree_conflicts=True, inspect_raw=True
    )
    g.write_visualizer_data(name)
    json_file = name + "/visdata.json"
    print_utils.cprint("Have a GroqView:\n    % " + print_utils.Colors.GREEN + "groqview --port 8888 " + json_file + print_utils.Colors.RESET, "")
    return iop_file, json_file
def invoke(devices, iop, pgm_num, ep_num, tensors, lastouts=None, buffers=None):
    """Low level interface to the device driver to access multiple programs. A higher level abstraction
    will be provided in a future release that offers access to multiple programs and entry points."""
    pgm = iop[pgm_num]
    ep = pgm.entry_points[ep_num]
    input_buffers, output_buffers = [], []
    for i, device in enumerate(devices):
        input_buffers.append(runtime.BufferArray(ep.input, 1)[0] if buffers is None else buffers[0][i])
        output_buffers.append(runtime.BufferArray(ep.output, 1)[0] if buffers is None else buffers[1][i])
        if ep.input.tensors:
            for input_tensor in ep.input.tensors:
                if input_tensor.name not in tensors[i]:
                    raise ValueError(f"Missing input tensor named {input_tensor.name}")
                input_tensor.from_host(tensors[i][input_tensor.name], input_buffers[i])
        device.invoke_nonblocking(input_buffers[i], output_buffers[i])
    l = len(devices)
    outs = [{} for _ in range(l)]
    i, checks = -1, list(range(l))
    while l != 0:
        i = (i + 1) % l
        idx = checks[i]
        if not output_buffers[idx].ready(): continue
        if devices[idx].check_faults(): print(devices[idx].dump_faults())
        del checks[i]; l -= 1
        if ep.output.tensors:
            for output_tensor in ep.output.tensors:
                result_tensor = lastouts[idx][output_tensor.name] if not lastouts is None else output_tensor.allocate_numpy_array()
                output_tensor.to_host(output_buffers[idx], result_tensor)
                outs[idx][output_tensor.name] = result_tensor
    return outs, [input_buffers, output_buffers]
def generateOffsetMap(offsets):
    superlane_size = 16
    superlane_count = 20
    bytes_per_element = superlane_size * superlane_count
    offsetMap = []
    for offset in offsets:
        offsetMapEntry = [ 255 ]*bytes_per_element
        splitOffset = (offset & 255, offset >> 8)
        for s in range(superlane_count):
            slot = s*superlane_size
            offsetMapEntry[slot:slot+2] = splitOffset
        offsetMap.append(offsetMapEntry)
    return np.asarray(offsetMap, dtype=np.uint8)
def to_graphviz(g, labels=None, ranks=None, constraints=None, reverse_constraints=None, edge_labels=None):
    return ("digraph {" + ";".join(str(x)+"->"+str(y) + ("[constraint=false]" if not constraints is None and (not x in constraints or not y in constraints[x]) else "") +
        ("" if edge_labels is None or not (x, y) in edge_labels else "[label=\"" + edge_labels[(x, y)] + "\"]") +
        (";"+str(y)+"->"+str(x) + "[style=invis]" if not reverse_constraints is None and x in reverse_constraints and y in reverse_constraints[x] else "") for x in g for y in g[x]) + ";" +
        ("" if labels is None else ";".join(str(x) + "[label=\"" + labels[x] + "\"]" for x in labels) + ";") +
        ("" if ranks is None else "".join("{rank=same; " + "; ".join(str(x) for x in rank) + ";}" for rank in ranks)) +
        "}")
def succ_to_pred(succ):
  pred = {x: set() for x in succ}
  for x in succ:
    for y in succ[x]:
      pred[y].add(x)
  return pred
def gate_op_finder():
    h, w = {}, {}
    for i in range(8): #entries are mtx: [a+bi e+fi] gate; [c+di g+hi] and (a+bi)*(c+di)=(ac-bd)+(cb+da)i
        h[i] = [8+i//4*4+i%4, 8+i//4*4+[3,2,0,1][i%4]]
    for i in range(8):
        h[8+i] = [8+8+i//2]
        w[8+i] = 1
    for i in range(6): #6 adders, last 2 adders are exits
        h[8+8+i] = [8+8+2+i] if i >= 4 else [8+8+4+i%2]
        w[8+8+i] = 1
    for i in range(2): #optional vxm_identity but likely needed for distributor aligned stream group output variant
        h[8+8+6+i] = [] if i >= 0 else [8+8+6+2+i]
        #w[8+8+6+i] = 0
    #for i in range(2): h[8+8+6+2+i] = []
    def gate_to_graphviz(extra_labels=None, edge_labels=None):
        return to_graphviz(h, labels={x: v + ("" if extra_labels is None else "\n" + extra_labels[x]) for x, v in {**{i: ("u" if i%4<2 else "g") + "_" + ("r" if i&1==0 else "i") + "_" + str(i//4) for i in range(8)},
            **{8+i: "g.mul" for i in range(8)}, **{8+8+i: "g.add" if not i in [0,2] else "g.sub" for i in range(6)}, 8+8+6: "real", 8+8+6+1: "imag"}.items()}, edge_labels=edge_labels)
    print(gate_to_graphviz())
    #first a proof that constraining matrix/gates to single hemisphere will not work
    impossible_entry_constraints = {i: list(range(8)) for i in range(8)}
    #groq_alu_finder(h, w, impossible_entry_constraints)
    entry_constraints = {i: set(range(8) if i%4<2 else range(8*8, 8*8+8)) for i in range(8)}
    exit_constraints = {8+8+6+i: set(range(8*(8+7), 8*8*2)) for i in range(2)}
    #exit_constraints = {8+8+6+i: set(range(8*7, 8*8)) for i in range(2)}
    exit_constraints_distributor = None #lambda d, subk, x: not (8+8+6+(1 if subk==8+8+6+0 else 0)) in d or (x % 8) // 2 == (d[8+8+6+(1 if subk==8+8+6+0 else 0)] % 8) // 2
    return groq_alu_finder(h, w, entry_constraints, exit_constraints_distributor, gate_to_graphviz)
def groq_alu_finder(h, w, entry_constraints=None, exit_constraints=None, gviz_func=None):
    g, weight = {}, {} #construct the Groq VXM graph
    for hemi in (WEST, EAST):
        for stage in range(8):
            for sg in range(8): #stream group
                g[hemi*8*8+stage*8+sg] = [] if stage==8-1 else [hemi*8*8+(stage+1)*8+sg]
                if (stage & 1) == 0: g[hemi*8*8+stage*8+sg][:0] = [2*8*8+(sg//4)*8 + (3-stage//2 if hemi else stage//2), 2*8*8+((sg-2)%8//4)*8+4 + (3-stage//2 if hemi else stage//2)]                
    for alu in range(16):
        g[2*8*8+alu] = [hemi*8*8+(((3-(alu % 4)) if hemi else alu % 4)*2+1)*8 + (((alu // 4) * 2 + i) % 8) for i in range(4) for hemi in (WEST, EAST)]
        weight[2*8*8+alu] = 1+((alu % 8) in [0,5,2,7]) #2 for large ALU, 1 otherwise, vxm_identity is a special effectively 0 weight use case during traversal
    entries = list(range(8)) + list(range(8*8, 8*8+8))
    exits = list(range(8*7, 8*8)) + list(range(8*(8+7), 8*8*2))
    pred, hpred = succ_to_pred(g), succ_to_pred(h)
    h = {x: set(h[x]) for x in h}
    dual_edge_constraints = {(2*8*8+alu, hemi*8*8+(((3-(alu % 4)) if hemi else alu % 4)*2)*8 + (((alu // 4) * 2 + i) % 8)): (1-hemi)*8*8+(((3-(alu % 4)) if 1-hemi else alu % 4)*2)*8 + (((alu // 4) * 2 + i) % 8)
        for alu in range(16) for i in range(4) for hemi in (WEST, EAST)} #cannot have same stream group in opposite direction enter a single ALU unit, a joiner node would also work
    #print(g, weight, entries, exits)    
    ranks = [list(range(8*stage, 8*(stage+1)))+list(range(8*(8+7-stage), 8*(8+8-stage))) for stage in range(8)] + [[2*8*8+j*4+i for j in range(4)] for i in range(4)]
    ranks = [ranks[i] for i in [0, 8, 1, 2, 9, 3, 4, 10, 5, 6, 11, 7]]
    labels = {**{hemi*8*8+stage*8+sg: "SG4[" + str(sg) + "]_" + ("W" if hemi==WEST else "E") + "@" + str(stage) for sg in range(8) for stage in range(8) for hemi in (EAST, WEST)},
        **{2*8*8+alu: "ALU" + str(alu) for alu in range(16)}}
    print(to_graphviz(g, labels=labels, ranks=ranks,
        constraints={x: set(g[x]) & set(ranks[i+1]) for i in range(len(ranks)-1) for x in ranks[i]},
        reverse_constraints={x: set(pred[x]) & set(ranks[i+1]) for i in range(len(ranks)-1) for x in ranks[i]}))
    hentries, hexits = {x for x in h if len(hpred[x]) == 0}, {x for x in h if len(h[x]) == 0}
    key = next(iter(hentries))
    if entry_constraints is None: entry_constraints = {x: set(entries) for x in set(hentries)}
    if exit_constraints is None: exit_constraints = {x: set(exits) for x in set(hexits)}
    d, dg, dgpred, sg, sgpred = {key: next(iter(entry_constraints[key]))}, {x: set() for x in h}, {x: set() for x in hpred}, {x: set() for x in g}, {x: set() for x in pred}
    s = [(False, key, None, None, iter(h[key]), None, None)] #stack for recursive DFS-based routine
    max_vxm_ident = 16 - len(w)
    #for num_vxm_ident in range(max_vxm_ident+1):
    bestmapping, bestsz = None, None
    while len(s) != 0:
        #print(s[-1])
        dir, k, subk, gk, it, git, undo = s.pop()
        if not undo is None:
            (sg if not dir else sgpred)[gk].remove(undo)
            (sgpred if not dir else sg)[undo].remove(gk)
            if subk in d and d[subk] == undo:
                (dg if not dir else dgpred)[k].remove(subk)
                (dgpred if not dir else dg)[subk].remove(k)
                if len(dg[subk]) == 0 and len(dgpred[subk]) == 0: del d[subk]
        if git is None:
            gk = d[k]
            git = iter((g if not dir else pred)[gk])
            for x in it:
                if x in (dg if not dir else dgpred)[k]: continue
                subk = x; break
            else:
                #print(k, d)
                chk = {x for x in d if not (h if not dir else hpred)[x] <= (dg if not dir else dgpred)[x]}
                if k in chk: continue #search failed
                if len(chk) == 0:
                    dir = not dir
                    chk = {x for x in d if not (h if not dir else hpred)[x] <= (dg if not dir else dgpred)[x]}
                if len(chk) != 0:
                    k = next(iter(chk))
                    s.append((dir, k, None, None, iter((h if not dir else hpred)[k]), None, None)) #continue target graph search from prior node or change direction
                else:
                    #print(bestmapping, len(s), sg)
                    if bestsz is None or len(sg) < bestsz:
                        bestmapping, bestsz = d.copy(), len(sg)
                        if not gviz_func is None: print(gviz_func({x: labels[bestmapping[x]] for x in bestmapping}, {(x, next(iter(h[x]))): labels[next(iter(sg[bestmapping[x]]))] for x in bestmapping if bestmapping[x] >= 2*8*8}))
                        print("ALU Path ( Length=", bestsz, ") found: ", bestmapping, sg); continue
                continue
        for x in git:
            #cannot have the same stream group in both directions, rather than add a joiner node to the graph, constraint added here
            #ALU allowed 2 inbound edges, 1 outbound edge, other nodes 1 inbound edge only but any of up to 3 outbound edges
            if x in (sg if not dir else sgpred)[gk] or len((sgpred if not dir else sg)[x]) >= ((2 if not dir else 1) if x >= 8*8*2 else (1 if not dir else 3)): continue
            if (((x, gk) if not dir else (gk, x)) in dual_edge_constraints) and ((dual_edge_constraints[(x, gk)] in sgpred[x]) if not dir else (dual_edge_constraints[(gk, x)] in sgpred[gk])): continue
            assignsubk = subk in hentries and x in entry_constraints[subk] or subk in hexits and (exit_constraints(d, subk, x) if callable(exit_constraints) else x in exit_constraints[subk]) or subk in w and x in weight and weight[x] >= w[subk]
            if assignsubk and subk in d and d[subk] != x: continue
            if not assignsubk and x >= 8*8*2: continue #vxm identity allowance possible here
            s.append((dir, k, subk, gk, it, git, x))
            (sg if not dir else sgpred)[gk].add(x)
            (sgpred if not dir else sg)[x].add(gk)
            if assignsubk:
                d[subk] = x
                (dg if not dir else dgpred)[k].add(subk)
                (dgpred if not dir else dg)[subk].add(k)
                s.append((dir, subk, None, None, iter((h if not dir else hpred)[subk]), None, None))
            else: s.append((dir, k, subk, x, it, iter((g if not dir else pred)[x]), None))
            break
        #else: pass #graph search failed so backtracking
    if bestsz is None: print("No ALU path possible")
#gate_op_finder(); assert False
class UnitarySimulator(g.Component):
    def __init__(self, num_qbits, reversedir=False, lastus=None, **kwargs):
        super().__init__(**kwargs)
        self.num_qbits, self.rev = num_qbits, reversedir
        #more efficient to just directly copy on the controlled rotation rather than doing unnecessary identity gate computations
        #self.identity2x2 = [g.from_data(np.zeros((1), dtype=np.float32), layout=get_slice1(WEST, 0, 0), broadcast=True),
        #                    g.from_data(np.ones((1), dtype=np.float32), layout=get_slice1(WEST, 0, 0), broadcast=True)]
        self.otherinit, self.copystore = [], []
        pow2qb = 1 << num_qbits
        num_inner_splits = (pow2qb+320-1)//320 #handle inner splits for >=9 qbits
        for hemi in (EAST, WEST) if reversedir else (WEST, EAST):
            self.otherinit.append(lastus.otherinit[hemi] if not lastus is None else tensor.create_storage_request(layout=get_slice8(hemi, s8range[0], s8range[-1], 0).replace(", S8", ", A" + str(pow2qb*num_inner_splits) + "(0-" + str(pow2qb*num_inner_splits-1) + "), S8")))
            self.copystore.append(lastus.copystore[hemi] if not lastus is None else tensor.create_storage_request(layout=get_slice8(hemi, s8range2[0], s8range2[-1], 0).replace(", S8", ", A" + str(pow2qb*num_inner_splits) + "(0-" + str(pow2qb*num_inner_splits-1) + "), S8")))
    def copymatrix(self, unitaryinit):
        unitaryinit = unitaryinit.read(streams=g.SG8[0], time=0)
        unitary = unitaryinit.write(name="unitary", storage_req=self.otherinit[WEST])
        copy = unitaryinit.write(name="initcopy", storage_req=self.copystore[WEST])
        resultother = self.create_memory_tensor(name="result", storage_req=self.otherinit[EAST], tensor_type=unitary.tensor_type)
        copyother = self.create_memory_tensor(name="copy", storage_req=self.copystore[EAST], tensor_type=copy.tensor_type)    
        return unitary, copy, resultother, copyother
    def cmppairs(a, b):
        return a[0].tolist() == b[0].tolist() and ((a[1] is None) and (b[1] is None) or a[1].tolist() == b[1].tolist())
    def idxmapgather(num_qbits):
        pow2qb = 1 << num_qbits
        idxmap = [np.arange(pow2qb).reshape(*([2]*num_qbits)).transpose(np.roll(np.arange(num_qbits), target_qbit)).reshape(-1, 2) for target_qbit in range(num_qbits)]
        idxmapsort = [x[x[:,0].argsort()] for x in idxmap]
        #idxmapm1 = [np.arange(1 << (num_qbits-1)).reshape(*([2]*(num_qbits-1))).transpose(np.roll(np.arange(num_qbits-1), target_qbit)).reshape(-1, 2) for target_qbit in range(num_qbits-1)]
        #idxmapm1 = [x[x[:,0].argsort()] for x in idxmapm1]
        idxmapm1 = [idxmapsort[i][:pow2qb//4,:] for i in range(num_qbits-1)]
        for target_qbit in range(num_qbits):
            for control_qbit in range(num_qbits):
                if target_qbit == control_qbit: assert UnitarySimulator.cmppairs(UnitarySimulator.idxmap(num_qbits, target_qbit, None), (idxmap[target_qbit], None))
                else:
                    idxs = idxmap[target_qbit]
                    oracle = UnitarySimulator.idxmap(num_qbits, target_qbit, control_qbit)
                    assert UnitarySimulator.cmppairs(oracle, (idxs[(idxs[:,0] & (1<<control_qbit)) != 0,:], idxs[(idxs[:,0] & (1<<control_qbit)) == 0,:]))
                    idxs = idxmapsort[target_qbit]
                    oracle = (oracle[0][oracle[0][:,0].argsort()], oracle[1][oracle[1][:,0].argsort()])
                    actual = (np.array(idxs[idxmapm1[(control_qbit - (control_qbit > target_qbit)) % (num_qbits-1)][:,1]]), np.array(idxs[idxmapm1[(control_qbit - (control_qbit > target_qbit)) % (num_qbits-1)][:,0]]))
                    assert UnitarySimulator.cmppairs(oracle, actual), (target_qbit, control_qbit, actual, oracle)
        return idxmapsort, idxmapm1 #target_qbit and control_qbit gather maps
    def idxmap(num_qbits, target_qbit, control_qbit):
        pow2qb = 1 << num_qbits
        t = np.roll(np.arange(num_qbits), target_qbit)
        idxs = np.arange(pow2qb).reshape(*([2]*num_qbits)).transpose(t).reshape(-1, 2)
        pairs = idxs if control_qbit is None else idxs[(idxs[:,0] & (1<<control_qbit)) != 0,:]
        if not control_qbit is None: bypasspairs = idxs[(idxs[:,0] & (1<<control_qbit)) == 0,:]
        else: bypasspairs = None
        #print(pairs, bypasspairs)
        return pairs, bypasspairs
    def intersection_range(r1, r2):
        mx, mn = max(r1[0], r2[0]), min(r1[-1], r2[-1])
        return (mx, mn) if mx < mn else None
    def difference_range(r1, r2):
        return None if r1 == r2 else (max(r2[0], r1[1]), r2[1])
    def difference_ranges(r, ranges):
        return [(None, None) if x is None else (UnitarySimulator.intersection_range(r, x), UnitarySimulator.difference_range(r, x)) for x in ranges]
    def smallest_contig_range(ranges):
        if all(r is None for r in ranges): return None
        ranges = list(filter(lambda k: not k is None, ranges))
        m = min(ranges)[0]
        return (m, min(map(lambda k: k[1] if k[0] == m else k[0], ranges))) 
    def transpose_null_share(tensors, schedules, gaps, time, tpose_shared):
        Allsplits = [[] for _ in tensors]
        while True:
            r = UnitarySimulator.smallest_contig_range([x[0] for x in schedules])
            if r is None: break
            diffs = UnitarySimulator.difference_ranges(r, [x[0] for x in schedules])
            for i, x in enumerate(diffs):
                Allsplits[i].append(None if x[0] is None else x[0][1] - x[0][0])
                if x[1] is None:
                    del schedules[i][0]
                    if len(schedules[i]) == 0: schedules[i].append(None) 
                else: schedules[i][0] = x[1]
        tensors = [y.split_vectors([x for x in Allsplits[i] if not x is None]) for i, y in enumerate(tensors)]
        t, Allindexes = 0, [0 for _ in tensors]
        for i, splits in enumerate(zip(*Allsplits)):
            with g.ResourceScope(name="dist" + str(i), is_buffered=False, time=time+t) as innerpred:
                res = tpose_shared([None if x is None else tensors[j][Allindexes[j]] for j, x in enumerate(splits)])
                for j, x in enumerate(splits):
                    if not x is None: tensors[j][Allindexes[j]] = res[j]
                Allindexes = [x + (0 if splits[j] is None else 1) for j, x in enumerate(Allindexes)]
            t += next(filter(lambda k: not k is None, splits))
            if t in gaps: t += gaps[t]
        return [g.concat(x, 0) for x in tensors] 
    def build(self, unitary, copy, target_qbit, control_qbit, gate, gatesel=None, tcqbitsel=None, derivdistro=None, inittime=0):
        if copy is None:
            with g.ResourceScope(name="initcopy", is_buffered=True, time=0) as pred:
                unitary, copy, _, _ = self.copymatrix(unitary)
        else: pred = None
        pow2qb = 1 << self.num_qbits
        num_inner_splits = 1 if gatesel is None else (pow2qb+320-1)//320
        innerdim = pow2qb if gatesel is None else 320
        usplit = np.array(g.split_vectors(unitary, [1] * (2*pow2qb*num_inner_splits))).reshape(pow2qb*num_inner_splits, 2)
        ucopysplit = np.array(g.split_vectors(copy, [1] * (2*pow2qb*num_inner_splits))).reshape(pow2qb*num_inner_splits, 2)
        if tcqbitsel is None:
            pairs, bypasspairs = UnitarySimulator.idxmap(self.num_qbits, target_qbit, control_qbit)
            u = [usplit[pairs[:,0],0], usplit[pairs[:,0],1], ucopysplit[pairs[:,1],0], ucopysplit[pairs[:,1],1]]
            ub = [np.array([])]*4 if control_qbit is None else [usplit[bypasspairs[:,0],0], usplit[bypasspairs[:,0],1], ucopysplit[bypasspairs[:,1],0], ucopysplit[bypasspairs[:,1],1]]
            revidx = np.argsort((pairs if control_qbit is None else np.hstack([bypasspairs, pairs])).transpose().flatten()).tolist()         
        r = 1 if control_qbit is None else 2
        with g.ResourceScope(name="rungate", is_buffered=True, time=0 if pred is None else None, predecessors=None if pred is None else [pred]) as pred:
            #(a+bi)*(c+di)=(ac-bd)+(ad+bc)i
            #gate[0] * p[0] - gate[1] * p[1] + gate[2] * p[2] - gate[3] * p[3]
            #gate[0] * p[1] + gate[1] * p[0] + gate[2] * p[3] + gate[3] * p[2]
            if gatesel is None:
                gatevals = g.split_vectors(gate, [1]*(2*2*2))
                gs = [g.concat_vectors([gatevals[i]]*(pow2qb//2*num_inner_splits//r)+[gatevals[i+4]]*(pow2qb//2*num_inner_splits//r), (pow2qb//r, pow2qb)).read(streams=g.SG4[2*i]) for i in range(4)] #, time=0 if i == 0 else None
            else:
                #gate = g.from_addresses(np.array(gate.addrs).reshape(-1, g.float32.size), pow2qb, g.float32, "gatedim")
                gatevals = np.array(g.split_vectors(gate, [1]*(gate.shape[0]))).reshape(gate.shape[0]//8, 2*2*2)
                gatesel_st = g.concat_vectors([gatesel[i].reshape(1,innerdim) for i in range(len(gatesel)) for _ in range(pow2qb//2*num_inner_splits//r)], (pow2qb*len(gatesel)//2*num_inner_splits//r, innerdim)).read(streams=g.SG4[1])
                gs = [g.mem_gather(g.concat_vectors(gatevals[:,i], (gate.shape[0]//8, innerdim)), gatesel_st, output_streams=[g.SG4[2*i]]) for i in range(4)]
            with g.ResourceScope(name="ident", is_buffered=False, time=0) as innerpred:
                if tcqbitsel is None:
                    us = [g.concat_vectors((ub[i%2].flatten().tolist() + ub[i%2+2].flatten().tolist() if i in [0,3] else []) + u[i].flatten().tolist()*2, (pow2qb*num_inner_splits if control_qbit is None or i in [0,3] else pow2qb//2*num_inner_splits, innerdim)).read(streams=g.SG4[2*i+1]) for i in range(4)]
                else:
                    if len(tcqbitsel) == 6 or len(tcqbitsel) == 8:
                        if len(tcqbitsel) == 8:
                            tqbitdistro, tqbitpairs0, tqbitpairs1, cqbitdistro, cqbitpairs0, cqbitpairs1, tcqbitdistro, cqbithighsel = tcqbitsel
                            cqbithighsel1 = g.split(cqbithighsel[1], num_splits=2)
                            tchighgather = g.distribute_8(g.concat([cqbithighsel1[0]]*2+[cqbithighsel1[1]]*2, 0).read(streams=g.SG1[24]), g.stack(pow2qb*num_inner_splits*[tcqbitdistro[1]], 0).read(streams=g.SG1[25]), bypass8=0b11111110, distributor_req=3+(4 if self.rev else 0), time=0)
                            cdistro = g.stack(pow2qb*num_inner_splits*[cqbitdistro[1]], 0).read(streams=g.SG1[16+1]) #, time=1 #slice 14->slice 41 = 44//4-14//4=8
                            lasttensor = [None]
                            def shared_tpose(tensors):
                                if tensors[1] is None:
                                    lasttensor[0] = g.transpose_null(tensors[0], transposer_req=3 if self.rev else 1, stream_order=[8], time=0)
                                    return lasttensor[0], None
                                if lasttensor[0].shape != tensors[1].shape: lt, lasttensor[0] = g.split(lasttensor[0], splits=[tensors[1].shape[0], lasttensor[0].shape[0] - tensors[1].shape[0]])
                                else: lt, lasttensor[0] = lasttensor[0], None
                                tchigh = g.mem_gather(cqbitpairs0[1], lt, output_streams=[g.SG1[16]]) #, time=-7
                                readcontrols = g.distribute_8(tchigh, tensors[1], bypass8=0b11111110, distributor_req=2+(4 if self.rev else 0)) #, time=-1
                                if tensors[0] is None: return None, g.transpose_null(readcontrols, transposer_req=3 if self.rev else 1, stream_order=[0], time=0)
                                else:
                                    readcontrols, lt = g.split(g.transpose_null(g.stack((readcontrols, tensors[0]), 1), transposer_req=3 if self.rev else 1, stream_order=[0, 8], time=0), num_splits=2, dim=1)
                                    lt = lt.reshape(lt.shape[0], lt.shape[-1])
                                    if lasttensor[0] is None: lasttensor[0] = lt
                                    else: lasttensor[0] = g.concat((lasttensor[0], lt), 0)
                                    return lt, readcontrols.reshape(readcontrols.shape[0], readcontrols.shape[-1])
                            _, readcontrols = UnitarySimulator.transpose_null_share([tchighgather, cdistro],
                                [[(x, min(x+15, tchighgather.shape[0])) for x in range(0, tchighgather.shape[0], 15)], [(15, 15+cdistro.shape[0])]], {}, 1, shared_tpose)
                        else:
                            tqbitdistro, tqbitpairs0, tqbitpairs1, cqbitdistro, cqbitpairs0, cqbitpairs1 = tcqbitsel
                            if self.num_qbits > 8:
                                for x in (tqbitpairs0, tqbitpairs1, cqbitpairs0, cqbitpairs1):
                                    for i in range(2): x[i] = g.split(x[i], num_splits=2)[target_qbit//8]
                            if self.num_qbits > 9:
                                for x in (cqbitpairs0, cqbitpairs1):
                                    for i in range(2): x[i] = g.split(x[i], num_splits=2)[control_qbit//8]
                            cdistro = g.stack(pow2qb*num_inner_splits*[cqbitdistro[1]], 0).read(streams=g.SG1[16+4])
                            readcontrols = g.distribute_8(g.stack([cqbitpairs0[1]]*2 + [cqbitpairs1[1]]*2, 0).reshape(pow2qb*num_inner_splits, innerdim).read(streams=g.SG1[16]), cdistro, bypass8=0b11111110, distributor_req=2+(4 if self.rev else 0))
                            readcontrols = g.transpose_null(readcontrols, transposer_req=3 if self.rev else 1, stream_order=[0], time=0)
                        tqb = g.mem_gather(tqbitpairs0[1], readcontrols, output_streams=[g.SG1[0]])
                        tqbp = g.mem_gather(tqbitpairs1[1], readcontrols, output_streams=[g.SG1[8]])
                    else:
                        tqbitdistro, tqbitpairs0, tqbitpairs1 = tcqbitsel
                        if self.num_qbits > 8:
                            for x in (tqbitpairs0, tqbitpairs1):
                                for i in range(2): x[i] = g.split(x[i], num_splits=2)[target_qbit//8]
                        tqb = g.concat_vectors([tqbitpairs0[1]]*2, (pow2qb*num_inner_splits, innerdim)).read(streams=g.SG1[0])
                        tqbp = g.concat_vectors([tqbitpairs1[1]]*2, (pow2qb*num_inner_splits, innerdim)).read(streams=g.SG1[8])
                    distro = g.stack(pow2qb*num_inner_splits*[tqbitdistro[1]], 0).read(streams=g.SG1[4])
                    readaddrs = g.distribute_lowest(tqb, distro, bypass8=0b11110000, distributor_req=0+(4 if self.rev else 0)) #.reinterpret(g.uint32)
                    readaddrpairs = g.distribute_lowest(tqbp, distro, bypass8=0b11110000, distributor_req=1+(4 if self.rev else 0)) #.reinterpret(g.uint32)
                    readaddrs, readaddrpairs = g.split(g.transpose_null(g.stack([readaddrs, readaddrpairs], 1), transposer_req=2 if self.rev else 0, stream_order=[0, 1, 2, 3, 8, 9, 10, 11]), dim=1, num_splits=2)
                    if len(gatesel) != 4 and len(tcqbitsel) == 6:
                        readaddrs = readaddrs.split(dim=0, num_splits=pow2qb*num_inner_splits)
                        readaddrpairs = readaddrpairs.split(dim=0, num_splits=pow2qb*num_inner_splits)
                        readaddrs, readaddrpairs = g.concat_vectors([(readaddrs if (i & (pow2qb*num_inner_splits//4)) == 0 else readaddrpairs)[i] for i in range(pow2qb//2*num_inner_splits)] + readaddrs[pow2qb//2*num_inner_splits:], (pow2qb*num_inner_splits, 1, 4, innerdim)), g.concat_vectors([(readaddrs if (i & (pow2qb*num_inner_splits//4)) == 0 else readaddrpairs)[i] for i in range(pow2qb//2*num_inner_splits)] + readaddrpairs[pow2qb//2*num_inner_splits:], (pow2qb*num_inner_splits, 1, 4, innerdim))
                    readaddrs, readaddrpairs = [x.reshape(pow2qb*num_inner_splits, innerdim) for x in g.split(readaddrs, dim=2, num_splits=4)], [x.reshape(pow2qb*num_inner_splits, innerdim) for x in g.split(readaddrpairs, dim=2, num_splits=4)]                    
                    #s8range
                    us = [g.stack([g.mem_gather(g.split_vectors(g.concat_vectors(x, (pow2qb*num_inner_splits, innerdim)).reinterpret(g.uint8).transpose(1, 0, 2), [pow2qb*num_inner_splits]*4)[j],
                                    *[z if control_qbit is None or i in [0,3] or len(gatesel)==4 else g.split_vectors(z, [pow2qb//2*num_inner_splits]*2)[1] for z in (readaddrs[j] if i<2 else readaddrpairs[j],)], output_streams=[g.SG1[4*(2*i+1)+j]]) for j in range(4)], 1).reinterpret(g.float32)
                            for i, x in enumerate((usplit[:,0], usplit[:,1], ucopysplit[:,0], ucopysplit[:,1]))]
                usb = [[]]*2
                if not control_qbit is None and (gatesel is None or len(gatesel) == 2):
                    for i in [0,3]:
                        usb[i%2], us[i] = g.split_vectors(us[i], [pow2qb//2*num_inner_splits, pow2qb//2*num_inner_splits])
                    #usb = [g.vxm_identity(usb[i], alus=[[rev_alu(13, self.rev),rev_alu(14, self.rev)][i]], time=0, output_streams=g.SG4[[1,7][i]]) for i in range(2)]
                    if derivdistro is None:
                        usb = [g.vxm_identity(usb[i], alus=[[rev_alu(15, self.rev),rev_alu(11, self.rev)][i]], time=0 if tcqbitsel is None or control_qbit is None else None, output_streams=g.SG4[[1,5][i]]) for i in range(2)]
                    else:
                        readddistro = g.concat_vectors([derivdistro.reshape(1, 320)]*(pow2qb//2*num_inner_splits), (pow2qb//2*num_inner_splits, 320)).read(streams=g.SG4[6])
                        usb = [g.mul(usb[i], readddistro, alus=[[rev_alu(15, self.rev),rev_alu(11, self.rev)][i]], time=0 if tcqbitsel is None or control_qbit is None else None, output_streams=g.SG4[[1,5][i]]) for i in range(2)]
            m1 = [g.mul(gs[i], us[i], alus=[[rev_alu(0, self.rev),rev_alu(4, self.rev),rev_alu(8, self.rev),rev_alu(12, self.rev)][i]], output_streams=g.SG4[[0,2,4,6][i]], time=(0 if control_qbit is None else pow2qb*num_inner_splits) if i==0 and (gatesel is None or len(gatesel) != 4) and (tcqbitsel is None or control_qbit is None) else None) for i in range(4)]
            m2 = [g.mul(gs[i], us[i^1], alus=[[rev_alu(2, self.rev),rev_alu(3, self.rev),rev_alu(10, self.rev),rev_alu(11, self.rev)][i]], output_streams=g.SG4[[3,3,5,5][i]]) for i in range(4)]
            a1 = [g.sub(m1[2*i], m1[2*i+1], alus=[[rev_alu(1, self.rev),rev_alu(9, self.rev)][i]], output_streams=g.SG4[[0,6][i]]) for i in range(2)]
            a2 = [g.add(m2[i], m2[2+i], alus=[[rev_alu(5, self.rev),rev_alu(6, self.rev)][i]], output_streams=g.SG4[[4,3][i]]) for i in range(2)]
            ri = [g.add(a1[0], a1[1], alus=[rev_alu(15, self.rev)], output_streams=g.SG4[1]),
                  g.add(a2[0], a2[1], alus=[rev_alu(7, self.rev)], output_streams=g.SG4[5])]
            if tcqbitsel is None:
                ri = g.concat_vectors(np.hstack([np.array(g.split_vectors(ri[i] if control_qbit is None else g.concat_vectors([usb[i], ri[i]], (pow2qb*num_inner_splits, innerdim)), [1]*(pow2qb*num_inner_splits)))[revidx].reshape(pow2qb*num_inner_splits, 1) for i in range(2)]).flatten().tolist(), (pow2qb*num_inner_splits*2, innerdim))
                result = ri.write(name="result", storage_req=self.otherinit[EAST])
                copy = ri.write(name="copy", storage_req=self.copystore[EAST])
            else:
                if len(tcqbitsel) == 6 or len(tcqbitsel) == 8:
                    if len(gatesel) != 4: ri = [g.concat_vectors([usb[i], ri[i]], (pow2qb*num_inner_splits, innerdim)) for i in range(2)]
                    cdistro = g.stack(pow2qb*num_inner_splits*[cqbitdistro[0]], 0).read(streams=g.SG1[16+1])
                    rigap, delay = 0 if len(gatesel) == 4 else 3*2, 4+4+1+4+2 #3 cycle ALU time and transposer time=4, IO crossing time=4, gather time=1, IO crossing time=4, distributor crossing time=2 (1 pre-entry, 1 for distributor operation)
                    tposediff = 4+4+44//4+5+3*3+44//4+4+2 #50, the ALU entry delay is 5 cycles -or- the read delay is 1 cycle + ALU entry delay of 4 cycles, but there is no ALU exit delay 
                    if len(tcqbitsel) == 8:
                        cqbithighsel0 = g.split(cqbithighsel[0], num_splits=2)
                        tchighgather = g.distribute_8(g.concat([cqbithighsel0[0]]*2+[cqbithighsel0[1]]*2, 0).read(streams=g.SG1[24]), g.stack(pow2qb*num_inner_splits*[tcqbitdistro[0]], 0).read(streams=g.SG1[25]), bypass8=0b11111110, distributor_req=3+(0 if self.rev else 4), time=tposediff)
                        #tchighgather = g.transpose_null(tchighgather, transposer_req=3 if self.rev else 1, stream_order=[8], time=0)
                        #tchigh = g.mem_gather(cqbitpairs0[0], tchighgather, output_streams=[g.SG1[16]])
                        lasttensor = [None]
                        def shared_tpose(tensors):
                            if tensors[0] is None and tensors[2] is None:
                                lasttensor[0] = g.transpose_null(tensors[1], transposer_req=1 if self.rev else 3, stream_order=[8], time=0)
                                return None, lasttensor[0], None
                            elif tensors[1] is None and tensors[2] is None:
                                return g.transpose_null(tensors[0], transposer_req=1 if self.rev else 3, stream_order=[4, 5, 6, 7], time=0), None, None
                            if lasttensor[0].shape != tensors[2].shape: lt, lasttensor[0] = g.split(lasttensor[0], splits=[tensors[2].shape[0], lasttensor[0].shape[0] - tensors[2].shape[0]])
                            else: lt, lasttensor[0] = lasttensor[0], None
                            tchigh = g.mem_gather(cqbitpairs0[0], lt, output_streams=[g.SG1[16]]) #, time=-7
                            writecontrols = g.distribute_8(tchigh, tensors[2], bypass8=0b11111110, distributor_req=2+(0 if self.rev else 4)) #, time=-1                                
                            if tensors[0] is None:
                                writecontrols, lt = g.split(g.transpose_null(g.stack((writecontrols, tensors[1]), 1), transposer_req=1 if self.rev else 3, stream_order=[0, 8], time=0), num_splits=2, dim=1)
                                lt = lt.reshape(lt.shape[0], lt.shape[-1])
                                if lasttensor[0] is None: lasttensor[0] = lt
                                else: lasttensor[0] = g.concat((lasttensor[0], lt), 0)
                                return None, lt, writecontrols.reshape(writecontrols.shape[0], writecontrols.shape[-1])                                
                            elif tensors[1] is None:
                                writecontrols, tensors[0] = g.split(g.transpose_null(g.concat([writecontrols.reshape(writecontrols.shape[0], 1, innerdim), tensors[0].reinterpret(g.uint8)], 1), transposer_req=1 if self.rev else 3, stream_order=[0, 4, 5, 6, 7], time=0), dim=1, splits=[1, 4])
                                return tensors[0].reinterpret(g.float32).reshape(tensors[0].shape[0], innerdim), None, writecontrols.reshape(writecontrols.shape[0], innerdim)                                  
                            else:
                                writecontrols, tensors[0], lt = g.split(g.transpose_null(g.concat([writecontrols.reshape(writecontrols.shape[0], 1, innerdim), tensors[0].reinterpret(g.uint8), tensors[1].reshape(tensors[1].shape[0], 1, tensors[1].shape[-1])], 1), transposer_req=1 if self.rev else 3, stream_order=[0, 4, 5, 6, 7, 8], time=0), splits=[1, 4, 1], dim=1)
                                lt = lt.reshape(lt.shape[0], lt.shape[-1])
                                if lasttensor[0] is None: lasttensor[0] = lt
                                else: lasttensor[0] = g.concat((lasttensor[0], lt), 0)
                                return tensors[0].reinterpret(g.float32).reshape(tensors[0].shape[0], innerdim), lt, writecontrols.reshape(writecontrols.shape[0], writecontrols.shape[-1])
                        ri[1], _, writecontrols = UnitarySimulator.transpose_null_share([ri[1], tchighgather, cdistro], [[(delay+delay, delay+delay+pow2qb*num_inner_splits)], [(x, min(x+delay, pow2qb*num_inner_splits)) for x in range(0, pow2qb*num_inner_splits, delay)], [(delay, delay+pow2qb*num_inner_splits)]], {}, 1+tposediff, shared_tpose)
                    else:
                        tchigh = g.stack([cqbitpairs0[0]]*2 + [cqbitpairs1[0]]*2, 0).read(streams=g.SG1[16])
                        writecontrols = g.distribute_8(tchigh, cdistro, bypass8=0b11111110, distributor_req=2 if self.rev else 6)
                        scheduleri = [(delay, delay+pow2qb*num_inner_splits)]
                        schedulewrite = [(0, pow2qb*num_inner_splits)]
                        gaps = {} if pow2qb*num_inner_splits >= delay else {pow2qb*num_inner_splits: delay-pow2qb*num_inner_splits}
                        #gaps = {pow2qb//2*num_inner_splits: rigap, delay+pow2qb//2*num_inner_splits: rigap} if pow2qb//2*num_inner_splits <= delay else {}
                        def shared_tpose(tensors):
                            if tensors[0] is None: return None, g.transpose_null(tensors[1], transposer_req=1 if self.rev else 3, stream_order=[0], time=0)
                            elif tensors[1] is None: return g.transpose_null(tensors[0], transposer_req=1 if self.rev else 3, stream_order=[4, 5, 6, 7], time=0), None
                            tensors[1], tensors[0] = g.split(g.transpose_null(g.concat([tensors[1].reshape(tensors[1].shape[0], 1, innerdim), tensors[0].reinterpret(g.uint8)], 1), transposer_req=1 if self.rev else 3, stream_order=[0, 4, 5, 6, 7], time=0), dim=1, splits=[1, 4])
                            return tensors[0].reinterpret(g.float32).reshape(tensors[0].shape[0], innerdim), tensors[1].reshape(tensors[1].shape[0], innerdim)                    
                        ri[1], writecontrols = UnitarySimulator.transpose_null_share([ri[1], writecontrols], [scheduleri, schedulewrite], gaps, tposediff if len(gatesel)==4 else tposediff-3*2, shared_tpose) #t=51 when gather transpose_null resource scope bases from but we are relative again to parent here
                    #writecontrols, ri[1] = g.split(g.transpose_null(g.concat([writecontrols.reshape(-1, 1, innerdim), ri[1].reinterpret(g.uint8)], 1), transposer_req=1 if self.rev else 3, stream_order=[0, 4, 5, 6, 7]), dim=1, splits=[1, 4])
                    #ri[1] = ri[1].reinterpret(g.float32).reshape(pow2qb*num_inner_splits, innerdim)
                    #writecontrols = writecontrols.reshape(pow2qb*num_inner_splits, innerdim)
                    dist_st = g.distribute_lowest(g.concat_vectors([g.mem_gather((tqbitpairs0 if (i & (pow2qb*num_inner_splits//4)) == 0 else tqbitpairs1)[0], x, output_streams=[g.SG1[0]]) for i, x in enumerate(writecontrols.split_vectors([1]*pow2qb*num_inner_splits))], (pow2qb*num_inner_splits, innerdim)), tqbitdistro[0].read(streams=g.SG1[12]), bypass8=0b11110000, distributor_req=0 if self.rev else 4)
                    #dist_st = g.distribute_lowest(g.mem_gather(g.stack([tqbitpairs0[0], tqbitpairs1[0]], dim=0).reshape(2, pow2qb//4*num_inner_splits, 2, innerdim).transpose(0,2,1,3), writecontrols, output_streams=[g.SG1[8]]), tqbitdistro[0].read(streams=g.SG1[12]), bypass8=0b11110000, distributor_req=1 if self.rev else 5)
                else:
                    dist_st = g.distribute_lowest(g.concat_vectors([tqbitpairs0[0], tqbitpairs1[0]], (pow2qb*num_inner_splits, innerdim)), tqbitdistro[0].read(streams=g.SG1[12]), bypass8=0b11110000, distributor_req=0 if self.rev else 4)
                    ri[1] = g.transpose_null(ri[1], transposer_req=1 if self.rev else 3, stream_order=[4, 5, 6, 7])
                writeaddrs, ri[0] = g.split(g.transpose_null(g.stack([dist_st, ri[0].reinterpret(g.uint8)], 1), transposer_req=0 if self.rev else 2, stream_order=[0, 1, 2, 3, 4, 5, 6, 7]), dim=1, num_splits=2)
                ri[0] = ri[0].reinterpret(g.float32).reshape(pow2qb*num_inner_splits, innerdim)
                result = g.from_addresses(np.array(self.otherinit[EAST].addresses).reshape(-1, g.float32.size), innerdim, g.float32, "result")
                copy = g.from_addresses(np.array(self.copystore[EAST].addresses).reshape(-1, g.float32.size), innerdim, g.float32, "copy")
                writeaddrs = [x.reshape(pow2qb*num_inner_splits, innerdim) for x in g.split(writeaddrs, dim=2, num_splits=4)]
                ri = [[x.reshape(pow2qb*num_inner_splits, innerdim) for x in g.split(ri[i].reinterpret(g.uint8), dim=1, num_splits=4)] for i in range(2)]
                for i in range(2):
                    for j in range(4):
                        g.mem_scatter(ri[i][j], g.split(g.split(copy.reshape(pow2qb*num_inner_splits, 2, innerdim), dim=1, num_splits=2)[i].reinterpret(g.uint8).reshape(pow2qb*num_inner_splits, 4, innerdim), dim=1, num_splits=4)[j], index_tensor=writeaddrs[j])
                        g.mem_scatter(ri[i][j], g.split(g.split(result.reshape(pow2qb*num_inner_splits, 2, innerdim), dim=1, num_splits=2)[i].reinterpret(g.uint8).reshape(pow2qb*num_inner_splits, 4, innerdim), dim=1, num_splits=4)[j], index_tensor=writeaddrs[j])
        return result, copy
    def unpack_broadcast(tensor, distmaps, temp_store, inner_splits, reversedir, width, writefn):
        #720=320+360 extra cost due to shifter cycles = 22 (first read delay)+19*19+17 (final write delay)
        inpstream, outpstream, bypass8, mask_bitmap = (g.SG4[1] if width == 4 else (g.SG2[2] if width == 2 else g.SG1[4]),
            g.SG4[0] if width == 4 else (g.SG2[0] if width == 2 else g.SG1[0]),
                0b11110000 if width == 4 else (0b11111100 if width == 2 else 0b11111110),
                0b0000 if width == 4 else (0b1100 if width == 2 else 0b1110)) 
        pred, cur_mt, outp = None, tensor, []        
        for i in range(20):
            if i != 0:
                with g.ResourceScope(name="shift" + str(i), is_buffered=True, predecessors=[pred], time=None) as pred: #time=16*inner_splits+(16*inner_splits+19*inner_splits)*(i-1)) as pred:
                    cur_mt = g.shift(cur_mt, 16, permutor_id=1 if reversedir else 0, shift_src=[inst.NEW_SRC]*width, dispatch_set=inst.DispatchSet.SET_0, input_streams=inpstream, output_streams=outpstream, time=0).write(name="outshift" + str(i), storage_req=temp_store)
            with g.ResourceScope(name="bcast" + str(i), is_buffered=True, predecessors=None if pred is None else [pred], time=0 if pred is None else None) as pred: #time=(16*inner_splits+19*inner_splits)*i) as pred:
                tobcast = g.distribute_8(g.stack([cur_mt]*16, 0), g.concat([distmaps]*inner_splits, 0), distributor_req=4 if reversedir else 0, bypass8=bypass8, map_stream_req=g.SG1[5])
                writefn(g.broadcast_lane_0(tobcast, permutor_req=1 if reversedir else 0, dispatch_set=inst.DispatchSet.SET_0, old_bitmap=[inst.NEW_SRC]*width, mask_bitmap=mask_bitmap, input_streams=outpstream, output_streams=outpstream, time=0))
    def compute_trace_real(unitary, num_qbits, ident, outp_storage): #using the copy could improve cycles
        pow2qb = 1 << num_qbits
        num_inner_splits = (pow2qb+320-1)//320
        #effective shape is address order so (num_inner_splits, pow2qb, 2, min(256, pow2qb))
        rows = g.concat_vectors([y
            for i, x in enumerate(g.split_vectors(unitary, [pow2qb*2]*num_inner_splits))
            for j, y in enumerate(g.split_vectors(x, [1]*(pow2qb*2)))
                if (j & 1) == 0 and j//2>=i*min(256, pow2qb) and j//2<(i+1)*min(256, pow2qb)], (pow2qb, min(256, pow2qb)))
        with g.ResourceScope(name="mask", is_buffered=True, time=0) as pred:
            rows = g.mul(rows, g.concat_vectors([ident]*num_inner_splits, (pow2qb, min(256, pow2qb))), time=0)
            rows = g.sum(rows, dims=[0])
            rows = rows.write(name="singledim", storage_req=outp_storage)
        #with g.ResourceScope(name="innerred", is_buffered=True, time=None, predecessors=[pred]) as pred:
        #    #rows = g.sum(g.concat_vectors([rows.reshape(pow2qb, min(256, pow2qb)), *([g.zeros((3, min(256, pow2qb)), dtype=g.float32, layout="-1, S12")]*pow2qb)], (4, pow2qb, min(256, pow2qb))).transpose(1,0,2), dims=None, time=0).write(name="trace", layout="-1, S4")
        #    rows = g.sum(g.concat_vectors([rows.reshape(1, min(256, pow2qb)), g.zeros((3, min(256, pow2qb)), dtype=g.float32, layout="-1, S12")], (4, min(256, pow2qb))), dims=[0,1], time=0).write(name="trace", layout="-1, S4, H1(W)")
        return rows
    def build_chain(num_qbits, max_gates, output_unitary=False):
        pow2qb = 1 << num_qbits
        debug = True
        pgm_pkg = g.ProgramPackage(name="us" + ("unit" if output_unitary else "") + str(num_qbits) + "-" + str(max_gates), output_dir="usiop", inspect_raw=debug, gen_vis_data=debug, check_stream_conflicts=debug, check_tensor_timing_conflicts=debug)
        num_inner_splits = (pow2qb+320-1)//320 #handle inner splits for >=9 qbits
        chainsize = 2 #40 if num_qbits == 2 else (10 if num_qbits == 10 else (16 if num_qbits == 9 else 20)) #min(max_gates, int(np.sqrt(6000*max_gates/(pow2qb*num_inner_splits/2)))) #6000*gates/chainsize == chainsize*pow2qb*num_inner_splits/2
        #if (chainsize & 1) != 0: chainsize += 1
        print("Number of qbits:", num_qbits, "Maximum gates:", max_gates, "Chain size:", chainsize)
        with pgm_pkg.create_program_context("init_us") as pcinitunitary:
            unitaryinit = g.input_tensor(shape=(pow2qb*2, pow2qb), dtype=g.float32, name="unitaryinit", layout="-1, H1(W), B1(1), A" + str(pow2qb*num_inner_splits) + "(0-" + str(pow2qb*num_inner_splits-1) + "), S8(0-8)") #get_slice8(WEST, 0, 7, 0)
            identaddr = (max_gates+1)//2*2; derivaddr = identaddr + 2
            gateidentderiv = [[g.from_data(np.repeat(np.eye(2, dtype=np.complex64).view(np.float32).flatten(), 320).reshape(2*2*2, 320), layout="-1, A2(" + str(identaddr) + "-" + str(identaddr+2-1) + "), S16(0-15), B1(0), H1(" + ("W" if hemi == WEST else "E") + ")"),
                g.zeros((2*2*2, 320), g.float32, layout="-1, A2(" + str(derivaddr) + "-" + str(derivaddr+2-1) + "), S16(0-15), B1(0), H1(" + ("W" if hemi == WEST else "E") + ")")] for hemi in (EAST, WEST)]
            identderivaddr = [g.from_data(np.array(([identaddr & 255, identaddr >> 8]+[0]*14)*20, dtype=np.uint8), name="identaddr0", layout=get_slice1(WEST, 0, 0) + ", A1(4085)"),
                              g.from_data(np.array(([(identaddr+1) & 255, (identaddr+1) >> 8]+[0]*14)*20, dtype=np.uint8), name="identaddr1", layout=get_slice1(WEST, 0, 0) + ", A1(4086)"),
                              g.from_data(np.array(([derivaddr & 255, derivaddr >> 8]+[0]*14)*20, dtype=np.uint8), name="derivaddr0", layout=get_slice1(WEST, 1, 0) + ", A1(4085)"),
                              g.from_data(np.array(([(derivaddr+1) & 255, (derivaddr+1) >> 8]+[0]*14)*20, dtype=np.uint8), name="derivaddr1", layout=get_slice1(WEST, 1, 0) + ", A1(4086)")]

            realgatemap = [[g.zeros((320,), g.uint32, layout=get_slice4(hemi, 17, 21, 1), name="realgatemap" + ("W" if hemi==WEST else "E")) for i in range(4)] for hemi in (EAST, WEST)]
            gatemap = [[g.zeros((320,), g.uint8, layout=get_slice1(hemi, 2, 0) + ", A1(4089)", name="gatemap0" + ("W" if hemi==WEST else "E")),
                        g.from_data(np.array(([1]+[0]*15)*20, dtype=np.uint8), layout=get_slice1(hemi, 2, 0) + ", A1(4090)", name="gatemap1" + ("W" if hemi==WEST else "E"))] for hemi in (EAST, WEST)]
            #gatemap = [[g.address_map(g.split_vectors(gates if hemi==EAST else othergates, [4]*((max_gates+1)//2*2))[0], np.array([0]*20), index_map_layout=get_slice4(hemi, 17, 21, 1)),
            #            g.address_map(g.split_vectors(gates if hemi==EAST else othergates, [4]*((max_gates+1)//2*2))[1], np.array([0]*20), index_map_layout=get_slice4(hemi, 17, 21, 1))] for hemi in (EAST, WEST)]
            
            gateinc = [g.from_data(np.array(([1*2]+[0]*15)*20, dtype=np.uint8), layout=get_slice1(hemi, 0, 0) + ", A1(4088)", name="gateinc" + ("W" if hemi==WEST else "E")) for hemi in (EAST, WEST)] #gather map is little endian byte order
            gateinc256 = [g.zeros((320,), layout=get_slice1(hemi, 1, 0) + ", A1(4088)", dtype=g.uint8, name="gateinc256" + ("W" if hemi==WEST else "E")) for hemi in (EAST, WEST)]
            gateinccount = [g.from_data(np.array(([0, 1*2]+[0]*14)*20, dtype=np.uint8), layout=get_slice1(hemi, 2, 0) + ", A1(4088)", name="gateinccount" + ("W" if hemi==WEST else "E")) for hemi in (EAST, WEST)]
            gateincmask = [g.from_data(np.array(([0, 1]+[0]*14)*20, dtype=np.uint8), layout=get_slice1(hemi, 3, 0) + ", A1(4088)", name="gateincmask" + ("W" if hemi==WEST else "E")) for hemi in (EAST, WEST)]

            targetqbitdistro = [[g.zeros((320,), dtype=g.uint8, layout=get_slice1(hemi, 15 if i==1 else 35, 1) + ", A1(4095)", name="targetqbitdistro" + ("W" if hemi==WEST else "E")) for i in range(2)] for hemi in (EAST, WEST)]
            controlqbitdistro = [[g.zeros((320,), dtype=g.uint8, layout=get_slice1(hemi, 14 if i==1 else 34, 1) + ", A1(4095)", name="controlqbitdistro" + ("W" if hemi==WEST else "E")) for i in range(2)] for hemi in (EAST, WEST)]
            if num_qbits >= 9: hightcqdistro = [[g.zeros((320,), dtype=g.uint8, layout=get_slice1(hemi, 13 if i==1 else 33, 1) + ", A1(4095)", name="tcqdistro" + ("W" if hemi==WEST else "E")) for i in range(2)] for hemi in (EAST, WEST)]
            #derivatedistro = g.zeros((320,), dtype=g.float32, layout=get_slice4(WEST, 0, 3, 0) + ", A1(4092)", name="derivatedistro")

            idxmapsort, idxmapm1 = UnitarySimulator.idxmapgather(num_qbits)
            
            idxmapsort = (np.repeat(np.stack(idxmapsort), num_inner_splits, axis=1).reshape(num_qbits, -1, num_inner_splits, 2) + (np.arange(num_inner_splits)*pow2qb).reshape(1, 1, num_inner_splits, 1)).reshape(num_qbits, -1, 2) 
            idxmapm1 = (np.repeat(np.stack(idxmapm1), num_inner_splits, axis=1).reshape(num_qbits-1, -1, num_inner_splits, 2)*num_inner_splits + (np.arange(num_inner_splits)).reshape(1, 1, num_inner_splits, 1)).reshape(num_qbits-1, -1, 2)
            
            idxmapsort = np.stack(((idxmapsort & 255).astype(np.uint8), (idxmapsort >> 8).astype(np.uint8))).transpose(3, 2, 1, 0).reshape(2, -1, num_qbits*2)
            if num_qbits % 8 != 0: idxmapsort = np.concatenate((idxmapsort, np.zeros((2, idxmapsort.shape[-2], 2*(8-num_qbits % 8)), dtype=np.uint8)), axis=2)
            idxmapsort = np.repeat(idxmapsort, 20, axis=1).reshape(2, -1, 320)
            idxmapm1 = np.stack(((idxmapm1 & 255).astype(np.uint8), (idxmapm1 >> 8).astype(np.uint8))).transpose(3, 2, 1, 0).reshape(2, -1, (num_qbits-1)*2)
            if (num_qbits-1) % 8 != 0: idxmapm1 = np.concatenate((idxmapm1, np.zeros((2, idxmapm1.shape[-2], 2*(8-(num_qbits-1) % 8)), dtype=np.uint8)), axis=2)
            idxmapm1 = np.repeat(idxmapm1, 20, axis=1).reshape(2, -1, 320)
            
            idxmapsort = idxmapsort.reshape(2, -1, 20, 2 if num_qbits > 8 else 1, 16).transpose(0, 3, 1, 2, 4).reshape(2, -1, 320)
            idxmapm1 = idxmapm1.reshape(2, -1, 20, 2 if num_qbits > 9 else 1, 16).transpose(0, 3, 1, 2, 4).reshape(2, -1, 320)
            if num_qbits > 8: idxmapm1 = np.stack((idxmapm1, (idxmapm1.reshape(2, 2 if num_qbits > 9 else 1, -1, 20, 8, 2) + np.array((0, pow2qb*num_inner_splits//2//256), dtype=np.uint8)).reshape(2, -1, 320)), axis=1).reshape(2, -1, 320) #must address idxmapsort again for target qbits >=8
            
            tseldim = pow2qb*(2 if num_qbits > 8 else 1)*num_inner_splits//2
            cseldim = pow2qb*(2 if num_qbits > 9 else 1)*(2 if num_qbits > 8 else 1)*num_inner_splits//4
            targetqbitpairs0 = [g.from_data(idxmapsort[0,:], layout=get_slice1(hemi, 43, 0) + ", A" + str(tseldim) + "(0-" + str(tseldim-1) + ")", name="targetqbitpairs0" + ("W" if hemi==WEST else "E")) for hemi in (EAST, WEST)]
            targetqbitpairs1 = [g.from_data(idxmapsort[1,:], layout=get_slice1(hemi, 42, 0) + ", A" + str(tseldim) + "(0-" + str(tseldim-1) + ")", name="targetqbitpairs1" + ("W" if hemi==WEST else "E")) for hemi in (EAST, WEST)]
            controlqbitpairs0 = [g.from_data(idxmapm1[0,:], layout=get_slice1(hemi, 41, 0) + ", A" + str(cseldim) + "(0-" + str(cseldim-1) + ")", name="controlqbitpairs0" + ("W" if hemi==WEST else "E")) for hemi in (EAST, WEST)]
            controlqbitpairs1 = [g.from_data(idxmapm1[1,:], layout=get_slice1(hemi, 41, 1) + ", A" + str(cseldim) + "(0-" + str(cseldim-1) + ")", name="controlqbitpairs1" + ("W" if hemi==WEST else "E")) for hemi in (EAST, WEST)]
            
            if num_qbits >= 9: #selecting from 2 at 9 qbits or 4 at 10 or more qbits possibilities and making a shared distributor mapping
                controlqbithighsel = [g.from_data(np.repeat(np.concatenate((np.concatenate((np.arange(cseldim, dtype=np.uint16).reshape(2 if num_qbits==9 else 4, -1).T.reshape(2 if num_qbits==9 else 4, -1).view(np.uint8).reshape(-1, 4 if num_qbits==9 else 8), np.zeros((cseldim//(2 if num_qbits==9 else 4), 12 if num_qbits==9 else 8), dtype=np.uint8)), axis=1), np.concatenate((np.arange(4096, 4096+cseldim, dtype=np.uint16).reshape(2 if num_qbits==9 else 4, -1).T.reshape(2 if num_qbits==9 else 4, -1).view(np.uint8).reshape(-1, 4 if num_qbits==9 else 8), np.zeros((cseldim//(2 if num_qbits==9 else 4), 12 if num_qbits==9 else 8), dtype=np.uint8)), axis=1))), 20, axis=0).reshape(-1,320), layout=get_slice1(hemi, 37, 1), name="controlqbithighsel" + ("W" if hemi==WEST else "E")) for hemi in (EAST, WEST)]

            qbitinc = g.from_data(np.array(([1]+[0]*15)*20, dtype=np.uint8), layout=get_slice1(WEST, 0, 0) + ", A1(4095)", name="qbitinc") #gather map is little endian byte order
            qbitinc256 = g.zeros((320,), layout=get_slice1(WEST, 1, 0) + ", A1(4094)", dtype=g.uint8, name="qbitinc256")
            qbitinccount = g.from_data(np.array(([0, 1]+[0]*14)*20, dtype=np.uint8), layout=get_slice1(WEST, 2, 0) + ", A1(4093)", name="qbitinccount")

            qbitmap = g.zeros((320,), g.uint8, layout=get_slice1(WEST, 1, 0) + ", A1(4095)", name="qbitmap") #g.address_map(targetqbits, np.array([0]*20), index_map_layout=get_slice1(WEST, 1, 0) + ", A1(4095)")

            resetqbitmaporig = g.zeros((320,), g.uint8, layout=get_slice1(WEST, 2, 0) + ", A1(4094)", name="resetqbitmaporig") #g.address_map(targetqbits, np.array([0]*20), index_map_layout=get_slice1(WEST, 2, 0) + ", A1(4094)")
            resetgatemapsorig = [[g.zeros((320,), g.uint8, layout=get_slice1(hemi, 22, 1), name="resetgatemapsorig0" + ("W" if hemi==WEST else "E")),
                    g.from_data(np.array(([1]+[0]*15)*20, dtype=np.uint8), layout=get_slice1(hemi, 23, 1), name="resetgatemapsorig1" + ("W" if hemi==WEST else "E"))] for hemi in (EAST, WEST)]
            #resetgatemaps = [[g.address_map(g.split_vectors(gates if hemi==EAST else othergates, [4]*((max_gates+1)//2*2))[0].reinterpret(g.uint8).split(dim=1, num_splits=4)[0], np.array([0]*20), index_map_layout=get_slice1(hemi, 22, 1)),
            #        g.address_map(g.split_vectors(gates if hemi==EAST else othergates, [4]*((max_gates+1)//2*2))[1].reinterpret(g.uint8).split(dim=1, num_splits=4)[0], np.array([0]*20), index_map_layout=get_slice1(hemi, 23, 1))] for hemi in (EAST, WEST)]

            onepoint = g.full((320,), 1.0, dtype=g.float32, layout=get_slice4(WEST, 4, 7, 0) + ", A1(4095)", name="onepoint")
            zeropads = [g.zeros((1, 320), dtype=g.uint8, layout=get_slice1(WEST, 17+z, 1) + ", A1(4095)", name="zeropads" + str(z)) for z in range(3)]
            if not output_unitary:
                identmat = g.eye(min(256, pow2qb), dtype=g.float32, layout=get_slice4(WEST, 4, 7, 0).replace(", S4", ", A" + str(min(256, pow2qb)) + "(" + str(4095-min(256, pow2qb)) + "-4094), S4"), name="identmat")
                outptrace = g.zeros((320,), dtype=g.float32, layout=get_slice4(WEST, 0, 3, 0) + ", A1(4091)", name="outptrace")
                g.add_mem_constraints([identmat], [onepoint], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
            resetzerosorig = g.zeros((320,), dtype=g.uint8, layout=get_slice1(WEST, 2, 0) + ", A1(4095)", name="resetzerosorig")
            distmaps = [g.from_data(np.array([[i] + [16]*319 for i in range(16)], dtype=np.uint8), name="distmaps", layout=get_slice1(hemi, 37, 1) + ", A16(4080-4095)") for hemi in (WEST, EAST)]
            lowmask = g.from_data(np.array(([7]*2+[0]*14)*20, dtype=np.uint8), name="lowmask", layout=get_slice1(WEST, 0, 0) + ", A1(4087)")
            midmask = g.from_data(np.array(([0x38]*2+[0]*14)*20, dtype=np.uint8), name="midmask", layout=get_slice1(WEST, 1, 0) + ", A1(4087)")
            shl1 = g.full((320,), 1, name="shl1", dtype=g.uint8, layout=get_slice1(WEST, 4, 0) + ", A1(3839)")
            shr2 = g.full((320,), 2, name="shr2", dtype=g.uint8, layout=get_slice1(WEST, 3, 0) + ", A1(4087)")
            shr6 = g.full((320,), 6, name="shr6", dtype=g.uint8, layout=get_slice1(WEST, 2, 0) + ", A1(4087)")
            adjmap = [g.from_data(np.array(([0, 1] + [16]*14)*20, dtype=np.uint8), name="adjmap", layout=get_slice1(WEST, 5+i, 0) + ", A1(3839)") for i in range(2)]
            
        with pgm_pkg.create_program_context("init_gates") as pcinit:
            us = UnitarySimulator(num_qbits)
            #unitary = g.input_tensor(shape=(pow2qb*2, pow2qb), dtype=g.float32, name="unitary", layout=get_slice8(WEST, s8range[0], s8range[-1], 0))
            #physical shapes for 9, 10, 11 qbits are (2, 1024, 4, (256, 256)), (4, 2048, 4, (256, 256, 256, 256)), (7, 4096, 4, (304, 304, 304, 304, 304, 304, 224))
            g.reserve_tensor(pcinitunitary, pcinit, unitaryinit)
            unitaryinitctxt = g.from_addresses(np.array(unitaryinit.storage_request.addresses.reshape(-1, g.float32.size), dtype=object), min(256, pow2qb), g.float32, "unitaryinitreset")
            unitaryinitctxt = g.concat_vectors([g.concat_inner_splits(g.split_vectors(x, [1]*num_inner_splits)) for x in g.split_vectors(unitaryinitctxt.reshape(num_inner_splits, pow2qb*2, min(256, pow2qb)).transpose(1, 0, 2), [num_inner_splits]*(pow2qb*2))], (pow2qb*2, pow2qb))
            with g.ResourceScope(name="makecopy", is_buffered=True, time=0) as pred:
                unitary, copy, otherunitary, othercopy = us.copymatrix(unitaryinitctxt)
            #gatescomb = g.input_tensor(shape=(max_gates+1)//2*2, 2*2*2, pow2qb), dtype=g.float32, name="gate", layout="-1, H2, S16(" + str(min(slices)) + "-" + str(max(slices)) + ")")
            #gates = g.input_tensor(shape=((max_gates+1)//2, 2*2*2, min(256, pow2qb)), dtype=g.float32, name="gate", layout="-1, A" + str((max_gates+1)//2*2) + "(0-" + str((max_gates+1)//2*2-1) + "), S16(0-15), B1(0), H1(E)") #get_slice16(EAST, list(range(16)), 0)) #, broadcast=True)
            #othergates = g.input_tensor(shape=((max_gates+1)//2, 2*2*2, min(256, pow2qb)), dtype=g.float32, name="othergate", layout="-1, A" + str((max_gates+1)//2*2) + "(0-" + str((max_gates+1)//2*2-1) + "), S16(0-15), B1(0), H1(W)") #get_slice16(WEST, list(range(16)), 0) #, broadcast=True)
            
            num_inner_gates = (2*2*2*((max_gates+1)//2)+320-1)//320 #(num_inner_gates, 320) or (num_inner_gates, 20, 16) order
            num_inner_qbits = (max_gates+320-1)//320
            gatespack = g.input_tensor(shape=(num_inner_gates*320,), dtype=g.float32, name="gate", layout=get_slice4(EAST, 32, 35, 1)) 
            othergatespack = g.input_tensor(shape=(num_inner_gates*320,), dtype=g.float32, name="othergate", layout=get_slice4(WEST, 32, 35, 1))
            dmaps = [tensor.shared_memory_tensor(mem_tensor=distmaps[hemi], name="distmaps" + str(hemi)) for hemi in (WEST, EAST)]
            with g.ResourceScope(name="initgates", is_buffered=True, time=0) as pred:
                for hemi in (WEST, EAST):
                    outp = []
                    def writefn(st):
                        i = len(outp) // num_inner_gates #order will be (20, num_inner_gates, 4, 4)
                        for j, x in enumerate(g.split_inner_splits(st)):
                            outp.append(x.write(name="initgate" + str(i) + str(j), storage_req=tensor.create_storage_request(layout="-1, A4(" + str(i*4+80*j) + "-" + str(i*4+80*j+3) + "), S16(0-15), B1(0), H1(" + ("W" if hemi==WEST else "E") + ")")))
                    temp_store = tensor.create_storage_request(layout=("-1, H1(" + ("W" if hemi==WEST else "E") + "), S4(39,40,42,43), B1(1)" if num_qbits == 10 else get_slice4(hemi, 40, 43, 1)).replace(", S4", ", A" + str(num_inner_gates) + "(" + str(4096-num_inner_gates) + "-" + "4095), S4"))
                    pack = othergatespack if hemi==WEST else gatespack
                    UnitarySimulator.unpack_broadcast(pack, dmaps[hemi], temp_store, num_inner_gates, hemi, 4, writefn)
                    g.add_mem_constraints(outp, outp, g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
                    if hemi == WEST: othergates = g.split(g.concat(outp, 0).reshape(20, num_inner_gates, 16, 320).transpose(1, 0, 2, 3).reshape(20*num_inner_gates*16, 320), [(max_gates+1)//2*2*2*2, num_inner_gates*320-(max_gates+1)//2*2*2*2])[0]
                    else: gates = g.split(g.concat(outp, 0).reshape(20, num_inner_gates, 16, 320).transpose(1, 0, 2, 3).reshape(20*num_inner_gates*16, 320), [(max_gates+1)//2*2*2*2, num_inner_gates*320-(max_gates+1)//2*2*2*2])[0]
            qbitinfo = g.input_tensor(shape=(num_inner_qbits*320,), dtype=g.uint16, name="qbits", layout=get_slice2(WEST, 39, 40, 1))
            with g.ResourceScope(name="initqbits", is_buffered=True, predecessors=[pred], time=None) as pred:
                outpt, outpc, outpd, outph = [], [], [], []
                lmask = tensor.shared_memory_tensor(mem_tensor=lowmask, name=lowmask.name + "init")
                mmask = tensor.shared_memory_tensor(mem_tensor=midmask, name=midmask.name + "init")
                s1 = tensor.shared_memory_tensor(mem_tensor=shl1, name=shl1.name + "init")
                s2 = tensor.shared_memory_tensor(mem_tensor=shr2, name=shr2.name + "init")
                s6 = tensor.shared_memory_tensor(mem_tensor=shr6, name=shr6.name + "init")
                amap = [tensor.shared_memory_tensor(mem_tensor=adjmap[i], name=adjmap[i].name + "init") for i in range(2)]
                def writefn(st):
                    #3 target qbits, 3 control qbits with one subtracted if > target_qbit, 0 if control qbit 
                    #target qbit formula: & 0x7 mask, <<1, +1/16 (3 ALU)
                    #control qbit formula: & 0x3f mask, >>2 (>>3, <<1), +1/16 (3 ALU)
                    #derivate formula: write only (0 for target, 1 for control, 2 for derivate)
                    #hightcqbit formula: >>6 +1/16 (2 ALU)
                    i = len(outpt) // num_inner_qbits
                    for j, x in enumerate(g.split_inner_splits(st)):
                        x = x.reinterpret(g.uint8).split(num_splits=2, dim=-2)
                        with g.ResourceScope(name="innersplitsalus", is_buffered=False, time=0):
                            am = amap[0].read(streams=g.SG1[7*4])
                            outpt.append(x[0].bitwise_and(lmask.read(streams=g.SG1[2*4]), alus=[0], output_streams=g.SG4[2]).left_shift(s1.read(streams=g.SG1[5*4]), alus=[5], output_streams=g.SG4[4]).add(am, alus=[10])
                                .write(name="target_qbits" + str(i) + str(j), storage_req=tensor.create_storage_request(layout=get_slice1(WEST, 37, 0) + ", A16(" + str(i*16+320*j) + "-" + str(i*16+320*j+15) + ")")))
                            outpc.append(x[0].bitwise_and(mmask.read(streams=g.SG1[6*4]), alus=[12], output_streams=g.SG4[6]).right_shift(s2.read(streams=g.SG1[4*4]), alus=[9], output_streams=g.SG4[6]).add(am, alus=[14])
                                .write(name="control_qbits" + str(i) + str(j), storage_req=tensor.create_storage_request(layout=get_slice1(WEST, 36, 0) + ", A16(" + str(i*16+320*j) + "-" + str(i*16+320*j+15) + ")")))
                            outpd.append(x[1].write(name="derivates" + str(i) + str(j), storage_req=tensor.create_storage_request(layout=get_slice1(WEST, 39, 0) + ", A16(" + str(i*16+320*j) + "-" + str(i*16+320*j+15) + ")")))
                            if num_qbits >= 9: outph.append(x[0].right_shift(s6.read(streams=g.SG1[1*4]), alus=[1], output_streams=g.SG4[1]).add(amap[1].read(streams=g.SG1[3*4]), alus=[2])
                                .write(name="high_tcqbits" + str(i) + str(j), storage_req=tensor.create_storage_request(layout=get_slice1(EAST, 36, 0) + ", A16(" + str(i*16+320*j) + "-" + str(i*16+320*j+15) + ")")))
                temp_store = tensor.create_storage_request(layout=get_slice2(hemi, 42, 43, 0).replace(", S2", ", A" + str(num_inner_qbits) + "(" + str(4096-num_inner_qbits) + "-" + "4095), S2"))
                UnitarySimulator.unpack_broadcast(qbitinfo, dmaps[WEST], temp_store, num_inner_qbits, WEST, 2, writefn)
                g.add_mem_constraints(outpt, outpt, g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
                g.add_mem_constraints(outpc, outpc, g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
                g.add_mem_constraints(outpd, outpd, g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
                if num_qbits >= 9: g.add_mem_constraints(outph, outph, g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
                targetqbits = g.split(g.concat(outpt, 0).reshape(20, num_inner_qbits, 16, 320).transpose(1, 0, 2, 3).reshape(20*num_inner_qbits*16, 320), [max_gates, num_inner_qbits*320-max_gates])[0]
                controlqbits = g.split(g.concat(outpc, 0).reshape(20, num_inner_qbits, 16, 320).transpose(1, 0, 2, 3).reshape(20*num_inner_qbits*16, 320), [max_gates, num_inner_qbits*320-max_gates])[0]
                derivates = g.split(g.concat(outpd, 0).reshape(20, num_inner_qbits, 16, 320).transpose(1, 0, 2, 3).reshape(20*num_inner_qbits*16, 320), [max_gates, num_inner_qbits*320-max_gates])[0]
                if num_qbits >= 9: hightcqbits = g.split(g.concat(outph, 0).reshape(20, num_inner_qbits, 16, 320).transpose(1, 0, 2, 3).reshape(20*num_inner_qbits*16, 320), [max_gates, num_inner_qbits*320-max_gates])[0]
            #targetqbits = g.input_tensor(shape=(max_gates, 320), dtype=g.uint8, name="target_qbits", layout=get_slice1(WEST, 37, 0) + ", A" + str(max_gates) + "(0-" + str(max_gates-1) + ")")
            #controlqbits = g.input_tensor(shape=(max_gates, 320), dtype=g.uint8, name="control_qbits", layout=get_slice1(WEST, 36, 0) + ", A" + str(max_gates) + "(0-" + str(max_gates-1) + ")")
            #derivates = g.input_tensor(shape=(max_gates, 320), dtype=g.uint8, name="derivates", layout=get_slice1(WEST, 39, 0) + ", A" + str(max_gates) + "(0-" + str(max_gates-1) + ")")
            #if num_qbits >= 9: hightcqbits = g.input_tensor(shape=(max_gates, 320), dtype=g.uint8, name="high_tcqbits", layout=get_slice1(EAST, 36, 0) + ", A" + str(max_gates) + "(0-" + str(max_gates-1) + ")")

            for x in (gateinc, gateinccount, gateincmask, targetqbitpairs0, targetqbitpairs1, controlqbitpairs0, controlqbitpairs1, zeropads) + ((controlqbithighsel,) if num_qbits >= 9 else ()):
                for y in x: tensor.shared_memory_tensor(mem_tensor=y, name=y.name + "init")
            for x in (qbitinc, qbitinccount, onepoint) + (() if output_unitary else (identmat,)): tensor.shared_memory_tensor(mem_tensor=x, name=x.name + "fin")        
            resetzeros = tensor.shared_memory_tensor(mem_tensor=resetzerosorig, name="resetzeros")
            resetqbitmap = tensor.shared_memory_tensor(mem_tensor=resetqbitmaporig, name="resetqbitmap")
            resetgatemaps = [[tensor.shared_memory_tensor(mem_tensor=resetgatemapsorig[hemi][i], name="resetgatemaps" + str(i) + ("W" if hemi==WEST else "E")) for i in range(2)] for hemi in (EAST, WEST)]
            with g.ResourceScope(name="resetgathercounts", is_buffered=True, time=None, predecessors=[pred]) as pred:
                #must reset ginc256, gatemap, qbitinc256, qbitmap or GFAULTs will occur due to bad addresses gathered/scattered
                tsrs = [resetzeros.read(streams=g.SG1[0], time=1).write(storage_req=qbitinc256.storage_request),
                    resetqbitmap.read(streams=g.SG1[0], time=2).write(storage_req=qbitmap.storage_request)]
                z = resetzeros.read(streams=g.SG1[0], time=0)
                tsrs += [z.write(storage_req=gateinc256[i].storage_request) for i in range(2)]
                tsrs += [g.concat([resetgatemaps[hemi][i].read(streams=g.SG1[0], time=i*4)]*4, 0).write(storage_req=gatemap[hemi][i].storage_request) for hemi in (EAST, WEST) for i in range(2)]
                g.add_mem_constraints(tsrs, tsrs, g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
            g.add_mem_constraints(gateinc + gateinc256 + gateinccount + [resetzeros, resetqbitmap], [gates, othergates, resetzeros, resetqbitmap], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
        
        #for reversedir in (False, True):
        #for target_qbit, control_qbit in ((0, None), (0, 1)) + (((8, None), (8, 1)) if num_qbits >= 9 else ()) + (((0, 9), (8, 9)) if num_qbits >= 10 else ()):
        #for target_qbit, control_qbit in ((0, 1),): # + (((8, 1),) if num_qbits >= 9 else ()) + (((0, 9), (8, 9)) if num_qbits >= 10 else ()):
        suffix = "universal" #("rev" if reversedir else "") + str(target_qbit) + "_" + str(control_qbit)
        with pgm_pkg.create_program_context("us_gate"+suffix) as pc:
            target_qbit, control_qbit = 0, 1
            #if not reversedir and target_qbit == 0 and control_qbit is None: print(gatemap[0].data, gatemap[1].data)            
            g.reserve_tensor(pcinitunitary, pcinit, unitaryinit)
            g.reserve_tensor(pcinit, pc, unitary)
            g.reserve_tensor(pcinit, pc, otherunitary)
            g.reserve_tensor(pcinit, pc, copy)
            g.reserve_tensor(pcinit, pc, othercopy)
            g.reserve_tensor(pcinit, pc, gates)
            g.reserve_tensor(pcinit, pc, othergates)
            g.reserve_tensor(pcinit, pc, targetqbits)
            g.reserve_tensor(pcinit, pc, controlqbits)
            if num_qbits >= 9: g.reserve_tensor(pcinit, pc, hightcqbits)
            g.reserve_tensor(pcinit, pc, derivates)
            onep = tensor.shared_memory_tensor(mem_tensor=onepoint, name="onep"+suffix)
            zs = [tensor.shared_memory_tensor(mem_tensor=zeropads[i], name="zs"+str(i)+suffix) for i in range(len(zeropads))]
            gmap = [[tensor.shared_memory_tensor(mem_tensor=gatemap[reversedir][i], name="gatemap" + str(reversedir) + str(i) + suffix) for i in range(2)] for reversedir in range(2)]
            realgmap = [[tensor.shared_memory_tensor(mem_tensor=realgatemap[reversedir][i], name="realgatemap" + str(reversedir) + str(i) + suffix) for i in range(4)] for reversedir in range(2)]
            ginc = [tensor.shared_memory_tensor(mem_tensor=gateinc[i], name="gateinc" + str(i) + suffix) for i in range(2)]
            ginc256 = [tensor.shared_memory_tensor(mem_tensor=gateinc256[i], name="gateinc256" + str(i) + suffix) for i in range(2)]
            ginccount = [tensor.shared_memory_tensor(mem_tensor=gateinccount[i], name="gateinccount" + str(i) +suffix) for i in range(2)]
            gincmask = [tensor.shared_memory_tensor(mem_tensor=gateincmask[i], name="gateincmask" + str(i) + suffix) for i in range(2)]
            tqbitdistro = [[tensor.shared_memory_tensor(mem_tensor=targetqbitdistro[i][(i+reversedir) % 2], name="tqbitdistro" + str(reversedir) + str(i) + suffix) for i in range(2)] for reversedir in range(2)]
            tqbitpairs0 = [tensor.shared_memory_tensor(mem_tensor=targetqbitpairs0[i], name="tqbitpairs0" + str(i) + suffix) for i in range(2)]
            tqbitpairs1 = [tensor.shared_memory_tensor(mem_tensor=targetqbitpairs1[i], name="tqbitpairs1" + str(i) + suffix) for i in range(2)]
            if not control_qbit is None:
                cqbitdistro = [[tensor.shared_memory_tensor(mem_tensor=controlqbitdistro[i][(i+reversedir) % 2], name="cqbitdistro" + str(reversedir) + str(i) + suffix) for i in range(2)] for reversedir in range(2)]
                cqbitpairs0 = [tensor.shared_memory_tensor(mem_tensor=controlqbitpairs0[i], name="cqbitpairs0" + str(i) + suffix) for i in range(2)]
                cqbitpairs1 = [tensor.shared_memory_tensor(mem_tensor=controlqbitpairs1[i], name="cqbitpairs1" + str(i) + suffix) for i in range(2)]
                if num_qbits >= 9:
                    tcqbitdistro = [[tensor.shared_memory_tensor(mem_tensor=hightcqdistro[i][(i+reversedir) % 2], name="tcqbitdistro" + str(reversedir) + str(i) + suffix) for i in range(2)] for reversedir in range(2)]
                    cqbithighsel = [tensor.shared_memory_tensor(mem_tensor=controlqbithighsel[i], name="controlqbithighsel" + str(i) + suffix) for i in range(2)]
            else:
                for i in range(2):
                    tensor.shared_memory_tensor(mem_tensor=controlqbitpairs0[i], name="cqbitpairs0" + str(i) + suffix)
                    tensor.shared_memory_tensor(mem_tensor=controlqbitpairs1[i], name="cqbitpairs1" + str(i) + suffix)
            for reversedir in range(2):
                g.add_mem_constraints([ginc[reversedir], ginc256[reversedir], ginccount[reversedir]], [othergates if reversedir else gates], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
            
            qmap = tensor.shared_memory_tensor(mem_tensor=qbitmap, name="qmap" + suffix)
            qinc = tensor.shared_memory_tensor(mem_tensor=qbitinc, name="qinc" + suffix)
            qinc256 = tensor.shared_memory_tensor(mem_tensor=qbitinc256, name="qinc256" + suffix)
            qinccount = tensor.shared_memory_tensor(mem_tensor=qbitinccount, name="qinccount" + suffix)

            for x in resetgatemapsorig:
                for y in x: tensor.shared_memory_tensor(mem_tensor=y, name=y.name + "calc")
            for x in (resetqbitmaporig, resetzerosorig) + (() if output_unitary else (identmat,)): tensor.shared_memory_tensor(mem_tensor=x, name=x.name + "calc")
            
            unitaryctxt = [g.from_addresses(np.array((otherunitary if reversedir else unitary).storage_request.addresses.reshape(-1, g.float32.size), dtype=object), 320, g.float32, "unitary" + str(reversedir) + suffix) for reversedir in range(2)]
            copyctxt = [g.from_addresses(np.array((othercopy if reversedir else copy).storage_request.addresses.reshape(-1, g.float32.size), dtype=object), 320, g.float32, "copy" + str(reversedir) + suffix) for reversedir in range(2)]
            gatesctxt = [g.from_addresses(np.array((othergates if reversedir else gates).storage_request.addresses.reshape(-1, g.float32.size), dtype=object), 320, g.float32, "gates" + str(reversedir) + suffix) for reversedir in range(2)]
            tqbits = g.from_addresses(np.array(targetqbits.storage_request.addresses.reshape(-1, g.uint8.size), dtype=object), 320, g.uint8, "targetqbits" + suffix)
            cqbits = g.from_addresses(np.array(controlqbits.storage_request.addresses.reshape(-1, g.uint8.size), dtype=object), 320, g.uint8, "controlqbits" + suffix)
            if num_qbits >= 9: htcqbits = g.from_addresses(np.array(hightcqbits.storage_request.addresses.reshape(-1, g.uint8.size), dtype=object), 320, g.uint8, "hightcqbits" + suffix)
            derivs = g.from_addresses(np.array(derivates.storage_request.addresses.reshape(-1, g.uint8.size), dtype=object), 320, g.uint8, "derivates" + suffix)
            idaddr = [tensor.shared_memory_tensor(mem_tensor=identderivaddr[i], name=identderivaddr[i].name + suffix) for i in range(4)]
            one, two = tensor.shared_memory_tensor(mem_tensor=shl1, name=shl1.name + suffix), tensor.shared_memory_tensor(mem_tensor=shr2, name=shr2.name + suffix)
            pred, reversedir = None, False
            for c in range(2):
                with g.ResourceScope(name="setgatherdistros" + str(c), is_buffered=True, time=0 if pred is None else None, predecessors=None if pred is None else [pred]) as pred:
                    qmapW_st = g.split(g.stack([qmap]*(1+1+(1+1 if not control_qbit is None else 0)+8), 0).read(streams=g.SG1_W[0], time=0), splits=[1, 1] + ([1, 1] if not control_qbit is None else []) + [8])
                    if not control_qbit is None and num_qbits >=9: qmapE_st = g.split(g.stack([qmap]*(1+1), 0).read(streams=g.SG1_E[0], time=1+1+(1+1)+8), splits=[1, 1])
                    for i in range(2):
                        g.mem_gather(tqbits, qmapW_st[0+i], output_streams=[g.SG1_E[1]]).write(name="targetqbitdistro" + str(i) + suffix, storage_req=tqbitdistro[reversedir][i].storage_request)
                        if not control_qbit is None:
                            g.mem_gather(cqbits, qmapW_st[2+i], output_streams=[g.SG1_E[2]]).write(name="controlqbitdistro" + str(i) + suffix, storage_req=cqbitdistro[reversedir][i].storage_request)
                            if num_qbits >= 9: g.mem_gather(htcqbits, qmapE_st[0+i], output_streams=[g.SG1_W[3]]).write(name="hightcqbitdistro" + str(i) + suffix, storage_req=tcqbitdistro[reversedir][i].storage_request)

                    d = g.mem_gather(derivs, qmapW_st[-1], output_streams=[g.SG1_E[12]])
                    updmap = g.split(g.bitwise_xor(g.mask(g.equal(d, g.stack([two]*8, 0).read(streams=g.SG1[2*4]), alus=[5], output_streams=g.SG4_E[4]), g.stack([idaddr[2]]*4+[idaddr[3]]*4, 0).read(streams=g.SG1[7*4]), alus=[10], output_streams=g.SG4[6]).vxm_identity(alus=[13], output_streams=g.SG4[6]),
                        g.bitwise_xor(g.mask(g.equal(d, g.stack([one]*8, 0).read(streams=g.SG1[0]), alus=[0], output_streams=g.SG4_E[0]), g.stack([idaddr[0]]*4+[idaddr[1]]*4, 0), alus=[1], output_streams=g.SG4[2]), 
                        g.mask_bar(d.vxm_identity(alus=[4], output_streams=g.SG4[4]), g.stack([gmap[reversedir][0]]*4+[gmap[reversedir][1]]*4, 0), alus=[9], output_streams=g.SG4[5]), alus=[6], output_streams=g.SG4[5]), alus=[11]), 0, num_splits=2)          
                    #updmap = g.split(g.bitwise_xor(g.mem_gather(derivs, qmapW_st[-1], output_streams=[g.SG1_E[12]]), g.stack([gmap[reversedir][0]]*4+[gmap[reversedir][1]]*4, 0)), 0, num_splits=2)
                    for i in range(2): updmap[i].reinterpret(g.uint32).write(name="realgatemap" + str(i) + suffix, storage_req=realgmap[reversedir][i].storage_request)
                    
                    updmap = g.split(g.stack([gmap[reversedir][0]]*4+[gmap[reversedir][1]]*4, 0).read(streams=g.SG1[8], time=4*i), 0, num_splits=2)
                    for i in range(2): updmap[i].reinterpret(g.uint32).write(name="realgatemap" + str(i+2) + suffix, storage_req=realgmap[reversedir][i+2].storage_request)
                tcmap = [list(reversed(x)) if reversedir else x for x in ((tqbitdistro[reversedir], tqbitpairs0, tqbitpairs1, cqbitdistro[reversedir], cqbitpairs0, cqbitpairs1) + ((tcqbitdistro[reversedir], cqbithighsel) if num_qbits >= 9 else ()) if not control_qbit is None else (tqbitdistro[reversedir], tqbitpairs0, tqbitpairs1))]
                with g.ResourceScope(name="rungate" + str(c), is_buffered=True, time=None, predecessors=[pred]) as pred:
                    newus = UnitarySimulator(num_qbits, reversedir, us)
                    newus.build(unitaryctxt[reversedir], copyctxt[reversedir], target_qbit, control_qbit, gatesctxt[reversedir], realgmap[reversedir], tcmap, None)
                with g.ResourceScope(name="incgate" + str(c), is_buffered=True, time=None, predecessors=[pred]) as pred:
                    updinc = g.stack([ginc256[reversedir]]*2, 0).add(g.stack([ginccount[reversedir]]*len(gmap[reversedir]), 0), time=0, alus=[3 if reversedir else 0], overflow_mode=g.OverflowMode.MODULAR)
                    updmap = g.split(g.stack(gmap[reversedir], 0).add(g.stack([ginc[reversedir]]*2, 0), alus=[7 if reversedir else 4], overflow_mode=g.OverflowMode.MODULAR).add(g.mask_bar(updinc, g.stack([gincmask[reversedir]]*2, 0))), 0, num_splits=2)
                    for i in range(2):
                        updmap[i].write(storage_req=gmap[reversedir][i].storage_request, name="nextgatemap" + str(i) + suffix)
                    g.split(updinc, 0, num_splits=2)[0].vxm_identity().write(storage_req=ginc256[reversedir].storage_request, name="nextgateinc256" + suffix)
                with g.ResourceScope(name="incqbit" + str(c), is_buffered=True, time=None, predecessors=[pred]) as pred:
                    updinc = qinc256.add(qinccount, time=0, alus=[0], overflow_mode=g.OverflowMode.MODULAR)
                    qmap.add(qinc, alus=[4], overflow_mode=g.OverflowMode.MODULAR).add(g.mask_bar(updinc, gincmask[reversedir])).write(storage_req=qmap.storage_request, name="nextqmap" + suffix)
                    updinc.vxm_identity().write(storage_req=qinc256.storage_request, name="nextqinc256" + suffix)
                reversedir = not reversedir
        #must validate all addresses are contiguous, and gather/scatter addresses are all on 0-address alignment by checking storage requests, should likely malloc to avoid
        assert {(x.hemi, x.slice, x.offset) for x in unitary.storage_request.addresses.reshape(-1).tolist()} == {(g.Hemisphere.WEST, x, i) for x in s8range for i in range(pow2qb*num_inner_splits)}
        assert {(x.hemi, x.slice, x.offset) for x in otherunitary.storage_request.addresses.reshape(-1).tolist()} == {(g.Hemisphere.EAST, x, i) for x in s8range for i in range(pow2qb*num_inner_splits)}
        assert {(x.hemi, x.slice, x.offset) for x in copy.storage_request.addresses.reshape(-1).tolist()} == {(g.Hemisphere.WEST, x, i) for x in s8range2 for i in range(pow2qb*num_inner_splits)}
        assert {(x.hemi, x.slice, x.offset) for x in othercopy.storage_request.addresses.reshape(-1).tolist()} == {(g.Hemisphere.EAST, x, i) for x in s8range2 for i in range(pow2qb*num_inner_splits)}
        
        assert {(x.hemi, x.slice, x.offset) for x in targetqbits.storage_request.addresses.reshape(-1).tolist()} == {(g.Hemisphere.WEST, x, i) for x in (37,) for i in range(max_gates)}
        assert {(x.hemi, x.slice, x.offset) for x in controlqbits.storage_request.addresses.reshape(-1).tolist()} == {(g.Hemisphere.WEST, x, i) for x in (36,) for i in range(max_gates)}
        assert {(x.hemi, x.slice, x.offset) for x in derivates.storage_request.addresses.reshape(-1).tolist()} == {(g.Hemisphere.WEST, x, i) for x in (39,) for i in range(max_gates)}
        assert {(x.hemi, x.slice, x.offset) for x in gates.storage_request.addresses.reshape(-1).tolist()} == {(g.Hemisphere.EAST, x, i) for x in range(16) for i in range((max_gates+1)//2*2)}
        assert {(x.hemi, x.slice, x.offset) for x in othergates.storage_request.addresses.reshape(-1).tolist()} == {(g.Hemisphere.WEST, x, i) for x in range(16) for i in range((max_gates+1)//2*2)}
        for i, hemi in enumerate((EAST, WEST)):
            assert {(x.hemi, x.slice, x.offset) for x in targetqbitpairs0[i].storage_request.addresses.reshape(-1).tolist()} == {(g.Hemisphere.WEST if hemi==WEST else g.Hemisphere.EAST, x, i) for x in (43,) for i in range(pow2qb//2*num_inner_splits*(2 if num_qbits > 8 else 1))}
            assert {(x.hemi, x.slice, x.offset) for x in targetqbitpairs1[i].storage_request.addresses.reshape(-1).tolist()} == {(g.Hemisphere.WEST if hemi==WEST else g.Hemisphere.EAST, x, i) for x in (42,) for i in range(pow2qb//2*num_inner_splits*(2 if num_qbits > 8 else 1))}
        
        #we return in raw address format, so the inner splits will be on the outer dimension!
        with pgm_pkg.create_program_context("final_us") as pcfinal:
            g.reserve_tensor(pcinitunitary, pcfinal, unitaryinit)
            g.reserve_tensor(pcinit, pcfinal, unitary)
            for x in (gateinc, gateinccount, gateincmask, targetqbitpairs0, targetqbitpairs1, controlqbitpairs0, controlqbitpairs1, zeropads):
                for y in x: tensor.shared_memory_tensor(mem_tensor=y, name=y.name + "fin")
            for x in resetgatemapsorig:
                for y in x: tensor.shared_memory_tensor(mem_tensor=y, name=y.name + "fin")
            for x in (qbitinc, qbitinccount, resetqbitmaporig, onepoint, resetzerosorig): tensor.shared_memory_tensor(mem_tensor=x, name=x.name + "fin")        
            unitaryres = g.from_addresses(np.array(unitary.storage_request.addresses.reshape(-1, g.float32.size), dtype=object), min(256, pow2qb), g.float32, "unitaryfin")
            if not output_unitary:
                unitaryres = UnitarySimulator.compute_trace_real(unitaryres, num_qbits,
                    tensor.shared_memory_tensor(mem_tensor=identmat, name="ident"),
                    tensor.shared_memory_tensor(mem_tensor=outptrace, name="outp").storage_request)
            unitaryres.set_program_output()
        """
        with pgm_pkg.create_program_context("finalrev_us") as pcfinal:
            g.reserve_tensor(pcinitunitary, pcfinal, unitaryinit)
            g.reserve_tensor(pcinit, pcfinal, otherunitary)
            for x in (gateinc, gateinccount, gateincmask, targetqbitpairs0, targetqbitpairs1, controlqbitpairs0, controlqbitpairs1, zeropads):
                for y in x: tensor.shared_memory_tensor(mem_tensor=y, name=y.name + "revfin")
            for x in resetgatemapsorig:
                for y in x: tensor.shared_memory_tensor(mem_tensor=y, name=y.name + "revfin")
            for x in (qbitinc, qbitinccount, resetqbitmaporig, onepoint, resetzerosorig): tensor.shared_memory_tensor(mem_tensor=x, name=x.name + "revfin")        
            unitaryrevres = g.from_addresses(np.array(otherunitary.storage_request.addresses.reshape(-1, g.float32.size), dtype=object), min(256, pow2qb), g.float32, "unitaryrevfin")
            if not output_unitary:
                unitaryrevres = UnitarySimulator.compute_trace_real(unitaryrevres, num_qbits,
                    tensor.shared_memory_tensor(mem_tensor=identmat, name="ident"), 
                    tensor.shared_memory_tensor(mem_tensor=outptrace, name="outp").storage_request) 
            unitaryrevres.set_program_output()
        """
        print_utils.infoc("\nAssembling model ...")
        iops = pgm_pkg.assemble(auto_agt_dim=3, skip_assembler=chainsize!=2)
        pgm_pkg = g.ProgramPackage(name="us" + ("unit" if output_unitary else "") + str(num_qbits) + "-" + str(max_gates), output_dir="usiop", inspect_raw=debug, gen_vis_data=debug, check_stream_conflicts=debug, check_tensor_timing_conflicts=debug)
        if chainsize != 2:
            import shutil
            for name in ("init_us", "init_gates", "us_gateuniversal", "final_us"):
                shutil.copyfile("usiop/topo_0/" + name + "/" + name + ".aa", "usiop/topo_0/" + name + "/" + name + ".0.aa")
            chain_aa("usiop/topo_0/us_gateuniversal/us_gateuniversal.0.aa", chainsize // 2)
            with pgm_pkg.create_program_context("init_us") as pcinitunitary:
                pgm_pkg.add_precompiled_program(pcinitunitary, "usiop/topo_0/init_us", "init_us")
            with pgm_pkg.create_program_context("init_gates") as pcinit:
                pgm_pkg.add_precompiled_program(pcinit, "usiop/topo_0/init_gates", "init_gates")
            with pgm_pkg.create_program_context("us_gate"+suffix) as pc:
                pgm_pkg.add_precompiled_program(pc, "usiop/topo_0/us_gateuniversal", "us_gateuniversal")
            with pgm_pkg.create_program_context("final_us") as pcfinal:
                pgm_pkg.add_precompiled_program(pcfinal, "usiop/topo_0/final_us", "final_us")
            print_utils.infoc("\nAssembling chained model ...")
            iops = pgm_pkg.assemble(auto_agt_dim=3)
        return {"iop": iops[0], "chainsize": chainsize, "max_gates": max_gates, "unitary": unitaryinit.name, "gates": gatespack.name, "othergates": othergatespack.name,
            "qbits": qbitinfo.name,
            #"targetqbits": targetqbits.name, "controlqbits": controlqbits.name, "derivates": derivates.name,
            "unitaryres": unitaryres.name} #, **({"hightcqbits" : hightcqbits.name} if num_qbits >= 9 else {})} #"unitaryrevres": unitaryrevres.name, 
    def build_all(max_levels, if_exists=False, output_unitary=False):
        import os, pickle
        if not if_exists and os.path.exists("usiop/usdata"):
            with open("usiop/usdata", 'rb') as f:
                d = pickle.load(f)
        else: d = {}
        for num_qbits in range(2, 10+1):
            max_gates = get_max_gates(num_qbits, max_levels)
            if not (num_qbits, max_gates, output_unitary) in d:
                d[(num_qbits, max_gates, output_unitary)] = UnitarySimulator.build_chain(num_qbits, max_gates, output_unitary=output_unitary)
                with open("usiop/usdata", 'wb') as f:
                    pickle.dump(d, f)
        return d
    def get_unitary_sim(num_qbits, max_gates, tensornames=None, output_unitary=False):
        pow2qb = 1 << num_qbits
        if tensornames is None: tensornames = UnitarySimulator.build_chain(num_qbits, max_gates, output_unitary)
        print(num_qbits, max_gates, tensornames)
        iop = runtime.IOProgram(tensornames["iop"])
        driver = runtime.Driver()
        device = driver.next_available_device()
        result = [None]
        import contextlib
        with contextlib.ExitStack() as exitstack:
            device_ = exitstack.enter_context(device)
            def closedevice(): exitstack.close()
            runfunc = [None]
            def loaddata():
                #for i in range(1+1+(2+(2 if num_qbits >= 9 else 0)+(2 if num_qbits >= 10 else 0))*2+2):
                #    device.load(iop[i], unsafe_keep_entry_points=True)
                device.load_all(iop, unsafe_keep_entry_points=True)
                num_inner_splits = (pow2qb+320-1)//320
                def actual(u, num_qbits, parameters, target_qbits, control_qbits):
                    num_gates = len(parameters)
                    padgates = 0 if (num_gates % tensornames["chainsize"]) == 0 else tensornames["chainsize"] - (num_gates % tensornames["chainsize"])
                    gateparams = [make_u3(parameters[i,:]) if target_qbits[i] == control_qbits[i] else make_cry(parameters[i,:]) for i in range(num_gates)] + [np.eye(2, dtype=np.complex128)]*padgates
                    num_qbits = (max_gates+320-1)//320*320
                    padqbits = 0 if (num_gates % num_qbits) == 0 else num_qbits - (num_gates % num_qbits)
                    target_qbits = np.concatenate((target_qbits, np.zeros(padqbits, dtype=target_qbits.dtype)))
                    control_qbits = np.concatenate((control_qbits, np.zeros(padqbits, dtype=control_qbits.dtype)))
                    num_gates += padgates
                    inputs = {}
                    inputs[tensornames["unitary"]] = np.ascontiguousarray(u.astype(np.complex64)).view(np.float32).reshape(pow2qb, pow2qb, 2).transpose(0, 2, 1).reshape(pow2qb*2, pow2qb)
                    invoke([device], iop, 0, 0, [inputs])
                    inputs = {}
                    #inputs[tensornames["gates"]] = np.concatenate([np.repeat(gateparams[i].astype(np.complex64).view(np.float32).flatten(), min(256, pow2qb)) for i in range(0, num_gates, 2)] + [np.zeros((2*2*2*min(256, pow2qb)), dtype=np.float32)]*((max_gates+1)//2-(num_gates-num_gates//2)))
                    inputs[tensornames["gates"]] = np.concatenate([gateparams[i].astype(np.complex64).view(np.float32).flatten() for i in range(0, num_gates, 2)] + [np.zeros((2*2*2), dtype=np.float32)]*((max_gates+1)//2-(num_gates-num_gates//2)))
                    #inputs[tensornames["othergates"]] = np.concatenate([np.repeat(gateparams[i].astype(np.complex64).view(np.float32).flatten(), min(256, pow2qb)) for i in range(1, num_gates, 2)] + [np.zeros((2*2*2*min(256, pow2qb)), dtype=np.float32)]*((max_gates+1)//2-num_gates//2))
                    inputs[tensornames["othergates"]] = np.concatenate([gateparams[i].astype(np.complex64).view(np.float32).flatten() for i in range(1, num_gates, 2)] + [np.zeros((2*2*2), dtype=np.float32)]*((max_gates+1)//2-num_gates//2))
                    #inputs[tensornames["targetqbits"]] = np.concatenate((np.repeat(np.hstack((target_qbits.astype(np.uint8)[:,np.newaxis]%8*2, target_qbits.astype(np.uint8)[:,np.newaxis]%8*2+1, np.array([[16]*14]*num_gates, dtype=np.uint8))), 20, axis=0).reshape(-1, 320), np.zeros((max_gates-num_gates, 320), dtype=np.uint8)))
                    adjcontrolqbits = np.where(control_qbits==target_qbits, 0, (control_qbits - (control_qbits > target_qbits)).astype(np.uint8))
                    deriv = control_qbits!=target_qbits
                    inputs[tensornames["qbits"]] = (target_qbits & 7) | ((adjcontrolqbits & 7) << 3)
                    if num_qbits == 9: inputs[tensornames["qbits"]] |= ((target_qbits>>3)<<6)
                    elif num_qbits == 10: inputs[tensornames["qbits"]] |= ((adjcontrolqbits>>3)<<6) | ((target_qbits>>3)<<7)
                    inputs[tensornames["qbits"]] = inputs[tensornames["qbits"]].astype(np.uint16) | (deriv.astype(np.uint16) << 8)     
                    #inputs[tensornames["controlqbits"]] = np.concatenate((np.repeat(np.hstack((adjcontrolqbits[:,np.newaxis]%8*2, adjcontrolqbits[:,np.newaxis]%8*2+1, np.array([[16]*14]*num_gates, dtype=np.uint8))), 20, axis=0).reshape(-1, 320), np.zeros((max_gates-num_gates, 320), dtype=np.uint8)))
                    #if num_qbits >= 9:
                    #    hightcq = (adjcontrolqbits//8 + (target_qbits//8)*2).astype(np.uint8) if num_qbits==10 else (target_qbits//8).astype(np.uint8)
                    #    inputs[tensornames["hightcqbits"]] = np.concatenate((np.repeat(np.hstack((hightcq[:,np.newaxis]*2, hightcq[:,np.newaxis]*2+1, np.array([[16]*14]*num_gates, dtype=np.uint8))), 20, axis=0).reshape(-1, 320), np.zeros((max_gates-num_gates, 320), dtype=np.uint8)))
                    #derivs = np.array([0 if target_qbits[i]==control_qbits[i] else (i//2*2) ^ ((max_gates+1)//2*2) for i in range(num_gates)], dtype=np.uint16)
                    #inputs[tensornames["derivates"]] = np.concatenate((np.repeat(np.hstack(((derivs & 255).astype(np.uint8)[:,np.newaxis], (derivs >> 8).astype(np.uint8)[:,np.newaxis], np.array([[0]*14]*num_gates, dtype=np.uint8))), 20, axis=0).reshape(-1, 320), np.zeros((max_gates-num_gates, 320), dtype=np.uint8)))                    
                    #inputs[tensornames["derivates"]] = np.zeros((max_gates, 320), dtype=np.uint8)
                    invoke([device], iop, 1, 0, [inputs])
                    for i in range(0, num_gates, tensornames["chainsize"]):
                        #progidx = int(1+1+(2+(2 if num_qbits >= 9 else 0)+(2 if num_qbits >= 10 else 0) if (i&1)!=0 else 0) + target_qbits[i]//8*2 + (0 if target_qbits[i] == control_qbits[i] else 1+(2+(target_qbits[i]//8==0))*(adjcontrolqbits[i]//8)))
                        #progidx = int(1+1+(1+(1 if num_qbits >= 9 else 0)+(2 if num_qbits >= 10 else 0) if (i&1)!=0 else 0) + target_qbits[i]//8 + (0 if target_qbits[i] == control_qbits[i] else 2*(adjcontrolqbits[i]//8)))
                        progidx = 1+1
                        #np.set_printoptions(threshold=sys.maxsize, formatter={'int':hex})
                        invoke([device], iop, progidx, 0, None, None, None)
                    progidx = 1+1+1 #1+1+(1+(1 if num_qbits >= 9 else 0)+(2 if num_qbits >= 10 else 0))*2+(num_gates&1) #1+1+(2+(2 if num_qbits >= 9 else 0)+(2 if num_qbits >= 10 else 0))*2+(num_gates&1)
                    res, _ = invoke([device], iop, progidx, 0, None, None, None)
                    if output_unitary:
                        result[0] = np.ascontiguousarray(res[0][tensornames["unitaryres" if (num_gates&1)==0 else "unitaryrevres"]].reshape(num_inner_splits, pow2qb, 2, min(256, pow2qb)).transpose(1, 0, 3, 2)).view(np.complex64).reshape(pow2qb, pow2qb).astype(np.complex128)
                    else:
                        result[0] = np.sum(res[0][tensornames["unitaryres" if (num_gates&1)==0 else "unitaryrevres"]])
                runfunc[0] = actual
        loaddata()
        actual = runfunc[0]
        return actual, result, closedevice
    def validate_alus():
        for alu in range(16):
            shape = (1, 320)
            with g.ProgramContext() as pc:
                t1 = g.input_tensor(shape, g.float32, name="t1")
                t2 = g.input_tensor(shape, g.float32, name="t2")
                result_st = t1.sub(t2, alus=[alu], time=0)
                result_mt = result_st.write(name="result", program_output=True)        
                compiled_iop, _ = compile_unit_test("validate_alus")        
            # Generate random input data and oracle for comparision.
            inp1 = np.random.random(size=t1.shape).astype(np.float32)
            inp2 = np.random.random(size=t2.shape).astype(np.float32)
            runner = tsp.create_tsp_runner(compiled_iop)
            inputs = {t1.name: inp1, t2.name: inp2}
            results = runner(**inputs)
            actual = results[result_mt.name]
            oracle = inp1 - inp2
            #assert np.allclose(results, oracle)
            max_atol = max(abs(oracle.reshape(-1) - actual.reshape(-1)))
            if max_atol == 0:
                print_utils.success(f"Test PASSED with a max tolerance of {max_atol}")
            else:
                print_utils.err(
                    f"Test FAILED with a max tolerance of {max_atol} (should be = 0)"
                )
    def unit_test(num_qbits):
        pow2qb = 1 << num_qbits
        use_identity = False
        target_qbit = np.random.randint(num_qbits)
        control_qbit = np.random.randint(num_qbits)
        if target_qbit == control_qbit: control_qbit = None
        #print(target_qbit, control_qbit)
        with g.ProgramContext() as pc:
            us = UnitarySimulator(num_qbits)
            unitary = g.input_tensor(shape=(pow2qb*2, pow2qb), dtype=g.float32, name="unitary", layout=get_slice8(WEST, 0, 7, 0))
            gate = g.input_tensor(shape=(1, 2*2*2, pow2qb), dtype=g.float32, name="gates", layout=get_slice16(EAST, list(range(16)), 0))
            output, _ = us.build(unitary, None, target_qbit, control_qbit, gate)
            output.set_program_output()
            iop_file, json_file = compile_unit_test("usunit")
        runner = tsp.create_tsp_runner(iop_file)
        u = np.eye(pow2qb) + 0j if use_identity else unitary_group.rvs(pow2qb)
        parameters = np.random.random(3)
        gateparams = make_u3(parameters) if control_qbit is None else make_cry(parameters)
        print_utils.infoc("\nRunning on HW ...")
        oracleres, result = [None], [None]
        def oracle():
            oracleres[0] = process_gates32(u, num_qbits, parameters.reshape(1, 3), np.array([target_qbit], dtype=np.uint8), np.array([control_qbit], dtype=np.uint8))
            #oracleres[0] = qiskit_oracle(u, num_qbits, parameters.reshape(1, 3), np.array([target_qbit]), np.array([control_qbit]))
        def actual():
            inputs = {}
            inputs[unitary.name] = np.ascontiguousarray(u.astype(np.complex64)).view(np.float32).reshape(pow2qb, pow2qb, 2).transpose(0, 2, 1).reshape(pow2qb*2, pow2qb)
            inputs[gate.name] = np.repeat(gateparams.astype(np.complex64).view(np.float32).flatten(), pow2qb)
            res = runner(**inputs)
            result[0] = np.ascontiguousarray(res[output.name].reshape(pow2qb, 2, pow2qb).transpose(0, 2, 1)).view(np.complex64).reshape(pow2qb, pow2qb).astype(np.complex128)
        oracle()
        actual()
        oracleres, result = oracleres[0], result[0]
        np.set_printoptions(formatter={'int':hex, 'complexfloat':lambda x:float(np.real(x)).hex()+'+'+float(np.imag(x)).hex()+'j'}, threshold=sys.maxsize, floatmode='unique')
        if not np.array_equal(oracleres, result): print(oracleres - result, oracleres, result, u)
        if np.allclose(result, oracleres):
            print_utils.success("\nQuantum Simulator Unit Test Success ...")
        else:
            print_utils.err("\nQuantum Simulator Unit Test Failure")
            print_utils.infoc(str(oracleres - result))
        
    def chain_test(num_qbits, max_gates, output_unitary=False):
        pow2qb = 1 << num_qbits
        num_gates, use_identity = max_gates, False

        u = np.eye(pow2qb) + 0j if use_identity else unitary_group.rvs(pow2qb)
        target_qbits = np.array([np.random.randint(num_qbits) for _ in range(num_gates)], dtype=np.uint8)
        control_qbits = np.array([np.random.randint(num_qbits) for _ in range(num_gates)], dtype=np.uint8)
        parameters = np.random.random((num_gates, 3))
        oracleres = [None]
        def oracle():
            oracleres[0] = process_gates32(u, num_qbits, parameters, target_qbits, control_qbits)
            #oracleres[0] = qiskit_oracle(u, num_qbits, parameters, target_qbits, control_qbits)
            if not output_unitary: oracleres[0] = np.trace(np.real(oracleres[0]))
        actual, result, closefunc = UnitarySimulator.get_unitary_sim(num_qbits, max_gates, output_unitary=output_unitary)
        oracle()
        actual(u, num_qbits, parameters, target_qbits, control_qbits)
        closefunc()
        oracleres, result = oracleres[0], result[0]
        #np.set_printoptions(formatter={'int':hex, 'complexfloat':lambda x:float(np.real(x)).hex()+'+'+float(np.imag(x)).hex()+'j'}, threshold=sys.maxsize, floatmode='unique')
        if not np.array_equal(oracleres, result): print(oracleres - result, oracleres, result, u)
        if np.allclose(result, oracleres):
            print_utils.success("\nQuantum Simulator Chain Test Success ...")
        else:
            print_utils.err("\nQuantum Simulator Chain Test Failure")
            #print(result[:10,:10], oracleres[:10,:10], result[-10:,-10:], oracleres[-10:,-10:])
            #print_utils.infoc(str(oracleres[~np.isclose(result, oracleres)]) + " " + str(result[~np.isclose(result, oracleres)]))
            print_utils.infoc(str(abs(oracleres[~np.isclose(result, oracleres)] - result[~np.isclose(result, oracleres)]) / abs(oracleres[~np.isclose(result, oracleres)])))
        #print(oracleres, result)
    def checkacc():
        #2 qbits works to chain 50, 3 to 22
        use_identity, max_levels = False, 6
        output_unitary = True
        d = UnitarySimulator.build_all(max_levels, output_unitary=output_unitary)
        acc, acc32 = {}, {}
        for num_qbits in range(2, 10+1):
            max_gates = get_max_gates(num_qbits, max_levels)
            pow2qb = 1 << num_qbits
            func, result, closefunc = UnitarySimulator.get_unitary_sim(num_qbits, max_gates, d[(num_qbits, max_gates, output_unitary)], output_unitary=output_unitary)
            u = np.eye(pow2qb) + 0j if use_identity else unitary_group.rvs(pow2qb)
            num_gates = 1
            parameters = np.random.random((num_gates, 3))
            for i in range(num_qbits):
                for j in range(num_qbits):
                    #if i == j: continue
                    func(u, num_qbits, parameters, np.array([i]*num_gates, dtype=np.uint8), np.array([j]*num_gates, dtype=np.uint8))
                    oracle = process_gates(u, num_qbits, parameters, np.array([i]*num_gates, dtype=np.uint8), np.array([j]*num_gates, dtype=np.uint8))
                    if not output_unitary: oracle = np.trace(np.real(oracle))
                    if not np.allclose(result[0], oracle): print("Fail", num_qbits, i, j, result[0][~np.isclose(result[0], oracle)], oracle[~np.isclose(result[0], oracle)])
            acc[num_qbits] = {}; acc32[num_qbits] = {}
            for num_gates in (max_gates,):#range(1, max_gates):
                print(num_gates)
                parameters = np.random.random((num_gates, 3))
                target_qbits = np.array([np.random.randint(num_qbits) for _ in range(num_gates)])
                control_qbits = np.array([np.random.randint(num_qbits) for _ in range(num_gates)])
                func(u, num_qbits, parameters, target_qbits, control_qbits)
                oracle = process_gates(u, num_qbits, parameters, target_qbits, control_qbits)
                if not output_unitary: oracle = np.trace(np.real(oracle))
                if not np.allclose(result[0], oracle): print("Fail", num_qbits, num_gates)
                acc[num_qbits][num_gates] = (np.amax(np.abs(oracle-result[0])), np.amax(np.abs(oracle-result[0])/np.abs(oracle)))
                oracle32 = process_gates32(u, num_qbits, parameters, target_qbits, control_qbits)
                if not output_unitary: oracle32 = np.trace(np.real(oracle32))
                acc32[num_qbits][num_gates] = (np.amax(np.abs(oracle-oracle32)), np.amax(np.abs(oracle-oracle32)/np.abs(oracle)))
            if not closefunc is None: closefunc()
        import matplotlib.pyplot as plt
        for i in range(2):
            fig, ax = plt.subplots()
            ax.set_title("Unitary Simulator Accuracy")
            ax.set(xlabel="# of gates", ylabel="Accuracy")
            for num_qbits in acc:
                max_gates = get_max_gates(num_qbits, max_levels)
                r = (max_gates,) #list(range(1, max_gates))
                ax.plot(r, [acc[num_qbits][j][i] for j in r], label="Groq " + ("Absolute" if i == 0 else "Relative") + " " + str(num_qbits) + " qbits")
                ax.plot(r, [acc32[num_qbits][j][i] for j in r], label="CPU float32 " + ("Absolute" if i == 0 else "Relative") + " " + str(num_qbits) + " qbits")
            ax.legend()
            fig.savefig("us_acc" + ("abs" if i==0 else "rel") + ".svg", format='svg')
        print(acc)
    def perfcompare():
        import timeit
        max_levels, batch_size = 6, 20
        output_unitary = False
        d = UnitarySimulator.build_all(max_levels, output_unitary=output_unitary)
        use_identity, max_levels = False, 6
        initfuncs = {"Groq": lambda nqb, mg: UnitarySimulator.get_unitary_sim(nqb, mg, d[(nqb, mg, output_unitary)])}
        testfuncs = {"Groq": None, "numpy": process_gates} #, "qiskit": qiskit_oracle}
        times, accuracy = {k: {} for k in testfuncs}, {k: {} for k in testfuncs}
        inittimesize = {"Groq": {}}
        for num_qbits in range(2, 10+1):
            max_gates = get_max_gates(num_qbits, max_levels)
            num_gates = max_gates
            pow2qb = 1 << num_qbits
            u = np.eye(pow2qb) + 0j if use_identity else unitary_group.rvs(pow2qb)
            target_qbits = np.array([np.random.randint(num_qbits) for _ in range(num_gates)])
            control_qbits = np.array([np.random.randint(num_qbits) for _ in range(num_gates)])
            parameters = np.random.random((num_gates, 3))
            for testfunc in testfuncs:
                if testfunc in initfuncs:
                    initres = [None]
                    def ifunc():
                        initres[0] = initfuncs[testfunc](num_qbits, max_gates)
                    t = timeit.timeit(ifunc, number=1) 
                    func, result, closefunc = initres[0]
                    inittimesize[testfunc][(num_qbits, max_gates)] = (t, os.path.getsize(d[(num_qbits, max_gates, output_unitary)]["iop"]))
                else:
                    result = [None]
                    def tf(u, n, p, t, c):
                        result[0] = testfuncs[testfunc](u, n, p, t, c)
                    func, closefunc = tf, None
                times[testfunc][num_qbits] = timeit.timeit(lambda: func(u, num_qbits, parameters, target_qbits, control_qbits), number=batch_size) / batch_size
                if not output_unitary: result[0] = np.trace(np.real(result[0]))
                accuracy[testfunc][num_qbits] = result[0] 
                print(testfunc, num_qbits, times[testfunc][num_qbits], accuracy[testfunc][num_qbits])
                if not closefunc is None: closefunc()
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_title("Unitary Simulator Performance (Average over Batch=" + str(batch_size) + ")")
        ax.set(xlabel="# of qbits", ylabel="Time (seconds)")
        for x in times:
            ax.plot(times[x].keys(), times[x].values(), label=x)
        ax.legend()
        fig.savefig("us_time.svg", format='svg')
        print(times, accuracy, inittimesize)
def get_max_gates(num_qbits, max_levels):
    max_gates = num_qbits+3*(num_qbits*(num_qbits-1)//2*max_levels)
    if (max_gates % 80) != 0: max_gates += (80 - max_gates % 80)
    return max_gates
def chain_aa(aafile, chainsize):
    with open(aafile, "r") as f:
        lines = f.readlines()
    import re #first determine the total cycles for the period size
    period = 0
    for line in lines:
        if len(line) == 0 or line[0] == '\n' or line[0] == '.' or line[:2] == '//': continue #ignore empty lines, directives and comments
        period = max(period, int(line[0:line.index(':')])) #time format with colon and 4 spaces "\d+:    "
    period += 1
    print("Atom Assembly Total Cycles", period)
    with open(aafile, "w") as f:
        duplines = []
        for line in lines:
            if len(line) == 0 or line[0] == '\n' or line[0] == '.' or line[:2] == '//':
                for c in range(1, chainsize):
                    for ln in duplines:
                        colon = ln.index(':')
                        v = str(int(ln[0:colon])+period*c)
                        f.write(v + ln[colon:])
                f.write(line)
                duplines = []
                continue
            duplines.append(line)
            f.write(line)
def main():
    max_levels=6
    UnitarySimulator.build_all(max_levels, output_unitary=False)
    #UnitarySimulator.build_all(max_levels, output_unitary=True)
    #test()
    #UnitarySimulator.distrib_depend()
    #[UnitarySimulator.idxmapgather(x) for x in range(10)]; assert False
    #import math; [(1<<x)*int(math.ceil((1<<x)/320)) for x in range(12)]
    #[1, 2, 4, 8, 16, 32, 64, 128, 256, 1024, 4096, 14336]
    #10 qbits max for single bank, 11 qbits requires dual chips [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 7, 26, 104]
    #import math; [math.ceil(((1<<x)*int(math.ceil((1<<x)/320)))/8192) for x in range(15)]
    #UnitarySimulator.validate_alus()
    num_qbits = 9
    #UnitarySimulator.unit_test(num_qbits)
    UnitarySimulator.chain_test(num_qbits, get_max_gates(num_qbits, max_levels), True)
    #UnitarySimulator.checkacc()
    #UnitarySimulator.perfcompare()
if __name__ == "__main__":
    main()
