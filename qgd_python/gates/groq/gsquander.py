import sys, os

import numpy as np
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

def qiskit_oracle(unitary, qbit_num, gates):
    from qiskit import Aer
    from qiskit import QuantumCircuit, execute
    backend = Aer.get_backend('unitary_simulator')
    circuit = QuantumCircuit(qbit_num)
    circuit.unitary(unitary, [i for i in range(qbit_num)])
    for op, target_qbit, control_qbit, parameters in gates:
        if op:
            circuit.u(parameters[0]*2, parameters[1], parameters[2], target_qbit)
        else:
            circuit.cry(parameters[0]*2, control_qbit, target_qbit)
    job = execute(circuit, backend)
    result=job.result()
    U3_qiskit = result.get_unitary(circuit)
    U3_qiskit = np.asarray(U3_qiskit)
    return U3_qiskit
def make_u3(parameters):
    return np.array(
        [[np.cos(parameters[0]*2/2), -np.exp(parameters[2]*1j)*np.sin(parameters[0]*2/2)],
         [np.exp(parameters[1]*1j)*np.sin(parameters[0]*2/2), np.exp((parameters[1]+parameters[2])*1j)*np.cos(parameters[0]*2/2)]])
def make_ry(parameters):
    return make_u3([parameters[0], 0, 0])
    #return np.array(
    #    [[np.cos(parameters[0]*2/2), -np.sin(parameters[0]*2/2)],
    #     [np.sin(parameters[0]*2/2), np.cos(parameters[0]*2/2)]])
def make_controlled(gate):
    return np.block([[np.eye(2), np.zeros((2, 2))], [np.zeros((2, 2)), gate]]) #[np.ix_(*([[0,2,1,3]]*2))]
def make_cry(parameters):
    return make_ry(parameters) #make_controlled(make_ry(parameters))
def apply_to_qbit(unitary, num_qbits, target_qbit, control_qbit, gate):
    pow2qb = 1 << num_qbits
    t = np.arange(num_qbits)
    if not control_qbit is None:
        t[:-1] = np.roll(t[:-1], (target_qbit - control_qbit) % num_qbits)
        gate = make_controlled(gate)
    t = np.roll(t, -target_qbit)
    idxs = np.arange(pow2qb).reshape(*([2]*num_qbits)).transpose(t).flatten().tolist()
    return np.kron(np.eye(pow2qb>>(1 if control_qbit is None else 2)), gate)[np.ix_(idxs, idxs)] @ unitary
def apply_to_qbit_loop(unitary, num_qbits, target_qbit, control_qbit, gate):
    pow2qb = 1 << num_qbits
    t = np.roll(np.arange(num_qbits), target_qbit)
    idxs = np.arange(pow2qb).reshape(*([2]*num_qbits)).transpose(t).reshape(-1, 2)
    for pair in (idxs if control_qbit is None else idxs[(idxs[:,0] & (1<<control_qbit)) != 0,:]).tolist():
        unitary[pair,:] = gate @ unitary[pair,:]
    return unitary
def process_gates(unitary, num_qbits, gates):
    unitary = np.copy(unitary)
    for op, target_qbit, control_qbit, parameters in gates:
        unitary = apply_to_qbit(unitary, num_qbits, target_qbit, control_qbit, make_u3(parameters) if op else make_cry(parameters))
    return unitary
def test():
    num_qbits, use_identity = 5, False
    pi = np.pi; parameters = np.array( [pi/2*0.32, pi*1.2, pi/2*0.89])
    pow2qb = 1 << num_qbits
    unitary = np.eye(pow2qb) + 0j if use_identity else unitary_group.rvs(pow2qb)
    for i in range(num_qbits):
        for j in range(num_qbits):
            if i == j: continue
            gates = [(True, i, None, parameters), (True, (i+1)%num_qbits, None, parameters),
                (False, i, j, parameters)
                ]
            actual, oracle = qiskit_oracle(unitary, num_qbits, gates), process_gates(unitary, num_qbits, gates)
            assert np.allclose(actual, oracle), (i, j, actual, oracle)
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
        base_name=name, gen_vis_data=True, check_stream_conflicts=True, #tree_conflicts=True, inspect_raw=True
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
        offsetMap.append(offsetMapEntry)]
    return np.asarray(offsetMap, dtype=np.uint8)
def to_graphviz(g, labels=None, ranks=None, constraints=None, reverse_constraints=None):
    return ("digraph {" + ";".join(str(x)+"->"+str(y) + ("[constraint=false]" if not constraints is None and (not x in constraints or not y in constraints[x]) else "")
        + (";"+str(y)+"->"+str(x) + "[style=invis]" if not reverse_constraints is None and x in reverse_constraints and y in reverse_constraints[x] else "") for x in g for y in g[x]) + ";" +
        ("" if labels is None else ";".join(str(x) + "[label=\"" + labels[x] + "\"]" for x in labels) + ";") +
        ("" if ranks is None else "".join("{rank=same; " + "; ".join(str(x) for x in rank) + ";}" for rank in ranks)) +
        "}")
def gate_op_finder():
    h, w = {}, {}
    for i in range(8): #entries are mtx: [a+bi e+fi] gate; [c+di g+hi] and (a+bi)*(c+di)=(ac-bd)+(cb+da)i
        h[i] = [8+i//4*4+i%4, 8+i//4*4+[3,2,0,1][i%4]]
    for i in range(8):
        h[8+i] = [8+8+i//2]
        w[8+i] = 1
    for i in range(6): #6 adders, last 2 adders are exits
        h[8+8+i] = [] if i >= 4 else [8+8+4+i%2]
        w[8+8+i] = 1
    print(to_graphviz(h, labels={**{i: ("u" if i%4<2 else "g") + "_" + ("r" if i&1==0 else "i") + "_" + str(i//4) for i in range(8)},
            **{8+i: "g.mul" for i in range(8)}, **{8+8+i: "g.add" if not i in [0,2] else "g.sub" for i in range(6)}}))
    return groq_alu_finder(h, w)
def groq_alu_finder(h, w):
    g, weight = {}, {} #construct the Groq VXM graph
    for hemi in (WEST, EAST):
        for stage in range(8):
            for sg in range(8): #stream group
                g[hemi*8*8+stage*8+sg] = [] if stage==8-1 else [hemi*8*8+(stage+1)*8+sg]
                if (stage & 1) == 0: g[hemi*8*8+stage*8+sg].extend([2*8*8+(sg//4)*8 + (3-stage//2 if hemi else stage//2), 2*8*8+((sg-2)%8//4)*8+4 + (3-stage//2 if hemi else stage//2)])                
    for alu in range(16):
        g[2*8*8+alu] = [hemi*8*8+(2*((alu % 4))+1 if hemi else 2*(alu % 4)+1)*8 + (((alu // 4) * 2 + i) % 8) for i in range(4) for hemi in (WEST, EAST)]
        weight[2*8*8+alu] = 1+((alu % 8) in [0,5,2,7]) #2 for large ALU, 1 otherwise, vxm_identity is a special effectively 0 weight use case during traversal
    entries = list(range(8)) + list(range(8*8, 8*8+8))
    exits = list(range(8*7, 8*8)) + list(range(8*(8+7), 8*8*2))
    print(g, weight, entries, exits)    
    ranks = [list(range(8*stage, 8*(stage+1)))+list(range(8*(8+7-stage), 8*(8+8-stage))) for stage in range(8)] + [[2*8*8+j*4+i for j in range(4)] for i in range(4)]
    ranks = [ranks[i] for i in [0, 8, 1, 2, 9, 3, 4, 10, 5, 6, 11, 7]]
    print(to_graphviz(g, labels={**{hemi*8*8+stage*8+sg: "SG1[" + str(sg) + "]_" + ("W" if hemi==WEST else "E") + "@" + str(stage) for sg in range(8) for stage in range(8) for hemi in (EAST, WEST)}, **{2*8*8+alu: "ALU" + str(alu) for alu in range(16)}}, ranks=ranks,
        constraints={x: set(g[x]) & set(ranks[i+1]) for i in range(len(ranks)-1) for x in ranks[i]},
        reverse_constraints={x: set(g[x]) & set(ranks[i-1]) for i in range(1, len(ranks)) for x in ranks[i]}))
class UnitarySimulator(g.Component):
    def __init__(self, num_qbits, reversedir=False, lastus=None, **kwargs):
        super().__init__(**kwargs)
        self.num_qbits, self.rev = num_qbits, reversedir
        self.otherinit = (lastus.uinit if reversedir else lastus.otherinit) if not lastus is None else tensor.create_storage_request(layout=get_slice8(EAST, s8range[0], s8range[-1], 0))
        #more efficient to just directly copy on the controlled rotation rather than doing unnecessary identity gate computations
        #self.identity2x2 = [g.from_data(np.zeros((1), dtype=np.float32), layout=get_slice1(WEST, 0, 0), broadcast=True),
        #                    g.from_data(np.ones((1), dtype=np.float32), layout=get_slice1(WEST, 0, 0), broadcast=True)]
        self.copystore = []
        for hemi in (EAST, WEST) if reversedir else (WEST, EAST):
            self.copystore.append(lastus.copystore[hemi] if not lastus is None else tensor.create_storage_request(layout=get_slice8(hemi, s8range2[0], s8range2[-1], 0)))
    def copymatrix(self, unitary):
        resultother = self.create_memory_tensor(name="result", storage_req=self.otherinit, tensor_type=unitary.tensor_type)
        copy = unitary.read(streams=g.SG8[0]).write(name="initcopy", storage_req=self.copystore[WEST], time=0)
        copyother = self.create_memory_tensor(name="copy", storage_req=self.copystore[EAST], tensor_type=copy.tensor_type)    
        return copy, resultother, copyother
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
    def build(self, unitary, copy, target_qbit, control_qbit, gate, gatesel=None, tcqbitsel=None, inittime=0):
        if copy is None:
            with g.ResourceScope(name="initcopy", is_buffered=True, time=0) as pred:
                copy, _, _ = self.copymatrix(unitary)
        else: pred = None
        pow2qb = 1 << self.num_qbits
        innerdim = pow2qb if gatesel is None else ((pow2qb+320-1)//320)*320
        usplit = np.array(g.split_vectors(unitary, [1] * (2*pow2qb))).reshape(pow2qb, 2)
        ucopysplit = np.array(g.split_vectors(copy, [1] * (2*pow2qb))).reshape(pow2qb, 2)        
        if tcqbitsel is None:
            pairs, bypasspairs = UnitarySimulator.idxmap(self.num_qbits, target_qbit, control_qbit)
            u = [usplit[pairs[:,0],0], usplit[pairs[:,0],1], ucopysplit[pairs[:,1],0], ucopysplit[pairs[:,1],1]]
            ub = [np.array([])]*4 if control_qbit is None else [usplit[bypasspairs[:,0],0], usplit[bypasspairs[:,0],1], ucopysplit[bypasspairs[:,1],0], ucopysplit[bypasspairs[:,1],1]]
            revidx = np.argsort((pairs if control_qbit is None else np.hstack([bypasspairs, pairs])).transpose().flatten()).tolist()         
        r = 1 if control_qbit is None else 2
        with g.ResourceScope(name="rungate", is_buffered=True, time=0 if pred is None else None, predecessors=None if pred is None else [pred]) as pred:
            #(a+bi)*(c+di)=(ac-bd)+(ad+bc)i
            #gate[0] * p[0] - gate[1] * p[1] + gate[2] * p[2] + gate[3] * p[3]
            #gate[0] * p[1] + gate[1] * p[0] + gate[2] * p[3] + gate[3] * p[2]
            if gatesel is None:
                gatevals = g.split_vectors(gate, [1]*(2*2*2))
                gs = [g.concat_vectors([gatevals[i]]*(pow2qb//2//r)+[gatevals[i+4]]*(pow2qb//2//r), (pow2qb//r, pow2qb)).read(streams=g.SG4[2*i]) for i in range(4)] #, time=0 if i == 0 else None                
            else:
                #gate = g.from_addresses(np.array(gate.addrs).reshape(-1, g.float32.size), pow2qb, g.float32, "gatedim")
                gatevals = np.array(g.split_vectors(gate, [1]*(gate.shape[0]))).reshape(gate.shape[0]//8, 2*2*2)
                gatesel_st = g.concat_vectors([gatesel[0].reshape(1,innerdim)]*(pow2qb//2//r)+[gatesel[1].reshape(1,innerdim)]*(pow2qb//2//r), (pow2qb//r, innerdim)).read(streams=g.SG4[1])
                gs = [g.mem_gather(g.concat_vectors(gatevals[:,i].tolist()*(pow2qb//2//r)+gatevals[:,i+4].tolist()*(pow2qb//2//r), (gate.shape[0]//8, pow2qb//r, innerdim)),
                                    gatesel_st, output_streams=[g.SG4[2*i]]) for i in range(4)]
                #gs = [g.mem_gather(g.concat_vectors(gatevals[:,i].tolist()*1+gatevals[:,i+4].tolist()*1, (gate.shape[0]//8, 2, innerdim)),
                #                    gatesel_st, output_streams=[g.SG4[2*i]]) for i in range(4)]
                #gatesel_st = g.split_vectors(g.concat_vectors([gatesel[0].reshape(1,innerdim)]*(pow2qb//2//r)+[gatesel[1].reshape(1,innerdim)]*(pow2qb//2//r), (pow2qb//r, innerdim)).read(streams=g.SG4[1]), [1]*(pow2qb//r))
                #gs = [g.concat_vectors([g.mem_gather(g.concat_vectors(gatevals[:,i+k*4].tolist(), (gate.shape[0]//8, innerdim)), gatesel_st[j+k*pow2qb//2//r], output_streams=[g.SG4[2*i]]) for j in range(pow2qb//2//r) for k in range(2)], (pow2qb//r, innerdim)) for i in range(4)]
            with g.ResourceScope(name="ident", is_buffered=False, time=0) as innerpred:                
                if tcqbitsel is None:
                    us = [g.concat_vectors((ub[i%2].flatten().tolist() + ub[i%2+2].flatten().tolist() if i in [0,3] else []) + u[i].flatten().tolist()*2, (pow2qb if control_qbit is None or i in [0,3] else pow2qb//2, innerdim)).read(streams=g.SG4[2*i+1]) for i in range(4)]
                else:
                    if len(tcqbitsel) == 6:
                        tqbitdistro, tqbitpairs0, tqbitpairs1, cqbitdistro, cqbitpairs0, cqbitpairs1 = tcqbitsel
                        cdistro = g.stack(pow2qb*[cqbitdistro[1]], 0).read(streams=g.SG1[16+4])
                        readcontrols = g.distribute_8(g.concat_vectors([cqbitpairs1[1], cqbitpairs1[1], cqbitpairs0[1], cqbitpairs0[1]], (pow2qb, innerdim)).read(streams=g.SG1[16]), cdistro, bypass8=0b11111110, distributor_req=2+(4 if self.rev else 0))
                        readcontrols = g.transpose_null(readcontrols, transposer_req=3 if self.rev else 1, stream_order=[0], time=0)                    
                        tqb = g.mem_gather(g.concat_vectors([tqbitpairs0[1], tqbitpairs0[1]], (pow2qb, innerdim)), readcontrols, output_streams=[g.SG1[0]])
                        tqbp = g.mem_gather(g.concat_vectors([tqbitpairs1[1], tqbitpairs1[1]], (pow2qb, innerdim)), readcontrols, output_streams=[g.SG1[8]])
                    else:
                        tqbitdistro, tqbitpairs0, tqbitpairs1 = tcqbitsel
                        tqb = g.concat_vectors([tqbitpairs0[1], tqbitpairs0[1]], (pow2qb, innerdim)).read(streams=g.SG1[0])
                        tqbp = g.concat_vectors([tqbitpairs1[1], tqbitpairs1[1]], (pow2qb, innerdim)).read(streams=g.SG1[8])
                    distro = g.stack(pow2qb*[tqbitdistro[1]], 0).read(streams=g.SG1[4])
                    readaddrs = g.distribute_lowest(tqb, distro, bypass8=0b11110000, distributor_req=0+(4 if self.rev else 0)) #.reinterpret(g.uint32)
                    readaddrpairs = g.distribute_lowest(tqbp, distro, bypass8=0b11110000, distributor_req=1+(4 if self.rev else 0)) #.reinterpret(g.uint32)
                    readaddrs, readaddrpairs = g.split(g.transpose_null(g.stack([readaddrs, readaddrpairs], 1), transposer_req=2 if self.rev else 0, stream_order=[0, 1, 2, 3, 8, 9, 10, 11]), dim=1, num_splits=2)
                    readaddrs, readaddrpairs = [x.reshape(pow2qb, innerdim) for x in g.split(readaddrs, dim=2, num_splits=4)], [x.reshape(pow2qb, innerdim) for x in g.split(readaddrpairs, dim=2, num_splits=4)]
                    #s8range
                    us = [g.stack([g.mem_gather(g.split_vectors(g.concat_vectors(x, (pow2qb, innerdim)).reinterpret(g.uint8).transpose(1, 0, 2), [pow2qb]*4)[j],
                                    *[z if control_qbit is None or i in [0,3] else g.split_vectors(z, [pow2qb//2]*2)[1] for z in (readaddrs[j] if i<2 else readaddrpairs[j],)], output_streams=[g.SG1[4*(2*i+1)+j]]) for j in range(4)], 1).reinterpret(g.float32)
                            for i, x in enumerate((usplit[:,0], usplit[:,1], ucopysplit[:,0], ucopysplit[:,1]))]
                usb = [[]]*2
                if not control_qbit is None:
                    for i in [0,3]:
                        usb[i%2], us[i] = g.split_vectors(us[i], [pow2qb//2, pow2qb//2])
                    #usb = [g.vxm_identity(usb[i], alus=[[rev_alu(13, self.rev),rev_alu(14, self.rev)][i]], time=0, output_streams=g.SG4[[1,7][i]]) for i in range(2)]
                    usb = [g.vxm_identity(usb[i], alus=[[rev_alu(15, self.rev),rev_alu(11, self.rev)][i]], time=0 if tcqbitsel is None or control_qbit is None else None, output_streams=g.SG4[[1,5][i]]) for i in range(2)]
            m1 = [g.mul(gs[i], us[i], alus=[[rev_alu(0, self.rev),rev_alu(4, self.rev),rev_alu(8, self.rev),rev_alu(12, self.rev)][i]], output_streams=g.SG4[[0,2,4,6][i]], time=(0 if control_qbit is None else pow2qb) if i==0 and (tcqbitsel is None or control_qbit is None) else None) for i in range(4)]
            m2 = [g.mul(gs[i], us[i^1], alus=[[rev_alu(2, self.rev),rev_alu(3, self.rev),rev_alu(10, self.rev),rev_alu(11, self.rev)][i]], output_streams=g.SG4[[3,3,5,5][i]]) for i in range(4)]
            a1 = [g.sub(m1[2*i], m1[2*i+1], alus=[[rev_alu(1, self.rev),rev_alu(9, self.rev)][i]], output_streams=g.SG4[[0,6][i]]) for i in range(2)]
            a2 = [g.add(m2[i], m2[2+i], alus=[[rev_alu(5, self.rev),rev_alu(6, self.rev)][i]], output_streams=g.SG4[[4,3][i]]) for i in range(2)]
            ri = [g.add(a1[0], a1[1], alus=[rev_alu(15, self.rev)], output_streams=g.SG4[1]),
                  g.add(a2[0], a2[1], alus=[rev_alu(7, self.rev)], output_streams=g.SG4[5])]
            if tcqbitsel is None:
                ri = g.concat_vectors(np.hstack([np.array(g.split_vectors(ri[i] if control_qbit is None else g.concat_vectors([usb[i], ri[i]], (pow2qb, innerdim)), [1]*(pow2qb)))[revidx].reshape(pow2qb, 1) for i in range(2)]).flatten().tolist(), (pow2qb*2, innerdim))
                result = ri.write(name="result", storage_req=self.otherinit)
                copy = ri.write(name="copy", storage_req=self.copystore[EAST])
            else:
                ri = [ri[i] if control_qbit is None else g.concat_vectors([usb[i], ri[i]], (pow2qb, innerdim)) for i in range(2)]
                if len(tcqbitsel) == 6:
                    tqbitdistro, tqbitpairs0, tqbitpairs1, cqbitdistro, cqbitpairs0, cqbitpairs1 = tcqbitsel
                    cdistro = g.stack(pow2qb*[cqbitdistro[0]], 0).read(streams=g.SG1[16+12])
                    writecontrols = g.distribute_8(g.concat_vectors([cqbitpairs1[0], cqbitpairs0[0], cqbitpairs1[0], cqbitpairs0[0]], (pow2qb, innerdim)).read(streams=g.SG1[16+8]), cdistro, bypass8=0b11111110, distributor_req=3 if self.rev else 7)
                    delay = 18 #1+4+1+2+4+1+4+1
                    ri[1] = g.concat_vectors([g.zeros((delay, innerdim), dtype=ri[1].dtype).read(streams=g.SG4[5]), ri[1]], (delay + pow2qb, innerdim))
                    writecontrols = g.concat_vectors([writecontrols, g.zeros((delay, innerdim), dtype=writecontrols.dtype).read(streams=g.SG1[8])], (pow2qb + delay, innerdim))
                    ri[1], writecontrols = g.split(g.transpose_null(g.concat([ri[1].reinterpret(g.uint8), writecontrols.reshape(pow2qb+delay, 1, innerdim)], 1), transposer_req=1 if self.rev else 3, stream_order=[4, 5, 6, 7, 8]), dim=1, splits=[4, 1])
                    #writecontrols = g.transpose_null(writecontrols.reshape(pow2qb+delay, innerdim), transposer_req=1 if self.rev else 3, stream_order=[8])
                    print(ri[1], writecontrols)
                    ri[1] = ri[1].reinterpret(g.float32).reshape(delay + pow2qb, innerdim)
                    ri[1] = g.split_vectors(ri[1], [delay, pow2qb])[1]
                    writecontrols = g.split_vectors(writecontrols, [pow2qb, delay])[0]
                    dist_st = g.distribute_lowest(g.mem_gather(g.concat_vectors([tqbitpairs0[0], tqbitpairs1[0]], (pow2qb, innerdim)), writecontrols.reshape(pow2qb, innerdim), output_streams=[g.SG1[8]]), tqbitdistro[0].read(streams=g.SG1[12]), bypass8=0b11110000, distributor_req=1 if self.rev else 5)
                else:
                    tqbitdistro, tqbitpairs0, tqbitpairs1 = tcqbitsel
                    dist_st = g.distribute_lowest(g.concat_vectors([tqbitpairs0[0], tqbitpairs1[0]], (pow2qb, innerdim)), tqbitdistro[0].read(streams=g.SG1[12]), bypass8=0b11110000, distributor_req=1 if self.rev else 5)
                    ri[1] = g.transpose_null(ri[1], transposer_req=1 if self.rev else 3, stream_order=[4, 5, 6, 7])
                ri[0], writeaddrs = g.split(g.transpose_null(g.stack([ri[0].reinterpret(g.uint8), dist_st], 1), transposer_req=0 if self.rev else 2, stream_order=[4, 5, 6, 7, 8, 9, 10, 11]), dim=1, num_splits=2)
                ri[0] = ri[0].reinterpret(g.float32).reshape(pow2qb, innerdim)
                result = g.from_addresses(np.array(self.otherinit.addresses).reshape(-1, g.float32.size), innerdim, g.float32, "result")
                copy = g.from_addresses(np.array(self.copystore[EAST].addresses).reshape(-1, g.float32.size), innerdim, g.float32, "copy")
                writeaddrs = [x.reshape(pow2qb, innerdim) for x in g.split(writeaddrs, dim=2, num_splits=4)]
                ri = [[x.reshape(pow2qb, innerdim) for x in g.split(ri[i].reinterpret(g.uint8), dim=1, num_splits=4)] for i in range(2)]
                for i in range(2 if control_qbit is None else 1):
                    for j in range(4):
                        g.mem_scatter(ri[i][j], g.split(g.split(copy.reshape(pow2qb, 2, innerdim), dim=1, num_splits=2)[i].reinterpret(g.uint8).reshape(pow2qb, 4, innerdim), dim=1, num_splits=4)[j], index_tensor=writeaddrs[j])
                        g.mem_scatter(ri[i][j], g.split(g.split(result.reshape(pow2qb, 2, innerdim), dim=1, num_splits=2)[i].reinterpret(g.uint8).reshape(pow2qb, 4, innerdim), dim=1, num_splits=4)[j], index_tensor=writeaddrs[j])
        return result, copy
    def unit_test(num_qbits):
        pow2qb = 1 << num_qbits
        use_identity = False
        target_qbit = np.random.randint(num_qbits)
        control_qbit = np.random.randint(num_qbits)
        if target_qbit == control_qbit: control_qbit = None
        #print(target_qbit, control_qbit)
        with g.ProgramContext() as pc:
            us = UnitarySimulator(num_qbits)
            unitary = g.input_tensor(shape=(pow2qb*2, pow2qb), dtype=g.float32, name="unitary", layout=get_slice8(WEST, s8range[0], s8range[-1], 0))
            gate = g.input_tensor(shape=(1, 2*2*2, pow2qb), dtype=g.float32, name="gates", layout=get_slice16(EAST, list(range(16)), 0))#, broadcast=True)
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
            oracleres[0] = qiskit_oracle(u, num_qbits, [(control_qbit is None, target_qbit, control_qbit, parameters)])
        def actual():
            inputs = {}
            inputs[unitary.name] = np.ascontiguousarray(u.astype(np.complex64)).view(np.float32).reshape(pow2qb, pow2qb, 2).transpose(0, 2, 1).reshape(pow2qb*2, pow2qb)
            inputs[gate.name] = np.repeat(gateparams.astype(np.complex64).view(np.float32).flatten(), pow2qb)
            res = runner(**inputs)
            result[0] = np.ascontiguousarray(res[output.name].reshape(pow2qb, 2, pow2qb).transpose(0, 2, 1)).view(np.complex64).reshape(pow2qb, pow2qb).astype(np.complex128)
        oracle()
        actual()
        oracleres, result = oracleres[0], result[0]
        np.set_printoptions(formatter={'int':hex}, threshold=sys.maxsize, floatmode='unique')
        if np.allclose(oracleres, result):
            print_utils.success("\nQuantum Simulator Unit Test Success ...")
        else:
            print_utils.err("\nQuantum Simulator Unit Test Failure")
            print_utils.infoc(str(oracleres - result))
        
    def chain_test(num_qbits, max_gates):
        num_gates, use_identity = max_gates, True
        pow2qb = 1 << num_qbits
        pgm_pkg = g.ProgramPackage(name="us", output_dir="us", gen_vis_data=True)
        print("Number of gates:", num_gates, "Maximum gates:", max_gates)
        with pgm_pkg.create_program_context("init_us") as pcinit:
            us = UnitarySimulator(num_qbits)
            unitary = g.input_tensor(shape=(pow2qb*2, pow2qb), dtype=g.float32, name="unitary", layout=get_slice8(WEST, s8range[0], s8range[-1], 0))
            us.uinit = unitary.storage_request
            with g.ResourceScope(name="makecopy", is_buffered=True, time=0) as pred:
                copy, otherunitary, othercopy = us.copymatrix(unitary)
            #gatescomb = g.input_tensor(shape=(max_gates+1)//2*2, 2*2*2, pow2qb), dtype=g.float32, name="gate", layout="-1, H2, S16(" + str(min(slices)) + "-" + str(max(slices)) + ")")
            gates = g.input_tensor(shape=((max_gates+1)//2, 2*2*2, pow2qb), dtype=g.float32, name="gate", layout=get_slice16(EAST, list(range(16)), 0)) #, broadcast=True)
            othergates = g.input_tensor(shape=((max_gates+1)//2, 2*2*2, pow2qb), dtype=g.float32, name="othergate", layout=get_slice16(WEST, list(range(16)), 0)) #, broadcast=True)
            gatemap = [[g.address_map(g.split_vectors(gates if hemi==EAST else othergates, [4]*((max_gates+1)//2*2))[0], np.array([0]*20), index_map_layout=get_slice4(hemi, 17, 21, 1)),
                        g.address_map(g.split_vectors(gates if hemi==EAST else othergates, [4]*((max_gates+1)//2*2))[1], np.array([0]*20), index_map_layout=get_slice4(hemi, 17, 21, 1))] for hemi in (EAST, WEST)]
            gateinc = [g.from_data(np.array(([1*2]+[0]*15)*20, dtype=np.uint8), layout=get_slice1(hemi, 0, 1)) for hemi in (EAST, WEST)] #gather map is little endian byte order
            gateinc256 = [g.zeros((320,), layout=get_slice1(hemi, 1, 1), dtype=g.uint8) for hemi in (EAST, WEST)]
            gateinccount = [g.from_data(np.array(([0, 1*2]+[0]*14)*20, dtype=np.uint8), layout=get_slice1(hemi, 2, 1)) for hemi in (EAST, WEST)]
            gateincmask = [g.from_data(np.array(([0, 1]+[0]*14)*20, dtype=np.uint8), layout=get_slice1(hemi, 3, 1)) for hemi in (EAST, WEST)]

            targetqbits = g.input_tensor(shape=(max_gates, 320), dtype=g.uint8, name="target_qbits", layout=get_slice1(WEST, 39, 0))
            controlqbits = g.input_tensor(shape=(max_gates, 320), dtype=g.uint8, name="control_qbits", layout=get_slice1(WEST, 37, 0))
            targetqbitdistro = [g.zeros((320,), dtype=g.uint8, layout=get_slice1(hemi, 37, 0)) for hemi in (EAST, WEST)]
            controlqbitdistro = [g.zeros((320,), dtype=g.uint8, layout=get_slice1(hemi, 36, 0)) for hemi in (EAST, WEST)]
            idxmapsort, idxmapm1 = UnitarySimulator.idxmapgather(num_qbits)
            #need to handle inner splits for >=9 qbits!!!
            idxmapsort = np.stack(((np.stack(idxmapsort) & 255).astype(np.uint8), (np.stack(idxmapsort) >> 8).astype(np.uint8))).transpose(3, 2, 1, 0).reshape(2, -1, num_qbits*2)
            if num_qbits % 8 != 0: idxmapsort = np.concatenate((idxmapsort, np.zeros((2, idxmapsort.shape[-2], 2*(8-num_qbits % 8)), dtype=np.uint8)), axis=2)
            idxmapsort = np.repeat(idxmapsort, 20, axis=1).reshape(2, -1, 320)
            idxmapm1 = np.stack(((np.stack(idxmapm1) & 255).astype(np.uint8), (np.stack(idxmapm1) >> 8).astype(np.uint8))).transpose(3, 2, 1, 0).reshape(2, -1, (num_qbits-1)*2)
            if (num_qbits-1) % 8 != 0: idxmapm1 = np.concatenate((idxmapm1, np.zeros((2, idxmapm1.shape[-2], 2*(8-(num_qbits-1) % 8)), dtype=np.uint8)), axis=2)
            idxmapm1 = np.repeat(idxmapm1, 20, axis=1).reshape(2, -1, 320)
            targetqbitpairs0 = [g.from_data(idxmapsort[0,:], layout=get_slice1(hemi, 43, 0)) for hemi in (EAST, WEST)]
            targetqbitpairs1 = [g.from_data(idxmapsort[1,:], layout=get_slice1(hemi, 42, 0)) for hemi in (EAST, WEST)]
            controlqbitpairs0 = [g.from_data(idxmapm1[0,:], layout=get_slice1(hemi, 41, 0)) for hemi in (EAST, WEST)]
            controlqbitpairs1 = [g.from_data(idxmapm1[1,:], layout=get_slice1(hemi, 40, 0)) for hemi in (EAST, WEST)]

            qbitinc = g.from_data(np.array(([1]+[0]*15)*20, dtype=np.uint8), layout=get_slice1(WEST, 0, 1)) #gather map is little endian byte order
            qbitinc256 = g.zeros((320,), layout=get_slice1(WEST, 1, 1), dtype=g.uint8)
            qbitinccount = g.from_data(np.array(([0, 1]+[0]*14)*20, dtype=np.uint8), layout=get_slice1(WEST, 2, 1))
            qbitmap = g.address_map(targetqbits, np.array([0]*20), index_map_layout=get_slice1(WEST, 1, 1))
           
            g.add_mem_constraints(gateinc + gateinc256 + gateinccount, [gates, othergates], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
        innerdim = ((pow2qb + 320-1) // 320) * 320
        for reversedir in (False, True):
            for target_qbit, control_qbit in ((0, None), (0, 1)):
                suffix = ("rev" if reversedir else "") + str(target_qbit) + "_" + str(control_qbit)
                with pgm_pkg.create_program_context("us_gate"+suffix) as pc:
                    #if not reversedir and target_qbit == 0 and control_qbit is None: print(gatemap[0].data, gatemap[1].data)
                    newus = UnitarySimulator(num_qbits, reversedir, us)
                    g.reserve_tensor(pcinit, pc, otherunitary if reversedir else unitary)
                    g.reserve_tensor(pcinit, pc, othercopy if reversedir else copy)
                    g.reserve_tensor(pcinit, pc, othergates if reversedir else gates)
                    g.reserve_tensor(pcinit, pc, targetqbits)
                    g.reserve_tensor(pcinit, pc, controlqbits)
                    gmap = [tensor.shared_memory_tensor(mem_tensor=gatemap[reversedir][i], name="gatemap" + suffix) for i in range(2)]
                    ginc = tensor.shared_memory_tensor(mem_tensor=gateinc[reversedir], name="gateinc" + suffix)
                    ginc256 = tensor.shared_memory_tensor(mem_tensor=gateinc256[reversedir], name="gateinc256" + suffix)
                    ginccount = tensor.shared_memory_tensor(mem_tensor=gateinccount[reversedir], name="gateinccount" + suffix)
                    gincmask = tensor.shared_memory_tensor(mem_tensor=gateincmask[reversedir], name="gateincmask" + suffix)
                    tqbitdistro = [tensor.shared_memory_tensor(mem_tensor=targetqbitdistro[i], name="tqbitdistro" + suffix) for i in range(2)]
                    tqbitpairs0 = [tensor.shared_memory_tensor(mem_tensor=targetqbitpairs0[i], name="tqbitpairs0" + suffix) for i in range(2)]
                    tqbitpairs1 = [tensor.shared_memory_tensor(mem_tensor=targetqbitpairs1[i], name="tqbitpairs1" + suffix) for i in range(2)]
                    if not control_qbit is None:
                        cqbitdistro = [tensor.shared_memory_tensor(mem_tensor=controlqbitdistro[i], name="cqbitdistro" + suffix) for i in range(2)]
                        cqbitpairs0 = [tensor.shared_memory_tensor(mem_tensor=controlqbitpairs0[i], name="cqbitpairs0" + suffix) for i in range(2)]
                        cqbitpairs1 = [tensor.shared_memory_tensor(mem_tensor=controlqbitpairs1[i], name="cqbitpairs1" + suffix) for i in range(2)]
                    g.add_mem_constraints([ginc, ginc256, ginccount], [othergates if reversedir else gates], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
                    
                    qmap = tensor.shared_memory_tensor(mem_tensor=qbitmap, name="qmap" + suffix)
                    qinc = tensor.shared_memory_tensor(mem_tensor=qbitinc, name="qinc" + suffix)
                    qinc256 = tensor.shared_memory_tensor(mem_tensor=qbitinc256, name="qinc256" + suffix)
                    qinccount = tensor.shared_memory_tensor(mem_tensor=qbitinccount, name="qinccount" + suffix)
                    
                    unitaryctxt = g.from_addresses(np.array((otherunitary if reversedir else unitary).storage_request.addresses.reshape(-1, g.float32.size), dtype=object), innerdim, g.float32, "unitary" + suffix)
                    copyctxt = g.from_addresses(np.array((othercopy if reversedir else copy).storage_request.addresses.reshape(-1, g.float32.size), dtype=object), innerdim, g.float32, "copy" + suffix)
                    gatesctxt = g.from_addresses(np.array((othergates if reversedir else gates).storage_request.addresses.reshape(-1, g.float32.size), dtype=object), innerdim, g.float32, "gates" + suffix)
                    tqbits = g.from_addresses(np.array(targetqbits.storage_request.addresses.reshape(-1, g.uint8.size), dtype=object), innerdim, g.uint8, "targetqbits" + suffix)
                    cqbits = g.from_addresses(np.array(controlqbits.storage_request.addresses.reshape(-1, g.uint8.size), dtype=object), innerdim, g.uint8, "controlqbits" + suffix)
                    with g.ResourceScope(name="setgatherdistros", is_buffered=True, time=0, predecessors=None) as pred:
                        for i in range(2):
                            g.mem_gather(tqbits, qmap, time=i).write(name="targetqbitdistro" + suffix, storage_req=tqbitdistro[i].storage_request)
                            if not control_qbit is None:
                                g.mem_gather(cqbits, qmap, time=2+i).write(name="controlqbitdistro" + suffix, storage_req=cqbitdistro[i].storage_request)
                    tcmap = [list(reversed(x)) if reversedir else x for x in ((tqbitdistro, tqbitpairs0, tqbitpairs1, cqbitdistro, cqbitpairs0, cqbitpairs1) if not control_qbit is None else (tqbitdistro, tqbitpairs0, tqbitpairs1))]
                    with g.ResourceScope(name="rungate", is_buffered=True, time=None, predecessors=[pred]) as pred:
                        newus.build(unitaryctxt, copyctxt, target_qbit, control_qbit, gatesctxt, gmap, tcmap)
                    with g.ResourceScope(name="incgate", is_buffered=True, time=None, predecessors=[pred]) as pred:
                        updinc = g.stack([ginc256]*2, 0).add(g.stack([ginccount]*2, 0), time=0, alus=[3 if reversedir else 0], overflow_mode=g.OverflowMode.MODULAR)
                        updmap = g.stack([g.split_vectors(gmap[0].reinterpret(g.uint8), [1]*4)[0],
                                          g.split_vectors(gmap[1].reinterpret(g.uint8), [1]*4)[0]], 0).add(g.stack([ginc]*2, 0), alus=[7 if reversedir else 4], overflow_mode=g.OverflowMode.MODULAR).add(g.mask_bar(updinc, g.stack([gincmask]*2, 0)))
                        updmap = g.split_vectors(updmap, [1]*2)
                        g.concat_vectors([updmap[0]]*4, (4, 320)).write(storage_req=gmap[0].storage_request, name="nextgatemap" + suffix)
                        g.concat_vectors([updmap[1]]*4, (4, 320)).write(storage_req=gmap[1].storage_request, name="nextgatemappair" + suffix)
                        g.split_vectors(updinc, [1]*2)[0].vxm_identity().write(storage_req=ginc256.storage_request, name="nextgateinc256" + suffix)
                    with g.ResourceScope(name="incqbit", is_buffered=True, time=None, predecessors=[pred]) as pred:
                        updinc = qinc256.add(qinccount, time=0, alus=[0], overflow_mode=g.OverflowMode.MODULAR)
                        qmap.add(qinc, alus=[4], overflow_mode=g.OverflowMode.MODULAR).add(g.mask_bar(updinc, gincmask)).write(storage_req=qmap.storage_request, name="nextqmap" + suffix)
                        updinc.vxm_identity().write(storage_req=qinc256.storage_request, name="nextqinc256" + suffix)                            
        with pgm_pkg.create_program_context("final_us") as pcfinal:
            g.reserve_tensor(pcinit, pcfinal, unitary)
            unitaryres = g.from_addresses(np.array(unitary.storage_request.addresses.reshape(-1, g.float32.size), dtype=object), pow2qb, g.float32, "unitaryfin")
            unitaryres.set_program_output()
        with pgm_pkg.create_program_context("finalrev_us") as pcfinal:
            g.reserve_tensor(pcinit, pcfinal, otherunitary)
            unitaryrevres = g.from_addresses(np.array(otherunitary.storage_request.addresses.reshape(-1, g.float32.size), dtype=object), pow2qb, g.float32, "unitaryrevfin")
            unitaryrevres.set_program_output()
        print_utils.infoc("\nAssembling model ...")
        iops = pgm_pkg.assemble()
        iop = runtime.IOProgram(iops[0])
        driver = runtime.Driver()
        device = driver.next_available_device()
        u = np.eye(pow2qb) + 0j if use_identity else unitary_group.rvs(pow2qb)
        target_qbits = [np.random.randint(num_qbits) for _ in range(num_gates)]
        control_qbits = target_qbits #[np.random.randint(num_qbits) for _ in range(num_gates)]
        print(target_qbits)
        parameters = np.random.random((num_gates, 3))
        gateparams = [make_u3(parameters[i,:]) if target_qbits[i] == control_qbits[i] else make_cry(parameters[i,:]) for i in range(num_gates)]
        oracleres, result = [None], [None]
        def oracle():
            oracleres[0] = qiskit_oracle(u, num_qbits, [(control_qbits[i] == target_qbits[i], target_qbits[i], control_qbits[i], parameters[i,:]) for i in range(num_gates)])
        with device:
            runfunc = [None]
            def loaddata():
                for i in range(1+2*((num_qbits+8-1)//8)*((num_qbits+8-1)//8)+2):
                    device.load(iop[i], unsafe_keep_entry_points=True)
                def actual():
                    inputs = {}
                    inputs[unitary.name] = np.ascontiguousarray(u.astype(np.complex64)).view(np.float32).reshape(pow2qb, pow2qb, 2).transpose(0, 2, 1).reshape(pow2qb*2, pow2qb)
                    inputs[gates.name] = np.concatenate([np.repeat(gateparams[i].astype(np.complex64).view(np.float32).flatten(), pow2qb) for i in range(0, num_gates, 2)] + [np.zeros((2*2*2*pow2qb), dtype=np.float32)]*((max_gates+1)//2-(num_gates-num_gates//2)))
                    inputs[othergates.name] = np.concatenate([np.repeat(gateparams[i].astype(np.complex64).view(np.float32).flatten(), pow2qb) for i in range(1, num_gates, 2)] + [np.zeros((2*2*2*pow2qb), dtype=np.float32)]*((max_gates+1)//2-num_gates//2))
                    inputs[targetqbits.name] = np.concatenate((np.repeat(np.hstack((np.array(target_qbits, dtype=np.uint8)[:,np.newaxis]%8*2, np.array(target_qbits, dtype=np.uint8)[:,np.newaxis]%8*2+1, np.array([[16]*14]*num_gates, dtype=np.uint8))), 20, axis=0).reshape(-1, 320), np.zeros((max_gates-num_gates, 320), dtype=np.uint8)))
                    adjcontrolqbits = [x-(x>target_qbits[i]) for i, x in enumerate(control_qbits)]
                    inputs[controlqbits.name] = np.concatenate((np.repeat(np.hstack((np.array(adjcontrolqbits, dtype=np.uint8)[:,np.newaxis]%8*2, np.array(adjcontrolqbits, dtype=np.uint8)[:,np.newaxis]%8*2+1, np.array([[16]*14]*num_gates, dtype=np.uint8))), 20, axis=0).reshape(-1, 320), np.zeros((max_gates-num_gates, 320), dtype=np.uint8)))
                    invoke([device], iop, 0, 0, [inputs])
                    for i in range(num_gates):
                        #progidx = 1+(num_qbits*num_qbits if (i&1)!=0 else 0)+target_qbits[i]*num_qbits+control_qbits[i]
                        progidx = 1+(((num_qbits+8-1)//8)*((num_qbits+8-1)//8) if (i&1)!=0 else 0) + (target_qbits[i]//8)*((num_qbits+8-1)//8) + (control_qbits[i]//8)
                        invoke([device], iop, progidx, 0, None, None, None)
                    res, _ = invoke([device], iop, 1+2*((num_qbits+8-1)//8)*((num_qbits+8-1)//8)+(num_gates&1), 0, None, None, None)
                    result[0] = np.ascontiguousarray(res[0][unitaryres.name if (num_gates&1)==0 else unitaryrevres.name].reshape(pow2qb, 2, pow2qb).transpose(0, 2, 1)).view(np.complex64).reshape(pow2qb, pow2qb).astype(np.complex128)
                runfunc[0] = actual
        loaddata()
        actual = runfunc[0]
        actual()
        oracle()
        oracleres, result = oracleres[0], result[0]
        if np.allclose(oracleres, result):
            print_utils.success("\nQuantum Simulator Chain Test Success ...")
        else:
            print_utils.err("\nQuantum Simulator Chain Test Failure")
            
            print_utils.infoc(str(oracleres[not np.isclose(oracleres, result)]) + " " + str(result[not np.isclose(oracleres, result)]))
        
def main():
    #test()
    #[UnitarySimulator.idxmapgather(x) for x in range(10)]; assert False
    #import math; [4*(1<<x)*int(math.ceil((1<<x)/320)) for x in range(12)]
    #[4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096, 16384, 57344]
    #10 qbits max for single bank, 11 qbits for dual bank solutions
    #import math; [math.ceil((4*(1<<x)*int(math.ceil((1<<x)/320))//8)/8192) for x in range(15)]
    num_qbits, max_levels = 3, 6
    max_gates = num_qbits+3*(num_qbits*(num_qbits-1)//2*max_levels)
    UnitarySimulator.unit_test(num_qbits)
    UnitarySimulator.chain_test(num_qbits, max_gates)
if __name__ == "__main__":
    main()
