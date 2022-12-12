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
def generateCurveOffsetMap(offsets):
    superlane_size = 16
    superlane_count = 20
    bytes_per_element = superlane_size * superlane_count
    curveOffsetMap = []
    for offset in offsets:
        offsetMapEntry = [ 255 ]*bytes_per_element
        splitOffset = (offset & 255, offset >> 8)
        for s in range(superlane_count):
            slot = s*superlane_size
            offsetMapEntry[slot:slot+2] = splitOffset
        ofsetMap.append(offsetMapEntry)
    return np.asarray(offsetMap, dtype=np.uint8)
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
        return unitary.read(streams=g.SG8[0]).write(name="initcopy", storage_req=self.copystore[WEST], time=0)
    def build(self, unitary, copy, target_qbit, control_qbit, gate, gatesel=None, inittime=0):
        if copy is None:
            with g.ResourceScope(name="initcopy", is_buffered=True, time=0) as pred:
                copy = self.copymatrix(unitary)
        else: pred = None
        pow2qb = 1 << self.num_qbits
        innerdim = pow2qb if gatesel is None else ((pow2qb+320-1)//320)*320
        t = np.roll(np.arange(self.num_qbits), target_qbit)
        idxs = np.arange(pow2qb).reshape(*([2]*self.num_qbits)).transpose(t).reshape(-1, 2)
        usplit = np.array(g.split_vectors(unitary, [1] * (2*pow2qb))).reshape(pow2qb, 2)
        ucopysplit = np.array(g.split_vectors(copy, [1] * (2*pow2qb))).reshape(pow2qb, 2)
        pairs = idxs if control_qbit is None else idxs[(idxs[:,0] & (1<<control_qbit)) != 0,:]
        u = [usplit[pairs[:,0],0], usplit[pairs[:,0],1], ucopysplit[pairs[:,1],0], ucopysplit[pairs[:,1],1]]
        if not control_qbit is None: bypasspairs = idxs[(idxs[:,0] & (1<<control_qbit)) == 0,:]
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
                gatesel_st = g.concat_vectors([gatesel.reshape(1,innerdim)]*(pow2qb//r), (pow2qb//r, innerdim)).read(streams=g.SG4[1])
                gs = [g.mem_gather(g.concat_vectors(gatevals[:,i].tolist()*(pow2qb//2//r)+gatevals[:,i+4].tolist()*(pow2qb//2//r), (gate.shape[0]//8, pow2qb//r, innerdim)),
                                    gatesel_st, output_streams=[g.SG4[2*i]]) for i in range(4)]
                #gatesel_st = g.split_vectors(g.concat_vectors([gatesel.reshape(1,innerdim)]*(pow2qb//r), (pow2qb//r, innerdim)).read(streams=g.SG4[1]), [1]*(pow2qb//r))
                #gs = [g.concat_vectors([g.mem_gather(g.concat_vectors(gatevals[:,i+k*4].tolist(), (gate.shape[0]//8, innerdim)), gatesel_st[j+k*pow2qb//2//r], output_streams=[g.SG4[2*i]]) for j in range(pow2qb//2//r) for k in range(2)], (pow2qb//r, innerdim)) for i in range(4)]
            us = [g.concat_vectors((ub[i%2].flatten().tolist() + ub[i%2+2].flatten().tolist() if i in [0,3] else []) + u[i].flatten().tolist()*2, (pow2qb if control_qbit is None or i in [0,3] else pow2qb//2, innerdim)).read(streams=g.SG4[2*i+1]) for i in range(4)]
            usb = [[]]*2
            if not control_qbit is None:
                for i in [0,3]:
                    usb[i%2], us[i] = g.split_vectors(us[i], [pow2qb//2, pow2qb//2])
                usb = [g.vxm_identity(usb[i], alus=[[rev_alu(13, self.rev),rev_alu(14, self.rev)][i]], time=0, output_streams=g.SG4[[1,7][i]]) for i in range(2)]
            m1 = [g.mul(gs[i], us[i], alus=[[rev_alu(0, self.rev),rev_alu(4, self.rev),rev_alu(8, self.rev),rev_alu(12, self.rev)][i]], output_streams=g.SG4[[0,2,4,6][i]], time=pow2qb if i==0 else None) for i in range(4)]
            m2 = [g.mul(gs[i], us[i^1], alus=[[rev_alu(2, self.rev),rev_alu(3, self.rev),rev_alu(10, self.rev),rev_alu(11, self.rev)][i]], output_streams=g.SG4[[3,3,5,5][i]]) for i in range(4)]
            a1 = [g.sub(m1[2*i], m1[2*i+1], alus=[[rev_alu(1, self.rev),rev_alu(9, self.rev)][i]], output_streams=g.SG4[[0,6][i]]) for i in range(2)]
            a2 = [g.add(m2[i], m2[2+i], alus=[[rev_alu(5, self.rev),rev_alu(6, self.rev)][i]], output_streams=g.SG4[[4,3][i]]) for i in range(2)]
            ri = [g.add(a1[0], a1[1], alus=[rev_alu(15, self.rev)], output_streams=g.SG4[0]),
                  g.add(a2[0], a2[1], alus=[rev_alu(7, self.rev)], output_streams=g.SG4[4])]
            ri = g.concat_vectors(np.hstack([np.array(g.split_vectors(ri[i] if control_qbit is None else g.concat_vectors([usb[i], ri[i]], (pow2qb, innerdim)), [1]*(pow2qb)))[revidx].reshape(pow2qb, 1) for i in range(2)]).flatten().tolist(), (pow2qb*2, innerdim))
            result = ri.write(name="result", storage_req=self.otherinit)
            copy = ri.write(name="copy", storage_req=self.copystore[EAST])
        return result, copy
    def unit_test(num_qbits):
        pow2qb = 1 << num_qbits
        use_identity = False
        target_qbit = np.random.randint(num_qbits)
        control_qbit = np.random.randint(num_qbits)
        if target_qbit == control_qbit: control_qbit = None
        print(target_qbit, control_qbit)
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
        num_gates, use_identity = 1, True
        pow2qb = 1 << num_qbits
        pgm_pkg = g.ProgramPackage(name="us", output_dir="us")
        print("Number of gates:", num_gates, "Maximum gates:", max_gates)
        with pgm_pkg.create_program_context("init_us") as pcinit:
            us = UnitarySimulator(num_qbits)
            unitary = g.input_tensor(shape=(pow2qb*2, pow2qb), dtype=g.float32, name="unitary", layout=get_slice8(WEST, s8range[0], s8range[-1], 0))
            us.uinit = unitary.storage_request
            #gatescomb = g.input_tensor(shape=(max_gates+1)//2*2, 2*2*2, pow2qb), dtype=g.float32, name="gate", layout="-1, H2, S16(" + str(min(slices)) + "-" + str(max(slices)) + ")")
            gates = g.input_tensor(shape=((max_gates+1)//2, 2*2*2, pow2qb), dtype=g.float32, name="gate", layout=get_slice16(EAST, list(range(16)), 0)) #, broadcast=True)
            othergates = g.input_tensor(shape=((max_gates+1)//2, 2*2*2, pow2qb), dtype=g.float32, name="othergate", layout=get_slice16(WEST, list(range(16)), 0)) #, broadcast=True)
            gatemap = [g.address_map(g.split_vectors(gates if hemi==EAST else othergates, [1]*((max_gates+1)//2*2*2*2))[0], np.array([0]*20), index_map_layout=get_slice4(hemi, 17, 21, 1)) for hemi in (EAST, WEST)]
            gateinc = [g.from_data(np.array(([1]+[0]*15)*20, dtype=np.uint8), layout=get_slice1(hemi, 0, 1)) for hemi in (EAST, WEST)] #gather map is little endian byte order
            gateinc256 = [g.zeros((320,), layout=get_slice1(hemi, 1, 1), dtype=g.uint8) for hemi in (EAST, WEST)]
            gateinccount = [g.from_data(np.array(([0, 1]+[0]*14)*20, dtype=np.uint8), layout=get_slice1(hemi, 2, 1)) for hemi in (EAST, WEST)]
            g.add_mem_constraints(gateinc + gateinc256 + gateinccount, [gates, othergates], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
            with g.ResourceScope(name="makecopy", is_buffered=True, time=0) as pred:
                copy = us.copymatrix(unitary)
        innerdim = ((pow2qb + 320-1) // 320) * 320
        for reversedir in (False, True):
            for target_qbit in range(num_qbits):
                for control_qbit in range(num_qbits):
                    if control_qbit == target_qbit: control_qbit = None                    
                    suffix = ("rev" if reversedir else "") + str(target_qbit) + "_" + str(control_qbit)
                    with pgm_pkg.create_program_context("us_gate"+suffix) as pc:
                        #if not reversedir and target_qbit == 0 and control_qbit is None: print(gatemap[0].data, gatemap[1].data)
                        newus = UnitarySimulator(num_qbits, reversedir, us)
                        g.reserve_tensor(pcrev if reversedir else pcinit, pc, otherunitary if reversedir else unitary)
                        g.reserve_tensor(pcrev if reversedir else pcinit, pc, othercopy if reversedir else copy)
                        g.reserve_tensor(pcinit, pc, othergates if reversedir else gates)
                        gmap = tensor.shared_memory_tensor(mem_tensor=gatemap[reversedir], name="gatemap" + suffix)
                        ginc = tensor.shared_memory_tensor(mem_tensor=gateinc[reversedir], name="gateinc" + suffix)
                        ginc256 = tensor.shared_memory_tensor(mem_tensor=gateinc256[reversedir], name="gateinc256" + suffix)
                        ginccount = tensor.shared_memory_tensor(mem_tensor=gateinccount[reversedir], name="gateinccount" + suffix)
                        g.add_mem_constraints([ginc, ginc256, ginccount], [othergates if reversedir else gates], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
                        unitaryctxt = g.from_addresses(np.array((otherunitary if reversedir else unitary).storage_request.addresses.reshape(-1, g.float32.size), dtype=object), innerdim, g.float32, "unitary" + suffix)
                        copyctxt = g.from_addresses(np.array((othercopy if reversedir else copy).storage_request.addresses.reshape(-1, g.float32.size), dtype=object), innerdim, g.float32, "copy" + suffix)
                        gatesctxt = g.from_addresses(np.array((othergates if reversedir else gates).storage_request.addresses.reshape(-1, g.float32.size), dtype=object), innerdim, g.float32, "gates" + suffix)                        
                        with g.ResourceScope(name="incgate", is_buffered=True, time=0, predecessors=None) as pred:
                            if not reversedir and target_qbit==0 and control_qbit is None:
                                otherunitary, othercopy = newus.build(unitaryctxt, copyctxt, target_qbit, control_qbit, gatesctxt, gmap)
                                pcrev = pc
                            else:
                                newus.build(unitaryctxt, copyctxt, target_qbit, control_qbit, gatesctxt, gmap)
                        with g.ResourceScope(name="incgate", is_buffered=True, time=None, predecessors=[pred]) as pred:
                            updinc = ginccount.add(ginc256, time=0, alus=[0])
                            updmap = g.split_vectors(gmap.reinterpret(g.uint8), [1]*4)[0].add(ginc, alus=[4]).add(g.mask_bar(updinc, ginc256))
                            g.concat_vectors([updmap]*4, (4, 320)).write(storage_req=gmap.storage_request, name="nextgatemap" + suffix)
                            updinc.vxm_identity().write(storage_req=ginccount.storage_request, name="nextgateinc256" + suffix)
        with pgm_pkg.create_program_context("final_us") as pcfinal:
            g.reserve_tensor(pcinit, pcfinal, unitary)
            unitaryres = g.from_addresses(np.array(unitary.storage_request.addresses.reshape(-1, g.float32.size), dtype=object), pow2qb, g.float32, "unitaryfin")
            unitaryres.set_program_output()
        with pgm_pkg.create_program_context("finalrev_us") as pcfinal:
            g.reserve_tensor(pcrev, pcfinal, otherunitary)
            unitaryrevres = g.from_addresses(np.array(otherunitary.storage_request.addresses.reshape(-1, g.float32.size), dtype=object), pow2qb, g.float32, "unitaryrevfin")
            unitaryrevres.set_program_output()
        print_utils.infoc("\nAssembling model ...")
        iops = pgm_pkg.assemble()
        iop = runtime.IOProgram(iops[0])
        driver = runtime.Driver()
        device = driver.next_available_device()
        u = np.eye(pow2qb) + 0j if use_identity else unitary_group.rvs(pow2qb)
        target_qbits = [0] #[np.random.randint(num_qbits) for _ in range(num_gates)]
        control_qbits = target_qbits # [np.random.randint(num_qbits) for _ in range(num_gates)]
        parameters = np.random.random((num_gates, 3))
        gateparams = [make_u3(parameters[i,:]) if target_qbits[i] == control_qbits[i] else make_cry(parameters[i,:]) for i in range(num_gates)]
        oracleres, result = [None], [None]
        def oracle():
            oracleres[0] = qiskit_oracle(u, num_qbits, [(control_qbits[i] == target_qbits[i], target_qbits[i], control_qbits[i], parameters[i,:]) for i in range(num_gates)])
        with device:
            runfunc = [None]
            def loaddata():
                for i in range(1+2*num_qbits*num_qbits+2):
                    device.load(iop[i], unsafe_keep_entry_points=True)
                def actual():
                    inputs = {}
                    inputs[unitary.name] = np.ascontiguousarray(u.astype(np.complex64)).view(np.float32).reshape(pow2qb, pow2qb, 2).transpose(0, 2, 1).reshape(pow2qb*2, pow2qb)
                    inputs[gates.name] = np.concatenate([np.repeat(gateparams[i].astype(np.complex64).view(np.float32).flatten(), pow2qb) for i in range(0, num_gates, 2)] + [np.zeros((2*2*2*pow2qb), dtype=np.float32)]*((max_gates+1)//2-(num_gates-num_gates//2)))
                    print(inputs[gates.name])
                    inputs[othergates.name] = np.concatenate([np.repeat(gateparams[i].astype(np.complex64).view(np.float32).flatten(), pow2qb) for i in range(1, num_gates, 2)] + [np.zeros((2*2*2*pow2qb), dtype=np.float32)]*((max_gates+1)//2-num_gates//2))
                    invoke([device], iop, 0, 0, [inputs])
                    for i in range(num_gates):
                        progidx = 1+(num_qbits*num_qbits if (i&1)!=0 else 0)+target_qbits[i]*num_qbits+control_qbits[i]
                        invoke([device], iop, progidx, 0, None, None, None)
                    res, _ = invoke([device], iop, 1+2*num_qbits*num_qbits+(num_gates&1), 0, None, None, None)
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
            print_utils.infoc(str(oracleres - result) + str(oracleres) + str(result))
        
def main():
    #test()
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
