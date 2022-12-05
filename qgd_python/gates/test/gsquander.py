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
def get_slice1(drctn, start, bank=0):
    return "-1, H1(" + ("W" if drctn==WEST else "E") + "), S1(" + str(start) + "), B1(" + str(bank) + ")"
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
class UnitarySimulator(g.Component):
    def __init__(self, num_qbits, **kwargs):
        super().__init__(**kwargs)
        self.num_qbits = num_qbits
        self.copystore = []
        self.eastinit = tensor.create_storage_request(layout=get_slice8(EAST, s8range[0], s8range[-1], 0))
        #more efficient to just directly copy on the controlled rotation rather than doing unnecessary identity gate computations
        self.identity2x2 = [g.from_data(np.zeros((1), dtype=np.float32), layout=get_slice1(WEST, 0, 0), broadcast=True),
                            g.from_data(np.ones((1), dtype=np.float32), layout=get_slice1(WEST, 0, 0), broadcast=True)]
        for hemi in (WEST, EAST):
            self.copystore.append(tensor.create_storage_request(layout=get_slice8(hemi, s8range2[0], s8range2[-1], 0)))
    def copymatrix(self, unitary):        
        return unitary.read(streams=g.SG8[0]).write(name="initcopy", storage_req=self.copystore[WEST], time=0)
    def build(self, unitary, copy, target_qbit, control_qbit, gate, inittime=0):
        if copy is None:
            with g.ResourceScope(name="initcopy", is_buffered=True, time=0) as pred:
                copy = self.copymatrix(unitary)
        pow2qb = 1 << self.num_qbits
        t = np.roll(np.arange(self.num_qbits), target_qbit)
        idxs = np.arange(pow2qb).reshape(*([2]*self.num_qbits)).transpose(t).reshape(-1, 2)
        usplit = np.array(g.split_vectors(unitary, [1] * (2*pow2qb))).reshape(pow2qb, 2)
        ucopysplit = np.array(g.split_vectors(copy, [1] * (2*pow2qb))).reshape(pow2qb, 2)
        gatevals = g.split_vectors(gate, [1]*(2*2*2))
        pairs = (idxs if control_qbit is None else idxs[(idxs[:,0] & (1<<control_qbit)) != 0,:])
        u = [usplit[pairs[:,0],0], usplit[pairs[:,0],1], ucopysplit[pairs[:,1],0], ucopysplit[pairs[:,1],1]]
        revidx = np.argsort(pairs[:,0])
        revidx = revidx.tolist() + (revidx+(pow2qb//2)).tolist()
        print(revidx)
        with g.ResourceScope(name="rungate", is_buffered=True, time=0 if pred is None else None, predecessors=None if pred is None else [pred]) as pred:
            #(a+bi)*(c+di)=(ac-bd)+(ad+bc)i
            #gate[0] * p[0] - gate[1] * p[1] + gate[2] * p[2] + gate[3] * p[3]
            #gate[0] * p[1] + gate[1] * p[0] + gate[2] * p[3] + gate[3] * p[2]
            gs = [g.concat_vectors([gatevals[i]]*(pow2qb//2)+[gatevals[i+4]]*(pow2qb//2), (pow2qb, pow2qb)).read(streams=g.SG4[2*i], time=0 if i == 0 else None) for i in range(4)]
            us = [g.concat_vectors(u[i].flatten().tolist()*2, (pow2qb, pow2qb)).read(streams=g.SG4[2*i+1]) for i in range(4)]
            m1 = [g.mul(gs[i], us[i], alus=[4*i], output_streams=g.SG4_E[i*2]) for i in range(4)]
            m2 = [g.mul(gs[i], us[i], alus=[4*i+1], output_streams=g.SG4_E[i*2+1]) for i in range(4)]
            a1 = [(g.sub if i==0 else g.add)(m1[2*i], m1[2*i+1], alus=[2+8*i], output_streams=g.SG4[3+i]) for i in range(2)]
            a2 = [g.add(m1[(2*i+1)%4], m1[(2*i+2)%4], alus=[6+8*i], output_streams=g.SG4[5+i]) for i in range(2)]
            ri = [g.add(a1[0], a1[1], alus=[7], output_streams=g.SG4[2]),
                  g.add(a2[0], a2[1], alus=[11], output_streams=g.SG4[4])]
            ri = g.concat_vectors(np.hstack([np.array(g.split_vectors(ri[i], [1]*pow2qb))[revidx].reshape(pow2qb, 1) for i in range(2)]).flatten().tolist(), (pow2qb*2, pow2qb))
            result = ri.write(name="result", storage_req=self.eastinit)
            copy = ri.write(name="copy", storage_req=self.copystore[EAST])
        return result, copy
    def unit_test(num_qbits):
        pow2qb = 1 << num_qbits
        target_qbit = np.random.randint(num_qbits)
        control_qbit = None#np.random.randint(num_qbits)
        if target_qbit == control_qbit: control_qbit = None
        with g.ProgramContext() as pc:
            us = UnitarySimulator(num_qbits)
            unitary = g.input_tensor(shape=(pow2qb*2, pow2qb), dtype=g.float32, name="unitary", layout=get_slice8(WEST, s8range[0], s8range[-1], 0))
            gate = g.input_tensor(shape=(1, 2*2*2, pow2qb), dtype=g.float32, name="gates", layout=get_slice16(WEST, list(range(16)), 0))#, broadcast=True)
            output, _ = us.build(unitary, None, target_qbit, control_qbit, gate)
            output.set_program_output()
            iop_file, json_file = compile_unit_test("usunit")
        runner = tsp.create_tsp_runner(iop_file)
        u = unitary_group.rvs(pow2qb)
        parameters = np.random.random(3)
        gateparams = make_u3(parameters) if control_qbit is None else make_cry(parameters)
        print_utils.infoc("\nRunning on HW ...")
        def oracle():
            return qiskit_oracle(u, num_qbits, [control_qbit is None, target_qbit, control_qbit, parameters])
        def actual():
            inputs = {}
            for i in range(parallel):
                inputs[unitary.name] = u.astype(np.complex32).view(np.float32).reshape(pow2qb, pow2qb, 2).transpose(0, 2, 1).reshape(pow2qb*2, poq2qb)
                inputs[gate.name] = gateparams
            res = runner(**inputs)
            result[0] = res[output.name].reshape(pow2qb, 2, pow2qb).transpose(0, 2, 1)
        oracle()
        actual()
        oracleres, result = oracleres[0], result[0]
        np.set_printoptions(formatter={'int':hex}, threshold=sys.maxsize, floatmode='unique')
        if np.all(oracleres == result):
            print_utils.success("\nQuantum Simulator Unit Test Success ...")
        else:
            print_utils.err("\nQuantum Simulator Test Failure")
            print_utils.infoc(str(oracleres) + " " + str(result))
        
    def chain_test(num_qbits, max_gates):
        pow2qb = 1 << num_qbits
        with pgm_pkg.create_program_context("init_us") as pcinit:
            us = UnitarySimulator(num_qbits)
            unitary = g.input_tensor(shape=(pow2qb, pow2qb), dtype=g.float32, name="unitary", layout=get_slice8(WEST, s8range[0], s8range[-1], 0))
            gates = g.input_tensor(shape=(max_gates, 2*2*2, 1), dtype=g.float32, name="gate", layout=get_slice16(WEST, list(range(16)), 0), broadcast=True)
            with g.ResourceScope(name="makecopy", is_buffered=True, time=0) as pred:
                copy = us.copymatrix(unitary)
        for op in (True, False):
            for target_qbit in range(num_qbits):
                for control_qbit in [None] if op else range(num_qbits):
                    with pgm_pkg.create_program_context("us_gate") as pc:
                        #gate = g.mem_gather()
                        us.build(unitary, target_qbit, control_qbit, gate)
        with pgm_pkg.create_program_context("final_us") as pcfinal:
            unitary.set_program_output()
        print_utils.infoc("\nAssembling model ...")
        iops = pgm_pkg.assemble()
        iop = runtime.IOProgram(iops[0])
        #device = runtime.devices[0]
        driver = runtime.Driver()
        device = driver.next_available_device()

def main():
    #test()
    num_qbits, max_gates = 8, 20
    UnitarySimulator.unit_test(num_qbits)
    #UnitarySimulator.chain_test(num_qbits, max_gates)
if __name__ == "__main__":
    main()
