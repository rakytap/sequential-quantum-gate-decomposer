import numpy as np
import scipy as sp
import pytest


sigmax = sp.sparse.csr_matrix(np.array([[0,1],
                   [1,0]])+0.j)
sigmay = sp.sparse.csr_matrix(np.array([[0,0+-1j],
                   [0+1j,0]])+0.j)
sigmaz = sp.sparse.csr_matrix(np.array([[1,0],
                   [0,-1]])+0.j)


def generate_hamiltonian(topology, n):

    Hamiltonian = sp.sparse.coo_matrix((2**n, 2**n), dtype=np.complex128)
    for i in topology:
        if i[0]==0:
            lhs_1 = sp.sparse.eye(1,format='coo')
        else:
            lhs_1 = sp.sparse.eye(2,format='coo')
        for k in range(i[0]-1):
            lhs_1 = sp.sparse.kron(lhs_1,sp.sparse.eye(2,format='coo'),format='coo')
        if i[0]==n-1:
            rhs_1 = sp.sparse.eye(1,format='coo')
        else:
            rhs_1 = sp.sparse.eye(2,format='coo')
        for k in range(n-i[0]-2):
            rhs_1 = sp.sparse.kron(rhs_1,sp.sparse.eye(2,format='coo'),format='coo')
        if i[1]==0:
            lhs_2 = sp.sparse.eye(1,format='coo')
        else:
            lhs_2 = sp.sparse.eye(2,format='coo')
        for k in range(i[1]-1):
            lhs_2 = sp.sparse.kron(lhs_2,sp.sparse.eye(2,format='coo'),format='coo')
        if i[1]==n-1:
            rhs_2 = sp.sparse.eye(1,format='coo')
        else:
            rhs_2 = sp.sparse.eye(2,format='coo')
        for k in range(n-i[1]-2):
            rhs_2 = sp.sparse.kron(rhs_2,sp.sparse.eye(2,format='coo'),format='coo')
        Hamiltonian += -0.5*sp.sparse.kron(sp.sparse.kron(lhs_1,sigmax,format='coo'),rhs_1,format='coo')@sp.sparse.kron(sp.sparse.kron(lhs_2 ,sigmax,format='coo'),rhs_2 ,format='coo')
        Hamiltonian += -0.5*sp.sparse.kron(sp.sparse.kron(lhs_1,sigmay,format='coo'),rhs_1,format='coo')@sp.sparse.kron(sp.sparse.kron(lhs_2 ,sigmay,format='coo'),rhs_2 ,format='coo')
        Hamiltonian += -0.5*sp.sparse.kron(sp.sparse.kron(lhs_1,sigmaz,format='coo'),rhs_1,format='coo')@sp.sparse.kron(sp.sparse.kron(lhs_2 ,sigmaz,format='coo'),rhs_2 ,format='coo')
    for i in range(n):
    
        if i==0:
            lhs_1 = sp.sparse.eye(1,format='coo')
        else:
            lhs_1 = sp.sparse.eye(2,format='coo')
        for k in range(i-1):
            lhs_1 = sp.sparse.kron(lhs_1,sp.sparse.eye(2,format='coo'),format='coo')
        if i==n-1:
            rhs_1 = sp.sparse.eye(1,format='coo')
        else:
            rhs_1 = sp.sparse.eye(2,format='coo')
        for k in range(n-i-2):
            rhs_1 = sp.sparse.kron(rhs_1,sp.sparse.eye(2,format='coo'),format='coo')
        Hamiltonian += -0.5*sp.sparse.kron(sp.sparse.kron(lhs_1,sigmaz,format='coo'),rhs_1,format='coo')
        

    return Hamiltonian.tocsr()






class Test_VQE:

    @staticmethod
    def _get_vqe_class():
        try:
            from squander.VQA.qgd_Variational_Quantum_Eigensolver_Base import (
                qgd_Variational_Quantum_Eigensolver_Base as Variational_Quantum_Eigensolver_Base,
            )
            return Variational_Quantum_Eigensolver_Base
        except Exception as exc:
            pytest.skip(f"VQE wrapper not available: {exc}")

    @staticmethod
    def _get_identity_config():
        return {
            "agent_lifetime": 500,
            "optimization_tolerance": -7.1,
            "max_inner_iterations": 10,
            "max_iterations": 50,
            "learning_rate": 2e-1,
            "agent_num": 64,
            "agent_exploration_rate": 0.3,
            "max_inner_iterations_adam": 50000,
        }

    @staticmethod
    def _get_story2_config():
        return {
            "max_inner_iterations": 4,
            "max_iterations": 1,
            "convergence_length": 2,
        }

    @staticmethod
    def _get_story2_noise():
        return [
            {
                "channel": "local_depolarizing",
                "target": 0,
                "after_gate_index": 0,
                "error_rate": 0.1,
            },
            {
                "channel": "amplitude_damping",
                "target": 1,
                "after_gate_index": 2,
                "gamma": 0.05,
            },
            {
                "channel": "phase_damping",
                "target": 0,
                "after_gate_index": 4,
                "lambda": 0.07,
            },
        ]

    def _build_identity_vqe(self, qbit_num=2, backend=None, Hamiltonian=None):
        if Hamiltonian is None:
            Hamiltonian = sp.sparse.eye(2**qbit_num, format="csr")
        VQE_cls = self._get_vqe_class()

        if backend is None:
            vqe = VQE_cls(Hamiltonian, qbit_num, self._get_identity_config())
        else:
            vqe = VQE_cls(
                Hamiltonian,
                qbit_num,
                self._get_identity_config(),
                backend=backend,
            )

        vqe.set_Ansatz("HEA")
        vqe.Generate_Circuit(1, 1)
        return vqe

    def _build_story2_density_vqe(self, qbit_num, optimizer=None):
        topology = [(idx, idx + 1) for idx in range(qbit_num - 1)]
        Hamiltonian = generate_hamiltonian(topology, qbit_num)
        VQE_cls = self._get_vqe_class()
        vqe = VQE_cls(
            Hamiltonian,
            qbit_num,
            self._get_story2_config(),
            backend="density_matrix",
            density_noise=self._get_story2_noise(),
        )
        if optimizer is not None:
            vqe.set_Optimizer(optimizer)
        vqe.set_Ansatz("HEA")
        vqe.Generate_Circuit(1, 1)
        return vqe, Hamiltonian

    def _build_story2_state_vector_vqe_with_density_noise(
        self, qbit_num, *, omit_backend=False
    ):
        topology = [(idx, idx + 1) for idx in range(qbit_num - 1)]
        Hamiltonian = generate_hamiltonian(topology, qbit_num)
        VQE_cls = self._get_vqe_class()
        kwargs = {"density_noise": self._get_story2_noise()}
        if not omit_backend:
            kwargs["backend"] = "state_vector"
        vqe = VQE_cls(
            Hamiltonian,
            qbit_num,
            self._get_story2_config(),
            **kwargs,
        )
        vqe.set_Ansatz("HEA")
        vqe.Generate_Circuit(1, 1)
        return vqe, Hamiltonian

    @staticmethod
    def _insert_story2_noise(base_circuit, density_noise):
        pytest.importorskip("qiskit")
        pytest.importorskip("qiskit_aer")

        from qiskit import QuantumCircuit
        from qiskit_aer.noise import (
            amplitude_damping_error,
            depolarizing_error,
            phase_damping_error,
        )

        noisy_circuit = QuantumCircuit(base_circuit.num_qubits)
        noise_by_gate = {}
        for noise_spec in density_noise:
            noise_by_gate.setdefault(noise_spec["after_gate_index"], []).append(
                noise_spec
            )

        for gate_index, instruction in enumerate(base_circuit.data):
            qargs = [
                noisy_circuit.qubits[base_circuit.find_bit(qubit).index]
                for qubit in instruction.qubits
            ]
            cargs = [
                noisy_circuit.clbits[base_circuit.find_bit(clbit).index]
                for clbit in instruction.clbits
            ]
            noisy_circuit.append(instruction.operation, qargs, cargs)

            for noise_spec in noise_by_gate.get(gate_index, []):
                target = noise_spec["target"]
                channel = noise_spec["channel"]
                if channel == "local_depolarizing":
                    noisy_circuit.append(
                        depolarizing_error(noise_spec["error_rate"], 1),
                        [noisy_circuit.qubits[target]],
                    )
                elif channel == "amplitude_damping":
                    noisy_circuit.append(
                        amplitude_damping_error(noise_spec["gamma"]),
                        [noisy_circuit.qubits[target]],
                    )
                elif channel == "phase_damping":
                    noisy_circuit.append(
                        phase_damping_error(noise_spec["lambda"]),
                        [noisy_circuit.qubits[target]],
                    )
                else:
                    raise ValueError(f"Unsupported Story 2 channel: {channel}")

        noisy_circuit.save_density_matrix()
        return noisy_circuit

    @staticmethod
    def _density_energy(hamiltonian, density_matrix):
        energy = np.trace(hamiltonian.dot(density_matrix))
        return float(np.real(energy)), float(np.imag(energy))

    def _get_story2_aer_reference(self, vqe, Hamiltonian):
        pytest.importorskip("qiskit_aer")

        from qiskit_aer import AerSimulator

        noisy_qiskit_circuit = self._insert_story2_noise(
            vqe.get_Qiskit_Circuit(),
            self._get_story2_noise(),
        )
        simulator = AerSimulator(method="density_matrix")
        result = simulator.run(noisy_qiskit_circuit, shots=1).result()
        aer_rho = np.asarray(result.data()["density_matrix"])
        return self._density_energy(Hamiltonian, aer_rho)

    def test_backend_argument_normalization(self):
        qbit_num = 2
        Hamiltonian = sp.sparse.eye(2**qbit_num, format="csr")
        VQE_cls = self._get_vqe_class()

        legacy_vqe = VQE_cls(Hamiltonian, qbit_num, self._get_identity_config())
        state_vector_vqe = VQE_cls(
            Hamiltonian,
            qbit_num,
            self._get_identity_config(),
            backend="state_vector",
        )
        density_matrix_vqe = VQE_cls(
            Hamiltonian,
            qbit_num,
            self._get_identity_config(),
            backend="density_matrix",
        )

        assert legacy_vqe.backend == "state_vector"
        assert state_vector_vqe.backend == "state_vector"
        assert density_matrix_vqe.backend == "density_matrix"

    def test_invalid_backend_rejected(self):
        qbit_num = 2
        Hamiltonian = sp.sparse.eye(2**qbit_num, format="csr")
        VQE_cls = self._get_vqe_class()

        with pytest.raises(ValueError, match="Unsupported backend"):
            VQE_cls(
                Hamiltonian,
                qbit_num,
                self._get_identity_config(),
                backend="invalid_backend",
            )

    def test_density_noise_normalization(self):
        qbit_num = 4
        Hamiltonian = sp.sparse.eye(2**qbit_num, format="csr")
        VQE_cls = self._get_vqe_class()

        vqe = VQE_cls(
            Hamiltonian,
            qbit_num,
            self._get_story2_config(),
            backend="density_matrix",
            density_noise=self._get_story2_noise(),
        )

        assert vqe.backend == "density_matrix"
        assert vqe.density_noise == [
            {
                "channel": "local_depolarizing",
                "target": 0,
                "after_gate_index": 0,
                "value": 0.1,
            },
            {
                "channel": "amplitude_damping",
                "target": 1,
                "after_gate_index": 2,
                "value": 0.05,
            },
            {
                "channel": "phase_damping",
                "target": 0,
                "after_gate_index": 4,
                "value": 0.07,
            },
        ]

    @pytest.mark.parametrize("omit_backend", [False, True])
    def test_state_vector_rejects_density_noise(self, omit_backend):
        qbit_num = 4
        vqe, _ = self._build_story2_state_vector_vqe_with_density_noise(
            qbit_num, omit_backend=omit_backend
        )

        parameters = np.linspace(
            0.05,
            0.05 * vqe.get_Parameter_Num(),
            vqe.get_Parameter_Num(),
            dtype=np.float64,
        )

        with pytest.raises(
            Exception,
            match="state_vector backend does not support density_matrix-only noise configuration",
        ):
            vqe.Optimization_Problem(parameters)

    def test_density_backend_rejects_unsupported_ansatz(self):
        qbit_num = 4
        topology = [(idx, idx + 1) for idx in range(qbit_num - 1)]
        Hamiltonian = generate_hamiltonian(topology, qbit_num)
        VQE_cls = self._get_vqe_class()
        vqe = VQE_cls(
            Hamiltonian,
            qbit_num,
            self._get_story2_config(),
            backend="density_matrix",
            density_noise=self._get_story2_noise(),
        )
        vqe.set_Ansatz("HEA_ZYZ")
        vqe.Generate_Circuit(1, 1)

        parameters = np.linspace(
            0.05,
            0.05 * vqe.get_Parameter_Num(),
            vqe.get_Parameter_Num(),
            dtype=np.float64,
        )

        with pytest.raises(Exception, match="only the HEA ansatz"):
            vqe.Optimization_Problem(parameters)

    def test_density_backend_requires_generated_hea_circuit(self):
        qbit_num = 4
        topology = [(idx, idx + 1) for idx in range(qbit_num - 1)]
        Hamiltonian = generate_hamiltonian(topology, qbit_num)
        VQE_cls = self._get_vqe_class()
        vqe = VQE_cls(
            Hamiltonian,
            qbit_num,
            self._get_story2_config(),
            backend="density_matrix",
            density_noise=self._get_story2_noise(),
        )

        with pytest.raises(Exception, match="requires a generated HEA circuit"):
            vqe.Optimization_Problem(np.array([], dtype=np.float64))

    def test_density_backend_rejects_invalid_after_gate_index(self):
        qbit_num = 4
        topology = [(idx, idx + 1) for idx in range(qbit_num - 1)]
        Hamiltonian = generate_hamiltonian(topology, qbit_num)
        VQE_cls = self._get_vqe_class()
        vqe = VQE_cls(
            Hamiltonian,
            qbit_num,
            self._get_story2_config(),
            backend="density_matrix",
            density_noise=[
                {
                    "channel": "local_depolarizing",
                    "target": 0,
                    "after_gate_index": 999,
                    "error_rate": 0.1,
                }
            ],
        )
        vqe.set_Ansatz("HEA")
        vqe.Generate_Circuit(1, 1)

        parameters = np.linspace(
            0.05,
            0.05 * vqe.get_Parameter_Num(),
            vqe.get_Parameter_Num(),
            dtype=np.float64,
        )

        with pytest.raises(
            Exception, match="after_gate_index exceeds generated gate count"
        ):
            vqe.Optimization_Problem(parameters)

    def test_density_backend_rejects_gradient_entrypoint(self):
        vqe, _ = self._build_story2_density_vqe(4)
        parameters = np.linspace(
            0.05,
            0.05 * vqe.get_Parameter_Num(),
            vqe.get_Parameter_Num(),
            dtype=np.float64,
        )

        with pytest.raises(Exception, match="does not support gradient-based optimization"):
            vqe.Optimization_Problem_Grad(parameters)

    def test_invalid_optimizer_rejected(self):
        vqe = self._build_identity_vqe()

        with pytest.raises(Exception, match="Unsupported optimizer"):
            vqe.set_Optimizer("NOT_A_REAL_OPTIMIZER")

    def test_invalid_ansatz_rejected(self):
        vqe = self._build_identity_vqe()

        with pytest.raises(Exception, match="Unsupported ansatz"):
            vqe.set_Ansatz("NOT_A_REAL_ANSATZ")

    def test_explicit_state_vector_matches_legacy_default(self):
        qbit_num = 2
        Hamiltonian = generate_hamiltonian([(0, 1)], qbit_num)
        legacy_vqe = self._build_identity_vqe(qbit_num=qbit_num, Hamiltonian=Hamiltonian)
        explicit_state_vector_vqe = self._build_identity_vqe(
            qbit_num=qbit_num,
            Hamiltonian=Hamiltonian,
            backend="state_vector",
        )

        assert legacy_vqe.backend == "state_vector"
        assert explicit_state_vector_vqe.backend == "state_vector"
        assert legacy_vqe.get_Parameter_Num() == explicit_state_vector_vqe.get_Parameter_Num()

        param_num = legacy_vqe.get_Parameter_Num()
        assert param_num > 0

        parameters = np.linspace(0.05, 0.05 * param_num, param_num, dtype=np.float64)
        legacy_energy = legacy_vqe.Optimization_Problem(parameters)
        explicit_state_vector_energy = explicit_state_vector_vqe.Optimization_Problem(parameters)

        assert np.isclose(legacy_energy, explicit_state_vector_energy, atol=1e-12)

    @pytest.mark.parametrize("qbit_num", [4, 6])
    def test_density_matrix_backend_anchor_fixed_parameter_smoke(self, qbit_num):
        density_vqe, Hamiltonian = self._build_story2_density_vqe(qbit_num)
        VQE_cls = self._get_vqe_class()
        state_vector_vqe = VQE_cls(
            Hamiltonian,
            qbit_num,
            self._get_story2_config(),
            backend="state_vector",
        )
        state_vector_vqe.set_Ansatz("HEA")
        state_vector_vqe.Generate_Circuit(1, 1)

        param_num = density_vqe.get_Parameter_Num()
        parameters = np.linspace(0.05, 0.05 * param_num, param_num, dtype=np.float64)

        density_energy = density_vqe.Optimization_Problem(parameters)
        state_vector_energy = state_vector_vqe.Optimization_Problem(parameters)

        assert density_vqe.backend == "density_matrix"
        assert np.isfinite(density_energy)
        assert np.isfinite(state_vector_energy)
        assert not np.isclose(density_energy, state_vector_energy, atol=1e-8)

    def test_density_matrix_backend_anchor_fixed_parameter_matches_aer_reference(self):
        density_vqe, Hamiltonian = self._build_story2_density_vqe(4)

        param_num = density_vqe.get_Parameter_Num()
        parameters = np.linspace(0.05, 0.05 * param_num, param_num, dtype=np.float64)
        density_vqe.set_Optimized_Parameters(parameters)

        density_energy = float(density_vqe.Optimization_Problem(parameters))
        aer_energy_real, aer_energy_imag = self._get_story2_aer_reference(
            density_vqe, Hamiltonian
        )

        assert density_vqe.backend == "density_matrix"
        assert np.isfinite(density_energy)
        assert np.isclose(density_energy, aer_energy_real, atol=1e-12)
        assert abs(aer_energy_imag) <= 1e-12

    def test_density_matrix_backend_bounded_optimization_trace(self):
        density_vqe, _ = self._build_story2_density_vqe(4, optimizer="COSINE")

        param_num = density_vqe.get_Parameter_Num()
        parameters = np.linspace(0.05, 0.05 * param_num, param_num, dtype=np.float64)
        density_vqe.set_Optimized_Parameters(parameters)

        initial_energy = density_vqe.Optimization_Problem(parameters)
        density_vqe.Start_Optimization()
        final_parameters = density_vqe.get_Optimized_Parameters()
        final_energy = density_vqe.Optimization_Problem(final_parameters)

        assert np.isfinite(initial_energy)
        assert np.isfinite(final_energy)
        assert final_parameters.shape == parameters.shape

    def test_density_backend_rejects_unsupported_optimizer_on_start(self):
        density_vqe, _ = self._build_story2_density_vqe(4, optimizer="ADAM")

        param_num = density_vqe.get_Parameter_Num()
        parameters = np.linspace(0.05, 0.05 * param_num, param_num, dtype=np.float64)
        density_vqe.set_Optimized_Parameters(parameters)

        with pytest.raises(
            Exception,
            match="supports only BAYES_OPT or COSINE optimization traces",
        ):
            density_vqe.Start_Optimization()

    def test_VQE_Identity(self):
        layers = 1
        blocks = 1

        qbit_num = 7
        Hamiltonian = sp.sparse.eye(2**qbit_num,format="csr")

        config = self._get_identity_config()
        
        # initiate the VQE object with the Hamiltonian
        VQE_cls = self._get_vqe_class()
        VQE_eye = VQE_cls(Hamiltonian, qbit_num, config)

        # set the optimization engine
        VQE_eye.set_Optimizer("GRAD_DESCEND")

        # set the ansatz variant (U3 rotations and CNOT gates)
        VQE_eye.set_Ansatz("HEA")

        # generate the circuit ansatz for the optimization
        VQE_eye.Generate_Circuit(layers, blocks)

        # create initial parameters 
        param_num  = VQE_eye.get_Parameter_Num()
        parameters = np.random.random( (param_num,) )

        VQE_eye.set_Optimized_Parameters(parameters)

        # start an etap of the optimization (max_inner_iterations iterations)
        VQE_eye.Start_Optimization()
        
        # retrieve QISKIT format of the optimized circuit
        quantum_circuit = VQE_eye.get_Qiskit_Circuit()

        # retrieve the optimized parameter
        parameters = VQE_eye.get_Optimized_Parameters()

        # evaluate the VQE energy at the optimized parameters
        Energy = VQE_eye.Optimization_Problem( parameters )


        assert (abs(Energy-1)<1e-4)
    
    def test_Heisenberg_XX(self):
    
        layers = 1
        blocks = 1

        qbit_num = 9
        topology = []
        for idx in range(qbit_num-1):
            topology.append( (idx, idx+1) )
            
        Hamiltonian = generate_hamiltonian(topology, qbit_num)
        
        config = {"agent_lifetime":10,
                  "max_inner_iterations":1000,
                  "max_iterations":1000,
                  "agent_num":3,
                  "agent_exploration_rate":0.5,
                  "max_inner_iterations_adam":500,
                  "convergence_length": 300}
        
        # initiate the VQE object with the Hamiltonian
        VQE_cls = self._get_vqe_class()
        VQE_Heisenberg = VQE_cls(Hamiltonian, qbit_num, config)

        # set the optimization engine
        VQE_Heisenberg.set_Optimizer("AGENTS")

        # set the ansatz variant (U3 rotations and CNOT gates)
        VQE_Heisenberg.set_Ansatz("HEA_ZYZ")

        # generate the circuit ansatz for the optimization        
        VQE_Heisenberg.Generate_Circuit(layers, blocks)

        # create initial parameters 
        param_num  = VQE_Heisenberg.get_Parameter_Num()
        parameters = np.random.random( (param_num,) )

        VQE_Heisenberg.set_Optimized_Parameters(parameters)

        # start an etap of the optimization (max_inner_iterations iterations)
        VQE_Heisenberg.Start_Optimization()

        # retrieve QISKIT format of the optimized circuit
        quantum_circuit = VQE_Heisenberg.get_Qiskit_Circuit()
        
        # print the quantum circuit
        lambdas, vecs = sp.sparse.linalg.eigs(Hamiltonian)

        # retrieve the optimized parameter
        parameters = VQE_Heisenberg.get_Optimized_Parameters()

        # evaluate the VQE energy at the optimized parameters
        Energy = VQE_Heisenberg.Optimization_Problem( parameters )
        
        print('Expected energy: ', np.real(lambdas[0]))
        print('Obtained energy: ', Energy)
        assert ((Energy - np.real(lambdas[0]))<1e-2)






