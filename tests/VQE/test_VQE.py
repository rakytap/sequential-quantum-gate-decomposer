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
    def _get_density_backend_config():
        return {
            "max_inner_iterations": 4,
            "max_iterations": 1,
            "convergence_length": 2,
        }

    @staticmethod
    def _get_density_backend_noise():
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

    @staticmethod
    def _get_required_local_noise_cases():
        return (
            {
                "case_name": "required_local_noise_4q_local_depolarizing_positive",
                "qbit_num": 4,
                "requested_noise_channel": "local_depolarizing",
                "expected_target": 0,
                "expected_after_gate_index": 0,
                "expected_value": 0.1,
                "density_noise": [
                    {
                        "channel": "local_depolarizing",
                        "target": 0,
                        "after_gate_index": 0,
                        "error_rate": 0.1,
                    }
                ],
            },
            {
                "case_name": "required_local_noise_4q_amplitude_damping_positive",
                "qbit_num": 4,
                "requested_noise_channel": "amplitude_damping",
                "expected_target": 1,
                "expected_after_gate_index": 2,
                "expected_value": 0.05,
                "density_noise": [
                    {
                        "channel": "amplitude_damping",
                        "target": 1,
                        "after_gate_index": 2,
                        "gamma": 0.05,
                    }
                ],
            },
            {
                "case_name": "required_local_noise_4q_phase_damping_positive",
                "qbit_num": 4,
                "requested_noise_channel": "phase_damping",
                "expected_target": 0,
                "expected_after_gate_index": 4,
                "expected_value": 0.07,
                "density_noise": [
                    {
                        "channel": "phase_damping",
                        "target": 0,
                        "after_gate_index": 4,
                        "lambda": 0.07,
                    }
                ],
            },
        )

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

    def _build_density_backend_vqe(self, qbit_num, optimizer=None, density_noise=None):
        topology = [(idx, idx + 1) for idx in range(qbit_num - 1)]
        Hamiltonian = generate_hamiltonian(topology, qbit_num)
        VQE_cls = self._get_vqe_class()
        if density_noise is None:
            density_noise = self._get_density_backend_noise()
        vqe = VQE_cls(
            Hamiltonian,
            qbit_num,
            self._get_density_backend_config(),
            backend="density_matrix",
            density_noise=density_noise,
        )
        if optimizer is not None:
            vqe.set_Optimizer(optimizer)
        vqe.set_Ansatz("HEA")
        vqe.Generate_Circuit(1, 1)
        return vqe, Hamiltonian

    def _build_state_vector_vqe_with_density_noise(
        self, qbit_num, *, omit_backend=False
    ):
        topology = [(idx, idx + 1) for idx in range(qbit_num - 1)]
        Hamiltonian = generate_hamiltonian(topology, qbit_num)
        VQE_cls = self._get_vqe_class()
        kwargs = {"density_noise": self._get_density_backend_noise()}
        if not omit_backend:
            kwargs["backend"] = "state_vector"
        vqe = VQE_cls(
            Hamiltonian,
            qbit_num,
            self._get_density_backend_config(),
            **kwargs,
        )
        vqe.set_Ansatz("HEA")
        vqe.Generate_Circuit(1, 1)
        return vqe, Hamiltonian

    @staticmethod
    def _insert_reference_noise(base_circuit, density_noise):
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
                    raise ValueError(
                        f"Unsupported density-backend channel: {channel}"
                    )

        noisy_circuit.save_density_matrix()
        return noisy_circuit

    @staticmethod
    def _density_energy(hamiltonian, density_matrix):
        energy = np.trace(hamiltonian.dot(density_matrix))
        return float(np.real(energy)), float(np.imag(energy))

    def _get_density_backend_aer_reference(self, vqe, Hamiltonian):
        pytest.importorskip("qiskit_aer")

        from qiskit_aer import AerSimulator

        noisy_qiskit_circuit = self._insert_reference_noise(
            vqe.get_Qiskit_Circuit(),
            self._get_density_backend_noise(),
        )
        simulator = AerSimulator(method="density_matrix")
        result = simulator.run(noisy_qiskit_circuit, shots=1).result()
        aer_rho = np.asarray(result.data()["density_matrix"])
        return self._density_energy(Hamiltonian, aer_rho)

    @staticmethod
    def _expected_bridge_operations(vqe):
        pytest.importorskip("qiskit")

        base_circuit = vqe.get_Qiskit_Circuit()
        noise_by_gate = {}
        for noise_spec in vqe.density_noise:
            noise_by_gate.setdefault(noise_spec["after_gate_index"], []).append(noise_spec)

        expected = []
        param_start = 0
        for gate_index, instruction in enumerate(base_circuit.data):
            qubit_indices = [
                base_circuit.find_bit(qubit).index for qubit in instruction.qubits
            ]
            gate_name = instruction.operation.name

            if gate_name == "u":
                expected.append(
                    {
                        "operation_class": "GateOperation",
                        "kind": "gate",
                        "name": "U3",
                        "is_unitary": True,
                        "source_gate_index": gate_index,
                        "target_qbit": qubit_indices[0],
                        "control_qbit": None,
                        "param_count": 3,
                        "param_start": param_start,
                        "fixed_value": None,
                    }
                )
                param_start += 3
            elif gate_name in {"cx", "cnot"}:
                expected.append(
                    {
                        "operation_class": "GateOperation",
                        "kind": "gate",
                        "name": "CNOT",
                        "is_unitary": True,
                        "source_gate_index": gate_index,
                        "target_qbit": qubit_indices[1],
                        "control_qbit": qubit_indices[0],
                        "param_count": 0,
                        "param_start": param_start,
                        "fixed_value": None,
                    }
                )
            else:
                raise ValueError(
                    f"Unsupported gate in expected bridge: {gate_name}"
                )

            for noise_spec in noise_by_gate.get(gate_index, []):
                expected.append(
                    {
                        "operation_class": "NoiseOperation",
                        "kind": "noise",
                        "name": noise_spec["channel"],
                        "is_unitary": False,
                        "source_gate_index": gate_index,
                        "target_qbit": noise_spec["target"],
                        "control_qbit": None,
                        "param_count": 0,
                        "param_start": param_start,
                        "fixed_value": noise_spec["value"],
                    }
                )

        return expected

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
            self._get_density_backend_config(),
            backend="density_matrix",
            density_noise=self._get_density_backend_noise(),
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
        vqe, _ = self._build_state_vector_vqe_with_density_noise(
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
            self._get_density_backend_config(),
            backend="density_matrix",
            density_noise=self._get_density_backend_noise(),
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
            self._get_density_backend_config(),
            backend="density_matrix",
            density_noise=self._get_density_backend_noise(),
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
            self._get_density_backend_config(),
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

    def test_density_backend_rejects_custom_gate_structure_source(self):
        from squander.gates.qgd_Circuit import qgd_Circuit

        qbit_num = 2
        topology = [(idx, idx + 1) for idx in range(qbit_num - 1)]
        Hamiltonian = generate_hamiltonian(topology, qbit_num)
        VQE_cls = self._get_vqe_class()
        vqe = VQE_cls(
            Hamiltonian,
            qbit_num,
            self._get_density_backend_config(),
            backend="density_matrix",
        )
        custom_circuit = qgd_Circuit(qbit_num)
        custom_circuit.add_H(0)
        custom_circuit.add_CNOT(1, 0)
        vqe.set_Gate_Structure(custom_circuit)

        with pytest.raises(
            Exception,
            match="unsupported circuit source in density backend path: custom_gate_structure",
        ):
            vqe.describe_density_bridge()

    def test_density_backend_rejects_unsupported_density_noise_channel(self):
        qbit_num = 1
        Hamiltonian = generate_hamiltonian([], qbit_num)
        VQE_cls = self._get_vqe_class()

        with pytest.raises(ValueError, match="Unsupported density-noise channel"):
            VQE_cls(
                Hamiltonian,
                qbit_num,
                self._get_density_backend_config(),
                backend="density_matrix",
                density_noise=[
                    {
                        "channel": "readout_noise",
                        "target": 0,
                        "after_gate_index": 0,
                        "value": 0.1,
                    }
                ],
            )

    def test_density_noise_aliases_normalize_to_required_local_models(self):
        from squander.VQA.qgd_Variational_Quantum_Eigensolver_Base import (
            _normalize_density_noise_spec,
        )

        normalized = _normalize_density_noise_spec(
            [
                {
                    "channel": "depolarizing",
                    "target": 0,
                    "after_gate_index": 0,
                    "error_rate": 0.1,
                },
                {
                    "channel": "dephasing",
                    "target": 1,
                    "after_gate_index": 2,
                    "lambda": 0.05,
                },
            ]
        )

        assert [item["channel"] for item in normalized] == [
            "local_depolarizing",
            "phase_damping",
        ]
        assert [item["target"] for item in normalized] == [0, 1]
        assert [item["after_gate_index"] for item in normalized] == [0, 2]
        assert [item["value"] for item in normalized] == pytest.approx([0.1, 0.05])

    @pytest.mark.parametrize(
        "channel",
        [
            "readout_noise",
            "correlated_multi_qubit_noise",
            "calibration_aware_noise",
            "non_markovian_noise",
        ],
    )
    def test_deferred_density_noise_families_fail_before_execution(self, channel):
        qbit_num = 1
        Hamiltonian = generate_hamiltonian([], qbit_num)
        VQE_cls = self._get_vqe_class()

        with pytest.raises(
            ValueError,
            match="Unsupported density-noise channel '{}'".format(channel),
        ):
            VQE_cls(
                Hamiltonian,
                qbit_num,
                self._get_density_backend_config(),
                backend="density_matrix",
                density_noise=[
                    {
                        "channel": channel,
                        "target": 0,
                        "after_gate_index": 0,
                        "value": 0.1,
                    }
                ],
            )

    def test_density_backend_rejects_negative_after_gate_index(self):
        qbit_num = 2
        topology = [(idx, idx + 1) for idx in range(qbit_num - 1)]
        Hamiltonian = generate_hamiltonian(topology, qbit_num)
        VQE_cls = self._get_vqe_class()

        with pytest.raises(Exception, match="after_gate_index must be non-negative"):
            VQE_cls(
                Hamiltonian,
                qbit_num,
                self._get_density_backend_config(),
                backend="density_matrix",
                density_noise=[
                    {
                        "channel": "local_depolarizing",
                        "target": 0,
                        "after_gate_index": -1,
                        "error_rate": 0.1,
                    }
                ],
            )

    def test_density_backend_rejects_target_out_of_range(self):
        qbit_num = 2
        topology = [(idx, idx + 1) for idx in range(qbit_num - 1)]
        Hamiltonian = generate_hamiltonian(topology, qbit_num)
        VQE_cls = self._get_vqe_class()

        with pytest.raises(Exception, match="target_qbit out of range"):
            VQE_cls(
                Hamiltonian,
                qbit_num,
                self._get_density_backend_config(),
                backend="density_matrix",
                density_noise=[
                    {
                        "channel": "phase_damping",
                        "target": 10,
                        "after_gate_index": 0,
                        "lambda": 0.1,
                    }
                ],
            )

    def test_density_backend_rejects_noise_value_out_of_range(self):
        qbit_num = 2
        topology = [(idx, idx + 1) for idx in range(qbit_num - 1)]
        Hamiltonian = generate_hamiltonian(topology, qbit_num)
        VQE_cls = self._get_vqe_class()

        with pytest.raises(Exception, match="noise values must be in \\[0, 1\\]"):
            VQE_cls(
                Hamiltonian,
                qbit_num,
                self._get_density_backend_config(),
                backend="density_matrix",
                density_noise=[
                    {
                        "channel": "amplitude_damping",
                        "target": 0,
                        "after_gate_index": 0,
                        "gamma": 1.5,
                    }
                ],
            )

    def test_density_backend_rejects_gradient_entrypoint(self):
        vqe, _ = self._build_density_backend_vqe(4)
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

    @pytest.mark.parametrize("qbit_num", [4, 6, 8, 10])
    def test_density_matrix_backend_anchor_fixed_parameter_smoke(self, qbit_num):
        density_vqe, Hamiltonian = self._build_density_backend_vqe(qbit_num)
        VQE_cls = self._get_vqe_class()
        state_vector_vqe = VQE_cls(
            Hamiltonian,
            qbit_num,
            self._get_density_backend_config(),
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

    @pytest.mark.parametrize("qbit_num", [4, 6])
    def test_density_matrix_bridge_metadata_matches_generated_anchor_circuit(
        self, qbit_num
    ):
        pytest.importorskip("qiskit")

        density_vqe, _ = self._build_density_backend_vqe(qbit_num)
        parameters = np.linspace(
            0.05,
            0.05 * density_vqe.get_Parameter_Num(),
            density_vqe.get_Parameter_Num(),
            dtype=np.float64,
        )
        density_vqe.set_Optimized_Parameters(parameters)
        bridge = density_vqe.describe_density_bridge()
        expected_operations = self._expected_bridge_operations(density_vqe)

        assert bridge["backend"] == "density_matrix"
        assert bridge["source_type"] == "generated_hea"
        assert bridge["qbit_num"] == qbit_num
        assert bridge["parameter_count"] == density_vqe.get_Parameter_Num()
        assert bridge["gate_count"] == density_vqe.get_Qiskit_Circuit().size()
        assert bridge["noise_count"] == len(density_vqe.density_noise)
        assert bridge["operation_count"] == len(expected_operations)
        assert len(bridge["operations"]) == len(expected_operations)

        for actual, expected in zip(bridge["operations"], expected_operations):
            assert actual["operation_class"] == expected["operation_class"]
            assert actual["kind"] == expected["kind"]
            assert actual["name"] == expected["name"]
            assert actual["is_unitary"] == expected["is_unitary"]
            assert actual["source_gate_index"] == expected["source_gate_index"]
            assert actual["target_qbit"] == expected["target_qbit"]
            assert actual["control_qbit"] == expected["control_qbit"]
            assert actual["param_count"] == expected["param_count"]
            assert actual["param_start"] == expected["param_start"]
            if expected["fixed_value"] is None:
                assert actual["fixed_value"] is None
            else:
                assert actual["fixed_value"] == pytest.approx(expected["fixed_value"])

    def test_fixed_parameter_artifact_includes_bridge_metadata(self):
        pytest.importorskip("qiskit")
        pytest.importorskip("qiskit_aer")

        from benchmarks.density_matrix.workflow_evidence.exact_density_vqe_validation import (
            run_fixed_parameter_case,
        )

        artifact = run_fixed_parameter_case(4)

        assert artifact["status"] == "completed"
        assert artifact["bridge_source_type"] == "generated_hea"
        assert artifact["bridge_operation_count"] == len(artifact["bridge_operations"])
        assert artifact["bridge_gate_count"] + artifact["bridge_noise_count"] == artifact["bridge_operation_count"]
        assert artifact["bridge_noise_count"] == len(artifact["density_noise"])
        assert artifact["bridge_parameter_count"] > 0
        assert artifact["bridge_operations"][0]["operation_class"] == "GateOperation"
        assert any(
            op["operation_class"] == "NoiseOperation"
            for op in artifact["bridge_operations"]
        )

    def test_required_local_noise_validation_models_execute_on_supported_path(self):
        for case in self._get_required_local_noise_cases():
            density_vqe, _ = self._build_density_backend_vqe(
                case["qbit_num"],
                density_noise=case["density_noise"],
            )
            parameters = np.linspace(
                0.05,
                0.05 * density_vqe.get_Parameter_Num(),
                density_vqe.get_Parameter_Num(),
                dtype=np.float64,
            )

            density_vqe.set_Optimized_Parameters(parameters)
            density_energy = density_vqe.Optimization_Problem(parameters)
            bridge = density_vqe.describe_density_bridge()
            noise_ops = [
                op
                for op in bridge["operations"]
                if op["operation_class"] == "NoiseOperation"
            ]

            assert density_vqe.backend == "density_matrix"
            assert np.isfinite(density_energy)
            assert bridge["source_type"] == "generated_hea"
            assert bridge["noise_count"] == 1
            assert len(noise_ops) == 1
            assert [op["name"] for op in noise_ops] == [case["requested_noise_channel"]]
            assert all(op["name"] != "depolarizing" for op in noise_ops)
            assert [op["target_qbit"] for op in noise_ops] == [case["expected_target"]]
            assert [op["source_gate_index"] for op in noise_ops] == [
                case["expected_after_gate_index"]
            ]
            assert [op["fixed_value"] for op in noise_ops] == pytest.approx(
                [case["expected_value"]]
            )

    def test_required_local_noise_bundle_schema(self):
        pytest.importorskip("qiskit")
        pytest.importorskip("qiskit_aer")

        from benchmarks.density_matrix.noise_support.required_local_noise_validation import (
            REQUIRED_LOCAL_NOISE_CASES,
            REQUIRED_LOCAL_NOISE_MODELS,
            build_artifact_bundle,
            run_validation,
        )

        results = run_validation(verbose=False)
        bundle = build_artifact_bundle(results)

        assert bundle["status"] == "pass"
        assert bundle["backend"] == "density_matrix"
        assert bundle["summary"]["total_cases"] == len(REQUIRED_LOCAL_NOISE_CASES)
        assert bundle["summary"]["passed_cases"] == len(REQUIRED_LOCAL_NOISE_CASES)
        assert bundle["summary"]["pass_rate"] == 1.0
        assert bundle["summary"]["required_cases"] == len(REQUIRED_LOCAL_NOISE_CASES)
        assert bundle["summary"]["required_passed_cases"] == len(REQUIRED_LOCAL_NOISE_CASES)
        assert bundle["summary"]["required_pass_rate"] == 1.0
        assert bundle["summary"]["optional_cases"] == 0
        assert bundle["summary"]["optional_cases_count_toward_mandatory_baseline"] == 0
        assert bundle["summary"]["mandatory_baseline_completed"]
        assert bundle["summary"]["whole_register_substitute_failures"] == 0
        assert set(bundle["requirements"]["required_local_noise_models"]) == set(
            REQUIRED_LOCAL_NOISE_MODELS
        )
        assert "required" in bundle["requirements"]["support_tier_vocabulary"]
        assert {case["requested_noise_channel"] for case in bundle["cases"]} == set(
            REQUIRED_LOCAL_NOISE_MODELS
        )
        assert all(case["status"] == "pass" for case in bundle["cases"])
        assert all(case["positive_slice_pass"] for case in bundle["cases"])
        assert all(case["support_tier"] == "required" for case in bundle["cases"])
        assert all(case["case_purpose"] == "mandatory_baseline" for case in bundle["cases"])
        assert all(case["counts_toward_mandatory_baseline"] for case in bundle["cases"])
        assert all(
            case["bridge_noise_sequence"] == [case["requested_noise_channel"]]
            for case in bundle["cases"]
        )

    def test_bridge_scope_mixed_bridge_case_passes(self):
        pytest.importorskip("qiskit")
        pytest.importorskip("qiskit_aer")

        from benchmarks.density_matrix.bridge_scope.bridge_validation import (
            MANDATORY_BRIDGE_MICROCASES,
            validate_bridge_microcase,
        )

        case = next(
            case
            for case in MANDATORY_BRIDGE_MICROCASES
            if case["case_name"] == "bridge_3q_mixed_local_noise_sequence"
        )
        result = validate_bridge_microcase(case, verbose=False)

        assert result["status"] == "pass"
        assert result["source_pass"]
        assert result["gate_pass"]
        assert result["noise_pass"]
        assert result["operation_match_pass"]
        assert result["execution_ready"]
        assert result["bridge_source_type"] == "generated_hea"
        assert result["bridge_noise_count"] == 3
        assert any(
            op["name"] == "CNOT" and op["operation_class"] == "GateOperation"
            for op in result["bridge_operations"]
        )

    def test_bridge_scope_bundle_schema(self):
        pytest.importorskip("qiskit")
        pytest.importorskip("qiskit_aer")

        from benchmarks.density_matrix.bridge_scope.bridge_validation import (
            MANDATORY_BRIDGE_MICROCASES,
            build_artifact_bundle,
            run_validation,
        )

        results = run_validation(verbose=False)
        bundle = build_artifact_bundle(results)

        assert bundle["status"] == "pass"
        assert bundle["backend"] == "density_matrix"
        assert bundle["requirements"]["source_type"] == "generated_hea"
        assert bundle["summary"]["total_cases"] == len(MANDATORY_BRIDGE_MICROCASES)
        assert bundle["summary"]["passed_cases"] == len(MANDATORY_BRIDGE_MICROCASES)
        assert bundle["summary"]["pass_rate"] == 1.0
        assert {case["qbit_num"] for case in bundle["cases"]} == {1, 2, 3}
        assert all(case["status"] == "pass" for case in bundle["cases"])
        assert all("bridge_operations" in case for case in bundle["cases"])

    def test_unsupported_bridge_bundle_schema(self):
        from benchmarks.density_matrix.bridge_scope.unsupported_bridge_validation import (
            UNSUPPORTED_CASE_BUILDERS,
            build_artifact_bundle,
            run_validation,
        )

        results = run_validation(verbose=False)
        bundle = build_artifact_bundle(results)

        assert bundle["status"] == "pass"
        assert bundle["backend"] == "density_matrix"
        assert bundle["summary"]["total_cases"] == len(UNSUPPORTED_CASE_BUILDERS)
        assert bundle["summary"]["unsupported_cases"] == len(
            UNSUPPORTED_CASE_BUILDERS
        )
        assert bundle["summary"]["error_match_count"] == len(
            UNSUPPORTED_CASE_BUILDERS
        )
        assert {case["unsupported_category"] for case in bundle["cases"]} == {
            "circuit_source",
            "lowering_path",
            "noise_insertion",
            "noise_type",
        }
        assert all(case["status"] == "unsupported" for case in bundle["cases"])
        assert all(case["error_match_pass"] for case in bundle["cases"])

    def test_unsupported_noise_bundle_schema(self):
        pytest.importorskip("qiskit")
        pytest.importorskip("qiskit_aer")

        from benchmarks.density_matrix.noise_support.unsupported_noise_validation import (
            UNSUPPORTED_NOISE_BOUNDARY_CASES,
            build_artifact_bundle,
            run_validation,
        )

        results = run_validation(verbose=False)
        bundle = build_artifact_bundle(results)

        assert bundle["status"] == "pass"
        assert bundle["backend"] == "density_matrix"
        assert bundle["summary"]["total_cases"] == len(UNSUPPORTED_NOISE_BOUNDARY_CASES)
        assert bundle["summary"]["unsupported_status_cases"] == len(
            UNSUPPORTED_NOISE_BOUNDARY_CASES
        )
        assert bundle["summary"]["error_match_count"] == len(
            UNSUPPORTED_NOISE_BOUNDARY_CASES
        )
        assert bundle["summary"]["pre_execution_failure_count"] == len(
            UNSUPPORTED_NOISE_BOUNDARY_CASES
        )
        assert bundle["summary"]["boundary_passed_cases"] == len(
            UNSUPPORTED_NOISE_BOUNDARY_CASES
        )
        assert bundle["summary"]["deferred_cases"] == 4
        assert bundle["summary"]["unsupported_cases"] == 4
        assert bundle["summary"]["support_tiers_present"] == [
            "deferred",
            "unsupported",
        ]
        assert bundle["summary"]["silent_substitution_failures"] == 0
        assert bundle["summary"]["silent_fallback_failures"] == 0
        assert set(bundle["summary"]["categories_present"]) == {
            "noise_insertion",
            "noise_target",
            "noise_type",
            "noise_value",
        }
        assert set(bundle["summary"]["boundary_classes_present"]) == {
            "configuration",
            "model_family",
            "schedule_element",
        }
        assert set(bundle["summary"]["failure_stages_present"]) == {
            "cxx_noise_spec_validation",
            "density_anchor_preflight",
            "python_normalization",
        }
        assert {
            case["first_unsupported_condition"] for case in bundle["cases"]
        } >= {
            "readout_noise",
            "correlated_multi_qubit_noise",
            "calibration_aware_noise",
            "non_markovian_noise",
            "after_gate_index_negative",
            "after_gate_index_exceeds_gate_count",
            "target_qbit_out_of_range",
            "noise_value_out_of_range",
        }
        assert all(case["status"] == "unsupported" for case in bundle["cases"])
        assert all(case["error_match_pass"] for case in bundle["cases"])
        assert all(case["unsupported_boundary_pass"] for case in bundle["cases"])
        assert all(
            not case["counts_toward_mandatory_baseline"] for case in bundle["cases"]
        )

    def test_exact_regime_workflow_case_includes_bridge_metadata(self):
        pytest.importorskip("qiskit")
        pytest.importorskip("qiskit_aer")

        from benchmarks.density_matrix.workflow_evidence.exact_density_vqe_validation import (
            build_exact_regime_parameter_sets,
            run_exact_regime_workflow_case,
        )

        density_vqe, _ = self._build_density_backend_vqe(4)
        parameter_set = build_exact_regime_parameter_sets(
            density_vqe.get_Parameter_Num(),
            count=1,
        )[0]
        result = run_exact_regime_workflow_case(
            4,
            parameter_set["parameter_set_id"],
            parameter_set["parameter_vector"],
        )

        assert result["status"] == "pass"
        assert result["bridge_source_type"] == "generated_hea"
        assert result["source_pass"]
        assert result["gate_pass"]
        assert result["noise_pass"]
        assert result["execution_ready"]
        assert result["bridge_supported_pass"]
        assert result["bridge_gate_count"] > 0
        assert result["bridge_noise_count"] == len(result["density_noise"])
        assert "U3" in result["bridge_gate_sequence"]
        assert "CNOT" in result["bridge_gate_sequence"]
        assert set(result["bridge_noise_sequence"]) == {
            "local_depolarizing",
            "amplitude_damping",
            "phase_damping",
        }
        assert result["support_tier"] == "required"
        assert result["case_purpose"] == "mandatory_baseline"
        assert result["counts_toward_mandatory_baseline"] is True

    def test_exact_regime_workflow_bundle_schema(self):
        pytest.importorskip("qiskit")
        pytest.importorskip("qiskit_aer")

        from benchmarks.density_matrix.workflow_evidence.exact_density_vqe_validation import (
            build_exact_regime_parameter_sets,
            build_exact_regime_workflow_bundle,
            run_optimization_trace,
            run_exact_regime_workflow_case,
        )

        density_vqe, _ = self._build_density_backend_vqe(4)
        parameter_set = build_exact_regime_parameter_sets(
            density_vqe.get_Parameter_Num(),
            count=1,
        )[0]
        result = run_exact_regime_workflow_case(
            4,
            parameter_set["parameter_set_id"],
            parameter_set["parameter_vector"],
        )
        trace_result = run_optimization_trace()
        bundle = build_exact_regime_workflow_bundle(
            [result],
            qubit_sizes=(4,),
            parameter_set_count=1,
            trace_result=trace_result,
        )

        assert result["status"] == "pass"
        assert result["workflow_completed"]
        assert result["energy_pass"]
        assert result["density_valid_pass"]
        assert result["trace_pass"]
        assert result["observable_pass"]
        assert result["qbit_num"] == 4
        assert bundle["status"] == "pass"
        assert bundle["summary"]["total_cases"] == 1
        assert bundle["summary"]["passed_cases"] == 1
        assert bundle["summary"]["unsupported_cases"] == 0
        assert bundle["summary"]["unsupported_status_cases"] == 0
        assert bundle["summary"]["bridge_supported_cases"] == 1
        assert bundle["summary"]["documented_10q_anchor_present"] is False
        assert bundle["summary"]["supported_trace_completed"] is True
        assert bundle["summary"]["supported_trace_case_name"] == trace_result.get(
            "case_name", "optimization_trace_4q"
        )
        assert bundle["summary"]["required_cases"] == 1
        assert bundle["summary"]["required_passed_cases"] == 1
        assert bundle["summary"]["required_pass_rate"] == 1.0
        assert bundle["summary"]["mandatory_baseline_case_count"] == 1
        assert bundle["summary"]["mandatory_baseline_passed_cases"] == 1
        assert bundle["summary"]["mandatory_baseline_completed"] is True
        assert bundle["summary"]["support_tiers_present"] == ["required"]
        assert bundle["summary"]["optional_cases_count_toward_mandatory_baseline"] == 0
        assert bundle["thresholds"]["absolute_energy_error"] == 1e-8

    def test_trace_artifact_includes_bridge_metadata(self):
        pytest.importorskip("qiskit")
        pytest.importorskip("qiskit_aer")

        from benchmarks.density_matrix.workflow_evidence.exact_density_vqe_validation import (
            run_optimization_trace,
        )

        artifact = run_optimization_trace()

        assert artifact["status"] == "completed"
        assert artifact["bridge_source_type"] == "generated_hea"
        assert artifact["source_pass"]
        assert artifact["gate_pass"]
        assert artifact["noise_pass"]
        assert artifact["execution_ready"]
        assert artifact["bridge_supported_pass"]
        assert artifact["bridge_gate_count"] > 0
        assert artifact["bridge_noise_count"] == len(artifact["density_noise"])
        assert artifact["support_tier"] == "required"
        assert artifact["case_purpose"] == "mandatory_baseline"
        assert artifact["counts_toward_mandatory_baseline"] is False
        assert artifact["required_validation_trace"] is True

    def test_trace_artifact_schema(self):
        pytest.importorskip("qiskit")
        pytest.importorskip("qiskit_aer")

        from benchmarks.density_matrix.workflow_evidence.exact_density_vqe_validation import (
            run_optimization_trace,
        )

        artifact = run_optimization_trace()

        assert artifact["status"] == "completed"
        assert artifact["workflow_completed"] is True
        assert artifact["trace_kind"] == "bounded_optimization_trace"
        assert artifact["optimizer"] == "COSINE"
        assert artifact["parameter_count"] == len(artifact["initial_parameters"])
        assert artifact["total_trace_runtime_ms"] >= 0.0
        assert artifact["process_peak_rss_kb"] > 0
        assert artifact["energy_improvement"] == (
            artifact["initial_energy"] - artifact["final_energy"]
        )
        assert artifact["optimizer_config"]["max_iterations"] == 1
        assert artifact["support_tier"] == "required"
        assert artifact["case_purpose"] == "mandatory_baseline"
        assert artifact["counts_toward_mandatory_baseline"] is False
        assert artifact["required_validation_trace"] is True

    def test_required_local_noise_workflow_bundle_schema(self, tmp_path):
        pytest.importorskip("qiskit")
        pytest.importorskip("qiskit_aer")

        from benchmarks.density_matrix.workflow_evidence.exact_density_vqe_validation import write_json
        from benchmarks.density_matrix.noise_support.required_local_noise_workflow_validation import (
            TRACE_ARTIFACT_FILENAME,
            TRACE_CASE_NAME,
            WORKFLOW_BUNDLE_FILENAME,
            build_artifact_bundle,
            run_validation,
            write_artifact_bundle,
        )

        workflow_results, trace_result, bundle = run_validation(
            qubit_sizes=(4,),
            parameter_set_count=1,
            verbose=False,
        )
        bundle = build_artifact_bundle(
            workflow_results,
            trace_result,
            qubit_sizes=(4,),
            parameter_set_count=1,
        )
        write_artifact_bundle(
            tmp_path / WORKFLOW_BUNDLE_FILENAME,
            bundle,
            trace_result=trace_result,
        )
        write_json(tmp_path / TRACE_ARTIFACT_FILENAME, trace_result)

        assert bundle["status"] == "pass"
        assert bundle["suite_name"] == "required_local_noise_workflow"
        assert bundle["summary"]["required_cases"] == 1
        assert bundle["summary"]["required_passed_cases"] == 1
        assert bundle["summary"]["required_pass_rate"] == 1.0
        assert bundle["summary"]["unsupported_status_cases"] == 0
        assert bundle["summary"]["required_trace_case_name"] == TRACE_CASE_NAME
        assert bundle["summary"]["required_trace_present"] is True
        assert bundle["summary"]["required_trace_completed"] is True
        assert bundle["summary"]["required_trace_bridge_supported"] is True
        assert (
            bundle["summary"]["required_trace_counts_toward_mandatory_baseline"]
            is False
        )
        assert trace_result["case_name"] == TRACE_CASE_NAME
        assert trace_result["support_tier"] == "required"
        assert (tmp_path / WORKFLOW_BUNDLE_FILENAME).exists()
        assert (tmp_path / TRACE_ARTIFACT_FILENAME).exists()

    def test_workflow_baseline_bundle_schema(self):
        pytest.importorskip("qiskit")
        pytest.importorskip("qiskit_aer")

        from benchmarks.density_matrix.validation_evidence.workflow_baseline_validation import (
            run_validation,
        )

        exact_regime_workflow_bundle, bundle = run_validation(
            qubit_sizes=(4,),
            parameter_set_count=1,
            verbose=False,
        )

        assert exact_regime_workflow_bundle["status"] == "pass"
        assert bundle["status"] == "pass"
        assert bundle["suite_name"] == "workflow_baseline_validation"
        assert bundle["backend"] == "density_matrix"
        assert bundle["summary"]["total_cases"] == 1
        assert bundle["summary"]["passed_cases"] == 1
        assert bundle["summary"]["unsupported_cases"] == 0
        assert bundle["summary"]["required_cases"] == 1
        assert bundle["summary"]["required_passed_cases"] == 1
        assert bundle["summary"]["required_pass_rate"] == 1.0
        assert bundle["summary"]["stable_case_ids_present"] is True
        assert bundle["summary"]["stable_parameter_set_ids_present"] is True
        assert bundle["summary"]["workflow_baseline_completed"] is True
        assert bundle["summary"]["documented_10q_anchor_present"] is False
        assert bundle["required_artifacts"]["exact_regime_workflow_reference"]["status"] == "pass"
        assert len(bundle["cases"]) == 1
        assert bundle["cases"][0]["parameter_set_id"] == "set_00"

    def test_workflow_baseline_missing_case_blocks_closure(self):
        pytest.importorskip("qiskit")
        pytest.importorskip("qiskit_aer")

        import copy

        from benchmarks.density_matrix.validation_evidence.workflow_baseline_validation import (
            build_artifact_bundle,
            run_validation,
        )

        exact_regime_workflow_bundle, bundle = run_validation(
            qubit_sizes=(4,),
            parameter_set_count=1,
            verbose=False,
        )
        broken_workflow_results = copy.deepcopy(bundle["cases"])
        omitted_case = broken_workflow_results.pop()

        broken_bundle = build_artifact_bundle(
            exact_regime_workflow_bundle,
            broken_workflow_results,
            qubit_sizes=(4,),
            parameter_set_count=1,
        )

        assert broken_bundle["status"] == "fail"
        assert broken_bundle["summary"]["stable_case_ids_present"] is False
        assert broken_bundle["summary"]["stable_parameter_set_ids_present"] is False
        assert broken_bundle["summary"]["workflow_baseline_completed"] is False
        assert (
            omitted_case["case_name"]
            in broken_bundle["summary"]["missing_mandatory_case_names"]
        )

    def test_trace_anchor_bundle_schema(self):
        pytest.importorskip("qiskit")
        pytest.importorskip("qiskit_aer")

        from benchmarks.density_matrix.validation_evidence.trace_anchor_validation import (
            run_validation,
        )

        workflow_baseline_bundle, trace_result, bundle = run_validation(
            qubit_sizes=(10,),
            parameter_set_count=1,
            verbose=False,
        )

        assert workflow_baseline_bundle["status"] == "pass"
        assert trace_result["status"] == "completed"
        assert bundle["status"] == "pass"
        assert bundle["suite_name"] == "trace_anchor_validation"
        assert bundle["summary"]["workflow_baseline_completed"] is True
        assert bundle["summary"]["documented_10q_anchor_present"] is True
        assert bundle["summary"]["required_trace_case_name"] == "optimization_trace_4q"
        assert bundle["summary"]["required_trace_present"] is True
        assert bundle["summary"]["required_trace_completed"] is True
        assert bundle["summary"]["required_trace_bridge_supported"] is True
        assert bundle["summary"]["trace_and_anchor_gate_completed"] is True
        assert bundle["required_artifacts"]["workflow_baseline_reference"]["status"] == "pass"
        assert bundle["trace_artifact"]["case_name"] == "optimization_trace_4q"

    def test_trace_anchor_missing_trace_marker_blocks_closure(self):
        pytest.importorskip("qiskit")
        pytest.importorskip("qiskit_aer")

        import copy

        from benchmarks.density_matrix.validation_evidence.trace_anchor_validation import (
            build_artifact_bundle,
            run_validation,
        )

        workflow_baseline_bundle, trace_result, _ = run_validation(
            qubit_sizes=(10,),
            parameter_set_count=1,
            verbose=False,
        )
        broken_trace_result = copy.deepcopy(trace_result)
        broken_trace_result["required_validation_trace"] = False

        bundle = build_artifact_bundle(
            workflow_baseline_bundle,
            broken_trace_result,
            qubit_sizes=(10,),
            parameter_set_count=1,
        )

        assert bundle["status"] == "fail"
        assert bundle["summary"]["required_trace_present"] is False
        assert bundle["summary"]["trace_and_anchor_gate_completed"] is False

    def test_metric_completeness_bundle_schema(self):
        pytest.importorskip("qiskit")
        pytest.importorskip("qiskit_aer")

        from benchmarks.density_matrix.validation_evidence.metric_completeness_validation import (
            run_validation,
        )

        _, _, _, bundle = run_validation(
            qubit_sizes=(10,),
            parameter_set_count=1,
            verbose=False,
        )

        assert bundle["status"] == "pass"
        assert bundle["suite_name"] == "metric_completeness_validation"
        assert bundle["summary"]["micro_cases_checked"] == 7
        assert bundle["summary"]["workflow_cases_checked"] == 1
        assert bundle["summary"]["trace_artifacts_checked"] == 1
        assert bundle["summary"]["micro_cases_missing_required_metrics"] == 0
        assert bundle["summary"]["workflow_cases_missing_required_metrics"] == 0
        assert bundle["summary"]["trace_artifacts_missing_required_metrics"] == 0
        assert bundle["summary"]["workflow_cases_with_stable_execution"] == 1
        assert bundle["summary"]["trace_execution_stability_pass"] is True
        assert bundle["summary"]["metric_completeness_gate_completed"] is True
        assert bundle["required_artifacts"]["local_correctness_reference"]["status"] == "pass"
        assert bundle["required_artifacts"]["workflow_baseline_reference"]["status"] == "pass"
        assert bundle["required_artifacts"]["trace_anchor_reference"]["status"] == "pass"

    def test_metric_completeness_missing_workflow_metric_blocks_gate(self):
        pytest.importorskip("qiskit")
        pytest.importorskip("qiskit_aer")

        import copy

        from benchmarks.density_matrix.validation_evidence.metric_completeness_validation import (
            build_artifact_bundle,
            run_validation,
        )

        local_correctness_bundle, workflow_baseline_bundle, trace_anchor_bundle, _ = run_validation(
            qubit_sizes=(10,),
            parameter_set_count=1,
            verbose=False,
        )
        broken_workflow_baseline_bundle = copy.deepcopy(workflow_baseline_bundle)
        broken_case = broken_workflow_baseline_bundle["cases"][0]
        del broken_case["process_peak_rss_kb"]

        bundle = build_artifact_bundle(
            local_correctness_bundle,
            broken_workflow_baseline_bundle,
            trace_anchor_bundle,
        )

        assert bundle["status"] == "fail"
        assert bundle["summary"]["workflow_cases_missing_required_metrics"] == 1
        assert "process_peak_rss_kb" in bundle["summary"]["missing_workflow_metric_fields_by_case"][
            broken_case["case_name"]
        ]
        assert bundle["summary"]["metric_completeness_gate_completed"] is False

    def test_interpretation_bundle_schema(self):
        pytest.importorskip("qiskit")
        pytest.importorskip("qiskit_aer")

        from benchmarks.density_matrix.validation_evidence.interpretation_validation import (
            run_validation,
        )

        metric_completeness_bundle, optional_bundle, unsupported_bundle, bundle = run_validation(
            qubit_sizes=(10,),
            parameter_set_count=1,
            verbose=False,
        )

        assert metric_completeness_bundle["status"] == "pass"
        assert optional_bundle["status"] == "pass"
        assert unsupported_bundle["status"] == "pass"
        assert bundle["status"] == "pass"
        assert bundle["suite_name"] == "validation_evidence_interpretation"
        assert bundle["summary"]["mandatory_artifacts_complete"] is True
        assert bundle["summary"]["optional_evidence_supplemental"] is True
        assert bundle["summary"]["unsupported_evidence_negative_only"] is True
        assert bundle["summary"]["main_validation_claim_completed"] is True
        assert bundle["summary"]["incomplete_mandatory_artifacts"] == []
        assert bundle["required_artifacts"]["metric_completeness_validation"]["status"] == "pass"
        assert (
            bundle["required_artifacts"]["optional_noise_reference"]["status"]
            == "pass"
        )
        assert bundle["required_artifacts"]["unsupported_noise_reference"]["status"] == "pass"

    def test_validation_interpretation_incomplete_mandatory_artifact_blocks_main_claim(self):
        pytest.importorskip("qiskit")
        pytest.importorskip("qiskit_aer")

        import copy

        from benchmarks.density_matrix.validation_evidence.interpretation_validation import (
            build_artifact_bundle,
            run_validation,
        )

        metric_completeness_bundle, optional_bundle, unsupported_bundle, _ = run_validation(
            qubit_sizes=(10,),
            parameter_set_count=1,
            verbose=False,
        )
        broken_metric_completeness_bundle = copy.deepcopy(metric_completeness_bundle)
        broken_metric_completeness_bundle["required_artifacts"]["workflow_baseline_reference"][
            "status"
        ] = "fail"

        bundle = build_artifact_bundle(
            broken_metric_completeness_bundle,
            optional_bundle,
            unsupported_bundle,
        )

        assert bundle["status"] == "fail"
        assert "workflow_baseline_reference" in bundle["summary"][
            "incomplete_mandatory_artifacts"
        ]
        assert bundle["summary"]["mandatory_artifacts_complete"] is False
        assert bundle["summary"]["main_validation_claim_completed"] is False

    def test_validation_evidence_publication_bundle_schema(self, tmp_path):
        pytest.importorskip("qiskit")
        pytest.importorskip("qiskit_aer")

        from benchmarks.density_matrix.validation_evidence.validation_evidence_publication_bundle import (
            ARTIFACT_FILENAME,
            generate_validation_evidence_publication_bundle,
        )

        bundle = generate_validation_evidence_publication_bundle(
            tmp_path,
            qubit_sizes=(10,),
            parameter_set_count=1,
            verbose=False,
        )

        assert bundle["status"] == "pass"
        assert bundle["suite_name"] == "validation_evidence_publication"
        assert bundle["summary"]["mandatory_artifact_count"] == 6
        assert bundle["summary"]["present_artifact_count"] == 6
        assert bundle["summary"]["status_match_count"] == 6
        assert bundle["summary"]["missing_artifact_count"] == 0
        assert bundle["summary"]["mismatched_status_count"] == 0
        assert bundle["summary"]["raw_trace_reference_pass"] is True
        assert bundle["provenance"]["git_revision"]
        assert {artifact["artifact_id"] for artifact in bundle["artifacts"]} == {
            "local_correctness_bundle",
            "workflow_baseline_bundle",
            "trace_anchor_bundle",
            "optimization_trace_4q",
            "metric_completeness_bundle",
            "interpretation_bundle",
        }
        assert (tmp_path / ARTIFACT_FILENAME).exists()

    def test_validation_evidence_publication_missing_trace_file_fails_bundle_validation(self, tmp_path):
        pytest.importorskip("qiskit")
        pytest.importorskip("qiskit_aer")

        from benchmarks.density_matrix.validation_evidence.validation_evidence_publication_bundle import (
            TRACE_ARTIFACT_FILENAME,
            generate_validation_evidence_publication_bundle,
            validate_validation_evidence_publication_bundle,
        )

        bundle = generate_validation_evidence_publication_bundle(
            tmp_path,
            qubit_sizes=(10,),
            parameter_set_count=1,
            verbose=False,
        )
        (tmp_path / TRACE_ARTIFACT_FILENAME).unlink()

        with pytest.raises(ValueError, match="missing artifact file"):
            validate_validation_evidence_publication_bundle(bundle, tmp_path)

    def test_noise_support_publication_bundle_schema(self, tmp_path):
        pytest.importorskip("qiskit")
        pytest.importorskip("qiskit_aer")

        from benchmarks.density_matrix.noise_support.noise_support_publication_bundle import (
            ARTIFACT_FILENAME,
            generate_noise_support_publication_bundle,
        )

        bundle = generate_noise_support_publication_bundle(
            tmp_path,
            qubit_sizes=(4,),
            parameter_set_count=1,
            verbose=False,
        )

        assert bundle["status"] == "pass"
        assert bundle["suite_name"] == "noise_support_publication_evidence"
        assert bundle["summary"]["mandatory_artifact_count"] == 6
        assert bundle["summary"]["present_artifact_count"] == 6
        assert bundle["summary"]["status_match_count"] == 6
        assert bundle["summary"]["missing_artifact_count"] == 0
        assert bundle["summary"]["mismatched_status_count"] == 0
        assert bundle["summary"]["workflow_trace_reference_pass"] is True
        assert bundle["provenance"]["git_revision"]
        assert {artifact["artifact_id"] for artifact in bundle["artifacts"]} == {
            "required_local_noise_bundle",
            "required_local_noise_micro_bundle",
            "optional_noise_classification_bundle",
            "unsupported_noise_bundle",
            "required_local_noise_workflow_bundle",
            "required_local_noise_trace_4q",
        }
        assert (tmp_path / ARTIFACT_FILENAME).exists()

    def test_exact_density_validation_bundle_manifest_schema(self, tmp_path):
        pytest.importorskip("qiskit")
        pytest.importorskip("qiskit_aer")

        from benchmarks.density_matrix.workflow_evidence.exact_density_vqe_validation import (
            MICRO_VALIDATION_BUNDLE_FILENAME,
            EXACT_REGIME_WORKFLOW_BUNDLE_FILENAME,
            build_exact_density_validation_bundle,
        )

        (tmp_path / MICRO_VALIDATION_BUNDLE_FILENAME).write_text("{}\n", encoding="utf-8")
        (tmp_path / EXACT_REGIME_WORKFLOW_BUNDLE_FILENAME).write_text(
            "{}\n", encoding="utf-8"
        )
        (tmp_path / "fixed_parameters_4q.json").write_text("{}\n", encoding="utf-8")
        (tmp_path / "fixed_parameters_6q.json").write_text("{}\n", encoding="utf-8")
        (tmp_path / "optimization_trace_4q.json").write_text("{}\n", encoding="utf-8")
        (tmp_path / "unsupported_state_vector_density_noise.json").write_text(
            "{}\n", encoding="utf-8"
        )

        micro_bundle = {
            "status": "pass",
            "summary": {"total_cases": 7, "passed_cases": 7, "pass_rate": 1.0},
        }
        workflow_bundle = {
            "status": "pass",
            "summary": {
                "total_cases": 40,
                "passed_cases": 40,
                "pass_rate": 1.0,
                "documented_10q_anchor_present": True,
            },
        }
        fixed_results = [
            {
                "status": "completed",
                "qbit_num": 4,
                "absolute_energy_error": 1e-12,
            },
            {
                "status": "completed",
                "qbit_num": 6,
                "absolute_energy_error": 1e-12,
            },
        ]
        trace_result = {
            "status": "completed",
            "optimizer": "COSINE",
            "parameter_count": 18,
            "workflow_completed": True,
            "initial_energy": 1.0,
            "final_energy": -1.0,
        }
        unsupported_result = {
            "status": "unsupported",
            "unsupported_category": "phase2_support_matrix",
            "unsupported_reason": "example",
        }

        bundle = build_exact_density_validation_bundle(
            tmp_path,
            fixed_results=fixed_results,
            trace_result=trace_result,
            unsupported_result=unsupported_result,
            micro_bundle=micro_bundle,
            workflow_bundle=workflow_bundle,
        )

        assert bundle["status"] == "pass"
        assert bundle["summary"]["mandatory_artifact_count"] == 6
        assert bundle["summary"]["present_artifact_count"] == 6
        assert bundle["summary"]["status_match_count"] == 6
        assert bundle["provenance"]["git_revision"]
        assert {artifact["artifact_id"] for artifact in bundle["artifacts"]} == {
            "micro_validation_bundle",
            "exact_regime_workflow_bundle",
            "fixed_parameters_4q",
            "fixed_parameters_6q",
            "optimization_trace_4q",
            "unsupported_state_vector_density_noise",
        }

    def test_bridge_publication_bundle_manifest_schema(self, tmp_path):
        from benchmarks.density_matrix.bridge_scope.bridge_publication_bundle import (
            BRIDGE_MICRO_VALIDATION_BUNDLE_FILENAME,
            BRIDGE_PUBLICATION_BUNDLE_FILENAME,
            UNSUPPORTED_BRIDGE_BUNDLE_FILENAME,
            build_bridge_publication_bundle,
        )
        from benchmarks.density_matrix.workflow_evidence.exact_density_vqe_validation import (
            OPTIMIZATION_TRACE_ARTIFACT_FILENAME,
            EXACT_REGIME_WORKFLOW_BUNDLE_FILENAME,
        )

        (tmp_path / "fixed_parameters_4q.json").write_text("{}\n", encoding="utf-8")
        (tmp_path / "fixed_parameters_6q.json").write_text("{}\n", encoding="utf-8")
        (tmp_path / OPTIMIZATION_TRACE_ARTIFACT_FILENAME).write_text(
            "{}\n", encoding="utf-8"
        )
        (tmp_path / EXACT_REGIME_WORKFLOW_BUNDLE_FILENAME).write_text(
            "{}\n", encoding="utf-8"
        )
        (tmp_path / BRIDGE_MICRO_VALIDATION_BUNDLE_FILENAME).write_text(
            "{}\n", encoding="utf-8"
        )
        (tmp_path / UNSUPPORTED_BRIDGE_BUNDLE_FILENAME).write_text(
            "{}\n", encoding="utf-8"
        )

        fixed_results = [
            {
                "status": "completed",
                "qbit_num": 4,
                "absolute_energy_error": 1e-12,
                "bridge_source_type": "generated_hea",
                "bridge_operation_count": 12,
                "bridge_noise_count": 3,
            },
            {
                "status": "completed",
                "qbit_num": 6,
                "absolute_energy_error": 1e-12,
                "bridge_source_type": "generated_hea",
                "bridge_operation_count": 18,
                "bridge_noise_count": 3,
            },
        ]
        trace_result = {
            "status": "completed",
            "case_name": "optimization_trace_4q",
            "optimizer": "COSINE",
            "parameter_count": 18,
            "workflow_completed": True,
            "bridge_supported_pass": True,
            "initial_energy": 1.0,
            "final_energy": -1.0,
        }
        bridge_micro_validation_bundle = {
            "status": "pass",
            "requirements": {"microcase_qubits": [1, 2, 3]},
            "summary": {"total_cases": 5, "passed_cases": 5, "pass_rate": 1.0},
        }
        unsupported_bridge_bundle = {
            "status": "pass",
            "requirements": {
                "required_categories": [
                    "circuit_source",
                    "lowering_path",
                    "noise_insertion",
                    "noise_type",
                ]
            },
            "summary": {
                "total_cases": 4,
                "unsupported_cases": 4,
                "error_match_count": 4,
                "required_case_count": 4,
            },
        }
        workflow_bundle = {
            "status": "pass",
            "summary": {
                "total_cases": 40,
                "passed_cases": 40,
                "pass_rate": 1.0,
                "unsupported_cases": 0,
                "bridge_supported_cases": 40,
                "documented_10q_anchor_present": True,
                "supported_trace_completed": True,
                "supported_trace_case_name": "optimization_trace_4q",
            },
        }

        bundle = build_bridge_publication_bundle(
            tmp_path,
            fixed_results=fixed_results,
            trace_result=trace_result,
            bridge_micro_validation_bundle=bridge_micro_validation_bundle,
            unsupported_bridge_bundle=unsupported_bridge_bundle,
            workflow_bundle=workflow_bundle,
        )

        assert bundle["status"] == "pass"
        assert bundle["summary"]["mandatory_artifact_count"] == 6
        assert bundle["summary"]["present_artifact_count"] == 6
        assert bundle["summary"]["status_match_count"] == 6
        assert bundle["provenance"]["git_revision"]
        assert {artifact["artifact_id"] for artifact in bundle["artifacts"]} == {
            "bridge_fixed_parameters_4q",
            "bridge_fixed_parameters_6q",
            "bridge_micro_validation_bundle",
            "bridge_exact_regime_workflow_bundle",
            "bridge_optimization_trace_4q",
            "unsupported_bridge_bundle",
        }
        assert (tmp_path / BRIDGE_PUBLICATION_BUNDLE_FILENAME).name == BRIDGE_PUBLICATION_BUNDLE_FILENAME

    def test_density_matrix_backend_anchor_fixed_parameter_matches_aer_reference(self):
        density_vqe, Hamiltonian = self._build_density_backend_vqe(4)

        param_num = density_vqe.get_Parameter_Num()
        parameters = np.linspace(0.05, 0.05 * param_num, param_num, dtype=np.float64)
        density_vqe.set_Optimized_Parameters(parameters)

        density_energy = float(density_vqe.Optimization_Problem(parameters))
        aer_energy_real, aer_energy_imag = self._get_density_backend_aer_reference(
            density_vqe, Hamiltonian
        )

        assert density_vqe.backend == "density_matrix"
        assert np.isfinite(density_energy)
        assert np.isclose(density_energy, aer_energy_real, atol=1e-12)
        assert abs(aer_energy_imag) <= 1e-12

    def test_density_matrix_backend_bounded_optimization_trace(self):
        density_vqe, _ = self._build_density_backend_vqe(4, optimizer="COSINE")

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
        density_vqe, _ = self._build_density_backend_vqe(4, optimizer="ADAM")

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






