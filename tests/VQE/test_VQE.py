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

    @staticmethod
    def _expected_story1_bridge_operations(vqe):
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
                raise ValueError(f"Unsupported Story 1 gate in expected bridge: {gate_name}")

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

    def test_density_backend_rejects_custom_gate_structure_source(self):
        from squander.gates.qgd_Circuit import qgd_Circuit

        qbit_num = 2
        topology = [(idx, idx + 1) for idx in range(qbit_num - 1)]
        Hamiltonian = generate_hamiltonian(topology, qbit_num)
        VQE_cls = self._get_vqe_class()
        vqe = VQE_cls(
            Hamiltonian,
            qbit_num,
            self._get_story2_config(),
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
                self._get_story2_config(),
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

    @pytest.mark.parametrize("qbit_num", [4, 6, 8, 10])
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

    @pytest.mark.parametrize("qbit_num", [4, 6])
    def test_density_matrix_bridge_metadata_matches_generated_anchor_circuit(
        self, qbit_num
    ):
        pytest.importorskip("qiskit")

        density_vqe, _ = self._build_story2_density_vqe(qbit_num)
        parameters = np.linspace(
            0.05,
            0.05 * density_vqe.get_Parameter_Num(),
            density_vqe.get_Parameter_Num(),
            dtype=np.float64,
        )
        density_vqe.set_Optimized_Parameters(parameters)
        bridge = density_vqe.describe_density_bridge()
        expected_operations = self._expected_story1_bridge_operations(density_vqe)

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

        from benchmarks.density_matrix.story2_vqe_density_validation import (
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

    def test_task3_story2_mixed_bridge_case_passes(self):
        pytest.importorskip("qiskit")
        pytest.importorskip("qiskit_aer")

        from benchmarks.density_matrix.task3_story2_bridge_validation import (
            MANDATORY_BRIDGE_MICROCASES,
            validate_bridge_microcase,
        )

        case = next(
            case
            for case in MANDATORY_BRIDGE_MICROCASES
            if case["case_name"] == "task3_story2_3q_mixed_local_noise_sequence"
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

    def test_task3_story2_bridge_bundle_schema(self):
        pytest.importorskip("qiskit")
        pytest.importorskip("qiskit_aer")

        from benchmarks.density_matrix.task3_story2_bridge_validation import (
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

    def test_task3_story3_unsupported_bridge_bundle_schema(self):
        from benchmarks.density_matrix.task3_story3_unsupported_bridge_validation import (
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

    def test_story4_workflow_bundle_schema(self):
        pytest.importorskip("qiskit")
        pytest.importorskip("qiskit_aer")

        from benchmarks.density_matrix.story2_vqe_density_validation import (
            build_story4_parameter_sets,
            build_story4_workflow_bundle,
            run_story4_workflow_case,
        )

        density_vqe, _ = self._build_story2_density_vqe(4)
        parameter_set = build_story4_parameter_sets(
            density_vqe.get_Parameter_Num(),
            count=1,
        )[0]
        result = run_story4_workflow_case(
            4,
            parameter_set["parameter_set_id"],
            parameter_set["parameter_vector"],
        )
        bundle = build_story4_workflow_bundle(
            [result],
            qubit_sizes=(4,),
            parameter_set_count=1,
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
        assert bundle["summary"]["documented_10q_anchor_present"] is False
        assert bundle["thresholds"]["absolute_energy_error"] == 1e-8

    def test_story5_trace_artifact_schema(self):
        pytest.importorskip("qiskit")
        pytest.importorskip("qiskit_aer")

        from benchmarks.density_matrix.story2_vqe_density_validation import (
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

    def test_story5_bundle_manifest_schema(self, tmp_path):
        pytest.importorskip("qiskit")
        pytest.importorskip("qiskit_aer")

        from benchmarks.density_matrix.story2_vqe_density_validation import (
            STORY2_MICRO_BUNDLE_FILENAME,
            STORY4_WORKFLOW_BUNDLE_FILENAME,
            build_story5_bundle,
        )

        (tmp_path / STORY2_MICRO_BUNDLE_FILENAME).write_text("{}\n", encoding="utf-8")
        (tmp_path / STORY4_WORKFLOW_BUNDLE_FILENAME).write_text(
            "{}\n", encoding="utf-8"
        )
        (tmp_path / "story2_fixed_4q.json").write_text("{}\n", encoding="utf-8")
        (tmp_path / "story2_fixed_6q.json").write_text("{}\n", encoding="utf-8")
        (tmp_path / "story2_trace_4q.json").write_text("{}\n", encoding="utf-8")
        (tmp_path / "story3_unsupported_state_vector_density_noise.json").write_text(
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

        bundle = build_story5_bundle(
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
            "story2_micro_validation_bundle",
            "story4_workflow_bundle",
            "story2_fixed_4q",
            "story2_fixed_6q",
            "story2_trace_4q",
            "story3_unsupported_state_vector_density_noise",
        }

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






