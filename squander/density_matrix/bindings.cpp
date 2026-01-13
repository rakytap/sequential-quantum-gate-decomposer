/*
Copyright 2025 SQUANDER Contributors

pybind11 Bindings for Density Matrix Module - Approach B Implementation
*/

#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "density_matrix.h"
#include "density_operation.h"
#include "gate_operation.h"
#include "matrix.h"
#include "matrix_real.h"
#include "noise_channel.h"
#include "noise_operation.h"
#include "noisy_circuit.h"

namespace py = pybind11;
using namespace squander::density;

// ===================================================================
// NumPy <-> DensityMatrix conversion helpers
// ===================================================================

DensityMatrix numpy_to_density_matrix(py::array_t<std::complex<double>> arr) {
  auto buf = arr.request();

  if (buf.ndim != 2) {
    throw std::runtime_error("Input must be 2D array");
  }
  if (buf.shape[0] != buf.shape[1]) {
    throw std::runtime_error("Input must be square matrix");
  }

  int dim = buf.shape[0];

  // Check if dimension is power of 2
  int temp = dim;
  while (temp > 1 && temp % 2 == 0) {
    temp /= 2;
  }
  if (temp != 1) {
    throw std::runtime_error("Matrix dimension must be power of 2");
  }

  // Copy data to DensityMatrix
  int qbit_num = 0;
  temp = dim;
  while (temp > 1) {
    temp /= 2;
    qbit_num++;
  }

  DensityMatrix rho(qbit_num);

  auto *src = static_cast<std::complex<double> *>(buf.ptr);
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      std::complex<double> val = src[i * dim + j];
      rho(i, j).real = val.real();
      rho(i, j).imag = val.imag();
    }
  }

  return rho;
}

py::array_t<std::complex<double>>
density_matrix_to_numpy(const DensityMatrix &rho) {
  int dim = rho.get_dim();

  py::array_t<std::complex<double>> result({dim, dim});
  auto buf = result.request();
  auto *ptr = static_cast<std::complex<double> *>(buf.ptr);

  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      QGD_Complex16 val = rho(i, j);
      ptr[i * dim + j] = std::complex<double>(val.real, val.imag);
    }
  }

  return result;
}

// ===================================================================
// Module Definition
// ===================================================================

PYBIND11_MODULE(_density_matrix_cpp, m) {
  m.doc() = R"pbdoc(
        SQUANDER Density Matrix Module - C++ Backend (Approach B)
        
        High-performance density matrix simulation with:
        - Optimized local kernel application (O(4^N) per gate)
        - Unified interface for gates and noise channels
        - Support for parametric noise
    )pbdoc";

  // ===============================================================
  // DensityMatrix class
  // ===============================================================

  py::class_<DensityMatrix>(m, "DensityMatrix", R"pbdoc(
        Quantum density matrix ρ for mixed-state representation.
        
        Features:
        - Automatic memory management
        - Optimized local unitary application
        - Quantum properties (purity, entropy, eigenvalues)
    )pbdoc")

      .def(py::init<int>(), py::arg("qbit_num"),
           "Create density matrix for n qubits (initialized to |0⟩⟨0|)")

      .def(py::init([](py::array_t<std::complex<double>> state) {
             auto buf = state.request();

             if (buf.ndim != 1) {
               throw std::runtime_error("State vector must be 1D");
             }

             int dim = buf.shape[0];

             Matrix state_vec(dim, 1);
             auto *src = static_cast<std::complex<double> *>(buf.ptr);
             for (int i = 0; i < dim; i++) {
               state_vec.get_data()[i].real = src[i].real();
               state_vec.get_data()[i].imag = src[i].imag();
             }

             return DensityMatrix(state_vec);
           }),
           py::arg("state_vector"), "Create from state vector: ρ = |ψ⟩⟨ψ|")

      .def_property_readonly("qbit_num", &DensityMatrix::get_qbit_num)
      .def_property_readonly("dim", &DensityMatrix::get_dim)

      .def(
          "trace",
          [](const DensityMatrix &self) {
            QGD_Complex16 tr = self.trace();
            return std::complex<double>(tr.real, tr.imag);
          },
          "Calculate trace: Tr(ρ)")

      .def("purity", &DensityMatrix::purity, "Calculate purity: Tr(ρ²)")
      .def("entropy", &DensityMatrix::entropy,
           "von Neumann entropy: S(ρ) = -Tr(ρ log₂ ρ)")
      .def("is_valid", &DensityMatrix::is_valid, py::arg("tol") = 1e-10,
           "Check if valid density matrix")
      .def("eigenvalues", &DensityMatrix::eigenvalues,
           "Get eigenvalues (sorted descending)")

      .def(
          "apply_unitary",
          [](DensityMatrix &self, py::array_t<std::complex<double>> U) {
            auto buf = U.request();
            if (buf.ndim != 2 || buf.shape[0] != buf.shape[1]) {
              throw std::runtime_error("Unitary must be square 2D array");
            }

            int dim = buf.shape[0];
            Matrix U_mat(dim, dim);
            auto *src = static_cast<std::complex<double> *>(buf.ptr);
            for (int i = 0; i < dim * dim; i++) {
              U_mat.get_data()[i].real = src[i].real();
              U_mat.get_data()[i].imag = src[i].imag();
            }

            self.apply_unitary(U_mat);
          },
          py::arg("U"), "Apply unitary: ρ → UρU†")

      .def(
          "apply_single_qubit_unitary",
          [](DensityMatrix &self, py::array_t<std::complex<double>> u,
             int target) {
            auto buf = u.request();
            if (buf.ndim != 2 || buf.shape[0] != 2 || buf.shape[1] != 2) {
              throw std::runtime_error("Kernel must be 2x2");
            }

            Matrix u_mat(2, 2);
            auto *src = static_cast<std::complex<double> *>(buf.ptr);
            for (int i = 0; i < 4; i++) {
              u_mat.get_data()[i].real = src[i].real();
              u_mat.get_data()[i].imag = src[i].imag();
            }

            self.apply_single_qubit_unitary(u_mat, target);
          },
          py::arg("u_2x2"), py::arg("target_qbit"),
          "Apply single-qubit unitary using optimized local kernel")

      .def("partial_trace", &DensityMatrix::partial_trace, py::arg("trace_out"),
           "Partial trace over specified qubits")
      .def("clone", &DensityMatrix::clone, "Create a deep copy")

      .def_static("maximally_mixed", &DensityMatrix::maximally_mixed,
                  py::arg("qbit_num"), "Create maximally mixed state: ρ = I/2^n")

      .def(
          "to_numpy",
          [](const DensityMatrix &self) {
            return density_matrix_to_numpy(self);
          },
          "Convert to NumPy array")

      .def_static(
          "from_numpy",
          [](py::array_t<std::complex<double>> arr) {
            return numpy_to_density_matrix(arr);
          },
          py::arg("array"), "Create from NumPy array")

      .def("__repr__", [](const DensityMatrix &self) {
        return "<DensityMatrix: " + std::to_string(self.get_qbit_num()) +
               " qubits, purity=" + std::to_string(self.purity()) + ">";
      });

  // ===============================================================
  // OperationInfo struct
  // ===============================================================

  py::class_<NoisyCircuit::OperationInfo>(m, "OperationInfo",
                                          "Information about a circuit operation")
      .def_readonly("name", &NoisyCircuit::OperationInfo::name)
      .def_readonly("is_unitary", &NoisyCircuit::OperationInfo::is_unitary)
      .def_readonly("param_count", &NoisyCircuit::OperationInfo::param_count)
      .def_readonly("param_start", &NoisyCircuit::OperationInfo::param_start)
      .def("__repr__", [](const NoisyCircuit::OperationInfo &self) {
        return "<OperationInfo: " + self.name +
               (self.is_unitary ? " (gate)" : " (noise)") +
               ", params=" + std::to_string(self.param_count) + ">";
      });

  // ===============================================================
  // NoisyCircuit class
  // ===============================================================

  py::class_<NoisyCircuit>(m, "NoisyCircuit", R"pbdoc(
        Noisy quantum circuit for density matrix simulation.
        
        Supports unitary gates and noise channels with unified interface.
        Uses optimized local kernel application for O(4^N) per-gate performance.
        
        Example:
            circuit = NoisyCircuit(2)
            circuit.add_H(0)
            circuit.add_CNOT(1, 0)
            circuit.add_depolarizing(2, error_rate=0.01)
            circuit.add_RZ(0)
            circuit.add_phase_damping(0)  # Parametric
            
            rho = DensityMatrix(2)
            params = np.array([0.5, 0.02])  # RZ angle, phase damping λ
            circuit.apply_to(params, rho)
    )pbdoc")

      .def(py::init<int>(), py::arg("qbit_num"),
           "Create empty circuit for n qubits")

      // Single-qubit constant gates
      .def("add_H", &NoisyCircuit::add_H, py::arg("target"), "Add Hadamard gate")
      .def("add_X", &NoisyCircuit::add_X, py::arg("target"), "Add Pauli-X gate")
      .def("add_Y", &NoisyCircuit::add_Y, py::arg("target"), "Add Pauli-Y gate")
      .def("add_Z", &NoisyCircuit::add_Z, py::arg("target"), "Add Pauli-Z gate")
      .def("add_S", &NoisyCircuit::add_S, py::arg("target"), "Add S gate")
      .def("add_Sdg", &NoisyCircuit::add_Sdg, py::arg("target"), "Add S† gate")
      .def("add_T", &NoisyCircuit::add_T, py::arg("target"), "Add T gate")
      .def("add_Tdg", &NoisyCircuit::add_Tdg, py::arg("target"), "Add T† gate")
      .def("add_SX", &NoisyCircuit::add_SX, py::arg("target"), "Add √X gate")

      // Single-qubit parametric gates
      .def("add_RX", &NoisyCircuit::add_RX, py::arg("target"),
           "Add RX rotation (1 param)")
      .def("add_RY", &NoisyCircuit::add_RY, py::arg("target"),
           "Add RY rotation (1 param)")
      .def("add_RZ", &NoisyCircuit::add_RZ, py::arg("target"),
           "Add RZ rotation (1 param)")
      .def("add_U1", &NoisyCircuit::add_U1, py::arg("target"),
           "Add U1 gate (1 param)")
      .def("add_U2", &NoisyCircuit::add_U2, py::arg("target"),
           "Add U2 gate (2 params)")
      .def("add_U3", &NoisyCircuit::add_U3, py::arg("target"),
           "Add U3 gate (3 params)")

      // Two-qubit constant gates
      .def("add_CNOT", &NoisyCircuit::add_CNOT, py::arg("target"),
           py::arg("control"), "Add CNOT gate")
      .def("add_CZ", &NoisyCircuit::add_CZ, py::arg("target"),
           py::arg("control"), "Add CZ gate")
      .def("add_CH", &NoisyCircuit::add_CH, py::arg("target"),
           py::arg("control"), "Add CH gate")

      // Two-qubit parametric gates
      .def("add_CRY", &NoisyCircuit::add_CRY, py::arg("target"),
           py::arg("control"), "Add CRY gate (1 param)")
      .def("add_CRZ", &NoisyCircuit::add_CRZ, py::arg("target"),
           py::arg("control"), "Add CRZ gate (1 param)")
      .def("add_CRX", &NoisyCircuit::add_CRX, py::arg("target"),
           py::arg("control"), "Add CRX gate (1 param)")
      .def("add_CP", &NoisyCircuit::add_CP, py::arg("target"),
           py::arg("control"), "Add CP gate (1 param)")

      // Noise channels
      .def(
          "add_depolarizing",
          [](NoisyCircuit &self, int qbit_num, py::object p) {
            if (p.is_none()) {
              self.add_depolarizing(qbit_num);
            } else {
              self.add_depolarizing(qbit_num, p.cast<double>());
            }
          },
          py::arg("qbit_num"), py::arg("error_rate") = py::none(),
          R"pbdoc(
            Add depolarizing noise: ρ → (1-p)ρ + p·I/2^n
            
            Args:
                qbit_num: Number of qubits the noise acts on
                error_rate: Fixed p, or None for parametric (1 param)
          )pbdoc")

      .def(
          "add_amplitude_damping",
          [](NoisyCircuit &self, int target, py::object gamma) {
            if (gamma.is_none()) {
              self.add_amplitude_damping(target);
            } else {
              self.add_amplitude_damping(target, gamma.cast<double>());
            }
          },
          py::arg("target"), py::arg("gamma") = py::none(),
          R"pbdoc(
            Add amplitude damping (T1 relaxation)
            
            Args:
                target: Target qubit index
                gamma: Fixed γ = 1-exp(-t/T1), or None for parametric (1 param)
          )pbdoc")

      .def(
          "add_phase_damping",
          [](NoisyCircuit &self, int target, py::object lambda_val) {
            if (lambda_val.is_none()) {
              self.add_phase_damping(target);
            } else {
              self.add_phase_damping(target, lambda_val.cast<double>());
            }
          },
          py::arg("target"), py::arg("lambda_param") = py::none(),
          R"pbdoc(
            Add phase damping (T2 dephasing)
            
            Args:
                target: Target qubit index
                lambda_param: Fixed λ = 1-exp(-t/T2), or None for parametric (1 param)
          )pbdoc")

      // Execution
      .def(
          "apply_to",
          [](NoisyCircuit &self, py::array_t<double> params,
             DensityMatrix &rho) {
            auto buf = params.request();
            self.apply_to(static_cast<double *>(buf.ptr), buf.size, rho);
          },
          py::arg("parameters"), py::arg("density_matrix"),
          "Apply circuit to density matrix")

      // Properties
      .def_property_readonly("qbit_num", &NoisyCircuit::get_qbit_num,
                             "Number of qubits")
      .def_property_readonly("parameter_num", &NoisyCircuit::get_parameter_num,
                             "Total number of parameters")
      .def("__len__", &NoisyCircuit::get_operation_count,
           "Number of operations")

      // Inspection
      .def("get_operation_info", &NoisyCircuit::get_operation_info,
           "Get list of all operations with their info")

      .def("__repr__", [](const NoisyCircuit &self) {
        return "<NoisyCircuit: " + std::to_string(self.get_qbit_num()) +
               " qubits, " + std::to_string(self.get_operation_count()) +
               " ops, " + std::to_string(self.get_parameter_num()) + " params>";
      });

  // ===============================================================
  // Legacy Noise Channels (for backward compatibility)
  // ===============================================================

  py::class_<NoiseChannel, std::shared_ptr<NoiseChannel>>(
      m, "NoiseChannel", "Base class for quantum noise channels (legacy)")
      .def("apply", &NoiseChannel::apply, py::arg("density_matrix"),
           "Apply noise channel to density matrix")
      .def("get_name", &NoiseChannel::get_name, "Get channel name");

  py::class_<DepolarizingChannel, NoiseChannel,
             std::shared_ptr<DepolarizingChannel>>(
      m, "DepolarizingChannel", "Depolarizing channel (legacy standalone)")
      .def(py::init<int, double>(), py::arg("qbit_num"), py::arg("error_rate"))
      .def_property_readonly("error_rate", &DepolarizingChannel::get_error_rate)
      .def_property_readonly("qbit_num", &DepolarizingChannel::get_qbit_num);

  py::class_<AmplitudeDampingChannel, NoiseChannel,
             std::shared_ptr<AmplitudeDampingChannel>>(
      m, "AmplitudeDampingChannel",
      "Amplitude damping channel (legacy standalone)")
      .def(py::init<int, double>(), py::arg("target_qbit"), py::arg("gamma"))
      .def_property_readonly("gamma", &AmplitudeDampingChannel::get_gamma)
      .def_property_readonly("target_qbit",
                             &AmplitudeDampingChannel::get_target_qbit);

  py::class_<PhaseDampingChannel, NoiseChannel,
             std::shared_ptr<PhaseDampingChannel>>(
      m, "PhaseDampingChannel", "Phase damping channel (legacy standalone)")
      .def(py::init<int, double>(), py::arg("target_qbit"), py::arg("lambda"))
      .def_property_readonly("lambda_param", &PhaseDampingChannel::get_lambda)
      .def_property_readonly("target_qbit",
                             &PhaseDampingChannel::get_target_qbit);

  // ===============================================================
  // Module metadata
  // ===============================================================

  m.attr("__version__") = "2.0.0";
}
