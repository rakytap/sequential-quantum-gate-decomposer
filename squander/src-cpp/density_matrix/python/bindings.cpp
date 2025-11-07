/*
Copyright 2025 SQUANDER Contributors

pybind11 Bindings for Density Matrix Module
*/

#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "Gate.h"
#include "density_circuit.h"
#include "density_matrix.h"
#include "matrix.h"
#include "matrix_real.h"
#include "noise_channel.h"

namespace py = pybind11;
using namespace squander::density;

// ===================================================================
// NumPy <-> DensityMatrix conversion helpers
// ===================================================================

/**
 * @brief Convert NumPy array to DensityMatrix
 */
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

/**
 * @brief Convert DensityMatrix to NumPy array
 */
py::array_t<std::complex<double>>
density_matrix_to_numpy(const DensityMatrix &rho) {
  int dim = rho.get_dim();

  // Create NumPy array
  py::array_t<std::complex<double>> result({dim, dim});
  auto buf = result.request();
  auto *ptr = static_cast<std::complex<double> *>(buf.ptr);

  // Copy data
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
        SQUANDER Density Matrix Module - C++ Backend
        
        Provides high-performance density matrix simulation for mixed quantum states.
        This is the C++ backend - use the Python wrapper in squander.density_matrix.
    )pbdoc";

  // ===============================================================
  // DensityMatrix class
  // ===============================================================

  py::class_<DensityMatrix>(m, "DensityMatrix", R"pbdoc(
        Quantum density matrix ρ for mixed-state representation.
        
        Inherits memory management from matrix_base, adds quantum-specific operations.
    )pbdoc")

      // Constructors
      .def(py::init<int>(), py::arg("qbit_num"),
           "Create density matrix for n qubits (initialized to |0⟩⟨0|)")

      .def(py::init([](py::array_t<std::complex<double>> state) {
             auto buf = state.request();

             if (buf.ndim != 1) {
               throw std::runtime_error("State vector must be 1D");
             }

             int dim = buf.shape[0];

             // Create Matrix wrapper for state vector
             Matrix state_vec(dim, 1);
             auto *src = static_cast<std::complex<double> *>(buf.ptr);
             for (int i = 0; i < dim; i++) {
               state_vec.get_data()[i].real = src[i].real();
               state_vec.get_data()[i].imag = src[i].imag();
             }

             return DensityMatrix(state_vec);
           }),
           py::arg("state_vector"), "Create from state vector: ρ = |ψ⟩⟨ψ|")

      // Properties
      .def_property_readonly("qbit_num", &DensityMatrix::get_qbit_num,
                             "Number of qubits")
      .def_property_readonly("dim", &DensityMatrix::get_dim,
                             "Matrix dimension (2^qbit_num)")

      // Methods
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
           "Check if valid density matrix (Hermitian, Tr=1, positive)")

      .def("eigenvalues", &DensityMatrix::eigenvalues,
           "Get eigenvalues (sorted descending)")

      .def(
          "apply_unitary",
          [](DensityMatrix &self, py::array_t<std::complex<double>> U) {
            auto buf = U.request();

            if (buf.ndim != 2) {
              throw std::runtime_error("Unitary must be 2D array");
            }
            if (buf.shape[0] != buf.shape[1]) {
              throw std::runtime_error("Unitary must be square");
            }

            int dim = buf.shape[0];

            // Create Matrix wrapper
            Matrix U_mat(dim, dim);
            auto *src = static_cast<std::complex<double> *>(buf.ptr);
            for (int i = 0; i < dim * dim; i++) {
              U_mat.get_data()[i].real = src[i].real();
              U_mat.get_data()[i].imag = src[i].imag();
            }

            self.apply_unitary(U_mat);
          },
          py::arg("U"), "Apply unitary: ρ → UρU†")

      .def("partial_trace", &DensityMatrix::partial_trace, py::arg("trace_out"),
           "Partial trace over specified qubits")

      .def("clone", &DensityMatrix::clone, "Create a deep copy")

      // Static methods
      .def_static("maximally_mixed", &DensityMatrix::maximally_mixed,
                  py::arg("qbit_num"),
                  "Create maximally mixed state: ρ = I/2^n")

      // NumPy conversion
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

      // Python special methods
      .def("__repr__",
           [](const DensityMatrix &self) {
             return "<DensityMatrix: " + std::to_string(self.get_qbit_num()) +
                    " qubits, purity=" + std::to_string(self.purity()) + ">";
           })

      .def("__str__", [](const DensityMatrix &self) {
        return "DensityMatrix(" + std::to_string(self.get_qbit_num()) +
               " qubits)";
      });

  // ===============================================================
  // DensityCircuit class
  // ===============================================================

  py::class_<DensityCircuit>(m, "DensityCircuit", R"pbdoc(
        Circuit for density matrix evolution.
        
        Provides same interface as qgd_Circuit but for density matrices.
    )pbdoc")

      .def(py::init<int>(), py::arg("qbit_num"),
           "Create empty circuit for n qubits")

      // Single-qubit gates
      .def("add_H", &DensityCircuit::add_H, py::arg("target"),
           "Add Hadamard gate")
      .def("add_X", &DensityCircuit::add_X, py::arg("target"),
           "Add Pauli-X gate")
      .def("add_Y", &DensityCircuit::add_Y, py::arg("target"),
           "Add Pauli-Y gate")
      .def("add_Z", &DensityCircuit::add_Z, py::arg("target"),
           "Add Pauli-Z gate")
      .def("add_S", &DensityCircuit::add_S, py::arg("target"), "Add S gate")
      .def("add_Sdg", &DensityCircuit::add_Sdg, py::arg("target"),
           "Add S† gate")
      .def("add_T", &DensityCircuit::add_T, py::arg("target"), "Add T gate")
      .def("add_Tdg", &DensityCircuit::add_Tdg, py::arg("target"),
           "Add T† gate")
      .def("add_SX", &DensityCircuit::add_SX, py::arg("target"), "Add √X gate")

      // Rotation gates
      .def("add_RX", &DensityCircuit::add_RX, py::arg("target"),
           "Add RX rotation gate")
      .def("add_RY", &DensityCircuit::add_RY, py::arg("target"),
           "Add RY rotation gate")
      .def("add_RZ", &DensityCircuit::add_RZ, py::arg("target"),
           "Add RZ rotation gate")
      .def("add_U1", &DensityCircuit::add_U1, py::arg("target"), "Add U1 gate")
      .def("add_U2", &DensityCircuit::add_U2, py::arg("target"), "Add U2 gate")
      .def("add_U3", &DensityCircuit::add_U3, py::arg("target"), "Add U3 gate")

      // Two-qubit gates
      .def("add_CNOT", &DensityCircuit::add_CNOT, py::arg("target"),
           py::arg("control"), "Add CNOT gate")
      .def("add_CZ", &DensityCircuit::add_CZ, py::arg("target"),
           py::arg("control"), "Add CZ gate")
      .def("add_CH", &DensityCircuit::add_CH, py::arg("target"),
           py::arg("control"), "Add CH gate")
      .def("add_CRY", &DensityCircuit::add_CRY, py::arg("target"),
           py::arg("control"), "Add CRY gate")
      .def("add_CRZ", &DensityCircuit::add_CRZ, py::arg("target"),
           py::arg("control"), "Add CRZ gate")
      .def("add_CRX", &DensityCircuit::add_CRX, py::arg("target"),
           py::arg("control"), "Add CRX gate")
      .def("add_CP", &DensityCircuit::add_CP, py::arg("target"),
           py::arg("control"), "Add CP (controlled-phase) gate")

      // Circuit application
      .def(
          "apply_to",
          [](DensityCircuit &self, py::array_t<double> params,
             DensityMatrix &rho) {
            auto buf = params.request();

            // Create Matrix_real wrapper
            Matrix_real params_mat(static_cast<double *>(buf.ptr), buf.shape[0],
                                   1);

            self.apply_to(params_mat, rho);
          },
          py::arg("parameters"), py::arg("density_matrix"),
          "Apply circuit to density matrix")

      // Properties
      .def_property_readonly("qbit_num", &DensityCircuit::get_qbit_num,
                             "Number of qubits")
      .def_property_readonly("parameter_num",
                             &DensityCircuit::get_parameter_num,
                             "Number of parameters")

      // Representation
      .def("__repr__", [](const DensityCircuit &self) {
        return "<DensityCircuit: " + std::to_string(self.get_qbit_num()) +
               " qubits>";
      });

  // ===============================================================
  // Noise Channels
  // ===============================================================

  py::class_<NoiseChannel, std::shared_ptr<NoiseChannel>>(
      m, "NoiseChannel", "Base class for quantum noise channels")
      .def("apply", &NoiseChannel::apply, py::arg("density_matrix"),
           "Apply noise channel to density matrix")
      .def("get_name", &NoiseChannel::get_name, "Get channel name");

  py::class_<DepolarizingChannel, NoiseChannel,
             std::shared_ptr<DepolarizingChannel>>(m, "DepolarizingChannel",
                                                   R"pbdoc(
            Depolarizing channel: ρ → (1-p)ρ + p·I/2^n
            
            Represents uniform noise that mixes state with maximally mixed state.
        )pbdoc")

      .def(py::init<int, double>(), py::arg("qbit_num"), py::arg("error_rate"),
           "Constructor with qubit number and error rate")

      .def_property_readonly("error_rate", &DepolarizingChannel::get_error_rate,
                             "Error rate p ∈ [0,1]")
      .def_property_readonly("qbit_num", &DepolarizingChannel::get_qbit_num,
                             "Number of qubits")

      .def("__repr__", [](const DepolarizingChannel &self) {
        return "<DepolarizingChannel: " + std::to_string(self.get_qbit_num()) +
               " qubits, p=" + std::to_string(self.get_error_rate()) + ">";
      });

  py::class_<AmplitudeDampingChannel, NoiseChannel,
             std::shared_ptr<AmplitudeDampingChannel>>(
      m, "AmplitudeDampingChannel",
      R"pbdoc(
            Amplitude damping (T1 relaxation): |1⟩ → |0⟩ decay
            
            Models energy relaxation. Parameter: γ = 1 - exp(-t/T1)
        )pbdoc")

      .def(py::init<int, double>(), py::arg("target_qbit"), py::arg("gamma"),
           "Constructor with target qubit and damping parameter")

      .def_property_readonly("gamma", &AmplitudeDampingChannel::get_gamma,
                             "Damping parameter γ = 1 - exp(-t/T1)")
      .def_property_readonly("target_qbit",
                             &AmplitudeDampingChannel::get_target_qbit,
                             "Target qubit index")

      .def("__repr__", [](const AmplitudeDampingChannel &self) {
        return "<AmplitudeDampingChannel: qubit=" +
               std::to_string(self.get_target_qbit()) +
               ", γ=" + std::to_string(self.get_gamma()) + ">";
      });

  py::class_<PhaseDampingChannel, NoiseChannel,
             std::shared_ptr<PhaseDampingChannel>>(m, "PhaseDampingChannel",
                                                   R"pbdoc(
            Phase damping (T2 dephasing): loss of coherence
            
            Models phase randomization. Parameter: λ = 1 - exp(-t/T2)
        )pbdoc")

      .def(py::init<int, double>(), py::arg("target_qbit"), py::arg("lambda"),
           "Constructor with target qubit and dephasing parameter")

      .def_property_readonly("lambda_param", &PhaseDampingChannel::get_lambda,
                             "Dephasing parameter λ = 1 - exp(-t/T2)")
      .def_property_readonly("target_qbit",
                             &PhaseDampingChannel::get_target_qbit,
                             "Target qubit index")

      .def("__repr__", [](const PhaseDampingChannel &self) {
        return "<PhaseDampingChannel: qubit=" +
               std::to_string(self.get_target_qbit()) +
               ", λ=" + std::to_string(self.get_lambda()) + ">";
      });

  // ===============================================================
  // Module metadata
  // ===============================================================

  m.attr("__version__") = "1.0.0";
}
