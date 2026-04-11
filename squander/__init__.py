# python exported interfaces of the SQUANDER package

# decomposition classes of narrow circuits (up to 10 qubits)
from squander.decomposition.qgd_N_Qubit_Decompositions_Wrapper import (
    qgd_N_Qubit_Decomposition as N_Qubit_Decomposition,
    qgd_N_Qubit_Decomposition_adaptive as N_Qubit_Decomposition_adaptive,
    qgd_N_Qubit_Decomposition_custom as N_Qubit_Decomposition_custom,
    qgd_N_Qubit_Decomposition_Tree_Search as N_Qubit_Decomposition_Tree_Search,
    qgd_N_Qubit_Decomposition_Tabu_Search as N_Qubit_Decomposition_Tabu_Search,
)

# State preparation (depends on adaptive decomposition)
from squander.decomposition.qgd_N_Qubit_State_Preparation_adaptive import qgd_N_Qubit_State_Preparation_adaptive as N_Qubit_State_Preparation_adaptive

# optimization of wide circuits (optimize wide circuits)
from squander.decomposition.qgd_Wide_Circuit_Optimization import qgd_Wide_Circuit_Optimization as Wide_Circuit_Optimization

# variational quantum solver
from squander.VQA.qgd_Variational_Quantum_Eigensolver_Base import qgd_Variational_Quantum_Eigensolver_Base as Variational_Quantum_Eigensolver
from squander.VQA.qgd_Generative_Quantum_Machine_Learning_Base import qgd_Generative_Quantum_Machine_Learning_Base as Generative_Quantum_Machine_Learning

#gates
from squander.gates.gates_Wrapper import (
    Gate,
    U1,
    U2,
    U3,
    H,
    X,
    Y,
    Z,
    T,
    Tdg,
    R,
    CH,
    CNOT,
    CZ,
    RX,
    RY,
    RZ,
    SX,
    SYC,
    CRY,
    CR,
    CROT,
    CRX,
    CRZ,
    CCX,
    CP,
    SWAP,
    CSWAP,
    RXX,
    RYY,
    RZZ,
    SXdg
)


# quantum circuit
from squander.gates.qgd_Circuit import qgd_Circuit as Circuit

#SABRE
from squander.synthesis.qgd_SABRE import qgd_SABRE as SABRE

# Qiskit IO
from squander.IO_interfaces import Qiskit_IO

import squander.utils


## NN component (experimental interface)
from squander.nn.qgd_nn import qgd_nn as NN



