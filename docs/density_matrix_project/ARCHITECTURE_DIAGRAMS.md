# NoisyCircuit Architecture - Mermaid Diagrams

## 1. Class Hierarchy

```mermaid
classDiagram
    direction TB
    
    %% Core Interface
    class IDensityOperation {
        <<interface>>
        +apply_to_density(params, param_count, rho)*
        +get_parameter_num()* int
        +get_name()* string
        +is_unitary()* bool
        +clone()* unique_ptr~IDensityOperation~
    }
    
    %% Gate Operations (Adapter Pattern)
    class GateOperation {
        -Gate* gate_
        -bool owns_
        +GateOperation(Gate*, bool owns)
        +apply_to_density(params, param_count, rho)
        +get_parameter_num() int
        +get_name() string
        +is_unitary() bool  // always true
        +clone() unique_ptr~IDensityOperation~
        +get_gate() Gate*
    }
    
    %% Noise Base Class
    class NoiseOperation {
        <<abstract>>
        +is_unitary() bool  // always false
    }
    
    %% Concrete Noise Channels
    class DepolarizingOp {
        -int qbit_num_
        -double fixed_error_rate_
        -bool is_parametric_
        +DepolarizingOp(qbit_num)
        +DepolarizingOp(qbit_num, error_rate)
        +apply_to_density(params, param_count, rho)
        +get_parameter_num() int
        +get_name() string
        +clone() unique_ptr~IDensityOperation~
    }
    
    class AmplitudeDampingOp {
        -int target_qbit_
        -double fixed_gamma_
        -bool is_parametric_
        +AmplitudeDampingOp(target)
        +AmplitudeDampingOp(target, gamma)
        +apply_to_density(params, param_count, rho)
        +get_parameter_num() int
        +get_name() string
        +clone() unique_ptr~IDensityOperation~
    }
    
    class PhaseDampingOp {
        -int target_qbit_
        -double fixed_lambda_
        -bool is_parametric_
        +PhaseDampingOp(target)
        +PhaseDampingOp(target, lambda)
        +apply_to_density(params, param_count, rho)
        +get_parameter_num() int
        +get_name() string
        +clone() unique_ptr~IDensityOperation~
    }
    
    %% Existing Gate Classes (from SQUANDER)
    class Gate {
        <<existing>>
        #int qbit_num
        #int target_qbit
        #int control_qbit
        +get_matrix(params) Matrix
        +calc_one_qubit_u3() Matrix
        +get_parameter_num() int
        +get_name() string
        +clone() Gate*
    }
    
    %% Inheritance relationships
    IDensityOperation <|.. GateOperation : implements
    IDensityOperation <|.. NoiseOperation : implements
    NoiseOperation <|-- DepolarizingOp : extends
    NoiseOperation <|-- AmplitudeDampingOp : extends
    NoiseOperation <|-- PhaseDampingOp : extends
    
    %% Composition
    GateOperation o-- Gate : wraps
```

## 2. NoisyCircuit Structure

```mermaid
classDiagram
    direction LR
    
    class NoisyCircuit {
        -int qbit_num_
        -vector~unique_ptr~IDensityOperation~~ operations_
        -vector~int~ param_starts_
        -int total_params_
        
        +NoisyCircuit(qbit_num)
        
        %% Gate Addition
        +add_H(target)
        +add_X(target)
        +add_Y(target)
        +add_Z(target)
        +add_S(target)
        +add_T(target)
        +add_RX(target)
        +add_RY(target)
        +add_RZ(target)
        +add_U3(target)
        +add_CNOT(target, control)
        +add_CZ(target, control)
        
        %% Noise Addition
        +add_depolarizing(qbit_num)
        +add_depolarizing(qbit_num, error_rate)
        +add_amplitude_damping(target)
        +add_amplitude_damping(target, gamma)
        +add_phase_damping(target)
        +add_phase_damping(target, lambda)
        
        %% Execution
        +apply_to(params, param_count, rho)
        
        %% Properties
        +get_qbit_num() int
        +get_parameter_num() int
        +get_operation_count() size_t
        +get_operation_info() vector~OperationInfo~
    }
    
    class DensityMatrix {
        -int qbit_num_
        -QGD_Complex16* data
        
        +DensityMatrix(qbit_num)
        +apply_single_qubit_unitary(u_2x2, target)
        +apply_two_qubit_unitary(u_4x4, target, control)
        +apply_local_unitary(kernel, targets)
        +trace() Complex
        +purity() double
        +entropy() double
        +is_valid() bool
    }
    
    class OperationInfo {
        <<struct>>
        +string name
        +bool is_unitary
        +int param_count
        +int param_start
    }
    
    NoisyCircuit "1" *-- "0..*" IDensityOperation : contains
    NoisyCircuit ..> DensityMatrix : modifies
    NoisyCircuit ..> OperationInfo : returns
```

## 3. Circuit Execution Flow

```mermaid
flowchart TB
    subgraph Input
        params["double* params"]
        rho["DensityMatrix& rho"]
    end
    
    subgraph NoisyCircuit["NoisyCircuit::apply_to()"]
        validate["Validate qubit count<br/>& parameter count"]
        loop["For each operation i"]
        
        subgraph OpExec["Operation Execution"]
            getStart["start = param_starts_[i]"]
            getCount["count = op->get_parameter_num()"]
            slice["op_params = params + start"]
            apply["op->apply_to_density(op_params, count, rho)"]
        end
    end
    
    subgraph Operations["IDensityOperation Polymorphism"]
        direction LR
        gate["GateOperation"]
        depol["DepolarizingOp"]
        amp["AmplitudeDampingOp"]
        phase["PhaseDampingOp"]
    end
    
    subgraph GateExec["GateOperation::apply_to_density()"]
        getMatrix["Get unitary matrix<br/>from Gate"]
        check1{{"1-qubit?"}}
        check2{{"2-qubit?"}}
        apply1["rho.apply_single_qubit_unitary()"]
        apply2["rho.apply_two_qubit_unitary()"]
        applyN["rho.apply_local_unitary()"]
    end
    
    subgraph NoiseExec["NoiseOperation::apply_to_density()"]
        kraus["Apply Kraus operators<br/>directly to density matrix"]
    end
    
    params --> validate
    rho --> validate
    validate --> loop
    loop --> getStart
    getStart --> getCount
    getCount --> slice
    slice --> apply
    
    apply -.-> gate
    apply -.-> depol
    apply -.-> amp
    apply -.-> phase
    
    gate --> getMatrix
    getMatrix --> check1
    check1 -->|Yes| apply1
    check1 -->|No| check2
    check2 -->|Yes| apply2
    check2 -->|No| applyN
    
    depol --> kraus
    amp --> kraus
    phase --> kraus
    
    apply1 --> loop
    apply2 --> loop
    applyN --> loop
    kraus --> loop
```

## 4. Parameter Flow

```mermaid
flowchart LR
    subgraph UserInput["User Input"]
        params["params = [0.5, 0.3, 0.1, 0.02, 0.01]"]
    end
    
    subgraph Circuit["NoisyCircuit"]
        direction TB
        op0["H(0)<br/>params: 0"]
        op1["RZ(1)<br/>params: 1<br/>start: 0"]
        op2["CNOT(1,0)<br/>params: 0"]
        op3["U3(0)<br/>params: 3<br/>start: 1"]
        op4["Depolarizing<br/>params: 1<br/>start: 4"]
        
        op0 --> op1 --> op2 --> op3 --> op4
    end
    
    subgraph Slicing["Parameter Slicing"]
        s0["H: (none)"]
        s1["RZ: params[0] = 0.5"]
        s2["CNOT: (none)"]
        s3["U3: params[1:4] = [0.3, 0.1, 0.02]"]
        s4["Depol: params[4] = 0.01"]
    end
    
    params --> Circuit
    op0 -.-> s0
    op1 -.-> s1
    op2 -.-> s2
    op3 -.-> s3
    op4 -.-> s4
    
    
```

## 5. Density Matrix Local Unitary Application

```mermaid
flowchart TB
    subgraph Input
        U["2×2 Unitary U"]
        rho["4×4 Density Matrix ρ<br/>(2 qubits)"]
        target["target_qubit = 0"]
    end
    
    subgraph Naive["Naive Approach O(8^N)"]
        direction TB
        expand["Expand U to 4×4:<br/>U_full = U ⊗ I"]
        mult1["ρ' = U_full × ρ"]
        mult2["ρ'' = ρ' × U_full†"]
    end
    
    subgraph Optimized["Optimized Local Kernel O(4^N)"]
        direction TB
        iterate["For each (i,j) in ρ"]
        extract["Extract target bits<br/>from i and j"]
        sum["Sum over kernel indices"]
        update["ρ'[i,j] = Σ U[i_t,a] × ρ[i',j'] × U*[j_t,b]"]
    end
    
    U --> expand
    rho --> expand
    expand --> mult1
    mult1 --> mult2
    
    U --> iterate
    rho --> iterate
    target --> iterate
    iterate --> extract
    extract --> sum
    sum --> update
    
    style Naive fill:#FFB6C1
    style Optimized fill:#90EE90
```

## 6. Noise Channel Mathematics

```mermaid
flowchart TB
    subgraph Depolarizing["Depolarizing Channel"]
        depol_in["ρ"]
        depol_formula["ρ' = (1-p)ρ + p·Tr(ρ)·I/d"]
        depol_out["ρ'"]
        depol_in --> depol_formula --> depol_out
    end
    
    subgraph AmplitudeDamping["Amplitude Damping (T1)"]
        amp_in["ρ"]
        amp_k0["K₀ = [[1,0],[0,√(1-γ)]]"]
        amp_k1["K₁ = [[0,√γ],[0,0]]"]
        amp_formula["ρ' = K₀ρK₀† + K₁ρK₁†"]
        amp_out["ρ'"]
        amp_in --> amp_formula
        amp_k0 --> amp_formula
        amp_k1 --> amp_formula
        amp_formula --> amp_out
    end
    
    subgraph PhaseDamping["Phase Damping (T2)"]
        phase_in["ρ"]
        phase_k0["K₀ = [[1,0],[0,√(1-λ)]]"]
        phase_k1["K₁ = [[0,0],[0,√λ]]"]
        phase_formula["ρ' = K₀ρK₀† + K₁ρK₁†"]
        phase_out["ρ'"]
        phase_in --> phase_formula
        phase_k0 --> phase_formula
        phase_k1 --> phase_formula
        phase_formula --> phase_out
    end
```

## 7. Python Binding Architecture

```mermaid
flowchart TB
    subgraph Python["Python Layer"]
        py_circuit["NoisyCircuit(n)"]
        py_add["circuit.add_H(0)<br/>circuit.add_CNOT(1,0)<br/>circuit.add_depolarizing(n, 0.01)"]
        py_apply["circuit.apply_to(params, rho)"]
        py_rho["DensityMatrix(n)"]
    end
    
    subgraph Pybind["pybind11 Bindings"]
        bind_circuit["py::class_<NoisyCircuit>"]
        bind_rho["py::class_<DensityMatrix>"]
        bind_lambda["Lambda wrappers for<br/>numpy array handling"]
    end
    
    subgraph CPP["C++ Layer"]
        cpp_circuit["squander::density::NoisyCircuit"]
        cpp_ops["IDensityOperation*<br/>(polymorphic)"]
        cpp_rho["squander::density::DensityMatrix"]
    end
    
    py_circuit --> bind_circuit
    py_add --> bind_circuit
    py_apply --> bind_lambda
    py_rho --> bind_rho
    
    bind_circuit --> cpp_circuit
    bind_lambda --> cpp_circuit
    bind_rho --> cpp_rho
    
    cpp_circuit --> cpp_ops
    cpp_ops --> cpp_rho
```

## 8. Design Patterns Used

```mermaid
mindmap
    root((NoisyCircuit<br/>Architecture))
        Interface Segregation
            IDensityOperation
                Unified interface
                Gates & Noise
        Adapter Pattern
            GateOperation
                Wraps existing Gate
                No Gate modification
        Strategy Pattern
            Different apply_to_density
                GateOperation: unitary
                NoiseOperation: Kraus
        Composite Pattern
            NoisyCircuit
                Contains operations
                Sequential execution
        Factory Methods
            add_H, add_X, etc.
            add_depolarizing, etc.
```

## 9. Memory Ownership

```mermaid
flowchart TB
    subgraph NoisyCircuit
        ops["vector<unique_ptr<IDensityOperation>>"]
    end
    
    subgraph GateOperation
        gate_ptr["Gate* gate_"]
        owns["bool owns_ = true"]
    end
    
    subgraph Gate["Gate (heap)"]
        gate_data["Gate data"]
    end
    
    subgraph NoiseOps["NoiseOperation instances"]
        depol["DepolarizingOp"]
        amp["AmplitudeDampingOp"]
        phase["PhaseDampingOp"]
    end
    
    ops -->|"unique_ptr owns"| GateOperation
    ops -->|"unique_ptr owns"| depol
    ops -->|"unique_ptr owns"| amp
    ops -->|"unique_ptr owns"| phase
    
    GateOperation -->|"owns_ ? delete : no-op"| Gate
    
    style ops fill:#90EE90
    style GateOperation fill:#87CEEB
    style Gate fill:#FFE4B5
```

---

## Quick Reference

| Component | Purpose | Key Method |
|-----------|---------|------------|
| `IDensityOperation` | Interface for all operations | `apply_to_density()` |
| `GateOperation` | Adapts Gate → IDensityOperation | Calls `apply_local_unitary()` |
| `NoiseOperation` | Base for noise channels | `is_unitary() = false` |
| `DepolarizingOp` | Global depolarizing noise | Direct density modification |
| `AmplitudeDampingOp` | T1 relaxation | Kraus operators |
| `PhaseDampingOp` | T2 dephasing | Kraus operators |
| `NoisyCircuit` | Orchestrates execution | `apply_to()` |
| `DensityMatrix` | Quantum state | `apply_single/two_qubit_unitary()` |

