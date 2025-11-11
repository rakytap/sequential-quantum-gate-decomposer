## Phase 1.1 Plan: `squander/density_matrix/core.py` (⏭️ Future Enhancement - NOT in Phase 1)

**⚠️ IMPORTANT:** This section shows a **conceptual** Python wrapper layer that is **NOT implemented in Phase 1**. 

**Phase 1 uses direct C++ bindings** (see actual `__init__.py` above). The wrapper layer shown below could be added in future phases for enhanced Python ergonomics (property decorators, convenience functions, etc.).

**What Phase 1 actually provides:**
- Direct C++ bindings via pybind11
- Method syntax (e.g., `rho.purity()` with parentheses)
- No convenience functions like `bell_state_density_matrix()`

**What this conceptual wrapper could add in the future:**
- Property syntax (e.g., `rho.purity` without parentheses)
- Convenience functions for common states
- Additional Python-specific features

**Conceptual code (for future reference):**

```python
# ⚠️ NOT IMPLEMENTED IN PHASE 1 - CONCEPTUAL ONLY
"""High-level Python interface wrapping C++ density matrix classes."""

import numpy as np
from typing import Optional, Union, List
from ._density_matrix_cpp import (
    DensityMatrix as _DensityMatrix,
    NoisyCircuit as _NoisyCircuit,
)


class DensityMatrix:  # ⚠️ Future enhancement
    """High-level wrapper for density matrix with NumPy integration.
    
    Examples:
        >>> from squander.density_matrix import DensityMatrix
        >>> rho = DensityMatrix(qbit_num=2)
        >>> print(rho.purity)  # 1.0 (pure state)
    """
    
    def __init__(
        self,
        qbit_num: Optional[int] = None,
        state_vector: Optional[np.ndarray] = None,
        data: Optional[np.ndarray] = None,
    ):
        """Create density matrix.
        
        Args:
            qbit_num: Number of qubits (creates |0⟩⟨0|).
            state_vector: Create from state vector ρ = |ψ⟩⟨ψ|.
            data: Create from existing density matrix data.
        """
        if qbit_num is not None:
            self._cpp_dm = _DensityMatrix(qbit_num)
        elif state_vector is not None:
            self._cpp_dm = _DensityMatrix(state_vector)
        elif data is not None:
            self._cpp_dm = _DensityMatrix.from_numpy(data)
        else:
            raise ValueError("Must provide qbit_num, state_vector, or data")
    
    @property
    def qbit_num(self) -> int:
        return self._cpp_dm.qbit_num
    
    @property
    def dim(self) -> int:
        return self._cpp_dm.dim
    
    @property
    def purity(self) -> float:
        return self._cpp_dm.purity()
    
    @property
    def entropy(self) -> float:
        return self._cpp_dm.entropy()
    
    def to_numpy(self) -> np.ndarray:
        """Export to NumPy array."""
        return self._cpp_dm.to_numpy()
    
    def is_valid(self, tol: float = 1e-10) -> bool:
        """Check if valid density matrix."""
        return self._cpp_dm.is_valid(tol)
    
    def apply_unitary(self, U: np.ndarray):
        """Apply unitary: ρ → UρU†."""
        self._cpp_dm.apply_unitary(U)
    
    def __repr__(self) -> str:
        return f"DensityMatrix(qubits={self.qbit_num}, purity={self.purity:.4f})"


class NoisyCircuit:
    """High-level wrapper for density matrix circuit evolution."""
    
    def __init__(self, qbit_num: int):
        self._cpp_circuit = _NoisyCircuit(qbit_num)
        self._qbit_num = qbit_num
    
    # Gate methods delegate to C++
    def add_H(self, target: int):
        self._cpp_circuit.add_H(target)
        return self
    
    def add_X(self, target: int):
        self._cpp_circuit.add_X(target)
        return self
    
    def add_CNOT(self, target: int, control: int):
        self._cpp_circuit.add_CNOT(target, control)
        return self
    
    def add_RZ(self, target: int):
        self._cpp_circuit.add_RZ(target)
        return self
    
    # ... other gates ...
    
    def apply_to(
        self,
        rho: DensityMatrix,
        parameters: Optional[np.ndarray] = None,
    ) -> None:
        """Apply circuit to density matrix."""
        if parameters is None:
            parameters = np.array([])
        self._cpp_circuit.apply_to(parameters, rho._cpp_dm)
    
    @property
    def qbit_num(self) -> int:
        return self._qbit_num
    
    def __repr__(self) -> str:
        return f"NoisyCircuit(qubits={self._qbit_num})"


def create_density_matrix(
    qbit_num: int,
    state: str = "zero"
) -> DensityMatrix:
    """Create density matrix with common initial states."""
    # Implementation similar to before
    pass


def bell_state_density_matrix(which: int = 0) -> DensityMatrix:  # ⚠️ Future
    """Create Bell state density matrix."""
    # Implementation
    pass


def ghz_state_density_matrix(qbit_num: int) -> DensityMatrix:  # ⚠️ Future
    """Create GHZ state density matrix."""
    # Implementation
    pass
```

**End of conceptual code**
