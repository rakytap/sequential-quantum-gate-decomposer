'''
Copyright 2020 Peter Rakyta, Ph.D.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

You should have received a copy of the GNU General Public License
along with this program.  If not, see http://www.gnu.org/licenses/.
'''

import numpy as np
import pytest
from itertools import permutations

from squander.gates.gates_Wrapper import Permutation
from squander.gates.qgd_Circuit import qgd_Circuit


class Test_Permutation:
    """Test class for Permutation gate"""

    def test_permutation_creation_identity(self):
        """
        Test creating identity permutation gates
        """
        for qbit_num in range(1, 6):
            # Identity permutation: [0, 1, 2, ..., n-1]
            pattern = list(range(qbit_num))
            perm_gate = Permutation(qbit_num, pattern)
            
            assert perm_gate.get_Parameter_Num() == 0
            pattern_retrieved = perm_gate.get_Pattern()
            assert pattern_retrieved == pattern

    def test_permutation_creation_swap(self):
        """
        Test creating swap permutation gates
        """
        for qbit_num in range(2, 6):
            # Swap first and last qubits: [n-1, 1, 2, ..., n-2, 0]
            pattern = list(range(qbit_num))
            pattern[0], pattern[-1] = pattern[-1], pattern[0]
            perm_gate = Permutation(qbit_num, pattern)
            
            pattern_retrieved = perm_gate.get_Pattern()
            assert pattern_retrieved == pattern

    def test_permutation_creation_reverse(self):
        """
        Test creating reverse permutation gates
        """
        for qbit_num in range(1, 6):
            # Reverse permutation: [n-1, n-2, ..., 1, 0]
            pattern = list(range(qbit_num))[::-1]
            perm_gate = Permutation(qbit_num, pattern)
            
            pattern_retrieved = perm_gate.get_Pattern()
            assert pattern_retrieved == pattern

    def test_permutation_creation_random(self):
        """
        Test creating random permutation gates
        """
        np.random.seed(42)
        for qbit_num in range(2, 6):
            # Random permutation
            pattern = list(range(qbit_num))
            np.random.shuffle(pattern)
            perm_gate = Permutation(qbit_num, pattern)
            
            pattern_retrieved = perm_gate.get_Pattern()
            assert pattern_retrieved == pattern

    def test_permutation_creation_invalid_size(self):
        """
        Test that creating permutation with wrong pattern size raises error
        """
        qbit_num = 3
        # Pattern too small
        with pytest.raises(ValueError, match="Pattern size.*does not match"):
            Permutation(qbit_num, [0, 1])
        
        # Pattern too large
        with pytest.raises(ValueError, match="Pattern size.*does not match"):
            Permutation(qbit_num, [0, 1, 2, 3])

    def test_permutation_creation_invalid_range(self):
        """
        Test that creating permutation with out-of-range indices raises error
        """
        qbit_num = 3
        # Negative index
        with pytest.raises(ValueError, match="out of range"):
            Permutation(qbit_num, [-1, 1, 2])
        
        # Index too large
        with pytest.raises(ValueError, match="out of range"):
            Permutation(qbit_num, [0, 1, 3])

    def test_permutation_creation_duplicates(self):
        """
        Test that creating permutation with duplicate values raises error
        """
        qbit_num = 3
        # Duplicate values
        with pytest.raises(ValueError, match="duplicate"):
            Permutation(qbit_num, [0, 1, 1])
        
        with pytest.raises(ValueError, match="duplicate"):
            Permutation(qbit_num, [0, 0, 2])

    def test_permutation_creation_invalid_type(self):
        """
        Test that creating permutation with invalid type raises error
        """
        qbit_num = 3
        # Tuple should work (converted to list)
        perm_gate = Permutation(qbit_num, (0, 1, 2))
        assert perm_gate.get_Pattern() == [0, 1, 2]
        
        # Non-integer values
        with pytest.raises(TypeError, match="pattern must contain integers"):
            Permutation(qbit_num, [0.0, 1.0, 2.0])
        
        with pytest.raises(TypeError, match="pattern must contain integers"):
            Permutation(qbit_num, ["0", "1", "2"])
        
        # Invalid type (not list or tuple)
        with pytest.raises(TypeError, match="pattern must be a list or tuple"):
            Permutation(qbit_num, "012")

    def test_permutation_get_pattern(self):
        """
        Test getting pattern from permutation gate
        """
        for qbit_num in range(1, 5):
            for pattern_tuple in permutations(range(qbit_num)):
                pattern = list(pattern_tuple)
                perm_gate = Permutation(qbit_num, pattern)
                retrieved_pattern = perm_gate.get_Pattern()
                assert retrieved_pattern == pattern

    def test_permutation_tuple_conversion(self):
        """
        Test that tuples are properly converted to lists
        """
        for qbit_num in range(1, 5):
            for pattern_tuple in permutations(range(qbit_num)):
                # Create with tuple
                perm_gate = Permutation(qbit_num, pattern_tuple)
                retrieved_pattern = perm_gate.get_Pattern()
                # Should return as list
                assert retrieved_pattern == list(pattern_tuple)
                assert isinstance(retrieved_pattern, list)
                
                # Set with tuple
                perm_gate.set_Pattern(pattern_tuple)
                retrieved_pattern = perm_gate.get_Pattern()
                assert retrieved_pattern == list(pattern_tuple)
                assert isinstance(retrieved_pattern, list)

    def test_permutation_set_pattern(self):
        """
        Test setting pattern on permutation gate
        """
        qbit_num = 4
        initial_pattern = [0, 1, 2, 3]
        perm_gate = Permutation(qbit_num, initial_pattern)
        
        # Set new pattern
        new_pattern = [3, 2, 1, 0]
        perm_gate.set_Pattern(new_pattern)
        assert perm_gate.get_Pattern() == new_pattern
        
        # Set another pattern
        another_pattern = [1, 0, 3, 2]
        perm_gate.set_Pattern(another_pattern)
        assert perm_gate.get_Pattern() == another_pattern

    def test_permutation_set_pattern_invalid(self):
        """
        Test that setting invalid pattern raises error
        """
        qbit_num = 3
        perm_gate = Permutation(qbit_num, [0, 1, 2])
        
        # Wrong size
        with pytest.raises(ValueError, match="Pattern size.*does not match"):
            perm_gate.set_Pattern([0, 1])
        
        # Out of range
        with pytest.raises(ValueError, match="out of range"):
            perm_gate.set_Pattern([0, 1, 3])
        
        # Duplicates
        with pytest.raises(ValueError, match="duplicate"):
            perm_gate.set_Pattern([0, 1, 1])
        
        # Invalid type (not list or tuple)
        with pytest.raises(TypeError, match="Pattern must be a list or tuple"):
            perm_gate.set_Pattern("012")
        
        # Tuple should work (converted to list)
        perm_gate.set_Pattern((0, 1, 2))
        assert perm_gate.get_Pattern() == [0, 1, 2]
        
        # Tuple with different pattern
        perm_gate.set_Pattern((2, 0, 1))
        assert perm_gate.get_Pattern() == [2, 0, 1]

    def test_permutation_get_matrix_identity(self):
        """
        Test that identity permutation gives identity matrix
        """
        for qbit_num in range(1, 5):
            pattern = list(range(qbit_num))
            perm_gate = Permutation(qbit_num, pattern)
            matrix = perm_gate.get_Matrix()
            
            expected = np.eye(2**qbit_num, dtype=np.complex128)
            error = np.linalg.norm(matrix - expected)
            assert error < 1e-10, f"Identity permutation failed for {qbit_num} qubits"

    def test_permutation_get_matrix_swap(self):
        """
        Test permutation matrix for swap operation
        """
        qbit_num = 2
        # Swap qubits: [1, 0]
        pattern = [1, 0]
        perm_gate = Permutation(qbit_num, pattern)
        matrix = perm_gate.get_Matrix()
        
        # For 2 qubits, swap should exchange |01> and |10>
        # Identity: |00> -> |00>, |01> -> |01>, |10> -> |10>, |11> -> |11>
        # Swap:     |00> -> |00>, |01> -> |10>, |10> -> |01>, |11> -> |11>
        expected = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ], dtype=np.complex128)
        
        error = np.linalg.norm(matrix - expected)
        assert error < 1e-10, "Swap permutation matrix incorrect"

    def test_permutation_get_matrix_unitary(self):
        """
        Test that permutation matrices are unitary
        """
        for qbit_num in range(1, 5):
            pattern = list(range(qbit_num))
            np.random.shuffle(pattern)
            perm_gate = Permutation(qbit_num, pattern)
            matrix = perm_gate.get_Matrix()
            
            # Check unitarity: U @ U^dagger = I
            unitary_check = matrix @ matrix.conj().T
            identity = np.eye(2**qbit_num, dtype=np.complex128)
            error = np.linalg.norm(unitary_check - identity)
            assert error < 1e-10, f"Matrix not unitary for pattern {pattern}"

    def test_permutation_apply_to_identity(self):
        """
        Test applying identity permutation to a state
        """
        for qbit_num in range(1, 5):
            pattern = list(range(qbit_num))
            perm_gate = Permutation(qbit_num, pattern)
            
            # Create random state
            matrix_size = 2**qbit_num
            state = np.random.rand(matrix_size) + 1j * np.random.rand(matrix_size)
            state = state / np.linalg.norm(state)
            
            state_copy = state.copy()
            perm_gate.apply_to(state_copy)
            
            # Identity should not change the state
            error = np.linalg.norm(state_copy - state)
            assert error < 1e-10, "Identity permutation changed state"

    def test_permutation_apply_to_swap(self):
        """
        Test applying swap permutation to a state
        """
        qbit_num = 2
        pattern = [1, 0]  # Swap qubits
        perm_gate = Permutation(qbit_num, pattern)
        
        # Create test state |01> = [0, 1, 0, 0]
        state = np.array([0, 1, 0, 0], dtype=np.complex128)
        perm_gate.apply_to(state)
        
        # After swap, should be |10> = [0, 0, 1, 0]
        expected = np.array([0, 0, 1, 0], dtype=np.complex128)
        error = np.linalg.norm(state - expected)
        assert error < 1e-10, "Swap permutation incorrect"

    def test_permutation_apply_to_matrix(self):
        """
        Test applying permutation to a matrix
        """
        qbit_num = 3
        pattern = [2, 0, 1]  # Rotate: 0->2, 1->0, 2->1
        perm_gate = Permutation(qbit_num, pattern)
        
        # Create test matrix
        matrix_size = 2**qbit_num
        test_matrix = np.random.rand(matrix_size, matrix_size) + 1j * np.random.rand(matrix_size, matrix_size)
        test_matrix = test_matrix / np.linalg.norm(test_matrix)
        
        # Apply permutation
        test_matrix_copy = test_matrix.copy()
        perm_gate.apply_to(test_matrix_copy)
        
        # Check that it's different (unless it's identity)
        if pattern != list(range(qbit_num)):
            assert not np.allclose(test_matrix_copy, test_matrix), "Permutation should change matrix"

    def test_permutation_composition(self):
        """
        Test that applying two permutations is equivalent to their composition
        """
        qbit_num = 3
        pattern1 = [1, 2, 0]  # Rotate left
        pattern2 = [2, 0, 1]  # Rotate right
        
        perm1 = Permutation(qbit_num, pattern1)
        perm2 = Permutation(qbit_num, pattern2)
        
        # Compose patterns: pattern2(pattern1(x))
        composed_pattern = [pattern2[pattern1[i]] for i in range(qbit_num)]
        perm_composed = Permutation(qbit_num, composed_pattern)
        
        # Create test state
        matrix_size = 2**qbit_num
        state = np.random.rand(matrix_size) + 1j * np.random.rand(matrix_size)
        state = state / np.linalg.norm(state)
        
        # Apply sequentially
        state_seq = state.copy()
        perm1.apply_to(state_seq)
        perm2.apply_to(state_seq)
        
        # Apply composed
        state_comp = state.copy()
        perm_composed.apply_to(state_comp)
        
        error = np.linalg.norm(state_seq - state_comp)
        assert error < 1e-10, "Composition of permutations incorrect"

    def test_permutation_inverse(self):
        """
        Test that applying permutation and its inverse gives identity
        """
        for qbit_num in range(2, 5):
            pattern = list(range(qbit_num))
            np.random.shuffle(pattern)
            
            # Compute inverse permutation
            inverse_pattern = [0] * qbit_num
            for i in range(qbit_num):
                inverse_pattern[pattern[i]] = i
            
            perm = Permutation(qbit_num, pattern)
            perm_inv = Permutation(qbit_num, inverse_pattern)
            
            # Create test state
            matrix_size = 2**qbit_num
            state = np.random.rand(matrix_size) + 1j * np.random.rand(matrix_size)
            state = state / np.linalg.norm(state)
            
            # Apply permutation then inverse
            state_transformed = state.copy()
            perm.apply_to(state_transformed)
            perm_inv.apply_to(state_transformed)
            
            error = np.linalg.norm(state_transformed - state)
            assert error < 1e-10, f"Inverse permutation failed for pattern {pattern}"

    def test_permutation_circuit_integration(self):
        """
        Test adding permutation gate to circuit
        """
        qbit_num = 3
        pattern = [2, 0, 1]
        
        circuit = qgd_Circuit(qbit_num)
        circuit.add_Permutation(pattern)
        
        gates = circuit.get_Gates()
        assert len(gates) == 1
        
        gate = gates[0]
        assert gate.get_Name() == "Permutation"
        retrieved_pattern = gate.get_Pattern()
        assert retrieved_pattern == pattern

    def test_permutation_circuit_multiple(self):
        """
        Test adding multiple permutation gates to circuit
        """
        qbit_num = 3
        
        circuit = qgd_Circuit(qbit_num)
        pattern1 = [1, 2, 0]
        pattern2 = [2, 0, 1]
        
        circuit.add_Permutation(pattern1)
        circuit.add_Permutation(pattern2)
        
        gates = circuit.get_Gates()
        assert len(gates) == 2
        
        assert gates[0].get_Pattern() == pattern1
        assert gates[1].get_Pattern() == pattern2

    def test_permutation_get_involved_qubits(self):
        """
        Test getting involved qubits from permutation gate
        """
        for qbit_num in range(1, 5):
            pattern = list(range(qbit_num))
            perm_gate = Permutation(qbit_num, pattern)
            
            involved_qbits = perm_gate.get_Involved_Qbits()
            # Permutation gate involves all qubits
            assert involved_qbits == list(range(qbit_num))

    def test_permutation_get_target_qubits(self):
        """
        Test getting target qubits from permutation gate
        """
        for qbit_num in range(1, 5):
            pattern = list(range(qbit_num))
            perm_gate = Permutation(qbit_num, pattern)
            
            target_qbits = perm_gate.get_Target_Qbits()
            # Permutation gate targets all qubits
            assert target_qbits == list(range(qbit_num))

    def test_permutation_get_control_qubits(self):
        """
        Test getting control qubits from permutation gate (should be empty)
        """
        for qbit_num in range(1, 5):
            pattern = list(range(qbit_num))
            perm_gate = Permutation(qbit_num, pattern)
            
            control_qbits = perm_gate.get_Control_Qbits()
            # Permutation gate has no control qubits
            assert control_qbits == []

    def test_permutation_large_patterns(self):
        """
        Test permutation gates with larger numbers of qubits
        """
        for qbit_num in [5, 6, 7]:
            # Test identity
            pattern = list(range(qbit_num))
            perm_gate = Permutation(qbit_num, pattern)
            matrix = perm_gate.get_Matrix()
            
            expected = np.eye(2**qbit_num, dtype=np.complex128)
            error = np.linalg.norm(matrix - expected)
            assert error < 1e-10, f"Large identity permutation failed for {qbit_num} qubits"
            
            # Test random permutation
            np.random.seed(42)
            pattern = list(range(qbit_num))
            np.random.shuffle(pattern)
            perm_gate = Permutation(qbit_num, pattern)
            
            # Check unitarity
            matrix = perm_gate.get_Matrix()
            unitary_check = matrix @ matrix.conj().T
            identity = np.eye(2**qbit_num, dtype=np.complex128)
            error = np.linalg.norm(unitary_check - identity)
            assert error < 1e-10, f"Large permutation not unitary for {qbit_num} qubits"

