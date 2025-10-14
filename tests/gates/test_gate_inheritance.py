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

import importlib
import pytest
import inspect

class Test_Gate_inheritance:
    """Tests for the inheritance structure of the CH gate in the Squander package"""

    @pytest.mark.parametrize("gate_class", [
        "CH",
        "CNOT",
        "CRY",
        "CZ",
        "H",
        "R",
        "RX",
        "RY",
        "RZ",
        "SX",
        "SYC",
        "U3",
        "X",
        "Y",
        "Z",
        'CSWAP'
    ])
    def test_gate_inheritance(self, gate_class):
        r"""
        Test that gates properly inherits from the base Gate class.
        """
        
        from squander import Gate
        
        control_gates = ["CH","CNOT","CRY","CZ","SYC"]
        multi_qubit_gates = ["CSWAP"]
        # Test class inheritance
        module = importlib.import_module('squander')
        gate_cls = getattr(module, gate_class)

        assert issubclass(gate_cls, Gate)
        
        # Create an instance
        if gate_class in control_gates:    
            gate_ins = gate_cls(3, 0, 1) # 3 qubits, target 0, control 1
        elif gate_class in multi_qubit_gates:
            gate_ins = gate_cls(3,[0,1],[2])
        else:
            gate_ins = gate_cls(3, 0)

        assert isinstance(gate_ins, Gate)

    @pytest.mark.parametrize("gate_class", [
        "CH",
        "CNOT",
        "CRY",
        "CZ",
        "H",
        "R",
        "RX",
        "RY",
        "RZ",
        "SX",
        "SYC",
        "U3",
        "X",
        "Y",
        "Z",
        'CSWAP'
    ])
    def test_gate_methods(self, gate_class):
        r"""
        Test that gate inherits methods from the base Gate class.
        """

        from squander import Gate

        control_gates = ["CH","CNOT","CRY","CZ","SYC"]
        multi_qubit_gates = ["CSWAP"]

        module = importlib.import_module('squander')
        gate_cls = getattr(module, gate_class)
        
        # Create an instance of the base Gate class to get its methods
        base_gate       = Gate(3)
        base_methods    = [name for name, obj in inspect.getmembers(base_gate) 
                           if (callable(obj) and not name.startswith('_'))]    
        # Create an instance
        if gate_class in control_gates:    
            gate_ins = gate_cls(3, 0, 1) # 3 qubits, target 0, control 1
        elif gate_class in multi_qubit_gates:
            gate_ins = gate_cls(3,[1,0],[2])
        else:
            gate_ins = gate_cls(3, 0)
        # Check that each method from the base class is available in the specific gate
        for method_name in base_methods:
            assert hasattr(gate_ins, method_name), f"{gate_class} is missing method {method_name}"

        # Test method results
        assert gate_ins.get_Target_Qbit() == 0
        if gate_class in control_gates:  
            assert gate_ins.get_Control_Qbit() == 1
        elif gate_class in multi_qubit_gates:
            assert gate_ins.get_Target_Qbits() == [0,1]
        name = gate_ins.get_Name()
        assert isinstance(name, str)
        assert len(name) > 0
