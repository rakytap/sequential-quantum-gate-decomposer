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

class Test_CH_inheritance:
    """Tests for the inheritance structure of the CH gate in the Squander package"""

    def test_CH_gate_inheritance(self):
        r"""
        Test that CH gate properly inherits from the base Gate class.
        """

        from squander import Gate
        from squander import CH
        
        # Test class inheritance
        assert issubclass(CH, Gate)
        
        # Create an instance
        ch_gate = CH(3, 0, 1)  # 3 qubits, target 0, control 1
        
        # Test instance inheritance
        assert isinstance(ch_gate, Gate)

    def test_CH_gate_methods(self):
        r"""
        Test that CH gate inherits methods from the base Gate class.
        """

        from squander import CH
        
        # Create a CH gate
        ch_gate = CH(3, 0, 1)
        
        # Test availability of inherited methods
        assert hasattr(ch_gate, 'get_Matrix')
        assert hasattr(ch_gate, 'apply_to')
        assert hasattr(ch_gate, 'get_Target_Qbit')
        assert hasattr(ch_gate, 'get_Control_Qbit')
        assert hasattr(ch_gate, 'set_Target_Qbit')
        assert hasattr(ch_gate, 'set_Control_Qbit')
        assert hasattr(ch_gate, 'get_Name')
        
        # Test method results
        assert ch_gate.get_Target_Qbit() == 0
        assert ch_gate.get_Control_Qbit() == 1
        name = ch_gate.get_Name()
        assert isinstance(name, str)
        assert name == "CH"
    
    