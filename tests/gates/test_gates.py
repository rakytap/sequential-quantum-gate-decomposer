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

import inspect


from squander.utils import get_unitary_from_qiskit_circuit_operator
from scipy.stats import unitary_group
import numpy as np

import qiskit
qiskit_version = qiskit.version.get_version_info()

from qiskit import QuantumCircuit
def make_u2(self, phi, lam, target_qbit):
    """
    This function is used to add a U2 gate to the circuit.
    It is a workaround for the fact that Qiskit does not have a U2 gate.
    """
    self.u( np.pi/2.0, phi, lam, target_qbit )
QuantumCircuit.u2 = make_u2
import qiskit_aer as Aer   
    
if qiskit_version[0] == '1' or qiskit_version[0] == '2':
    from qiskit import transpile
else :
    from qiskit import execute
    



class Test_operations:
    """This is a test class of the python iterface to the gates of the QGD package"""



    def test_SYC_creation(self):
        r"""
        This method is called by pytest. 
        Test to create an instance of CH gate.

        """

        from squander import SYC


        # number of qubits
        qbit_num = 3

        # target qbit
        target_qbit = 2

        # control_qbit
        control_qbit = 1       

        # creating an instance of the C++ class
        SYC_gate = SYC( qbit_num, target_qbit, control_qbit )


    def test_Operation_Block_creation(self):
        r"""
        This method is called by pytest. 
        Test to create an instance of Operation_Block class.

        """

        from squander.gates.qgd_Circuit_Wrapper import qgd_Circuit_Wrapper        

        # number of qubits
        qbit_num = 3
     

        # creating an instance of the C++ class
        cCircuit = qgd_Circuit_Wrapper( qbit_num )



    def test_Operation_Block_add_operations(self):
        r"""
        This method is called by pytest. 
        Test to add operations to a block of gates

        """

        from squander.gates.qgd_Circuit_Wrapper import qgd_Circuit_Wrapper        

        # number of qubits
        qbit_num = 3
     

        # creating an instance of the C++ class
        cCircuit = qgd_Circuit_Wrapper( qbit_num )



        # target qbit
        target_qbit = 0

        # add U3 gate to the block
        cCircuit.add_U3( target_qbit )


        # target qbit
        target_qbit = 0

        # control_qbit
        control_qbit = 1  

        # add CNOT gate to the block
        cCircuit.add_CNOT( target_qbit, control_qbit )





    def test_Operation_Block_add_block(self):
        r"""
        This method is called by pytest. 
        Test to add operations to a block of gates

        """

        from squander.gates.qgd_Circuit_Wrapper import qgd_Circuit_Wrapper        

        # number of qubits
        qbit_num = 3
     

        # creating an instance of the C++ class
        layer = qgd_Circuit_Wrapper( qbit_num )

        # target qbit
        target_qbit = 0

        # add U3 gate to the block
        layer.add_U3( target_qbit )



        # target qbit
        target_qbit = 0

        # control_qbit
        control_qbit = 1  

        # add CNOT gate to the block
        layer.add_CNOT( control_qbit, target_qbit )


        # add RX gate to the block
        layer.add_RX( target_qbit )

        # creating an instance of the C++ class
        cCircuit = qgd_Circuit_Wrapper( qbit_num )   

        # add inner operation block to the outher operation block
        cCircuit.add_Circuit( layer )



    def perform_gate_matrix_testing( self, gate_obj ):
        """
        This method is called to test the individual gates 
        Test by comparing the gate matrix of squander gate to Qiskit implementation

        """

        is_controlled_gate = (len(gate_obj.__name__) > 1) and ((gate_obj.__name__[0] == 'C') or (gate_obj.__name__[-4:] == "SWAP" or gate_obj.__name__[-3:] == "RXX"))
        is_3qbit_gate = (gate_obj.__name__[:3] == 'CCX' or gate_obj.__name__[:5] == 'CSWAP')
        for qbit_num in range(3,7):

            # target qbit
            target_qbit = qbit_num-3

            # creating an instance of the C++ class
            if not is_controlled_gate:
                # single qbit gate
                squander_gate = gate_obj( qbit_num, target_qbit )
            elif gate_obj.__name__ == 'SWAP':
                # SWAP gate uses vector of target qubits
                # For Qiskit compatibility: swap target_qbit with control_qbit (qbit_num-1)
                control_qbit = qbit_num-1
                squander_gate = gate_obj(qbit_num, [target_qbit, control_qbit])
            elif gate_obj.__name__ == 'RXX':
                # SWAP gate uses vector of target qubits
                # For Qiskit compatibility: swap target_qbit with control_qbit (qbit_num-1)
                control_qbit = qbit_num-1
                squander_gate = gate_obj(qbit_num, [target_qbit, control_qbit])
            elif gate_obj.__name__ == 'CCX':
                # CCX gate uses target_qbit and vector of control qubits
                control_qbit = qbit_num-2
                control_qbit2 = qbit_num-1
                squander_gate = gate_obj(qbit_num, target_qbit, [control_qbit, control_qbit2])
            elif gate_obj.__name__ == 'CSWAP':
                # CSWAP gate uses vectors for both target and control qubits
                target_qbit2 = qbit_num-2
                control_qbit = qbit_num-1
                squander_gate = gate_obj(qbit_num, [target_qbit, target_qbit2], [control_qbit])
            elif is_controlled_gate and not is_3qbit_gate:
                # gate with control qbit
                control_qbit = qbit_num-1
                squander_gate = gate_obj( qbit_num, target_qbit, control_qbit )
            

            
            

        #SQUANDER

            # get the gate matrix     

            # determine the number of input arguments of the get_Matrix function
            parameter_num = squander_gate.get_Parameter_Num()
            if parameter_num == 0:
                gate_matrix_squander = squander_gate.get_Matrix( )
            elif parameter_num > 0: # Inconsistent use of tabs and spaces in indentation
                parameters = ( np.random.rand( parameter_num )*2-1.0 ) * np.pi
                gate_matrix_squander = squander_gate.get_Matrix( parameters )


        #QISKIT

            # Create a Quantum Circuit acting on the q register
            circuit = QuantumCircuit(qbit_num)

            # Add the gate
            gate_name = gate_obj.__name__

            #special cases:
            if gate_name == "U3":
                gate_name = "u"
            
            elif gate_name == "U1":
                gate_name = "p"

            elif gate_name == "CNOT":
                gate_name = "cx"                
                

            gate_name = gate_name.lower()
            

            gate_adding_fnc = getattr(circuit, gate_name )
            if not is_controlled_gate and parameter_num == 0:
                # add parameter-free single qbit gate to Qiskit circuit
                gate_adding_fnc(target_qbit)

            elif not is_controlled_gate and parameter_num > 0:
                # add single qbit gate to Qiskit circuit
                parameters_QISKIT = list(parameters)

                #Squander uses half of the theta function for numerical performance
                if gate_name != "p" and gate_name != "u2":
                    parameters_QISKIT[0] = parameters_QISKIT[0]*2 
                gate_adding_fnc( *parameters_QISKIT, target_qbit)

            elif is_controlled_gate and parameter_num == 0:
                # add parameter-free two-qbit controlled gate to Qiskit circuit
                if gate_name != "ccx" and gate_name != "cswap":
                    control_qbit = qbit_num-1
                    gate_adding_fnc(control_qbit, target_qbit)
                else:
                    control_qbit = qbit_num-2
                    control_qbit2 = qbit_num-1
                    gate_adding_fnc(control_qbit2, control_qbit, target_qbit)

            elif is_controlled_gate and parameter_num > 0:
                # add parameter-free two-qbit controlled gate to Qiskit circuit
                parameters_QISKIT = list(parameters)

                #Squander uses half of the theta function for numerical performance
                if gate_name != "cp":
                    parameters_QISKIT[0] = parameters_QISKIT[0]*2 
                gate_adding_fnc( *parameters_QISKIT, control_qbit, target_qbit)


            # the unitary matrix from the result object
            gate_matrix_qiskit = get_unitary_from_qiskit_circuit_operator( circuit )
            gate_matrix_qiskit = np.asarray(gate_matrix_qiskit)

            #the difference between the SQUANDER and the qiskit result
            delta_matrix=gate_matrix_squander-gate_matrix_qiskit

            # compute norm of matrix
            error=np.linalg.norm(delta_matrix)

            #print("Get_matrix: The difference between the SQUANDER and the qiskit result is: " , np.around(error,2))
            if error >= 1e-3:
                print(f"ERROR: {gate_obj.__name__} with qbit_num={qbit_num}, error={error}")
            assert( error < 1e-3 )        


    def perform_gate_apply_to_testing( self, gate_obj ):
        """
        This method is called to test the individual gates 
        Test by comparing the effect of a gate on an input state 
        by squander gate to Qiskit implementation

        """


        is_controlled_gate = (len(gate_obj.__name__) > 1) and  ((gate_obj.__name__[0] == 'C') or gate_obj.__name__[-4:] == "SWAP")
        is_3qbit_gate = (gate_obj.__name__[:3] == 'CCX' or gate_obj.__name__[:5] == 'CSWAP')
        for qbit_num in range(3,7):

            # target qbit
            target_qbit = qbit_num-3

            # creating an instance of the C++ class
            if not is_controlled_gate:
                # single qbit gate
                squander_gate = gate_obj( qbit_num, target_qbit )
            elif gate_obj.__name__ == 'SWAP':
                # SWAP gate uses vector of target qubits
                # For Qiskit compatibility: swap target_qbit with control_qbit (qbit_num-1)
                control_qbit = qbit_num-1
                squander_gate = gate_obj(qbit_num, [target_qbit, control_qbit])
            elif gate_obj.__name__ == 'RXX':
                # SWAP gate uses vector of target qubits
                # For Qiskit compatibility: swap target_qbit with control_qbit (qbit_num-1)
                control_qbit = qbit_num-1
                squander_gate = gate_obj(qbit_num, [target_qbit, control_qbit])

            elif gate_obj.__name__ == 'CCX':
                # CCX gate uses target_qbit and vector of control qubits
                control_qbit = qbit_num-2
                control_qbit2 = qbit_num-1
                squander_gate = gate_obj(qbit_num, target_qbit, [control_qbit, control_qbit2])
            elif gate_obj.__name__ == 'CSWAP':
                # CSWAP gate uses vectors for both target and control qubits
                target_qbit2 = qbit_num-2
                control_qbit = qbit_num-1
                squander_gate = gate_obj(qbit_num, [target_qbit, target_qbit2], [control_qbit])
            elif is_controlled_gate and not is_3qbit_gate:
                # gate with control qbit
                control_qbit = qbit_num-1
                squander_gate = gate_obj( qbit_num, target_qbit, control_qbit )


            # matrix size of the unitary
            matrix_size = 1 << qbit_num #pow(2, qbit_num )

            initial_state_real = np.random.uniform(-1.0,1.0, (matrix_size,) )
            initial_state_imag = np.random.uniform(-1.0,1.0, (matrix_size,) )
            initial_state = initial_state_real + initial_state_imag*1j
            initial_state = initial_state/np.linalg.norm(initial_state)

            

            

        #SQUANDER

            state_squander = initial_state.copy()


            # determine the number of input arguments of the get_Matrix function
            parameter_num = squander_gate.get_Parameter_Num()
            if parameter_num == 0:
                squander_gate.apply_to( state_squander )
            elif parameter_num > 0: # Inconsistent use of tabs and spaces in indentation
                parameters = ( np.random.rand( parameter_num )*2-1.0 ) * np.pi
                squander_gate.apply_to( state_squander, parameters )


        #QISKIT

            # Create a Quantum Circuit acting on the q register
            circuit_qiskit = QuantumCircuit(qbit_num)
            circuit_qiskit.initialize( initial_state )

            # Add the gate
            gate_name = gate_obj.__name__

            #special cases:
            if gate_name == "U3":
                gate_name = "u"
            
            elif gate_name == "U1":
                gate_name = "p"
    
            elif gate_name == "CNOT":
                gate_name = "cx"

            gate_name = gate_name.lower()
            

            gate_adding_fnc = getattr(circuit_qiskit, gate_name )
            if not is_controlled_gate and parameter_num == 0:
                # add parameter-free single qbit gate to Qiskit circuit
                gate_adding_fnc(target_qbit)

            elif not is_controlled_gate and parameter_num > 0:
                # add single qbit gate to Qiskit circuit
                parameters_QISKIT = list(parameters)

                #Squander uses half of the theta function for numerical performance
                if gate_name != "p" and gate_name != "u2":
                    parameters_QISKIT[0] = parameters_QISKIT[0]*2 
                gate_adding_fnc( *parameters_QISKIT, target_qbit)

            elif is_controlled_gate and parameter_num == 0:
                # add parameter-free two-qbit controlled gate to Qiskit circuit
                if gate_name != "ccx" and gate_name != "cswap":
                    control_qbit = qbit_num-1
                    gate_adding_fnc(control_qbit, target_qbit)
                else:
                    control_qbit = qbit_num-2
                    control_qbit2 = qbit_num-1
                    gate_adding_fnc(control_qbit2, control_qbit, target_qbit)
                    


            elif is_controlled_gate and parameter_num > 0:
                # add parameter-free two-qbit controlled gate to Qiskit circuit
                parameters_QISKIT = list(parameters)

                #Squander uses half of the theta function for numerical performance
                if gate_name != "cp":
                    parameters_QISKIT[0] = parameters_QISKIT[0]*2 
                control_qbit = qbit_num-1
                gate_adding_fnc( *parameters_QISKIT, control_qbit, target_qbit)


            # Execute and get the state vector
            if qiskit_version[0] == '1' or qiskit_version[0] == '2':
    
                circuit_qiskit.save_statevector()
    
                backend = Aer.AerSimulator(method='statevector')
                compiled_circuit = transpile(circuit_qiskit, backend)
                result = backend.run(compiled_circuit).result()
        
                state_QISKIT = result.get_statevector(compiled_circuit)		
       
        
            elif qiskit_version[0] == '0':
    
                # Select the StatevectorSimulator from the Aer provider
                simulator = Aer.get_backend('statevector_simulator')	
        
                backend = Aer.get_backend('aer_simulator')
                result = execute(circuit_qiskit, simulator).result()
        
                state_QISKIT = result.get_statevector(circuit_qiskit)

            state_QISKIT = np.array(state_QISKIT)


            # compute norm of matrix
            error=np.linalg.norm( state_squander-state_QISKIT )

            assert( error < 1e-3 )        


    def test_gates(self):
        """
        This method is called by pytest. 
        Test by comparing all of the squander gates to Qiskit implementation

        """

        import squander.gates.gates_Wrapper as gates

        for name in dir(gates):
            obj = getattr(gates, name)

            if inspect.isclass(obj):
                
                if name == "SYC" or name == "Gate" or name=="CR" or name=="CROT":
                    continue

                print(f"testing gate: {name}")

                self.perform_gate_matrix_testing( obj )
                if name == "SWAP" or name =="RXX":
                    continue
                self.perform_gate_apply_to_testing( obj )












      

