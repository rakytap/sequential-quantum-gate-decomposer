# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 14:42:56 2020
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

@author: Peter Rakyta, Ph.D.
"""
## \file example_Renyi_etropy.py
## \brief Simple example python code to evaluate the second Rényi entropy of a parametric quantum circuit at random parameter set.

from squander import N_Qubit_Decomposition_adaptive  


class Test_Renyi_entropy:
    """This is a test the calculation of Renyi entropy"""





    def create_custom_gate_structure(self, qbit_num, level_num=2):
        """
        Add layers to disentangle the 3rd qubit from the others
        linear chain with IBM native operations

        """

        from squander import Circuit


        # creating an instance of the wrapper class Circuit
        Circuit_ret = Circuit( qbit_num )  


        Layer = Circuit( qbit_num )
        for target_qbit in range(qbit_num):
            Layer.add_U3(target_qbit, True, True, True )

        Circuit_ret.add_Circuit( Layer )
			   


        for idx in range(0,level_num):

            for target_qbit in range(1, qbit_num-1, 2):

                    # creating an instance of the wrapper class Circuit
                    Layer = Circuit( qbit_num )

                    Layer.add_CNOT( target_qbit=target_qbit, control_qbit=target_qbit+1 )
                    Layer.add_U3( target_qbit, True, True, True )
                    Layer.add_U3( target_qbit+1, True, True, True )

                    Circuit_ret.add_Circuit( Layer )


            for target_qbit in range(0, qbit_num-1, 2):

                    # creating an instance of the wrapper class Circuit
                    Layer = Circuit( qbit_num )
        
                    Layer.add_CNOT( target_qbit=target_qbit, control_qbit=target_qbit+1 )
                    Layer.add_U3( target_qbit, True, True, True )
                    Layer.add_U3( target_qbit+1, True, True, True )

                    Circuit_ret.add_Circuit( Layer )




          
        return Circuit_ret


    def test_Renyi_entropy(self):

        import numpy as np

    
        # the number of qubits spanning the unitary
        qbit_num  = 22
        level_num = 15

        gate_structure = self.create_custom_gate_structure( qbit_num, level_num )



        # get the number of parameters
        num_of_parameters = gate_structure.get_Parameter_Num()


        # create random parameter set
        parameters = np.random.uniform( 0.0, 2*np.pi, (num_of_parameters,) )

        # calculate the second Rényi entropy

        qubit_list = [0,1]


        entropy = gate_structure.get_Second_Renyi_Entropy( parameters=parameters, qubit_list=qubit_list )
        print( 'The second Renyi entropy is:',  entropy) 


        page_entropy = len(qubit_list) * np.log(2.0) - 1.0/( pow(2, qbit_num-2*len(qubit_list)+1) )

        print( 'The page entropy: ', page_entropy)


        assert( np.abs( page_entropy-entropy) < 1e-1 )











