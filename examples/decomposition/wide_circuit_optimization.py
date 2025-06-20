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
## \file wide_circuit_optimization.py
## \brief Simple example python code demonstrating a wide circuit optimization

import squander.decomposition.qgd_Wide_Circuit_Optimization as Wide_Circuit_Optimization




import time
    

if __name__ == '__main__':


    squander_config = {  
            'strategy': "TreeSearch", 
            'test_subcircuits': True,
            'test_final_circuit': True,
            'max_partition_size': 3
    }

    start_time = time.time()

    wide_circuit_optimizer = Wide_Circuit_Optimization.qgd_Wide_Circuit_Optimization( squander_config )

    wide_circuit_optimizer.OptimizeWideCircuit()

    print("--- %s seconds elapsed during optimization ---" % (time.time() - start_time))


