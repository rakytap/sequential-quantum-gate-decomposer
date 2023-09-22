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
'''

import vqe_utils
import numpy as np
import time
import random


# load preconstructed optimized parameters
vqe_utils.load_preconstructed_data('optimized_parameters_vqe.npy', 'parameter_values_vqe.npy')

start = time.time() 
random.seed(start)

# determine random parameter value alpha
alpha = random.random()*4*np.pi
print('alpha=', alpha)

# determine the quantum circuit at parameter value alpha
qc = vqe_utils.get_interpolated_circuit( alpha )


time_loc = time.time() - start
print('average time: ', time_loc, 's.')



