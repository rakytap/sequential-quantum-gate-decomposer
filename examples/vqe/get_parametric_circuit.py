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



