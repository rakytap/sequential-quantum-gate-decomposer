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

You should have received a copy of the GNU General Public License
along with this program.  If not, see http://www.gnu.org/licenses/.

@author: Peter Rakyta, Ph.D.
"""
## \file test_Global_Phase.py
## \brief Functionality test cases for the N_Qubit_Decomposition_adaptive class.



# cerate unitary q-bit matrix
from scipy.stats import unitary_group
import numpy as np


try:
    from mpi4py import MPI
    MPI_imported = True
except ModuleNotFoundError:
    MPI_imported = False

from squander import N_Qubit_Decomposition_adaptive
from scipy.io import loadmat

def test_global_phase():
	# load the unitary from file
	data = loadmat('Umtx.mat')
	# The unitary to be decomposed  
	Umtx = data['Umtx']
	# creating a class to decompose the unitary
	cDecompose = N_Qubit_Decomposition_adaptive( Umtx.conj().T, level_limit_max=5, level_limit_min=0 )
	angl = cDecompose.get_Global_Phase()
	Umtx_assert = cDecompose.get_Unitary()

	assert(np.abs(angl)<2*np.pi)

	angl_new = np.pi/3
	cDecompose.set_Global_Phase(angl_new)
	global_phase_factor = np.sqrt(2)*np.cos(angl_new)+1j*np.sqrt(2)*np.sin(angl_new)
	cDecompose.apply_Global_Phase_Factor()
	angl_assert = cDecompose.get_Global_Phase()

	assert(np.abs(angl_assert)<1e-8)
	Umtx_assert_new=cDecompose.get_Unitary()
	assert(np.sum(np.abs(Umtx_assert*global_phase_factor-Umtx_assert_new))<0.00001)


	   
