[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4508680.svg)](https://doi.org/10.5281/zenodo.4508680)
 <a href="https://trackgit.com">
<img src="https://us-central1-trackgit-analytics.cloudfunctions.net/token/ping/l0sfey1m19at85951dwl" alt="trackgit-views" />
</a>

# Sequential Quantum Gate Decomposer (SQUANDER)


The Sequential Quantum Gate Decomposer (SQUANDER) package provides a novel solution to decompose any n-qubit unitaries into a sequence of one-qubit rotations and two-qubit controlled gates (such as controlled NOT or controlled phase gate). SQUANDER utilizes two novel gate synthesis techniques reported in Refereces [1] and [2].
(i) To synthesize general unitaries SQUANDER applies periodic layers of two-qubit and parametric one-qubit gates to an n-qubit unitary such that the resulting unitary is 1-qubit decoupled, i.e., is a tensor product of an (n-1)-qubit and a 1-qubit unitary. Continuing the decoupling procedure sequentially one arrives at the end to a full decomposition of the original unitary into 1- and 2-qubit gates. SQUANDER provides lower CNOT counts for generic n-qubit unitaries (up to n=6)  than the previously provided lower bounds.
(ii) An adaptive circuit compression is used to optimize quantum circuit by the application of parametric two-qubit gates in the synthesis process. The utilization of these parametric two-qubit gates in the circuit design allows one to transform the discrete combinatorial problem of circuit synthesis into an optimization problem over continuous variables. The circuit is then compressed by a sequential removal of two-qubit gates from the design, while the remaining building blocks are continuously adapted to the reduced gate structure by iterated learning cycles.


The SQUANDER library is written in C/C++ providing a Python interface via [C++ extensions](https://docs.python.org/3/library/ctypes.html).
The present package is supplied with Python building script and CMake tools to ease its deployment.
The SQUANDER package can be built with both Intel and GNU compilers, and can be link against various CBLAS libraries installed on the system.
(So far the CLBAS libraries of the GNU Scientific Library, OpenBLAS and the Intel MKL packages were tested.)
In the following we briefly summarize the steps to build, install and use the SQUANDER package. 


The project was supported by grant OTKA PD123927 and by the Ministry of Innovation and Technology and the National Research, Development and Innovation
Office within the Quantum Information National Laboratory of Hungary.

Find the documantation of the SQUANDER package at [CodeDocs[xyz]](https://codedocs.xyz/rakytap/sequential-quantum-gate-decomposer/)



### Contact Us

Have a question about the SQUANDER package? Don't hesitate to contact us at the following e-mails:

* Zoltán Zimborás (researcher): zimboras.zoltan@wigner.hu
* Peter Rakyta (developer): peter.rakyta@ttk.elte.hu



### Dependencies

The dependencies necessary to compile and build the SQUANDER package are the followings:

* [CMake](https://cmake.org/) (>=3.10.2)
* C++/C [Intel](https://software.intel.com/content/www/us/en/develop/tools/compilers/c-compilers.html) (>=14.0.1) or [GNU](https://gcc.gnu.org/) (>=v4.8.1) compiler
* [TBB](https://github.com/oneapi-src/oneTBB) library (shipped with tbb-devel Python package)
* [Intel MKL](https://software.intel.com/content/www/us/en/develop/tools/math-kernel-library.html) (optional)
* [OpenBlas](https://www.openblas.net/) (optional, recommended)
* [LAPACKE](https://www.netlib.org/lapack/lapacke.html)
* [Doxygen](https://www.doxygen.nl/index.html) (optional)

The Python interface of SQUANDER was developed and tested with Python 3.6-3.9.
The SQUANDER Python interface needs the following packages to be installed on the system:

* [Qiskit](https://qiskit.org/documentation/install.html)
* [Numpy](https://numpy.org/install/)
* [scipy](https://www.scipy.org/install.html)
* [scikit-build](https://scikit-build.readthedocs.io/en/latest/)
* [tbb-devel](https://pypi.org/project/tbb-devel/) (containing the TBB Library)




### Install SQUANDER from Python Package Index (PyPI)

Since version 1.7.1 the SQUANDER package is accessible at Python Package Index (PyPI). The package can be installed on linux systems following the steps outlined below:

$ pip install numpy swig tbb-devel wheel scikit-build ninja qiskit

$ pip install squander

### Download the SQUANDER package

The developer version of the Quantum Gate Decomposer package can be downloaded from github repository [https://github.com/rakytap/quantum-gate-decomposer/tree/master](https://github.com/rakytap/quantum-gate-decomposer/tree/master).
After the package is downloaded into the directory **path/to/SQUANDER/package** (which would be the path to the source code of the SQUANDER package), one can proceed to the compilation steps described in the next section.

### How to build the SQUANDER package on Unix/Linux/MacOS

The SQUANDER package is equipped with a Python build script and CMake tools to ease the compilation and the deployment of the package.
The SQUANDER package is parallelized via Threading Building Block (TBB) libraries. If TBB is not present in the system, it can be easily installed via python package [tbb-devel](https://pypi.org/project/tbb-devel/).
Alternatively the TBB libraries can be installed via apt or yum utility (sudo apt install libtbb-dev) or it can be downloaded from [https://github.com/oneapi-src/oneTBB](https://github.com/oneapi-src/oneTBB)   and built from source. 
In this case one should supply the necessary environment variables pointing to the header and library files of the TBB package. For newer
Intel compilers the TBB package is part of the Intel compiler package, similarly to the MKL package. If the TBB library is located at non-standrad path or the SQUANDER package is compiled with GNU compiler, then setting the

$ export TBB_LIB_DIR=path/to/TBB/lib(64)

$ export TBB_INC_DIR=path/to/TBB/include

environment variables are sufficient for successful compilation. 
When the TBB library is installed via a python package, setting the environment variables above is not necessary.
The SQUANDER package with C++ Python extensions can be compiled via the Python script **setup.py** located in the root directory of the SQUANDER package.
The script automatically finds out the CBLAS library working behind the numpy package and uses it in further linking. 
The **setup.py** script also build the C++ library of the SQUANDER package by making the appropriate CMake calls. 



### Developer build


We recommend to install the SQUANDER package in the Anaconda environment. In order to install the necessary requirements, follow the steps below:

Creating new python environment: 

$ conda create -n qgd

Activate the new anaconda environment

$ conda activate qgd

Install dependencies:

$ conda install numpy scipy pip pytest scikit-build tbb-devel

$ pip install qiskit matplotlib 

After the basic environment variables are set and the dependencies are installed, the compilation of the package can be started by the Python command:

$ python3 setup.py build_ext

The command above starts the compilation of the SQUANDER C++ library and builds the necessary C++ Python interface extensions of the SQUANDER package in place.
After a successful build, one can register the SQUANDER package in the Python distribution in developer (i.e. editable) mode by command:

$ python -m pip install -e .


### Binary distribution

After the environment variables are set it is possible to build wheel binaries of the SQUANDER package. 
In order to launch the compilation process from python, **[scikit-build](https://scikit-build.readthedocs.io/en/latest/)** package is necessary.
(It is optional to install the ninja package which speeds up the building process by parallel compilation.)
The binary wheel can be constructed by command

$ python3 setup.py bdist_wheel

in the root directory of the SQUADER package.
The created SQUANDER wheel can be installed on the local machine by issuing the command from the directory **path/to/SQUANDER/package/dist**

$ pip3 install qgd-*.whl

We notice, that the created wheel is not portable, since it contains hard coded link to external libraries (TBB and CBLAS).


### Source distribution

A portable source distribution of the SQUANDER package can be created by a command launched from the root directory of the SQUANDER package:

$ python3 setup.py sdist

In order to create a source distribution it is not necessary to set the environment variables, since this script only collects the necessary files and pack them into a tar ball located in the directory **path/to/SQUANDER/package/dist**. 
In order to install the SQUANDER package from source tar ball, see the previous section discussing the initialization of the environment variables.
The package can be compiled and installed by the command

$ pip3 install qgd-*.tar.gz

issued from directory **path/to/SQUANDER/package/dist**
(It is optional to install the ninja package which speeds up the building process by parallel compilation.)


### Build and Install on Microsoft Windows with Microsoft Visual C++

CMake must be in the path and able to find the MSVC compiler e.g.

$ set PATH=%PATH%;C:\Program Files\cmake\bin

Now set the TBB and BLAS folders and build via:

$ set TBB_LOCATION=<Python_Folder>/LocalCache/local-packages

$ set TBB_INC_DIR=%TBB_LOCATION%/Library/include

$ set TBB_LIB_DIR=%TBB_LOCATION%/Library/lib

$ set LIB=<BLAS_Location>/lib;<LAPACK_Location>/lib

$ python setup.py build_ext -DTBB_HEADER=<TBB_Location>\Library\include\

Installation merely requires copying DLL files (if they are not in the path):

$ copy _skbuild\win-amd64-3.9\cmake-install\bin .\qgd_python\decomposition

$ copy "%TBB_LOCATION%\Library\bin\tbb12.dll" .\qgd_python\decomposition

$ copy "%TBB_LOCATION%\Library\bin\tbbmalloc.dll" .\qgd_python\decomposition

Verify the installation:

$ python -m pytest


### How to use

The algorithm implemented in the SQUANDER package intends to transform the given unitary into an identity matrix via a sequence of two-qubit and one-qubit gate operations applied on the unitary. 
Thus, in order to get the decomposition of a unitary, one should rather provide the complex transpose of this unitary as the input for the SQUANDER decomposing process, as can be seen in the examples.


## Python Interface

The SQUANDER package contains a Python interface allowing the access of the functionalities of the SQUANDER package from Python. 
The usage of the SQUANDER Python interface is demonstrated in the example files in the directory **examples** located in the directory **path/to/SQUANDER/package**, or in test files located in sub-directories of **path/to/SQUANDER/package/qgd_python/*/test**. 

### Example code snippet

Here we provide an example to use the SQUANDER package. The following python interface is accessible from version 1.8.0. 
In this example we use two optimization engines for the decomposition:
1. An evolutionary engine called AGENTS
2. Second order gradient descend algorithm (BFGS)

Firstly we construct a Python map to set hyper-parameters during the gate synthesis.

        #Python map containing hyper-parameters
        config = { 'max_outer_iterations': 1, 
                    'max_inner_iterations_agent': 25000, 
                    'max_inner_iterations_compression': 10000,
                    'max_inner_iterations' : 500,
                    'max_inner_iterations_final': 5000, 		
                    'Randomized_Radius': 0.3, 
                    'randomized_adaptive_layers': 1,
                    'optimization_tolerance_agent': 1e-2,
                    'optimization_tolerance': 1e-8,
                    'agent_num': 10}
 
Next we initialize the decomposition class with the unitary Umtx to be decomposed. 

        # creating a class to decompose the unitary
        from squander import N_Qubit_Decomposition_adaptive
        cDecompose = N_Qubit_Decomposition_adaptive( Umtx.conj().T, config=config )

The verbosity of the execution output can be controlled by the function call 

        # setting the verbosity of the decomposition
        cDecompose.set_Verbose( 3 )


We construct the initial trial gate structure for the optimization consisting of 2 levels of adaptive layer. 
(1 level is made of qubit_num*(qubit_num-1) two-qubit building blocks if all-to-all connectivity is assumed)


        # adding decomposing layers to the gate structure
        levels = 2
        for idx in range(levels):
            cDecompose.add_Adaptive_Layers()

        cDecompose.add_Finalyzing_Layer_To_Gate_Structure()
        
We can construct an initial parameters set for the optimization by retrieving the number of free parameters. If the initial parameter set is not set, random parameters are used by default.

        # setting intial parameter set
        parameter_num = cDecompose.get_Parameter_Num()
        parameters = np.zeros( (parameter_num,1), dtype=np.float64 )
        cDecompose.set_Optimized_Parameters( parameters )

We can choose between several engines to solve the optimization problem. Here we use an evolutionary based algorithm named 'AGENTS'

        # setting optimizer
        cDecompose.set_Optimizer("AGENTS")
	
The optimization process is started by the function call	

        # starting the decomposition
        cDecompose.get_Initial_Circuit()
	
The optimization process terminates by either reaching the tolerance 'optimization_tolerance_agent' or by reaching the maximal iteration number 'max_inner_iterations_agent', or if the engines identifies a convergence to a local minimum. The SQUANDER framework enables one to continue the optimization using a different engine. In particular we set a second order gradient descend method 'BFGS' 
In order to achieve the best performance one can play around with the hyper-parameters in the map 'config'. (Optimization strategy AGENTS is good in avoiding local minima or get through flat areas of the optimization landscape. Then a gradient descend method can be used for faster convergence toward a solution.)

        # setting optimizer
        cDecompose.set_Optimizer("BFGS")

        # continue the decomposition with a second optimizer method
        cDecompose.get_Initial_Circuit()
	
After solving the optimization problem for the initial gate structure, we can initiate gate compression iterations. (This step can be omited.)	

        # starting compression iterations
        cDecompose.Compress_Circuit()
	
By finalizing the gate structure we replace the CRY gates with CNOT gates. (CRY gates with small rotation angle are approximately expressed with a single CNOT gate, so further optimization process needs to be initiated.)

        # finalize the gate structure (replace CRY gates with CNOT gates)
        cDecompose.Finalize_Circuit()

Finally, we can retrieve the decomposed quantum circuit in QISKIT format.

        # get the decomposing operations
        quantum_circuit = cDecompose.get_Quantum_Circuit()



### How to cite us

If you have found our work useful for your research project, please cite us by

[1] Péter Rakyta, Zoltán Zimborás, Approaching the theoretical limit in quantum gate decomposition, Quantum 6, 710 (2022). <br>
[2] Péter Rakyta, Zoltán Zimborás, Efficient quantum gate decomposition via adaptive circuit compression, arXiv:2203.04426. <br>
[3] Peter Rakyta, Gregory Morse, Jakab Nádori, Zita Majnay-Takács, Oskar Mencer, Zoltán Zimborás, Highly optimized quantum circuits synthesized via data-flow engines, arXiv:2211.07685







