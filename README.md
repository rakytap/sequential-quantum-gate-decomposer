[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4508680.svg)](https://doi.org/10.5281/zenodo.4508680)
 <a href="https://trackgit.com">
<img src="https://us-central1-trackgit-analytics.cloudfunctions.net/token/ping/l0sfey1m19at85951dwl" alt="trackgit-views" />
</a>

# Sequential Quantum Gate Decomposer (SQUANDER)

The Sequential Quantum Gate Decomposer (SQUANDER) package introduces innovative techniques for training parametric quantum circuits based on qubits. SQUANDER offers Python interfaces to facilitate a variety of numerical experiments, encompassing quantum gate synthesis, variational quantum eigensolver, and state preparation. Leveraging two gate synthesis methods, as outlined in References [1], [2], and [3], SQUANDER excels in decomposing n-qubit unitaries into sequences of one-qubit rotations and two-qubit controlled gates. The computational backbone involves a parallel C/C++ framework and vectorized AVX gate kernels, enhancing the efficiency of underlying numerical simulations. Beyond conventional first (gradient descent and ADAM) and second-order (BFGS) gradient-based optimizers, SQUANDER integrates an innovative gradient-free optimization technique detailed in Reference [4]. This technique exhibits robust numerical behavior, particularly effective in circumventing barren plateaus. The handcrafted optimization strategies within SQUANDER are designed to accommodate the periodicity inherent in the optimization landscape, ensuring resilient numerical efficiency.



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

The Python interface of SQUANDER was developed and tested with Python 3.6-3.10.
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

$ conda install numpy scipy pip pytest scikit-build tbb-devel ninja

$ pip install qiskit qiskit-aer matplotlib 

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

$ copy _skbuild\win-amd64-3.9\cmake-install\bin .\squander\decomposition

$ copy "%TBB_LOCATION%\Library\bin\tbb12.dll" .\squander\decomposition

$ copy "%TBB_LOCATION%\Library\bin\tbbmalloc.dll" .\squander\decomposition

Verify the installation:

$ python -m pytest


### How to use

he Squander package provides high perfromance computational library to

* decompose unitaries into a quantum circuti composed from single- and two-qubit gates.
* simulate the evolution of state vectors under the effect of quantum circuits
* generate quantum circuit for the pusrpose of state preparation 
* run variational quantum algorithms with the incorporated quantum computer simulator.

## Python Interface

The SQUANDER package contains a Python interface allowing the access of the functionalities of the SQUANDER package from Python. 
The usage of the SQUANDER Python interface is demonstrated in the example files in the directory **examples** located in the directory **path/to/SQUANDER/package**, or in test files located in sub-directories of **path/to/SQUANDER/package/qgd_python/*/test**. 



### How to cite us

If you have found our work useful for your research project, please cite us by

[1] Péter Rakyta, Zoltán Zimborás, Approaching the theoretical limit in quantum gate decomposition, Quantum 6, 710 (2022). <br>
[2] Péter Rakyta, Zoltán Zimborás, Efficient quantum gate decomposition via adaptive circuit compression, arXiv:2203.04426. <br>
[3] Peter Rakyta, Gregory Morse, Jakab Nádori, Zita Majnay-Takács, Oskar Mencer, Zoltán Zimborás, Highly optimized quantum circuits synthesized via data-flow engines, Journal of Computational Physics 500, 112756 (2024). <br>
[4] Jakab Nádori, Gregory Morse, Zita Majnay-Takács, Zoltán Zimborás, Péter Rakyta, The promising path of evolutionary optimization to avoid barren plateaus, arXiv:2402.05227.








