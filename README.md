[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4508680.svg)](https://doi.org/10.5281/zenodo.4508680)

# Sequential Quantum Gate Decomposer (SQUANDER)


The Sequential Quantum Gate Decomposer (SQUANDER) package provides a novel solution to decompose any n-qubit unitary into a sequence of one-qubit rotations and two-qubit controlled gates (such as controlled NOT or controlled phase gate). SQUANDER utilizes a new gate synthesis technique that applies periodic layers of two-qubit and parametric one-qubit gates to an n-qubit unitary such that the resulting unitary is 1-qubit decoupled, i.e., is a tensor product of an (n-1)-qubit and a 1-qubit unitary. Continuing the decoupling procedure sequentially one arrives at the end to a full decomposition of the original unitary into 1- and 2-qubit gates. SQUANDER provides lower CNOT counts for generic n-qubit unitaries (up to n=6)  than the previously provided lower bounds.

The SQUANDER library is written in C/C++ providing a Python interface via [C++ extensions](https://docs.python.org/3/library/ctypes.html).
The present package is supplied with Python building script and CMake tools to ease its deployment.
The SQUANDER package can be built with both Intel and GNU compilers, and can be link against various CBLAS libraries installed on the system.
(So far the CLBAS libraries of the GNU Scientific Library, OpenBLAS and the Intel MKL packages were tested.)
In the following we briefly summarize the steps to build, install and use the SQUANDER package. 


The project was supported by grant OTKA PD123927 and by the Ministry of Innovation and Technology and the National Research, Development and Innovation
Office within the Quantum Information National Laboratory of Hungary.





### Contact Us

Have a question about the SQUANDER package? Don't hesitate to contact us at the following e-mails:

* Zoltán Zimborás (researcher): zimboras.zoltan@wigner.hu
* Peter Rakyta (developer): peter.rakyta@ttk.elte.hu



### Dependencies

The optimization algorithm of SQUANDER relies on the [multimin](https://www.gnu.org/software/gsl/doc/html/multimin.html) component of the [GNU Scientific Library](https://www.gnu.org/software/gsl/doc/html/index.html). 
We developed and tested the SQUANDER package with GNU Scientific Library of version 2.5, 2.6 and 2.7.
The dependencies necessary to compile and build the SQUANDER package are the followings:

* [CMake](https://cmake.org/) (>=3.10.2)
* [GNU Scientific Library](https://www.gnu.org/software/gsl/doc/html/index.html) (>=2.5)
* C++/C [Intel](https://software.intel.com/content/www/us/en/develop/tools/compilers/c-compilers.html) (>=14.0.1) or [GNU](https://gcc.gnu.org/) (>=v4.8.1) compiler
* [TBB](https://github.com/oneapi-src/oneTBB) library
* [Intel MKL](https://software.intel.com/content/www/us/en/develop/tools/math-kernel-library.html) (optional)
* [OpenBlas](https://www.openblas.net/) (optional, recommended)
* [LAPACKE](https://www.netlib.org/lapack/lapacke.html)
* [Doxygen](https://www.doxygen.nl/index.html) (optional)

The Python interface of SQUANDER was developed and tested with Python 3.6, 3.7 and 3.8.
The SQUANDER Python interface needs the following packages to be installed on the system:

* [Qiskit](https://qiskit.org/documentation/install.html)
* [Numpy](https://numpy.org/install/)
* [scipy](https://www.scipy.org/install.html)
* [scikit-build](https://scikit-build.readthedocs.io/en/latest/)
* [tbb-devel](https://pypi.org/project/tbb-devel/)



### How to build GNU Scientific Library

If the GNU Scientific Library is not installed on the system, it can be easily easily downloaded and deployed from source by the end user even without administrative privileges.
The GNU Scientific Library can be downloaded from the site [https://www.gnu.org/software/gsl/](https://www.gnu.org/software/gsl/).
After the downloaded package is extracted somewhere in the home directory of the user (**path/to/gsl/source**), one should configure the building environment using the **configure** tool.
Depending on the individual settings the default compiler to be invoked might be different from HPC to HPC. 
To ensure the usage of the GNU compiler, the following shell command should be executed inside the directory **path/to/gsl/source**:

$ ./configure --prefix=path/to/gsl FC=gfortran CC=gcc CXX=g++

(Similarly, Intel compiler can be forced by setting FC=ifort CC=icc and CXX=icpc.)
The installation directory of the compiled GNU Scientific Library is given by **--prefix=path/to/gsl** (which is different from the directory path of 
the source files given by **path/to/gslsource**).
To install GNU Scientific Library the user should have read and write permissions on the path **path/to/gsl** (which might be for example /home/username/gsl).
After the successful configuration the GNU Scientific Library can be compiled by the shell command

$ make

The compilation of the GNU Scientific Library takes some time. When the compilation is done, the package (including the C header files and the static and shared libraries) is installed into the directory **path/to/gsl** by the shell command:

$ make install

### Download the SQUANDER package

The developer version of the Quantum Gate Decomposer package can be downloaded from github repository [https://github.com/rakytap/quantum-gate-decomposer/tree/master](https://github.com/rakytap/quantum-gate-decomposer/tree/master).
After the package is downloaded into the directory **path/to/SQUANDER/package** (which would be the path to the source code of the SQUANDER package), one can proceed to the compilation steps described in the next section.

### How to build the SQUANDER package

The SQUANDER package is equipped with a Python build script and CMake tools to ease the compilation and the deployment of the package.
To ensure that SQUANDER package would find the necessary libraries and header files during compilation time it is advised to define the following environment variables:

$ export GSL_LIB_DIR=path/to/gsl/lib(64)

$ export GSL_INC_DIR=path/to/gsl/include

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



### In-place build

After the basic environment variables are set, the compilation can be started by the Python command:

$ python3 setup.py build_ext

The command above starts the compilation of the C++ library and builds the necessary C++ Python extensions for the Python interface of the SQUANDER package in place.


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


### How to use

The algorithm implemented in the SQUANDER package intends to transform the given unitary into an identity matrix via a sequence of two-qubit and one-qubit gate operations applied on the unitary. 
Thus, in order to get the decomposition of a unitary, one should rather provide the complex transpose of this unitary as the input for the SQUANDER decomposing process, as can be seen in the examples.


## Python Interface

The SQUANDER package contains a Python interface allowing the access of the functionalities of the SQUANDER package from Python. 
The usage of the SQUANDER Python interface is demonstrated in the example files in the directory **examples** located in the directory **path/to/SQUANDER/package**, or in test files located in sub-directories of **path/to/SQUANDER/package/qgd_python/*/test**. 
The example files import the necessary **qgd_python** modules containing the wrapper classes to interface with the C++ @QGD library. 
(So the $QGD package need to be installed or the compiled package needs to be added to the Python search path.)


### How to cite us

If you have found our work useful for your research project, please cite us by

Rakyta,Peter, & Zimborás,Zoltán. (2021, February 4). Sequential Quantum Gate Decomposer (SQUANDER) (Version 1.4). Zenodo. http://doi.org/10.5281/zenodo.4508680




