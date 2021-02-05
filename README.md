# Quantum Gate Decomposer

The project was supported by OTKA PD123927 grant and by the Ministry of Innovation and Technology and the National Research, Development and Innovation
Office within the Quantum Information National Laboratory of Hungary.


Quantum Gate Decomposer (QGD) is a heuristic method to decompose an arbitrary NxN Unitary matrix into a sequence of U3 and CNOT gates. 
The QGD library is written in C/C++ providing a Python interface via [C++ extensions](https://docs.python.org/3/library/ctypes.html).
The present package is supplied with Python building script and CMake tools to ease its deployment.
The QGD package can be built with both Intel and GNU compilers, and can be link against various CBLAS libraries installed on the system.
(So far the CLBAS libraries of the GNU Scientific Library, OpenBLAS and the Intel MKL packages were tested.)
In the following we briefly summarize the steps to build, install and use the QGD package. 



### Dependencies

The optimization algorithm of QGD relies on the [multimin](https://www.gnu.org/software/gsl/doc/html/multimin.html) component of the [GNU Scientific Library](https://www.gnu.org/software/gsl/doc/html/index.html). 
We developed and tested the QGD package with GNU Scientific Library of version 2.5 and 2.6.
The dependencies necessary to compile and build the QGD package are the followings:

* [CMake](https://cmake.org/) (>=3.10.2)
* [GNU Scientific Library](https://www.gnu.org/software/gsl/doc/html/index.html) (>=2.5)
* C++/C [Intel](https://software.intel.com/content/www/us/en/develop/tools/compilers/c-compilers.html) (>=14.0.1) or [GNU](https://gcc.gnu.org/) (>=v4.8.1) compiler
* [TBB](https://github.com/oneapi-src/oneTBB) library
* [Intel MKL](https://software.intel.com/content/www/us/en/develop/tools/math-kernel-library.html) (optional)
* [OpenBlas](https://www.openblas.net/) (optional)
* [Doxygen](https://www.doxygen.nl/index.html) (optional)

The Python interface of QGD was developed and tested with Python 3.6 and 3.7.
The QGD Python interface needs the following packages to be installed on the system:

* [Qiskit](https://qiskit.org/documentation/install.html)
* [Numpy](https://numpy.org/install/)
* [scipy](https://www.scipy.org/install.html)



### How to install GNU Scientific Library

If the GNU Scientific Library is not installed on the system, it can be easily easily downloaded and deployed from source by the end user even without administrative privileges.
The GNU Scientific Library can be downloaded from the site [https://www.gnu.org/software/gsl/](https://www.gnu.org/software/gsl/).
After the downloaded package is extracted somewhere in the home directory of the user (**path/to/gslsource**), one should configure the compiling environment using the **configure** tool.
Depending on the individual settings the default compiler to be invoked might be different from HPC to HPC. 
To ensure the usage of the GNU compiler, the following shell command should be executed inside the directory **path/to/gslsource**:

$ ./configure --prefix=path/to/gsl FC=gfortran CC=gcc CXX=g++

(Similarly, Intel compiler can be forced by setting FC=ifort CC=icc and CXX=icpc.)
The installation directory of the compiled GNU Scientific Library is given by **--prefix=path/to/gsl** (which is different from the directory path of 
the source files given by **path/to/gslsource**).
To install GNU Scientific Library the user should have read and write permissions on the path **path/to/gsl** (which might be for example /home/username/gsl).
After the successful configuration the GNU Scientific Library can be compiled by the shell command

$ make

The compilation of the GNU Scientific Library takes some time. When the compilation is done, the package (including the C header files and the static and shared libraries) is installed into the directory **path/to/gsl** by the shell command:

$ make install

### Download the Quantum Gate Decomposer package

The developer version of the Quantum Gate Decomposer package can be downloaded from github repository [https://github.com/rakytap/quantum-gate-decomposer/tree/master](https://github.com/rakytap/quantum-gate-decomposer/tree/master).
After the downloaded package is extracted into the directory **path/to/QGD/package** (which would be the path to the source code of the QGD package), one can proceed to the compilation steps described in the next section.

### How to build the Quantum Gate Decomposer package

The QGD package is equipped with a Python build script and CMake tools to ease the compilation and the deployment of the package.
To ensure that QGD package would find the necessary libraries and header files during compilation time it is advised to define the following environment variables:

$ export GSL_LIB_DIR=path/to/gsl/lib(64)

$ export GSL_INC_DIR=path/to/gsl/include

The QGD package is parallelized via Threading Building Block (TBB) libraries. If TBB is not present in the system, it can be easily installed via the apt utility (sudo apt install libtbb-dev) or it can be downloaded and built from source at 
[https://github.com/oneapi-src/oneTBB](https://github.com/oneapi-src/oneTBB) 
(Notice, that the newest version of the TBB library under the branch onetbb_2021 is aiming to be compatible with the Intel oneApi programming model. It does not contain all the functionalities of previous TBB libraries yet, so QGD package is not compatible with this version of the TBB library. 
For compilation of the QGD package please use previous versions of the library, such as tbb_2017, tbb_2018, tbb_2019 or tbb_2020.)
To exploit TBB parallelism, one should supply the necessary environment variables pointing to the header and library files of the TBB package. For newer
Intel compilers the TBB package is part of the Intel compiler package, similarly to the MKL package. If the TBB library is located at non-standrad path or the QGD package is compiled with GNU compiler, then setting the

$ export TBB_LIB_DIR=path/to/TBB/lib(64)

$ export TBB_INC_DIR=path/to/TBB/include

environment variables are sufficient for successful compilation. 
The QGD package with C++ Python extensions can be compiled via the Python script **setup.py** located in the root directory of the QGD package.
The script automatically finds out the CBLAS library working behind the numpy package and uses it in further linking. 
The **setup.py** script also build the C++ library of the QGD package by making the appropriate CMake calls. 


After the basic environment variables are set, the compilation can be started by the Python command:

$ python3 setup.py build_ext

The command above starts the compilation of the C++ library and builds the necessary C++ Python extensions for the Python interface of the QGD package.


### How to use

The algorithm implemented in the QGD package intends to transform the given unitary into an identity matrix via a sequence of CNOT and U3 operations applied on the unitary. 
Thus, in order to get the decomposition of a unitary, one should rather provide the complex transpose of this unitary as the input for the QGD decomposing process, as can be seen in the examples.

## Standalone executable

During the compilation processes of the QGD package standalone test executable are also built. 
These executable can be launched from directory **/path/to/QGD/package/build/lib*/test_standalone** by command:

$ ./decomposition_test

and

$ ./custom_gate_structure_test

The standalone test programs starts the decomposition of general unitaries.
Alternatively the test programs can be run by the command

$ ctest

from directory **/path/to/QGD/package/build**. 
The sources of these examples are located in **path/to/QGD/package/test_standalone/** and show simple test cases of the usage of the QGD package. 

## Python Interface

The QGD package contains a Python interface allowing the access of the functionalities of the QGD package from Python. 
The usage of the QGD Python interface is demonstrated in the example file **example.py** located in the directory **path/to/QGD/package**, or in test files located in sub-directories of **path/to/QGD/package/qgd_python/*/test**. 
The **example.py** script can be run similarly to any python scripts, while the test files can be invoked by the **pytest** utility.
The example files import the necessary **qgd_python** modules containing the wrapper classes to interface with the C++ QGD library.

It should be noted, however, that the python interface functions are implemented only for few functionalities of the whole QGD API. 
Another desired interface functions can be implemented following the sources of already implemented interface functions in *.cpp source files located in directory
**path/to/QGD/package/qgd_python**
