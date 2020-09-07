# Quantum Gate Decomposer

Quantum Gate Decomposer (QGD) is an optimization method to decompose an arbitrary NxN Unitary matrix into a sequence of U3 and CNOT gates. 
It is written in C/C++ providing a simple Python interface via [ctypes](https://docs.python.org/3/library/ctypes.html) and a possibility to run QGD as a standalone C executable.
The present package is supplied with automake tools to ease its deployment.
The QGD package can be built with both Intel and GNU compilers, and link against various CBLAS libraries installed on the system.
(So far the CLBAS libraries of the GNU Scientific Library and the Intel MKL packages were tested.)
In the following we briefly summarize the steps to build, install and use the QGD package. 

The project was supported ... HUNQT ...

### Dependencies

The optimization algorithm of QGD relies on the [multimin](https://www.gnu.org/software/gsl/doc/html/multimin.html) component of the [GNU Scientific Library](https://www.gnu.org/software/gsl/doc/html/index.html). 
We developed and tested the QGD package with GNU Scientific Library of version 2.5 and 2.6.
The dependencies necessary to compile and build the QGD package are the followings:

* [automake](https://www.gnu.org/software/automake/) (for further development purposes)
* [autoconf](https://www.gnu.org/software/autoconf/) (for further development purposes)
* [libtool](https://www.gnu.org/software/libtool/)
* [make](https://www.gnu.org/software/make/)
* [GNU Scientific Library](https://www.gnu.org/software/gsl/doc/html/index.html) (>=2.5)
* C++/C [Intel](https://software.intel.com/content/www/us/en/develop/tools/compilers/c-compilers.html) or [GNU](https://gcc.gnu.org/) compiler
* [Intel MKL](https://software.intel.com/content/www/us/en/develop/tools/math-kernel-library.html) (optional)

The Python interface of QGD was developed and tested with Python 3.6 and 3.7.
The QGD Python interface needs the following packages to be installed on the system:

* [Qiskit](https://qiskit.org/documentation/install.html)
* [Numpy](https://numpy.org/install/)
* [scipy](https://www.scipy.org/install.html)
* [ctypes](https://docs.python.org/3/library/ctypes.html)



### How to build GNU Scientific Library

If the GNU Scientific Library is not installed on the system, it can be easily compiled and deployed by the end user without administrative privileges.
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

The developer version Quantum Gate Decomposer package can be downloaded from github repository [https://github.com/rakytap/quantum-gate-decomposer/tree/C++](https://github.com/rakytap/quantum-gate-decomposer/tree/C++).
After the downloaded package is extracted into the directory **path/to/qgdsource** (which would be the path to the source code of the QGD package), one can proceed to the compilation steps described in the next section.

### How to deploy the Quantum Gate Decomposer package

Similarly to GNU Scientific Library, the QGD package is also equipped with automake tools to ease the compilation and the deployment of the package.
To ensure that QGD package would find the necessary libraries and header files during compilation time it is advised to define the following environment variables:

$ export GSL_LIB_DIR=path/to/gsl/lib64

$ export GSL_INC_DIR=path/to/gsl/include

When using Intel compiler equipped with Intel MKL on a HPC, the compiler can find his way to the MKL libraries automatically through the environment variable MKLROOT which points to the root directory of the MKL package. 
If the Intel environment variables are not set, they can be initialized by the shell command:

$ source /opt/intel/composerxe/bin/compilervars.sh intel64

where **/opt/intel/composerxe** is the path to the Intel compiler package location which might be different from the given one.
(This step can be omitted when using GNU compiler, or when we do not have intention to use Intel MKL)
After the basic environment variables are set, the compilation can be configured by the command executed in the source directory **path/to/qgdsource** of the QGD package:

$ ./configure --prefix=path/to/qgd CC=gcc CXX=g++

where **path/to/qgd** is the installation path of the Quantum Gate Decomposer package.

The installation directory of the compiled QGD package is given by **--prefix=path/to/qgd** (which is different from the directory path of the source files given by **path/to/qgdsource**).
The user should have read and write permissions on the path **path/to/qgd** (which can be for example /home/username/qgd).
Another optional flag **--enable-ffast-math** enables the compiler's floating-point optimization (which is usually enabled by default in Intel compilers settings). 
While in general this optimization is considered to be dangerous, the runtime performance might be significantly increased due to this optimization. 
Try to compile without and with this flag and compare the performance and stability of the resulted binaries (for further information see section **How to use**).
We notice, that the QGD Python interface does not fully support this optimization resulting in lower performance than a standalone C applications.
On the other hand, if one choses Intel compiler to built the QGD package, the following configuration settings should be invoked:

$ ./configure --prefix=path/to/qgd --with-mkl CC=icc CXX=icpc

The **--with-mkl** flag sets the appropriate linking of the QGD package with the Intel MKL package.
If the flag is missing from the configuration than the CBLAS library of the GNU Scientific Library is used for linear algebra operations.
After the successful configuration the QGD package can be compiled by the shell command executed in the directory **path/to/qgdsource**:

$ make

After a successful compilation of the QGD package, it can be installed into the directory **path/to/gsl** by the shell command:

$ make install

The installation procedure will copy all the C header files, the static and shared libraries needed for further developments and the python interface files including a simple python example file **example.py** into the installation destination defined by the **path/to*qgd** path.


### How to use

The algorithm implemented in the QGD package intends to transform the given unitary into an identity matrix via a sequence of CNOT and U3 operations applied on the unitary. 
Thus, in order to really get the decomposition of a unitary, one should rather provide the complex transpose of this unitary as the input for the QGD decomposing process.

## Standalone executable

During the compilation and the installation processes of the QGD package a standalone executable was also built and copied into the directory **path/to/gsl/bin**. 
This executable can be executed by a command

$ ./decomposition_test

and it starts a decomposition of a random general unitary matrix. 
The source of this example is located in **path/to/qgdsource/test_standalone/** and shows a simple test case of the usage of the QGD package. 
The Doxygen documentation of the QGD API can be also generated in order fully exploit the functionalities of the QGD package (for further details see section **Doxygen manual** at the end of this manual).

## Python Interface

The QGD package contains a simple python interface allowing the access of the functionalities of the QGD package from Python. 
The usage of the QGD Python interface is demonstrated in the example file **example.py** located in the directory **path/to/qgd** and can be run similarly to any python scripts.
The example file imports the **qgd_python** module containing the wrapper class for the decomposition of a given unitary matrix.
The wrapper class loads the shared library libqgd.so and performs the data conversion between the python and C sides.

It should be noted, however, that the python interface functions are implemented only for few functionalities of the whole QGD API. 
Another desired interface functions can be implemented following the source of already implemented interface function in source file **python_interface.cpp** in the main directory of the QGD source code.


### Development

The QGD API enables the extension of the capabilities of the QGD packages into further projects. 
The QGD header files needed for the compilation of the project are provided in the directory **path/to/qgd/include**. 
The compiled object files should than be linked against the static or shared QGD libraries libqgd.a or libqgd.so, respectively,
located in the directory **path/to/qgd/lib64**.
In order to exploit the speedup comming from the floating point optimization, we notice that Intel's compiler (usually) use this optimization by default,
but when linking with GNU compiler, the -ffast-math option must be append when linking against the QGD API library is done.
The full documentation of the QGD API can be accessed by a Doxygen manual which can be accessed and recreated by steps described in the following section

## Doxygen manual

----- how to create doxygen API documentation --------
