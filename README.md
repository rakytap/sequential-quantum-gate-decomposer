# Quantum Gate Decomposer

Quantum Gate Decomposer (QGD) is an optimization method to decompose an arbitrary NxN Unitary matrix into a sequence of U3 and CNOT gates. 
It is written in C/C++ providing a simple Python interface via ctypes-A foreign function library for Python and a possibility to run QGD as a standalone C executable.
(Although the Python interface and the standalone executable are linked with the same libraries, our tests showed a substantial decrement in performance of the python interface compared to the native C executable.)
The present package is supplied with automake tools to ease its deployment.
Although QGD can be built with gnu, the best performance of the package can be obtained using Intel compiler integrating its Math Kernel Library ([MKL](https://software.intel.com/content/www/us/en/develop/tools/math-kernel-library.html)). 
Since Intel compiler is present on almost each HPC's, we briefly summarize the steps to build and install the QGD package using Intel compiler.

The project was supported ... HUNQT ...

### Dependencies

The optimization algorithm of QGD relies on the [multimin](https://www.gnu.org/software/gsl/doc/html/multimin.html) optimization component of the [GNU Scientific Library](https://www.gnu.org/software/gsl/doc/html/index.html). 
We developed and tested the QGD package with GNU Scientific Library of version 2.5 and 2.6.

* automake (for further development purposes)
* autoconf (for further development purposes)
* libtool
* GNU Scientific Library (>=2.5)
* Intel (recommended) or GNU compiler
* Intel MKL (optional, but strongly recommended)

The Python interface of QGD was developed and tested with Python 3.6 and 3.7
QGD Python interface needs the following packages to be installed on the system:

* [Qiskit](https://qiskit.org/documentation/install.html)
* [Numpy](https://numpy.org/install/)
* [scipy](https://www.scipy.org/install.html)



### How to build GNU Scientific Library

If the GNU Scientific Library is not installed on the system, it can be easily compiled and deployed by the end user without administrative privileges.
The GNU Scientific Library can be downloaded from the site [https://www.gnu.org/software/gsl/](https://www.gnu.org/software/gsl/).
After the downloaded package is extracted somewhere in the home directory of the user (path/to/gslsource), one should configure the compiling environment using the **configure** tool.
To ensure the usage of Intel compilers, the following shell command should be executed inside the directory path/to/gslsource:

$ ./configure --prefix=path/to/gsl CC=icc CXX=icpc

The installation directory of the compiled GNU Scientific Library is given by **--prefix=path/to/gsl** (which is different from the directory path of the source files given by path/to/gslsource).
The end user should have read and write permissions on the path **path/to/gsl** (for example /home/username/gsl).
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

Usually, when the Intel compiler module is loaded on the HPC, the compiler can find his way to the MKL libraries automatically through the environment variable MKLROOT which points to the root irectory of the MKL package. 
If the Intel environment variables are not set, they can be initialized by the shell command:

$ source /opt/intel/composerxe/bin/compilervars.sh intel64

where **/opt/intel/composerxe** is the path to the Intel compiler package location which might be different from the given one.
After the environment variables are set, the compilation environment can be configured by the command executed in the source directory **path/to/qgdsource** of the QGD package:

$ ./configure --prefix=path/to/qgd CC=icc CXX=icpc

where **path/to/qgd** is the installation path of the Quantum Gate Decomposer package.

The installation directory of the compiled QGD package is given by **--prefix=path/to/qgd** (which is different from the directory path of the source files given by **path/to/qgdsource**).
The end user should have read and write permissions on the path **path/to/qgd** (for example /home/username/qgd).
After the successful configuration the QGD package can be compiled by the shell command executed in the directory **path/to/qgdsource**:

$ make

After a successful compilation of the QGD package, it can be installed into the directory **path/to/gsl** by the shell command:

$ make install

The installation procedure will copy all the C header files, the static and shared libraries needed for further developments and the python interface files including a simple python example file **example.py** into the installation destination defined by the **path/to*qgd** path.


### How to use

The algorithm implemented in the QGD package intends to transform the given unitary into an identity matrix via a sequence of CNOT and U3 operations applied on the unitary. 
Thus, in order to get the decomposition of the unitary, one should rather provide the complex transpose of the unitary as the input for the decomposition process.

## Standalone executable

During the compilation and the instalaltion processes of the QGD package a standalone executable was also built and copied into the directory **path/to/gsl/bin**. 
This executable can be executed by a command

$ ./decomposition_test

and it starts a decomposition of a random general unitary matrix. 
The source of this example is located in **path/to/qgdsource/test_standalone/** and shows a simple testcase of the usage of the QGD package on source code level. 
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
The compiled object files should than be linked against the static or shared QGD library libqgd.a or libqgd.so, respectively,
located in the directory **path/to/qgd/lib64**.
To resolve all the references during the linkage, one should also link against the corresponding libraries of the 
GNU Scientific Library located in the GSL_LIB_DIR environment variable set above, and against the Intel MKL libraries if they were used in the compilation of the QGD package.
The full documentation of the QGD API can be accessed by a Doxygen manual which can be accessed and recreated by steps described in the following section

## Doxygen manual

----- how to create doxygen API documentation --------
