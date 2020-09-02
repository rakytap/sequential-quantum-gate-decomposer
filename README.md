# Quantum Gate Decomposer

Quantum Gate Decomposer (QGD) is an optimization method to decompose an arbitrary NxN Unitary matrix into a sequence of U3 and CNOT gates. 
It is written in C/C++ providing a simple Python interface via ctypes-A foreign function library for Python and a possibility to run QGD as a standalone C executable.
(Although the Python interface and the standalone executable are linked with the same libraries, our tests showed a substantial decrement in performance of the python interface compared to the native C executable.)
The present package is supplied with automake tools to ease its deployment.
Although QGD can be built with gnu, the best performance of the package can be obtained using Intel compiler integrating its Math Kernel Library ([MKL](https://software.intel.com/content/www/us/en/develop/tools/math-kernel-library.html)). 
Since Intel compiler is present on almost each HPC's, we briefly summarize the steps to build and install the QGD package using Intel compiler.

### Dependencies

The optimization algorithm of QGD relies on the [multimin](https://www.gnu.org/software/gsl/doc/html/multimin.html) optimization component of the [GNU Scientific Library](https://www.gnu.org/software/gsl/doc/html/index.html). 
We developed and tested the QGD package with GNU Scientific Library of version 2.5 and 2.6.

* automake (only for development)
* autoconf (only for development)
* libtool
* GNU Scientific Library (>=2.5)
* Intel (recommended) or GNU compiler
* Intel MKL (optional, but recommended)

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

The downloaded package should be extracted into the directory **path/to/qgdsource**.

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

The installation directory will contain all the C header files, the static and shared libraries needed for further developments and the python interface files including a simple python example file **example.py**.



### How to use

The usage of the decomposer is demonstrated in test files in the file **test/decomposition.py**. The file contains several examples to calculate the decomposition of two, three and four qubit unitaries. The test examples can be run by the file ** run_tests.py** in the main directory with command:

*$ python3 run_tests.py*

The algorithm implemented in the code decomposes a given unitary into an identity matrix. 
In order to get the decomposition of a unitary into U3 and CNOT gates instead of operations transforming it to unity, one should give the complex transpose of the unitary as the input for the decomposition.

