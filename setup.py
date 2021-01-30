from distutils.command.install_data import install_data
from setuptools import find_packages, setup, Extension
from Cython.Build import cythonize
from setuptools.command.build_ext import build_ext
from setuptools.command.install_lib import install_lib
from setuptools.command.install_scripts import install_scripts
import struct
import pathlib
import os
import shutil
import numpy as np


CQGD_LIB_DIR = None
CQGD_LIBRARY_NAME = 'qgd'

extra_link_args = []
libraries = []

# checking BLAS library (MKL/OPenBlas/ATLAS)
blas_info = np.__config__.get_info('blas_opt_info')

# adding linking options for CBLAS
for item in blas_info.get("library_dirs"," "):
    extra_link_args = extra_link_args + ["-L"+item]
    os.environ["CBLAS_LIB_DIR"] = item

# adding linking options for CBLAS
BLAS_TYPE = None
for item in blas_info.get("libraries"," "):
    if item.startswith('mkl'):
        BLAS_TYPE = 'MKL'
    elif item.startswith('openblas'):
        BLAS_TYPE = 'OPENBLAS'

    libraries = libraries + [item]
    print(item)

# adding TBB include flags to the compiler if necessary
extra_compiler_flags = []
TBB_INC_DIR = os.getenv('TBB_INC_DIR')
if not( TBB_INC_DIR is None):
    extra_compiler_flags = ['-I'+TBB_INC_DIR]



# ***************************************************************************************************
# ************ Compiling C++ library ****************************************************************
# ***************************************************************************************************

class CMakeLibrary(Extension):
    """
    An extension to run the cmake build

    This simply overrides the base extension class so that setuptools
    doesn't try to build your sources for you
    """

    def __init__(self, name, sources=[]):
        super().__init__(name = name, sources = sources)

class InstallCMakeLibsData(install_data):
    """
    Just a wrapper to get the install data into the state-info ????

    Listing the installed files in the egg-info guarantees that
    all of the package files will be uninstalled when the user
    uninstalls your package through pip
    """

    def run(self):
        """
        Outfiles are the libraries that were built using cmake
        """

        # There seems to be no other way to do this; I tried listing the
        # libraries during the execution of the InstallCMakeLibs.run() but
        # setuptools never tracked them, seems like setuptools wants to
        # track the libraries through package data more than anything...
        # help would be appriciated

        self.outfiles = self.distribution.data_files

class InstallCMakeLibs(install_lib):
    """
    Get the libraries from the parent distribution, use those as the outfiles

    Skip building anything; everything is already built, forward libraries to
    the installation step
    """

    def run(self):
        """
        Copy libraries from the bin directory and place them as appropriate
        """

        self.announce("Moving library files", level=3)

        # We have already built the libraries in the previous build_ext step

        self.skip_build = True

        bin_dir = self.distribution.bin_dir

        # Depending on the files that are generated from your cmake
        # build chain, you may need to change the below code, such that
        # your files are moved to the appropriate location when the installation
        # is run

        libs = [os.path.join(bin_dir, _lib) for _lib in 
                os.listdir(bin_dir) if 
                os.path.isfile(os.path.join(bin_dir, _lib)) and 
                os.path.splitext(_lib)[1] in [".dll", ".so"]
                and not (_lib.startswith("python") or _lib.startswith('libpiquasso'))]

        for lib in libs:

            shutil.move(lib, os.path.join(self.build_dir,
                                          os.path.basename(lib)))

        # Mark the libs for installation, adding them to 
        # distribution.data_files seems to ensure that setuptools' record 
        # writer appends them to installed-files.txt in the package's egg-info
        #
        # Also tried adding the libraries to the distribution.libraries list, 
        # but that never seemed to add them to the installed-files.txt in the 
        # egg-info, and the online recommendation seems to be adding libraries 
        # into eager_resources in the call to setup(), which I think puts them 
        # in data_files anyways. 
        # 
        # What is the best way?

        # These are the additional installation files that should be
        # included in the package, but are resultant of the cmake build
        # step; depending on the files that are generated from your cmake
        # build chain, you may need to modify the below code

        self.distribution.data_files = [os.path.join(self.install_dir, 
                                                     os.path.basename(lib))
                                        for lib in libs]

        # Must be forced to run after adding the libs to data_files

        self.distribution.run_command("install_data")

        super().run()

class InstallCMakeScripts(install_scripts):
    """
    Install the scripts in the build dir
    """

    def run(self):
        """
        Copy the required directory to the build directory and super().run()
        """

        self.announce("Moving scripts files", level=3)

        # Scripts were already built in a previous step

        self.skip_build = True

        bin_dir = self.distribution.bin_dir

        scripts_dirs = [os.path.join(bin_dir, _dir) for _dir in
                        os.listdir(bin_dir) if
                        os.path.isdir(os.path.join(bin_dir, _dir))]

        for scripts_dir in scripts_dirs:

            shutil.move(scripts_dir,
                        os.path.join(self.build_dir,
                                     os.path.basename(scripts_dir)))

        # Mark the scripts for installation, adding them to 
        # distribution.scripts seems to ensure that the setuptools' record 
        # writer appends them to installed-files.txt in the package's egg-info

        self.distribution.scripts = scripts_dirs

        super().run()

class BuildCMakeExt(build_ext):
    """
    Builds using cmake instead of the python setuptools implicit build
    """

    def run(self):
        """
        Perform build_cmake before doing the 'normal' stuff
        """

        for extension in self.extensions:

            if extension.name == CQGD_LIBRARY_NAME:

                self.build_cmake(extension)

        # Here we only build the C++ library not the python extension
        #super().run() 

    def build_cmake(self, extension: Extension):
        """
        The steps required to build the extension
        """

        self.announce("Preparing the build environment", level=3)

        extension_path = pathlib.Path(self.get_ext_fullpath(extension.name))

        os.makedirs(self.build_lib, exist_ok=True)
        os.makedirs(extension_path.parent.absolute(), exist_ok=True)

        # Now that the necessary directories are created, build

        self.announce("Configuring cmake project", level=3)

        # Change your cmake arguments below as necessary
        # Below is just an example set of arguments for building Blender as a Python module
        global BLAS_TYPE
        if BLAS_TYPE == 'MKL':
            blas_flag = '-DUSE_MKL=ON'
        elif BLAS_TYPE == 'OPENBLAS':
            blas_flag = '-USE_OPENBLAS=ON'

        SOURCE_DIR=os.getcwd()

        os.chdir(self.build_lib)
        self.spawn(['cmake',
                    '-DCMAKE_BUILD_TYPE=Release', blas_flag, ' ', SOURCE_DIR])
        os.chdir(SOURCE_DIR)

        self.announce("Building binaries", level=3)

        self.spawn(["cmake", "--build", self.build_lib])

        # setting the library path for other Cython extensions
        global CQGD_LIB_DIR 
        CQGD_LIB_DIR = self.build_lib
        
        # After build_ext is run, the following commands will run:
        # 
        # install_lib
        # install_scripts
        # 
        # These commands are subclassed above to avoid pitfalls that
        # setuptools tries to impose when installing these, as it usually
        # wants to build those libs and scripts as well or move them to a
        # different place. See comments above for additional information


# first build the C++ library with CMAKE
setup(name='CPiquasso',
      version='0.1',
      packages=find_packages(),
      ext_modules=[CMakeLibrary(name=CQGD_LIBRARY_NAME)],
      description='The C++ library for the Picasso project',
      long_description=open("./README.md", 'r').read(),
      long_description_content_type="text/markdown",
      keywords="test, cmake, extension",
      classifiers=["Intended Audience :: Developers",
                   "License :: OSI Approved :: "
                   "Apache License 2.0.",
                   "Natural Language :: English",
                   "Programming Language :: C",
                   "Programming Language :: C++"],
      license='Apache License 2.0.',
      cmdclass={
          'build_ext': BuildCMakeExt,
          'install_data': InstallCMakeLibsData,
          'install_lib': InstallCMakeLibs,
          'install_scripts': InstallCMakeScripts
          }
    )



# ***************************************************************************************************
# ************ Compiling Cython extensions **********************************************************
# ***************************************************************************************************


class BuildCythonExt(build_ext):
    """
    Class to override the default destination path of Cython extensions
    """

    def run(self):
        """
        Override the deafult path of the library before doing the 'normal' stuff
        """
        self.build_lib = EXTENSION_PATH

        super().run() 



# compiler options for cython extensions
extra_compiler_flags = extra_compiler_flags + ['-std=c++11', '-I' + os.path.join('common', 'include'), '-I' + os.path.join('operations', 'include'), '-I' + os.path.join('decomposition', 'include'), '-DCYTHON']
extra_link_args = extra_link_args + ['-L' + CQGD_LIB_DIR]
runtime_library_dirs = [CQGD_LIB_DIR]
libraries= libraries + [CQGD_LIBRARY_NAME]

"""
extensions = [{'name': 'qgd_CNOT', 'path': os.path.join('qgd_python', 'gates' )},
              {'name': 'qgd_Operation_Block', 'path': os.path.join('qgd_python', 'gates' )},
              {'name': 'qgd_N_Qubit_Decomposition', 'path': os.path.join('qgd_python')}
             ]


for ext in extensions:
    # building individual extensions
    print('building ' + ext['name'] + '  extension')
    EXTENSION_PATH = ext['path']
    extensions = [Extension(ext['name'],
                         language='c++11', 
                         sources=[os.path.join(EXTENSION_PATH, ext['name']+'.pyx')],
                         extra_compile_args=extra_compiler_flags,
                         runtime_library_dirs=runtime_library_dirs,
                         libraries=libraries,
                         extra_link_args=extra_link_args,
             )]

    setup(
           ext_modules = cythonize(extensions), 
           include_dirs=[np.get_include()], 
           cmdclass={
              'build_ext': BuildCythonExt
           })
"""

extensions = [{'name': 'qgd_U3', 'path': os.path.join('qgd_python', 'gates' )},
              {'name': 'qgd_CNOT', 'path': os.path.join('qgd_python', 'gates' )},
              {'name': 'qgd_Operation_Block', 'path': os.path.join('qgd_python', 'gates' )},
              {'name': 'qgd_N_Qubit_Decomposition_Wrapper', 'path': os.path.join('qgd_python' )}
             ]

#os.environ["CC"] = "g++"

for ext in extensions:
    # building individual extensions
    print('building ' + ext['name'] + '  extension')
    EXTENSION_PATH = ext['path']

    setup(name=ext['name'], version="1.0",
      ext_modules=[
         Extension(ext['name'], 
                    sources=[os.path.join(EXTENSION_PATH, ext['name']+'.cpp')],
                    language='c++', 
                    include_dirs=[np.get_include()],
                    extra_compile_args=extra_compiler_flags,
                    runtime_library_dirs=runtime_library_dirs,
                    libraries=libraries,
                    extra_link_args=extra_link_args),
         ],
        cmdclass={
              'build_ext': BuildCythonExt
           })


