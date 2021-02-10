# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 14:42:56 2020
Copyright (C) 2020 Peter Rakyta, Ph.D.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see http://www.gnu.org/licenses/.

@author: Peter Rakyta, Ph.D.
"""
## \file setup.py
## \brief Building script for the SQUANDER package

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sysconfig
import sys
import pathlib
import os
import numpy as np


EXT_NAME = 'qgd'


class CMakeExtension(Extension):
    """
    An extension to run the cmake build

    This simply overrides the base extension class so that setuptools
    doesn't try to build your sources for you
    """

    def __init__(self, name, sources=[]):
        super().__init__(name = name, sources = sources)



class BuildCMakeExt(build_ext):
    """
    Builds using cmake instead of the python setuptools implicit build
    """

    def run(self):
        """
        Perform build_cmake before doing the 'normal' stuff
        """

        for extension in self.extensions:

            if extension.name == EXT_NAME:

                self.build_cmake(extension)

        # We build python extension by CMake calls
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

        # checking BLAS library (MKL/OpenBlas/ATLAS)
        blas_info = np.__config__.get_info('blas_opt_info')

        # adding macro values options for CBLAS usage
        blas_flag = ' ' 
        for item in blas_info.get("libraries"," "):
            if item.startswith('mkl'):
                blas_flag = '-DUSE_MKL=ON'
            elif item.startswith('openblas'):
                blas_flag = '-DUSE_OPENBLAS=ON'

       

        # get the platform specific extension suffix
        ext_suffix = '-DEXT_SUFFIX='+sysconfig.get_config_var('EXT_SUFFIX')


        # get Python executable to help out Cmake when multiple python interpreters are installed
        python_executable = '-DPYTHON_EXECUTABLE={}'.format(sys.executable)

        # get the source directory
        SOURCE_DIR=os.getcwd()

        os.chdir(self.build_lib)
        self.spawn(['cmake',
                    '-DCMAKE_BUILD_TYPE=Release', ext_suffix, ' ', python_executable, ' ', blas_flag, ' ', SOURCE_DIR])
        os.chdir(SOURCE_DIR)

        self.announce("Building binaries", level=3)
        self.spawn(["cmake", "--build", self.build_lib])
        





# building the shared libraries (extensions) with CMAKE
setup(name=EXT_NAME,
      version='1.5',
      url="https://github.com/rakytap/sequential-quantum-gate-decomposer", 
      maintainer="Peter Rakyta",
      maintainer_email="peter.rakyta@ttk.elte.hu",
      install_requires=[
        "numpy==1.19.4",
        "scipy==1.5.4",
      ],
      description='The C++ binding for the SQUANDER package',
      long_description=open("./README.md", 'r').read(),
      long_description_content_type="text/markdown",
      keywords="test, cmake, extension",
      classifiers=["Intended Audience :: Developers",
                   "License :: OSI Approved :: "
                   "GNU General Public License v3.0",
                   "Natural Language :: English",
                   "Programming Language :: C",
                   "Programming Language :: C++"],
      license='GNU General Public License v3.0',
      ext_modules=[CMakeExtension(name=EXT_NAME)],
      cmdclass={
          'build_ext': BuildCMakeExt,
          }
    )



