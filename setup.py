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



from skbuild import setup
from setuptools import find_packages


setup(
    name="qgd",
    packages=find_packages(
        exclude=(
            "test_standalone", "test_standalone.*",
        )
    ),
    version='1.7',
    url="https://github.com/rakytap/sequential-quantum-gate-decomposer", 
    maintainer="Peter Rakyta",
    maintainer_email="peter.rakyta@ttk.elte.hu",
    include_package_data=True,
    install_requires=[
        "numpy>=1.19.2",
	"tbb-devel",
	"qiskit",
    ],
    tests_require=["pytest"],
    description='The C++ binding for the SQUANDER package',
    long_description=open("./README.md", 'r').read(),
    long_description_content_type="text/markdown",
    keywords="test, cmake, extension",
    classifiers=[
        "Intended Audience :: Developers",
        "License :: OSI Approved :: "
        "GNU General Public License v3.0",
        "Natural Language :: English",
        "Programming Language :: C",
        "Programming Language :: C++"
    ],
    license='GNU General Public License v3.0',
)






