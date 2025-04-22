# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 14:42:56 2020
Copyright 2020 Peter Rakyta, Ph.D.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

You should have received a copy of the GNU General Public License
along with this program.  If not, see http://www.gnu.org/licenses/.

@author: Peter Rakyta, Ph.D.
"""
## \file setup.py
## \brief Building script for the SQUANDER package



from skbuild import setup
from setuptools import find_packages

from skbuild.cmaker import get_cmake_version
setup(
    name="squander",
    packages= [ 'squander' ],
    version='1.9.2',
    url="https://github.com/rakytap/sequential-quantum-gate-decomposer", 
    maintainer="Peter Rakyta",
    maintainer_email="peter.rakyta@ttk.elte.hu",
    include_package_data=True,
    package_data = {
        '': ['README.md', 'MANIFEST.in', 'pyproject.toml']
    },  
    tests_require=["pytest"],
    description='C++ library with Python interface to train quantum circuits, quantum gate synthesis and state preparation.',
    long_description=open("./README.md", 'r').read(),
    long_description_content_type="text/markdown",
    keywords="test, cmake, extension",
    classifiers=[
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Programming Language :: C",
        "Programming Language :: C++"
    ],
    license='Apache-2.0 license',
)






