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
from skbuild.command.build_ext import build_ext
from setuptools import Command
import os
import shutil
import glob
from pathlib import Path


class CopyLibraryCommand(Command):
    """Custom command to copy qgd library from cmake-install to package directory"""

    description = "Copy qgd library to package directory"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        """Copy qgd.dll/qgd.so from cmake-install to squander/"""
        import platform

        # Determine library extension and possible names
        if platform.system() == "Windows":
            lib_names = ["qgd.dll"]
        elif platform.system() == "Darwin":
            lib_names = ["qgd.dylib", "libqgd.dylib"]
        else:
            # Linux: try both with and without lib prefix
            lib_names = ["libqgd.so", "qgd.so"]

        # Find the library in cmake-install directory
        build_dir = Path("_skbuild")
        if not build_dir.exists():
            print("Warning: _skbuild directory not found")
            return

        # Search for the library in cmake-install
        found = False
        for install_dir in build_dir.rglob("cmake-install"):
            squander_dir = install_dir / "squander"
            if not squander_dir.exists():
                continue

            # Try all possible library names
            for lib_name in lib_names:
                lib_path = squander_dir / lib_name
                if lib_path.exists():
                    # Determine destination name (use qgd.* format)
                    dest_name = f"qgd{lib_path.suffix}"
                    dest = Path("squander") / dest_name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(lib_path, dest)
                    print(f"Copied {lib_path} to {dest}")
                    found = True
                    break

            if found:
                break

        # Also check if library is already in squander/ from build
        if not found:
            for lib_name in lib_names:
                build_lib = Path("squander") / lib_name
                if build_lib.exists():
                    print(f"Library already exists at {build_lib}")
                    found = True
                    break

            # Also check for libqgd.so if we're on Linux
            if not found and platform.system() != "Windows":
                for pattern in ["libqgd.*", "qgd.*"]:
                    for lib_file in Path("squander").glob(pattern):
                        if lib_file.is_file():
                            print(f"Library already exists at {lib_file}")
                            found = True
                            break
                    if found:
                        break

        if not found:
            # Debug: list what's actually in the cmake-install directory
            print("Debug: Searching for library files...")
            for install_dir in build_dir.rglob("cmake-install"):
                squander_dir = install_dir / "squander"
                if squander_dir.exists():
                    print(f"  Checking {squander_dir}")
                    for item in squander_dir.iterdir():
                        if item.is_file() and (
                            ".so" in item.name
                            or ".dll" in item.name
                            or ".dylib" in item.name
                        ):
                            print(f"    Found: {item}")
            print(f"Warning: Could not find qgd library in cmake-install directory")
            print(f"  Tried: {lib_names}")


# Custom build_ext that runs after CMake install
class CustomBuildExt(build_ext):
    def run(self):
        # Run the normal build_ext
        super().run()
        # Copy the library
        copy_cmd = CopyLibraryCommand(self.distribution)
        copy_cmd.run()


setup(
    packages=["squander"],
    include_package_data=True,
    package_data={
        "": ["README.md", "MANIFEST.in", "pyproject.toml"],
        "squander": ["*.dll", "*.so", "*.dylib"],  # Include shared libraries
    },
    cmdclass={
        "build_ext": CustomBuildExt,
    },
)
