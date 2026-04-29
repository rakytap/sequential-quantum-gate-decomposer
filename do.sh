#!/bin/bash
set -euo pipefail

export TBB_INC_DIR=~/.local/include
export TBB_LIB_DIR=~/.local/lib

python setup.py build_ext "$@"