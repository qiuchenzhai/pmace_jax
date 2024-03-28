#!/bin/bash
# This script just installs pmace_jax along with all requirements
# for the package, demos, and documentation.
# However, it does not remove the existing installation of pmace_jax.

conda activate pmace_jax
cd ..
pip install -r requirements.txt
pip install -e .
pip install -r demo/requirements.txt
pip install -r docs/requirements.txt 
cd dev_scripts

