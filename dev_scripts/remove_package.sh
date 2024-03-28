#!/bin/bash
# This script purges the docs and environment

cd ..
/bin/rm -r docs/build
/bin/rm -r dist
/bin/rm -r pmace_jax.egg-info
/bin/rm -r build

pip uninstall pmace_jax

cd dev_scripts
